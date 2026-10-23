"""Solver and post-process HDF5 read/write."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import List, Tuple

import h5py
import numpy as np

from heom_tools.common import heom_h5, mc_naming
from heom_tools.common.heom_h5 import decode_attr as _decode_attr_common
from heom_tools.common.mc_merge import ensure_shard_group_merged, fold_shard_groups

from .config import PostConfig, cfg_pol_str, parse_window, tw_index_to_attr
from .constants import H5_COMPLEX, REQUIRED_SIGNALS
from .h5_spectra import array_for_token, h5_spectra_fingerprint, token_spec

_RESULTS_DATASET = "resp"


def is_rank_shard_filename(filename: str) -> bool:
    """True for an MC worker shard basename (ends ``_rank{N}.h5``)."""
    return mc_naming.is_rank_shard(os.path.basename(filename))


def ensure_base_merged(shard_path: str) -> str:
    """Merge sibling MC shards for one ``<base>_<key>`` group and return the
    merged single-shot 2DES HDF5 path."""
    return ensure_shard_group_merged(
        shard_path,
        _RESULTS_DATASET,
        log_prefix="twodes-post",
    )


def fold_rank_groups(cwd: str) -> None:
    """Fold every MC shard group in `cwd` into ``<base>_<key>_r<lo>-<hi>.h5``."""
    fold_shard_groups(cwd, _RESULTS_DATASET, log_prefix="twodes-post")


def to_h5_complex(arr: np.ndarray) -> np.ndarray:
    out = np.empty(arr.shape, dtype=H5_COMPLEX)
    out["r"] = arr.real
    out["i"] = arr.imag
    return out


def from_h5_complex(data: np.ndarray) -> np.ndarray:
    """Restore complex array from HDF5 compound or native complex storage."""
    if getattr(data.dtype, "names", None):
        return data["r"] + 1j * data["i"]
    return np.asarray(data)


def decode_attr(val):
    """Normalize h5py attributes via ``heom_tools.common.heom_h5.decode_attr``.

    Single decode path for the whole tools tree — do not reimplement here.
    """
    return _decode_attr_common(val)


def split_csv_attr(val) -> List[str]:
    text = decode_attr(val)
    if not isinstance(text, str):
        text = str(text)
    return text.split(",")


def read_params_attrs(par, *, path: str = "") -> Tuple[List[str], List[str]]:
    prefix = f"{path}: " if path else ""
    for key in ("signals", "pathways"):
        if key not in par:
            raise ValueError(f"{prefix}missing parameters attribute {key!r}")
    return split_csv_attr(par["signals"]), split_csv_attr(par["pathways"])


def write_h5_complex_ds(group, name: str, arr: np.ndarray) -> None:
    ds = group.create_dataset(name, shape=arr.shape, dtype=H5_COMPLEX)
    ds[...] = to_h5_complex(arr)


def is_h5_complex_compound(ds) -> bool:
    """True when the on-disk HDF5 type is compound ``{r,i}`` (solver contract).

    Recent h5py versions present such datasets as NumPy ``complex128`` in
    ``ds.dtype``; check the HDF5 type id, not the Python view.
    """
    try:
        t = ds.id.get_type()
    except AttributeError:
        return False
    if not isinstance(t, h5py.h5t.TypeCompoundID):
        return False
    if t.get_nmembers() != 2:
        return False
    names = {t.get_member_name(i) for i in range(2)}
    return names == {b"r", b"i"}


def require_2des_signals(signals: List[str]) -> None:
    missing = REQUIRED_SIGNALS - set(signals)
    if missing:
        raise ValueError(
            f"2DES post needs signals {sorted(REQUIRED_SIGNALS)}; missing {sorted(missing)}"
        )


def is_solver_2des_h5(path: str, *, verbose: bool = False) -> bool:
    """True when *path* looks like a solver 2DES HDF5 (not post-processed output)."""
    try:
        with h5py.File(path, "r") as f:
            for ds in ("/results/tau_fs", "/results/tw_fs", "/results/t_fs", "/results/resp"):
                if ds not in f:
                    if verbose:
                        print(f"twodes-post: not 2DES HDF5 {path}: missing dataset {ds}", file=sys.stderr)
                    return False
            if "/parameters" not in f:
                if verbose:
                    print(f"twodes-post: not 2DES HDF5 {path}: missing group /parameters", file=sys.stderr)
                return False
            signals, pathways = read_params_attrs(f["/parameters"].attrs, path=path)
            require_2des_signals(signals)
            if not pathways:
                if verbose:
                    print(f"twodes-post: not 2DES HDF5 {path}: empty pathways", file=sys.stderr)
                return False
            return True
    except (OSError, ValueError, KeyError) as e:
        if verbose:
            print(f"twodes-post: not 2DES HDF5 {path}: {e}", file=sys.stderr)
        return False


def load_meta(path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Lightweight read for skip checks (no response tensor)."""
    with h5py.File(path, "r") as f:
        heom_h5.require_complete(f)
        if "/results/tw_fs" not in f:
            raise ValueError(f"{path}: missing dataset /results/tw_fs")
        if "/parameters" not in f:
            raise ValueError(f"{path}: missing group /parameters")
        tw = f["/results/tw_fs"][...]
        signals, pathways = read_params_attrs(f["/parameters"].attrs, path=path)
    return tw, signals, pathways


def load_axes(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Time axes and metadata only (no response tensor)."""
    with h5py.File(path, "r") as f:
        heom_h5.require_complete(f)
        for ds in ("/results/tau_fs", "/results/tw_fs", "/results/t_fs", "/results/resp"):
            if ds not in f:
                raise ValueError(f"{path}: missing dataset {ds}")
        if "/parameters" not in f:
            raise ValueError(f"{path}: missing group /parameters")
        tau = f["/results/tau_fs"][...]
        tw = f["/results/tw_fs"][...]
        t = f["/results/t_fs"][...]
        signals, pathways = read_params_attrs(f["/parameters"].attrs, path=path)
    require_2des_signals(signals)
    return tau, tw, t, signals, pathways


def contract_resp_slice(resp: np.ndarray) -> None:
    if resp.shape[0] != 3:
        raise ValueError(
            f"expected 3 invariants (I1,I2,I3), got {resp.shape[0]}. "
            "Run the solver with orientation: isotropic.")


def contract(resp, weights):
    return np.tensordot(weights, resp, axes=([0], [0]))


def read_contracted_tw(resp_ds, k: int, weights: np.ndarray) -> np.ndarray:
    """One Tw index: read resp[...,k,:] and contract invariants (no full 5D tensor)."""
    r = resp_ds[..., k, :]
    resp = from_h5_complex(r[...]) if getattr(r.dtype, "names", None) else np.asarray(r)
    contract_resp_slice(resp)
    return contract(resp, weights)


def _source_precision_mode(src_path: str) -> str:
    """Carry the solver build's precision provenance onto the derived spectrum so
    a lone 2DES post file self-documents fp64 vs mixed_fp32. Legacy sources (pre
    mixed-precision build) have no attr and were all FP64 — same convention as the
    MC merge guard (heom_tools.common.mc_merge)."""
    try:
        with h5py.File(src_path, "r") as src:
            v = src.attrs.get("precision_mode")
    except OSError:
        return "fp64"
    if v is None:
        return "fp64"
    return v.decode() if isinstance(v, bytes) else v


def write_post_h5(path, *, w_tau, w_t, tw_vals, spec_by_pathway, spec_total,
                  cfg: PostConfig, source_h5, pathway, signals, pathways):
    h5_tokens = cfg.h5_spectra or []
    outdir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(suffix=".h5", dir=outdir)
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["precision_mode"] = _source_precision_mode(source_h5)
            res = f.create_group("results")
            ax = res.create_group("axes")
            ax.create_dataset("w_tau_cm", data=w_tau)
            ax.create_dataset("w_t_cm", data=w_t)
            ax.create_dataset("tw_fs", data=tw_vals)
            spec_g = res.create_group("spectra")
            for tok in h5_tokens:
                spec = token_spec(tok)
                arr = array_for_token(
                    tok, spec_by_pathway=spec_by_pathway, spec_total=spec_total,
                )
                pg = spec_g.require_group(spec.h5_group)
                write_h5_complex_ds(pg, spec.h5_dataset, arr)
            par = f.create_group("parameters")
            pol = cfg.pol
            st = os.stat(source_h5)
            par.attrs["spectrum_type"] = "2des_post"
            par.attrs["pol"] = cfg_pol_str(pol)
            par.attrs["pathway_sum"] = pathway
            par.attrs["dw"] = float(cfg.dw)
            par.attrs["w0_cm"] = float(cfg.w0)
            par.attrs["signals"] = ",".join(signals)
            par.attrs["pathways"] = ",".join(pathways)
            par.attrs["source_h5"] = os.path.basename(source_h5)
            par.attrs["source_mtime"] = int(st.st_mtime_ns)
            par.attrs["source_size"] = int(st.st_size)
            par.attrs["tw_index"] = tw_index_to_attr(cfg.tw_index)
            par.attrs["time_domain"] = int(bool(cfg.time_domain))
            par.attrs["fft_method"] = str(cfg.fft_method)
            win = parse_window(cfg.window)
            par.attrs["window"] = json.dumps(win)
            par.attrs["levels"] = int(cfg.levels)
            par.attrs["lines"] = json.dumps(list(cfg.lines or []))
            fp = h5_spectra_fingerprint(h5_tokens, write_h5=True)
            par.attrs["h5_spectra"] = json.dumps(fp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
