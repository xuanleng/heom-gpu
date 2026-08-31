"""Read twodes_post frequency-domain HDF5."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np

from heom_tools.twodes.post.config import parse_window
from heom_tools.twodes.post.h5_spectra import H5_SPECTRA_TOKENS, token_spec
from heom_tools.twodes.post.io import decode_attr, from_h5_complex


def spectrum_h5_path(token: str) -> str:
    """HDF5 dataset path for one h5_spectra token."""
    spec = token_spec(token)
    return f"/results/spectra/{spec.h5_group}/{spec.h5_dataset}"


def spectrum_plot_label(token: str) -> str:
    """Panel title fragment, e.g. 'Re 2des', 'Re rp-gsb'."""
    return f"Re {normalize_spectrum_token(token)}"


def normalize_spectrum_token(raw: str) -> str:
    tok = str(raw).strip().lower()
    token_spec(tok)
    return tok


def discover_available_spectra(f: h5py.File) -> Tuple[str, ...]:
    """Tokens present under /results/spectra (stable h5_spectra order)."""
    out: List[str] = []
    for tok in sorted(H5_SPECTRA_TOKENS):
        if spectrum_h5_path(tok) in f:
            out.append(tok)
    return tuple(out)


@dataclass(frozen=True)
class Post2desFile:
    """Metadata and axes for one post-processed 2DES HDF5 (no spectrum tensor loaded)."""

    path: str
    label: str
    w_tau: np.ndarray
    w_t: np.ndarray
    tw_fs: np.ndarray
    pol: str
    window: Optional[List[float]]
    levels: int
    lines: List[float]
    spectrum: str
    available_spectra: Tuple[str, ...]
    dw: Optional[float] = None
    w0_cm: Optional[float] = None


def _attr_text(val, default: str = "") -> str:
    """decode_attr → str for JSON / equality checks."""
    if val is None:
        return default
    v = decode_attr(val)
    if v is None:
        return default
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v if isinstance(v, str) else str(v)


def _optional_float_attr(par, key: str) -> Optional[float]:
    if key not in par:
        return None
    v = decode_attr(par[key])
    if v is None:
        return None
    return float(v)


def _validate_spectrum_stack(path: str, ds: h5py.Dataset, tw_fs: np.ndarray) -> None:
    if ds.ndim != 3:
        raise ValueError(
            f"{path}: {ds.name} must be 3D (n_tw, n_wtau, n_wt), got shape {ds.shape}",
        )
    if ds.shape[0] != tw_fs.size:
        raise ValueError(
            f"{path}: tw axis length {tw_fs.size} != spectrum axis 0 {ds.shape[0]}",
        )


def is_post_2des_h5(path: str) -> bool:
    """True when *path* looks like a twodes-post frequency HDF5.

    Never raises for probe use: missing groups, wrong attrs, or non-HDF5
    paths all return False (including AttributeError from bad fallbacks).
    """
    try:
        with h5py.File(path, "r") as f:
            if "parameters" not in f:
                return False
            if _attr_text(f["parameters"].attrs.get("spectrum_type", "")) != "2des_post":
                return False
            if "/results/axes/w_tau_cm" not in f:
                return False
            return bool(discover_available_spectra(f))
    except (OSError, KeyError, ValueError, AttributeError, TypeError):
        return False


def load_post_2des_h5(path: str, *, spectrum: str = "2des") -> Post2desFile:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    tok = normalize_spectrum_token(spectrum)
    label = os.path.basename(path)
    with h5py.File(path, "r") as f:
        if "parameters" not in f:
            raise ValueError(f"{path}: missing group /parameters")
        par = f["parameters"].attrs
        if _attr_text(par.get("spectrum_type", "")) != "2des_post":
            raise ValueError(f"{path}: not a twodes_post output (spectrum_type != 2des_post)")
        for ax in ("/results/axes/w_tau_cm", "/results/axes/w_t_cm", "/results/axes/tw_fs"):
            if ax not in f:
                raise ValueError(f"{path}: missing dataset {ax}")
        available = discover_available_spectra(f)
        if not available:
            raise ValueError(f"{path}: no known spectra under /results/spectra")
        if tok not in available:
            have = ", ".join(available)
            raise ValueError(
                f"{path}: spectrum {tok!r} not stored (have: {have}); "
                f"re-run twodes-post with write_h5 and h5_spectra",
            )
        ds_path = spectrum_h5_path(tok)
        ds = f[ds_path]
        w_tau = np.asarray(f["/results/axes/w_tau_cm"][...], dtype=np.float64)
        w_t = np.asarray(f["/results/axes/w_t_cm"][...], dtype=np.float64)
        tw_fs = np.asarray(f["/results/axes/tw_fs"][...], dtype=np.float64)
        _validate_spectrum_stack(path, ds, tw_fs)
        pol = _attr_text(par.get("pol", ""))
        window = None
        if "window" in par:
            window = parse_window(json.loads(_attr_text(par["window"])))
        levels = int(par["levels"]) if "levels" in par else 25
        lines = json.loads(_attr_text(par["lines"])) if "lines" in par else []
        dw = _optional_float_attr(par, "dw")
        w0_cm = _optional_float_attr(par, "w0_cm")
    return Post2desFile(
        path=os.path.abspath(path),
        label=label,
        w_tau=w_tau,
        w_t=w_t,
        tw_fs=tw_fs,
        pol=pol,
        window=window,
        levels=levels,
        lines=list(lines or []),
        spectrum=tok,
        available_spectra=available,
        dw=dw,
        w0_cm=w0_cm,
    )


def decode_plane(raw) -> np.ndarray:
    if getattr(raw.dtype, "names", None):
        return from_h5_complex(raw)
    return np.asarray(raw, dtype=np.complex128)


def read_spectrum_planes(meta: Post2desFile, tw_indices: Sequence[int]) -> List[np.ndarray]:
    """Read several Tw slices for meta.spectrum in one HDF5 open."""
    if not tw_indices:
        return []
    ds_path = spectrum_h5_path(meta.spectrum)
    with h5py.File(meta.path, "r") as f:
        if ds_path not in f:
            raise ValueError(f"{meta.path}: missing dataset {ds_path}")
        ds = f[ds_path]
        return [decode_plane(ds[int(k)]) for k in tw_indices]


def read_spectrum_plane(meta: Post2desFile, tw_index: int) -> np.ndarray:
    """Read one Tw slice from disk (complex 2D)."""
    return read_spectrum_planes(meta, [tw_index])[0]
