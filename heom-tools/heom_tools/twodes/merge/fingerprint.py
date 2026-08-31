"""Physics fingerprint for solver 2DES HDF5 merge compatibility checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from heom_tools.twodes.post.io import decode_attr, from_h5_complex, read_params_attrs

_ARRAY_RTOL = 0.0
_ARRAY_ATOL = 0.0

_PARAM_ATTRS = (
    "L_cutoff", "L_total", "basis", "dt", "expansion_method", "filter_threshold",
    "n_bath_modes_total", "n_extra_poles", "n_sites", "n_steps", "npa",
    "orientation", "output_interval", "pathways", "signals", "sink_enabled",
    "spectrum_type", "static_disorder_model", "temperature", "use_filtering", "w0_cm",
)

_PARAM_DATASETS = (
    "dipoles", "eta_cm", "gamma_cm", "gamma_inverse_fs", "hamiltonian_cm",
    "static_disorder_fwhm_cm", "strengths",
)

_DERIVED_DATASETS = (
    "bath_c_cm2", "bath_nu_cm", "eigenvalues_relative_cm", "eigenvectors",
    "pade_chi", "pade_zeta_dimensionless",
)


@dataclass(frozen=True)
class FieldDiff:
    field: str
    reference: str
    other: str
    ref_value: Any
    other_value: Any


@dataclass
class SolverFingerprint:
    path: str
    tau: np.ndarray
    t: np.ndarray
    tw: np.ndarray
    resp_shape: Tuple[int, ...]
    signals: Tuple[str, ...]
    pathways: Tuple[str, ...]
    param_attrs: Dict[str, Any]
    param_datasets: Dict[str, np.ndarray]
    derived_datasets: Dict[str, np.ndarray]


def _read_dataset(group: h5py.Group, name: str) -> Optional[np.ndarray]:
    if name not in group:
        return None
    raw = np.asarray(group[name][...])
    return from_h5_complex(raw)


def load_fingerprint(path: str) -> SolverFingerprint:
    with h5py.File(path, "r") as f:
        if "/results/resp" not in f:
            raise ValueError(f"{path}: missing /results/resp")
        tau = np.asarray(f["/results/tau_fs"][...], dtype=np.float64).ravel()
        t = np.asarray(f["/results/t_fs"][...], dtype=np.float64).ravel()
        tw = np.asarray(f["/results/tw_fs"][...], dtype=np.float64).ravel()
        resp_ds = f["/results/resp"]
        resp_shape = tuple(int(x) for x in resp_ds.shape)
        signals, pathways = read_params_attrs(f["/parameters"].attrs, path=path)
        par = f["/parameters"]
        attrs = {key: decode_attr(par.attrs[key]) for key in _PARAM_ATTRS if key in par.attrs}
        param_ds = {
            name: _read_dataset(par, name)
            for name in _PARAM_DATASETS
            if name in par
        }
        derived_ds: Dict[str, np.ndarray] = {}
        if "/derived" in f:
            der = f["/derived"]
            for name in _DERIVED_DATASETS:
                arr = _read_dataset(der, name)
                if arr is not None:
                    derived_ds[name] = arr
    return SolverFingerprint(
        path=path,
        tau=tau,
        t=t,
        tw=tw,
        resp_shape=resp_shape,
        signals=tuple(signals),
        pathways=tuple(pathways),
        param_attrs=attrs,
        param_datasets=param_ds,
        derived_datasets=derived_ds,
    )


def _as_numeric(a: np.ndarray) -> np.ndarray:
    """Native float/complex view; compound {r,i} → complex128."""
    return from_h5_complex(np.asarray(a))


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = _as_numeric(a)
    b = _as_numeric(b)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.complexfloating) or np.issubdtype(b.dtype, np.complexfloating):
        return np.allclose(a, b, rtol=_ARRAY_RTOL, atol=_ARRAY_ATOL)
    if np.issubdtype(a.dtype, np.floating):
        return np.allclose(a, b, rtol=_ARRAY_RTOL, atol=_ARRAY_ATOL)
    return np.array_equal(a, b)


def _fmt_scalar(v) -> str:
    if np.iscomplexobj(v):
        return f"{complex(v):.6g}"
    return f"{float(np.real(v)):.6g}"


def _array_mismatch_pair(a: np.ndarray, b: np.ndarray) -> Tuple[str, str]:
    """Human-readable ref/other notes; samples are centred on argmax |Δ|."""
    a = _as_numeric(a)
    b = _as_numeric(b)
    if a.shape != b.shape:
        return f"shape={a.shape}", f"shape={b.shape}"
    if a.size == 0:
        return f"shape={a.shape} empty", f"shape={b.shape} empty"
    delta = np.abs(a.astype(np.complex128, copy=False) - b.astype(np.complex128, copy=False))
    max_abs = float(np.max(delta))
    n_diff = int(np.count_nonzero(delta > 0.0))
    flat = int(np.argmax(delta))
    at = tuple(int(i) for i in np.unravel_index(flat, a.shape))
    # Neighbourhood on the ravel around the peak so corner diffs are visible.
    lo = max(0, flat - 1)
    hi = min(a.size, flat + 2)
    prev_a = np.array2string(a.ravel()[lo:hi], precision=6, separator=", ")
    prev_b = np.array2string(b.ravel()[lo:hi], precision=6, separator=", ")
    head = (
        f"shape={a.shape} max|Δ|={max_abs:.6g} at={at} "
        f"n≠={n_diff}/{a.size}"
    )
    return (
        f"{head} peak={_fmt_scalar(a[at])} near={prev_a}",
        f"{head} peak={_fmt_scalar(b[at])} near={prev_b}",
    )


def diff_fingerprints(ref: SolverFingerprint, other: SolverFingerprint) -> List[FieldDiff]:
    """Return structural/physics mismatches (tw axis excluded)."""
    diffs: List[FieldDiff] = []
    ref_label, other_label = ref.path, other.path

    def add(field: str, rv, ov) -> None:
        diffs.append(FieldDiff(field, ref_label, other_label, rv, ov))

    if not _arrays_equal(ref.tau, other.tau):
        rv, ov = _array_mismatch_pair(ref.tau, other.tau)
        add("results/tau_fs", rv, ov)
    if not _arrays_equal(ref.t, other.t):
        rv, ov = _array_mismatch_pair(ref.t, other.t)
        add("results/t_fs", rv, ov)

    ref_static = ref.resp_shape[:4] + ref.resp_shape[5:]
    other_static = other.resp_shape[:4] + other.resp_shape[5:]
    if ref_static != other_static:
        add("results/resp static dims", ref_static, other_static)

    if ref.signals != other.signals:
        add("parameters/signals", ref.signals, other.signals)
    if ref.pathways != other.pathways:
        add("parameters/pathways", ref.pathways, other.pathways)

    keys = sorted(set(ref.param_attrs) | set(other.param_attrs))
    for key in keys:
        rv, ov = ref.param_attrs.get(key), other.param_attrs.get(key)
        if rv != ov:
            add(f"parameters@{key}", rv, ov)

    ds_keys = sorted(set(ref.param_datasets) | set(other.param_datasets))
    for key in ds_keys:
        ra, oa = ref.param_datasets.get(key), other.param_datasets.get(key)
        if ra is None or oa is None:
            add(f"parameters/{key}", "present" if ra is not None else "missing",
                "present" if oa is not None else "missing")
        elif not _arrays_equal(ra, oa):
            rv, ov = _array_mismatch_pair(ra, oa)
            add(f"parameters/{key}", rv, ov)

    der_keys = sorted(set(ref.derived_datasets) | set(other.derived_datasets))
    for key in der_keys:
        ra, oa = ref.derived_datasets.get(key), other.derived_datasets.get(key)
        if ra is None or oa is None:
            add(f"derived/{key}", "present" if ra is not None else "missing",
                "present" if oa is not None else "missing")
        elif not _arrays_equal(ra, oa):
            rv, ov = _array_mismatch_pair(ra, oa)
            add(f"derived/{key}", rv, ov)

    return diffs


def format_diff_report(diffs: List[FieldDiff]) -> str:
    lines = ["twodes-merge: physics/metadata mismatch — merge refused"]
    for d in diffs:
        lines.append(f"  {d.field}:")
        lines.append(f"    ref  ({d.reference}): {d.ref_value!r}")
        lines.append(f"    other ({d.other}): {d.other_value!r}")
    return "\n".join(lines)
