"""Slice selection, windowing, normalization, grid alignment (no matplotlib)."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from heom_tools.twodes.viz import cut_window

from .io import Post2desFile, load_post_2des_h5, read_spectrum_planes, spectrum_plot_label

_TW_ATOL_FS = 1e-6
_PEAK_FLOOR = 1e-30


@dataclass(frozen=True)
class PlotSlice:
    """One waiting-time Re(spectrum) map ready to draw."""

    label: str
    tw_fs: float
    w_tau: np.ndarray
    w_t: np.ndarray
    data: np.ndarray


@dataclass(frozen=True)
class CompareSlices:
    """Raw pair and difference on a common grid."""

    a: PlotSlice
    b: PlotSlice
    delta: np.ndarray
    resampled_b: bool


@dataclass(frozen=True)
class DiagonalNormRef:
    """Reference Re at a shared (i_tau, i_t) grid point for one waiting-time slice."""

    i_tau: int
    i_t: int
    w_tau_cm: float
    w_t_cm: float
    re_value: float
    method: str


@dataclass(frozen=True)
class PlotNormConfig:
    """Display scaling: default shared raw min/max, or per-T diagonal reference."""

    diagonal: bool = False
    ref_freq: Optional[float] = None

    def validate(self) -> None:
        if not self.diagonal and self.ref_freq is not None:
            raise ValueError("--norm-ref-freq requires --norm diagonal")


def norm_figure_caption(norm: PlotNormConfig, *, spectrum: str = "2des") -> str:
    """Short on-figure label for the active normalization mode."""
    label = spectrum_plot_label(spectrum)
    if not norm.diagonal:
        return f"Colorbar: shared min/max {label} across T"
    if norm.ref_freq is not None:
        return (
            f"Norm: each T / diagonal Re at {norm.ref_freq:g} cm$^{{-1}}$ "
            f"(shared grid)"
        )
    return "Norm: each T / Re at first-panel max diagonal (shared grid)"


@dataclass(frozen=True)
class PreparedPlotData:
    slices: List[PlotSlice]
    vmin: float
    vmax: float
    norm_log: List[str]


@dataclass(frozen=True)
class SinglePlotRequest:
    meta: Post2desFile
    slice: PlotSlice


@dataclass(frozen=True)
class SliceSeriesRequest:
    """Ordered waiting-time selection; HDF5 planes loaded when rendering."""

    meta: Post2desFile
    selection: List[Tuple[int, float]]


@dataclass(frozen=True)
class ComparePlotRequest:
    meta: Post2desFile
    compare: CompareSlices


PlotRequest = Union[SinglePlotRequest, SliceSeriesRequest, ComparePlotRequest]


def _tw_values_text(tw_fs: np.ndarray) -> str:
    tw = np.asarray(tw_fs, dtype=np.float64).ravel()
    return ", ".join(f"{v:g}" for v in tw)


def resolve_tw_by_fs(tw_fs: np.ndarray, tw_fs_target: float) -> int:
    """Return index where tw_fs equals tw_fs_target (strict, no nearest-neighbor)."""
    tw = np.asarray(tw_fs, dtype=np.float64).ravel()
    if tw.size == 0:
        raise ValueError("empty tw_fs axis")
    target = float(tw_fs_target)
    hits = np.flatnonzero(np.isclose(tw, target, rtol=0.0, atol=_TW_ATOL_FS))
    if hits.size == 0:
        raise ValueError(
            f"T={target:g} fs not in tw_fs [{_tw_values_text(tw)}]; "
            f"--tw must match exactly",
        )
    if hits.size > 1:
        raise ValueError(
            f"T={target:g} fs matches multiple tw_fs entries in "
            f"[{_tw_values_text(tw)}]",
        )
    return int(hits[0])


def resolve_tw_selection(
    tw_fs: np.ndarray,
    targets: List[float],
) -> List[Tuple[int, float]]:
    """Map each --tw target to (index, canonical tw_fs); reject duplicates."""
    tw = np.asarray(tw_fs, dtype=np.float64).ravel()
    seen: set[float] = set()
    out: List[Tuple[int, float]] = []
    for target in targets:
        k = resolve_tw_by_fs(tw_fs, target)
        v = float(tw[k])
        if v in seen:
            raise ValueError(f"duplicate --tw {v:g} fs")
        seen.add(v)
        out.append((k, v))
    return out


def resolve_tw_list(tw_fs: np.ndarray, targets: List[float]) -> List[float]:
    return [v for _, v in resolve_tw_selection(tw_fs, targets)]


def default_tw_fs(tw_fs: np.ndarray) -> float:
    tw = np.asarray(tw_fs, dtype=np.float64).ravel()
    if tw.size == 1:
        return float(tw[0])
    raise ValueError(
        f"multiple waiting times ({tw.size}); pass --tw (fs). "
        f"Available: {_tw_values_text(tw)}",
    )


def all_tw_selection(meta: Post2desFile) -> List[Tuple[int, float]]:
    """All waiting times in ascending tw_fs order."""
    tw_arr = np.asarray(meta.tw_fs, dtype=np.float64).ravel()
    order = np.argsort(tw_arr)
    return resolve_tw_selection(meta.tw_fs, [float(tw_arr[i]) for i in order])


def _axes_match(x_a: np.ndarray, x_b: np.ndarray, *, atol: float = 1e-6) -> bool:
    a = np.asarray(x_a, dtype=np.float64).ravel()
    b = np.asarray(x_b, dtype=np.float64).ravel()
    return a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=0.0)


def resample_2d(
    w_tau_src: np.ndarray,
    w_t_src: np.ndarray,
    z: np.ndarray,
    w_tau_dst: np.ndarray,
    w_t_dst: np.ndarray,
) -> np.ndarray:
    """Bilinear resample onto destination grid; destination must lie in source span.

    Delegates to ``project_spec_2d`` (same kernel as twodes-post). Out-of-range
    destination axes raise — no silent endpoint extrapolation.
    """
    from heom_tools.twodes.post.fft import project_spec_2d

    z = np.asarray(z)
    out = project_spec_2d(
        np.asarray(z, dtype=np.complex128),
        np.asarray(w_tau_src, dtype=np.float64).ravel(),
        np.asarray(w_t_src, dtype=np.float64).ravel(),
        np.asarray(w_tau_dst, dtype=np.float64).ravel(),
        np.asarray(w_t_dst, dtype=np.float64).ravel(),
    )
    if np.iscomplexobj(z):
        return out
    return np.asarray(out.real, dtype=np.float64)


def shared_re_range(slices: List[PlotSlice]) -> Tuple[float, float]:
    """Finite (min, max) of Re data across slices in one pass; (0.0, 1.0) when empty."""
    lo, hi = np.inf, -np.inf
    for sl in slices:
        arr = np.asarray(sl.data, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            v = float(np.min(finite))
            if v < lo:
                lo = v
            v = float(np.max(finite))
            if v > hi:
                hi = v
    if np.isinf(lo):
        return 0.0, 1.0
    return float(lo), float(hi)


def _nearest_axis_index(axis: np.ndarray, target: float) -> int:
    arr = np.asarray(axis, dtype=np.float64).ravel()
    return int(np.argmin(np.abs(arr - float(target))))


def _diagonal_band_samples(
    w_tau: np.ndarray,
    w_t: np.ndarray,
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w_tau = np.asarray(w_tau, dtype=np.float64).ravel()
    w_t = np.asarray(w_t, dtype=np.float64).ravel()
    z = np.asarray(data, dtype=np.float64)
    i_tau = np.arange(w_tau.size, dtype=np.int64)
    i_t = np.array([_nearest_axis_index(w_t, wt) for wt in w_tau], dtype=np.int64)
    vals = z[i_tau, i_t]
    return i_tau, i_t, vals, 0.5 * (w_tau + w_t[i_t])


def resolve_diagonal_reference(
    w_tau: np.ndarray,
    w_t: np.ndarray,
    data: np.ndarray,
    *,
    target_freq: Optional[float] = None,
) -> DiagonalNormRef:
    """Pick a diagonal reference index (ω_τ≈ω_t) for per-T normalization."""
    if target_freq is not None:
        freq = float(target_freq)
        i_tau = _nearest_axis_index(w_tau, freq)
        i_t = _nearest_axis_index(w_t, freq)
        method = f"frequency {freq:g} cm^-1"
    else:
        i_tau_all, i_t_all, vals, w_mid = _diagonal_band_samples(w_tau, w_t, data)
        k = int(np.argmax(np.abs(vals)))
        i_tau = int(i_tau_all[k])
        i_t = int(i_t_all[k])
        method = f"max Re on diagonal near {w_mid[k]:g} cm^-1"
    wtau = float(np.asarray(w_tau, dtype=np.float64).ravel()[i_tau])
    wt = float(np.asarray(w_t, dtype=np.float64).ravel()[i_t])
    re_val = float(np.asarray(data, dtype=np.float64)[i_tau, i_t])
    return DiagonalNormRef(
        i_tau=i_tau,
        i_t=i_t,
        w_tau_cm=wtau,
        w_t_cm=wt,
        re_value=re_val,
        method=method,
    )


def normalize_slice_at_grid_ref(
    sl: PlotSlice,
    i_tau: int,
    i_t: int,
    *,
    method: str,
) -> Tuple[PlotSlice, DiagonalNormRef]:
    """Divide slice by its own Re at a fixed (ω_τ, ω_t) grid index."""
    z = np.asarray(sl.data, dtype=np.float64)
    wtau = float(np.asarray(sl.w_tau, dtype=np.float64).ravel()[i_tau])
    wt = float(np.asarray(sl.w_t, dtype=np.float64).ravel()[i_t])
    re_val = float(z[i_tau, i_t])
    ref = DiagonalNormRef(
        i_tau=i_tau,
        i_t=i_t,
        w_tau_cm=wtau,
        w_t_cm=wt,
        re_value=re_val,
        method=method,
    )
    if abs(re_val) <= _PEAK_FLOOR:
        raise ValueError(
            f"T={sl.tw_fs:g} fs: diagonal reference Re={re_val:g} at "
            f"({wtau:g}, {wt:g}) cm^-1 is too small to normalize",
        )
    inv = 1.0 / re_val
    return PlotSlice(
        label=sl.label,
        tw_fs=sl.tw_fs,
        w_tau=sl.w_tau,
        w_t=sl.w_t,
        data=z * inv,
    ), ref


def format_diagonal_norm_log(
    refs: List[Tuple[float, DiagonalNormRef]],
    norm: PlotNormConfig,
    *,
    anchor_tw_fs: float,
    anchor_ref: DiagonalNormRef,
) -> List[str]:
    out = [
        "normalization: diagonal (shared grid from first panel; "
        "each T divided by its own Re at that point)",
    ]
    if norm.ref_freq is not None:
        out.append(
            f"  reference grid: nearest diagonal to {norm.ref_freq:g} cm^-1 "
            f"(from T={anchor_tw_fs:g} fs)",
        )
    else:
        out.append(
            f"  reference grid: max diagonal Re of first panel "
            f"(T={anchor_tw_fs:g} fs) at "
            f"ω_τ={anchor_ref.w_tau_cm:g}, ω_t={anchor_ref.w_t_cm:g} cm^-1 "
            f"({anchor_ref.method})",
        )
    for tw_fs, ref in refs:
        out.append(
            f"  T={tw_fs:g} fs: divide by Re={ref.re_value:g} at "
            f"ω_τ={ref.w_tau_cm:g}, ω_t={ref.w_t_cm:g} cm^-1 "
            f"(grid {ref.i_tau},{ref.i_t}) → ref maps to 1.0",
        )
    return out


def prepare_plot_normalization(
    slices: List[PlotSlice],
    norm: PlotNormConfig,
    *,
    spectrum: str = "2des",
) -> PreparedPlotData:
    """Apply display normalization and shared colorbar limits."""
    norm.validate()
    if not slices:
        raise ValueError("plot normalization requires at least one slice")

    if not norm.diagonal:
        vmin, vmax = shared_re_range(slices)
        label = spectrum_plot_label(spectrum)
        log = [
            f"display: raw {label}, shared colorbar",
            f"  vmin={vmin:g}  vmax={vmax:g}",
        ]
        return PreparedPlotData(
            slices=list(slices),
            vmin=vmin,
            vmax=vmax,
            norm_log=log,
        )

    target_freq = float(norm.ref_freq) if norm.ref_freq is not None else None
    anchor = slices[0]
    anchor_ref = resolve_diagonal_reference(
        anchor.w_tau, anchor.w_t, anchor.data, target_freq=target_freq,
    )
    refs: List[Tuple[float, DiagonalNormRef]] = []
    out: List[PlotSlice] = []
    for i, sl in enumerate(slices):
        method = anchor_ref.method if i == 0 else (
            f"shared grid from first panel (T={anchor.tw_fs:g} fs)"
        )
        scaled, ref = normalize_slice_at_grid_ref(
            sl, anchor_ref.i_tau, anchor_ref.i_t, method=method,
        )
        out.append(scaled)
        refs.append((sl.tw_fs, ref))
    vmin, vmax = shared_re_range(out)
    return PreparedPlotData(
        slices=out,
        vmin=vmin,
        vmax=vmax,
        norm_log=format_diagonal_norm_log(
            refs, norm, anchor_tw_fs=anchor.tw_fs, anchor_ref=anchor_ref,
        ),
    )


def warn_compare_metadata_mismatch(a: Post2desFile, b: Post2desFile) -> None:
    if a.pol != b.pol:
        warnings.warn(
            f"compare: pol differs ({a.label}={a.pol!r} vs {b.label}={b.pol!r}); "
            f"Δ map mixes polarization schemes — using first file labels",
            UserWarning,
            stacklevel=2,
        )
    if (
        a.dw is not None and b.dw is not None
        and not np.isclose(float(a.dw), float(b.dw), rtol=0.0, atol=0.0)
    ):
        warnings.warn(
            f"compare: dw differs ({a.dw:g} vs {b.dw:g} cm^-1); "
            f"grids may need resample",
            UserWarning,
            stacklevel=2,
        )
    if (
        a.w0_cm is not None and b.w0_cm is not None
        and not np.isclose(float(a.w0_cm), float(b.w0_cm), rtol=0.0, atol=0.0)
    ):
        warnings.warn(
            f"compare: w0_cm differs ({a.w0_cm:g} vs {b.w0_cm:g}); "
            f"frequency frames may disagree",
            UserWarning,
            stacklevel=2,
        )
    if a.window != b.window:
        warnings.warn(
            f"compare: window differs ({a.label}={a.window!r} vs {b.label}={b.window!r}); "
            f"using first file",
            UserWarning,
            stacklevel=2,
        )
    if a.levels != b.levels:
        warnings.warn(
            f"compare: levels differ ({a.levels} vs {b.levels}); using first file",
            UserWarning,
            stacklevel=2,
        )
    if a.lines != b.lines:
        warnings.warn(
            f"compare: lines differ; using first file",
            UserWarning,
            stacklevel=2,
        )


def _slice_from_plane(
    meta: Post2desFile,
    *,
    tw_fs: float,
    plane: np.ndarray,
    window: Optional[List[float]],
) -> PlotSlice:
    win = window if window is not None else meta.window
    x, y, d = cut_window(meta.w_tau, meta.w_t, np.real(plane), win)
    return PlotSlice(
        label=meta.label,
        tw_fs=tw_fs,
        w_tau=x,
        w_t=y,
        data=d,
    )


def materialize_slices(
    request: SliceSeriesRequest,
    window: Optional[List[float]],
) -> List[PlotSlice]:
    indices = [k for k, _ in request.selection]
    planes = read_spectrum_planes(request.meta, indices)
    return [
        _slice_from_plane(request.meta, tw_fs=v, plane=plane, window=window)
        for (_, v), plane in zip(request.selection, planes)
    ]


def extract_real_slice(
    meta: Post2desFile,
    *,
    tw_fs: float,
    window: Optional[List[float]],
) -> PlotSlice:
    k, v = resolve_tw_selection(meta.tw_fs, [tw_fs])[0]
    plane = read_spectrum_planes(meta, [k])[0]
    return _slice_from_plane(meta, tw_fs=v, plane=plane, window=window)


def align_to_reference(
    ref: PlotSlice,
    other: PlotSlice,
) -> Tuple[PlotSlice, bool]:
    """Resample *other* onto *ref* frequency grid when axes differ."""
    if _axes_match(ref.w_tau, other.w_tau) and _axes_match(ref.w_t, other.w_t):
        return other, False
    warnings.warn(
        f"frequency grids differ ({other.label} vs {ref.label}); "
        f"resampling {other.label} onto {ref.label}",
        UserWarning,
        stacklevel=2,
    )
    data = resample_2d(other.w_tau, other.w_t, other.data, ref.w_tau, ref.w_t)
    return PlotSlice(
        label=other.label,
        tw_fs=other.tw_fs,
        w_tau=ref.w_tau,
        w_t=ref.w_t,
        data=data,
    ), True


def prepare_compare(
    a: Post2desFile,
    b: Post2desFile,
    *,
    tw_fs: float,
    window: Optional[List[float]],
) -> CompareSlices:
    warn_compare_metadata_mismatch(a, b)
    k_a, v_a = resolve_tw_selection(a.tw_fs, [tw_fs])[0]
    k_b, v_b = resolve_tw_selection(b.tw_fs, [tw_fs])[0]
    plane_a, plane_b = read_spectrum_planes(a, [k_a])[0], read_spectrum_planes(b, [k_b])[0]
    sl_a = _slice_from_plane(a, tw_fs=v_a, plane=plane_a, window=window)
    sl_b = _slice_from_plane(b, tw_fs=v_b, plane=plane_b, window=window)
    if not np.isclose(sl_a.tw_fs, sl_b.tw_fs, rtol=0.0, atol=_TW_ATOL_FS):
        raise ValueError(
            f"compare: resolved T differs ({sl_a.label} T={sl_a.tw_fs:g} fs vs "
            f"{sl_b.label} T={sl_b.tw_fs:g} fs) for --tw {tw_fs:g}",
        )
    sl_b_aligned, resampled = align_to_reference(sl_a, sl_b)
    delta = sl_a.data - sl_b_aligned.data
    return CompareSlices(a=sl_a, b=sl_b_aligned, delta=delta, resampled_b=resampled)


def build_series_request(
    path: str,
    tw: Optional[List[float]],
    window: Optional[List[float]],
    *,
    spectrum: str = "2des",
    all_tw_if_none: bool,
) -> SliceSeriesRequest:
    """Build ordered tw selection; does not load spectrum planes."""
    ds = load_post_2des_h5(path, spectrum=spectrum)
    if tw is None:
        selection = (
            all_tw_selection(ds) if all_tw_if_none
            else resolve_tw_selection(ds.tw_fs, [default_tw_fs(ds.tw_fs)])
        )
    else:
        selection = resolve_tw_selection(ds.tw_fs, [float(t) for t in tw])
    return SliceSeriesRequest(meta=ds, selection=selection)


def build_animation_request(
    path: str,
    tw: Optional[List[float]],
    window: Optional[List[float]],
    *,
    spectrum: str = "2des",
) -> SliceSeriesRequest:
    """Backward-compatible alias for animation series (all tw_fs when --tw omitted)."""
    return build_series_request(
        path, tw, window, spectrum=spectrum, all_tw_if_none=True,
    )


def build_plot_request(
    path: str,
    tw: Optional[List[float]],
    window: Optional[List[float]],
    *,
    spectrum: str = "2des",
) -> Union[SinglePlotRequest, SliceSeriesRequest]:
    """Load one post HDF5, resolve --tw, return a static plot request."""
    path = os.path.abspath(path)
    ds = load_post_2des_h5(path, spectrum=spectrum)
    tw_list = [float(t) for t in tw] if tw is not None else [default_tw_fs(ds.tw_fs)]
    if len(tw_list) == 1:
        sl = extract_real_slice(ds, tw_fs=tw_list[0], window=window)
        return SinglePlotRequest(meta=ds, slice=sl)
    selection = resolve_tw_selection(ds.tw_fs, tw_list)
    return SliceSeriesRequest(meta=ds, selection=selection)


def build_compare_request(
    path_a: str,
    path_b: str,
    tw: Optional[List[float]],
    window: Optional[List[float]],
    *,
    spectrum: str = "2des",
) -> ComparePlotRequest:
    """Load two post HDF5 files and build a compare request."""
    path_a = os.path.abspath(path_a)
    path_b = os.path.abspath(path_b)
    if tw is not None and len(tw) != 1:
        raise ValueError("compare accepts exactly one --tw value")
    ds_a = load_post_2des_h5(path_a, spectrum=spectrum)
    ds_b = load_post_2des_h5(path_b, spectrum=spectrum)
    tw_fs = float(tw[0]) if tw is not None else default_tw_fs(ds_a.tw_fs)
    cmp = prepare_compare(ds_a, ds_b, tw_fs=tw_fs, window=window)
    return ComparePlotRequest(meta=ds_a, compare=cmp)


def tw_summary(request: PlotRequest) -> str:
    if isinstance(request, SinglePlotRequest):
        return f"{request.slice.tw_fs:g}"
    if isinstance(request, SliceSeriesRequest):
        return ", ".join(f"{v:g}" for _, v in request.selection)
    return f"{request.compare.a.tw_fs:g}"
