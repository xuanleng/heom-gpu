"""Polarization contraction, Tw-slice computation, and FFT grid."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import h5py

from .config import effective_fft_method
from .constants import M_INV, PATHWAY_PLOT_ORDER, PRESETS, STANDARD_PATHWAYS
from .fft import (
    cross_check_two_d_fft,
    freq_axis_cm,
    freq_axis_dw_cm,
    nfft_len,
    project_spec_2d,
    two_d_fft,
)
from .h5_spectra import SpectraComputeNeeds
from .io import read_contracted_tw
from .types import OutputPlan, PathwayPlan, SlabMap, SpecMap, TwSlice


def u_vector(angles_deg):
    t = np.radians(angles_deg)

    def cos_diff(a, b):
        return np.cos(t[a] - t[b])

    return np.array([
        cos_diff(0, 1) * cos_diff(2, 3),
        cos_diff(0, 2) * cos_diff(1, 3),
        cos_diff(0, 3) * cos_diff(1, 2),
    ])


def polarization_weights(pol):
    if isinstance(pol, str):
        key = pol.lower()
        if key == "perpendicular":
            key = "perp"
        if key in PRESETS:
            angles = PRESETS[key]
        else:
            angles = tuple(float(x) for x in pol.split(","))
            if len(angles) != 4:
                raise ValueError(f"pol expects a preset or 'a,b,c,d', got {pol!r}")
    else:
        angles = pol
    # Orientational averaging: contract I1/I2/I3 with u(θ), divide by 30 (see Refs_papers/orientational_averaging).
    return (u_vector(angles) @ M_INV) / 30.0


def ordered_pathways(pathways: List[str]) -> List[str]:
    out = [p for p in PATHWAY_PLOT_ORDER if p in pathways]
    out.extend(p for p in pathways if p not in out)
    return out


def pathways_complete(pathways: List[str]) -> bool:
    return STANDARD_PATHWAYS.issubset(set(pathways))


def pathway_plan(pathways: List[str]) -> PathwayPlan:
    """Pathway sum for totals; plots always show per-pathway rows when present."""
    pidx = list(range(len(pathways)))
    complete = pathways_complete(pathways)
    return PathwayPlan(
        pidx=pidx,
        pathway_sum="all" if complete else ",".join(ordered_pathways(pathways)),
    )


def spec_2des_total(spec_total: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Frequency-domain 2DES: rephasing + nonrephasing totals (same scale)."""
    try:
        return spec_total["rephasing"] + spec_total["nonrephasing"]
    except KeyError as e:
        raise ValueError(
            "2DES spectrum needs both rephasing and nonrephasing signals; "
            f"missing {e.args[0]!r} in {list(spec_total)}"
        ) from e


def combined_freq_rows(
    spec_p: SpecMap,
    spec_t: Dict[str, np.ndarray],
    s2des: Optional[np.ndarray],
    pathways: List[str],
) -> List[Tuple[str, np.ndarray]]:
    rows: List[Tuple[str, np.ndarray]] = []
    if s2des is not None:
        rows.append(("2des", s2des))
    for sig in ("rephasing", "nonrephasing"):
        if sig not in spec_t:
            continue
        rows.append((sig, spec_t[sig]))
        for pname in ordered_pathways(pathways):
            if pname in spec_p and sig in spec_p[pname]:
                rows.append((f"{sig} {pname}", spec_p[pname][sig]))
    return rows


def combined_time_rows(
    time_p: SlabMap,
    time_t: Dict[str, np.ndarray],
    has_2des: bool,
    pathways: List[str],
) -> List[Tuple[str, Dict[str, np.ndarray]]]:
    rows: List[Tuple[str, Dict[str, np.ndarray]]] = []
    if has_2des:
        rows.append(("2des", time_t))
    for pname in ordered_pathways(pathways):
        if pname in time_p:
            rows.append((pname, time_p[pname]))
    return rows


def time_domain_combo(slabs: Dict[str, np.ndarray], tau: np.ndarray):
    rp = slabs.get("rephasing")
    nr = slabs.get("nonrephasing")
    if rp is None and nr is None:
        raise ValueError("time_domain_combo: need at least one signal slab")
    if rp is not None and nr is not None:
        left = np.fliplr(rp.T)
        combo = np.concatenate((left[:, :-1], nr.T), axis=1)
        x = np.concatenate((-np.flip(tau)[:-1], tau))
    elif rp is not None:
        combo, x = np.fliplr(rp.T), -np.flip(tau)
    else:
        combo, x = nr.T, tau
    return x, combo


def response_slab(S: np.ndarray, si: int, pi: int) -> np.ndarray:
    """τ×t slab from per-Tw contracted S[signal, pathway, τ, t]."""
    return S[si, pi, :, :]


def _pathway_indices(
    n_pathways: int,
    needs: SpectraComputeNeeds,
    *,
    need_time_path: bool,
) -> List[int]:
    indices = {pi for pi, _ in needs.pathway_fft}
    if need_time_path:
        indices.update(range(n_pathways))
    return sorted(indices)


def _run_fft(
    slab: np.ndarray,
    sname: str,
    n_tau: int,
    n_t: int,
    *,
    dtau: float,
    dt: float,
    fft_method: str,
    produce: str,
    label: str,
    check_stats: List[Dict[str, float]],
) -> np.ndarray:
    if fft_method == "check":
        spec, stats = cross_check_two_d_fft(
            slab, sname, n_tau, n_t, dtau=dtau, dt=dt, label=label)
        check_stats.append(stats)
        return spec
    return two_d_fft(slab, sname, n_tau, n_t, method=produce, dtau=dtau, dt=dt)


def _print_fft_check_summary(k: int, check_stats: List[Dict[str, float]]) -> None:
    if not check_stats:
        return
    sep_max = max(s["axis_vs_separable"] for s in check_stats)
    m3_max = max(s["axis_vs_legacy_mode3"] for s in check_stats)
    print(
        f"fft check tw{k}: n={len(check_stats)} separable_max={sep_max:.2e} "
        f"legacy_mode3_max={m3_max:.2e}"
    )


def _maybe_project(
    spec: np.ndarray,
    *,
    w_tau_src: Optional[np.ndarray],
    w_t_src: Optional[np.ndarray],
    w_tau_dst: Optional[np.ndarray],
    w_t_dst: Optional[np.ndarray],
) -> np.ndarray:
    if w_tau_dst is None:
        return spec
    assert w_tau_src is not None and w_t_src is not None and w_t_dst is not None
    return project_spec_2d(spec, w_tau_src, w_t_src, w_tau_dst, w_t_dst)


def compute_tw_slice(
    S: np.ndarray,
    signals: List[str],
    pathways: List[str],
    pidx: List[int],
    k: int,
    n_tau: int,
    n_t: int,
    *,
    needs: SpectraComputeNeeds,
    need_time: bool,
    need_time_path: bool,
    fft_method: str = "axis",
    dtau: float = 1.0,
    dt: float = 1.0,
    w_tau_src: Optional[np.ndarray] = None,
    w_t_src: Optional[np.ndarray] = None,
    w_tau_dst: Optional[np.ndarray] = None,
    w_t_dst: Optional[np.ndarray] = None,
) -> TwSlice:
    produce = effective_fft_method(fft_method)
    check_stats: List[Dict[str, float]] = []
    sig_index = {name: i for i, name in enumerate(signals)}
    path_fft_by_pi: Dict[int, set] = {}
    for pi, sname in needs.pathway_fft:
        path_fft_by_pi.setdefault(pi, set()).add(sname)

    def _fft_and_project(slab, sname, label):
        spec = _run_fft(
            slab, sname, n_tau, n_t,
            dtau=dtau, dt=dt,
            fft_method=fft_method, produce=produce,
            label=label, check_stats=check_stats,
        )
        return _maybe_project(
            spec,
            w_tau_src=w_tau_src, w_t_src=w_t_src,
            w_tau_dst=w_tau_dst, w_t_dst=w_t_dst,
        )

    spec_path: SpecMap = {}
    time_path: SlabMap = {}
    for pi in _pathway_indices(len(pathways), needs, need_time_path=need_time_path):
        pname = pathways[pi]
        fft_signals = path_fft_by_pi.get(pi, set())
        if need_time_path:
            time_path[pname] = {}
        if fft_signals:
            spec_path[pname] = {}
        for si, sname in enumerate(signals):
            slab = response_slab(S, si, pi)
            if need_time_path:
                time_path[pname][sname] = slab
            if sname in fft_signals:
                spec_path[pname][sname] = _fft_and_project(
                    slab, sname, f"tw{k} {pname}",
                )

    spec_total: Dict[str, np.ndarray] = {}
    time_total: Dict[str, np.ndarray] = {}
    for sname in needs.total_fft:
        si = sig_index[sname]
        slab = sum(response_slab(S, si, p) for p in pidx)
        if need_time:
            time_total[sname] = slab
        spec_total[sname] = _fft_and_project(slab, sname, f"tw{k} total")

    if fft_method == "check":
        _print_fft_check_summary(k, check_stats)

    spec_2des = None
    if needs.compute_2des and {"rephasing", "nonrephasing"}.issubset(spec_total):
        spec_2des = spec_2des_total(spec_total)

    return TwSlice(
        k=k,
        spec_path=spec_path,
        spec_total=spec_total,
        spec_2des=spec_2des,
        time_path=time_path if need_time_path else None,
        time_total=time_total if need_time else None,
    )


def stack_spectra(
    slices: Dict[int, TwSlice],
    tw_order: List[int],
    signals: List[str],
):
    if not tw_order:
        return {}, {}
    first = slices[tw_order[0]]
    path_keys = list(first.spec_path.keys())
    path_signals = {
        p: [s for s in signals if s in first.spec_path.get(p, {})]
        for p in path_keys
    }
    total_signals = list(first.spec_total.keys())
    by_path = {p: {s: [] for s in path_signals[p]} for p in path_keys}
    total = {s: [] for s in total_signals}
    s2des = []
    for k in tw_order:
        sl = slices[k]
        for p in path_keys:
            for s in path_signals[p]:
                by_path[p][s].append(sl.spec_path[p][s])
        for s in total_signals:
            total[s].append(sl.spec_total[s])
        if sl.spec_2des is not None:
            s2des.append(sl.spec_2des)
    for p in path_keys:
        for s in path_signals[p]:
            by_path[p][s] = np.stack(by_path[p][s], axis=0)
    for s in total_signals:
        total[s] = np.stack(total[s], axis=0)
    out_total: Dict[str, Any] = dict(total)
    if s2des:
        out_total["2des"] = np.stack(s2des, axis=0)
    return by_path, out_total


def require_fft_time_axes(tau: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
    if len(tau) < 2:
        raise ValueError(f"tau_fs needs >= 2 points for FFT spacing, got {len(tau)}")
    if len(t) < 2:
        raise ValueError(f"t_fs needs >= 2 points for FFT spacing, got {len(t)}")
    dtau = float(tau[1] - tau[0])
    dt = float(t[1] - t[0])
    if dtau <= 0.0:
        raise ValueError("tau_fs must be strictly increasing")
    if dt <= 0.0:
        raise ValueError("t_fs must be strictly increasing")
    if len(tau) > 2 and not np.allclose(np.diff(tau), dtau):
        raise ValueError("tau_fs must be uniformly spaced")
    if len(t) > 2 and not np.allclose(np.diff(t), dt):
        raise ValueError("t_fs must be uniformly spaced")
    return dtau, dt


def fft_grid(tau: np.ndarray, t: np.ndarray, *, dw: float, w0: float):
    """Pad lengths for the native FFT plus the published ``w0+k·dw`` axes.

    ``dw`` is the *output* lattice step, not the independent Fourier resolve
    (that is ``FS_TO_CM/(n_data·dt)``; see runtime ``ω resolve`` log line).

    Returns
    -------
    n_tau, n_t :
        Native zero-pad lengths (FFT work size).
    w_tau, w_t :
        Exact ``dw`` lattices (what HDF5 / plots / compares use).
    dtau, dt :
        Time-grid spacings (fs).
    w_tau_native, w_t_native :
        Native ``fftfreq`` axes; used only to project onto ``(w_tau, w_t)``.
    """
    from .config import print_fft_grid_info

    dtau, dt = require_fft_time_axes(tau, t)
    n_tau_data = len(tau)
    n_t_data = len(t)
    n_tau = nfft_len(n_tau_data, dtau, dw)
    n_t = nfft_len(n_t_data, dt, dw)
    w_tau_native = freq_axis_cm(n_tau, dtau, w0)
    w_t_native = freq_axis_cm(n_t, dt, w0)
    w_tau = freq_axis_dw_cm(dw, w0, n_tau, dtau)
    w_t = freq_axis_dw_cm(dw, w0, n_t, dt)
    print_fft_grid_info(
        dtau=dtau,
        dt=dt,
        n_tau_data=n_tau_data,
        n_tau_fft=n_tau,
        n_t_data=n_t_data,
        n_t_fft=n_t,
        dw=dw,
        w0=w0,
        n_w_tau=len(w_tau),
        n_w_t=len(w_t),
    )
    return n_tau, n_t, w_tau, w_t, dtau, dt, w_tau_native, w_t_native


def build_tw_slices(
    h5: str,
    weights: np.ndarray,
    signals: List[str],
    pathways: List[str],
    pidx: List[int],
    tw_indices: List[int],
    n_tau: int,
    n_t: int,
    plan: OutputPlan,
    fft_method: str,
    *,
    dtau: float,
    dt: float,
    w_tau_src: Optional[np.ndarray] = None,
    w_t_src: Optional[np.ndarray] = None,
    w_tau_dst: Optional[np.ndarray] = None,
    w_t_dst: Optional[np.ndarray] = None,
) -> Dict[int, TwSlice]:
    slices: Dict[int, TwSlice] = {}
    with h5py.File(h5, "r") as f:
        resp_ds = f["/results/resp"]
        for k in tw_indices:
            s_k = read_contracted_tw(resp_ds, k, weights)
            slices[k] = compute_tw_slice(
                s_k, signals, pathways, pidx, k, n_tau, n_t,
                needs=plan.spectra_needs,
                need_time=plan.need_time,
                need_time_path=plan.need_time_path,
                fft_method=fft_method,
                dtau=dtau, dt=dt,
                w_tau_src=w_tau_src, w_t_src=w_t_src,
                w_tau_dst=w_tau_dst, w_t_dst=w_t_dst,
            )
    return slices
