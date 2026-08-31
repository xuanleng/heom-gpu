"""LAS spectrum alignment, residuals, and metrics (no matplotlib).

Naming (legend, config-diff overlay):
  ``A vs B`` = from baseline A looking at subject B.
  Arrow (config overlay): ``value(A) -> value(B)``.
  Δ formula is always explicit on the axis (not encoded only in ``vs`` order):
    theory–theory: ``1 vs 2`` → ΔI = I₂ − I₁, axis ``ΔI (2 − 1)``;
    theory–experiment: ``1 vs exp`` → ΔI = I₁ − I_exp, axis ``ΔI (1 − exp)``.

L² relative error in residual-panel legends uses the same ``x_residual`` grid
as ΔI and the terminal MAE/RMSE table: ‖Δ‖₂ / ‖I_ref‖₂ on that grid.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import h5py
import numpy as np

from heom_tools.common.spectrum import ExperimentSpectrum
from heom_tools.common.spectrum_metrics import (
    interp_clamped,
    l2_relative_percent_on_grid,
    peak_normalize,
)

_DENSE_GRID_MIN_POINTS = 512

EXP_LEGEND_LABEL = "exp"

ResidualRef = Literal["experiment", "first_theory"]


@dataclass(frozen=True)
class TheorySpectrum:
    label: str
    omega_cm: np.ndarray
    intensity: np.ndarray


@dataclass(frozen=True)
class ResidualCurve:
    theory_label: str
    theory_idx: int
    delta: np.ndarray
    ref: ResidualRef
    ref_label: str


@dataclass(frozen=True)
class LasAligned:
    theories: tuple[TheorySpectrum, ...]
    baseline_label: str
    experiment: Optional[ExperimentSpectrum]
    x_residual: np.ndarray
    x_lo: float
    x_hi: float
    intensity_span: float
    residuals: list[ResidualCurve]
    use_aligned_top: bool
    top_aligned_y: Optional[list[np.ndarray]]


def load_theory_from_spectrum_h5(path: str) -> TheorySpectrum:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    label = os.path.basename(path)
    with h5py.File(path, "r") as f:
        if "omega_cm" not in f:
            raise ValueError(f"{path}: missing dataset 'omega_cm' (need *_spectrum.h5)")
        omega = np.asarray(f["omega_cm"][...], dtype=np.float64).ravel()
        if "intensity_total_norm" in f:
            y = np.asarray(f["intensity_total_norm"][...], dtype=np.float64).ravel()
        elif "intensity_total" in f:
            y_raw = np.asarray(f["intensity_total"][...], dtype=np.float64).ravel()
            y = peak_normalize(y_raw)
        else:
            raise ValueError(f"{path}: need intensity_total_norm or intensity_total")
    if omega.shape != y.shape:
        raise ValueError(f"{path}: omega_cm shape {omega.shape} != intensity shape {y.shape}")
    return TheorySpectrum(label=label, omega_cm=omega, intensity=y)


def _ref_intensity_on_residual_grid(r: ResidualCurve, aligned: LasAligned) -> np.ndarray:
    x = aligned.x_residual
    if r.ref == "experiment":
        if aligned.experiment is None:
            raise ValueError("experiment reference without experiment spectrum")
        exp = aligned.experiment
        return interp_clamped(x, exp.omega_cm, exp.intensity)
    baseline = aligned.theories[0]
    return interp_clamped(x, baseline.omega_cm, baseline.intensity)


def residual_l2_rel_percent(r: ResidualCurve, aligned: LasAligned) -> float:
    return l2_relative_percent_on_grid(
        _ref_intensity_on_residual_grid(r, aligned),
        r.delta,
    )


def l2_percent_vs_experiment(aligned: LasAligned) -> Optional[float]:
    """Baseline theory L² vs experiment on ``x_residual``, or ``None`` if absent."""
    for r in aligned.residuals:
        if r.ref == "experiment" and r.theory_idx == 0:
            return residual_l2_rel_percent(r, aligned)
    return None


def l2_percent_vs_baseline(aligned: LasAligned) -> dict[int, float]:
    """Subject theory index → L² vs baseline theory (indices start at 1)."""
    out: dict[int, float] = {}
    for r in aligned.residuals:
        if r.ref == "first_theory":
            out[r.theory_idx] = residual_l2_rel_percent(r, aligned)
    return out


def format_residual_legend_label(r: ResidualCurve, aligned: LasAligned) -> str:
    base = residual_curve_label(r)
    pct = residual_l2_rel_percent(r, aligned)
    return f"{base} (L² {pct:.1f}%)"


def _diff_metrics(diff: np.ndarray) -> dict[str, float]:
    d = np.asarray(diff, dtype=np.float64).ravel()
    if d.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan")}
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    max_abs = float(np.max(np.abs(d)))
    return {"mae": mae, "rmse": rmse, "max_abs": max_abs}


def max_abs_residual(residuals: list[ResidualCurve], ref: ResidualRef) -> float:
    m = 0.0
    for r in residuals:
        if r.ref != ref:
            continue
        arr = np.asarray(r.delta, dtype=np.float64).ravel()
        if arr.size:
            m = max(m, float(np.max(np.abs(arr))))
    return m


def _uniq_sorted_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    xs = np.sort(x)
    return xs[np.concatenate([[True], np.diff(xs) > 0])]


def _theory_intensity_span(
    theories: tuple[TheorySpectrum, ...] | list[TheorySpectrum],
    w_lo: float,
    w_hi: float,
) -> float:
    peak = 0.0
    for t in theories:
        om = np.asarray(t.omega_cm, dtype=np.float64).ravel()
        y = np.asarray(t.intensity, dtype=np.float64).ravel()
        m = (om >= w_lo) & (om <= w_hi)
        if np.any(m):
            peak = max(peak, float(np.max(y[m])))
    return peak if peak > 0 else 1.0


def residual_curve_label(r: ResidualCurve) -> str:
    if r.ref == "experiment":
        # Theory (subject) vs experiment (reference); Δ = I_theory − I_exp.
        return f"{r.theory_idx + 1} vs {EXP_LEGEND_LABEL}"
    # Baseline-first: from theory 1 looking at this curve; Δ = I_n − I_1.
    return f"1 vs {r.theory_idx + 1}"


def _ref_display_name(ref: ResidualRef, ref_label: str) -> str:
    if ref == "experiment":
        return f"experiment ({ref_label})"
    return f"first theory ({ref_label})"


def print_metrics_table(omega: np.ndarray, residuals: list[ResidualCurve]) -> None:
    print(f"[LAS compare] residuals (common grid, n={omega.size})", flush=True)
    print(
        f"{'curve':<32} {'reference':<28} {'MAE':>12} {'RMSE':>12} {'max|Δ|':>12}",
        flush=True,
    )
    for r in residuals:
        m = _diff_metrics(r.delta)
        curve = residual_curve_label(r)
        ref = _ref_display_name(r.ref, r.ref_label)
        if len(ref) > 27:
            ref = ref[:24] + "..."
        print(
            f"{curve:<32} {ref:<28} {m['mae']:12.5e} {m['rmse']:12.5e} {m['max_abs']:12.5e}",
            flush=True,
        )


def _omega_union_in_window(
    theories: tuple[TheorySpectrum, ...] | list[TheorySpectrum],
    w_lo: float,
    w_hi: float,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for t in theories:
        om = np.asarray(t.omega_cm, dtype=np.float64).ravel()
        m = (om >= w_lo) & (om <= w_hi)
        if np.any(m):
            parts.append(om[m])
    if not parts:
        return np.array([], dtype=np.float64)
    return np.unique(np.concatenate(parts))


def _residual_grid_theory_only(
    theories: tuple[TheorySpectrum, ...] | list[TheorySpectrum],
) -> np.ndarray:
    """Union of all theory ω samples (no baseline-only special case)."""
    parts: list[np.ndarray] = []
    for t in theories:
        om = np.asarray(t.omega_cm, dtype=np.float64).ravel()
        if om.size:
            parts.append(om)
    if not parts:
        raise ValueError("no ω samples in theory spectra")
    return np.unique(np.concatenate(parts))


def _aligned_top_intensities(
    theories: list[TheorySpectrum],
    x_ref: np.ndarray,
) -> list[np.ndarray]:
    return [
        interp_clamped(x_ref, t.omega_cm, t.intensity) for t in theories
    ]


def _residual_grid_with_experiment(
    theories: list[TheorySpectrum],
    experiment: ExperimentSpectrum,
) -> np.ndarray:
    w_lo = float(np.min(experiment.omega_cm))
    w_hi = float(np.max(experiment.omega_cm))
    x_res = _omega_union_in_window(theories, w_lo, w_hi)
    if x_res.size >= 2:
        return x_res
    n_exp_u = _uniq_sorted_x(experiment.omega_cm).size
    n_pts = max(_DENSE_GRID_MIN_POINTS, 4 * max(n_exp_u, 2))
    warnings.warn(
        "no theory ω samples in the experimental ω range (or grid has <2 points); "
        f"residuals use a dense linear grid ({n_pts} points) in "
        f"[{w_lo:.4g}, {w_hi:.4g}] cm⁻¹; interpolate experiment onto this grid",
        UserWarning,
        stacklevel=2,
    )
    return np.linspace(w_lo, w_hi, int(n_pts), dtype=np.float64)


def _build_residuals_with_experiment(
    theories: list[TheorySpectrum],
    experiment: ExperimentSpectrum,
    x_res: np.ndarray,
) -> list[ResidualCurve]:
    baseline = theories[0]
    y_exp_r = interp_clamped(x_res, experiment.omega_cm, experiment.intensity)
    y0 = interp_clamped(x_res, baseline.omega_cm, baseline.intensity)
    out: list[ResidualCurve] = [
        ResidualCurve(
            theory_label=baseline.label,
            theory_idx=0,
            delta=y0 - y_exp_r,
            ref="experiment",
            ref_label=experiment.label,
        )
    ]
    for k, t in enumerate(theories[1:], start=1):
        yt = interp_clamped(x_res, t.omega_cm, t.intensity)
        out.append(
            ResidualCurve(
                theory_label=t.label,
                theory_idx=k,
                delta=yt - y0,
                ref="first_theory",
                ref_label=baseline.label,
            )
        )
    return out


def _build_residuals_theory_only(
    theories: list[TheorySpectrum], x_ref: np.ndarray,
) -> list[ResidualCurve]:
    baseline = theories[0]
    y0 = interp_clamped(x_ref, baseline.omega_cm, baseline.intensity)
    out: list[ResidualCurve] = []
    for k, t in enumerate(theories[1:], start=1):
        yi = interp_clamped(x_ref, t.omega_cm, t.intensity)
        out.append(
            ResidualCurve(
                theory_label=t.label,
                theory_idx=k,
                delta=yi - y0,
                ref="first_theory",
                ref_label=baseline.label,
            )
        )
    return out


def align_las_compare(
    theories: list[TheorySpectrum],
    experiment: Optional[ExperimentSpectrum],
) -> LasAligned:
    if not theories:
        raise ValueError("no theory spectra")

    baseline = theories[0]

    if experiment is not None:
        x_res = _residual_grid_with_experiment(theories, experiment)
        w_lo = float(np.min(experiment.omega_cm))
        w_hi = float(np.max(experiment.omega_cm))
        return LasAligned(
            theories=tuple(theories),
            baseline_label=baseline.label,
            experiment=experiment,
            x_residual=x_res,
            x_lo=w_lo,
            x_hi=w_hi,
            intensity_span=_theory_intensity_span(theories, w_lo, w_hi),
            residuals=_build_residuals_with_experiment(theories, experiment, x_res),
            use_aligned_top=False,
            top_aligned_y=None,
        )

    x_ref = _residual_grid_theory_only(theories)
    w_lo = float(x_ref[0])
    w_hi = float(x_ref[-1])
    intensity_rows = _aligned_top_intensities(theories, x_ref)
    residuals = (
        []
        if len(theories) < 2
        else _build_residuals_theory_only(theories, x_ref)
    )
    return LasAligned(
        theories=tuple(theories),
        baseline_label=baseline.label,
        experiment=None,
        x_residual=x_ref,
        x_lo=w_lo,
        x_hi=w_hi,
        intensity_span=_theory_intensity_span(theories, w_lo, w_hi),
        residuals=residuals,
        use_aligned_top=True,
        top_aligned_y=intensity_rows,
    )
