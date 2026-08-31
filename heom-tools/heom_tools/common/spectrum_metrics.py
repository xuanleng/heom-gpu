"""Shared spectrum metrics: interpolation, peak normalization, L² relative error."""
from __future__ import annotations

import numpy as np


def _sort_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("omega and intensity length mismatch")
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    return x[order], y[order]


def interp_clamped(xq: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interp with sorted xp; constant extrapolation beyond endpoints."""
    xs, ys = _sort_xy(x, y)
    xq = np.asarray(xq, dtype=np.float64)
    if xs.size < 2:
        return np.full_like(xq, float(ys[0]) if xs.size else np.nan, dtype=np.float64)
    return np.interp(xq, xs, ys)


def peak_normalize_factor(y: np.ndarray) -> float:
    """Max-abs scale for peak normalization; 1.0 when empty or all-zero."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    return peak if peak > 0 else 1.0


def peak_normalize(y: np.ndarray) -> np.ndarray:
    """Divide by :func:`peak_normalize_factor` (max-norm, peak = 1)."""
    return np.asarray(y, dtype=np.float64) / peak_normalize_factor(y)


def l2_relative_diff(
    omega_ref: np.ndarray,
    i_ref: np.ndarray,
    omega_test: np.ndarray,
    i_test: np.ndarray,
) -> float:
    """L² relative difference ‖I_test − I_ref‖₂ / ‖I_ref‖₂ on the reference ω grid.

    The test spectrum is interpolated onto ``omega_ref`` (sorted-safe via
    :func:`interp_clamped`). For metrics on an existing residual grid Δ and
    I_ref already sampled there, use :func:`l2_relative_percent_on_grid`.
    """
    omega_ref = np.asarray(omega_ref, dtype=np.float64).ravel()
    i_ref = np.asarray(i_ref, dtype=np.float64).ravel()
    i_on_ref = interp_clamped(omega_ref, omega_test, i_test)
    denom = float(np.sqrt(np.sum(i_ref ** 2)))
    if denom == 0.0:
        raise ValueError("reference spectrum has zero L² norm")
    return float(np.sqrt(np.sum((i_on_ref - i_ref) ** 2)) / denom)


def l2_relative_percent_on_grid(ref_on_grid: np.ndarray, delta: np.ndarray) -> float:
    """L² rel. error (%) on a common grid: 100 × ‖Δ‖₂ / ‖I_ref‖₂."""
    ref = np.asarray(ref_on_grid, dtype=np.float64).ravel()
    d = np.asarray(delta, dtype=np.float64).ravel()
    denom = float(np.sqrt(np.sum(ref ** 2)))
    if denom == 0.0:
        raise ValueError("reference spectrum has zero L² norm on residual grid")
    return 100.0 * float(np.sqrt(np.sum(d ** 2)) / denom)
