"""Frequency-domain 2D map panel drawing."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .norm import make_norm


def apply_freq_window(ax, window):
    lo, hi = float(window[0]), float(window[1])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    return lo, hi


def draw_freq_panel(
    ax,
    w_tau: np.ndarray,
    w_t: np.ndarray,
    data2d: np.ndarray,
    *,
    title: str,
    mpl,
    plt_mod,
    cmap,
    levels: int,
    lines: List[float],
    window: Optional[List[float]],
    nonnegative: bool = False,
    norm=None,
    colorbar: bool = True,
) -> None:
    """Draw one frequency-domain 2D map (data already window-cut)."""
    z = np.asarray(data2d, dtype=np.float64).T
    if norm is None:
        norm = make_norm(z, mpl, nonnegative=nonnegative)
    if window:
        lo, hi = float(window[0]), float(window[1])
        extent = [lo, hi, lo, hi]
        diag_lo, diag_hi = apply_freq_window(ax, window)
    else:
        extent = [w_tau.min(), w_tau.max(), w_t.min(), w_t.max()]
        diag_lo = max(w_tau.min(), w_t.min())
        diag_hi = min(w_tau.max(), w_t.max())
    ax.imshow(
        z, origin="lower", aspect="auto", cmap=cmap, norm=norm,
        extent=extent, interpolation="spline36",
    )
    ax.contour(w_tau, w_t, z, levels=levels, colors="k", linewidths=0.4)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k-", lw=1.0)
    for v in lines:
        ax.axvline(v, color="k", lw=0.7)
        ax.axhline(v, color="k", lw=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\omega_\tau$ (cm$^{-1}$)")
    ax.set_ylabel(r"$\omega_t$ (cm$^{-1}$)")
    ax.set_title(title)
    if colorbar:
        cb = plt_mod.colorbar(ax.images[-1], ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)
