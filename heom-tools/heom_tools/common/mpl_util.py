"""Shared matplotlib helpers for HEOM post-processing packages."""
from __future__ import annotations

import os
from typing import Callable

PNG_DPI = 300

THEORY_LINE_STYLES: tuple[str, ...] = ("-", "--", "-.", ":")
TWIN_YAXIS_COLOR = "0.25"
RESIDUAL_ZERO_LINE_STYLE = ":"

PARAM_OVERLAY_FONT = 8
PARAM_OVERLAY_XY = (0.02, 0.98)
PARAM_OVERLAY_BBOX = {
    "boxstyle": "round,pad=0.35",
    "facecolor": "white",
    "alpha": 0.88,
    "edgecolor": "0.75",
    "linewidth": 0.8,
}


def draw_axes_monospace_overlay(ax, text: str) -> None:
    """Upper-left monospace parameter / annotation box (axes coordinates)."""
    ax.text(
        *PARAM_OVERLAY_XY,
        text.strip() or "(unavailable)",
        transform=ax.transAxes,
        fontsize=PARAM_OVERLAY_FONT,
        verticalalignment="top",
        horizontalalignment="left",
        family="monospace",
        bbox=PARAM_OVERLAY_BBOX,
        zorder=12,
    )


def theory_color(theory_idx: int) -> str:
    return f"C{theory_idx % 10}"


def theory_linestyle(theory_idx: int) -> str:
    return THEORY_LINE_STYLES[theory_idx % len(THEORY_LINE_STYLES)]


def theory_plot_style(theory_idx: int) -> dict[str, str]:
    return {"color": theory_color(theory_idx), "linestyle": theory_linestyle(theory_idx)}


def pyplot(backend: str = "Agg"):
    import matplotlib
    matplotlib.use(backend, force=True)
    import matplotlib.pyplot as plt
    return matplotlib, plt


def save_matplotlib_figure(
    draw_fn: Callable,
    out_path: str,
    *,
    dpi: int = PNG_DPI,
) -> None:
    """Render with Agg and write ``out_path`` (no interactive window)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    _, plt_mod = pyplot("Agg")
    fig = draw_fn(plt_mod)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt_mod.close(fig)


def show_matplotlib_figure(draw_fn: Callable, *, backend: str = "TkAgg") -> None:
    """Open one interactive window; blocks until the user closes it."""
    _, plt_mod = pyplot(backend)
    draw_fn(plt_mod)
    plt_mod.show(block=True)
