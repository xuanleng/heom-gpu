"""Matplotlib figures for post-processed 2DES HDF5."""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np

from heom_tools.twodes.post.cmap import asymmetric_diverging_cmap, diverging_cmap
from heom_tools.twodes.viz import draw_freq_panel

from .core import (
    ComparePlotRequest,
    CompareSlices,
    PlotNormConfig,
    PlotRequest,
    PlotSlice,
    SinglePlotRequest,
    SliceSeriesRequest,
    materialize_slices,
    norm_figure_caption,
    prepare_plot_normalization,
    shared_re_range,
)
from .io import spectrum_plot_label

_FIG_DPI = 200
_ANIM_DPI = 150
_SINGLE_FIGSIZE = (6.5, 5.5)
_MULTI_PANEL_WIDTH = 4.6
_MULTI_FIGSIZE_H = 5.5
_COMPARE_FIGSIZE = (14.0, 4.8)
_RESIDUAL_PAD = 1.1


def _panel_title(tw_fs: float, pol: str, spectrum: str) -> str:
    return f"T={tw_fs:g} fs  {pol}  {spectrum_plot_label(spectrum)}"


def _freq_span_cm(slices: List[PlotSlice], window) -> float:
    if window is not None:
        return abs(float(window[1]) - float(window[0]))
    sl = slices[0]
    return float(np.max(sl.w_tau) - np.min(sl.w_tau))


def _multi_x_tick_step(span_cm: float, n_panels: int) -> float:
    """Pick a round cm^-1 step for ~4–6 x ticks; scale down when many panels."""
    n_ticks = max(4, min(6, 10 - n_panels))
    rough = span_cm / max(n_ticks - 1, 1)
    for step in (50.0, 100.0, 200.0, 250.0, 500.0):
        if step >= rough:
            return step
    return 500.0


def _apply_multi_axis_ticks(
    axes,
    plt_mod,
    *,
    n_panels: int,
    span_cm: float,
) -> None:
    step = _multi_x_tick_step(span_cm, n_panels)
    locator = plt_mod.matplotlib.ticker.MultipleLocator(step)
    axes[0].xaxis.set_major_locator(locator)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)


def _pick_colorbar_step(raw: float) -> float:
    if raw <= 0.0:
        return 1.0
    if raw >= 1.0:
        exp = 10.0 ** np.floor(np.log10(raw))
        for mult in (1.0, 2.0, 5.0, 10.0):
            step = mult * exp
            if step >= raw * 0.9:
                return step
        return 10.0 * exp
    for step in (0.05, 0.1, 0.2, 0.25, 0.5):
        if step >= raw * 0.9:
            return step
    return max(float(raw), 0.05)


def _shared_color_scale(vmin: float, vmax: float, mpl):
    """Diverging scale on [vmin, vmax]; white at 0 when 0 lies in range."""
    lo, hi = float(vmin), float(vmax)
    base = diverging_cmap()
    if hi <= lo:
        return base, mpl.colors.Normalize(vmin=lo, vmax=hi or lo + 1.0)
    if lo < 0.0 < hi:
        cmap = asymmetric_diverging_cmap(base, lo, hi, vcenter=0.0)
    elif lo >= 0.0:
        cmap = asymmetric_diverging_cmap(base, lo, hi, vcenter=lo)
    else:
        cmap = asymmetric_diverging_cmap(base, lo, hi, vcenter=hi)
    return cmap, mpl.colors.Normalize(vmin=lo, vmax=hi)


def _colorbar_ticks(vmin: float, vmax: float) -> np.ndarray:
    """Ticks from data min/max; include 0 when it lies in range."""
    lo, hi = float(vmin), float(vmax)
    if hi <= lo:
        return np.array([lo])
    if lo < 0.0 < hi:
        step = _pick_colorbar_step((hi - lo) / 5.0)
        n_pos = int(np.floor(hi / step + 1e-9))
        n_neg = int(np.floor(-lo / step + 1e-9))
        ticks = [step * k for k in range(-n_neg, n_pos + 1)]
        out = np.array(sorted(set(ticks)), dtype=np.float64)
        out[np.abs(out) < 1e-12] = 0.0
        if lo not in out:
            out = np.insert(out, 0, lo)
        if hi not in out:
            out = np.append(out, hi)
        return np.array(sorted(set(out)), dtype=np.float64)
    return np.linspace(lo, hi, 5)


def _format_colorbar_tick(t: float, span: float) -> str:
    if abs(t) < 1e-12:
        return "0.00"
    if span >= 10.0:
        return f"{t:.0f}"
    if span >= 1.0:
        return f"{t:.2f}"
    return f"{t:.3f}"


def _attach_fixed_colorbar(cax, plt_mod, *, norm, cmap, ticks: np.ndarray):
    """Colorbar decoupled from panel artists; ticks are the only y labels."""
    mpl = plt_mod.matplotlib
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt_mod.colorbar(sm, cax=cax)
    ticks = np.asarray(ticks, dtype=np.float64)
    span = max(abs(float(ticks[0])), abs(float(ticks[-1]))) if ticks.size else 1.0
    labels = [_format_colorbar_tick(t, span) for t in ticks]
    axis = cbar.ax
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(FixedFormatter(labels))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.tick_params(which="both", length=3)
    return cbar


def _stamp_norm_caption(fig, norm: PlotNormConfig, *, spectrum: str) -> None:
    fig.text(
        0.5,
        0.01,
        norm_figure_caption(norm, spectrum=spectrum),
        ha="center",
        va="bottom",
        fontsize=8,
        transform=fig.transFigure,
    )


def _finish_figure_layout(fig) -> None:
    """Leave room for the bottom norm caption without tight_layout/colorbar fights."""
    fig.subplots_adjust(bottom=0.10)


def _agg_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_figure(fig, out_path: str, plt_mod) -> None:
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt_mod.close(fig)


def render_single_figure(
    sl: PlotSlice,
    plt_mod,
    *,
    pol: str,
    spectrum: str,
    levels: int,
    lines: List[float],
    window,
    norm_config: PlotNormConfig,
):
    prepared = prepare_plot_normalization(
        [sl], norm_config, spectrum=spectrum,
    )
    sl = prepared.slices[0]
    cmap, norm = _shared_color_scale(prepared.vmin, prepared.vmax, plt_mod.matplotlib)
    fig, ax = plt_mod.subplots(1, 1, figsize=_SINGLE_FIGSIZE)
    title = _panel_title(sl.tw_fs, pol, spectrum)
    draw_freq_panel(
        ax, sl.w_tau, sl.w_t, sl.data, title=title,
        mpl=plt_mod.matplotlib, plt_mod=plt_mod,
        cmap=cmap, levels=levels, lines=lines, window=window,
        nonnegative=False, norm=norm,
    )
    _stamp_norm_caption(fig, norm_config, spectrum=spectrum)
    _finish_figure_layout(fig)
    return fig, prepared.norm_log


def render_multi_figure(
    slices: List[PlotSlice],
    plt_mod,
    *,
    pol: str,
    spectrum: str,
    levels: int,
    lines: List[float],
    window,
    norm_config: PlotNormConfig,
):
    if not slices:
        raise ValueError("multi plot requires at least one waiting time")
    prepared = prepare_plot_normalization(
        slices, norm_config, spectrum=spectrum,
    )
    slices = prepared.slices
    disp_lo, disp_hi = prepared.vmin, prepared.vmax
    ticks = _colorbar_ticks(disp_lo, disp_hi)
    cmap, norm = _shared_color_scale(disp_lo, disp_hi, plt_mod.matplotlib)
    n = len(slices)
    fig_w = _MULTI_PANEL_WIDTH * n + 0.7
    fig = plt_mod.figure(figsize=(fig_w, _MULTI_FIGSIZE_H))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1.0] * n + [0.05], wspace=0.08)
    axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, n):
        axes.append(fig.add_subplot(gs[0, i], sharey=axes[0], sharex=axes[0]))
    cax = fig.add_subplot(gs[0, n])
    for i, (ax, sl) in enumerate(zip(axes, slices)):
        draw_freq_panel(
            ax, sl.w_tau, sl.w_t, sl.data, title=_panel_title(sl.tw_fs, pol, spectrum),
            mpl=plt_mod.matplotlib, plt_mod=plt_mod,
            cmap=cmap, levels=levels, lines=lines, window=window,
            nonnegative=False, norm=norm, colorbar=False,
        )
        if i > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
    _apply_multi_axis_ticks(
        axes, plt_mod, n_panels=n, span_cm=_freq_span_cm(slices, window),
    )
    _attach_fixed_colorbar(cax, plt_mod, norm=norm, cmap=cmap, ticks=ticks)
    _stamp_norm_caption(fig, norm_config, spectrum=spectrum)
    _finish_figure_layout(fig)
    return fig, prepared.norm_log


def render_compare_figure(
    cmp: CompareSlices,
    plt_mod,
    *,
    spectrum: str,
    levels: int,
    lines: List[float],
    window: Optional[List[float]],
):
    spec_label = spectrum_plot_label(spectrum)
    ab_lo, ab_hi = shared_re_range([cmp.a, cmp.b])
    ab_cmap, ab_norm = _shared_color_scale(ab_lo, ab_hi, plt_mod.matplotlib)
    div_cmap = diverging_cmap()
    fig, axes = plt_mod.subplots(1, 3, figsize=_COMPARE_FIGSIZE)
    panels = (
        (cmp.a, spec_label),
        (cmp.b, spec_label),
    )
    for ax, (sl, title) in zip(axes[:2], panels):
        draw_freq_panel(
            ax, sl.w_tau, sl.w_t, sl.data, title=title,
            mpl=plt_mod.matplotlib, plt_mod=plt_mod,
            cmap=ab_cmap, levels=levels, lines=lines, window=window,
            nonnegative=False, norm=ab_norm,
        )
    peak = float(np.max(np.abs(cmp.delta))) if cmp.delta.size else 0.0
    residual_norm = None
    if peak > 0.0:
        v = peak * _RESIDUAL_PAD
        residual_norm = plt_mod.matplotlib.colors.Normalize(vmin=-v, vmax=v)
    draw_freq_panel(
        axes[2], cmp.a.w_tau, cmp.a.w_t, cmp.delta,
        title="Δ (A − B)",
        mpl=plt_mod.matplotlib, plt_mod=plt_mod,
        cmap=div_cmap, levels=levels, lines=lines, window=window,
        nonnegative=False, norm=residual_norm,
    )
    fig.suptitle(f"T={cmp.a.tw_fs:g} fs", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_plot(
    request: PlotRequest,
    plt_mod,
    *,
    levels: int,
    lines: List[float],
    window,
    norm_config: PlotNormConfig,
):
    if isinstance(request, SinglePlotRequest):
        fig, norm_log = render_single_figure(
            request.slice, plt_mod,
            pol=request.meta.pol, spectrum=request.meta.spectrum,
            levels=levels, lines=lines, window=window,
            norm_config=norm_config,
        )
        return fig, norm_log
    if isinstance(request, SliceSeriesRequest):
        slices = materialize_slices(request, window)
        fig, norm_log = render_multi_figure(
            slices, plt_mod,
            pol=request.meta.pol, spectrum=request.meta.spectrum,
            levels=levels, lines=lines, window=window,
            norm_config=norm_config,
        )
        return fig, norm_log
    if isinstance(request, ComparePlotRequest):
        fig = render_compare_figure(
            request.compare, plt_mod, spectrum=request.meta.spectrum,
            levels=levels, lines=lines, window=window,
        )
        return fig, []
    raise TypeError(f"unsupported plot request type: {type(request)!r}")


def write_plot_png(
    request: PlotRequest,
    out_path: str,
    *,
    levels: Optional[int] = None,
    window=None,
    norm_config: Optional[PlotNormConfig] = None,
) -> List[str]:
    meta = request.meta
    lv = int(levels) if levels is not None else meta.levels
    win = window if window is not None else meta.window
    norm = norm_config or PlotNormConfig()
    plt = _agg_plt()
    fig, norm_log = render_plot(
        request, plt, levels=lv, lines=meta.lines, window=win, norm_config=norm,
    )
    _save_figure(fig, out_path, plt)
    return norm_log


def _draw_panel_no_colorbar(
    ax,
    sl: PlotSlice,
    plt_mod,
    *,
    pol: str,
    spectrum: str,
    cmap,
    norm,
    levels: int,
    lines: List[float],
    window,
) -> None:
    title = _panel_title(sl.tw_fs, pol, spectrum)
    draw_freq_panel(
        ax, sl.w_tau, sl.w_t, sl.data, title=title,
        mpl=plt_mod.matplotlib, plt_mod=plt_mod,
        cmap=cmap, levels=levels, lines=lines, window=window,
        nonnegative=False, norm=norm, colorbar=False,
    )


def write_animation(
    request: SliceSeriesRequest,
    out_path: str,
    *,
    fmt: str,
    fps: float,
    interval_ms: Optional[int] = None,
    levels: Optional[int] = None,
    window=None,
    norm_config: Optional[PlotNormConfig] = None,
) -> List[str]:
    """Export spectrum animation with the same --norm modes as static multi-T plots."""
    import matplotlib.animation as animation

    meta = request.meta
    n_frames = len(request.selection)
    if n_frames < 2:
        raise ValueError("animation needs at least 2 frames")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Axes3D.*",
            category=UserWarning,
        )

    lv = int(levels) if levels is not None else meta.levels
    win = window if window is not None else meta.window
    norm = norm_config or PlotNormConfig()
    slices = materialize_slices(request, win)
    prepared = prepare_plot_normalization(
        slices, norm, spectrum=meta.spectrum,
    )
    disp_lo, disp_hi = prepared.vmin, prepared.vmax
    ticks = _colorbar_ticks(disp_lo, disp_hi)
    plt = _agg_plt()
    cmap, color_norm = _shared_color_scale(disp_lo, disp_hi, plt.matplotlib)

    fig = plt.figure(figsize=_SINGLE_FIGSIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    sl0 = prepared.slices[0]
    _draw_panel_no_colorbar(
        ax, sl0, plt, pol=meta.pol, spectrum=meta.spectrum,
        cmap=cmap, norm=color_norm,
        levels=lv, lines=meta.lines, window=win,
    )
    _attach_fixed_colorbar(cax, plt, norm=color_norm, cmap=cmap, ticks=ticks)
    _stamp_norm_caption(fig, norm, spectrum=meta.spectrum)
    _finish_figure_layout(fig)

    play_fps = max(float(fps), 0.1)
    frame_interval = (
        int(interval_ms) if interval_ms is not None
        else int(1000.0 / play_fps)
    )

    def _update(frame: int):
        ax.clear()
        sl = prepared.slices[frame]
        _draw_panel_no_colorbar(
            ax, sl, plt, pol=meta.pol, spectrum=meta.spectrum,
            cmap=cmap, norm=color_norm,
            levels=lv, lines=meta.lines, window=win,
        )
        return ()

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=frame_interval, blit=False,
    )

    fmt_l = str(fmt).lower().lstrip(".")
    try:
        if fmt_l == "mp4":
            writer = animation.FFMpegWriter(fps=play_fps, dpi=_ANIM_DPI)
            ani.save(out_path, writer=writer)
        elif fmt_l == "gif":
            writer = animation.PillowWriter(fps=play_fps)
            ani.save(out_path, writer=writer, dpi=_ANIM_DPI)
        else:
            raise ValueError(f"unsupported animation format {fmt!r} (use mp4 or gif)")
    except (RuntimeError, OSError, ValueError) as exc:
        if fmt_l == "mp4":
            raise RuntimeError(
                "mp4 export failed (install ffmpeg and ensure it is on PATH)",
            ) from exc
        raise
    finally:
        plt.close(fig)
    return prepared.norm_log
