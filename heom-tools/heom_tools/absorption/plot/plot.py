"""LAS comparison matplotlib figures."""
from __future__ import annotations

from typing import Optional

from .align import (
    EXP_LEGEND_LABEL,
    LasAligned,
    ResidualCurve,
    ResidualRef,
    format_residual_legend_label,
    max_abs_residual,
)
from heom_tools.common.mpl_util import (
    RESIDUAL_ZERO_LINE_STYLE,
    TWIN_YAXIS_COLOR,
    draw_axes_monospace_overlay,
    save_matplotlib_figure,
    show_matplotlib_figure,
    theory_plot_style,
)

_RESIDUAL_Y_PAD = 1.1
_RESIDUAL_FLOOR_FRAC = 1.0 / 6.0


def _residual_ylabel_left(r: ResidualCurve) -> str:
    if r.ref == "experiment":
        return rf"$\Delta I$ ({r.theory_idx + 1} $-$ {EXP_LEGEND_LABEL})"
    return rf"$\Delta I$ ({r.theory_idx + 1} $-$ 1)"


def _residual_ylabel_right(vs_theory: list[ResidualCurve]) -> str:
    if len(vs_theory) == 1:
        n = vs_theory[0].theory_idx + 1
        return rf"$\Delta I$ ({n} $-$ 1)"
    return r"$\Delta I$ (n $-$ 1)"


def _style_twin_yaxis(ax_bot2, color: str) -> None:
    ax_bot2.set_ylabel(ax_bot2.get_ylabel(), color=color)
    ax_bot2.tick_params(axis="y", colors=color, labelcolor=color)
    ax_bot2.spines["right"].set_color(color)


def _half_span_vs_experiment(residuals: list[ResidualCurve], intensity_span: float) -> float:
    m = max_abs_residual(residuals, "experiment")
    half = m * _RESIDUAL_Y_PAD if m > 0 else 0.0
    half = max(half, intensity_span * _RESIDUAL_FLOOR_FRAC)
    return half if half > 0 else 1.0


def _half_span_vs_first_theory(residuals: list[ResidualCurve]) -> float | None:
    m = max_abs_residual(residuals, "first_theory")
    if m <= 0:
        return None
    return m * _RESIDUAL_Y_PAD


def _residuals_by_ref(residuals: list[ResidualCurve], ref: ResidualRef) -> list[ResidualCurve]:
    return [r for r in residuals if r.ref == ref]


def _plot_residual(
    ax,
    x_plot,
    r: ResidualCurve,
    *,
    label: str,
) -> object:
    style = theory_plot_style(r.theory_idx)
    return ax.plot(x_plot, r.delta, lw=1.5, label=label, **style)[0]


def _render_residual_panel(
    ax_bot,
    *,
    aligned: LasAligned,
    x_plot,
    residuals: list[ResidualCurve],
) -> None:
    import warnings

    vs_exp = _residuals_by_ref(residuals, "experiment")
    vs_theory = _residuals_by_ref(residuals, "first_theory")
    half_r = _half_span_vs_first_theory(residuals)
    use_twin = bool(vs_exp and vs_theory and half_r is not None)

    bot_handles: list = []
    bot_labels: list[str] = []

    if use_twin:
        ax_bot2 = ax_bot.twinx()
        for r in vs_exp:
            label = format_residual_legend_label(r, aligned)
            line = _plot_residual(ax_bot, x_plot, r, label=label)
            bot_handles.append(line)
            bot_labels.append(label)
        for r in vs_theory:
            label = format_residual_legend_label(r, aligned)
            line = _plot_residual(ax_bot2, x_plot, r, label=label)
            bot_handles.append(line)
            bot_labels.append(label)
        half_l = _half_span_vs_experiment(residuals, aligned.intensity_span)
        ax_bot.set_ylim(-half_l, half_l)
        ax_bot2.set_ylim(-half_r, half_r)
        ax_bot.set_ylabel(_residual_ylabel_left(vs_exp[0]))
        ax_bot2.set_ylabel(_residual_ylabel_right(vs_theory))
        _style_twin_yaxis(ax_bot2, TWIN_YAXIS_COLOR)
    else:
        if vs_exp and vs_theory and half_r is None:
            warnings.warn(
                "theory–theory residuals are zero on the residual grid; "
                "using single y-axis (vs experiment scale)",
                UserWarning,
                stacklevel=2,
            )
        for r in residuals:
            label = format_residual_legend_label(r, aligned)
            line = _plot_residual(ax_bot, x_plot, r, label=label)
            bot_handles.append(line)
            bot_labels.append(label)
        if vs_exp:
            half = _half_span_vs_experiment(residuals, aligned.intensity_span)
            ylabel = _residual_ylabel_left(vs_exp[0])
        elif half_r is not None:
            half = half_r
            ylabel = _residual_ylabel_right(vs_theory) if vs_theory else r"$\Delta I$"
        else:
            half = 1.0
            ylabel = r"$\Delta I$"
        ax_bot.set_ylim(-half, half)
        ax_bot.set_ylabel(ylabel)

    if bot_handles:
        ax_bot.legend(bot_handles, bot_labels, loc="upper right", fontsize=8, framealpha=0.9)

    ax_bot.axhline(0.0, color="0.5", lw=0.8, ls=RESIDUAL_ZERO_LINE_STYLE)
    ax_bot.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax_bot.set_xlim(aligned.x_lo, aligned.x_hi)
    ax_bot.grid(alpha=0.3)


def render_las_compare_figure(
    aligned: LasAligned,
    plt_mod,
    *,
    config_diff_text: Optional[str] = None,
    title: Optional[str] = None,
):
    x_plot = aligned.x_residual
    residuals = list(aligned.residuals)
    show_residual = bool(residuals)

    if show_residual:
        fig, (ax_top, ax_bot) = plt_mod.subplots(
            2, 1, sharex=True, figsize=(9, 7.0),
            gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )
    else:
        fig, ax_top = plt_mod.subplots(1, 1, figsize=(9, 5.0), constrained_layout=True)
        ax_bot = None

    if aligned.use_aligned_top:
        if aligned.top_aligned_y is None:
            raise ValueError("use_aligned_top requires top_aligned_y intensities")
        for i, t in enumerate(aligned.theories):
            ax_top.plot(
                x_plot, aligned.top_aligned_y[i], lw=1.8, label=t.label,
                **theory_plot_style(i),
            )
    else:
        for i, t in enumerate(aligned.theories):
            ax_top.plot(
                t.omega_cm, t.intensity, lw=1.8, label=t.label,
                **theory_plot_style(i),
            )

    if aligned.experiment is not None:
        ax_top.scatter(
            aligned.experiment.omega_cm,
            aligned.experiment.intensity,
            s=16, facecolors="none", edgecolors="k", linewidths=0.8,
            label=aligned.experiment.label, zorder=5,
        )

    ax_top.set_xlim(aligned.x_lo, aligned.x_hi)
    ax_top.set_ylabel("intensity (max-norm)")
    if title is None:
        title = (
            "linear absorption comparison" if show_residual else "linear absorption"
        )
    ax_top.set_title(title)
    ax_top.legend(loc="best", fontsize=8)
    ax_top.grid(alpha=0.3)
    if config_diff_text:
        draw_axes_monospace_overlay(ax_top, config_diff_text)

    if ax_bot is not None:
        _render_residual_panel(
            ax_bot,
            aligned=aligned,
            x_plot=x_plot,
            residuals=residuals,
        )
    else:
        ax_top.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    return fig


def plot_las_compare_png(
    aligned: LasAligned,
    out_path: str,
    *,
    config_diff_text: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    save_matplotlib_figure(
        lambda plt_mod: render_las_compare_figure(
            aligned, plt_mod, config_diff_text=config_diff_text, title=title,
        ),
        out_path,
    )


def show_las_compare_interactive(
    aligned: LasAligned,
    *,
    config_diff_text: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    show_matplotlib_figure(
        lambda plt_mod: render_las_compare_figure(
            aligned, plt_mod, config_diff_text=config_diff_text, title=title,
        ),
    )
