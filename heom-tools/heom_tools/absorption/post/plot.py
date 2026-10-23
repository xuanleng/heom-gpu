"""Matplotlib figures for absorption post-processing."""
from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np

from .core import AbsorptionPostResult
from heom_tools.common.mpl_util import PNG_DPI, draw_axes_monospace_overlay, save_matplotlib_figure, show_matplotlib_figure
from .panel import RunParametersPanel, load_run_parameters_panel
from heom_tools.common.spectrum import ExperimentSpectrum

_SPECTRUM_FIG_WIDTH_IN = 9.0
_SPECTRUM_FIG_HEIGHT_IN = 5.5
_EXP_OVERLAY_EDGE_COLORS = ("r", "darkviolet", "darkgoldenrod", "teal", "brown", "deeppink")
_STICK_LABEL = {
    "absorption": r"stick ($|\mu|^2_{\mathrm{eg}}/3$, no norm)",
    "fluorescence": r"stick ($p_k\,|\mu|^2_{\mathrm{eg}}/3$, no norm)",
}


def _tab10_cmap(plt_mod):
    try:
        return plt_mod.colormaps.get_cmap("tab10")
    except (AttributeError, ValueError, KeyError):
        import matplotlib.cm as mpl_cm
        return mpl_cm.get_cmap("tab10")


def _draw_spectrum_axes(
    ax,
    result: AbsorptionPostResult,
    experiments: Optional[Iterable[ExperimentSpectrum]],
    *,
    show_sticks: bool,
    annotate_peaks: bool,
    normalize: bool,
    plt_mod,
) -> None:
    omega = result.omega_cm
    total = result.intensity_total_norm if normalize else result.intensity_total
    per = result.intensity_per_exciton_norm if normalize else result.intensity_per_exciton

    z_line = 2
    z_total = 4
    z_scatter = 5
    z_peak = 6
    ncol = 2 if result.n_excitons > 3 else 1

    cmap = _tab10_cmap(plt_mod)
    for k in range(result.n_excitons):
        st = result.sticks[k]
        label = (
            f"exc {k + 1} (site {st.dominant_site_index + 1}, "
            f"{st.dominant_site_fraction * 100:.0f}%)"
        )
        ax.plot(omega, per[:, k], lw=1.5, color=cmap(k % 10), label=label, zorder=z_line)

    kind = result.spec.spectrum_type
    ax.plot(
        omega,
        total,
        lw=2.5,
        color="C0",
        label=f"HEOM {kind} (total)",
        zorder=z_total,
    )

    if experiments:
        for ei, experiment in enumerate(experiments):
            c = _EXP_OVERLAY_EDGE_COLORS[ei % len(_EXP_OVERLAY_EDGE_COLORS)]
            ax.scatter(
                experiment.omega_cm,
                experiment.intensity,
                facecolors="none",
                edgecolors=c,
                label=experiment.label,
                zorder=z_scatter,
            )

    if show_sticks and result.sticks:
        omega_sticks = np.array([s.omega_cm for s in result.sticks])
        osc_sticks = np.array([s.oscillator_strength for s in result.sticks])
        width = max(abs(omega[-1] - omega[0]) * 0.005, 1.0)
        ax.bar(
            omega_sticks,
            osc_sticks,
            width=width,
            alpha=0.45,
            color="C2",
            label=_STICK_LABEL.get(kind, _STICK_LABEL["absorption"]),
            zorder=1,
        )
        for st in result.sticks:
            tag = f"site {st.dominant_site_index + 1} ({st.dominant_site_fraction * 100:.0f}%)"
            ax.annotate(
                tag,
                xy=(st.omega_cm, st.oscillator_strength),
                xytext=(2, 4),
                textcoords="offset points",
                fontsize=8,
                color="C2",
                zorder=z_total + 1,
            )
            ax.axvline(st.omega_cm, ls="--", lw=0.7, color="C2", alpha=0.5, zorder=1)

    if annotate_peaks:
        for x, y in result.peaks_total:
            ax.annotate(
                f"{x:.0f}",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                zorder=z_peak + 1,
            )
            ax.plot([x], [y], "*", color="C3", zorder=z_peak)

    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel("Intensity (norm)" if normalize else "Intensity")
    ax.set_title(f"linear {kind} | N={result.spec.N - 1} pigments")
    ax.legend(loc="upper right", fontsize=8, ncol=ncol)
    ax.grid(alpha=0.3)
    ymin, ymax = ax.get_ylim()
    if ymax > ymin:
        pad = max((ymax - ymin) * 0.01, 0.02)
        ax.set_ylim(ymin, ymax + pad)


def _draw_parameters_overlay(ax, panel: RunParametersPanel) -> None:
    draw_axes_monospace_overlay(ax, panel.format_overlay_lines())


def _make_spectrum_figure(
    plt_mod,
    result: AbsorptionPostResult,
    experiments: Optional[Iterable[ExperimentSpectrum]],
    panel: RunParametersPanel,
    *,
    show_sticks: bool,
    annotate_peaks: bool,
    normalize: bool,
):
    fig, ax = plt_mod.subplots(figsize=(_SPECTRUM_FIG_WIDTH_IN, _SPECTRUM_FIG_HEIGHT_IN))
    _draw_spectrum_axes(
        ax,
        result,
        experiments,
        show_sticks=show_sticks,
        annotate_peaks=annotate_peaks,
        normalize=normalize,
        plt_mod=plt_mod,
    )
    _draw_parameters_overlay(ax, panel)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
    return fig


def _make_correlation_figure(plt_mod, result: AbsorptionPostResult):
    import matplotlib.lines as mlines

    t = result.spec.time_fs
    summed = result.ct_iso_sum_check
    total = result.ct_iso_total
    per = result.ct_iso_per_exciton
    fig, (ax_re, ax_im) = plt_mod.subplots(2, 1, figsize=(12, 8.5))

    cmap = _tab10_cmap(plt_mod)
    for k in range(result.n_excitons):
        st = result.sticks[k]
        label_re = (
            fr"$C_{{{k+1}}}(t)$ (site {st.dominant_site_index + 1}, "
            fr"{st.dominant_site_fraction * 100:.0f}\%)"
        )
        c = cmap(k % 10)
        ax_re.plot(t, per[:, k].real, lw=1.4, color=c, alpha=0.95, label=label_re)
        ax_im.plot(t, per[:, k].imag, lw=1.4, color=c, alpha=0.95)

    ax_re.plot(
        t, total.real, "-", lw=3.5, alpha=0.55, color="0.15",
        label=r"$\mathrm{Re}\,C_{\mathrm{iso}}(t)$",
    )
    ax_re.plot(
        t, summed.real, "--", lw=2.0, color="C1",
        label=r"$\mathrm{Re}\,\sum_k C_k(t)$",
    )
    ax_re.set_xlabel("t (fs)")
    ax_re.set_ylabel(r"$\mathrm{Re}\,C(t)$")
    ax_re.set_title(
        "dipole correlation (isotropic) | "
        f"max $|\\sum_k C_k - C_{{\\mathrm{{iso}}}}|$ = {result.consistency_max_error:.2e}"
    )
    ncol = 2 if result.n_excitons > 2 else 1
    ax_re.legend(loc="best", fontsize=7.5, ncol=ncol)
    ax_re.grid(alpha=0.3)

    ax_im.plot(
        t, total.imag, "-", lw=3.5, alpha=0.55, color="0.15",
        label=r"$\mathrm{Im}\,C_{\mathrm{iso}}(t)$",
    )
    ax_im.plot(
        t, summed.imag, "--", lw=2.0, color="C1",
        label=r"$\mathrm{Im}\,\sum_k C_k(t)$",
    )
    ax_im.set_xlabel("t (fs)")
    ax_im.set_ylabel(r"$\mathrm{Im}\,C(t)$")

    h_im, _ = ax_im.get_legend_handles_labels()
    k_note = mlines.Line2D(
        [], [], color="0.45", lw=1.4,
        label=r"$\mathrm{Im}\,C_k(t)$ (colors match $\mathrm{Re}\,C_k$ above)",
    )
    ax_im.legend(handles=list(h_im) + [k_note], loc="best", fontsize=9)
    ax_im.grid(alpha=0.3)
    plt_mod.tight_layout()
    return fig


def plot_absorption_spectrum(
    result: AbsorptionPostResult,
    out_path: str,
    experiments: Optional[Iterable[ExperimentSpectrum]] = None,
    *,
    panel: Optional[RunParametersPanel] = None,
    show_sticks: bool = True,
    annotate_peaks: bool = True,
    normalize: bool = True,
) -> None:
    if panel is None:
        panel = load_run_parameters_panel(result.spec.filename)

    save_matplotlib_figure(
        lambda plt_mod: _make_spectrum_figure(
            plt_mod,
            result,
            experiments,
            panel,
            show_sticks=show_sticks,
            annotate_peaks=annotate_peaks,
            normalize=normalize,
        ),
        out_path,
        dpi=PNG_DPI,
    )


def plot_correlation_figure(
    result: AbsorptionPostResult,
    out_path: str,
) -> None:
    save_matplotlib_figure(
        lambda plt_mod: _make_correlation_figure(plt_mod, result),
        out_path,
        dpi=PNG_DPI,
    )


def show_absorption_plots_interactive(
    result: AbsorptionPostResult,
    experiments: Optional[Iterable[ExperimentSpectrum]] = None,
    *,
    panel: Optional[RunParametersPanel] = None,
    show_sticks: bool = True,
    normalize: bool = True,
) -> None:
    """Open spectrum + correlation figures in one blocking GUI session."""
    if panel is None:
        panel = load_run_parameters_panel(result.spec.filename)

    def draw_both(plt_mod):
        _make_spectrum_figure(
            plt_mod,
            result,
            experiments,
            panel,
            show_sticks=show_sticks,
            annotate_peaks=True,
            normalize=normalize,
        )
        _make_correlation_figure(plt_mod, result)

    show_matplotlib_figure(draw_both)


def write_plots(
    result: AbsorptionPostResult,
    plot_dir: str,
    base: str,
    experiments: Optional[list[ExperimentSpectrum]],
    *,
    normalize: bool = True,
    show_sticks: bool = True,
) -> list[str]:
    """Write ``{base}_spectrum.png`` and ``{base}_corr.png`` (Agg only).

    ``base`` is the raw input stem, e.g. ``foo`` → ``foo_spectrum.png``.
    """
    os.makedirs(plot_dir, exist_ok=True)
    panel = load_run_parameters_panel(result.spec.filename)
    outputs = [
        os.path.join(plot_dir, f"{base}_spectrum.png"),
        os.path.join(plot_dir, f"{base}_corr.png"),
    ]
    plot_absorption_spectrum(
        result,
        outputs[0],
        experiments=experiments,
        panel=panel,
        show_sticks=show_sticks,
        normalize=normalize,
    )
    plot_correlation_figure(result, outputs[1])
    return outputs
