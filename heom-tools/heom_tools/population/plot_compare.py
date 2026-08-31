"""Two-run population comparison figures."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .analysis import (
    create_comparison_result,
    population_l2_rel_percent,
    print_analysis,
    windowed_series,
)
from .config_diff import draw_params_diff_overlay, load_compare_diff_overlay
from .draw import (
    FIG_SIZE_COMPARE,
    add_sink_curves,
    configure_matplotlib,
    plot_comparison_panel,
    plot_difference_panel,
    refresh_legend,
    save_figure,
)
from .io import parse_population_file
from .models import ComparisonResult
from .util import basename_noext


def compare_dual_basis(
    result: ComparisonResult,
    output_filename: Optional[str] = None,
    time_limit: Optional[float] = None,
    *,
    show: bool = True,
    verbose: bool = False,
):
    if not result.has_exciton:
        raise ValueError("Result must include exciton series for dual-basis comparison")

    configure_matplotlib()
    assert result.exciton is not None

    time_plot, site, exciton = windowed_series(result, time_limit)
    assert exciton is not None

    n_sites = site.pops1.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_sites))

    file1_name = basename_noext(result.data1.filename)
    file2_name = basename_noext(result.data2.filename)

    site_labels = result.data1.site_labels[:n_sites]
    exciton_labels = (result.data1.exciton_labels or [])[:n_sites]

    fig = plt.figure(figsize=FIG_SIZE_COMPARE)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.06,
        hspace=0.28,
        wspace=0.15,
    )

    axes = np.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            axes[r, c] = fig.add_subplot(gs[r, c])

    plot_comparison_panel(
        axes[0, 0],
        time_plot,
        site.pops1,
        site.pops2,
        site_labels,
        colors,
        title="Site Basis",
        show_xlabel=False,
        show_ylabel=True,
    )

    if add_sink_curves(axes[0, 0], result.data1, result.data2, time_limit):
        refresh_legend(axes[0, 0], n_sites, fontsize=6)

    diff_overlay = load_compare_diff_overlay(
        result.data1.filename,
        result.data2.filename,
        result.data1.params,
        result.data2.params,
    )
    draw_params_diff_overlay(axes[0, 0], diff_overlay)

    plot_comparison_panel(
        axes[0, 1],
        time_plot,
        exciton.pops1,
        exciton.pops2,
        exciton_labels,
        colors,
        title="Exciton Basis",
        show_xlabel=False,
        show_ylabel=False,
    )

    plot_difference_panel(
        axes[1, 0],
        time_plot,
        site.diff,
        site_labels,
        colors,
        title="Difference (2 − 1) · Site",
        show_xlabel=True,
        show_ylabel=True,
        l2_percent=population_l2_rel_percent(site.pops1, site.pops2),
    )

    plot_difference_panel(
        axes[1, 1],
        time_plot,
        exciton.diff,
        exciton_labels,
        colors,
        title="Difference (2 − 1) · Exciton",
        show_xlabel=True,
        show_ylabel=False,
        l2_percent=population_l2_rel_percent(exciton.pops1, exciton.pops2),
    )

    fig.suptitle(
        f"(1 —){file1_name}  |  (2 - -){file2_name}",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )

    print_analysis(result, time_limit=time_limit, verbose=verbose)

    default = f"{file1_name}_vs_{file2_name}_dual_comparison.png"
    save_figure(
        fig,
        output_filename,
        default,
        show=show,
        saved_message="双表象对比图已保存至: {path}",
    )
    return fig, axes


def compare_populations(
    file1: str,
    file2: str,
    output_filename: Optional[str] = None,
    time_limit: Optional[float] = None,
    *,
    show: bool = True,
    verbose: bool = False,
):
    configure_matplotlib()
    data1 = parse_population_file(file1, verbose=False)
    data2 = parse_population_file(file2, verbose=False)

    result = create_comparison_result(data1, data2)

    if result.has_exciton:
        return compare_dual_basis(
            result,
            output_filename,
            time_limit,
            show=show,
            verbose=verbose,
        )

    time_plot, site, _ = windowed_series(result, time_limit)
    n_sites = site.pops1.shape[1]
    labels = data1.site_labels[:n_sites]

    fig = plt.figure(figsize=FIG_SIZE_COMPARE)
    axes = np.empty(2, dtype=object)
    axes[0] = fig.add_subplot(2, 1, 1)
    axes[1] = fig.add_subplot(2, 1, 2)

    file1_name = basename_noext(file1)
    file2_name = basename_noext(file2)

    colors = plt.cm.tab10(np.linspace(0, 1, n_sites))

    plot_comparison_panel(
        axes[0],
        time_plot,
        site.pops1,
        site.pops2,
        labels,
        colors,
        title="Population Dynamics Comparison",
    )
    if add_sink_curves(axes[0], data1, data2, time_limit):
        refresh_legend(axes[0], n_sites)

    diff_overlay = load_compare_diff_overlay(
        file1, file2, data1.params, data2.params,
    )
    draw_params_diff_overlay(axes[0], diff_overlay)

    l2_percent = population_l2_rel_percent(site.pops1, site.pops2)
    plot_difference_panel(
        axes[1],
        time_plot,
        site.diff,
        labels,
        colors,
        title=f"Population Difference (2 − 1): {file2_name} − {file1_name}",
        l2_percent=l2_percent,
    )

    fig.text(
        0.5,
        0.01,
        f"Line style:  (1) Solid (—) = {file1_name}  |  (2) Dashed (- -) = {file2_name}",
        ha="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85, pad=0.5),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1], pad=2.5, h_pad=3.0)

    print_analysis(result, time_limit=time_limit, verbose=verbose)

    default = f"{file1_name}_vs_{file2_name}_comparison.png"
    save_figure(
        fig,
        output_filename,
        default,
        show=show,
        saved_message="对比图已保存至: {path}",
    )
    return fig, axes
