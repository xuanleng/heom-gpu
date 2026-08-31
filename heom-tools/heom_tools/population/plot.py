"""Single-run population dynamics figures."""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .draw import (
    FIG_SIZE_DUAL_PANEL,
    FIG_SIZE_SINGLE,
    configure_matplotlib,
    draw_hamiltonian_heatmap,
    draw_parameters_textbox,
    plot_population_panel,
    save_figure,
)
from .io import parse_population_file
from .models import PopulationData
from .util import apply_time_limit, basename_noext, require_nonempty_time


def plot_dual_basis(
    data: PopulationData,
    output_filename: Optional[str] = None,
    time_limit: Optional[float] = None,
    *,
    show: bool = True,
):
    configure_matplotlib()

    if data.is_dual_basis:
        time_plot, pops_site_plot, pops_exciton_plot = apply_time_limit(
            data.time, time_limit, data.populations_site, data.populations_exciton
        )
    else:
        time_plot, pops_site_plot = apply_time_limit(
            data.time, time_limit, data.populations_site
        )
        pops_exciton_plot = None

    require_nonempty_time(time_plot, time_limit, data_time=data.time)

    n_sites = pops_site_plot.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_sites))

    fig = plt.figure(figsize=FIG_SIZE_DUAL_PANEL)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.98,
        top=0.94,
        bottom=0.04,
        hspace=0.25,
        wspace=0.12,
        height_ratios=[1, 1],
    )

    axes = np.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            axes[r, c] = fig.add_subplot(gs[r, c])

    site_title = "Site Basis" + (
        " (Primary)" if data.primary_basis == "site" else ""
    )
    plot_population_panel(
        axes[0, 0],
        time_plot,
        pops_site_plot,
        data.site_labels,
        site_title,
        colors,
        sink_data=data if data.has_sink else None,
    )

    if data.is_dual_basis and pops_exciton_plot is not None:
        exciton_title = "Exciton Basis" + (
            " (Primary)" if data.primary_basis == "exciton" else ""
        )
        plot_population_panel(
            axes[0, 1],
            time_plot,
            pops_exciton_plot,
            data.exciton_labels or [],
            exciton_title,
            colors,
        )
    else:
        axes[0, 1].axis("off")
        axes[0, 1].text(
            0.5,
            0.5,
            "Exciton data not available",
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
            alpha=0.5,
            transform=axes[0, 1].transAxes,
        )

    if data.hamiltonian_cm is not None:
        draw_hamiltonian_heatmap(axes[1, 0], data.hamiltonian_cm, data.w0_cm)
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(
            0.5,
            0.5,
            "Hamiltonian not available",
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
            alpha=0.5,
            transform=axes[1, 0].transAxes,
        )

    if data.params:
        draw_parameters_textbox(axes[1, 1], data.params)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "Parameters not available",
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
            alpha=0.5,
            transform=axes[1, 1].transAxes,
        )

    fig.suptitle(
        f"Population Dynamics: {os.path.basename(data.filename)}",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    default = f"{basename_noext(data.filename)}_dual_basis_plot.png"
    save_figure(
        fig,
        output_filename,
        default,
        show=show,
        saved_message="2x2 布局图像已保存至: {path}",
    )
    return fig, axes


def plot_single_population(
    filename: str,
    output_filename: Optional[str] = None,
    time_limit: Optional[float] = None,
    *,
    show: bool = True,
    verbose: bool = False,
):
    configure_matplotlib()
    data = parse_population_file(filename, verbose=verbose)

    if filename.endswith(".h5") or filename.endswith(".hdf5") or data.is_dual_basis:
        return plot_dual_basis(data, output_filename, time_limit, show=show)

    time_plot, pops_plot = apply_time_limit(data.time, time_limit, data.populations_site)
    require_nonempty_time(time_plot, time_limit, data_time=data.time)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    colors = plt.cm.tab10(np.linspace(0, 1, data.n_sites))

    metadata: list[str] = []
    if data.temperature:
        metadata.append(f"T = {data.temperature}")
    if data.total_time > 0:
        metadata.append(f"Total time = {data.total_time:.1f} fs")

    plot_population_panel(
        ax,
        time_plot,
        pops_plot,
        data.site_labels,
        f"Population Dynamics: {os.path.basename(filename)}",
        colors,
        sink_data=data if data.has_sink else None,
        metadata_lines=metadata or None,
    )

    plt.tight_layout(pad=2.0)

    default = f"{basename_noext(filename)}_plot.png"
    save_figure(
        fig,
        output_filename,
        default,
        show=show,
        saved_message="图像已保存至: {path}",
    )
    return fig, ax
