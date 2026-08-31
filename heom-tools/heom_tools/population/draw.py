"""Shared matplotlib drawing helpers for population figures."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .analysis import format_compare_l2_label
from .models import PopulationData
from .util import apply_time_limit, legend_cols

FIG_SIZE_SINGLE = (12, 7)
FIG_SIZE_COMPARE = (14, 10)
FIG_SIZE_DUAL_PANEL = (14, 14)
DPI = 300

_MPL_CONFIGURED = False


def configure_matplotlib() -> None:
    """Idempotent rcParams setup (call from figure entry points, not on import)."""
    global _MPL_CONFIGURED
    if _MPL_CONFIGURED:
        return
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    _MPL_CONFIGURED = True


def save_figure(
    fig,
    output_filename: Optional[str],
    default_name: str,
    *,
    show: bool,
    saved_message: str,
) -> str:
    if output_filename is None:
        output_filename = default_name
    fig.savefig(output_filename, dpi=DPI, bbox_inches="tight")
    print(saved_message.format(path=output_filename))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_filename


def plot_sink_overlay(
    ax,
    time: np.ndarray,
    N_captured: np.ndarray,
    Phi: float,
    t50: float,
) -> None:
    ax.plot(
        time,
        N_captured,
        "k--",
        lw=2.5,
        alpha=0.8,
        label=f"N_captured\n(Φ={Phi:.3f})",
    )
    ax.fill_between(time, 0, N_captured, alpha=0.1, color="gray")

    if t50 > 0 and t50 < time[-1]:
        ax.axvline(x=t50, color="red", linestyle=":", linewidth=2, alpha=0.7)
        y_t50 = Phi / 2
        ax.plot(t50, y_t50, "ro", markersize=6)
        ax.annotate(
            f"t₅₀={t50:.0f}fs",
            xy=(t50, y_t50),
            xytext=(t50 + time[-1] * 0.05, y_t50 + 0.05),
            fontsize=9,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
        )


def draw_hamiltonian_heatmap(ax, H_cm: np.ndarray, w0_cm: float = 0.0) -> None:
    n = len(H_cm)
    H_full = H_cm.copy().astype(float)
    for i in range(n):
        H_full[i, i] += w0_cm

    H_offdiag = H_cm.copy().astype(float)
    np.fill_diagonal(H_offdiag, 0)
    vmax = np.max(np.abs(H_offdiag)) if np.any(H_offdiag) else 1

    im = ax.imshow(H_offdiag, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    for i in range(n):
        for j in range(n):
            val = H_full[i, j]
            if i == j:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=True,
                        facecolor="#FFFFCC",
                        edgecolor="black",
                        lw=0.5,
                    )
                )
            text_color = (
                "white" if abs(H_offdiag[i, j]) > vmax * 0.6 and i != j else "black"
            )
            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if i == j else "normal",
                color=text_color,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"|{j + 1}⟩" for j in range(n)], fontsize=9)
    ax.set_yticklabels([f"⟨{i + 1}|" for i in range(n)], fontsize=9)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Coupling (cm⁻¹)", fontsize=9)


def draw_parameters_textbox(ax, params: dict) -> None:
    ax.axis("off")
    lines = []

    n_sites = params.get("n_sites", "N/A")
    T = params.get("temperature")
    T_str = f"{T:.0f}" if T else "N/A"
    lines.append(f"System:  N = {n_sites}    T = {T_str} K")

    eta = params.get("eta_cm")
    gamma = params.get("gamma_cm")
    eta_str = f"{eta:.1f}" if eta else "N/A"
    gamma_str = f"{gamma:.1f}" if gamma else "N/A"
    lines.append(f"Bath:    η = {eta_str} cm⁻¹    γ = {gamma_str} cm⁻¹")

    K = params.get("n_extra_poles", "N/A")
    L = params.get("L_cutoff", "N/A")
    M = params.get("M_total", "N/A")
    lines.append(f"HEOM:    K = {K}    L = {L}    M = {M}")

    dt = params.get("dt")
    total = params.get("total_time")
    dt_str = f"{dt:.4f}" if dt else "N/A"
    total_str = f"{total:.0f}" if total else "N/A"
    lines.append(f"Time:    dt = {dt_str} fs    Total = {total_str} fs")

    init = params.get("initial_state", "N/A")
    lines.append(f"Initial: {init}")

    if params.get("sink_enabled"):
        sink_site = params.get("sink_site", 0) + 1
        gamma_sink = params.get("gamma_sink_fs", 0)
        lines.append(f"Sink:    Site {sink_site}    γ = {gamma_sink:.4f} fs⁻¹")

    text = "\n".join(lines)
    ax.text(
        0.05,
        0.5,
        text,
        transform=ax.transAxes,
        fontsize=11,
        fontfamily="monospace",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", edgecolor="gray", alpha=0.9),
    )


def plot_population_panel(
    ax,
    time: np.ndarray,
    pops: np.ndarray,
    names: list[str],
    title: str,
    colors,
    *,
    sink_data: Optional[PopulationData] = None,
    metadata_lines: Optional[list[str]] = None,
) -> None:
    n_sites = pops.shape[1]
    for i in range(n_sites):
        ax.plot(time, pops[:, i], lw=2, label=names[i], color=colors[i], alpha=0.8)

    if sink_data is not None and sink_data.has_sink:
        mask = sink_data.time <= time[-1]
        plot_sink_overlay(
            ax,
            sink_data.time[mask],
            sink_data.N_captured[mask],
            sink_data.Phi,
            sink_data.t50,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (fs)", fontsize=12)
    ax.set_ylabel("Population", fontsize=12)
    ax.legend(loc="best", fontsize=9, ncol=legend_cols(n_sites), framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if metadata_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(metadata_lines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )


def add_sink_curves(
    ax,
    data1: PopulationData,
    data2: Optional[PopulationData],
    time_limit: Optional[float] = None,
) -> bool:
    added = False

    if data1.has_sink:
        t1, N1 = apply_time_limit(data1.time, time_limit, data1.N_captured)
        ax.plot(
            t1,
            N1,
            "k-",
            lw=2.5,
            alpha=0.7,
            label=f"N_cap\n(Φ₁={data1.Phi:.3f})",
        )
        ax.fill_between(t1, 0, N1, alpha=0.08, color="black")
        added = True

    if data2 is not None and data2.has_sink:
        t2, N2 = apply_time_limit(data2.time, time_limit, data2.N_captured)
        ax.plot(
            t2,
            N2,
            color="gray",
            linestyle="--",
            lw=2.5,
            alpha=0.7,
            label=f"N_cap\n(Φ₂={data2.Phi:.3f})",
        )
        added = True

    return added


def refresh_legend(ax, n_sites: int, *, fontsize: float = 7) -> None:
    """Rebuild legend after late artists (e.g. sink curves) are added."""
    ax.legend(
        loc="best",
        fontsize=fontsize,
        ncol=legend_cols(n_sites),
        framealpha=0.95,
        columnspacing=0.7,
        handlelength=1.5,
        handletextpad=0.5,
    )


def plot_comparison_panel(
    ax,
    time: np.ndarray,
    pops1: np.ndarray,
    pops2: np.ndarray,
    names: list[str],
    colors,
    *,
    title: Optional[str] = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    n_sites = pops1.shape[1]
    for i in range(n_sites):
        ax.plot(
            time,
            pops1[:, i],
            lw=2,
            label=names[i],
            color=colors[i],
            linestyle="-",
            alpha=0.7,
        )
    for i in range(n_sites):
        ax.plot(
            time,
            pops2[:, i],
            lw=2,
            label="_nolegend_",
            color=colors[i],
            linestyle="--",
            alpha=0.7,
        )

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel("Time (fs)", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Population", fontsize=11)

    refresh_legend(ax, n_sites)
    ax.grid(True, alpha=0.3)


def plot_difference_panel(
    ax,
    time: np.ndarray,
    diff: np.ndarray,
    names: list[str],
    colors,
    *,
    title: Optional[str] = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    l2_percent: Optional[float] = None,
) -> None:
    """Plot Δ = file2 − file1 (baseline-first rule; axis spells ``(2 − 1)``)."""
    n_sites = diff.shape[1]
    for i in range(n_sites):
        ax.plot(time, diff[:, i], lw=2, label=names[i], color=colors[i], alpha=0.7)

    ax.axhline(y=0, color="black", linestyle=":", linewidth=1)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel("Time (fs)", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Difference (2 − 1)", fontsize=11)

    refresh_legend(ax, n_sites)
    ax.grid(True, alpha=0.3)

    if l2_percent is not None:
        ax.text(
            0.02,
            0.95,
            format_compare_l2_label(l2_percent),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor="0.75",
                pad=0.3,
            ),
        )
