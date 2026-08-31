"""Smoke tests for las-plot matplotlib figures."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from heom_tools.absorption.plot.align import TheorySpectrum, align_las_compare
from heom_tools.absorption.plot.plot import plot_las_compare_png, render_las_compare_figure
from heom_tools.common.mpl_util import pyplot
from heom_tools.common.spectrum import ExperimentSpectrum


def _theory(label: str, omega: np.ndarray, intensity: np.ndarray) -> TheorySpectrum:
    return TheorySpectrum(label=label, omega_cm=omega, intensity=intensity)


def test_render_las_compare_figure_legend_and_config_overlay() -> None:
    om = np.linspace(12000, 12100, 11)
    y = np.sin((om - 12050) / 20.0) * 0.5 + 0.5
    t0 = _theory("run_a_spectrum.h5", om, y)
    t1 = _theory("run_b_spectrum.h5", om, y * 1.01)
    exp = ExperimentSpectrum(omega_cm=om.copy(), intensity=y.copy(), label="exp.csv")
    aligned = align_las_compare([t0, t1], experiment=exp)
    config_text = "1 vs 2: L_total  20 -> 15"

    _, plt_mod = pyplot("Agg")
    fig = render_las_compare_figure(aligned, plt_mod, config_diff_text=config_text)
    try:
        ax_top, ax_bot = fig.axes[:2]
        top_labels = [t.get_text() for t in ax_top.get_legend().get_texts()]
        bot_labels = [t.get_text() for t in ax_bot.get_legend().get_texts()]
        assert "run_a_spectrum.h5" in top_labels
        assert not any("L²" in lab for lab in top_labels)
        assert any("1 vs exp" in lab and "L²" in lab for lab in bot_labels)
        assert any("1 vs 2" in lab and "L²" in lab for lab in bot_labels)
        overlay = [c for c in ax_top.get_children() if hasattr(c, "get_text")]
        overlay_text = " ".join(getattr(c, "get_text")() for c in overlay if c.get_text())
        assert "1 vs 2:" in overlay_text
        assert "L_total" in overlay_text
    finally:
        plt_mod.close(fig)


def test_plot_las_compare_png_writes_file() -> None:
    om = np.linspace(12000, 12100, 11)
    t0 = _theory("a_spectrum.h5", om, np.ones_like(om))
    t1 = _theory("b_spectrum.h5", om, np.ones_like(om) * 0.5)
    aligned = align_las_compare([t0, t1], experiment=None)

    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/compare.png"
        plot_las_compare_png(aligned, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0


def test_single_theory_plot_no_residual_panel() -> None:
    om = np.linspace(12000, 12100, 11)
    t0 = _theory("solo_spectrum.h5", om, np.ones_like(om))
    aligned = align_las_compare([t0], experiment=None)
    assert aligned.residuals == []

    _, plt_mod = pyplot("Agg")
    fig = render_las_compare_figure(aligned, plt_mod, title="linear absorption")
    try:
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == "linear absorption"
    finally:
        plt_mod.close(fig)


def test_single_theory_with_experiment_has_residual() -> None:
    om = np.linspace(12000, 12100, 11)
    y = np.ones_like(om)
    t0 = _theory("solo_spectrum.h5", om, y)
    exp = ExperimentSpectrum(omega_cm=om.copy(), intensity=y * 0.9, label="exp.csv")
    aligned = align_las_compare([t0], experiment=exp)
    assert len(aligned.residuals) == 1
    assert aligned.residuals[0].ref == "experiment"
