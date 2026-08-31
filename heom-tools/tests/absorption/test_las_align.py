"""Tests for heom_tools.absorption.plot.align."""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from heom_tools.absorption.plot.align import (
    TheorySpectrum,
    align_las_compare,
    format_residual_legend_label,
    l2_percent_vs_baseline,
    l2_percent_vs_experiment,
    print_metrics_table,
    residual_curve_label,
    residual_l2_rel_percent,
)
from heom_tools.common.spectrum import ExperimentSpectrum
from heom_tools.common.spectrum_metrics import (
    interp_clamped,
    l2_relative_diff,
    l2_relative_percent_on_grid,
)


def _theory(label: str, omega: np.ndarray, intensity: np.ndarray) -> TheorySpectrum:
    return TheorySpectrum(label=label, omega_cm=omega, intensity=intensity)


def test_interp_clamped_endpoints() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    xq = np.array([-1.0, 0.5, 2.5])
    got = interp_clamped(xq, x, y)
    np.testing.assert_allclose(got, [0.0, 0.5, 0.0])


def test_align_two_theories_residual_vs_first() -> None:
    om = np.linspace(12000, 12100, 11)
    t0 = _theory("a_post.h5", om, np.ones_like(om))
    t1 = _theory("b_post.h5", om, np.ones_like(om) * 0.5)
    aligned = align_las_compare([t0, t1], experiment=None)
    assert aligned.use_aligned_top
    assert len(aligned.residuals) == 1
    r = aligned.residuals[0]
    assert r.ref == "first_theory"
    np.testing.assert_allclose(r.delta, -0.5, rtol=0, atol=1e-12)


def test_align_with_experiment_mae_zero_on_grid() -> None:
    om = np.linspace(12000, 12100, 11)
    y = np.sin((om - 12050) / 20.0) * 0.5 + 0.5
    theory = _theory("t_post.h5", om, y)
    exp = ExperimentSpectrum(omega_cm=om.copy(), intensity=y.copy(), label="exp.csv")
    aligned = align_las_compare([theory], experiment=exp)
    assert aligned.experiment is not None
    assert len(aligned.residuals) == 1
    delta = aligned.residuals[0].delta
    assert float(np.max(np.abs(delta))) < 1e-12


def test_print_metrics_table_runs() -> None:
    om = np.linspace(12000, 12100, 5)
    t0 = _theory("a.h5", om, np.ones(5))
    t1 = _theory("b.h5", om, np.ones(5) * 0.9)
    aligned = align_las_compare([t0, t1], experiment=None)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_metrics_table(aligned.x_residual, aligned.residuals)
    out = buf.getvalue()
    assert "[LAS compare]" in out
    assert "MAE" in out
    assert residual_curve_label(aligned.residuals[0]) in out


def test_residual_curve_labels() -> None:
    om = np.linspace(12000, 12100, 5)
    t0 = _theory("a.h5", om, np.ones(5))
    t1 = _theory("b.h5", om, np.ones(5) * 0.9)
    aligned = align_las_compare([t0, t1], experiment=None)
    assert residual_curve_label(aligned.residuals[0]) == "1 vs 2"


def test_l2_percent_vs_baseline_two_theories() -> None:
    om = np.linspace(12000, 12100, 11)
    t0 = _theory("a.h5", om, np.ones_like(om))
    t1 = _theory("b.h5", om, np.ones_like(om) * 0.5)
    aligned = align_las_compare([t0, t1], experiment=None)
    pct = l2_percent_vs_baseline(aligned)
    assert 0 not in pct
    assert pct[1] == pytest.approx(50.0, rel=1e-9)
    assert l2_percent_vs_experiment(aligned) is None


def test_l2_percent_split_refs_with_experiment() -> None:
    om = np.linspace(12000, 12100, 11)
    y = np.sin((om - 12050) / 20.0) * 0.5 + 0.5
    t0 = _theory("a.h5", om, y)
    t1 = _theory("b.h5", om, y * 1.1)
    exp = ExperimentSpectrum(omega_cm=om.copy(), intensity=y.copy(), label="exp.csv")
    aligned = align_las_compare([t0, t1], experiment=exp)
    assert l2_percent_vs_experiment(aligned) == pytest.approx(0.0, abs=1e-10)
    assert l2_percent_vs_baseline(aligned)[1] == pytest.approx(10.0, rel=1e-6)


def test_theory_only_residual_grid_is_omega_union() -> None:
    """Subjects' ω samples must appear on the residual grid (not baseline-only)."""
    om0 = np.array([12000.0, 12050.0, 12100.0])
    om1 = np.array([12000.0, 12025.0, 12050.0, 12075.0, 12100.0])
    t0 = _theory("a.h5", om0, np.ones_like(om0))
    t1 = _theory("b.h5", om1, np.ones_like(om1) * 0.5)
    aligned = align_las_compare([t0, t1], experiment=None)
    np.testing.assert_allclose(aligned.x_residual, om1)
    assert 12025.0 in aligned.x_residual
    assert 12075.0 in aligned.x_residual


def test_empty_omega_raises() -> None:
    t0 = _theory("a.h5", np.array([]), np.array([]))
    with pytest.raises(ValueError, match="no ω samples"):
        align_las_compare([t0], experiment=None)


def test_format_residual_legend_label() -> None:
    om = np.linspace(12000, 12100, 11)
    t0 = _theory("a.h5", om, np.ones_like(om))
    t1 = _theory("b.h5", om, np.ones_like(om) * 0.5)
    aligned = align_las_compare([t0, t1], experiment=None)
    r = aligned.residuals[0]
    assert format_residual_legend_label(r, aligned) == "1 vs 2 (L² 50.0%)"


def test_l2_relative_diff_matches_constant_scale() -> None:
    om = np.linspace(12000, 12100, 8)
    ref = np.ones_like(om)
    test = np.ones_like(om) * 1.1
    assert l2_relative_diff(om, ref, om, test) == pytest.approx(0.1, rel=1e-9)


def test_l2_relative_diff_unsorted_omega() -> None:
    om_sorted = np.linspace(12000, 12100, 8)
    ref = np.ones_like(om_sorted)
    test = np.ones_like(om_sorted) * 1.1
    om_rev = om_sorted[::-1].copy()
    assert l2_relative_diff(om_rev, ref[::-1], om_rev, test[::-1]) == pytest.approx(
        0.1, rel=1e-9,
    )


def test_residual_l2_matches_delta_on_common_grid() -> None:
    om_dense = np.linspace(12000, 12100, 21)
    om_sparse = om_dense[::2]
    y = np.sin((om_dense - 12050) / 20.0) * 0.5 + 0.5
    t0 = _theory("a.h5", om_dense, y)
    t1 = _theory("b.h5", om_sparse, y[::2] * 1.1)
    exp = ExperimentSpectrum(omega_cm=om_sparse.copy(), intensity=y[::2].copy(), label="exp.csv")
    aligned = align_las_compare([t0, t1], experiment=exp)
    for r in aligned.residuals:
        if r.ref == "experiment":
            ref_y = interp_clamped(aligned.x_residual, exp.omega_cm, exp.intensity)
        else:
            ref_y = interp_clamped(aligned.x_residual, t0.omega_cm, t0.intensity)
        expected = l2_relative_percent_on_grid(ref_y, r.delta)
        assert residual_l2_rel_percent(r, aligned) == pytest.approx(expected, rel=1e-12)
        if r.ref == "experiment":
            assert l2_percent_vs_experiment(aligned) == pytest.approx(expected, rel=1e-12)
        else:
            assert l2_percent_vs_baseline(aligned)[r.theory_idx] == pytest.approx(
                expected, rel=1e-12,
            )
