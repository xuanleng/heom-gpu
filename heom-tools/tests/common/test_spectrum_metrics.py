"""heom_tools.common.spectrum_metrics unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from heom_tools.common.spectrum_metrics import (
    interp_clamped,
    l2_relative_diff,
    l2_relative_percent_on_grid,
    peak_normalize,
    peak_normalize_factor,
)


def test_interp_clamped_endpoints() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    xq = np.array([-1.0, 0.5, 2.5])
    got = interp_clamped(xq, x, y)
    np.testing.assert_allclose(got, [0.0, 0.5, 0.0])


def test_peak_normalize() -> None:
    y = np.array([0.0, 4.0, -2.0])
    assert peak_normalize_factor(y) == 4.0
    np.testing.assert_allclose(peak_normalize(y), [0.0, 1.0, -0.5])
    assert peak_normalize_factor(np.array([])) == 1.0


def test_l2_relative_diff_constant_scale() -> None:
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


def test_l2_relative_percent_on_grid() -> None:
    ref = np.ones(4)
    delta = np.ones(4) * 0.1
    assert l2_relative_percent_on_grid(ref, delta) == pytest.approx(10.0, rel=1e-12)
