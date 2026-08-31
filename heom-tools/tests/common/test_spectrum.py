"""Tests for heom_tools.common.spectrum."""
from __future__ import annotations

import numpy as np
import pytest

from heom_tools.common.spectrum import (
    default_absorption_frequency_window_cm,
    find_peaks_safe,
)


def test_default_absorption_frequency_window_cm() -> None:
    w_lo, w_hi = default_absorption_frequency_window_cm(np.array([0.0, 100.0, 200.0]))
    np.testing.assert_allclose([w_lo, w_hi], [37.5, 262.5])

    w_lo, w_hi = default_absorption_frequency_window_cm(np.array([0.0, 12100.0]))
    assert abs((w_hi - w_lo) - 500.0) < 1e-9
    assert abs(0.5 * (w_lo + w_hi) - 12100.0) < 1e-9


def test_find_peaks_safe_height_frac() -> None:
    xv = np.arange(11, dtype=float)
    y_two = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])

    pk_strict = find_peaks_safe(xv, y_two, height_frac=0.5)
    assert len(pk_strict) == 1
    assert abs(pk_strict[0][0] - 2.0) < 1e-9

    pk_loose = find_peaks_safe(xv, y_two, height_frac=0.05, distance_cm=1.0)
    assert len(pk_loose) == 2

    with pytest.raises(TypeError):
        find_peaks_safe(xv, y_two, prominence_frac=0.5)  # type: ignore[call-arg]
