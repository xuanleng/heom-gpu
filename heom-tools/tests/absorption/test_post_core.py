"""Unit tests for ``heom_tools.absorption.post`` core helpers and panels."""
from __future__ import annotations

import os
import tempfile
import warnings

import h5py
import numpy as np
import pytest

from heom_tools.absorption.post.core import (
    AbsorptionPostResult,
    _source_precision_mode,
    _sticks_table,
)
from heom_tools.absorption.post.panel import RunParametersPanel, load_run_parameters_panel
from heom_tools.absorption.post.util import format_duration
from heom_tools.common.spectrum import (
    EXPERIMENT_TOKEN_RE,
    ExperimentSpectrum,
    StickEntry,
    default_absorption_frequency_window_cm,
    discover_experiment_spectra_auto,
    fft_to_spectrum,
    find_peaks_safe,
)


def test_default_absorption_frequency_window_cm() -> None:
    w_lo, w_hi = default_absorption_frequency_window_cm(np.array([0.0, 100.0, 200.0]))
    np.testing.assert_allclose([w_lo, w_hi], [37.5, 262.5])
    w_lo, w_hi = default_absorption_frequency_window_cm(np.array([0.0, 12100.0]))
    assert abs((w_hi - w_lo) - 500.0) < 1e-9
    assert abs(0.5 * (w_lo + w_hi) - 12100.0) < 1e-9
    w_lo, w_hi = default_absorption_frequency_window_cm(np.array([0.0, 100.0, 100.0]))
    assert abs((w_hi - w_lo) - 500.0) < 1e-9


def test_experiment_spectrum_rejects_all_zero_intensities() -> None:
    fd, zpath = tempfile.mkstemp(suffix=".csv", text=True)
    os.close(fd)
    try:
        with open(zpath, "w", encoding="utf-8") as zf:
            zf.write("1000,0\n2000,0\n")
        with pytest.raises(ValueError, match="all zero"):
            ExperimentSpectrum.load(zpath)
    finally:
        os.unlink(zpath)


def test_find_peaks_safe_height_frac() -> None:
    xv = np.arange(5, dtype=float)
    y_plateau = np.array([0.05, 0.9, 0.9, 0.9, 0.05])
    assert find_peaks_safe(xv, y_plateau) == []
    y_peak = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    pk = find_peaks_safe(xv, y_peak)
    assert len(pk) == 1 and abs(pk[0][0] - 2.0) < 1e-9

    xv2 = np.arange(11, dtype=float)
    y_two = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    pk_strict = find_peaks_safe(xv2, y_two, height_frac=0.5)
    assert len(pk_strict) == 1 and abs(pk_strict[0][0] - 2.0) < 1e-9
    pk_loose = find_peaks_safe(xv2, y_two, height_frac=0.05, distance_cm=1.0)
    assert len(pk_loose) == 2

    with pytest.raises(TypeError):
        find_peaks_safe(xv2, y_two, prominence_frac=0.5)  # type: ignore[call-arg]


def test_absorption_post_result_empty_consistency() -> None:
    fget = AbsorptionPostResult.consistency_max_error.fget

    class _EmptyC:
        ct_iso_sum_check = np.array([], dtype=np.complex128)
        ct_iso_total = np.array([], dtype=np.complex128)

    assert fget(_EmptyC()) == 0.0


def test_source_precision_mode_propagation() -> None:
    with tempfile.TemporaryDirectory() as d:
        # explicit stamp is carried through verbatim
        mixed = os.path.join(d, "mixed.h5")
        with h5py.File(mixed, "w") as f:
            f.attrs["precision_mode"] = "mixed_fp32"
        assert _source_precision_mode(mixed) == "mixed_fp32"

        # legacy (no attr) defaults to fp64 — matches mc_merge convention
        legacy = os.path.join(d, "legacy.h5")
        with h5py.File(legacy, "w") as f:
            f.attrs["task"] = "absorption"
        assert _source_precision_mode(legacy) == "fp64"

        # unreadable/missing file also defaults to fp64, never raises
        assert _source_precision_mode(os.path.join(d, "nope.h5")) == "fp64"


def test_fft_to_spectrum_rejects_bad_grid() -> None:
    for bad_dt in (0.0, -1.0):
        with pytest.raises(ValueError, match="dt_fs"):
            fft_to_spectrum(np.ones(4, dtype=np.complex128), dt_fs=bad_dt)
    with pytest.raises(ValueError, match="dw_cm"):
        fft_to_spectrum(np.ones(4, dtype=np.complex128), dt_fs=1.0, dw_cm=0.0)


def test_sticks_table_dominant_site_index() -> None:
    st_tab = _sticks_table([
        StickEntry(1.0, 0.5, 1, 0.75),
        StickEntry(2.0, 0.3, 0, 1.0),
    ])
    assert st_tab[0, 2] == 2.0
    assert st_tab[1, 2] == 1.0
    with pytest.raises(ValueError, match="dominant_site_index"):
        _sticks_table([StickEntry(1.0, 0.5, -1, 0.0)])


def test_discover_experiment_spectra_auto_token_gate() -> None:
    accept_names = [
        "fmo-las-exp-77K.csv",
        "experimental_FMO_77K.csv",
        "experiment_data.txt",
        "data_exp.csv",
        "exp.csv",
        "EXP_uppercase.csv",
    ]
    reject_names = [
        "export.csv",
        "unexpected.csv",
        "expansion_log.csv",
        "expensive_test.csv",
        "expert_review.csv",
        "abs_source_spectrum.h5",
        "abs_source_post.h5",
    ]
    with tempfile.TemporaryDirectory() as exp_tmp:
        for fn in accept_names + reject_names:
            with open(os.path.join(exp_tmp, fn), "w", encoding="utf-8") as f:
                f.write("12000.0,0.5\n12100.0,1.0\n12200.0,0.5\n")

        loaded = discover_experiment_spectra_auto(exp_tmp)
        loaded_basenames = sorted(s.label for s in loaded)
        assert loaded_basenames == sorted(accept_names)

        for bad in ("export", "unexpected", "expansion", "expensive_test", "expert_review"):
            assert EXPERIMENT_TOKEN_RE.search(bad + ".csv") is None


def test_format_duration() -> None:
    assert format_duration(None) == "—"
    assert format_duration(0.5) == "0.500s"
    assert format_duration(59.999) == "59.999s"
    assert format_duration(60.0) == "1m 0s"
    assert format_duration(3661.0) == "1h 1m 1s"
    assert format_duration(90061.0) == "1d 1h 1m 1s"


def test_run_parameters_panel_overlay() -> None:
    panel = RunParametersPanel(
        temperature_K=77.0,
        eta_cm=35.0,
        gamma_inverse_fs=18.9,
        static_disorder_fwhm_cm=0.0,
        n_extra_poles=5,
        L_cutoff=14,
        L_total=0,
        npa=0,
        tolerance=1.0e-15,
        dt_fs=2.0,
        n_steps=50000,
        computation_time_s=3723.0,
    )
    overlay = panel.format_overlay_lines()
    for section in (
        "Spectral density & bath",
        "Truncation",
        "Time evolution",
        "Runtime",
    ):
        assert section in overlay
    assert "  T          = 77.0 K" in overlay
    assert "  K          = 5" in overlay
    assert "  tolerance  = 1.0e-15" in overlay
    assert "  total      = 1h 2m 3s" in overlay


def test_load_run_parameters_panel_disorder_warnings() -> None:
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        disorder_h5 = tf.name
    try:
        with h5py.File(disorder_h5, "w") as f:
            params = f.create_group("/parameters")
            params.attrs["static_disorder_fwhm_cm"] = 100.0
            params.attrs["temperature"] = 77.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bad_panel = load_run_parameters_panel(disorder_h5)
        msgs = [str(w.message) for w in caught]
        assert any("L_total" in m for m in msgs)
        assert any("npa" in m for m in msgs)
        assert bad_panel.L_total is None
        assert bad_panel.npa is None
        assert "  L_total    = —" in bad_panel.format_overlay_lines()
    finally:
        os.unlink(disorder_h5)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        no_disorder_h5 = tf.name
    try:
        with h5py.File(no_disorder_h5, "w") as f:
            params = f.create_group("/parameters")
            params.attrs["static_disorder_fwhm_cm"] = 0.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ok_panel = load_run_parameters_panel(no_disorder_h5)
        assert not any("L_total" in str(w.message) for w in caught)
        assert ok_panel.L_total == 0
        assert ok_panel.npa == 0
    finally:
        os.unlink(no_disorder_h5)
