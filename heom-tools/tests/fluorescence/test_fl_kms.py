"""KMS fit and the physics-identity guard."""
import numpy as np
import pytest

from heom_tools.fluorescence.post.kms import KB_CM, fit_kms, require_same_physics
from tests.fluorescence._synth import YAML, write_raw

T = 77.0


def _pair(omega):
    I_A = np.exp(-0.5 * ((omega - 12100.0) / 60.0) ** 2) \
        + 0.6 * np.exp(-0.5 * ((omega - 12350.0) / 60.0) ** 2)
    return I_A, np.exp(-(omega - 12000.0) / (KB_CM * T)) * I_A


def test_exact_boltzmann_ratio_recovers_temperature():
    w = np.linspace(11700.0, 12700.0, 2001)
    I_A, I_F = _pair(w)
    fit = fit_kms(w, I_F, w, I_A, T)
    assert fit.rel_error < 1e-6
    assert fit.omega_cm.size > 100
    assert np.all(fit.weights > 0)


def test_absorption_on_another_grid_is_interpolated():
    w = np.linspace(11700.0, 12700.0, 2001)
    wa = np.linspace(11650.0, 12750.0, 3001)
    I_A, _ = _pair(wa)
    _, I_F = _pair(w)
    assert fit_kms(w, I_F, wa, I_A, T).rel_error < 1e-4


def test_blue_emission_is_rejected():
    w = np.linspace(11700.0, 12700.0, 2001)
    I_A, I_F = _pair(w)
    with pytest.raises(ValueError, match="not red"):
        fit_kms(w, I_A, w, I_F, T)


def test_no_overlap_is_rejected():
    w = np.linspace(11000.0, 13000.0, 4001)
    I_A = np.exp(-0.5 * ((w - 12600.0) / 20.0) ** 2)
    I_F = np.exp(-0.5 * ((w - 11400.0) / 20.0) ** 2)
    with pytest.raises(ValueError, match="overlap"):
        fit_kms(w, I_F, w, I_A, T)


def test_same_core_physics_passes_across_task_blocks(tmp_path):
    fl, ab = str(tmp_path / "fl.h5"), str(tmp_path / "abs.h5")
    write_raw(fl, "fluorescence", n_outputs=8)
    write_raw(ab, "absorption", n_outputs=8)
    assert require_same_physics(fl, ab)


def test_different_bath_is_rejected(tmp_path):
    fl, ab = str(tmp_path / "fl.h5"), str(tmp_path / "abs.h5")
    write_raw(fl, "fluorescence", n_outputs=8)
    write_raw(ab, "absorption", n_outputs=8,
              yaml=YAML.format(task="absorption").replace("eta: 70.0", "eta: 80.0"))
    with pytest.raises(ValueError, match="system/bath/filter differ"):
        require_same_physics(fl, ab)
