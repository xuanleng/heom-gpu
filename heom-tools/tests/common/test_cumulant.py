"""Monomer cumulant reference: mirror symmetry, reorganization energy, g(t)."""
import numpy as np
import pytest

from heom_tools.common.cumulant import _CM_FS, monomer_correlations, reorganization_cm

KB_CM = 0.695034800      # cm⁻¹ / K


def drude_matsubara(lam, gamma, T, K):
    """Drude-Lorentz C(t) on Matsubara ladders, real ν ⇒ c̄ = c*."""
    beta = 1.0 / (KB_CM * T)
    c = [lam * gamma * (1.0 / np.tan(beta * gamma / 2.0) - 1j)]
    nu = [gamma]
    for k in range(1, K + 1):
        vk = 2.0 * np.pi * k / beta
        c.append(4.0 * lam * gamma / beta * vk / (vk**2 - gamma**2))
        nu.append(vk)
    c = np.asarray(c, dtype=np.complex128)
    return c, c.conj(), np.asarray(nu, dtype=np.complex128)


def oscillator_pair():
    """A complex-conjugate ladder pair with arbitrary c: c̄_1 = c_2*, c̄_2 = c_1*."""
    nu = np.array([15.0 + 340.0j, 15.0 - 340.0j])
    c = np.array([900.0 - 2300.0j, 700.0 + 1100.0j])
    return c, np.array([c[1].conj(), c[0].conj()]), nu


T_FS = np.linspace(0.0, 2000.0, 4001)


@pytest.mark.parametrize("coeffs", [drude_matsubara(35.0, 53.088, 77.0, 3),
                                    oscillator_pair()])
def test_mirror_symmetry_about_bare_energy(coeffs):
    c, cbar, nu = coeffs
    lam = reorganization_cm(c, nu)
    E = 123.0
    C_A, C_F = monomer_correlations(c, cbar, nu, 0.7, E, T_FS)
    rot = np.exp(1j * (E - lam) * T_FS * _CM_FS)
    np.testing.assert_allclose(C_F * rot, np.conj(C_A * rot), rtol=0, atol=1e-12)


@pytest.mark.parametrize("coeffs", [drude_matsubara(35.0, 53.088, 77.0, 3),
                                    oscillator_pair()])
def test_exact_pairing_gives_g_conjugate_form(coeffs):
    c, cbar, nu = coeffs
    lam = reorganization_cm(c, nu)
    E, res = -40.0, 0.3
    C_A, C_F = monomer_correlations(c, cbar, nu, res, E, T_FS)
    t = T_FS * _CM_FS
    g = np.sum(c[:, None] / nu[:, None]**2
               * (np.exp(-nu[:, None] * t) + nu[:, None] * t - 1.0), axis=0) + res * t
    np.testing.assert_allclose(C_F, np.exp(-1j * (E - 2.0 * lam) * t - np.conj(g)),
                               rtol=0, atol=1e-12)


def test_no_imaginary_part_means_no_stokes_shift():
    c = np.array([1200.0, 300.0], dtype=np.complex128)
    nu = np.array([53.0, 400.0], dtype=np.complex128)
    assert reorganization_cm(c, nu) == 0.0
    C_A, C_F = monomer_correlations(c, c, nu, 0.0, 10.0, T_FS)
    np.testing.assert_array_equal(C_A, C_F)


def test_reorganization_recovers_drude_lambda():
    for lam in (35.0, 80.0):
        c, _, nu = drude_matsubara(lam, 53.088, 298.0, 4)
        assert reorganization_cm(c, nu) == pytest.approx(lam, rel=1e-13)


def test_lineshape_function_matches_double_integral():
    c, cbar, nu = drude_matsubara(35.0, 53.088, 77.0, 3)
    t_fs = np.linspace(0.0, 400.0, 400001)
    t = t_fs * _CM_FS
    C = np.sum(c[:, None] * np.exp(-nu[:, None] * t), axis=0)
    dt = t[1] - t[0]
    inner = np.concatenate(([0.0], np.cumsum(0.5 * (C[1:] + C[:-1]) * dt)))
    g_num = np.concatenate(([0.0], np.cumsum(0.5 * (inner[1:] + inner[:-1]) * dt)))
    C_A, _ = monomer_correlations(c, cbar, nu, 0.0, 0.0, t_fs)
    np.testing.assert_allclose(np.exp(-g_num), C_A, rtol=0, atol=1e-9)


def test_initial_instantaneous_frequencies():
    c, cbar, nu = drude_matsubara(35.0, 53.088, 77.0, 3)
    E = 50.0
    t_fs = np.array([0.0, 1e-3])
    C_A, C_F = monomer_correlations(c, cbar, nu, 0.0, E, t_fs)
    w = lambda C: -np.angle(C[1] / C[0]) / (t_fs[1] * _CM_FS)
    assert w(C_A) == pytest.approx(E, abs=1e-3)
    assert w(C_F) == pytest.approx(E - 2.0 * 35.0, abs=1e-3)


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="one length"):
        monomer_correlations([1.0], [1.0, 2.0], [1.0], 0.0, 0.0, T_FS)
    with pytest.raises(ValueError, match="Re ν"):
        monomer_correlations([1.0], [1.0], [-1.0], 0.0, 0.0, T_FS)
