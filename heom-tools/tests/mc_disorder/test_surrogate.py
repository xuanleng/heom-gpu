"""Pure-numpy tests for the stick-spectrum control-variate surrogate.

No solver / GPU / config needed: the surrogate takes H0 and mu as arrays.
"""
import numpy as np

from heom_tools.mc_disorder import surrogate as sur


def _grid(lo=11000.0, hi=13000.0, n=400):
    return np.linspace(lo, hi, n)


def test_single_site_peak_position_and_area():
    """1-site H: peak at E0+ξ, area under c = |mu|² (Gaussian normalised)."""
    H0 = np.array([[12000.0]])
    mu = np.array([[1.0, 0.0, 0.0]])
    omega = _grid(11000.0, 13000.0, 4000)
    c = sur.stick_spectrum([200.0], H0, mu, omega, width_cm=50.0)
    # Peak near 12200.
    assert abs(omega[np.argmax(c)] - 12200.0) < 2.0
    # Area ≈ oscillator strength |mu|² = 1.
    area = np.trapezoid(c, omega)
    assert abs(area - 1.0) < 1e-3


def test_zero_disorder_reproducible_and_deterministic():
    H0 = np.array([[12000.0, 100.0], [100.0, 12100.0]])
    mu = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    omega = _grid()
    a = sur.stick_spectrum(np.zeros(2), H0, mu, omega, 60.0)
    b = sur.stick_spectrum(np.zeros(2), H0, mu, omega, 60.0)
    np.testing.assert_array_equal(a, b)
    # Two eigenpeaks → two local maxima in a wide-enough, fine-enough grid.
    assert a.sum() > 0.0


def test_total_area_conserved_under_disorder():
    """Broadening is unitary: total area = Σ|d_k|² is independent of ξ (dipole
    magnitude is basis-independent for an orthonormal eigenbasis)."""
    H0 = np.array([[12000.0, 80.0, 10.0],
                   [80.0, 12150.0, 60.0],
                   [10.0, 60.0, 12080.0]])
    mu = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.3], [0.5, 0.0, 1.0]])
    omega = _grid(10500.0, 13500.0, 8000)
    tot_mu = np.sum(mu * mu)
    for xi in (np.zeros(3), np.array([150.0, -120.0, 90.0]), np.array([-300.0, 40.0, 200.0])):
        c = sur.stick_spectrum(xi, H0, mu, omega, 40.0)
        assert abs(np.trapezoid(c, omega) - tot_mu) < 5e-3 * tot_mu


def test_width_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        sur.stick_spectrum([0.0], np.array([[1.0]]), np.array([[1.0, 0, 0]]),
                           _grid(), width_cm=0.0)


def test_mean_stick_matches_single_when_sigma_zero():
    H0 = np.array([[12000.0, 100.0], [100.0, 12100.0]])
    mu = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    omega = _grid()
    mean = sur.mean_stick_spectrum(H0, mu, np.zeros(2), omega, 60.0,
                                   n_samples=1000, seed=1)
    single = sur.stick_spectrum(np.zeros(2), H0, mu, omega, 60.0)
    np.testing.assert_allclose(mean, single, rtol=1e-12)


def test_mean_stick_reproducible():
    H0 = np.array([[12000.0, 100.0], [100.0, 12100.0]])
    mu = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    omega = _grid(11000.0, 13000.0, 200)
    sig = sur.sigma_from_fwhm(np.full(2, 120.0))
    a = sur.mean_stick_spectrum(H0, mu, sig, omega, 60.0, n_samples=5000, seed=7)
    b = sur.mean_stick_spectrum(H0, mu, sig, omega, 60.0, n_samples=5000, seed=7)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------
# control_variate_apply: the estimator algebra (independent of any spectrum)
# --------------------------------------------------------------------------

def test_cv_perfect_correlation_recovers_exact_mean():
    """KILLER: if O_i == c_i, β=1, ρ=1, and the CV estimate collapses to the
    exact ⟨c⟩ (zero variance). This pins down the estimator algebra: the WRONG
    formula ⟨c⟩+mean(O−βc) would give 2⟨c⟩ here."""
    rng = np.random.default_rng(0)
    n_omega = 5
    c_nodes = rng.standard_normal((50, n_omega)) + 3.0
    o_nodes = c_nodes.copy()
    c_mean = np.full(n_omega, 12.34)   # arbitrary "exact" mean
    est, beta, rho = sur.control_variate_apply(o_nodes, c_nodes, c_mean)
    np.testing.assert_allclose(beta, 1.0, atol=1e-9)
    np.testing.assert_allclose(rho, 1.0, atol=1e-9)
    np.testing.assert_allclose(est, c_mean, atol=1e-9)


def test_cv_zero_correlation_returns_plain_mean():
    """Independent c → β≈0, estimate ≈ plain mean(O) (no help, no harm)."""
    rng = np.random.default_rng(1)
    o = rng.standard_normal((20000, 1))
    c = rng.standard_normal((20000, 1))
    est, beta, rho = sur.control_variate_apply(o, c, np.array([0.0]))
    assert abs(beta[0]) < 0.05
    assert abs(est[0] - o.mean()) < 0.05


def test_cv_flat_surrogate_is_noop():
    """Var(c)=0 (flat surrogate) → β=0, estimate = mean(O), never NaN."""
    rng = np.random.default_rng(2)
    o = rng.standard_normal((100, 3))
    c = np.ones((100, 3)) * 7.0
    est, beta, rho = sur.control_variate_apply(o, c, np.full(3, 7.0))
    np.testing.assert_allclose(beta, 0.0)
    np.testing.assert_allclose(est, o.mean(axis=0))
    assert np.all(np.isfinite(rho))


def test_cv_fixed_beta_is_unbiased_shape():
    """Fixed-β path (production): estimate = mean(O) − β(mean(c) − ⟨c⟩)."""
    rng = np.random.default_rng(3)
    o = rng.standard_normal((100, 4))
    c = rng.standard_normal((100, 4))
    cm = np.zeros(4)
    beta_fixed = np.full(4, 0.5)
    est, beta, _ = sur.control_variate_apply(o, c, cm, beta=beta_fixed)
    np.testing.assert_array_equal(beta, beta_fixed)
    expect = o.mean(axis=0) - beta_fixed * (c.mean(axis=0) - cm)
    np.testing.assert_allclose(est, expect)
