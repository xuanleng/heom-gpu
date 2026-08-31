"""Pure metric tests for the smoothness probe (numpy only; no GPU/solver).

The probe's orchestration (generate→run→merge→spectrum) needs the solver + a
GPU; here we lock down the pure pieces the
whole judgment rests on — the L² relative-difference metric and the closed-form
convergence core. They live in probe_core (numpy-only, zero heavy imports even
lazily), so loading it for this test needs only numpy.
"""
import numpy as np
import pytest

from heom_tools.mc_disorder import probe_core as probe


def test_l2_identical_is_zero():
    w = np.linspace(0, 100, 51)
    y = np.sin(w / 10.0) ** 2
    assert probe.l2_relative_diff(w, y, w, y) == pytest.approx(0.0, abs=1e-12)


def test_l2_relative_scale():
    # A uniform 10% offset on a flat spectrum is a 10% relative L² diff.
    w = np.linspace(0, 10, 100)
    y = np.full_like(w, 2.0)
    assert probe.l2_relative_diff(w, y, w, y * 1.1) == pytest.approx(0.1, rel=1e-9)


def test_l2_interpolates_onto_reference_grid():
    # Test spectrum on a DIFFERENT, coarser grid; a linear ramp interpolates
    # exactly, so the metric must still be ~0 (grid drift must not fake error).
    w_ref = np.linspace(0, 10, 101)
    w_test = np.linspace(0, 10, 11)
    ramp = lambda x: 3.0 * x + 1.0
    assert probe.l2_relative_diff(w_ref, ramp(w_ref), w_test, ramp(w_test)) \
        == pytest.approx(0.0, abs=1e-12)


def test_l2_symmetry_denominator_is_reference():
    # The denominator is ‖reference‖, so the metric is NOT symmetric — swapping
    # ref/test rescales by ‖ref‖/‖test‖. Guards against a silent denom swap.
    w = np.linspace(0, 1, 50)
    a = np.full_like(w, 1.0)
    b = np.full_like(w, 2.0)
    d_ab = probe.l2_relative_diff(w, a, w, b)   # ‖b-a‖/‖a‖ = 1.0
    d_ba = probe.l2_relative_diff(w, b, w, a)   # ‖a-b‖/‖b‖ = 0.5
    assert d_ab == pytest.approx(1.0, rel=1e-9)
    assert d_ba == pytest.approx(0.5, rel=1e-9)


def test_l2_zero_reference_raises():
    w = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="zero L"):
        probe.l2_relative_diff(w, np.zeros_like(w), w, np.ones_like(w))


# --------------------------------------------------------------------------
# 2-D (2des) observable: flattened map on a shared integer index grid.
# This is the whole 1-D↔2-D unification — the SAME l2_relative_diff is reused
# on flattened 2-D spectra, and because every node shares one FFT grid the
# 'grid' is a plain arange whose interp is an identity.
# --------------------------------------------------------------------------

def test_l2_flattened_2d_matches_full_array_norm():
    # A flattened 2-D map with an integer index grid must give exactly the
    # element-wise relative L² — no interpolation artifact from the fake axis.
    rng = np.random.default_rng(1)
    ref2d = rng.standard_normal((13, 17))
    test2d = ref2d + 0.05 * rng.standard_normal((13, 17))
    grid = np.arange(ref2d.size, dtype=np.float64)
    got = probe.l2_relative_diff(grid, ref2d.reshape(-1),
                                 grid, test2d.reshape(-1))
    want = (np.sqrt(np.sum((test2d - ref2d) ** 2))
            / np.sqrt(np.sum(ref2d ** 2)))
    assert got == pytest.approx(want, rel=1e-12)


def test_l2_flattened_2d_identity_is_zero():
    rng = np.random.default_rng(2)
    m = rng.standard_normal((8, 8))
    grid = np.arange(m.size, dtype=np.float64)
    assert probe.l2_relative_diff(grid, m.reshape(-1), grid, m.reshape(-1)) \
        == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Closed-form convergence core (mc-probe converge)
# --------------------------------------------------------------------------

def test_eff_variance_matches_numpy():
    rng = np.random.default_rng(0)
    O = rng.standard_normal((50, 7))
    assert np.allclose(probe.eff_variance(O), O.var(axis=0, ddof=1))


def test_antithetic_kills_linear_response():
    # Purely linear response O(ξ)=a·ξ: each pair (ξ,−ξ) averages to 0, so the
    # per-eval antithetic variance must be ~0 while iid variance is a²·Var(ξ)>0.
    rng = np.random.default_rng(1)
    xi = rng.standard_normal(64)
    a = 3.0
    rows = np.empty((128, 1))
    rows[0::2, 0] = a * xi           # node 2i:  +ξ_i
    rows[1::2, 0] = -a * xi          # node 2i+1: −ξ_i
    v_anti = probe.eff_variance_antithetic(rows)
    v_iid = probe.eff_variance(rows)
    assert v_anti[0] < 1e-20
    assert v_iid[0] > 1.0


def test_antithetic_keeps_even_response():
    # Purely even response O(ξ)=O(−ξ): antithetic can't help — per-eval variance
    # equals 2×(pair-mean var). With both rows equal, pair mean = the value, so
    # v_eff_anti = 2·Var(value), i.e. no reduction per eval vs the pair count.
    rng = np.random.default_rng(2)
    val = rng.standard_normal(32) ** 2       # even, node-dependent
    rows = np.empty((64, 1))
    rows[0::2, 0] = val
    rows[1::2, 0] = val                      # O(+ξ)=O(−ξ)
    v_anti = probe.eff_variance_antithetic(rows)
    assert v_anti[0] == pytest.approx(2.0 * val.var(ddof=1), rel=1e-9)


def test_eff_variance_antithetic_odd_rows_raise():
    with pytest.raises(ValueError, match="even row count"):
        probe.eff_variance_antithetic(np.zeros((5, 3)))


def test_conv_amplitude_and_nstar_roundtrip():
    # Two rows μ±d → eff_variance = 2d² per column; A = √(Σ2d²)/‖μ‖.
    mu = np.array([3.0, 4.0])          # ‖μ‖ = 5
    d = np.array([0.5, 0.5])
    O = np.vstack([mu + d, mu - d])
    v = probe.eff_variance(O)
    assert np.allclose(v, 2 * d ** 2)
    A = probe.conv_amplitude(v, mu)
    assert A == pytest.approx(np.sqrt(np.sum(2 * d ** 2)) / 5.0, rel=1e-12)
    # err(N)=A/√N ⇒ N*(tol)=(A/tol)²; err at N* must hit ~tol.
    tol = 0.01
    N = probe.n_star(A, tol)
    assert N == int(np.ceil((A / tol) ** 2))
    assert A / np.sqrt(N) <= tol + 1e-12


def test_nstar_even_for_pairs():
    A = 0.1
    n = probe.n_star(A, 0.014142, even=True)   # (0.1/0.014142)²≈50 → even 50
    assert n % 2 == 0


def test_control_variate_removes_correlated_variance():
    # c = O exactly ⇒ fixed β=1, residual O−(c−⟨c⟩)=⟨c⟩ constant ⇒ variance→0.
    rng = np.random.default_rng(3)
    O = rng.standard_normal((40, 5)) + 10.0
    c = O.copy()
    c_mean = c.mean(axis=0)
    beta = probe.fixed_beta(O, c, c_mean)
    assert np.allclose(beta, 1.0, atol=1e-9)
    r = probe.cv_residual(O, c, c_mean, beta)
    assert np.all(probe.eff_variance(r) < 1e-18)


def test_analyze_pool_shapes_and_iid_ratio():
    rng = np.random.default_rng(4)
    M, W = 64, 6
    O_iid = rng.standard_normal((M, W)) + 5.0
    O_anti = rng.standard_normal((M, W)) + 5.0
    c_iid = rng.standard_normal((M, W))
    c_anti = rng.standard_normal((M, W))
    c_mean = np.zeros(W)
    res = probe.analyze_pool(O_iid, O_anti, c_iid, c_anti, c_mean,
                             tols=[0.02, 0.01])
    for fam in ("iid", "iid_cv", "antithetic", "antithetic_cv"):
        assert set(res[fam]["nstar"]) == {0.02, 0.01}
        assert res[fam]["A"] > 0
    assert res["_var_ratio"]["iid"] == pytest.approx(1.0, rel=1e-12)
