#!/usr/bin/env python3
"""Cheap stick-spectrum surrogate for the control-variate variance reduction
.

The physical insight: the *hard* part of the disordered absorption spectrum is
the inhomogeneous broadening — the peaks wandering with the site-energy disorder
ξ. That wandering is captured, to microsecond cost, by diagonalising the
single-excitation Hamiltonian H0 + diag(ξ) and dressing each excitonic
eigenstate with a FIXED homogeneous lineshape. The result c(ω; ξ) is a pure
function of ξ (no HEOM, no bath) that is strongly correlated with the expensive
HEOM observable O_HEOM(ω; ξ) — exactly what a control variate needs.

This module is PURE numpy and config-agnostic: it takes the base Hamiltonian
H0 [n_site, n_site] and per-site transition dipoles mu [n_site, 3] as arrays.
Loading them from an absorption config is the probe's job (component ④), keeping
this math out of any hot path (it stays in Python).

Unbiasedness note: the control variate ⟨O⟩ ≈ ⟨c⟩ + mean(O − β c) is unbiased
for ANY choice of c and ANY lineshape width — the width only tunes the variance
reduction ρ, never correctness. So `width_cm` is a free efficiency knob.
"""
import numpy as np

# Reuse the single source of truth for the FWHM->sigma conversion.
from heom_tools.mc_disorder.generate import SIGMA_PER_FWHM


def stick_spectrum(delta_eps, H0, mu, omega_cm, width_cm):
    """c(ω; ξ) for ONE disorder realisation.

    Parameters
    ----------
    delta_eps : (n_site,)   site-energy offsets ξ (cm^-1), added to diag(H0).
    H0        : (n_site, n_site)  base single-excitation Hamiltonian (cm^-1).
    mu        : (n_site, 3)  per-site transition dipole vectors.
    omega_cm  : (n_omega,)  frequency grid (cm^-1), same grid as O_HEOM.
    width_cm  : float        Gaussian homogeneous width (>0), efficiency knob.

    Returns
    -------
    (n_omega,) real stick spectrum broadened by a normalised Gaussian:
        c(ω) = Σ_k |d_k|² · N(ω; E_k, width_cm),
    where E_k, V are the eigenpairs of H0 + diag(ξ) and d_k = Σ_m V[m,k] μ_m is
    the excitonic transition dipole (oscillator strength |d_k|²).
    """
    delta_eps = np.asarray(delta_eps, dtype=np.float64)
    H0 = np.asarray(H0, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    omega_cm = np.asarray(omega_cm, dtype=np.float64)
    if width_cm <= 0.0:
        raise ValueError("width_cm must be > 0")

    H = H0 + np.diag(delta_eps)
    E, V = np.linalg.eigh(H)          # E:(n_site,)  V:(n_site,n_site) columns=states
    d = V.T @ mu                      # (n_site, 3) excitonic dipoles
    osc = np.einsum("ki,ki->k", d, d)  # (n_site,) oscillator strengths
    # Normalised Gaussian per eigenpeak, summed weighted by oscillator strength.
    z = (omega_cm[:, None] - E[None, :]) / width_cm
    g = np.exp(-0.5 * z * z) / (width_cm * np.sqrt(2.0 * np.pi))
    return g @ osc                    # (n_omega,)


def mean_stick_spectrum(H0, mu, sigma_cm, omega_cm, width_cm,
                        n_samples=1_000_000, seed=0, batch=20_000):
    """⟨c(ω)⟩ estimated over `n_samples` iid Gaussian draws ξ ~ N(0, diag σ²).

    Computed on the SAME grid and lineshape as `stick_spectrum` (consistency is
    the control variate). Cheap enough for many draws; batched to bound
    memory. Deterministic given `seed`.
    """
    H0 = np.asarray(H0, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma_cm = np.asarray(sigma_cm, dtype=np.float64)
    omega_cm = np.asarray(omega_cm, dtype=np.float64)
    n_site = H0.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC0]))

    acc = np.zeros(omega_cm.shape[0], dtype=np.float64)
    done = 0
    while done < n_samples:
        m = min(batch, n_samples - done)
        xi = rng.standard_normal((m, n_site)) * sigma_cm[None, :]
        for r in range(m):
            acc += stick_spectrum(xi[r], H0, mu, omega_cm, width_cm)
        done += m
    return acc / n_samples


def sigma_from_fwhm(fwhm_cm):
    """Per-site σ (cm^-1) from FWHM, using the generator's single conversion."""
    return np.asarray(fwhm_cm, dtype=np.float64) * SIGMA_PER_FWHM


def control_variate_apply(o_nodes, c_nodes, c_mean, beta=None):
    """Combine per-node HEOM spectra with the stick surrogate (Decision C).

    Parameters
    ----------
    o_nodes : (N, n_omega)  per-node HEOM observable O_HEOM(ω; ξ_i).
    c_nodes : (N, n_omega)  per-node surrogate c(ω; ξ_i), SAME grid.
    c_mean  : (n_omega,)    exact ⟨c(ω)⟩ (from mean_stick_spectrum).
    beta    : (n_omega,) or None. If None, the full-β estimate β(ω) =
              Cov_i(O, c)/Var_i(c) is regressed per bin (study/probe mode).

    Returns
    -------
    (estimate, beta, rho):
        estimate : (n_omega,)  CV estimate  mean(O) − β·(mean(c) − ⟨c⟩)
                   (standard control variate; unbiased for any β).
        beta     : (n_omega,)  the β used.
        rho      : (n_omega,)  per-bin corr(O, c) (variance-reduction predictor
                   1/(1−ρ²)); NaN where Var(c)=0.
    """
    o_nodes = np.asarray(o_nodes, dtype=np.float64)
    c_nodes = np.asarray(c_nodes, dtype=np.float64)
    c_mean = np.asarray(c_mean, dtype=np.float64)
    if o_nodes.shape != c_nodes.shape:
        raise ValueError("o_nodes and c_nodes must have the same shape (N, n_omega)")

    o_bar = o_nodes.mean(axis=0)
    c_bar = c_nodes.mean(axis=0)
    do = o_nodes - o_bar[None, :]
    dc = c_nodes - c_bar[None, :]
    cov = (do * dc).mean(axis=0)
    var_c = (dc * dc).mean(axis=0)
    var_o = (do * do).mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        rho = cov / np.sqrt(var_c * var_o)
        beta_used = (cov / var_c) if beta is None else np.asarray(beta, dtype=np.float64)
    # Where the surrogate is flat (Var(c)=0) the CV correction is 0.
    beta_used = np.where(np.isfinite(beta_used), beta_used, 0.0)
    rho = np.where(np.isfinite(rho), rho, 0.0)

    estimate = o_bar - beta_used * (c_bar - c_mean)
    return estimate, beta_used, rho
