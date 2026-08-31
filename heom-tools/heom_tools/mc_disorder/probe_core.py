#!/usr/bin/env python3
"""Pure sampling-convergence math for `mc-probe` (numpy ONLY — no GPU, no I/O,
no heavy imports, not even lazy ones).

This is the unit-tested heart that every mc-probe verdict rests on, split out of
probe.py so it imports anywhere with just numpy. Two blocks:

  * the L² relative-difference metric (1-D spectra AND flattened 2-D maps share
    it — implemented in ``heom_tools.common.spectrum_metrics``, re-exported here),
  * the closed-form 1/√N convergence core: from a per-node pool O[M,ω] the
    plain-mean estimator's L² sampling error obeys err(N)=A/√N, so N*(tol)=
    (A/tol)². antithetic and CV only change the effective variance v_eff —
    everything else is the SAME formula, so the methods are one code path.
"""
import numpy as np

from heom_tools.common.spectrum_metrics import l2_relative_diff


# --------------------------------------------------------------------------
# Closed-form convergence core
#
# "Minimum evaluations to converge" — answered analytically, not by resampling.
# From a per-node pool of raw spectra O[M,ω] the plain-mean estimator with N
# evaluations has per-ω variance v_eff(ω)/N, so its L² sampling error obeys a
# clean 1/√N law:  err(N) = A/√N,  A = ‖√Σ_ω v_eff‖ / ‖μ‖.  Inverting gives
# N*(tol) = (A/tol)².  antithetic and CV only change v_eff — everything else is
# the same formula, so the three methods are one code path.
# --------------------------------------------------------------------------

def eff_variance(O):
    """Per-ω per-evaluation variance of the plain-mean estimator (iid rows)."""
    return np.asarray(O, dtype=np.float64).var(axis=0, ddof=1)


def eff_variance_antithetic(O):
    """O = 2K rows as K antithetic pairs (rows 2i, 2i+1 = ±ξ_i). One pair costs
    2 evaluations, so the per-EVALUATION variance is 2·Var[pair-mean] — this is
    < iid's whenever the odd (linear-in-ξ) response dominates."""
    O = np.asarray(O, dtype=np.float64)
    if O.shape[0] % 2:
        raise ValueError("antithetic pool needs an even row count (±ξ pairs)")
    p = 0.5 * (O[0::2] + O[1::2])
    return 2.0 * p.var(axis=0, ddof=1)


def fixed_beta(O, c, c_mean):
    """Per-ω control-variate slope β(ω)=cov(O,c)/var(c) over the whole pool
    (FIXED β — the production-safe channel; per-subset β overfits at small N)."""
    O = np.asarray(O, dtype=np.float64)
    cc = np.asarray(c, dtype=np.float64) - np.asarray(c_mean, dtype=np.float64)
    Od = O - O.mean(axis=0)
    ccd = cc - cc.mean(axis=0)
    num = (Od * ccd).mean(axis=0)
    den = (ccd * ccd).mean(axis=0)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def cv_residual(O, c, c_mean, beta):
    """Per-node control-variate residual O_i − β·(c_i − ⟨c⟩): same mean as O
    (β·0 in expectation), lower variance. Feed to eff_variance[_antithetic]."""
    cc = np.asarray(c, dtype=np.float64) - np.asarray(c_mean, dtype=np.float64)
    return np.asarray(O, dtype=np.float64) - beta * cc


def conv_amplitude(v_eff, mu):
    """A in err(N)=A/√N: total sampling std per √evaluation, relative to ‖μ‖."""
    mu_l2 = float(np.sqrt(np.sum(np.asarray(mu, dtype=np.float64) ** 2)))
    if mu_l2 == 0.0:
        raise ValueError("pool mean has zero L² norm")
    return float(np.sqrt(np.sum(np.asarray(v_eff, dtype=np.float64)))) / mu_l2


def n_star(A, tol, even=False):
    """Smallest N reaching self-convergence tol: N*=(A/tol)² (even for pairs)."""
    n = int(np.ceil((A / tol) ** 2))
    if even and n % 2:
        n += 1
    return n


def analyze_pool(O_iid, O_anti, c_iid, c_anti, c_mean, tols):
    """Closed-form convergence from a per-node pool. Returns per-method
    (A, {tol: N*}) plus the effective-variance ratios. Pure numpy — the
    unit-tested heart of `mc-probe converge`.

    Methods: iid, iid_cv, antithetic, antithetic_cv. μ (the converged spectrum
    the estimators approach) is the iid pool mean (unbiased, no ± pairing)."""
    mu = np.asarray(O_iid, dtype=np.float64).mean(axis=0)
    b_iid = fixed_beta(O_iid, c_iid, c_mean)
    b_anti = fixed_beta(O_anti, c_anti, c_mean)
    v = {
        "iid": eff_variance(O_iid),
        "iid_cv": eff_variance(cv_residual(O_iid, c_iid, c_mean, b_iid)),
        "antithetic": eff_variance_antithetic(O_anti),
        "antithetic_cv": eff_variance_antithetic(
            cv_residual(O_anti, c_anti, c_mean, b_anti)),
    }
    out = {}
    for fam, v_eff in v.items():
        A = conv_amplitude(v_eff, mu)
        even = fam.startswith("antithetic")
        out[fam] = {"A": A,
                    "nstar": {t: n_star(A, t, even=even) for t in tols}}
    base = np.sum(v["iid"])
    out["_var_ratio"] = {fam: float(base / np.sum(vv)) for fam, vv in v.items()}
    out["_mu"] = mu
    return out
