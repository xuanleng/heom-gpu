"""Exact monomer correlation functions on the HEOM's own exponential expansion.

With C(t) = Σ_k c_k e^{−ν_k t}
(t ≥ 0), c̄_k the coefficients of C*(t) on the same ladders, and Δ the Markov
residue, the (e,g)-block hierarchy of one pigment solves in closed form:

    g(t)   = Σ_k c_k/ν_k² (e^{−ν_k t} + ν_k t − 1) + Δ t
    C_A(t) = exp[−iEt − g(t)]
    C_F(t) = exp[−iEt − g(t) − Σ_k (c_k − c̄_k)(1 − e^{−ν_k t})/ν_k²]

C_F is written with c̄ exactly as the hierarchy sees it (equilibrium ADOs
a_k = −i(c_k − c̄_k)/ν_k), so it is the L → ∞ limit of the HEOM even if the
pairing were imperfect. With exact pairing it equals exp[−i(E−2λ)t − g*(t)].
E is the H_eff (vertical) energy in any frame the caller likes; C(0) = 1.
"""
from __future__ import annotations

import numpy as np

from heom_tools.common.spectrum import AU2FS, AU2PERCM

_CM_FS = 1.0 / (AU2PERCM * AU2FS)   # (cm⁻¹)·fs → radians


def reorganization_cm(c_cm2, nu_cm) -> float:
    """λ = −Im Σ_k c_k/ν_k (= −Im ∫₀^∞ C(t) dt), cm⁻¹."""
    c = np.asarray(c_cm2, dtype=np.complex128)
    nu = np.asarray(nu_cm, dtype=np.complex128)
    return float(-np.sum(c / nu).imag)


def monomer_correlations(c_cm2, cbar_cm2, nu_cm, residue_cm: float,
                         E_cm: float, t_fs, *, cm_fs: float = _CM_FS
                         ) -> tuple[np.ndarray, np.ndarray]:
    """(C_A(t), C_F(t)) for one pigment; inputs as in HDF5 ``/derived``.

    ``cm_fs`` is the (cm⁻¹)·fs → rad factor. The solver's unit constants differ
    from CODATA by ~1e-8; comparisons below that level must pass the solver's.
    """
    c = np.asarray(c_cm2, dtype=np.complex128)[:, None]
    cbar = np.asarray(cbar_cm2, dtype=np.complex128)[:, None]
    nu = np.asarray(nu_cm, dtype=np.complex128)[:, None]
    if not (c.shape == cbar.shape == nu.shape):
        raise ValueError(f"c, cbar, nu must have one length: {c.shape}, {cbar.shape}, {nu.shape}")
    if np.any(nu.real <= 0.0):
        raise ValueError("every ladder needs Re ν > 0")
    t = np.asarray(t_fs, dtype=np.float64)[None, :] * cm_fs    # cm (so ν t is radians)

    decay = np.exp(-nu * t)
    g = np.sum(c / nu**2 * (decay + nu * t - 1.0), axis=0) + residue_cm * t[0]
    phase = -1j * E_cm * t[0]
    C_A = np.exp(phase - g)
    C_F = C_A * np.exp(-np.sum((c - cbar) * (1.0 - decay) / nu**2, axis=0))
    return C_A, C_F
