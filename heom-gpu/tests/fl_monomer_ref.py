"""NumPy monomer hierarchy: the (e,g) block of one pigment, Shi-scaled, RK4.

Reads the expansion straight from a solver HDF5 (``/derived/bath_{c,cbar,nu}``,
``markov_residue_cm``) so it checks the propagation and the equilibrium
start, not a second derivation of the coefficients.

On the (e,g) block V^×X = X and XV = 0, so for every ADO n

    dX_n/dt = −(iE + n·ν + Δ) X_n − i Σ_k √((n_k+1) s_k) X_{n+e_k}
              − i Σ_k c_k √(n_k/s_k) X_{n−e_k},       s_k = √(|c_k||c̄_k|).

Start: absorption X_0 = 1; fluorescence X_n = Π_k (a_k/√s_k)^{n_k}/√(n_k!),
a_k = −i(c_k − c̄_k)/ν_k (the exact monomer equilibrium, scaled). Returns X_0(t).
"""
from __future__ import annotations

import itertools
import math

import h5py
import numpy as np

# The solver's own constants (include/types.hpp), not CODATA: they differ by
# ~1e-8, which over a few hundred fs of phase is far above the 1e-10 gate.
CM_TO_AU, FS_TO_AU = 4.556335e-6, 41.341374
CM_FS = CM_TO_AU * FS_TO_AU          # (cm⁻¹)(fs) → rad


def _hierarchy(D: int, L: int):
    rows = [tuple(np.bincount(c, minlength=D)) if c else (0,) * D
            for tot in range(L + 1)
            for c in itertools.combinations_with_replacement(range(D), tot)]
    look = {r: i for i, r in enumerate(rows)}
    M = len(rows)
    plus = np.full((M, D), M)
    minus = np.full((M, D), M)
    for i, r in enumerate(rows):
        for k in range(D):
            plus[i, k] = look.get(r[:k] + (r[k] + 1,) + r[k + 1:], M)
            if r[k]:
                minus[i, k] = look[r[:k] + (r[k] - 1,) + r[k + 1:]]
    return np.array(rows), plus, minus


def monomer_x0(h5_path: str, task: str) -> tuple[np.ndarray, np.ndarray]:
    """(time_fs, X_0(t)/X_0(0)) on the file's own grid, L, dt and expansion."""
    with h5py.File(h5_path, "r") as f:
        c = f["/derived/bath_c_cm2"][...].astype(np.complex128)
        cbar = f["/derived/bath_cbar_cm2"][...].astype(np.complex128)
        nu = f["/derived/bath_nu_cm"][...].astype(np.complex128)
        delta = float(f["/derived/markov_residue_cm"][0])
        E = float(f["/parameters/hamiltonian_cm"][1, 1])
        p = f["/parameters"].attrs
        L, dt, every = int(p["L_cutoff"]), float(p["dt"]), int(p["output_interval"])
        n_out = f["/results/time"].shape[0]
        if f["/parameters/hamiltonian_cm"].shape != (2, 2):
            raise ValueError("fl_monomer_ref handles one pigment (with_ground N = 2)")

    s = np.sqrt(np.abs(c) * np.abs(cbar))
    N, plus, minus = _hierarchy(len(c), L)
    isf = np.exp(-0.5 * np.array([sum(math.lgamma(x + 1) for x in r) for r in N]))
    if task == "fluorescence":
        a = -1j * (c - cbar) / nu
        X = np.prod(np.power((a / np.sqrt(s))[None, :], N), axis=1) * isf
    else:
        X = np.zeros(len(N), np.complex128)
        X[0] = 1.0
    decay = 1j * E + N @ nu + delta
    up = -1j * np.sqrt((N + 1) * s[None, :])
    dn = -1j * c[None, :] * np.sqrt(N / s[None, :])

    def rhs(X):
        Xp = np.append(X, 0.0)
        return -decay * X + np.sum(up * Xp[plus] + dn * Xp[minus], axis=1)

    h = dt * CM_FS
    out = np.empty(n_out, np.complex128)
    out[0] = X[0]
    for i in range(1, n_out):
        for _ in range(every):
            k1 = rhs(X); k2 = rhs(X + 0.5 * h * k1)
            k3 = rhs(X + 0.5 * h * k2); k4 = rhs(X + h * k3)
            X = X + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = X[0]
    return np.arange(n_out) * every * dt, out / out[0]
