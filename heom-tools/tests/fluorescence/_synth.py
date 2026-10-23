"""Synthetic raw solver HDF5 for flu-post tests (C++ writer's group layout).

Coupled dimer, Gaussian exciton lines. Absorption: ρ_k0(t) = μ⁺_k0 e^{−iE_k t − σ²t²/2};
fluorescence: same times p_k = e^{−βE_k}/Z with E_k → E_k − βσ². The pair then
satisfies I_F ∝ e^{−βω} I_A exactly, so a KMS fit must return T.
"""
from __future__ import annotations

import h5py
import numpy as np

from heom_tools.common.cumulant import _CM_FS

KB_CM = 0.695034800
T_K = 77.0
SIGMA_CM = 60.0
W0_CM = 12200.0
H_EFF = np.array([[-65.0, 100.0], [100.0, 135.0]])     # relative, λ on the diagonal
REORG = np.array([35.0, 35.0])
DIPOLES = np.array([[0.98, 0.2, 0.0], [0.3, -0.8, 0.5]])
DIPOLES /= np.linalg.norm(DIPOLES, axis=1, keepdims=True)
STRENGTHS = np.array([1.0, 0.7])

YAML = """system:
  n_states: 2
  temperature: 77.0
  hamiltonian:
    - [12100.0, 100.0]
    - [100.0, 12300.0]
  w0: 12200.0
bath:
  spectral_density: "Drude-Lorentz"
  eta: 70.0
  gamma_inverse: 100.0
  expansion_method: pade
  L_cutoff: 3
filter:
  enabled: false
{task}:
  dt: 2
"""


def dimer_eigen():
    E, U = np.linalg.eigh(H_EFF)
    N = 3
    evec = np.zeros((N, N), dtype=np.complex128)
    evec[0, 0] = 1.0
    evec[1:, 1:] = U
    return np.concatenate(([0.0], E)), evec


def rho_eq_dimer(pops: np.ndarray) -> np.ndarray:
    E, evec = dimer_eigen()
    U = evec[1:, 1:]
    rho = np.zeros((3, 3), dtype=np.complex128)
    rho[1:, 1:] = (U * pops) @ U.conj().T
    return rho


def boltzmann(E: np.ndarray) -> np.ndarray:
    w = np.exp(-(E - E.min()) / (KB_CM * T_K))
    return w / w.sum()


def write_raw(path: str, task: str, *, n_outputs: int = 2000, dt_fs: float = 2.0,
              pops: np.ndarray | None = None, yaml: str | None = None,
              with_rho_eq: bool = True) -> None:
    E, evec = dimer_eigen()
    U = evec[1:, 1:]
    t = np.arange(n_outputs) * dt_fs
    tc = t * _CM_FS
    beta = 1.0 / (KB_CM * T_K)
    mu_plus = np.zeros((3, 3, 3))
    mu_plus[:, 1:, 0] = (STRENGTHS[:, None] * DIPOLES).T
    if pops is None:
        pops = boltzmann(E[1:])
    shift = beta * SIGMA_CM**2 if task == "fluorescence" else 0.0
    weight = pops if task == "fluorescence" else np.ones(2)
    rho = np.zeros((3, n_outputs, 3, 3), dtype=np.complex128)
    for a in range(3):
        m_exc = U.conj().T @ mu_plus[a, 1:, 0]                     # μ⁺_{k0} in excitons
        amp = weight[:, None] * m_exc[:, None] * np.exp(
            -1j * (E[1:, None] - shift) * tc - 0.5 * (SIGMA_CM * tc) ** 2)
        rho[a, :, 1:, 0] = (U @ amp).T

    H = np.zeros((3, 3))
    H[1:, 1:] = H_EFF
    with h5py.File(path, "w") as f:
        f.attrs.update(task=task, status="complete", config_hash="synthetic",
                       simulation_time_seconds=1.0)
        p = f.create_group("parameters")
        p.attrs.update(spectrum_type=task, basis="with_ground", n_steps=n_outputs - 1,
                       output_interval=1, dt=dt_fs, temperature=T_K, w0_cm=W0_CM,
                       eta_cm=70.0, gamma_cm=53.0)
        p.create_dataset("dipoles", data=DIPOLES)
        p.create_dataset("strengths", data=STRENGTHS)
        p.create_dataset("hamiltonian_cm", data=H)
        p.create_dataset("reorg_cm", data=REORG)
        p.create_dataset("config_yaml", data=yaml if yaml is not None else YAML.format(task=task))
        d = f.create_group("derived")
        d.create_dataset("eigenvalues_relative_cm", data=E)
        d.create_dataset("eigenvectors", data=evec)
        d.create_dataset("markov_residue_cm", data=np.zeros(2))
        r = f.create_group("results")
        r.create_dataset("time", data=t)
        r.create_dataset("rho_xyz", data=rho)
        if task == "fluorescence" and with_rho_eq:
            r.create_dataset("rho_eq", data=rho_eq_dimer(pops))
            g = r.create_group("equilibration")
            for k in ("time_fs", "delta", "ratio", "estimate"):
                g.create_dataset(k, data=np.array([50.0, 100.0]))
            g.attrs.update(t_eq_fs=100.0, tolerance=1e-6, check_interval_fs=50.0,
                           t_max_fs=1000.0)
