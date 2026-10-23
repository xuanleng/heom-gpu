"""Fluorescence post-processing core: compute + HDF5 write (no matplotlib).

The linear part (correlation,
exciton split, FFT, peaks) is the absorption code verbatim
(``common/linear_spectrum.py``); this module adds the emission sticks p_k f_k,
the three population diagnostics, the optional ω³ factor and the KMS fit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import h5py
import numpy as np

from heom_tools.common.linear_spectrum import (
    LinearPostResult,
    compute_linear_post,
    exciton_indices,
    write_linear_h5,
)
from heom_tools.common.spectrum import (
    SpectrumData,
    build_stick_spectrum,
    default_absorption_frequency_window_cm,
)
from heom_tools.fluorescence.post.kms import KB_CM, KMSFit


@dataclass
class Populations:
    """Exciton populations (ascending H_eff energy), three sources side by side."""
    heom: Optional[np.ndarray]     # (U† ρ_eq U)_kk; None for MC-merged input
    boltz_heff: np.ndarray         # e^{−βE_k}/Z, E_k = H_eff eigenvalues
    boltz_hs: np.ndarray           # (U† e^{−βH_S} U)_kk / Z, H_S = H_eff − diag(λ)

    @property
    def stick_weights(self) -> str:
        return "heom" if self.heom is not None else "boltzmann_heff"

    @property
    def weights(self) -> np.ndarray:
        return self.heom if self.heom is not None else self.boltz_heff


@dataclass
class FluorescencePostResult:
    linear: LinearPostResult
    populations: Populations
    omega3: Optional[np.ndarray] = None   # S_em = (ω/ω_ref)³ I_F, absolute ω
    omega_ref_cm: Optional[float] = None
    kms: Optional[KMSFit] = None


def _require_reorg(spec: SpectrumData) -> np.ndarray:
    if spec.reorg_cm is None:
        raise ValueError(f"{spec.filename!r} has no /parameters/reorg_cm "
                         "(fluorescence output needs heom_solver ≥ 2.1.1)")
    return spec.reorg_cm


def _boltzmann_diag(H_cm: np.ndarray, U: np.ndarray, kT: float) -> np.ndarray:
    """diag(U† e^{−H/kT} U)/Z for a Hermitian H on the excited block."""
    w, V = np.linalg.eigh(H_cm)
    p = np.exp(-(w - w.min()) / kT)
    rho = (V * (p / p.sum())) @ V.conj().T
    return np.real(np.einsum("ik,ij,jk->k", U.conj(), rho, U))


def compute_populations(spec: SpectrumData) -> Populations:
    if spec.temperature_K <= 0.0:
        raise ValueError(f"temperature = {spec.temperature_K} K; Boltzmann weights need T > 0")
    kT = KB_CM * spec.temperature_K
    exc = exciton_indices(spec)
    U = spec.eigenvectors[1:, exc]                                # excited rows only
    E = spec.eigenvalues_relative_cm[exc]
    H_eff = (U * E) @ U.conj().T
    H_s = H_eff - np.diag(_require_reorg(spec))

    w = np.exp(-(E - E.min()) / kT)
    heom = None
    if spec.rho_eq is not None:
        heom = np.real(np.einsum("ik,ij,jk->k", U.conj(), spec.rho_eq[1:, 1:], U))
    return Populations(heom=heom, boltz_heff=w / w.sum(),
                       boltz_hs=_boltzmann_diag(H_s, U, kT))


def default_fluorescence_frequency_window_cm(spec: SpectrumData) -> tuple[float, float]:
    """Absorption default window with the red edge pushed out by 2·max λ."""
    lo, hi = default_absorption_frequency_window_cm(spec.eigenvalues_absolute_cm,
                                                    eigenvectors=spec.eigenvectors)
    return lo - 2.0 * float(np.max(_require_reorg(spec))), hi


def compute_fluorescence_post(
    spec: SpectrumData,
    wmin_cm: float,
    wmax_cm: float,
    *,
    omega3: bool = False,
    strict_ground_w0: bool = True,
) -> FluorescencePostResult:
    """Emission spectrum, exciton split, sticks p_k·f_k, populations, optional ω³."""
    pops = compute_populations(spec)
    sticks = [replace(st, oscillator_strength=st.oscillator_strength * float(p))
              for st, p in zip(build_stick_spectrum(spec.eigenvalues_absolute_cm,
                                                    spec.eigenvectors, spec.dipoles,
                                                    spec.strengths),
                               pops.weights)]
    lin = compute_linear_post(spec, wmin_cm, wmax_cm, sticks,
                              strict_ground_w0=strict_ground_w0)
    out = FluorescencePostResult(linear=lin, populations=pops)
    if omega3:
        w_ref = float(lin.omega_cm[int(np.argmax(lin.intensity_total))])
        out.omega_ref_cm = w_ref
        out.omega3 = (lin.omega_cm / w_ref) ** 3 * lin.intensity_total
    return out


def write_fluorescence_h5(result: FluorescencePostResult, out_path: str) -> None:
    """``write_linear_h5`` layout + populations / equilibrium / ω³ / KMS groups."""
    lin, pops, spec = result.linear, result.populations, result.linear.spec
    write_linear_h5(lin, out_path, spectrum_type="fluorescence",
                    generated_by="heom_tools.fluorescence.post", entry_point="flu-post")
    with h5py.File(out_path, "a") as f:
        f["sticks"].attrs["stick_weights"] = pops.stick_weights
        g = f.create_group("populations")
        g.attrs["order"] = "excitons by ascending H_eff energy"
        if pops.heom is not None:
            g.create_dataset("heom", data=pops.heom)
        g.create_dataset("boltzmann_heff", data=pops.boltz_heff)
        g.create_dataset("boltzmann_hs", data=pops.boltz_hs)
        if spec.rho_eq is not None:
            f["source"].create_dataset("rho_eq", data=spec.rho_eq)
        if spec.reorg_cm is not None:
            f["source"].create_dataset("reorg_cm", data=spec.reorg_cm)
        if spec.equilibration is not None:
            f["source"].attrs["t_eq_fs"] = spec.equilibration.t_eq_fs
            f["source"].attrs["eq_tolerance"] = spec.equilibration.tolerance
        if result.omega3 is not None:
            ds = f.create_dataset("intensity_omega3", data=result.omega3)
            ds.attrs["omega_ref_cm"] = result.omega_ref_cm
            ds.attrs["definition"] = "(omega/omega_ref)^3 * intensity_total, absolute omega"
        if result.kms is not None:
            result.kms.write(f.create_group("kms"))
