"""Absorption post-processing core: compute + HDF5 write (no matplotlib)."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import h5py
import numpy as np

from heom_tools.common.spectrum import (
    SpectrumData,
    StickEntry,
    build_mu_minus_site,
    build_stick_spectrum,
    compute_correlation,
    cut_to_window_batch,
    fft_to_spectra_batch,
    find_peaks_safe,
    ground_eigen_index,
)
from heom_tools.common.spectrum_metrics import peak_normalize_factor

_COMPLETENESS_WARN_TOL = 1e-9
COMPLETENESS_WARN_TOL = _COMPLETENESS_WARN_TOL

_SOURCE_PROVENANCE_PATHS = {
    "hamiltonian_cm": "/parameters/hamiltonian_cm",
    "eigenvalues_relative_cm": "/derived/eigenvalues_relative_cm",
}


@dataclass
class AbsorptionPostResult:
    """All derived absorption data that should live in one HDF5 artifact."""

    spec: SpectrumData
    ct_xyz: np.ndarray
    ct_iso_total: np.ndarray
    ct_iso_per_exciton: np.ndarray
    ct_iso_sum_check: np.ndarray
    omega_cm: np.ndarray
    intensity_total: np.ndarray
    intensity_per_exciton: np.ndarray
    sticks: list[StickEntry]
    peaks_total: list[tuple[float, float]]
    peaks_per_exciton: list[list[tuple[float, float]]]
    w0_cm: float
    wmin_cm: float
    wmax_cm: float

    @property
    def n_excitons(self) -> int:
        return self.ct_iso_per_exciton.shape[1]

    @property
    def consistency_max_error(self) -> float:
        d = self.ct_iso_sum_check - self.ct_iso_total
        if d.size == 0:
            return 0.0
        return float(np.max(np.abs(d)))

    @property
    def norm(self) -> float:
        return peak_normalize_factor(self.intensity_total)

    @property
    def intensity_total_norm(self) -> np.ndarray:
        return self.intensity_total / self.norm

    @property
    def intensity_per_exciton_norm(self) -> np.ndarray:
        return self.intensity_per_exciton / self.norm


def compute_absorption_post(
    spec: SpectrumData,
    wmin_cm: float,
    wmax_cm: float,
    *,
    strict_ground_w0: bool = True,
) -> AbsorptionPostResult:
    """Compute total absorption, exciton decomposition, sticks, and peaks."""
    if spec.n_outputs < 1:
        raise ValueError("compute_absorption_post requires n_outputs >= 1")

    v = spec.eigenvectors
    g_eig = ground_eigen_index(v)
    ground_rel = float(spec.eigenvalues_relative_cm[g_eig])
    if abs(ground_rel) > 1e-3:
        msg = (
            f"eigenvalues_relative_cm[ground_eigen_index={g_eig}] = {ground_rel:.3e}; "
            "with_ground rotating frame expects Hilbert |g⟩ at relative energy 0"
        )
        if strict_ground_w0:
            raise RuntimeError(
                f"{msg}; check HDF5 Hamiltonian / w0_cm or pass --w0 for FFT axis only"
            )
        warnings.warn(
            f"{msg}; continuing (--w0 override): FFT ω-axis uses w0_cm={spec.w0_cm:g} cm⁻¹; "
            "sticks use uncorrected eigenvalues_absolute_cm.",
            UserWarning,
            stacklevel=2,
        )

    n_hilbert = spec.N
    exciton_idxs = sorted(
        (k for k in range(n_hilbert) if k != g_eig),
        key=lambda k: spec.eigenvalues_absolute_cm[k],
    )
    n_excitons = len(exciton_idxs)

    mu_minus_site = build_mu_minus_site(spec.dipoles, spec.strengths, n_hilbert)
    ct_xyz = compute_correlation(spec.rho_xyz, mu_minus_site)
    ct_iso_total = ct_xyz.mean(axis=0)

    v_dag = v.conj().T

    mu_minus_exc = np.einsum("ij,ajk,kl->ail", v_dag, mu_minus_site, v)
    rho_exc = np.einsum("ij,atjk,kl->atil", v_dag, spec.rho_xyz, v)

    ct_per_alpha = np.zeros((3, spec.n_outputs, n_excitons), dtype=np.complex128)
    for j, k in enumerate(exciton_idxs):
        ct_per_alpha[:, :, j] = (
            mu_minus_exc[:, g_eig, k][:, None] * rho_exc[:, :, k, g_eig]
        )

    ct_iso_per_exciton = ct_per_alpha.mean(axis=0)
    ct_iso_sum_check = ct_iso_per_exciton.sum(axis=1)

    w0 = spec.w0_cm
    ct_batch = np.vstack(
        (ct_iso_total[np.newaxis, :], ct_iso_per_exciton.T),
    )
    omega, specs_full = fft_to_spectra_batch(ct_batch, dt_fs=spec.dt_fs, w0_cm=w0)

    omega_cut, specs_cut = cut_to_window_batch(omega, specs_full, wmin_cm, wmax_cm)
    intensity_total = specs_cut[0].real
    intensity_per_exciton = specs_cut[1:].real.T

    sticks = build_stick_spectrum(
        spec.eigenvalues_absolute_cm,
        spec.eigenvectors,
        spec.dipoles,
        spec.strengths,
    )
    if len(sticks) != n_excitons:
        raise RuntimeError(
            f"internal: len(sticks)={len(sticks)} != n_excitons={n_excitons}"
        )

    norm = peak_normalize_factor(intensity_total)
    peaks_total = find_peaks_safe(omega_cut, intensity_total / norm)
    peaks_per_exciton = [
        find_peaks_safe(omega_cut, intensity_per_exciton[:, k] / norm)
        for k in range(n_excitons)
    ]

    return AbsorptionPostResult(
        spec=spec,
        ct_xyz=ct_xyz,
        ct_iso_total=ct_iso_total,
        ct_iso_per_exciton=ct_iso_per_exciton,
        ct_iso_sum_check=ct_iso_sum_check,
        omega_cm=omega_cut,
        intensity_total=intensity_total,
        intensity_per_exciton=intensity_per_exciton,
        sticks=sticks,
        peaks_total=peaks_total,
        peaks_per_exciton=peaks_per_exciton,
        w0_cm=w0,
        wmin_cm=float(wmin_cm),
        wmax_cm=float(wmax_cm),
    )


def _sticks_table(sticks: list[StickEntry]) -> np.ndarray:
    table = np.zeros((len(sticks), 4), dtype=np.float64)
    for i, st in enumerate(sticks):
        if st.dominant_site_index < 0:
            raise ValueError(
                f"sticks[{i}].dominant_site_index = {st.dominant_site_index} < 0; "
                "with_ground basis requires a valid 0-based pigment index"
            )
        table[i, 0] = st.omega_cm
        table[i, 1] = st.oscillator_strength
        table[i, 2] = float(st.dominant_site_index + 1)
        table[i, 3] = st.dominant_site_fraction
    return table


def _peaks_total_table(peaks: list[tuple[float, float]]) -> np.ndarray:
    if not peaks:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(peaks, dtype=np.float64)


def _peaks_per_exciton_table(peaks_per_exciton: list[list[tuple[float, float]]]) -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    for k, peaks in enumerate(peaks_per_exciton, 1):
        for omega, intensity in peaks:
            rows.append((float(k), float(omega), float(intensity)))
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _copy_source_provenance_datasets(src_path: str, dst_group, names) -> None:
    with h5py.File(src_path, "r") as src:
        for name in names:
            location = _SOURCE_PROVENANCE_PATHS.get(name, name)
            if location in src and name not in dst_group:
                dst_group.create_dataset(name, data=src[location][...])


def _source_precision_mode(src_path: str) -> str:
    """Carry the solver build's precision provenance onto the derived spectrum
    so a lone ``*_spectrum.h5`` self-documents fp64 vs mixed_fp32. Legacy sources
    (pre mixed-precision build) have no attr and were all FP64 — same convention
    as the MC merge guard (heom_tools.common.mc_merge)."""
    try:
        with h5py.File(src_path, "r") as src:
            v = src.attrs.get("precision_mode")
    except OSError:
        return "fp64"
    if v is None:
        return "fp64"
    return v.decode() if isinstance(v, bytes) else v


def write_absorption_h5(result: AbsorptionPostResult, out_path: str) -> None:
    """Write all numeric post-processing outputs into one HDF5 file."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f.attrs["generated_by"] = "heom_tools.absorption.post"
        f.attrs["entry_point"] = "las-post"
        f.attrs["source_h5"] = os.path.abspath(result.spec.filename)
        f.attrs["precision_mode"] = _source_precision_mode(result.spec.filename)
        f.attrs["spectrum_type"] = "absorption"
        f.attrs["basis"] = result.spec.basis
        f.attrs["w0_cm"] = result.w0_cm
        f.attrs["wmin_cm"] = result.wmin_cm
        f.attrs["wmax_cm"] = result.wmax_cm
        f.attrs["normalization"] = result.norm
        f.attrs["consistency_max_error"] = result.consistency_max_error
        f.attrs["n_excitons"] = result.n_excitons

        f.create_dataset("omega_cm", data=result.omega_cm)
        f.create_dataset("intensity_total", data=result.intensity_total)
        f.create_dataset("intensity_total_norm", data=result.intensity_total_norm)
        f.create_dataset("intensity_per_exciton", data=result.intensity_per_exciton)
        f.create_dataset("intensity_per_exciton_norm", data=result.intensity_per_exciton_norm)

        corr = f.create_group("correlation")
        corr.create_dataset("time_fs", data=result.spec.time_fs)
        corr.create_dataset("ct_xyz_real", data=result.ct_xyz.real)
        corr.create_dataset("ct_xyz_imag", data=result.ct_xyz.imag)
        corr.create_dataset("ct_iso_real", data=result.ct_iso_total.real)
        corr.create_dataset("ct_iso_imag", data=result.ct_iso_total.imag)
        corr.create_dataset("ct_iso_per_exciton_real", data=result.ct_iso_per_exciton.real)
        corr.create_dataset("ct_iso_per_exciton_imag", data=result.ct_iso_per_exciton.imag)
        corr.create_dataset("ct_iso_sum_check_real", data=result.ct_iso_sum_check.real)
        corr.create_dataset("ct_iso_sum_check_imag", data=result.ct_iso_sum_check.imag)

        sticks = f.create_group("sticks")
        stick_ds = sticks.create_dataset("table", data=_sticks_table(result.sticks))
        stick_ds.attrs["columns"] = (
            "omega_cm,oscillator_strength,dominant_site_index_1based,dominant_site_fraction"
        )

        peaks = f.create_group("peaks")
        total_ds = peaks.create_dataset("total", data=_peaks_total_table(result.peaks_total))
        total_ds.attrs["columns"] = "omega_cm,intensity_norm"
        per_ds = peaks.create_dataset(
            "per_exciton", data=_peaks_per_exciton_table(result.peaks_per_exciton)
        )
        per_ds.attrs["columns"] = "exciton_index_1based,omega_cm,intensity_norm"

        src = f.create_group("source")
        src.create_dataset("dipoles", data=result.spec.dipoles)
        src.create_dataset("strengths", data=result.spec.strengths)
        src.create_dataset("eigenvalues_absolute_cm", data=result.spec.eigenvalues_absolute_cm)
        src.create_dataset("eigenvectors_real", data=result.spec.eigenvectors.real)
        src.create_dataset("eigenvectors_imag", data=result.spec.eigenvectors.imag)
        _copy_source_provenance_datasets(
            result.spec.filename,
            src,
            names=("hamiltonian_cm", "eigenvalues_relative_cm"),
        )
        src.attrs["temperature_K"] = result.spec.temperature_K
        src.attrs["eta_cm"] = result.spec.eta_cm
        src.attrs["gamma_bath_cm"] = result.spec.gamma_cm

