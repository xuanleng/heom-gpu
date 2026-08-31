#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared spectrum operators: HDF5 load, dipole rebuild, FFT, sticks, peaks.

Drivers call these functions and own plotting and filenames. This module
does not write HDF5 or draw. It may emit UserWarning, for example when a
dipole is renormalized.

Units are atomic units: AU2FS fs per au time, AU2PERCM cm^-1 per Hartree.
find_peaks is pure NumPy.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import h5py

from heom_tools.common import heom_h5


# ---------------------------------------------------------------------------
# Physical constants, matching system_info.cpp.
# ---------------------------------------------------------------------------

AU2FS    = 2.418884254e-2     # 1 au_time = AU2FS fs
AU2PERCM = 219474.63068       # 1 Hartree = AU2PERCM cm⁻¹
DW_DEFAULT_CM = 2.0           # Upper bound on the FFT frequency step.

# Zero-padding length:
#     Num = int(1.0 / (dw * dt * FFT_GRID_SCALE_LAS))
FFT_GRID_SCALE_LAS = 3e-5


# ---------------------------------------------------------------------------
# HDF5 load
# ---------------------------------------------------------------------------

# Group paths used by ``heom_solver absorption`` (group-layout schema, see
# src/output/hdf5_metadata.cpp). Listed here so the sniffer and the loader
# can never drift on which datasets are mandatory.
_SPECTRUM_H5_REQUIRED_PATHS: tuple[str, ...] = (
    "/results/rho_xyz",
    "/results/time",
    "/parameters/dipoles",
    "/parameters/strengths",
    "/parameters/hamiltonian_cm",
    "/derived/eigenvalues_relative_cm",
    "/derived/eigenvectors",
)


def is_raw_with_ground_absorption_h5(path: str) -> bool:
    """Cheap structural check for ``heom_solver absorption`` raw HDF5.

    Requires the group-layout paths in ``_SPECTRUM_H5_REQUIRED_PATHS`` to
    exist and ``/parameters/basis == 'with_ground'``. Shape and numerical
    validity stay with ``load_spectrum_h5``.
    """
    return absorption_h5_skip_reason(path) is None


def _decode_h5_attr_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


def absorption_h5_skip_reason(path: str) -> Optional[str]:
    """Return a human-readable reason when ``path`` is not raw absorption input."""
    try:
        with h5py.File(path, "r") as f:
            if "omega_cm" in f:
                return "processed spectrum HDF5 (use las-plot / las-compare on this file)"
            task = _decode_h5_attr_text(f.attrs.get("task", ""))
            if task and task != "absorption":
                return (
                    f"task={task!r} (need `heom_solver absorption` output; "
                    "filename prefix such as dis100 is ignored)"
                )
            missing = [p for p in _SPECTRUM_H5_REQUIRED_PATHS if p not in f]
            if missing:
                short = [p.split("/")[-1] for p in missing]
                return f"missing absorption datasets: {', '.join(short)}"
            basis = heom_h5.param(f, "basis", None)
            if basis != "with_ground":
                return f"basis={basis!r} (need with_ground)"
    except OSError as e:
        return str(e)
    return None


_PROCESSED_ABSORPTION_SUFFIXES: tuple[str, ...] = ("_spectrum", "_post")


def is_processed_absorption_h5_filename(filename: str) -> bool:
    """True for post-processed absorption HDF5 basenames (current or legacy)."""
    return any(filename.endswith(f"{suffix}.h5") for suffix in _PROCESSED_ABSORPTION_SUFFIXES)


def processed_absorption_h5_path(source_h5: str) -> str:
    """``<dir>/<stem>_spectrum.h5`` next to the raw solver output."""
    stem = basename_noext(source_h5)
    directory = os.path.dirname(os.path.abspath(source_h5))
    return os.path.join(directory, f"{stem}_spectrum.h5")


def has_processed_absorption_sibling(source_h5: str) -> bool:
    """True when either ``*_spectrum.h5`` or legacy ``*_post.h5`` exists."""
    stem = basename_noext(source_h5)
    directory = os.path.dirname(os.path.abspath(source_h5))
    return any(
        os.path.isfile(os.path.join(directory, f"{stem}{suffix}.h5"))
        for suffix in _PROCESSED_ABSORPTION_SUFFIXES
    )


@dataclass
class SpectrumData:
    """Fields unpacked from a raw absorption HDF5 file.

    Computation fields only. Other datasets stay in the source file
    and are copied by the driver.
    """
    filename: str
    rho_xyz:    np.ndarray         # complex [3, n_outputs, N, N]
    time_fs:    np.ndarray         # float   [n_outputs]
    dipoles:    np.ndarray         # float   [N_pigments, 3]
    strengths:  np.ndarray         # float   [N_pigments]
    N:          int
    n_outputs:  int
    dt_fs:      float
    n_steps:    int
    output_interval: int
    spectrum_type: str
    basis: str
    temperature_K: float
    w0_cm: float
    eta_cm: float
    gamma_cm: float
    eigenvalues_absolute_cm: np.ndarray    # [N], ascending. The ground state is not necessarily index 0.
    eigenvectors: np.ndarray               # complex [N, N], columns are eigenvectors

    @property
    def n_pigments(self) -> int:
        return self.dipoles.shape[0]

    @property
    def eigenvalues_relative_cm(self) -> np.ndarray:
        """absolute - w0_cm. Rotating-frame detuning; the ground state is near 0."""
        return self.eigenvalues_absolute_cm - self.w0_cm

    @property
    def ground_eigen_index(self) -> int:
        """Sorted-eigenbasis index of Hilbert |g⟩ (row 0)."""
        return ground_eigen_index(self.eigenvectors)

    @property
    def exciton_eigen_indices(self) -> list[int]:
        """All sorted-eigenbasis indices except the ground."""
        g = self.ground_eigen_index
        return [k for k in range(self.N) if k != g]


def load_spectrum_h5(path: str) -> SpectrumData:
    """Load a ``heom_solver absorption`` HDF5 (new group layout).

    Stage 1 only supports ``basis == "with_ground"`` — other bases raise
    ``RuntimeError`` so the |g⟩=index-0 assumption downstream never silently
    breaks. Schema lives in ``src/output/hdf5_metadata.cpp``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"spectrum file not found: {path}")

    with h5py.File(path, "r") as f:
        missing = [p for p in _SPECTRUM_H5_REQUIRED_PATHS if p not in f]
        if missing:
            raise ValueError(
                f"spectrum HDF5 missing required paths {missing}: {path!r}"
            )

        # /results/rho_xyz is already complex128 (compound {r,i}); h5py
        # exposes it as a single complex ndarray, no real/imag merging.
        rho_xyz  = np.asarray(f["/results/rho_xyz"][...],        dtype=np.complex128)
        time_fs  = np.asarray(f["/results/time"][...],           dtype=np.float64)
        dipoles  = np.asarray(f["/parameters/dipoles"][...],     dtype=np.float64)
        strengths = np.asarray(f["/parameters/strengths"][...],  dtype=np.float64)
        eigenvectors = np.asarray(f["/derived/eigenvectors"][...], dtype=np.complex128)

        # Absolute eigenvalues are derived (relative + w0_cm); keep the
        # ``_absolute_cm`` field name for backward compatibility but compute
        # the value here so consumers continue to work.
        eig_abs = heom_h5.eigenvalues_absolute_cm(f)

        spectrum_type = heom_h5.param(f, "spectrum_type", "absorption")
        basis         = heom_h5.param(f, "basis",         "unknown")
        n_steps         = int(heom_h5.param(f, "n_steps",         time_fs.shape[0] - 1))
        output_interval = int(heom_h5.param(f, "output_interval", 1))
        temperature_K   = float(heom_h5.param(f, "temperature",   0.0))
        w0_cm           = float(heom_h5.param(f, "w0_cm",         0.0))
        eta_cm          = float(heom_h5.param(f, "eta_cm",        0.0))
        gamma_cm        = float(heom_h5.param(f, "gamma_cm",      0.0))
        dt_attr         = heom_h5.param(f, "dt", None)

    if basis != "with_ground":
        raise RuntimeError(
            f"spectrum tools require basis='with_ground'; got "
            f"basis='{basis}' in {path}"
        )

    dipoles = normalize_dipoles(dipoles)

    n_pigments = dipoles.shape[0]
    N = n_pigments + 1                    # with_ground: N = n_states + 1
    n_outputs  = time_fs.shape[0]
    if rho_xyz.shape != (3, n_outputs, N, N):
        raise ValueError(
            f"rho_xyz shape mismatch: got {rho_xyz.shape}, "
            f"expected (3, {n_outputs}, {N}, {N})"
        )
    if eigenvectors.shape != (N, N):
        raise ValueError(
            f"eigenvectors shape mismatch: got {eigenvectors.shape}, "
            f"expected ({N}, {N})"
        )

    return SpectrumData(
        filename = path,
        rho_xyz  = rho_xyz,
        time_fs  = time_fs,
        dipoles  = dipoles,
        strengths = strengths,
        N        = N,
        n_outputs = n_outputs,
        dt_fs    = _sample_dt_fs(time_fs, dt_attr),
        n_steps  = n_steps,
        output_interval = output_interval,
        spectrum_type = spectrum_type,
        basis    = basis,
        temperature_K = temperature_K,
        w0_cm    = w0_cm,
        eta_cm   = eta_cm,
        gamma_cm = gamma_cm,
        eigenvalues_absolute_cm = eig_abs,
        eigenvectors = eigenvectors,
    )


def _sample_dt_fs(time_fs: np.ndarray, dt_attr) -> float:
    """Sampling interval of the saved correlation data.

    Use the median of ``diff(time_fs)`` so the FFT sees the spacing of
    **stored** samples — when ``output_interval > 1`` the HDF5 ``dt``
    attribute is the internal propagation step, not the spacing of the
    ``time`` dataset. Falls back to the parameter attribute when only one
    time point exists (degenerate but cheap to support).
    """
    if time_fs.size > 1:
        return float(np.median(np.diff(time_fs)))
    return float(dt_attr) if dt_attr is not None else 0.0


def normalize_dipoles(dipoles: np.ndarray,
                      tolerance: float = 1e-6,
                      label: str = "dipoles") -> np.ndarray:
    """Normalize dipole direction vectors, matching the C++ YAML parser.

    The direction belongs in `dipoles`; the magnitude belongs in `strengths`.
    If a legacy / hand-written HDF5 file stores non-unit directions, normalize
    them here and warn instead of silently scaling oscillator strengths twice.
    """
    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(f"{label} must have shape [n_pigments, 3], got {dipoles.shape}")

    out = np.array(dipoles, dtype=np.float64, copy=True)
    for i in range(out.shape[0]):
        norm = float(np.linalg.norm(out[i]))
        if norm < 1e-12:
            raise ValueError(f"{label}[{i}] has zero length")
        if abs(norm - 1.0) >= tolerance:
            warnings.warn(
                f"{label}[{i}] is not unit length (||d|| = {norm:.6e}); "
                "auto-normalising. Put magnitude into 'strengths'.",
                UserWarning,
                stacklevel=2,
            )
            out[i] /= norm
    return out


# ---------------------------------------------------------------------------
# Rebuild dipole operators (site basis, with_ground)
# ---------------------------------------------------------------------------

def build_mu_minus_site(dipoles: np.ndarray,
                        strengths: np.ndarray,
                        N: int) -> np.ndarray:
    """Site-basis lowering operators mu-_alpha, three Cartesian directions.

    Same layout as DipoleOperator::from_pigment_dipoles:
        mu+[alpha][i+1, 0] = strengths[i] * dipoles[i, alpha]
        mu- = adjoint(mu+)
    so mu- is nonzero only in the (0, i+1) block.

    Returns complex [3, N, N].
    """
    if dipoles.shape[0] != strengths.shape[0]:
        raise ValueError("dipoles 与 strengths 长度必须一致")
    n_pig = dipoles.shape[0]
    if n_pig + 1 != N:
        raise ValueError(
            f"with_ground basis 要求 N = n_pigments + 1, "
            f"got N={N}, n_pigments={n_pig}"
        )

    # mu-[alpha, 0, i+1] = strengths[i] * dipoles[i, alpha]
    mu_minus = np.zeros((3, N, N), dtype=np.complex128)
    mu_minus[:, 0, 1:] = (strengths[:, None] * dipoles).T   # [3, n_pig]
    return mu_minus


# ---------------------------------------------------------------------------
# Correlation C_alpha(t) = Tr(mu-_alpha * rho_alpha(t))
# ---------------------------------------------------------------------------

def compute_correlation(rho_xyz: np.ndarray,
                        mu_minus: np.ndarray) -> np.ndarray:
    """C_alpha(t) = Tr(mu-_alpha * rho_alpha(t)) for three directions.

    rho_xyz  complex [3, n_outputs, N, N]
    mu_minus complex [3, N, N]
    -> Ct    complex [3, n_outputs]

    One einsum for the trace.
    """
    if rho_xyz.shape[0] != 3 or mu_minus.shape[0] != 3:
        raise ValueError("rho_xyz / mu_minus 第 0 维必须是 3 (x,y,z)")
    # Tr(A B) = Σ_{ij} A_{ij} B_{ji}
    # → Σ_{ij} mu_minus[α,i,j] * rho_xyz[α,t,j,i]
    return np.einsum("aij,atji->at", mu_minus, rho_xyz)


# ---------------------------------------------------------------------------
# FFT to the frequency-domain absorption spectrum.
# ---------------------------------------------------------------------------

def _fft_grid_length(n_time: int, dt_fs: float, dw_cm: float) -> int:
    if dt_fs <= 0:
        raise ValueError(f"FFT requires dt_fs > 0, got {dt_fs}")
    if dw_cm <= 0:
        raise ValueError(f"FFT requires dw_cm > 0, got {dw_cm}")
    nfft = int(1.0 / (dw_cm * dt_fs * FFT_GRID_SCALE_LAS))
    if (nfft % 2) == 0:
        nfft += 1
    return max(nfft, n_time)


def fft_to_spectra_batch(ct_batch: np.ndarray,
                         dt_fs: float,
                         w0_cm: float = 0.0,
                         dw_cm: float = DW_DEFAULT_CM) -> tuple[np.ndarray, np.ndarray]:
    """C(t) batch → I(ω).

    ``ct_batch``: complex ``[n_time]`` or ``[n_series, n_time]``.
    Returns ``omega_cm`` ``[N']`` and complex spectra ``[n_series, N']``.
    """
    ct = np.asarray(ct_batch, dtype=np.complex128)
    if ct.ndim == 1:
        ct = ct[np.newaxis, :]
    elif ct.ndim != 2:
        raise ValueError(
            f"fft_to_spectra_batch expects 1D or 2D input, got shape {ct.shape}"
        )
    if ct.shape[1] == 0:
        raise ValueError("fft_to_spectra_batch: empty time series")

    nfft = _fft_grid_length(ct.shape[1], dt_fs, dw_cm)
    dt_au = dt_fs / AU2FS

    # ifft, not fft: the absorption convention is the e^{+i*w*t} transform.
    # fft would mirror the spectrum onto negative frequencies.
    spec = np.fft.fftshift(np.fft.ifft(ct, nfft, axis=-1), axes=-1)
    omega = (
        np.fft.fftshift(np.fft.fftfreq(nfft, d=dt_au)) * 2.0 * np.pi * AU2PERCM
        + w0_cm
    )
    return omega, spec


def fft_to_spectrum(ct: np.ndarray,
                    dt_fs: float,
                    w0_cm: float = 0.0,
                    dw_cm: float = DW_DEFAULT_CM) -> tuple[np.ndarray, np.ndarray]:
    """C(t) to I(omega). ``ct`` is a complex 1D array."""
    omega, specs = fft_to_spectra_batch(ct, dt_fs=dt_fs, w0_cm=w0_cm, dw_cm=dw_cm)
    return omega, specs[0]


def cut_to_window(omega_cm: np.ndarray, spec: np.ndarray,
                  wmin_cm: float, wmax_cm: float) -> tuple[np.ndarray, np.ndarray]:
    """Cut the spectrum to [wmin, wmax], expanding the edge by one point."""
    omega_cut, specs_cut = cut_to_window_batch(omega_cm, spec, wmin_cm, wmax_cm)
    return omega_cut, specs_cut


def cut_to_window_batch(
    omega_cm: np.ndarray,
    specs: np.ndarray,
    wmin_cm: float,
    wmax_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch ``cut_to_window``: ``specs`` is 1D ``[n_omega]`` or 2D ``[n_series, n_omega]``."""
    if wmin_cm >= wmax_cm:
        raise ValueError(f"wmin_cm ({wmin_cm}) 必须 < wmax_cm ({wmax_cm})")
    i1 = max(int(np.argmin(np.abs(omega_cm - wmin_cm))) - 1, 0)
    i2 = min(int(np.argmin(np.abs(omega_cm - wmax_cm))) + 2, len(omega_cm))
    omega_cut = omega_cm[i1:i2]
    arr = np.asarray(specs)
    if arr.ndim == 1:
        return omega_cut, arr[i1:i2]
    if arr.ndim != 2:
        raise ValueError(f"cut_to_window_batch expects 1D or 2D specs, got shape {arr.shape}")
    return omega_cut, arr[:, i1:i2]


def ground_eigen_index(eigenvectors: np.ndarray, *, tol: float = 0.99) -> int:
    """Return sorted-eigenbasis index of Hilbert |g⟩ (with_ground row 0).

    Eigen's ascending sort can place negative-detuning excitons before the
    ground (rotating-frame energy 0). Hilbert |g⟩ is always index 0.
    """
    v = np.asarray(eigenvectors)
    if v.ndim != 2 or v.shape[0] < 1:
        raise ValueError(f"eigenvectors must be 2D, got shape {v.shape}")
    overlaps = np.abs(v[0, :])
    g = int(np.argmax(overlaps))
    if overlaps[g] < tol:
        raise RuntimeError(
            f"cannot identify ground eigenstate: max |⟨g|ψ_k⟩|² = {overlaps[g]:.4f} "
            f"< {tol}; with_ground Hilbert |g⟩ is index 0"
        )
    return g


def default_absorption_frequency_window_cm(
    eigenvalues_absolute_cm: np.ndarray,
    *,
    eigenvectors: np.ndarray | None = None,
    spread_scale: float = 1.5,
    single_exciton_total_width_cm: float = 500.0,
) -> tuple[float, float]:
    """Default absolute window (cm^-1) when --wmin / --wmax are omitted.

    Uses excited eigenvalues only, skipping the ground-state index.

    Two or more excitons: spread = max - min, effective = spread_scale * spread,
    return median ± 0.75*effective.

    One exciton: total width ``single_exciton_total_width_cm`` (default 500),
    centered on that eigenvalue.

    A numerical spread of 0 uses half of that single-exciton width (±250).
    """
    e = np.asarray(eigenvalues_absolute_cm, dtype=np.float64).ravel()
    if e.size < 2:
        raise ValueError("eigenvalues_absolute_cm must include ground + at least one excitation")
    if eigenvectors is not None:
        g = ground_eigen_index(eigenvectors)
        exc = np.delete(e, g)
    else:
        exc = e[1:]
    if exc.size == 1:
        half = 0.5 * float(single_exciton_total_width_cm)
        c = float(exc[0])
        return c - half, c + half

    med = float(np.median(exc))
    spread = float(np.max(exc) - np.min(exc))
    effective = float(spread_scale) * spread
    half_span = 0.75 * effective
    if half_span < 1e-9:
        half_span = 0.5 * float(single_exciton_total_width_cm)
    return med - half_span, med + half_span


# ---------------------------------------------------------------------------
# Peak picking. Pure NumPy.
# ---------------------------------------------------------------------------

def find_peaks_safe(x: np.ndarray, y: np.ndarray,
                    height_frac: float = 0.05,
                    distance_cm: float = 30.0) -> list[tuple[float, float]]:
    """Return [(x_peak_cm, y_peak), ...].

    Strict local maxima ``y[i-1] < y[i] > y[i+1]``, then an absolute height
    cut ``height_frac * max(y)``, then one peak per ``distance_cm`` neighborhood.
    An empty list means no peak. ``height_frac`` is an absolute height cut,
    not a prominence.
    """
    n = len(y)
    if n < 3:
        return []
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    distance = max(int(round(distance_cm / max(abs(dx), 1e-12))), 1)
    threshold = max(float(np.max(y)) * height_frac, 1e-12)

    # Strict local maxima, then the height cut.
    mid = y[1:-1]
    mask = (mid > y[:-2]) & (mid > y[2:]) & (mid >= threshold)
    cand_idx = np.flatnonzero(mask) + 1
    if cand_idx.size == 0:
        return []

    # Greedy by amplitude: drop the smaller peak inside ``distance``.
    order = np.argsort(-y[cand_idx])
    chosen_idx: list[int] = []
    for i in cand_idx[order].tolist():
        if all(abs(i - j) >= distance for j in chosen_idx):
            chosen_idx.append(i)
    chosen_idx.sort()
    return [(float(x[i]), float(y[i])) for i in chosen_idx]


# ---------------------------------------------------------------------------
# Overlay an experimental spectrum.
# ---------------------------------------------------------------------------

# Token-based basename gate: ``exp`` / ``experiment`` / ``experimental`` must be
# an independent token (``. _ -`` or string boundary), not a substring.
EXPERIMENT_TOKEN_RE = re.compile(
    r"(?:^|[._-])(?:exp|experiment|experimental)(?:[._-]|$)",
    re.IGNORECASE,
)


def discover_experiment_spectra_auto(data_dir: str) -> list["ExperimentSpectrum"]:
    """Load experiment spectra from ``data_dir`` by token-matched basenames.

    Non-loadable paths are skipped with a warning. Sorted by basename.
    """
    if not os.path.isdir(data_dir):
        return []
    names = sorted(fn for fn in os.listdir(data_dir) if not fn.startswith("."))
    paths: list[str] = []
    for fn in names:
        if not EXPERIMENT_TOKEN_RE.search(fn):
            continue
        full = os.path.join(data_dir, fn)
        if os.path.isfile(full):
            paths.append(full)
    loaded: list[ExperimentSpectrum] = []
    for pth in paths:
        try:
            loaded.append(ExperimentSpectrum.load(pth))
        except (OSError, ValueError, FileNotFoundError) as e:
            warnings.warn(f"skipped experimental candidate {pth!r}: {e}")
    return loaded


@dataclass
class ExperimentSpectrum:
    """Experimental absorption: two columns, cm^-1 and intensity."""
    omega_cm: np.ndarray
    intensity: np.ndarray
    label: str = "exp"

    @classmethod
    def load(cls, path: str, label: Optional[str] = None) -> "ExperimentSpectrum":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"experimental spectrum file not found: {path}")
        # CSV or whitespace-separated.
        try:
            data = np.loadtxt(path, delimiter=",")
        except ValueError:
            data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{path}: expected 2-column data (omega_cm, intensity)")
        intens = np.asarray(data[:, 1], dtype=np.float64)
        denom = float(np.max(np.abs(intens)))
        if denom < 1e-30:
            raise ValueError(f"{path}: intensity column is all zero (cannot normalize)")
        return cls(
            omega_cm  = data[:, 0],
            intensity = intens / denom,
            label     = label or os.path.basename(path),
        )


# ---------------------------------------------------------------------------
# Stick spectrum from hamiltonian_cm, dipoles, and strengths.
# ---------------------------------------------------------------------------

@dataclass
class StickEntry:
    """One stick: eigenvalue, isotropic oscillator strength |mu|^2/3, dominant site.

    ``dominant_site_index`` is the 0-based pigment with the largest |V|^2
    on eigenvector row j+1. For an exciton in the with_ground basis that
    index is always valid.
    """
    omega_cm: float
    oscillator_strength: float
    dominant_site_index: int
    dominant_site_fraction: float


def build_stick_spectrum(eigenvalues_cm: np.ndarray,
                         eigenvectors: np.ndarray,
                         dipoles: np.ndarray,
                         strengths: np.ndarray) -> list[StickEntry]:
    """Stick spectrum of each exciton in the with_ground basis.

        |mu_eg^(k)|^2 = (1/3) sum_alpha |sum_j strengths[j] * dipoles[j, alpha] * V[j+1, k]|^2

    Dominant site = argmax_j |V[j+1, k]|^2.
    """
    N = eigenvectors.shape[0]
    n_pig = dipoles.shape[0]
    if N != n_pig + 1:
        raise ValueError(f"stick: N={N} 与 n_pigments+1={n_pig+1} 不一致")

    g = ground_eigen_index(eigenvectors)
    exciton_idxs = [k for k in range(N) if k != g]
    V_pig = eigenvectors[1:, exciton_idxs]                # [n_pig, n_exc]
    mu_alpha = np.einsum("j,ja,jk->ak", strengths, dipoles, V_pig)  # [3, n_exc]
    osc_all = (np.abs(mu_alpha) ** 2).sum(axis=0) / 3.0   # [n_exc]

    site_amp2 = np.abs(V_pig) ** 2                        # [n_pig, n_exc]
    site_amp2_sum = site_amp2.sum(axis=0)                 # [n_exc]
    if not np.all(site_amp2_sum > 0):
        raise RuntimeError(
            "degenerate excitonic eigenvector: ||V[1:, k]||² = 0 for some k; "
            "with_ground basis guarantees this is non-zero, source HDF5 is corrupt"
        )
    j_star = np.argmax(site_amp2, axis=0)                 # [n_exc]
    frac = site_amp2[j_star, np.arange(j_star.size)] / site_amp2_sum

    eig_exc = np.asarray(eigenvalues_cm[exciton_idxs], dtype=np.float64)
    return [
        StickEntry(
            omega_cm = float(eig_exc[k]),
            oscillator_strength = float(osc_all[k]),
            dominant_site_index = int(j_star[k]),
            dominant_site_fraction = float(frac[k]),
        )
        for k in range(eig_exc.size)
    ]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_two_column(path: str, x: np.ndarray, y: np.ndarray, header: str = "") -> None:
    """Save two comma-separated columns. Legacy helper; drivers prefer HDF5."""
    np.savetxt(
        path,
        np.column_stack([x, y]),
        delimiter=",",
        header=header,
        comments="# ",
    )


def basename_noext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
