"""Production 2D FFT for 2DES post-processing."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .constants import FS_TO_CM

FFT_METHODS = ("axis", "legacy_mode3", "check")
# Config / CLI production surface — legacy_mode3 stays importable for tests only.
CONFIG_FFT_METHODS = ("axis", "check")
FFT_CHECK_RTOL = 1e-9
_FFT_DEPRECATED_ALIASES = {"fft2": "legacy_mode3"}


def dw_phys_cm(n_data: int, dt_fs: float) -> float:
    """Independent Fourier spacing from the *unpadded* record length (cm^-1).

    Zero-padding densifies the FFT output but does not improve this value.
    """
    if n_data < 1:
        raise ValueError(f"n_data must be >= 1, got {n_data}")
    if dt_fs <= 0.0:
        raise ValueError(f"time step must be positive, got {dt_fs}")
    return float(FS_TO_CM) / (float(n_data) * float(dt_fs))


def nfft_len(n, dt_fs, dw_cm):
    """Zero-pad length so native FFT bin spacing ≈ ``dw_cm``.

    Uses ``N ≈ FS_TO_CM / (dw · dt)`` (odd, ≥ n). The published frequency
    axis is *not* this native spacing — see ``freq_axis_dw_cm``. ``dw`` only
    sizes the pad so projection onto ``w0 + k·dw`` is well sampled; physical
    resolution remains ``dw_phys_cm(n, dt)``.
    """
    if dw_cm <= 0.0:
        raise ValueError(f"dw must be positive, got {dw_cm}")
    if dt_fs <= 0.0:
        raise ValueError(f"time step must be positive, got {dt_fs}")
    nf = int(FS_TO_CM / (dw_cm * dt_fs))
    if nf % 2 == 0:
        nf += 1
    return max(nf, n)


def fft_time_weight(dtau: float, dt: float) -> float:
    """Discrete → continuous Fourier measure: ΣΣ S_ij e^{…} · Δτ · Δt.

    Without this factor, absolute spectral amplitudes scale with the time-grid
    spacings and cannot be compared across different (dtau, dt_detect). Peak-
    normalized lineshape gates are unaffected. Defaults of 1.0 keep unit tests
    bit-identical to the pre-weight API.
    """
    if dtau <= 0.0 or dt <= 0.0:
        raise ValueError(f"dtau and dt must be positive, got dtau={dtau}, dt={dt}")
    return float(dtau) * float(dt)


def _pad_slab(S: np.ndarray, n_tau: int, n_t: int) -> np.ndarray:
    """Zero-pad slab[τ, t] to (n_tau, n_t)."""
    out = np.zeros((n_tau, n_t), dtype=complex)
    n0 = min(n_tau, S.shape[0])
    n1 = min(n_t, S.shape[1])
    out[:n0, :n1] = S[:n0, :n1]
    return out


def fft_axis_plus(x, axis, n):
    """Time → frequency along one axis: ifft (numpy divides by transform length n)."""
    return np.fft.fftshift(np.fft.ifft(x, n=n, axis=axis), axes=axis)


def fft_axis_minus(x, axis, n):
    """τ-axis rephasing sign: fft (no 1/n factor) instead of ifft on axis 0 only."""
    return np.fft.fftshift(np.fft.fft(x, n=n, axis=axis), axes=axis)


def two_d_fft_separable(S, signal, n_tau, n_t, *, dtau: float = 1.0, dt: float = 1.0):
    """Same physics as two_d_fft_axis via 2D ifft2 (nr) or fft(τ)+ifft(t) (rp)."""
    pad = _pad_slab(S, n_tau, n_t)
    w = fft_time_weight(dtau, dt)
    if signal == "nonrephasing":
        return np.fft.fftshift(np.fft.ifft2(pad)) * n_tau * w
    if signal == "rephasing":
        spec = np.fft.fft(pad, n=n_tau, axis=0)
        spec = np.fft.ifft(spec, n=n_t, axis=1)
        return np.fft.fftshift(spec, axes=(0, 1)) * w
    raise ValueError(f"unknown signal {signal}")


def two_d_fft_axis(S, signal, n_tau, n_t, *, dtau: float = 1.0, dt: float = 1.0):
    """Sequential τ/t FFT: nr via two_d_fft_separable; rp=fft(τ)+ifft(t)."""
    if signal == "nonrephasing":
        return two_d_fft_separable(S, signal, n_tau, n_t, dtau=dtau, dt=dt)
    if signal == "rephasing":
        return (fft_axis_plus(fft_axis_minus(S, 0, n_tau), 1, n_t)
                * fft_time_weight(dtau, dt))
    raise ValueError(f"unknown signal {signal}")


def _fft_max_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    den = float(max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-30))
    return float(np.max(np.abs(a - b)) / den)


def cross_check_two_d_fft(
    S, signal, n_tau, n_t, *, dtau: float = 1.0, dt: float = 1.0, label: str = ""
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Verify axis≡separable; report axis vs mode-3 fft2. Returns axis spectrum."""
    from .fft_legacy import two_d_fft_legacy_mode3

    axis = two_d_fft_axis(S, signal, n_tau, n_t, dtau=dtau, dt=dt)
    sep = two_d_fft_separable(S, signal, n_tau, n_t, dtau=dtau, dt=dt)
    # Legacy stays unweighted internally; apply the same measure for the check.
    m3 = two_d_fft_legacy_mode3(S, signal, n_tau, n_t) * fft_time_weight(dtau, dt)
    err_sep = _fft_max_rel_err(axis, sep)
    if err_sep > FFT_CHECK_RTOL:
        raise ValueError(
            f"FFT axis vs separable mismatch {label} {signal}: rel_err={err_sep:.3e}"
        )
    stats = {
        "axis_vs_separable": err_sep,
        "axis_vs_legacy_mode3": _fft_max_rel_err(axis, m3),
    }
    return axis, stats


def two_d_fft(S, signal, n_tau, n_t, method: str = "axis", *,
              dtau: float = 1.0, dt: float = 1.0):
    """Dispatch 2D FFT. method: axis | legacy_mode3 (not for production).

    Multiplies by ``dtau * dt`` (fs²) so absolute amplitudes are comparable
    across time grids. Pass dtau=dt=1.0 (default) for the historical unweighted
    scale used by unit tests.
    """
    if method == "axis":
        return two_d_fft_axis(S, signal, n_tau, n_t, dtau=dtau, dt=dt)
    if method == "legacy_mode3":
        from .fft_legacy import two_d_fft_legacy_mode3

        return two_d_fft_legacy_mode3(S, signal, n_tau, n_t) * fft_time_weight(dtau, dt)
    raise ValueError(f"fft_method must be one of {FFT_METHODS}, got {method!r}")


def freq_axis_cm(n, d_fs, w0_cm):
    """Native FFT bin centres: ``w0 + fftfreq(n, d)·FS_TO_CM``.

    Spacing is ``FS_TO_CM / (n·d_fs)``, which only *approximates* a requested
    output ``dw`` after ``nfft_len``. Prefer ``freq_axis_dw_cm`` for the
    stored / compared axis. Independent resolve is still ``dw_phys_cm`` of
    the unpadded length.
    """
    return np.fft.fftshift(np.fft.fftfreq(n, d=d_fs)) * FS_TO_CM + w0_cm


def freq_axis_dw_cm(dw_cm: float, w0_cm: float, n_fft: int, d_fs: float) -> np.ndarray:
    """Exact published lattice ``ω = w0 + k·dw`` covering native FFT support.

    ``dw`` is the output sample step (for HDF5 / plots / cross-dt compare),
    not the physical Fourier resolution. Same ``(dw, w0)`` always yields the
    same sample set for every ``k`` inside the native Nyquist window of
    ``(n_fft, d_fs)`` — shared ω pixels where supports overlap, no figure-
    side interpolation.
    """
    if dw_cm <= 0.0:
        raise ValueError(f"dw must be positive, got {dw_cm}")
    if d_fs <= 0.0:
        raise ValueError(f"time step must be positive, got {d_fs}")
    if n_fft < 2:
        raise ValueError(f"n_fft must be >= 2, got {n_fft}")
    native = freq_axis_cm(n_fft, d_fs, w0_cm)
    lo = float(native[0])
    hi = float(native[-1])
    k_min = int(np.ceil((lo - w0_cm) / dw_cm - 1e-12))
    k_max = int(np.floor((hi - w0_cm) / dw_cm + 1e-12))
    if k_max < k_min:
        raise ValueError(
            f"dw={dw_cm} cm^-1 lattice empty inside native support "
            f"[{lo:.6g}, {hi:.6g}] cm^-1 (w0={w0_cm})"
        )
    return w0_cm + np.arange(k_min, k_max + 1, dtype=np.float64) * dw_cm


def _axis_interp_weights(x_src: np.ndarray, x_dst: np.ndarray):
    """Linear-interp indices/weights; ``x_dst`` must lie in ``[x_src[0], x_src[-1]]``."""
    if x_src.ndim != 1 or x_dst.ndim != 1:
        raise ValueError("frequency axes must be 1-D")
    if len(x_src) < 2:
        raise ValueError("source frequency axis needs >= 2 points")
    if not np.all(np.diff(x_src) > 0.0):
        raise ValueError("source frequency axis must be strictly increasing")
    if len(x_dst) == 0:
        raise ValueError("destination frequency axis is empty")
    if float(x_dst[0]) < float(x_src[0]) - 1e-9 or float(x_dst[-1]) > float(x_src[-1]) + 1e-9:
        raise ValueError(
            f"destination ω outside native FFT support "
            f"[{x_src[0]:.6g}, {x_src[-1]:.6g}] ← "
            f"[{x_dst[0]:.6g}, {x_dst[-1]:.6g}]"
        )
    i1 = np.searchsorted(x_src, x_dst, side="right")
    i1 = np.clip(i1, 1, len(x_src) - 1)
    i0 = i1 - 1
    x0 = x_src[i0]
    x1 = x_src[i1]
    denom = x1 - x0
    w1 = np.divide(x_dst - x0, denom, out=np.zeros_like(x_dst), where=denom != 0.0)
    w0 = 1.0 - w1
    return i0, i1, w0, w1


def project_spec_2d(
    spec: np.ndarray,
    w_tau_src: np.ndarray,
    w_t_src: np.ndarray,
    w_tau_dst: np.ndarray,
    w_t_dst: np.ndarray,
) -> np.ndarray:
    """Bilinear project a complex (ωτ, ωt) map onto a destination ω lattice."""
    if spec.shape != (len(w_tau_src), len(w_t_src)):
        raise ValueError(
            f"spec shape {spec.shape} != ({len(w_tau_src)}, {len(w_t_src)})"
        )
    ix0, ix1, wx0, wx1 = _axis_interp_weights(w_tau_src, w_tau_dst)
    iy0, iy1, wy0, wy1 = _axis_interp_weights(w_t_src, w_t_dst)
    tmp = wx0[:, None] * spec[ix0, :] + wx1[:, None] * spec[ix1, :]
    return tmp[:, iy0] * wy0[None, :] + tmp[:, iy1] * wy1[None, :]
