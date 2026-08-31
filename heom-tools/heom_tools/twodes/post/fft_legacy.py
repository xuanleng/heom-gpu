"""Legacy 2D FFT (refs fft_2d_data mode 3) — not for production."""
from __future__ import annotations

import numpy as np

from .fft import _pad_slab


def two_d_fft_legacy_mode3(S, signal, n_tau, n_t):
    """fft2 on [t, tau], then flips.

    Legacy reference sign convention — not equal to axis (do not use for production).
    """
    pad_tt = _pad_slab(S, n_tau, n_t).T  # [t, τ] as in fft_2d_data.py
    spec = np.fft.fftshift(np.fft.fft2(pad_tt))
    if signal == "rephasing":
        spec = np.flipud(spec)
    elif signal == "nonrephasing":
        spec = np.flipud(np.fliplr(spec))
    else:
        raise ValueError(f"unknown signal {signal}")
    return spec.T  # [ω_τ, ω_t]


two_d_fft_fft2 = two_d_fft_legacy_mode3  # deprecated name
