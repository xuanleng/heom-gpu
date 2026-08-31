"""Frequency-axis windowing for 2DES maps."""
from __future__ import annotations

import numpy as np

# Extra bins kept outside the requested [lo, hi] so contours/imshow edges
# are not clipped flush against the window. Asymmetry is intentional padding
# for the upper edge (imshow extent still uses the config window).
WINDOW_PAD_LO = 1
WINDOW_PAD_HI = 2


def window_slice(axis, lo, hi):
    n = len(axis)
    if n < 2:
        return slice(None)
    i0 = int(np.argmin(np.abs(axis - lo))) - WINDOW_PAD_LO
    i1 = int(np.argmin(np.abs(axis - hi))) + WINDOW_PAD_HI
    i0 = max(0, min(i0, n - 2))
    i1 = max(i0 + 2, min(i1, n))
    return slice(i0, i1)


def cut_window(x, y, data, window):
    if not window:
        return x, y, data
    lo, hi = float(window[0]), float(window[1])
    if lo >= hi:
        raise ValueError(f"window lo must be < hi, got {lo} {hi}")
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    in_x = int(np.sum((x_arr >= lo) & (x_arr <= hi)))
    in_y = int(np.sum((y_arr >= lo) & (y_arr <= hi)))
    if in_x < 2 or in_y < 2:
        raise ValueError(
            f"window [{lo:g}, {hi:g}] contains fewer than 2 bins on the "
            f"frequency axes ({in_x} on ω_τ, {in_y} on ω_t); widen the window "
            f"or lower dw",
        )
    sx, sy = window_slice(x_arr, lo, hi), window_slice(y_arr, lo, hi)
    return x_arr[sx], y_arr[sy], data[sx, sy]
