"""Matplotlib normalization helpers (no matplotlib import at module load)."""
from __future__ import annotations

import numpy as np


def make_norm(data, mpl, *, nonnegative=False):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return mpl.colors.Normalize()
    vmin = float(finite.min())
    vmax = float(finite.max())
    if nonnegative:
        hi = vmax if vmax > 0.0 else 1.0
        return mpl.colors.Normalize(vmin=0.0, vmax=hi)
    if vmin == vmax:
        a = max(abs(vmin), 1e-30)
        return mpl.colors.Normalize(vmin=-a, vmax=a)
    a = max(abs(vmin), abs(vmax))
    return mpl.colors.Normalize(vmin=-a, vmax=a)
