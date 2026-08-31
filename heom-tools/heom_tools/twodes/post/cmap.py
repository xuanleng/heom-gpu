"""2DES diverging colormap (blue → white → red, white at centre = 0)."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Embedded lookup table — do not import from legacy reference scripts.
_DIVERGING_RGB = [
    [0.4, 0.2, 1], [0.36, 0.2, 1], [0.32, 0.2, 1], [0.28, 0.2, 1],
    [0.24, 0.2, 1], [0.2, 0.2, 1], [0.16, 0.2, 1], [0.12, 0.2, 1],
    [0.08, 0.2, 1], [0.04, 0.2, 1], [0, 0.2, 1],
    [0.0232278, 0.241931, 0.994872], [0.0464555, 0.283861, 0.989744],
    [0.0696833, 0.325792, 0.984615], [0.092911, 0.367722, 0.979487],
    [0.116139, 0.409653, 0.974359], [0.139367, 0.451584, 0.969231],
    [0.162594, 0.493514, 0.964103], [0.185822, 0.535445, 0.958974],
    [0.20905, 0.577376, 0.953846], [0.232278, 0.619306, 0.948718],
    [0.255505, 0.661237, 0.94359], [0.278733, 0.703167, 0.938462],
    [0.301961, 0.745098, 0.933333], [0.379521, 0.773421, 0.940741],
    [0.457081, 0.801743, 0.948148], [0.534641, 0.830065, 0.955556],
    [0.6122, 0.858388, 0.962963], [0.68976, 0.88671, 0.97037],
    [0.76732, 0.915033, 0.977778], [0.84488, 0.943355, 0.985185],
    [0.92244, 0.971678, 0.992593], [1, 1, 1],
    [0.875, 1, 0.875], [0.75, 1, 0.75], [0.625, 1, 0.625], [0.5, 1, 0.5],
    [0.375, 1, 0.375], [0.25, 1, 0.25], [0.125, 1, 0.125], [0, 1, 0],
    [0.125, 0.980392, 0], [0.25, 0.960784, 0], [0.375, 0.941176, 0],
    [0.5, 0.921569, 0], [0.625, 0.901961, 0], [0.75, 0.882353, 0],
    [0.875, 0.862745, 0], [1, 0.843137, 0], [1, 0.766488, 0],
    [1, 0.68984, 0], [1, 0.613191, 0], [1, 0.536542, 0], [1, 0.459893, 0],
    [1, 0.383244, 0], [1, 0.306595, 0], [1, 0.229947, 0], [1, 0.153298, 0],
    [1, 0.0766488, 0], [1, 0, 0], [0.9, 0, 0.05], [0.8, 0, 0.1],
    [0.7, 0, 0.15], [0.6, 0, 0.2],
]


def diverging_cmap():
    return LinearSegmentedColormap.from_list("twodes_div", _DIVERGING_RGB)


def asymmetric_diverging_cmap(
    base: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    *,
    vcenter: float = 0.0,
    n: int = 256,
) -> LinearSegmentedColormap:
    """Resample *base* so its centre (white) sits at *vcenter* on a linear [vmin, vmax] bar."""
    lo, hi, z = float(vmin), float(vmax), float(vcenter)
    if lo >= hi:
        return base
    if z <= lo:
        colors = base(np.linspace(0.5, 1.0, n))
        return LinearSegmentedColormap.from_list("twodes_div_asym", colors)
    if z >= hi:
        colors = base(np.linspace(0.0, 0.5, n))
        return LinearSegmentedColormap.from_list("twodes_div_asym", colors)
    frac = (z - lo) / (hi - lo)
    t = np.linspace(0.0, 1.0, n)
    src = np.empty(n, dtype=np.float64)
    low = t <= frac
    if frac > 0.0:
        src[low] = 0.5 * t[low] / frac
    else:
        src[low] = 0.0
    high = ~low
    tail = 1.0 - frac
    if tail > 0.0:
        src[high] = 0.5 + 0.5 * (t[high] - frac) / tail
    else:
        src[high] = 1.0
    colors = base(src)
    return LinearSegmentedColormap.from_list("twodes_div_asym", colors)


def positive_cmap(base):
    colors = base(np.linspace(0.5, 1.0, 256))
    return LinearSegmentedColormap.from_list("twodes_pos", colors)
