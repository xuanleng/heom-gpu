"""KMS / Kennard–Stepanov check: I_F(ω) = (Z_g/Z_e) e^{−βω} I_A(ω).

On the overlap where both spectra exceed ``floor`` of their own
peak, fit ln(I_F/I_A) linearly in ω; T_eff = −1/(k_B·slope). Weights are
1/var(ln I_F − ln I_A) for errors proportional to each spectrum's peak:
var ∝ 1/a² + 1/f² with a, f the peak-normalised intensities. The spectra must
weight C(t_0) by one half (``spectrum.fft_to_spectra_batch`` does): a full
weight adds a flat baseline that sits right at ``floor`` and drags T_eff by
several percent.
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from heom_tools.common.config_hash import physics_hash

KB_CM = 0.695034800      # cm⁻¹ / K
KMS_FLOOR = 1e-2


@dataclass
class KMSFit:
    T_K: float
    T_eff_K: float
    slope_per_cm: float
    intercept: float
    floor: float
    omega_cm: np.ndarray        # overlap grid used in the fit
    log_ratio: np.ndarray       # ln(I_F/I_A) there
    weights: np.ndarray
    fluorescence_norm: np.ndarray  # peak-normalised fitted I_F, whole fluorescence grid
    absorption_norm: np.ndarray    # peak-normalised fitted I_A, interpolated onto it

    @property
    def rel_error(self) -> float:
        return abs(self.T_eff_K - self.T_K) / self.T_K

    def write(self, g: h5py.Group) -> None:
        for k in ("T_K", "T_eff_K", "slope_per_cm", "intercept", "floor", "rel_error"):
            g.attrs[k] = getattr(self, k)
        g.create_dataset("omega_cm", data=self.omega_cm)
        g.create_dataset("log_ratio", data=self.log_ratio)
        g.create_dataset("weights", data=self.weights)
        g.create_dataset("fluorescence_norm", data=self.fluorescence_norm)
        g.create_dataset("absorption_norm", data=self.absorption_norm)

    @classmethod
    def read(cls, g: h5py.Group) -> "KMSFit":
        return cls(**{k: float(g.attrs[k])
                      for k in ("T_K", "T_eff_K", "slope_per_cm", "intercept", "floor")},
                   **{k: g[k][...] for k in ("omega_cm", "log_ratio", "weights",
                                             "fluorescence_norm", "absorption_norm")})


def fit_kms(omega_cm, I_F, omega_abs_cm, I_A, T_K: float,
            floor: float = KMS_FLOOR) -> KMSFit:
    """Fit T_eff on the fluorescence grid; I_A is interpolated onto it."""
    w = np.asarray(omega_cm, dtype=np.float64)
    f = np.asarray(I_F, dtype=np.float64) / np.max(I_F)
    wa = np.asarray(omega_abs_cm, dtype=np.float64)
    if np.any(np.diff(w) <= 0) or np.any(np.diff(wa) <= 0):
        raise ValueError("KMS fit needs ascending frequency grids")
    a = np.interp(w, wa, np.asarray(I_A, dtype=np.float64) / np.max(I_A),
                  left=0.0, right=0.0)
    m = (a > floor) & (f > floor)
    if np.count_nonzero(m) < 3:
        raise ValueError(f"absorption and fluorescence overlap above {floor:g} of their "
                         f"peaks in {np.count_nonzero(m)} points; widen the window "
                         "(--wmin/--wmax) or check that the two files belong together")
    x, y = w[m], np.log(f[m] / a[m])
    wt = 1.0 / (1.0 / a[m] ** 2 + 1.0 / f[m] ** 2)
    slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(wt))
    if slope >= 0.0:
        raise ValueError(f"ln(I_F/I_A) slope = {slope:.3e} ≥ 0: emission is not red of "
                         "absorption on the overlap — not a thermalised pair")
    return KMSFit(T_K=float(T_K), T_eff_K=float(-1.0 / (KB_CM * slope)),
                  slope_per_cm=float(slope), intercept=float(intercept), floor=floor,
                  omega_cm=x, log_ratio=y, weights=wt, fluorescence_norm=f,
                  absorption_norm=a)


def _config_yaml(path: str) -> str:
    with h5py.File(path, "r") as f:
        if "/parameters/config_yaml" not in f:
            raise ValueError(f"{path!r} has no /parameters/config_yaml")
        v = f["/parameters/config_yaml"][()]
    return v.decode() if isinstance(v, bytes) else str(v)


def require_same_physics(fl_path: str, abs_path: str) -> str:
    """Task-less physics_hash (system + bath + filter) must agree; returns it."""
    h_fl = physics_hash(_config_yaml(fl_path), None)
    h_abs = physics_hash(_config_yaml(abs_path), None)
    if h_fl != h_abs:
        raise ValueError(f"system/bath/filter differ: {fl_path!r} hash {h_fl} vs "
                         f"{abs_path!r} hash {h_abs}; KMS compares one physical system")
    return h_fl
