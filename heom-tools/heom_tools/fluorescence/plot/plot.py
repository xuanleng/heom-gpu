"""Emission figure: I_F (+ excitons, sticks), optional I_A and ln(I_F/I_A) fit.

Takes plain arrays so ``flu-post`` (in-memory result) and ``flu-plot``
(processed ``*_spectrum.h5``) draw the same picture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np

from heom_tools.common.mpl_util import PNG_DPI, save_matplotlib_figure, show_matplotlib_figure
from heom_tools.fluorescence.post.kms import KMSFit


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


@dataclass
class EmissionView:
    label: str
    omega_cm: np.ndarray
    total_norm: np.ndarray
    per_exciton_norm: np.ndarray        # [n_omega, n_exc]
    stick_omega_cm: np.ndarray
    stick_weight: np.ndarray
    stick_weights: str                  # "heom" | "boltzmann_heff"
    kms: Optional[KMSFit] = None

    @classmethod
    def from_result(cls, result, label: str) -> "EmissionView":
        lin = result.linear
        return cls(label=label, omega_cm=lin.omega_cm, total_norm=lin.intensity_total_norm,
                   per_exciton_norm=lin.intensity_per_exciton_norm,
                   stick_omega_cm=np.array([s.omega_cm for s in lin.sticks]),
                   stick_weight=np.array([s.oscillator_strength for s in lin.sticks]),
                   stick_weights=result.populations.stick_weights, kms=result.kms)

    @classmethod
    def from_processed_h5(cls, path: str) -> "EmissionView":
        with h5py.File(path, "r") as f:
            st = _text(f.attrs.get("spectrum_type", ""))
            if st != "fluorescence":
                raise ValueError(f"{path!r}: spectrum_type={st!r}, need a flu-post output")
            table = f["sticks/table"][...]
            return cls(label=_text(f.attrs["source_h5"]).rsplit("/", 1)[-1],
                       omega_cm=f["omega_cm"][...],
                       total_norm=f["intensity_total_norm"][...],
                       per_exciton_norm=f["intensity_per_exciton_norm"][...],
                       stick_omega_cm=table[:, 0], stick_weight=table[:, 1],
                       stick_weights=_text(f["sticks"].attrs["stick_weights"]),
                       kms=KMSFit.read(f["kms"]) if "kms" in f else None)


def _draw(plt_mod, v: EmissionView):
    rows = 2 if v.kms is not None else 1
    fig, axes = plt_mod.subplots(rows, 1, figsize=(9.0, 4.2 * rows + 1.0), squeeze=False)
    ax = axes[0, 0]
    for k in range(v.per_exciton_norm.shape[1]):
        ax.plot(v.omega_cm, v.per_exciton_norm[:, k], lw=1.2, label=f"exc {k + 1}")
    ax.plot(v.omega_cm, v.total_norm, lw=2.5, color="C0", label="HEOM fluorescence")
    if v.kms is not None:
        ax.plot(v.omega_cm, v.kms.absorption_norm, "--", lw=2.0, color="k",
                label="absorption (KMS partner)")
    if v.stick_weight.size:
        w = v.stick_weight / max(float(v.stick_weight.max()), 1e-300)
        ax.bar(v.stick_omega_cm, w, width=max(np.ptp(v.omega_cm) * 0.005, 1.0),
               alpha=0.45, color="C2", label=f"stick p_k f_k ({v.stick_weights}, max=1)")
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel("Intensity (norm)")
    ax.set_title(f"steady-state fluorescence | {v.label}")
    ax.legend(fontsize=8, ncol=2 if v.per_exciton_norm.shape[1] > 3 else 1)
    ax.grid(alpha=0.3)
    if v.kms is not None:
        k = v.kms
        ax2 = axes[1, 0]
        ax2.plot(k.omega_cm, k.log_ratio, ".", ms=3, label=r"$\ln(I_F/I_A)$")
        ax2.plot(k.omega_cm, k.slope_per_cm * k.omega_cm + k.intercept, "-", color="C3",
                 label=(fr"fit: $T_{{\rm eff}}$ = {k.T_eff_K:.2f} K "
                        fr"(T = {k.T_K:g} K, rel. err {k.rel_error:.2e})"))
        ax2.set_xlabel(r"$\omega$ (cm$^{-1}$)")
        ax2.set_ylabel(r"$\ln(I_F/I_A)$")
        ax2.set_title(f"KMS on overlap > {k.floor:g} of both peaks")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def write_emission_plot(v: EmissionView, out_path: str) -> str:
    save_matplotlib_figure(lambda plt_mod: _draw(plt_mod, v), out_path, dpi=PNG_DPI)
    return out_path


def show_emission_plot(v: EmissionView) -> None:
    show_matplotlib_figure(lambda plt_mod: _draw(plt_mod, v))
