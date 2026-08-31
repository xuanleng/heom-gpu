"""Parameter overlay panel for absorption spectrum PNG figures."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import h5py

from heom_tools.common import heom_h5
from .util import format_duration


@dataclass
class RunParametersPanel:
    """Bath and truncation scalars for the spectrum PNG parameter panel."""

    eta_cm: Optional[float] = None
    gamma_inverse_fs: Optional[float] = None
    static_disorder_fwhm_cm: Optional[float] = None
    temperature_K: Optional[float] = None
    n_extra_poles: Optional[int] = None
    L_cutoff: Optional[int] = None
    L_total: Optional[int] = None
    npa: Optional[int] = None
    tolerance: Optional[float] = None
    dt_fs: Optional[float] = None
    n_steps: Optional[int] = None
    computation_time_s: Optional[float] = None

    @staticmethod
    def _fmt_float(v: Optional[float], prec: int, suffix: str) -> str:
        if v is None:
            return "—"
        return f"{v:{prec}} {suffix}".strip()

    @staticmethod
    def _fmt_int(v: Optional[int]) -> str:
        return "—" if v is None else str(int(v))

    def format_overlay_lines(self) -> str:
        """Upper-left overlay: bath, truncation, time evolution, runtime."""
        return "\n".join(
            [
                "Spectral density & bath",
                f"  T          = {self._fmt_float(self.temperature_K, 1, 'K')}",
                f"  η          = {self._fmt_float(self.eta_cm, 1, 'cm⁻¹')}",
                f"  γ⁻¹        = {self._fmt_float(self.gamma_inverse_fs, 1, 'fs')}",
                f"  FWHM       = {self._fmt_float(self.static_disorder_fwhm_cm, 1, 'cm⁻¹')}",
                "Truncation",
                f"  K          = {self._fmt_int(self.n_extra_poles)}",
                f"  L_cutoff   = {self._fmt_int(self.L_cutoff)}",
                f"  L_total    = {self._fmt_int(self.L_total)}",
                f"  npa        = {self._fmt_int(self.npa)}",
                f"  tolerance  = {'—' if self.tolerance is None else f'{self.tolerance:.1e}'}",
                "Time evolution",
                f"  dt         = {self._fmt_float(self.dt_fs, 1, 'fs')}",
                f"  n_steps    = {self._fmt_int(self.n_steps)}",
                "Runtime",
                f"  total      = {format_duration(self.computation_time_s)}",
            ]
        )


def _disorder_panel_int(
    raw,
    attr_name: str,
    source_h5: str,
    disorder_on: bool,
) -> Optional[int]:
    if raw is None:
        if disorder_on:
            warnings.warn(
                f"parameter panel: static disorder on (FWHM>0) but missing "
                f"/parameters attr {attr_name!r} in {source_h5!r}",
                UserWarning,
                stacklevel=2,
            )
        return None if disorder_on else 0
    return int(raw)


def load_run_parameters_panel(source_h5: str) -> RunParametersPanel:
    """Load panel fields from raw absorption HDF5 ``/parameters`` attrs."""
    if not os.path.isfile(source_h5):
        warnings.warn(
            f"parameter panel: source HDF5 not found: {source_h5!r}",
            UserWarning,
            stacklevel=2,
        )
        return RunParametersPanel()

    with h5py.File(source_h5, "r") as f:
        fwhm = heom_h5.param(f, "static_disorder_fwhm_cm")
        disorder_on = fwhm is not None and float(fwhm) > 0.0

        L_total = _disorder_panel_int(
            heom_h5.get_attr(f["/parameters"], "L_total"),
            "L_total",
            source_h5,
            disorder_on,
        )
        npa = _disorder_panel_int(
            heom_h5.get_attr(f["/parameters"], "npa"),
            "npa",
            source_h5,
            disorder_on,
        )

        panel = RunParametersPanel(
            eta_cm=heom_h5.param(f, "eta_cm"),
            gamma_inverse_fs=heom_h5.param(f, "gamma_inverse_fs"),
            static_disorder_fwhm_cm=fwhm,
            temperature_K=heom_h5.param(f, "temperature"),
            n_extra_poles=heom_h5.param(f, "n_extra_poles"),
            L_cutoff=heom_h5.param(f, "L_cutoff"),
            L_total=L_total,
            npa=npa,
            tolerance=heom_h5.param(f, "filter_threshold"),
            dt_fs=heom_h5.param(f, "dt"),
            n_steps=heom_h5.param(f, "n_steps"),
            computation_time_s=heom_h5.root_attr(f, "simulation_time_seconds"),
        )

    for attr_name, value in (
        ("eta_cm", panel.eta_cm),
        ("gamma_inverse_fs", panel.gamma_inverse_fs),
        ("static_disorder_fwhm_cm", panel.static_disorder_fwhm_cm),
        ("temperature", panel.temperature_K),
        ("n_extra_poles", panel.n_extra_poles),
        ("L_cutoff", panel.L_cutoff),
        ("filter_threshold", panel.tolerance),
        ("dt", panel.dt_fs),
        ("n_steps", panel.n_steps),
    ):
        if value is None:
            warnings.warn(
                f"parameter panel: missing /parameters attr {attr_name!r} "
                f"in {source_h5!r}",
                UserWarning,
                stacklevel=2,
            )
    return panel
