"""Absorption post-processing core: compute + HDF5 write (no matplotlib)."""
from __future__ import annotations

from heom_tools.common.linear_spectrum import (  # noqa: F401  (re-exported)
    LinearPostResult,
    _sticks_table,
    _source_precision_mode,
    compute_linear_post,
    write_linear_h5,
)
from heom_tools.common.spectrum import SpectrumData, build_stick_spectrum

_COMPLETENESS_WARN_TOL = 1e-9
COMPLETENESS_WARN_TOL = _COMPLETENESS_WARN_TOL

AbsorptionPostResult = LinearPostResult


def compute_absorption_post(
    spec: SpectrumData,
    wmin_cm: float,
    wmax_cm: float,
    *,
    strict_ground_w0: bool = True,
) -> AbsorptionPostResult:
    """Compute total absorption, exciton decomposition, sticks, and peaks."""
    sticks = build_stick_spectrum(
        spec.eigenvalues_absolute_cm,
        spec.eigenvectors,
        spec.dipoles,
        spec.strengths,
    )
    return compute_linear_post(spec, wmin_cm, wmax_cm, sticks,
                               strict_ground_w0=strict_ground_w0)


def write_absorption_h5(result: AbsorptionPostResult, out_path: str) -> None:
    """Write all numeric post-processing outputs into one HDF5 file."""
    write_linear_h5(result, out_path, spectrum_type="absorption",
                    generated_by="heom_tools.absorption.post", entry_point="las-post")
