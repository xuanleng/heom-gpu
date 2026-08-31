"""Absorption post-processing: FFT and *_spectrum.h5."""
from __future__ import annotations

from typing import Any

from heom_tools.common.spectrum import EXPERIMENT_TOKEN_RE, discover_experiment_spectra_auto

from .core import AbsorptionPostResult, compute_absorption_post, write_absorption_h5
from .io import discover_unprocessed_absorption_h5_flat, resolve_frequency_window
from .util import format_duration

__all__ = [
    "AbsorptionPostResult",
    "EXPERIMENT_TOKEN_RE",
    "RunParametersPanel",
    "compute_absorption_post",
    "discover_experiment_spectra_auto",
    "discover_unprocessed_absorption_h5_flat",
    "format_duration",
    "load_run_parameters_panel",
    "main",
    "resolve_frequency_window",
    "run_one",
    "write_absorption_h5",
]


def __getattr__(name: str) -> Any:
    if name == "main":
        from .cli import main

        return main
    if name == "run_one":
        from .cli import run_one

        return run_one
    if name == "RunParametersPanel":
        from .panel import RunParametersPanel

        return RunParametersPanel
    if name == "load_run_parameters_panel":
        from .panel import load_run_parameters_panel

        return load_run_parameters_panel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
