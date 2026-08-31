"""Population-dynamics post-processing.

The public helpers are imported here. The CLI loads through :func:`main`.
"""
from __future__ import annotations

from typing import Any

from .analysis import create_comparison_result, print_analysis
from .io import parse_population_file
from .models import BasisSeries, ComparisonResult, PopulationData
from .plot import plot_dual_basis, plot_single_population
from .plot_compare import compare_populations

__all__ = [
    "BasisSeries",
    "ComparisonResult",
    "PopulationData",
    "batch_plot_directory",
    "compare_populations",
    "create_comparison_result",
    "main",
    "parse_population_file",
    "plot_dual_basis",
    "plot_single_population",
    "print_analysis",
]


def __getattr__(name: str) -> Any:
    if name == "main":
        from .cli import main

        return main
    if name == "batch_plot_directory":
        from .cli import batch_plot_directory

        return batch_plot_directory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
