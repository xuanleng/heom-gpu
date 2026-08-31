"""Absorption plotting and comparison."""
from __future__ import annotations

from .align import (
    LasAligned,
    ResidualCurve,
    TheorySpectrum,
    align_las_compare,
    load_theory_from_spectrum_h5,
    print_metrics_table,
)
from .cli import main
from .plot import plot_las_compare_png, render_las_compare_figure, show_las_compare_interactive
from .run import allocate_output_png_path

__all__ = [
    "LasAligned",
    "ResidualCurve",
    "TheorySpectrum",
    "align_las_compare",
    "allocate_output_png_path",
    "load_theory_from_spectrum_h5",
    "main",
    "plot_las_compare_png",
    "print_metrics_table",
    "render_las_compare_figure",
    "show_las_compare_interactive",
]
