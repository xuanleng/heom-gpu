"""Parameter-difference overlay text for population comparison figures."""
from __future__ import annotations

import os
from typing import Any, Optional

from heom_tools.common.config_diff import (
    format_vs_lines,
    values_equal,
    load_config_diff_summary,
    print_config_yaml_diffs,
)
from heom_tools.common.mpl_util import draw_axes_monospace_overlay

_MISSING = object()


def _is_hdf5(path: str) -> bool:
    return path.endswith(".h5") or path.endswith(".hdf5")


def params_diff_entries(
    baseline: dict[str, Any],
    other: dict[str, Any],
) -> list[tuple[str, Any, Any]]:
    keys = sorted(set(baseline) | set(other))
    out: list[tuple[str, Any, Any]] = []
    for key in keys:
        av = baseline.get(key, _MISSING)
        bv = other.get(key, _MISSING)
        if not values_equal(av, bv):
            out.append((key, av, bv))
    return out


def format_params_diff_overlay(
    baseline: dict[str, Any],
    other: dict[str, Any],
    *,
    baseline_idx: int = 0,
    other_idx: int = 2,
) -> Optional[str]:
    entries = params_diff_entries(baseline, other)
    if not entries:
        return None
    lines = format_vs_lines(baseline_idx, other_idx, entries)
    return "\n".join(lines)


def load_compare_diff_overlay(
    file1: str,
    file2: str,
    params1: Optional[dict[str, Any]] = None,
    params2: Optional[dict[str, Any]] = None,
    *,
    print_diff: bool = True,
) -> Optional[str]:
    """Build monospace overlay for file1 (baseline) vs file2 (subject).

    Same rule as las-plot: prefix ``1 vs 2`` means from baseline looking at
    subject; arrow is ``value(1) -> value(2)``. Difference panels use
    Δ = file2 − file1 and label the formula ``(2 − 1)``.

    Prefers ``parameters/config_yaml`` when both inputs are dynamics HDF5;
    falls back to the parsed ``params`` summary dicts from :class:`PopulationData`.
    """
    if _is_hdf5(file1) and _is_hdf5(file2):
        summary = load_config_diff_summary([file1, file2], baseline_idx=0)
        if summary is not None:
            if print_diff:
                print_config_yaml_diffs(summary=summary)
            return summary.overlay_text

    if params1 and params2:
        text = format_params_diff_overlay(params1, params2)
        if text:
            if print_diff:
                print("[config diff]", flush=True)
                for line in text.splitlines():
                    print(f"  {line}", flush=True)
                print(flush=True)
            return text

    return None


def draw_params_diff_overlay(ax, text: Optional[str]) -> None:
    if text:
        draw_axes_monospace_overlay(ax, text)
