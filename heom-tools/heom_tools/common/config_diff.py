"""Compare ``parameters/config_yaml`` across solver HDF5 inputs."""
from __future__ import annotations

import os
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from heom_tools.common.h5_config import read_source_config_yaml_dict

_MISSING = object()


@dataclass(frozen=True)
class ConfigDiffBlock:
    """Diffs of one theory curve vs baseline (legend indices are 1-based).

    Overlay prefix uses ``{baseline} vs {other}`` (from baseline looking at other).
    Arrow body is always ``value(baseline) -> value(other)``; Δ plots use
    ``other − baseline`` and spell the formula explicitly (e.g. ``(2 − 1)``).
    """

    other_idx: int
    entries: list[tuple[str, Any, Any]]


@dataclass(frozen=True)
class ConfigDiffSummary:
    blocks: list[ConfigDiffBlock]
    overlay_text: str


def _short_config_key(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _display_keys_for_entries(entries: list[tuple[str, Any, Any]]) -> list[str]:
    """Legend-safe keys: disambiguate colliding short names with ``parent.key``."""
    paths = [key_path for key_path, _, _ in entries]
    shorts = [_short_config_key(path) for path in paths]
    dup_shorts = {short for short, count in Counter(shorts).items() if count > 1}
    display: list[str] = []
    for path, short in zip(paths, shorts):
        if short not in dup_shorts:
            display.append(short)
        elif "." in path:
            display.append(".".join(path.rsplit(".", 2)[-2:]))
        else:
            display.append(path)
    return display


def _is_matrix_like(value: Any) -> bool:
    if value is _MISSING:
        return False
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)
            return arr.ndim >= 2 or arr.size > 64
        except (TypeError, ValueError):
            return False
    return False


def _format_diff_side_value(value: Any) -> str:
    if _is_matrix_like(value):
        return "(matrix)"
    return _format_config_value(value)


def _format_diff_pair(baseline_value: Any, other_value: Any) -> tuple[str, str]:
    if _is_matrix_like(baseline_value) or _is_matrix_like(other_value):
        if not values_equal(baseline_value, other_value):
            return "(matrix differs)", "(matrix differs)"
    return _format_config_value(baseline_value), _format_config_value(other_value)


def _format_diff_body(
    display_key: str,
    baseline_value: Any,
    other_value: Any,
    *,
    key_width: int,
    baseline_width: int,
) -> str:
    left, right = _format_diff_pair(baseline_value, other_value)
    return f"{display_key:<{key_width}}  {left:<{baseline_width}} -> {right}"


def format_vs_lines(
    baseline_idx: int,
    other_idx: int,
    entries: list[tuple[str, Any, Any]],
) -> list[str]:
    if not entries:
        return []
    prefix = f"{baseline_idx + 1} vs {other_idx}: "
    indent = " " * len(prefix)
    display_keys = _display_keys_for_entries(entries)
    key_width = max(len(key) for key in display_keys)
    left_values = [_format_diff_side_value(av) for _, av, _ in entries]
    baseline_width = max(len(text) for text in left_values)
    lines: list[str] = []
    for i, ((_, av, bv), display_key) in enumerate(zip(entries, display_keys)):
        body = _format_diff_body(
            display_key, av, bv, key_width=key_width, baseline_width=baseline_width,
        )
        lines.append((prefix if i == 0 else indent) + body)
    return lines


def _flatten_config(obj: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key in sorted(obj):
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_config(obj[key], path)
    else:
        yield prefix, obj


def values_equal(a: Any, b: Any) -> bool:
    if a is _MISSING or b is _MISSING:
        return False
    if type(a) is not type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) == float(b)
        return False
    if isinstance(a, (list, tuple)):
        try:
            return bool(np.allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0, equal_nan=True))
        except (TypeError, ValueError):
            return a == b
    return a == b


def _format_config_value(value: Any) -> str:
    if value is _MISSING:
        return "(missing)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        if _is_matrix_like(value):
            return "(matrix)"
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.size <= 64:
            rows = ["[" + ", ".join(f"{x:g}" for x in row) + "]" for row in arr]
            return "[" + "; ".join(rows) + "]"
        text = repr(value)
        if len(text) > 120:
            return text[:117] + "..."
        return text
    text = repr(value)
    if len(text) > 120:
        return text[:117] + "..."
    return text


def config_yaml_diff_entries(
    baseline: dict[str, Any],
    other: dict[str, Any],
) -> list[tuple[str, Any, Any]]:
    """Return ``(path, baseline_value, other_value)`` for differing keys."""
    base_flat = dict(_flatten_config(baseline))
    other_flat = dict(_flatten_config(other))
    keys = sorted(set(base_flat) | set(other_flat))
    out: list[tuple[str, Any, Any]] = []
    for key in keys:
        av = base_flat.get(key, _MISSING)
        bv = other_flat.get(key, _MISSING)
        if not values_equal(av, bv):
            out.append((key, av, bv))
    return out


def collect_config_yaml_diffs(
    h5_paths: list[str],
    baseline_idx: int = 0,
) -> tuple[int, list[ConfigDiffBlock]]:
    """Return ``(baseline_legend_idx, diff blocks)`` for each non-baseline theory file."""
    if len(h5_paths) < 2:
        return 0, []

    configs: list[dict[str, Any] | None] = []

    for path in h5_paths:
        try:
            configs.append(read_source_config_yaml_dict(path))
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"config diff: skipped {os.path.basename(path)!r}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            configs.append(None)

    baseline_cfg = configs[baseline_idx]
    if baseline_cfg is None:
        return baseline_idx + 1, []

    blocks: list[ConfigDiffBlock] = []
    for idx, cfg in enumerate(configs):
        if idx == baseline_idx or cfg is None:
            continue
        diffs = config_yaml_diff_entries(baseline_cfg, cfg)
        if diffs:
            blocks.append(ConfigDiffBlock(other_idx=idx + 1, entries=diffs))
    return baseline_idx + 1, blocks


def _summary_from_blocks(
    blocks: list[ConfigDiffBlock],
    *,
    baseline_idx: int = 0,
) -> ConfigDiffSummary:
    lines: list[str] = []
    for block in blocks:
        lines.extend(format_vs_lines(baseline_idx, block.other_idx, block.entries))
    return ConfigDiffSummary(blocks=blocks, overlay_text="\n".join(lines))


def load_config_diff_summary(
    h5_paths: list[str],
    baseline_idx: int = 0,
) -> Optional[ConfigDiffSummary]:
    """Load YAML diffs once for CLI print + figure overlay."""
    _, blocks = collect_config_yaml_diffs(h5_paths, baseline_idx=baseline_idx)
    if not blocks:
        return None
    return _summary_from_blocks(blocks, baseline_idx=baseline_idx)


def format_config_diff_overlay(
    h5_paths: list[str],
    baseline_idx: int = 0,
) -> Optional[str]:
    """Monospace overlay text for compare figures; ``None`` if no differences."""
    summary = load_config_diff_summary(h5_paths, baseline_idx=baseline_idx)
    return summary.overlay_text if summary else None


def print_config_yaml_diffs(
    h5_paths: list[str] | None = None,
    baseline_idx: int = 0,
    *,
    summary: ConfigDiffSummary | None = None,
) -> None:
    """Print ``[config diff]`` for each non-baseline theory file."""
    if summary is None:
        if h5_paths is None:
            return
        summary = load_config_diff_summary(h5_paths, baseline_idx=baseline_idx)
    if summary is None:
        return

    print("[config diff]", flush=True)
    for block in summary.blocks:
        for line in format_vs_lines(baseline_idx, block.other_idx, block.entries):
            print(f"  {line}", flush=True)
    print(flush=True)
