"""Tests for heom_tools.common.config_diff."""
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import h5py
import pytest

from heom_tools.common.config_diff import (
    _display_keys_for_entries,
    format_vs_lines,
    config_yaml_diff_entries,
    format_config_diff_overlay,
    load_config_diff_summary,
    print_config_yaml_diffs,
)


def _write_config_h5(path: str, yaml_text: str) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("parameters/config_yaml", data=yaml_text)


def test_config_yaml_diff_entries() -> None:
    base = {"time_evolution": {"dt": 1.0, "n_steps": 100}}
    other = {"time_evolution": {"dt": 2.0, "n_steps": 100}}
    entries = config_yaml_diff_entries(base, other)
    assert len(entries) == 1
    assert entries[0][0] == "time_evolution.dt"


def test_load_config_diff_summary_from_h5() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        a = str(Path(tmp) / "a.h5")
        b = str(Path(tmp) / "b.h5")
        _write_config_h5(a, "time_evolution:\n  dt: 1.0\n")
        _write_config_h5(b, "time_evolution:\n  dt: 2.0\n")

        summary = load_config_diff_summary([a, b], baseline_idx=0)
        assert summary is not None
        assert "dt" in summary.overlay_text


def test_config_yaml_diff_nested_keys_sorted() -> None:
    base = {"truncation": {"npa": 2, "L_total": 12}, "bath": {"eta": 70.0}}
    other = {"truncation": {"npa": 1, "L_total": 12}, "bath": {"eta": 70.0, "static_disorder_fwhm": 100.0}}
    diffs = config_yaml_diff_entries(base, other)
    keys = [d[0] for d in diffs]
    assert keys == ["bath.static_disorder_fwhm", "truncation.npa"]
    assert diffs[0][1] is not None and diffs[1][2] == 1


def test_display_keys_disambiguate_collision() -> None:
    entries = [
        ("truncation.npa", 2, 1),
        ("bath.npa", 0, 2),
    ]
    assert _display_keys_for_entries(entries) == ["truncation.npa", "bath.npa"]


def test_matrix_diff_shows_compact_label() -> None:
    ham = [[1.0, 0.0], [0.0, 2.0]]
    ham2 = [[1.0, 0.1], [0.0, 2.0]]
    entries = config_yaml_diff_entries(
        {"system": {"hamiltonian": ham}},
        {"system": {"hamiltonian": ham2}},
    )
    lines = format_vs_lines(0, 2, entries)
    assert len(lines) == 1
    assert "(matrix differs)" in lines[0]


def test_format_vs_lines_multiline_aligned() -> None:
    base = {"truncation": {"npa": 2, "L_total": 12}, "bath": {"eta": 70.0}}
    other = {
        "truncation": {"npa": 1, "L_total": 0},
        "bath": {"eta": 70.0, "static_disorder_fwhm": 100.0},
    }
    entries = config_yaml_diff_entries(base, other)
    lines = format_vs_lines(0, 2, entries)
    assert len(lines) == 3
    assert lines[0].startswith("1 vs 2:")
    indent = " " * len("1 vs 2: ")
    assert lines[1].startswith(indent)
    assert lines[2].startswith(indent)
    arrow_cols = [line.index(" -> ") for line in lines]
    assert len(set(arrow_cols)) == 1


def test_print_config_yaml_diffs_identical_silent() -> None:
    yaml_text = "time_evolution:\n  dt: 1.0\n"
    with tempfile.TemporaryDirectory() as tmp:
        a = str(Path(tmp) / "a.h5")
        b = str(Path(tmp) / "b.h5")
        _write_config_h5(a, yaml_text)
        _write_config_h5(b, yaml_text)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_config_yaml_diffs([a, b])
        assert buf.getvalue() == ""
        assert format_config_diff_overlay([a, b]) is None


def test_load_config_diff_summary_identical_returns_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        a = str(Path(tmp) / "a.h5")
        b = str(Path(tmp) / "b.h5")
        yaml_text = "time_evolution:\n  dt: 1.0\n"
        _write_config_h5(a, yaml_text)
        _write_config_h5(b, yaml_text)

        assert load_config_diff_summary([a, b], baseline_idx=0) is None
