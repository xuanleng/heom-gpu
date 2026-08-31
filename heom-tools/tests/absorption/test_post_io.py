"""Tests for heom_tools.absorption.post.io."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile

import h5py
import pytest

from heom_tools.common.h5_config import read_source_config_yaml_dict, resolve_source_h5
from heom_tools.common.config_diff import (
    format_config_diff_overlay,
    load_config_diff_summary,
    print_config_yaml_diffs,
)


def _write_source_h5(path: str, yaml_text: str) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("parameters/config_yaml", data=yaml_text)


def _write_spectrum_h5(path: str, source_h5: str) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("omega_cm", data=[12000.0, 12100.0])
        f.create_dataset("intensity_total_norm", data=[1.0, 0.5])
        f.attrs["source_h5"] = os.path.abspath(source_h5)


def test_resolve_spectrum_missing_source_h5_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spec = os.path.join(tmp, "broken_spectrum.h5")
        with h5py.File(spec, "w") as f:
            f.create_dataset("omega_cm", data=[12000.0])
        with pytest.raises(ValueError, match="source_h5"):
            resolve_source_h5(spec)


def test_resolve_and_read_via_spectrum_chain() -> None:
    yaml_a = "truncation:\n  npa: 2\n"
    yaml_b = "truncation:\n  npa: 1\n"
    with tempfile.TemporaryDirectory() as tmp:
        src_a = os.path.join(tmp, "run_a.h5")
        src_b = os.path.join(tmp, "run_b.h5")
        spec_a = os.path.join(tmp, "run_a_spectrum.h5")
        spec_b = os.path.join(tmp, "run_b_post.h5")
        _write_source_h5(src_a, yaml_a)
        _write_source_h5(src_b, yaml_b)
        _write_spectrum_h5(spec_a, src_a)
        _write_spectrum_h5(spec_b, src_b)

        assert resolve_source_h5(spec_a) == os.path.abspath(src_a)
        assert read_source_config_yaml_dict(spec_b)["truncation"]["npa"] == 1

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_config_yaml_diffs([spec_a, spec_b])
        out = buf.getvalue()
        assert "[config diff]" in out
        assert "1 vs 2:" in out
        assert "npa" in out and "2" in out and "-> 1" in out

        overlay = format_config_diff_overlay([spec_a, spec_b])
        assert overlay is not None
        assert overlay.startswith("1 vs 2:")
        assert "npa" in overlay and "-> 1" in overlay


def test_load_config_diff_summary_via_spectrum_h5() -> None:
    yaml_a = "truncation:\n  npa: 2\n"
    yaml_b = "truncation:\n  npa: 1\n"
    with tempfile.TemporaryDirectory() as tmp:
        src_a = os.path.join(tmp, "run_a.h5")
        src_b = os.path.join(tmp, "run_b.h5")
        spec_a = os.path.join(tmp, "run_a_spectrum.h5")
        spec_b = os.path.join(tmp, "run_b_post.h5")
        _write_source_h5(src_a, yaml_a)
        _write_source_h5(src_b, yaml_b)
        _write_spectrum_h5(spec_a, src_a)
        _write_spectrum_h5(spec_b, src_b)

        summary = load_config_diff_summary([spec_a, spec_b])
        assert summary is not None
        assert summary.overlay_text == format_config_diff_overlay([spec_a, spec_b])
