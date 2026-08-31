"""Tests for heom_tools.common.heom_h5."""
from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from heom_tools.common import heom_h5


def test_decode_attr_bytes_and_scalar() -> None:
    assert heom_h5.decode_attr(b"abc") == "abc"
    assert heom_h5.decode_attr(np.int64(7)) == 7


def test_param_scalar_attr_and_per_site_dataset() -> None:
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        path = tf.name
    try:
        with h5py.File(path, "w") as f:
            params = f.create_group("parameters")
            params.attrs["w0_cm"] = 12000.0
            params.create_dataset("eta_cm", data=np.array([35.0, 40.0]))

        with h5py.File(path, "r") as f:
            assert heom_h5.param(f, "w0_cm") == 12000.0
            assert heom_h5.param(f, "eta_cm") == 35.0
            np.testing.assert_allclose(
                heom_h5.param_array(f, "eta_cm"),
                [35.0, 40.0],
            )
    finally:
        Path(path).unlink(missing_ok=True)


def test_task_and_run_kind() -> None:
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        path = tf.name
    try:
        with h5py.File(path, "w") as f:
            f.attrs["task"] = "dynamics"
        with h5py.File(path, "r") as f:
            assert heom_h5.task(f) == "dynamics"
            assert heom_h5.is_dynamics_run(f)
            assert not heom_h5.is_absorption_run(f)
    finally:
        Path(path).unlink(missing_ok=True)


def test_eigenvalues_absolute_cm() -> None:
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        path = tf.name
    try:
        with h5py.File(path, "w") as f:
            params = f.create_group("parameters")
            params.attrs["w0_cm"] = 100.0
            derived = f.create_group("derived")
            derived.create_dataset(
                "eigenvalues_relative_cm",
                data=np.array([0.0, 50.0, 120.0]),
            )
        with h5py.File(path, "r") as f:
            np.testing.assert_allclose(
                heom_h5.eigenvalues_absolute_cm(f),
                [100.0, 150.0, 220.0],
            )
    finally:
        Path(path).unlink(missing_ok=True)
