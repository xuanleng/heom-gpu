"""Read ``parameters/config_yaml`` from solver HDF5 files."""
from __future__ import annotations

import os
from typing import Any

import h5py

from heom_tools.common import heom_h5

_CONFIG_YAML_PATH = "parameters/config_yaml"


def _read_scalar_text_dataset(ds) -> str | None:
    if ds.shape != ():
        return None
    string_dtype = h5py.check_string_dtype(ds.dtype)
    if string_dtype is None and ds.dtype.kind not in ("S", "U", "O"):
        return None
    return heom_h5.decode_attr(ds[()])


def _decode_h5_text(value) -> str:
    return heom_h5.decode_attr(value)


def resolve_source_h5(path: str) -> str:
    """Raw solver HDF5 for a processed spectrum file, or *path* itself."""
    abs_path = os.path.abspath(path)
    with h5py.File(abs_path, "r") as f:
        if "source_h5" in f.attrs:
            return os.path.abspath(_decode_h5_text(f.attrs["source_h5"]))
        if "omega_cm" in f:
            raise ValueError(f"{path}: spectrum HDF5 missing source_h5 attr")
    return abs_path


def read_config_yaml_dict(path: str) -> dict[str, Any]:
    """Parse ``parameters/config_yaml`` from a solver HDF5 file."""
    with h5py.File(path, "r") as f:
        text = (
            _read_scalar_text_dataset(f[_CONFIG_YAML_PATH])
            if _CONFIG_YAML_PATH in f
            else None
        )
    if not text:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("config diff requires PyYAML (pip install pyyaml)") from exc
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        return {}
    return data


def read_source_config_yaml_dict(path: str) -> dict[str, Any]:
    """Parse ``parameters/config_yaml`` from the solver HDF5 behind *path*."""
    return read_config_yaml_dict(resolve_source_h5(path))
