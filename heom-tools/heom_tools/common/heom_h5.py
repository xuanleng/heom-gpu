#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin helpers that know the heom_solver HDF5 group layout.

The C++ writer (src/output/hdf5_metadata.cpp) emits four sibling groups:

    /                attrs: task, config_hash, simulation_time_seconds
    /parameters/     every user knob + Hamiltonian + dipoles/strengths
                     + initial_density_matrix + config_yaml.
                     Per-site bath inputs (eta_cm, gamma_cm, gamma_inverse_fs,
                     static_disorder_fwhm_cm) are 1-D [n_bath] datasets.
    /derived/        eigenvalues_relative_cm, eigenvectors,
                     bath_c_cm2 / bath_nu_cm (2-D [n_bath][K+1]),
                     M_total, final_active_ados
    /results/        time + density_matrix | rho_xyz
    /provenance/     cpu/gpu/os/cuda/compiler + config_path

This module is intentionally small: attribute lookups across the four groups
become one-liners (`heom_h5.param(f, "w0_cm")`) so every post-processing tool
spells the layout the same way.

Pure helpers — no caching, no `Run` objects, no I/O abstraction beyond
h5py.  Use it with `with h5py.File(...) as f: ...`.
"""

from __future__ import annotations

import h5py
import numpy as np


__all__ = [
    "decode_attr",
    "get_attr",
    "param",
    "param_array",
    "derived",
    "provenance",
    "root_attr",
    "task",
    "eigenvalues_absolute_cm",
    "is_dynamics_run",
    "is_absorption_run",
]


def decode_attr(v):
    """h5py vlen strings come back as ``bytes`` on some platforms; numpy 0-d
    scalars come back as ``np.int64`` / ``np.float64``. Normalise both into
    plain Python types so caller code never branches on dtype.
    """
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v.item()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except (ValueError, TypeError):
            return v
    return v


def get_attr(loc, name, default=None):
    """Read attribute ``name`` from any open h5py file / group, returning
    ``default`` if absent. Always decoded via :func:`decode_attr`.
    """
    if name not in loc.attrs:
        return default
    return decode_attr(loc.attrs[name])


def param(f: h5py.File, name: str, default=None):
    """Read a scalar parameter from ``/parameters``.

    Per-site upgrade (2026-06-04): the bath inputs ``eta_cm`` / ``gamma_cm`` /
    ``gamma_inverse_fs`` / ``static_disorder_fwhm_cm`` moved from scalar
    attributes to 1-D ``[n_bath]`` datasets (one value per pigment). To stay
    back-compatible with old files AND give scalar call sites (labels, prints) a
    sensible value, this returns:
      * the attribute if present (old files, and still-scalar knobs), else
      * element 0 of the dataset ``/parameters/<name>`` if it exists (the
        Default / site-0 value; homogeneous runs have all entries equal), else
      * ``default``.
    Use :func:`param_array` when you need the full per-site vector.
    """
    params = f["/parameters"]
    if name in params.attrs:
        return decode_attr(params.attrs[name])
    if name in params and isinstance(params[name], h5py.Dataset):
        arr = np.asarray(params[name][...])
        if arr.ndim == 0:
            return decode_attr(arr)
        if arr.size > 0:
            return decode_attr(arr.reshape(-1)[0])
    return default


def param_array(f: h5py.File, name: str, default=None):
    """Read a per-site parameter from ``/parameters`` as a 1-D numpy array.

    Returns the full ``[n_bath]`` vector for the per-site bath inputs. Falls
    back to wrapping a scalar attribute in a length-1 array (old files), or
    ``default`` when neither form is present.
    """
    params = f["/parameters"]
    if name in params and isinstance(params[name], h5py.Dataset):
        return np.atleast_1d(np.asarray(params[name][...]))
    if name in params.attrs:
        return np.atleast_1d(np.asarray(decode_attr(params.attrs[name])))
    return default


def derived(f: h5py.File, name: str, default=None):
    """Read attribute from ``/derived``."""
    return get_attr(f["/derived"], name, default)


def provenance(f: h5py.File, name: str, default=None):
    """Read attribute from ``/provenance``."""
    return get_attr(f["/provenance"], name, default)


def root_attr(f: h5py.File, name: str, default=None):
    """Read attribute from the file root (``task``, ``config_hash``,
    ``simulation_time_seconds``).
    """
    return get_attr(f, name, default)


def task(f: h5py.File) -> str:
    """Return the run's task string (``"dynamics"`` or ``"absorption"``).

    Throws KeyError if the attribute is missing — every solver-written file
    has it; missing means the file is not a heom_solver output.
    """
    t = root_attr(f, "task", None)
    if t is None:
        raise KeyError(
            "missing root attribute 'task'; file is not a heom_solver output "
            "(or pre-dates the 2026-05-21 group-layout refactor)"
        )
    return t


def is_dynamics_run(f: h5py.File) -> bool:
    # Total predicate: a file without a 'task' attribute (e.g. an MC partial
    # file sitting in the same dir) is simply not a dynamics run — answer
    # False instead of raising, so directory scans stay robust.
    return root_attr(f, "task", None) == "dynamics"


def is_absorption_run(f: h5py.File) -> bool:
    return root_attr(f, "task", None) == "absorption"


def eigenvalues_absolute_cm(f: h5py.File) -> np.ndarray:
    """Compute ``eigenvalues_relative_cm + w0_cm`` from the file.

    The absolute spectrum used to live on disk; the new layout stores only
    the relative form to avoid the two arrays drifting apart. This helper
    keeps tooling code that wants absolute eV / cm⁻¹ as a one-liner.
    """
    rel = np.asarray(f["/derived/eigenvalues_relative_cm"][...], dtype=np.float64)
    w0 = float(param(f, "w0_cm", 0.0))
    return rel + w0
