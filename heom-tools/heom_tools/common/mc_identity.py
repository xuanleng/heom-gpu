"""Disorder identity strings — must match C++ ``load_realizations()`` (mc_realizations.cpp).

The merge key is ``fnv1a64(config_hash + '|' + disorder_identity)``. Python resume
and manifest tooling must build the same ``generator|seed=…|sigma=…`` suffix the
solver writes into shard ``/mc`` attrs.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "format_sigma_cm_for_identity",
    "disorder_identity_from_attrs",
    "disorder_identity_from_draws",
]


def format_sigma_cm_for_identity(sigma_cm) -> str:
    """Comma-separated σ list matching ``std::ostringstream`` default ``<< double``."""
    sigma = np.asarray(sigma_cm, dtype=np.float64).reshape(-1)
    return ",".join(format(float(x), ".6g") for x in sigma)


def disorder_identity_from_attrs(
    *,
    estimator: str,
    generator: str,
    master_seed: int,
    sigma_cm,
    sg_level: int | None = None,
) -> str:
    """Build identity from the node-file attributes (mean vs quadrature)."""
    sig = format_sigma_cm_for_identity(sigma_cm)
    if estimator == "quadrature":
        if sg_level is None:
            raise ValueError("quadrature identity requires sg_level")
        return f"{generator}|level={int(sg_level)}|sigma={sig}"
    return f"{generator}|seed={int(master_seed)}|sigma={sig}"


def disorder_identity_from_draws(draws_path: str) -> str:
    """Read a node file and return the C++-compatible disorder_identity string."""
    import h5py

    with h5py.File(draws_path, "r") as h:
        est = str(h.attrs.get("estimator", ""))
        gen = str(h.attrs["generator"])
        sigma = h.attrs["sigma_cm"]
        sg_level = int(h.attrs["sg_level"]) if "sg_level" in h.attrs else None
        seed = int(h.attrs["master_seed"])
    return disorder_identity_from_attrs(
        estimator=est,
        generator=gen,
        master_seed=seed,
        sigma_cm=sigma,
        sg_level=sg_level,
    )
