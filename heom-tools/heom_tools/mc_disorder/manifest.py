"""RQMC multi-replica manifest I/O (mc-disorder-gen ↔ mc-disorder-run)."""
from __future__ import annotations

import os

import numpy as np
import yaml

__all__ = ["write_rqmc_manifest", "read_rqmc_manifest", "assert_rqmc_draws_file"]


def _read_n_node(draws):
    try:
        import h5py
        with h5py.File(draws, "r") as h:
            return int(h["delta_eps"].shape[0])
    except Exception:
        return None


def assert_rqmc_draws_file(path: str) -> None:
    """Fail fast if a manifest entry is not a scrambled-Sobol node file."""
    import h5py

    with h5py.File(path, "r") as h:
        est = str(h.attrs.get("estimator", ""))
        gen = str(h.attrs.get("generator", ""))
    if est != "mean" or not gen.startswith("sobol_gauss"):
        raise ValueError(
            f"'{path}' is not an rqmc node file "
            f"(estimator={est!r}, generator={gen!r})")


def write_rqmc_manifest(manifest_path, *, master_seed, n_node, n_site,
                        fwhm_cm, sigma_cm, n_replicas, draws_files):
    """Write a small manifest for mc-disorder-run to consume."""
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path)) or "."
    rel_files = []
    for p in draws_files:
        ap = os.path.abspath(p)
        try:
            rel_files.append(os.path.relpath(ap, manifest_dir))
        except ValueError:
            rel_files.append(ap)
    payload = {
        "method": "rqmc",
        "master_seed": int(master_seed),
        "n_node": int(n_node),
        "n_site": int(n_site),
        "fwhm_cm": [float(x) for x in np.asarray(fwhm_cm).reshape(-1)],
        "sigma_cm": [float(x) for x in np.asarray(sigma_cm).reshape(-1)],
        "n_replicas": int(n_replicas),
        "draws_files": rel_files,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def read_rqmc_manifest(manifest_path):
    """Load and validate an mc-disorder-gen RQMC manifest; resolve draws paths."""
    with open(manifest_path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    if doc.get("method") != "rqmc":
        raise ValueError(
            f"'{manifest_path}' is not an rqmc manifest (method={doc.get('method')!r})")
    n_replicas = int(doc.get("n_replicas", 0))
    rel_files = doc.get("draws_files") or []
    if n_replicas < 1:
        raise ValueError(f"'{manifest_path}' has invalid n_replicas={n_replicas}")
    if len(rel_files) != n_replicas:
        raise ValueError(
            f"'{manifest_path}' n_replicas={n_replicas} but "
            f"len(draws_files)={len(rel_files)}")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path)) or "."
    resolved = []
    for rel in rel_files:
        p = rel if os.path.isabs(rel) else os.path.join(manifest_dir, rel)
        if not os.path.isfile(p):
            raise ValueError(f"manifest draws_file missing: '{p}'")
        assert_rqmc_draws_file(p)
        resolved.append(os.path.abspath(p))
    n_expect = doc.get("n_node")
    for p in resolved:
        n_rows = _read_n_node(p)
        if n_rows is None:
            raise ValueError(f"cannot read node count from '{p}'")
        if n_expect is not None and n_rows != int(n_expect):
            raise ValueError(
                f"'{p}' has n_node={n_rows}, manifest expects {n_expect}")
    n_node = int(n_expect) if n_expect is not None else _read_n_node(resolved[0])
    n_site = doc.get("n_site")
    return {
        "method": "rqmc",
        "n_replicas": n_replicas,
        "n_node": n_node,
        "n_site": int(n_site) if n_site is not None else None,
        "master_seed": int(doc["master_seed"]) if "master_seed" in doc else None,
        "fwhm_cm": np.asarray(doc.get("fwhm_cm") or [], dtype=np.float64),
        "draws_files": resolved,
    }
