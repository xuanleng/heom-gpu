"""Shared HEOM YAML I/O for mc-disorder tools.

Single source of truth for save_path / draws_file resolution, aligned with the
solver's task-scoped schema (``<task>.output.save_path``) and legacy
``output.<task>.save_path``.
"""
from __future__ import annotations

import os

import yaml

__all__ = [
    "load_doc",
    "save_doc",
    "save_path_from_doc",
    "read_save_path",
    "patch_save_path_in_doc",
    "resolve_draws_path",
    "read_run_paths",
    "n_states_from_doc",
    "read_n_states",
    "patch_draws_file_in_doc",
    "strip_legacy_bath_disorder",
    "patch_probe_2des_tw",
    "write_probe_cfg",
    "write_calibration_cfg",
    "load_h0_mu",
]


def load_doc(config: str) -> dict:
    with open(config, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_doc(doc: dict, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)


def save_path_from_doc(doc: dict, task: str) -> str | None:
    """Return raw save_path string or None."""
    doc = doc or {}
    task_blk = doc.get(task)
    if isinstance(task_blk, dict):
        out = task_blk.get("output") or {}
        if out.get("save_path"):
            return out["save_path"]
    legacy = (doc.get("output") or {}).get(task) or {}
    if legacy.get("save_path"):
        return legacy["save_path"]
    return None


def patch_save_path_in_doc(doc: dict, task: str, save_path: str) -> None:
    """Write save_path into task-scoped or legacy schema (match existing layout)."""
    sp = save_path if save_path.endswith("/") else save_path + "/"
    if isinstance(doc.get(task), dict):
        doc.setdefault(task, {}).setdefault("output", {})["save_path"] = sp
    else:
        doc.setdefault("output", {}).setdefault(task, {})["save_path"] = sp


def read_save_path(config: str, task: str) -> str:
    sp = save_path_from_doc(load_doc(config), task)
    if not sp:
        raise ValueError(
            f"'{config}' has no save_path for task '{task}' "
            f"(expected {task}.output.save_path or output.{task}.save_path)")
    return sp if sp.endswith("/") else sp + "/"


def resolve_draws_path(doc: dict, config: str) -> str | None:
    sd = (doc or {}).get("static_disorder") or {}
    draws = sd.get("draws_file")
    if not draws:
        return None
    if not os.path.isabs(draws):
        cfg_dir = os.path.dirname(os.path.abspath(config))
        draws = os.path.join(cfg_dir, draws)
    return draws


def read_run_paths(config: str, task: str) -> tuple[str, str]:
    """Return (save_path, draws_path) with one yaml read."""
    doc = load_doc(config)
    sp = save_path_from_doc(doc, task)
    if not sp:
        raise ValueError(
            f"'{config}' has no save_path for task '{task}' "
            f"(expected {task}.output.save_path or output.{task}.save_path)")
    if not sp.endswith("/"):
        sp += "/"
    draws = resolve_draws_path(doc, config)
    if not draws:
        raise ValueError(
            f"'{config}' has no static_disorder.draws_file; MC needs a node "
            "file (generate one with mc-disorder-gen)")
    return sp, draws


def n_states_from_doc(doc: dict) -> int:
    sysd = (doc or {}).get("system") or (doc or {}).get("physics") or {}
    if "n_states" not in sysd:
        raise ValueError("config has no system.n_states")
    return int(sysd["n_states"])


def read_n_states(config: str) -> int:
    return n_states_from_doc(load_doc(config))


def patch_draws_file_in_doc(doc: dict, draws_path: str) -> None:
    doc.setdefault("static_disorder", {})["draws_file"] = os.path.abspath(draws_path)


def strip_legacy_bath_disorder(doc: dict) -> None:
    """Drop bath.static_disorder_fwhm — rejected by the refactored solver."""
    if isinstance(doc.get("bath"), dict):
        doc["bath"].pop("static_disorder_fwhm", None)


def patch_probe_2des_tw(doc: dict) -> None:
    """Force cheapest 2DES probe: single waiting time T=0."""
    doc.setdefault("2des", {})["tw"] = ["0"]


def write_probe_cfg(
    base_cfg: str,
    *,
    draws_path: str,
    save_path: str,
    work_dir: str,
    task: str = "absorption",
) -> str:
    """Temp config for mc-probe: draws + scratch save_path, solver-aligned schema."""
    doc = load_doc(base_cfg)
    patch_draws_file_in_doc(doc, draws_path)
    strip_legacy_bath_disorder(doc)
    patch_save_path_in_doc(doc, task, save_path)
    if task == "2des":
        patch_probe_2des_tw(doc)
    cfg_path = os.path.join(work_dir, "probe_cfg.yaml")
    save_doc(doc, cfg_path)
    return cfg_path


def write_calibration_cfg(
    base_cfg: str,
    task: str,
    scratch_save_path: str,
    *,
    cfg_path: str,
) -> str:
    """Temp config for mc-disorder-calibrate: absolute draws + scratch save_path."""
    doc = load_doc(base_cfg)
    draws = resolve_draws_path(doc, base_cfg)
    if draws:
        patch_draws_file_in_doc(doc, draws)
    patch_save_path_in_doc(doc, task, os.path.abspath(scratch_save_path))
    save_doc(doc, cfg_path)
    return cfg_path


def load_h0_mu(doc: dict):
    """Excited-state H0 [n,n] and dipoles μ [n,3] for stick surrogate / arena."""
    import numpy as np

    sysd = doc.get("system") or doc.get("physics") or {}
    H0 = np.array(sysd["hamiltonian"], dtype=np.float64)
    # Task-scoped configs (fmo.yaml): dipoles live under system; legacy under spectroscopy.
    dip_src = sysd if "dipoles" in sysd else doc.get("spectroscopy") or sysd
    dip = np.array(dip_src["dipoles"], dtype=np.float64)
    strengths = np.array(
        dip_src.get("strengths", sysd.get("strengths", np.ones(len(dip)))),
        dtype=np.float64,
    )
    mu = dip * strengths[:, None]
    return H0, mu
