"""Merge solver 2DES HDF5 along tw_fs / resp axis 4."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import h5py
import numpy as np
import yaml

from heom_tools.common.config_hash import physics_hash
from heom_tools.twodes.post.io import from_h5_complex, is_solver_2des_h5, write_h5_complex_ds

from .fingerprint import (
    FieldDiff,
    SolverFingerprint,
    diff_fingerprints,
    format_diff_report,
    load_fingerprint,
)

MergeOverlapPolicy = Literal["fail", "keep-first", "keep-last"]
DEFAULT_OVERLAP_RTOL = 1e-9
# Cluster waiting times that differ only by float noise as one T.
_TW_GROUP_ATOL_FS = 1e-6


@dataclass(frozen=True)
class TwSliceRef:
    tw_fs: float
    file_index: int
    slice_index: int
    path: str


@dataclass(frozen=True)
class OverlapConflict:
    tw_fs: float
    sources: Tuple[TwSliceRef, ...]
    max_rel_err: float


def _read_resp_slice(path: str, slice_index: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        raw = f["/results/resp"][..., slice_index, :]
    if getattr(raw.dtype, "names", None):
        return from_h5_complex(raw)
    return np.asarray(raw)


def _max_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    den = float(max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-30))
    return float(np.max(np.abs(a - b)) / den)


def _collect_slice_refs(paths: Sequence[str]) -> List[TwSliceRef]:
    refs: List[TwSliceRef] = []
    for fi, path in enumerate(paths):
        fp = load_fingerprint(path)
        for ki, tw_val in enumerate(fp.tw):
            refs.append(TwSliceRef(float(tw_val), fi, int(ki), path))
    refs.sort(key=lambda r: (r.tw_fs, r.file_index))
    return refs


def _group_by_tw(
    refs: List[TwSliceRef],
    *,
    atol_fs: float = _TW_GROUP_ATOL_FS,
) -> Dict[float, List[TwSliceRef]]:
    """Group slices by waiting time with absolute tolerance (not exact float ==)."""
    groups: Dict[float, List[TwSliceRef]] = {}
    keys: List[float] = []
    for ref in refs:
        matched = None
        for key in keys:
            if abs(ref.tw_fs - key) <= atol_fs:
                matched = key
                break
        if matched is None:
            keys.append(ref.tw_fs)
            groups[ref.tw_fs] = [ref]
        else:
            groups[matched].append(ref)
    return groups


def _resolve_overlaps(
    groups: Dict[float, List[TwSliceRef]],
    *,
    on_overlap: MergeOverlapPolicy,
    rtol: float,
) -> Tuple[List[TwSliceRef], List[OverlapConflict], List[str]]:
    chosen: List[TwSliceRef] = []
    conflicts: List[OverlapConflict] = []
    notes: List[str] = []

    for tw_val in sorted(groups.keys()):
        srcs = groups[tw_val]
        if len(srcs) == 1:
            chosen.append(srcs[0])
            continue

        slices = [_read_resp_slice(s.path, s.slice_index) for s in srcs]
        max_err = 0.0
        for sl in slices[1:]:
            max_err = max(max_err, _max_rel_err(slices[0], sl))

        if max_err <= rtol:
            chosen.append(srcs[0])
            notes.append(
                f"T={tw_val:g} fs: duplicate in {len(srcs)} file(s); "
                f"resp agree (rel_err<={rtol:g}) — kept {os.path.basename(srcs[0].path)}",
            )
            continue

        conflicts.append(OverlapConflict(tw_val, tuple(srcs), max_err))
        if on_overlap == "fail":
            continue
        pick = srcs[0] if on_overlap == "keep-first" else srcs[-1]
        chosen.append(pick)
        notes.append(
            f"T={tw_val:g} fs: overlap mismatch (max_rel_err={max_err:.3e}); "
            f"policy={on_overlap} — kept {os.path.basename(pick.path)}",
        )

    return chosen, conflicts, notes


def format_overlap_report(conflicts: List[OverlapConflict]) -> str:
    lines = [
        "twodes-merge: overlapping waiting times with different resp data",
        "Re-run with --on-overlap keep-first|keep-last to pick one source,",
        "or remove duplicate-Tw inputs.",
    ]
    for c in conflicts:
        lines.append(f"  T={c.tw_fs:g} fs  max_rel_err={c.max_rel_err:.3e}")
        for s in c.sources:
            lines.append(f"    {os.path.basename(s.path)}  (tw index {s.slice_index})")
    return "\n".join(lines)


def update_config_yaml_tw(yaml_text: str, tw_values: Sequence[float]) -> str:
    """Set spectroscopy.tw to the merged waiting-time list."""
    doc = yaml.safe_load(yaml_text) or {}
    if not isinstance(doc, dict):
        raise ValueError("config_yaml root must be a mapping")
    spec = doc.setdefault("spectroscopy", {})
    if not isinstance(spec, dict):
        raise ValueError("config_yaml spectroscopy must be a mapping")
    tw_out = [float(v) for v in tw_values]
    if all(abs(v - round(v)) < 1e-6 for v in tw_out):
        spec["tw"] = [str(int(round(v))) for v in tw_out]
    else:
        spec["tw"] = [f"{v:g}" for v in tw_out]
    return yaml.dump(doc, sort_keys=False, allow_unicode=True)


def validate_merge_inputs(paths: Sequence[str]) -> Tuple[SolverFingerprint, List[FieldDiff]]:
    if len(paths) < 2:
        raise ValueError("twodes-merge needs at least two solver 2DES HDF5 files")
    abs_paths = [os.path.abspath(p) for p in paths]
    for path in abs_paths:
        if not os.path.exists(path):
            raise ValueError(f"not found: {path}")
        if not is_solver_2des_h5(path):
            raise ValueError(f"not a solver 2DES HDF5: {path}")

    ref_fp = load_fingerprint(abs_paths[0])
    all_diffs: List[FieldDiff] = []
    for path in abs_paths[1:]:
        all_diffs.extend(diff_fingerprints(ref_fp, load_fingerprint(path)))
    return ref_fp, all_diffs


def merge_solver_2des(
    paths: Sequence[str],
    out_path: str,
    *,
    on_overlap: MergeOverlapPolicy = "fail",
    overlap_rtol: float = DEFAULT_OVERLAP_RTOL,
) -> Dict[str, object]:
    """Merge inputs into *out_path* (metadata from first file). Returns summary dict."""
    abs_paths = [os.path.abspath(p) for p in paths]
    out_path = os.path.abspath(out_path)
    if out_path in abs_paths:
        raise ValueError("output path must not be one of the input files")

    ref_fp, diffs = validate_merge_inputs(abs_paths)
    if diffs:
        raise ValueError(format_diff_report(diffs))

    refs = _collect_slice_refs(abs_paths)
    groups = _group_by_tw(refs)
    chosen, conflicts, notes = _resolve_overlaps(
        groups, on_overlap=on_overlap, rtol=overlap_rtol,
    )
    if conflicts and on_overlap == "fail":
        raise ValueError(format_overlap_report(conflicts))

    slices = [_read_resp_slice(r.path, r.slice_index) for r in chosen]
    merged_tw = np.array([r.tw_fs for r in chosen], dtype=np.float64)
    merged_resp = np.stack(slices, axis=4)

    outdir = os.path.dirname(out_path) or "."
    os.makedirs(outdir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".h5", dir=outdir)
    os.close(fd)
    try:
        shutil.copy2(abs_paths[0], tmp)
        with h5py.File(tmp, "r+") as f:
            res = f["/results"]
            del res["tw_fs"]
            res.create_dataset("tw_fs", data=merged_tw)

            del res["resp"]
            # Same compound {r,i} contract as the solver / twodes-post writers.
            write_h5_complex_ds(res, "resp", merged_resp)

            par = f["/parameters"]
            if "config_yaml" in par:
                raw = par["config_yaml"][()]
                yaml_text = raw.decode() if isinstance(raw, bytes) else str(raw)
                new_yaml = update_config_yaml_tw(yaml_text, merged_tw)
                del par["config_yaml"]
                par.create_dataset("config_yaml", data=np.bytes_(new_yaml.encode("utf-8")))
                # Physics identity of the merged config (new tw list) — same
                # scheme the solver uses, so it stays consistent with config_yaml.
                f.attrs["config_hash"] = physics_hash(new_yaml)

            sources = [
                {"path": os.path.basename(p), "tw_fs": load_fingerprint(p).tw.tolist()}
                for p in abs_paths
            ]
            par.attrs["merge_sources"] = json.dumps(sources)
            par.attrs["merge_overlap_policy"] = on_overlap
            par.attrs["merge_tw_fs"] = json.dumps(merged_tw.tolist())

        os.replace(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    return {
        "out": out_path,
        "tw_fs": merged_tw.tolist(),
        "n_tw": len(merged_tw),
        "sources": [os.path.basename(p) for p in abs_paths],
        "notes": notes,
        "overlap_conflicts": len(conflicts),
    }
