"""Filesystem discovery and source HDF5 summary printing for absorption CLI."""
from __future__ import annotations

import os
import warnings
from collections import defaultdict
from typing import Optional

import h5py
import numpy as np

from heom_tools.common import heom_h5
from heom_tools.common.mc_merge import (
    ensure_shard_group_merged,
    fold_shard_groups,
)
from heom_tools.common import mc_naming
from .util import format_duration
from heom_tools.common.spectrum import (
    SpectrumData,
    default_absorption_frequency_window_cm,
    has_processed_absorption_sibling,
    is_processed_absorption_h5_filename,
    linear_h5_skip_reason,
    processed_absorption_h5_path,
)

__all__ = [
    "discover_unprocessed_absorption_h5_flat",
    "discover_unprocessed_linear_h5_flat",
    "count_dynamics_h5_flat",
    "ensure_base_merged",
    "fold_rank_groups",
    "format_h5_attr_value",
    "has_processed_absorption_sibling",
    "is_processed_absorption_h5_filename",
    "is_rank_shard_filename",
    "print_source_config_yaml",
    "print_source_run_info",
    "print_source_summary",
    "processed_absorption_h5_path",
    "resolve_frequency_window",
]

# Shard/merged naming lives in ONE place: heom_tools.common.mc_naming. Absorption
# only pins the /results dataset it averages (rho_xyz); the merge primitive and
# the name grammar are otherwise task-agnostic.
_RESULTS_DATASET = "rho_xyz"


def is_rank_shard_filename(filename: str) -> bool:
    """True for an MC worker shard basename (ends ``_rank{N}.h5``)."""
    return mc_naming.is_rank_shard(os.path.basename(filename))


def ensure_base_merged(shard_path: str) -> str:
    """Given one shard, merge ALL its sibling shards sharing the same
    ``<base>_<key>`` (across timestamps and intervals) into
    ``<dir>/<base>_<key>_r<lo>-<hi>.h5`` and return that merged path."""
    return ensure_shard_group_merged(
        shard_path,
        _RESULTS_DATASET,
        log_prefix="[las-post]",
    )


def fold_rank_groups(cwd: str) -> None:
    """Ensure every ``<base>_<key>`` shard group in `cwd` has its merged
    ``<base>_<key>_r<lo>-<hi>.h5`` for the current coverage."""
    fold_shard_groups(cwd, _RESULTS_DATASET, log_prefix="[las-post]")


def discover_unprocessed_absorption_h5_flat(cwd: str) -> list[str]:
    """Non-recursive scan: raw absorption ``*.h5`` missing ``*_spectrum.h5``."""
    return discover_unprocessed_linear_h5_flat(cwd, "absorption")


def discover_unprocessed_linear_h5_flat(cwd: str, task: str) -> list[str]:
    """Non-recursive scan: raw ``task`` ``*.h5`` missing ``*_spectrum.h5``."""
    if not os.path.isdir(cwd):
        return []
    names = sorted(
        fn for fn in os.listdir(cwd)
        if fn.endswith(".h5") and not fn.startswith(".")
    )
    out: list[str] = []
    skipped: dict[str, list[str]] = defaultdict(list)
    for fn in names:
        if is_processed_absorption_h5_filename(fn):
            continue
        if is_rank_shard_filename(fn):
            continue  # shards are folded into <base>.h5 before discovery
        src = os.path.join(cwd, fn)
        if not os.path.isfile(src):
            continue
        if has_processed_absorption_sibling(src):
            continue
        reason = linear_h5_skip_reason(src, task)
        if reason is not None:
            skipped[reason].append(fn)
            continue
        out.append(os.path.abspath(src))

    for reason, files in sorted(skipped.items()):
        if len(files) == 1:
            warnings.warn(
                f"--auto: skipped {files[0]!r} ({reason})",
                UserWarning,
                stacklevel=2,
            )
            continue
        sample = ", ".join(files[:2])
        extra = f" … +{len(files) - 2} more" if len(files) > 2 else ""
        warnings.warn(
            f"--auto: skipped {len(files)} files ({reason}): {sample}{extra}",
            UserWarning,
            stacklevel=2,
        )
    return out


def count_dynamics_h5_flat(cwd: str) -> int:
    """Count ``task=dynamics`` HDF5 files in a non-recursive directory scan."""
    if not os.path.isdir(cwd):
        return 0
    n = 0
    for fn in os.listdir(cwd):
        if not fn.endswith(".h5") or fn.startswith("."):
            continue
        path = os.path.join(cwd, fn)
        if not os.path.isfile(path):
            continue
        try:
            with h5py.File(path, "r") as f:
                if heom_h5.is_dynamics_run(f):
                    n += 1
        except OSError:
            continue
    return n


def _decode_h5_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


def format_h5_attr_value(value) -> str:
    if isinstance(value, (bytes, str)):
        return _decode_h5_text(value)
    arr = np.asarray(value)
    if arr.shape == ():
        return _decode_h5_text(arr.item())
    return f"<shape={arr.shape}, dtype={arr.dtype}>"


def _read_scalar_text_dataset(ds) -> Optional[str]:
    if ds.shape != ():
        return None
    string_dtype = h5py.check_string_dtype(ds.dtype)
    if string_dtype is None and ds.dtype.kind not in ("S", "U", "O"):
        return None
    return _decode_h5_text(ds[()])


def _strip_yaml_inline_comment(line: str) -> str:
    """Remove trailing ``# ...`` outside YAML string quotes."""
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if i == 0 or line[i - 1].isspace():
                return line[:i].rstrip()
    return line.rstrip()


def _config_yaml_print_lines(text: str) -> list[str]:
    """YAML body: no comments, no blank lines, uniform +2 space indent."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        stripped = _strip_yaml_inline_comment(line)
        if not stripped.strip():
            continue
        lines.append(f"  {stripped}")
    return lines


_CONFIG_YAML_PATH = "parameters/config_yaml"

_PROVENANCE_ENV_KEYS: tuple[str, ...] = (
    "cpu_model",
    "gpu_model",
    "os",
    "cuda_version",
    "compiler",
)


def _run_info_entries(f: h5py.File) -> list[tuple[str, object]]:
    lines: list[tuple[str, object]] = []

    m_total = heom_h5.derived(f, "M_total", None)
    if m_total is not None:
        lines.append(("M_total", m_total))

    if "/provenance" in f:
        for key in _PROVENANCE_ENV_KEYS:
            if key in f["/provenance"].attrs:
                lines.append((key, heom_h5.provenance(f, key)))

    sim_s = heom_h5.root_attr(f, "simulation_time_seconds", None)
    if sim_s is not None:
        lines.append(("simulation_time", format_duration(float(sim_s))))

    return lines


def _config_yaml_lines_from_file(f: h5py.File) -> list[str]:
    if _CONFIG_YAML_PATH not in f:
        return []
    text = _read_scalar_text_dataset(f[_CONFIG_YAML_PATH])
    if text is None:
        return []
    return _config_yaml_print_lines(text)


def print_source_summary(path: str) -> None:
    """Print ``[run_info]`` and ``[config_yaml]`` from one source HDF5 read."""
    with h5py.File(path, "r") as f:
        run_info = _run_info_entries(f)
        yaml_lines = _config_yaml_lines_from_file(f)

    if run_info:
        print("[run_info]")
        for key, value in run_info:
            print(f"  {key} = {format_h5_attr_value(value)}")

    if yaml_lines:
        print("[config_yaml]")
        for line in yaml_lines:
            print(line)


def print_source_run_info(path: str) -> None:
    """Print ``[run_info]`` only (opens the file once)."""
    with h5py.File(path, "r") as f:
        run_info = _run_info_entries(f)
    if not run_info:
        return
    print("[run_info]")
    for key, value in run_info:
        print(f"  {key} = {format_h5_attr_value(value)}")


def print_source_config_yaml(path: str) -> None:
    """Print ``[config_yaml]`` only (opens the file once)."""
    with h5py.File(path, "r") as f:
        yaml_lines = _config_yaml_lines_from_file(f)
    if not yaml_lines:
        return
    print("[config_yaml]")
    for line in yaml_lines:
        print(line)


def resolve_frequency_window(
    wmin_cli: Optional[float],
    wmax_cli: Optional[float],
    spec: SpectrumData,
) -> tuple[float, float]:
    lo, hi = default_absorption_frequency_window_cm(
        spec.eigenvalues_absolute_cm,
        eigenvectors=spec.eigenvectors,
    )
    if wmin_cli is not None:
        lo = float(wmin_cli)
    if wmax_cli is not None:
        hi = float(wmax_cli)
    return lo, hi
