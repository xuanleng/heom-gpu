"""Discovery, MC shard folding and frequency window for flu-post.

Everything task-agnostic is the absorption code (``absorption/post/io.py``);
this module only pins the task name and the log prefix.
"""
from __future__ import annotations

from typing import Optional

from heom_tools.absorption.post.io import discover_unprocessed_linear_h5_flat
from heom_tools.common.mc_merge import ensure_shard_group_merged, fold_shard_groups
from heom_tools.common.spectrum import SpectrumData

from .core import default_fluorescence_frequency_window_cm

_RESULTS_DATASET = "rho_xyz"
_LOG = "[flu-post]"


def discover_unprocessed_fluorescence_h5_flat(cwd: str) -> list[str]:
    return discover_unprocessed_linear_h5_flat(cwd, "fluorescence")


def ensure_base_merged(shard_path: str) -> str:
    return ensure_shard_group_merged(shard_path, _RESULTS_DATASET, log_prefix=_LOG)


def fold_rank_groups(cwd: str) -> None:
    fold_shard_groups(cwd, _RESULTS_DATASET, log_prefix=_LOG)


def resolve_frequency_window(wmin_cli: Optional[float], wmax_cli: Optional[float],
                             spec: SpectrumData) -> tuple[float, float]:
    lo, hi = default_fluorescence_frequency_window_cm(spec)
    return (lo if wmin_cli is None else float(wmin_cli),
            hi if wmax_cli is None else float(wmax_cli))
