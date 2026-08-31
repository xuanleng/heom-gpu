"""Small helpers shared across population I/O, plotting, and analysis."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

LEGEND_COLS = {6: 2, 9: 3}
MAX_METADATA_LINES = 2000


def basename_noext(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]


def legend_cols(n_sites: int) -> int:
    for threshold in sorted(LEGEND_COLS.keys()):
        if n_sites <= threshold:
            return LEGEND_COLS[threshold]
    return 4


def apply_time_limit(
    time: np.ndarray, limit: Optional[float], *arrays: np.ndarray
) -> tuple:
    """Slice ``time`` and peer arrays to ``time <= limit``.

    ``limit is None`` means no window. ``limit <= 0`` is invalid and raises.
    """
    if limit is None:
        return (time, *arrays)
    if limit <= 0:
        raise ValueError(f"time_limit must be positive, got {limit!r}")
    mask = time <= limit
    return (time[mask], *(a[mask] for a in arrays))


def require_nonempty_time(
    time: np.ndarray,
    time_limit: Optional[float],
    *,
    data_time: Optional[np.ndarray] = None,
) -> None:
    """Raise if a time window produced no samples."""
    if time.size > 0:
        return
    src = data_time if data_time is not None and data_time.size else None
    if src is not None:
        span = f"data range {float(src[0]):g}–{float(src[-1]):g} fs"
    else:
        span = "no samples in source time array"
    raise ValueError(f"time window is empty (time_limit={time_limit!r}; {span})")


def interp_populations(
    time_common: np.ndarray,
    time: np.ndarray,
    populations: np.ndarray,
    n_sites: int,
) -> np.ndarray:
    """Interpolate ``populations[:, :n_sites]`` onto ``time_common``."""
    return np.column_stack(
        [np.interp(time_common, time, populations[:, i]) for i in range(n_sites)]
    )


def common_time_grid(time1: np.ndarray, time2: np.ndarray) -> np.ndarray:
    t_min = max(time1[0], time2[0])
    t_max = min(time1[-1], time2[-1])
    grid = np.unique(np.concatenate([time1, time2]))
    return grid[(grid >= t_min) & (grid <= t_max)]
