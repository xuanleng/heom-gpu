"""Small display helpers (no HDF5 / numpy physics)."""
from __future__ import annotations

from typing import Optional


def format_duration(seconds: Optional[float]) -> str:
    """Human-readable wall-clock tiers matching the C++ console formatter."""
    if seconds is None:
        return "—"
    total = int(seconds)
    d = total // 86400
    h = (total % 86400) // 3600
    m = (total % 3600) // 60
    s = total % 60
    if d > 0:
        return f"{d}d {h}h {m}m {s}s"
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{seconds:.3f}s"
