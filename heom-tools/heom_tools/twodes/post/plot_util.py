"""Deprecated shim — import from heom_tools.twodes.viz instead.

Removal target: after 2026-09. New code must not import this module.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "heom_tools.twodes.post.plot_util is deprecated; "
    "import from heom_tools.twodes.viz instead (remove after 2026-09)",
    DeprecationWarning,
    stacklevel=2,
)

from heom_tools.twodes.viz.freq_panel import apply_freq_window, draw_freq_panel
from heom_tools.twodes.viz.norm import make_norm

__all__ = ["apply_freq_window", "draw_freq_panel", "make_norm"]
