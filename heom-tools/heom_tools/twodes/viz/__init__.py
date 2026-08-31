"""Shared frequency-window and matplotlib panel helpers (twodes-post/plot internal).

Not part of the student-facing CLI surface; helpers for ``heom_tools.twodes.post``
and ``heom_tools.twodes.plot`` only.
"""
from __future__ import annotations

from .freq_panel import apply_freq_window, draw_freq_panel
from .norm import make_norm
from .window import WINDOW_PAD_HI, WINDOW_PAD_LO, cut_window, window_slice

__all__ = [
    "WINDOW_PAD_HI",
    "WINDOW_PAD_LO",
    "apply_freq_window",
    "cut_window",
    "draw_freq_panel",
    "make_norm",
    "window_slice",
]
