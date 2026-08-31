"""Shared constants for 2DES post-processing."""
from __future__ import annotations

from typing import Dict

import numpy as np

# ν(cm^-1) = f(Hz)/c with c in cm/fs → Δω = FS_TO_CM / (N·Δt_fs).
FS_TO_CM = 33356.40952
# Published ω lattice step (cm^-1), not independent spectral resolution.
# Physical resolve ≈ FS_TO_CM / (n_data · dt); zero-pad only densifies the FFT.
DW_DEFAULT_CM = 10.0
H5_COMPLEX = np.dtype([("r", "f8"), ("i", "f8")])

M_INV = np.array([[4.0, -1.0, -1.0],
                  [-1.0, 4.0, -1.0],
                  [-1.0, -1.0, 4.0]])

PRESETS = {
    "ap": (0.0, 0.0, 0.0, 0.0),
    "perp": (0.0, 0.0, 90.0, 90.0),
    # Thyrhaug Nat. Chem. 2018 SI §1 / Zanni–Hochstrasser angles.
    # Global sign vs the older (−45, 45, 90, 0) preset is an experimental
    # spectrum phase-calibration choice (match measured maps), not a
    # change in the isotropic averaging theory.
    "double_crossed": (45.0, -45.0, 90.0, 0.0),
}

REQUIRED_SIGNALS = frozenset({"rephasing", "nonrephasing"})
PATHWAY_PLOT_ORDER = ("GSB", "SE", "ESA")
STANDARD_PATHWAYS = frozenset(PATHWAY_PLOT_ORDER)

H5_PATHWAY_SUFFIX = {
    "gsb": "GSB",
    "se": "SE",
    "esa": "ESA",
}

SpecMap = Dict[str, Dict[str, np.ndarray]]
SlabMap = Dict[str, Dict[str, np.ndarray]]

DEFAULTS = {
    "out": None,
    "scan_dir": None,
    "pol": "ap",
    "tw_index": "all",
    "w0": 0.0,
    "dw": DW_DEFAULT_CM,
    "window": None,
    "levels": 25,
    "lines": [],
    "plot": True,
    "time_domain": True,
    "write_h5": False,
    "h5_spectra": None,
    "force": False,
    "fft_method": "axis",
}

FFT_RUNTIME_NOTES = {
    "axis": "production — mixed fft(τ)/ifft(t); nr × n_tau to match rp; × Δτ·Δt",
    "check": "diagnostic — cross-check every FFT (pathway + total); one summary per Tw",
    "legacy_mode3": "NOT in config — refs/fft_2d_data mode 3; spectra ≠ axis; tests only",
}

CONFIG_CANDIDATES = ("twodes_post.yaml", "twodes_post.yml", "twodes_post.json")
