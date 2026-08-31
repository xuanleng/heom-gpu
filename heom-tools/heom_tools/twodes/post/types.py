"""Shared datatypes for 2DES post-processing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import PostConfig
from .constants import SlabMap, SpecMap
from .h5_spectra import SpectraComputeNeeds, resolve_spectra_compute_needs

@dataclass
class TwSlice:
    """One waiting-time index: spectra and optional time slabs."""

    k: int
    spec_path: SpecMap = field(default_factory=dict)
    spec_total: Dict[str, np.ndarray] = field(default_factory=dict)
    spec_2des: Optional[np.ndarray] = None
    time_path: Optional[SlabMap] = None
    time_total: Optional[Dict[str, np.ndarray]] = None


@dataclass
class PathwayPlan:
    """Pathway sum/breakdown derived from solver HDF5 metadata."""

    pidx: List[int]
    pathway_sum: str


@dataclass
class OutputPlan:
    """What compute_tw_slice and writers need — derived once from cfg + pathways."""

    write_h5: bool
    do_plot: bool
    h5_spectra: Optional[List[str]]
    spectra_needs: SpectraComputeNeeds
    need_time: bool
    need_time_path: bool

    @classmethod
    def from_cfg(cls, cfg: PostConfig, *, pathways: List[str]) -> "OutputPlan":
        write_h5 = bool(cfg.write_h5)
        do_plot = bool(cfg.plot)
        td = bool(cfg.time_domain)
        h5_spectra = cfg.h5_spectra if write_h5 else None
        spectra_needs = resolve_spectra_compute_needs(
            do_plot=do_plot,
            h5_spectra=h5_spectra,
            pathways=pathways,
        )
        return cls(
            write_h5=write_h5,
            do_plot=do_plot,
            h5_spectra=h5_spectra,
            spectra_needs=spectra_needs,
            need_time=do_plot and td,
            need_time_path=do_plot and td and bool(pathways),
        )


@dataclass
class PlotCtx:
    """Frequency/time axes and display scalars only (no matplotlib handles)."""

    w_tau: np.ndarray
    w_t: np.ndarray
    window: Optional[List[float]]
    levels: int
    lines: List[float]
    tau: np.ndarray = field(default_factory=lambda: np.array([]))
    t: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class OutputCtx:
    tw: np.ndarray
    stem: str
    outdir: str
    ptag: str
    post_path: str
    plot_paths: List[str]
    sidecar: str
    tw_indices: List[int]


@dataclass
class PrecheckResult:
    """Skip/stale decision and output paths — computed once per input file."""

    should_skip: bool
    skip_msg: str
    stale: bool
    out_ctx: OutputCtx


@dataclass
class ProcessNeeds:
    stem: str
    outdir: str
    ptag: str
    out_h5: str
    sidecar: str
    tw_indices: List[int]
    plot_paths: List[str]
    need_h5: bool
    need_plot: bool
    tw_compute: List[int]
