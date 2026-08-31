"""2DES post-processing."""
from __future__ import annotations

from .cache import check_skip_outputs, precheck
from .cli import execute, main, process, run_auto
from .config import PostConfig, find_config, load_config, pol_tag, post_h5_path, print_runtime_config
from .constants import DEFAULTS, DW_DEFAULT_CM, FS_TO_CM
from .core import compute_tw_slice, polarization_weights, time_domain_combo
from .fft import (
    FFT_METHODS,
    dw_phys_cm,
    fft_time_weight,
    freq_axis_cm,
    freq_axis_dw_cm,
    nfft_len,
    project_spec_2d,
    two_d_fft,
)
from .io import from_h5_complex, load_axes, load_meta, write_post_h5
from .types import PrecheckResult, TwSlice
from heom_tools.twodes.viz import cut_window, make_norm

__all__ = [
    "DEFAULTS",
    "DW_DEFAULT_CM",
    "FFT_METHODS",
    "FS_TO_CM",
    "PostConfig",
    "PrecheckResult",
    "TwSlice",
    "check_skip_outputs",
    "compute_tw_slice",
    "cut_window",
    "dw_phys_cm",
    "execute",
    "fft_time_weight",
    "find_config",
    "freq_axis_cm",
    "freq_axis_dw_cm",
    "from_h5_complex",
    "load_axes",
    "load_config",
    "load_meta",
    "main",
    "make_norm",
    "nfft_len",
    "pol_tag",
    "polarization_weights",
    "post_h5_path",
    "precheck",
    "print_runtime_config",
    "process",
    "project_spec_2d",
    "run_auto",
    "time_domain_combo",
    "two_d_fft",
    "write_post_h5",
]
