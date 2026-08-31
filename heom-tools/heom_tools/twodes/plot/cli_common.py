"""Shared argparse helpers for twodes plot CLIs."""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

from heom_tools.twodes.post.config import parse_window, tw_tag
from heom_tools.twodes.post.h5_spectra import H5_SPECTRA_TOKENS

from .core import PlotNormConfig


def parse_window_arg(values: Optional[List[float]]) -> Optional[List[float]]:
    if values is None:
        return None
    return parse_window(list(values), validate=True)


def spectrum_filename_token(spectrum: str) -> str:
    return str(spectrum).strip().lower()


def build_norm_config(args: argparse.Namespace) -> PlotNormConfig:
    return PlotNormConfig(
        diagonal=getattr(args, "norm", None) == "diagonal",
        ref_freq=getattr(args, "norm_ref_freq", None),
    )


def format_wrote(path: str) -> str:
    abspath = os.path.abspath(path)
    try:
        nbytes = os.path.getsize(abspath)
    except OSError:
        return f"wrote {abspath}"
    if nbytes >= 1 << 20:
        size = f"{nbytes / (1 << 20):.1f} MB"
    elif nbytes >= 1 << 10:
        size = f"{nbytes / (1 << 10):.1f} KB"
    else:
        size = f"{nbytes} B"
    return f"wrote {abspath} ({size})"


def print_norm_log(prefix: str, lines: List[str]) -> None:
    for line in lines:
        print(f"{prefix}: {line}")


def add_tw_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--tw",
        type=float,
        nargs="+",
        default=None,
        metavar="FS",
        help="waiting time(s) in fs; exact tw_fs match (omit if single T in file)",
    )


def add_window_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="frequency window cm^-1 for both axes (default: HDF5 parameters.window)",
    )


def add_levels_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--levels",
        type=int,
        default=None,
        metavar="N",
        help="contour levels (default: HDF5 parameters.levels)",
    )


def add_spectrum_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--spectrum",
        choices=sorted(H5_SPECTRA_TOKENS),
        default="2des",
        metavar="TOKEN",
        help=(
            "spectrum component (default: 2des); must be stored in post HDF5 "
            "(twodes_post h5_spectra tokens)"
        ),
    )


def add_norm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--norm",
        choices=("diagonal",),
        default=None,
        help="per-T diagonal reference (default: shared raw min/max colorbar)",
    )
    p.add_argument(
        "--norm-ref-freq",
        type=float,
        default=None,
        metavar="CM1",
        help="with --norm diagonal: reference frequency cm^-1 (nearest ω_τ≈ω_t grid point)",
    )


def static_output_path(h5_path: str, *, spectrum: str, tw_fs: float) -> str:
    stem = os.path.splitext(os.path.basename(h5_path))[0]
    outdir = os.path.dirname(os.path.abspath(h5_path)) or "."
    spec = spectrum_filename_token(spectrum)
    return os.path.join(outdir, f"{stem}_{spec}_{tw_tag(tw_fs)}.png")


def series_output_path(h5_path: str, *, spectrum: str, tw_values: List[float]) -> str:
    stem = os.path.splitext(os.path.basename(h5_path))[0]
    outdir = os.path.dirname(os.path.abspath(h5_path)) or "."
    spec = spectrum_filename_token(spectrum)
    tags = "_".join(tw_tag(v) for v in tw_values)
    return os.path.join(outdir, f"{stem}_{spec}_{tags}.png")


def animation_output_path(h5_path: str, fmt: str, *, spectrum: str = "2des") -> str:
    stem = os.path.splitext(os.path.basename(h5_path))[0]
    outdir = os.path.dirname(os.path.abspath(h5_path)) or "."
    spec = spectrum_filename_token(spectrum)
    return os.path.join(outdir, f"{stem}_{spec}_anim.{fmt}")


def compare_output_path(
    path_a: str,
    path_b: str,
    *,
    spectrum: str,
    tw_fs: float,
) -> str:
    stem_a = os.path.splitext(os.path.basename(path_a))[0]
    stem_b = os.path.splitext(os.path.basename(path_b))[0]
    outdir = os.path.dirname(os.path.abspath(path_a)) or "."
    spec = spectrum_filename_token(spectrum)
    return os.path.join(
        outdir,
        f"twodes_compare_{spec}_{stem_a}_vs_{stem_b}_{tw_tag(tw_fs)}.png",
    )


def resolve_animation_timing(
    fps: Optional[float],
    interval_ms: Optional[float],
) -> tuple[float, int]:
    if interval_ms is not None:
        ms = float(interval_ms)
        if ms <= 0.0:
            raise ValueError("--interval must be positive (milliseconds)")
        return 1000.0 / ms, int(round(ms))
    play_fps = 5.0 if fps is None else float(fps)
    if play_fps <= 0.0:
        raise ValueError("--fps must be positive")
    return play_fps, int(round(1000.0 / play_fps))
