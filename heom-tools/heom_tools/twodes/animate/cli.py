"""CLI for 2DES waiting-time animation export."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings
from typing import Optional

from heom_tools.twodes.plot.cli_common import (
    add_levels_arg,
    add_norm_args,
    add_spectrum_arg,
    add_tw_arg,
    add_window_arg,
    animation_output_path,
    build_norm_config,
    format_wrote,
    parse_window_arg,
    print_norm_log,
    resolve_animation_timing,
)
from heom_tools.twodes.plot.core import build_animation_request, tw_summary
from heom_tools.twodes.plot.plot import write_animation

_EPILOG = """\
examples:
  %(prog)s sim_ap_post.h5 --format gif
  %(prog)s sim_ap_post.h5 --format mp4 --fps 4
  %(prog)s sim_ap_post.h5 --format gif --interval 300
  %(prog)s sim_ap_post.h5 --format gif --norm diagonal
  %(prog)s sim_ap_post.h5 --format gif --tw 0 500 1000 6000

  python -m heom_tools.twodes.animate sim_ap_post.h5 --format gif

output (beside POST_H5):
  <stem>_<spectrum>_anim.mp4 / <stem>_<spectrum>_anim.gif
"""

_DESCRIPTION = """\
Export waiting-time animations from twodes-post *_post.h5 files.

Input: frequency-domain HDF5 from twodes-post. Uses the same colorbar policy as
twodes-plot (shared raw min/max across T, or --norm diagonal per frame).
Needs at least two waiting times (omit --tw to use all tw_fs in the file).

MP4 export requires ffmpeg on PATH.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="twodes-animate",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument(
        "h5",
        metavar="POST_H5",
        help="post-processed HDF5 from twodes-post",
    )
    p.add_argument(
        "--format",
        choices=("mp4", "gif"),
        required=True,
        metavar="FMT",
        help="animation container format",
    )
    add_tw_arg(p)
    add_window_arg(p)
    add_levels_arg(p)
    add_spectrum_arg(p)
    add_norm_args(p)
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        metavar="N",
        help="frames per second (default: 5; ignored if --interval set)",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=None,
        metavar="MS",
        help="milliseconds per frame (overrides --fps; e.g. 300 ≈ 3.3 fps)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"twodes-animate: {exc}", file=sys.stderr)
        return 2

    h5 = os.path.abspath(args.h5)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            window = parse_window_arg(args.window)
            norm_config = build_norm_config(args)
            norm_config.validate()
            spectrum = str(args.spectrum)
            anim = build_animation_request(
                h5, args.tw, window, spectrum=spectrum,
            )
            if len(anim.selection) < 2:
                raise ValueError(
                    "animation needs at least 2 waiting times "
                    "(omit --tw to use all tw_fs in the file, or pass multiple --tw values)",
                )
            out = animation_output_path(h5, args.format, spectrum=spectrum)
            play_fps, frame_ms = resolve_animation_timing(args.fps, args.interval)
            norm_log = write_animation(
                anim, out, fmt=args.format, fps=play_fps,
                interval_ms=frame_ms, levels=args.levels, window=window,
                norm_config=norm_config,
            )
            print(
                f"twodes-animate: spectrum={spectrum}  "
                f"animation T={tw_summary(anim)} fs "
                f"({len(anim.selection)} frames, {play_fps:g} fps, {frame_ms} ms/frame)",
            )
            print_norm_log("twodes-animate", norm_log)
            print(format_wrote(out))
            print("note: output is beside the input POST_H5 (not the shell cwd)")
            for w in caught:
                if issubclass(w.category, UserWarning):
                    print(f"warning: {w.message}", file=sys.stderr)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"twodes-animate: {exc}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1
