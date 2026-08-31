"""CLI for 2DES spectrum comparison plots."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings
from typing import Optional

from heom_tools.twodes.plot.cli_common import (
    add_levels_arg,
    add_spectrum_arg,
    add_tw_arg,
    add_window_arg,
    build_norm_config,
    compare_output_path,
    format_wrote,
    parse_window_arg,
)
from heom_tools.twodes.plot.core import build_compare_request, tw_summary
from heom_tools.twodes.plot.plot import write_plot_png

_EPILOG = """\
examples:
  %(prog)s run_a_ap_post.h5 run_b_ap_post.h5 --tw 500
  %(prog)s run_a_ap_post.h5 run_b_ap_post.h5 --spectrum rp-gsb

  python -m heom_tools.twodes.compare a_ap_post.h5 b_ap_post.h5 --tw 500

output (beside first POST_H5):
  twodes_compare_<spectrum>_<a>_vs_<b>_<T>fs.png
"""

_DESCRIPTION = """\
Compare two twodes-post *_post.h5 files at one waiting time.

Produces a 1×3 panel: spectrum A, spectrum B, raw difference (A − B).
Exactly one --tw is allowed (omit when each file has a single waiting time).
If frequency grids differ, the second spectrum is resampled onto the first.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="twodes-compare",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument(
        "h5_a",
        metavar="POST_H5_A",
        help="first post-processed HDF5",
    )
    p.add_argument(
        "h5_b",
        metavar="POST_H5_B",
        help="second post-processed HDF5",
    )
    add_tw_arg(p)
    add_window_arg(p)
    add_levels_arg(p)
    add_spectrum_arg(p)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"twodes-compare: {exc}", file=sys.stderr)
        return 2

    path_a = os.path.abspath(args.h5_a)
    path_b = os.path.abspath(args.h5_b)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            if args.tw is not None and len(args.tw) != 1:
                raise ValueError("compare accepts exactly one --tw value")
            window = parse_window_arg(args.window)
            norm_config = build_norm_config(args)
            if norm_config.diagonal:
                raise ValueError("--norm diagonal is not supported in compare mode")
            norm_config.validate()
            spectrum = str(args.spectrum)
            request = build_compare_request(
                path_a, path_b, args.tw, window, spectrum=spectrum,
            )
            out = compare_output_path(
                path_a, path_b,
                spectrum=spectrum,
                tw_fs=request.compare.a.tw_fs,
            )
            norm_log = write_plot_png(
                request, out, levels=args.levels, window=window,
                norm_config=norm_config,
            )
            print(f"twodes-compare: spectrum={spectrum}  T={tw_summary(request)} fs")
            if request.compare.resampled_b:
                print("note: resampled second spectrum onto first frequency grid")
            print(format_wrote(out))
            print("note: output is beside the first POST_H5 (not the shell cwd)")
            for w in caught:
                if issubclass(w.category, UserWarning):
                    print(f"warning: {w.message}", file=sys.stderr)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"twodes-compare: {exc}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1
