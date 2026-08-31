"""CLI for static 2DES post HDF5 plotting."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings
from typing import Optional

from .cli_common import (
    add_levels_arg,
    add_norm_args,
    add_spectrum_arg,
    add_tw_arg,
    add_window_arg,
    build_norm_config,
    format_wrote,
    parse_window_arg,
    print_norm_log,
    series_output_path,
    static_output_path,
)
from .core import (
    SinglePlotRequest,
    SliceSeriesRequest,
    build_plot_request,
    tw_summary,
)
from .plot import write_plot_png

_EPILOG = """\
examples:
  %(prog)s sim_ap_post.h5 --tw 500
  %(prog)s sim_ap_post.h5 --tw 0 500 1000
  %(prog)s sim_ap_post.h5 --tw 0 500 1000 --spectrum rp-gsb
  %(prog)s sim_ap_post.h5 --tw 0 500 1000 --norm diagonal --norm-ref-freq 12350

  python -m heom_tools.twodes.plot sim_ap_post.h5 --tw 500

output (beside POST_H5):
  <stem>_<spectrum>_<T>fs.png
  <stem>_<spectrum>_<T0>_<T1>_...png   (multiple --tw, shared colorbar)
"""

_DESCRIPTION = """\
Plot Re(spectrum) PNGs from twodes-post *_post.h5 files.

Input: frequency-domain HDF5 from twodes-post (run twodes-post on solver .h5 first).
One POST_H5 → single panel, or multiple --tw → 1×N row with shared colorbar.
Default: raw data, shared min/max colorbar across T (white at 0 when 0 is in range).
--norm diagonal: per-T diagonal reference.
--tw must match tw_fs exactly (omit when the file has only one waiting time).

For waiting-time animations use twodes-animate.
For A/B comparison use twodes-compare.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="twodes-plot",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument(
        "h5",
        metavar="POST_H5",
        help="post-processed HDF5 from twodes-post",
    )
    add_tw_arg(p)
    add_window_arg(p)
    add_levels_arg(p)
    add_spectrum_arg(p)
    add_norm_args(p)
    return p


def _output_path(request, h5_path: str) -> str:
    spectrum = request.meta.spectrum
    if isinstance(request, SinglePlotRequest):
        return static_output_path(
            h5_path, spectrum=spectrum, tw_fs=request.slice.tw_fs,
        )
    if isinstance(request, SliceSeriesRequest):
        return series_output_path(
            h5_path,
            spectrum=spectrum,
            tw_values=[v for _, v in request.selection],
        )
    raise TypeError(f"unsupported request type: {type(request)!r}")


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"twodes-plot: {exc}", file=sys.stderr)
        return 2

    h5 = os.path.abspath(args.h5)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            window = parse_window_arg(args.window)
            norm_config = build_norm_config(args)
            norm_config.validate()
            spectrum = str(args.spectrum)
            request = build_plot_request(h5, args.tw, window, spectrum=spectrum)
            out = _output_path(request, h5)
            norm_log = write_plot_png(
                request, out, levels=args.levels, window=window,
                norm_config=norm_config,
            )
            print(f"twodes-plot: spectrum={spectrum}  T={tw_summary(request)} fs")
            print_norm_log("twodes-plot", norm_log)
            print(format_wrote(out))
            print("note: output is beside the input POST_H5 (not the shell cwd)")
            for w in caught:
                if issubclass(w.category, UserWarning):
                    print(f"warning: {w.message}", file=sys.stderr)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"twodes-plot: {exc}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1
