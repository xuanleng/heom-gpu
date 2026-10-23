"""CLI for merging solver 2DES HDF5 files along waiting time."""
from __future__ import annotations

import argparse
import os
import sys
import traceback

from .core import merge_solver_2des


_DESCRIPTION = """\
Merge solver 2DES HDF5 files that share the same physics/grid but different tw_fs.

Validates /parameters, /derived, tau/t axes, and resp static dimensions against the
first input. Overlapping waiting times are compared slice-by-slice; mismatches are
reported and merge is refused unless --on-overlap keep-first|keep-last is set.

Output metadata (parameters, derived, provenance) is taken from the first file.
parameters/config_yaml 2des.tw and root config_hash are updated to the
merged waiting-time list.
"""

_EPILOG = """\
examples:
  %(prog)s -o fmo_merged.h5 run_a.h5 run_b.h5 run_c.h5
  %(prog)s -o out.h5 a.h5 b.h5 --on-overlap keep-first

then:
  twodes-post fmo_merged.h5
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="twodes-merge",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument(
        "h5",
        nargs="+",
        metavar="H5",
        help="solver 2DES HDF5 paths (first file is the metadata reference)",
    )
    p.add_argument(
        "-o", "--output",
        required=True,
        metavar="OUT",
        help="merged solver 2DES HDF5 path",
    )
    p.add_argument(
        "--on-overlap",
        choices=("fail", "keep-first", "keep-last"),
        default="fail",
        help="when the same T appears in multiple inputs with different resp (default: fail)",
    )
    p.add_argument(
        "--overlap-rtol",
        type=float,
        default=1e-9,
        metavar="RTOL",
        help="relative tolerance for duplicate-Tw resp agreement (default: 1e-9)",
    )
    return p


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    try:
        summary = merge_solver_2des(
            args.h5,
            args.output,
            on_overlap=args.on_overlap,
            overlap_rtol=float(args.overlap_rtol),
        )
        tw_text = ", ".join(f"{v:g}" for v in summary["tw_fs"])
        print(f"twodes-merge: wrote {summary['out']}")
        print(f"  sources : {', '.join(summary['sources'])}")
        print(f"  tw_fs   : [{tw_text}]  ({summary['n_tw']} points)")
        for note in summary.get("notes", []):
            print(f"  note    : {note}")
        if summary.get("overlap_conflicts"):
            print(
                f"  warning : resolved {summary['overlap_conflicts']} overlap "
                f"conflict(s) with --on-overlap {args.on_overlap}",
                file=sys.stderr,
            )
        return 0
    except ValueError as e:
        print(f"twodes-merge: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"twodes-merge: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
