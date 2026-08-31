"""CLI for two-run population comparison (console: ``pop-compare``)."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from .plot_compare import compare_populations

_DOC = """Compare two heom_solver population dynamics HDF5 exports.

Baseline is the first file (1); subject is the second (2).
Difference panels show Δ = (2 − 1) and a global L² over all times × sites.

Examples:
    pop-compare run_a.h5 run_b.h5
    pop-compare run_a.h5 run_b.h5 --time-limit 2000 --no-show
    pop-compare run_a.h5 run_b.h5 --verbose

For a single-run plot use pop-plot:
    pop-plot run_a.h5
"""


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_DOC.split("\n\n")[0],
        epilog=_DOC.split("\n\n", 1)[1],
    )
    p.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="exactly two dynamics HDF5 files (baseline, subject)",
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=None,
        metavar="FS",
        help="upper time bound in fs (e.g. 2000)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="save figures without opening interactive windows",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print detailed per-site comparison tables",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)
    show = not args.no_show

    if len(args.files) != 2:
        print(_DOC)
        if len(args.files) == 1:
            print(
                "错误: pop-compare 需要恰好两个文件；单次运行请用: "
                f"pop-plot {args.files[0]}",
                file=sys.stderr,
            )
        elif not args.files:
            print("错误: pop-compare 需要恰好两个文件", file=sys.stderr)
        else:
            print(
                f"错误: pop-compare 需要恰好两个文件，收到 {len(args.files)} 个",
                file=sys.stderr,
            )
        return 1

    try:
        compare_populations(
            args.files[0],
            args.files[1],
            time_limit=args.time_limit,
            show=show,
            verbose=args.verbose,
        )
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
