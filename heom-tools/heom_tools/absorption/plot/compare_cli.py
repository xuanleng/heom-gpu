"""Multi-theory LAS comparison CLI (console: ``las-compare``)."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from .run import run_las_pipeline

_DOC = """Compare linear absorption spectra (LAS), optional experiment overlay.

Baseline is the first theory file (1); others are subjects (2, 3, …).
Residual panels show Δ = (n − 1) or (n − exp) with global L² labels.

Examples:
  las-compare run/a_spectrum.h5 run/b_spectrum.h5
  las-compare a_spectrum.h5 b_spectrum.h5 --experiment exp.csv
  python -m heom_tools.absorption.plot.compare_cli a_spectrum.h5 b_spectrum.h5

For a single-theory plot use las-plot:
  las-plot run/a_spectrum.h5
"""


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=_DOC.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_DOC.split("\n\n", 1)[1],
    )
    p.add_argument(
        "h5",
        nargs="*",
        help="two or more *_spectrum.h5 files (first = baseline; legacy *_post.h5 ok)",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default=None,
        metavar="FILE",
        help="explicit experimental 2-column file (overrides auto-discovery)",
    )
    p.add_argument(
        "--no-auto-experiment",
        action="store_true",
        help="do not auto-discover experiment files next to the first .h5",
    )
    p.add_argument(
        "--no-config-diff",
        action="store_true",
        help="hide the parameter-diff text box on the figure (default: show)",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if len(args.h5) < 2:
        print(_DOC)
        if len(args.h5) == 1:
            print(
                "错误: las-compare 需要至少两个理论谱；单条请用: "
                f"las-plot {args.h5[0]}",
                file=sys.stderr,
            )
        else:
            print("错误: las-compare 需要至少两个 *_spectrum.h5", file=sys.stderr)
        return 1

    return run_las_pipeline(
        args.h5,
        cmd_name="las-compare",
        output_prefix="las_compare",
        title="linear absorption comparison",
        experiment_path=args.experiment,
        auto_experiment=not args.no_auto_experiment,
        no_config_diff=args.no_config_diff,
    )


if __name__ == "__main__":
    raise SystemExit(main())
