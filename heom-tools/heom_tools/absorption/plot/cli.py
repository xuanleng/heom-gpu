"""Single-spectrum LAS plot CLI (console: ``las-plot``)."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from .run import run_las_pipeline

_DOC = """Plot one linear absorption spectrum (optional experiment overlay).

Examples:
  las-plot run/a_spectrum.h5
  las-plot run/a_spectrum.h5 --experiment exp.csv
  python -m heom_tools.absorption.plot run/a_spectrum.h5

For multi-theory comparison use las-compare:
  las-compare run/a_spectrum.h5 run/b_spectrum.h5
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
        help="exactly one *_spectrum.h5 (legacy *_post.h5 also accepted)",
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
        help="do not auto-discover experiment files next to the .h5",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if len(args.h5) != 1:
        print(_DOC)
        if len(args.h5) >= 2:
            print(
                "错误: las-plot 只画单条理论谱；对比请用: "
                f"las-compare {' '.join(args.h5[:2])}",
                file=sys.stderr,
            )
        else:
            print("错误: las-plot 需要恰好一个 *_spectrum.h5", file=sys.stderr)
        return 1

    return run_las_pipeline(
        args.h5,
        cmd_name="las-plot",
        output_prefix="las_plot",
        title="linear absorption",
        experiment_path=args.experiment,
        auto_experiment=not args.no_auto_experiment,
        no_config_diff=True,
    )
