"""CLI for single-run population dynamics plotting (console: ``pop-plot``)."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt

from .plot import plot_single_population

_DOC = """Population dynamics plotter for heom_solver HDF5 exports (single run).

Examples:
    pop-plot file.h5
    pop-plot file.h5 --time-limit 2000
    pop-plot file.h5 --verbose
    pop-plot --batch data/ --output-dir plots/
    python -m heom_tools.population file.h5

For two-run comparison use pop-compare:
    pop-compare run_a.h5 run_b.h5
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
        help="exactly one dynamics HDF5; omit with --batch",
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=None,
        metavar="FS",
        help="upper time bound in fs (e.g. 2000)",
    )
    p.add_argument(
        "--batch",
        nargs="?",
        const=".",
        default=None,
        metavar="DIR",
        help="batch-plot all .h5/.hdf5 in DIR (default: cwd)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="output directory for --batch (default: same as input)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="save figures without opening interactive windows",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print HDF5 parameter / sink analysis when loading",
    )
    return p.parse_args(argv)


def batch_plot_directory(
    directory: str,
    output_dir: Optional[str] = None,
    time_limit: Optional[float] = None,
) -> int:
    if not os.path.isdir(directory):
        print(f"错误: '{directory}' 不是有效目录", file=sys.stderr)
        return 1

    h5_files = sorted(
        f
        for f in os.listdir(directory)
        if f.endswith(".h5") or f.endswith(".hdf5")
    )

    if not h5_files:
        print(f"目录 '{directory}' 下未找到 .h5/.hdf5 文件")
        return 0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")

    original_backend = plt.get_backend()
    plt.switch_backend("Agg")

    print(f"批量绘图: 在 '{directory}' 下发现 {len(h5_files)} 个 HDF5 文件")
    if time_limit is not None:
        print(f"时间范围: 0-{time_limit} fs")
    print("-" * 60)

    success, failed = 0, 0
    try:
        for i, fname in enumerate(h5_files, 1):
            filepath = os.path.join(directory, fname)
            base_name = os.path.splitext(fname)[0]
            save_dir = output_dir if output_dir else directory
            out_path = os.path.join(save_dir, f"{base_name}_dual_basis_plot.png")

            print(f"[{i}/{len(h5_files)}] {fname} ... ", end="", flush=True)
            try:
                plot_single_population(
                    filepath,
                    output_filename=out_path,
                    time_limit=time_limit,
                    show=False,
                    verbose=False,
                )
                print("ok")
                success += 1
            except Exception as e:
                print(f"失败: {e}")
                failed += 1
    finally:
        plt.switch_backend(original_backend)

    print("-" * 60)
    print(f"批量绘图完成: {success} 成功, {failed} 失败")
    return 1 if failed else 0


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)
    show = not args.no_show

    if args.batch is not None:
        if args.files:
            print("错误: --batch 不能与 FILE 参数同时使用", file=sys.stderr)
            return 1
        return batch_plot_directory(args.batch, args.output_dir, args.time_limit)

    if not args.files:
        print(_DOC)
        print("错误: 请提供恰好一个文件，或使用 --batch", file=sys.stderr)
        return 1

    if len(args.files) >= 2:
        print(
            "错误: pop-plot 只画单次运行；对比请用: "
            f"pop-compare {args.files[0]} {args.files[1]}",
            file=sys.stderr,
        )
        return 1

    try:
        plot_single_population(
            args.files[0],
            time_limit=args.time_limit,
            show=show,
            verbose=args.verbose,
        )
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    return 0
