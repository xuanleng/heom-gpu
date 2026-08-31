"""CLI for unified absorption HDF5 post-processing."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings
from dataclasses import replace
from typing import Optional

try:
    import h5py  # noqa: F401
except ImportError as e:
    raise SystemExit("las-post requires h5py: pip install h5py") from e

from .core import (
    COMPLETENESS_WARN_TOL,
    AbsorptionPostResult,
    compute_absorption_post,
    write_absorption_h5,
)
from .io import (
    discover_unprocessed_absorption_h5_flat,
    count_dynamics_h5_flat,
    ensure_base_merged,
    fold_rank_groups,
    is_rank_shard_filename,
    print_source_summary,
    processed_absorption_h5_path,
    resolve_frequency_window,
)
from .plot import show_absorption_plots_interactive, write_plots
from heom_tools.common.spectrum import (
    ExperimentSpectrum,
    basename_noext,
    discover_experiment_spectra_auto,
    load_spectrum_h5,
)

_PROCESSING_ERRORS = (OSError, ValueError, RuntimeError, KeyError, TypeError)

_DOC = """Unified post-processing for `heom_solver absorption ...` HDF5 output.

Default output: <input_dir>/<input_base>_spectrum.h5

With no H5 argument, scans the current directory for raw absorption *.h5 files
without sibling *_spectrum.h5 and post-processes each one (MC rank shards are
merged first). Run from the data directory, or pass an explicit H5 path.

Run: las-post
     las-post <input.h5>
     python -m heom_tools.absorption.post <input.h5>
"""


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_DOC.split("\n\n")[0],
        epilog=(
            "Writes analysis HDF5 and PNG figures next to the input .h5 "
            "(combined absorption spectrum PNG + time-domain correlation PNG)."
        ),
    )
    p.add_argument(
        "h5",
        nargs="?",
        default=None,
        metavar="H5",
        help=".h5 written by `heom_solver absorption ...` (omit to scan cwd)",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--wmin", type=float, default=None, help="lower frequency bound (cm^-1)")
    p.add_argument("--wmax", type=float, default=None, help="upper frequency bound (cm^-1)")
    p.add_argument(
        "--w0",
        type=float,
        default=None,
        metavar="CM",
        help="override w0_cm for FFT frequency axis only",
    )
    p.add_argument("--no-plots", action="store_true", help="skip PNG / Tk plotting")
    p.add_argument("--show", action="store_true", help="open Tk windows after saving PNGs")
    args = p.parse_args(argv)
    if args.auto and args.h5:
        p.error("cannot combine --auto with H5 path")
    return args


def _print_experiment_overlays(experiments: list[ExperimentSpectrum]) -> None:
    if not experiments:
        return
    print("[experimental overlays] (auto-discovered, max-normalized)")
    for ex in experiments:
        print(f"   {ex.label}")


def _run_one_compute_and_write(
    h5_path: str,
    args: argparse.Namespace,
    *,
    experiments: Optional[list[ExperimentSpectrum]] = None,
    print_experiments: bool = True,
) -> tuple[AbsorptionPostResult, str, str, list[ExperimentSpectrum]]:
    spec = load_spectrum_h5(h5_path)
    w0_override = args.w0 is not None
    if w0_override:
        spec = replace(spec, w0_cm=float(args.w0))
    omega_min, omega_max = resolve_frequency_window(args.wmin, args.wmax, spec)

    abs_h5 = os.path.abspath(h5_path)
    plot_dir = os.path.dirname(abs_h5)
    out_path = processed_absorption_h5_path(h5_path)

    print(f"[source] {os.path.basename(h5_path)}")
    print(f"  input = {abs_h5}")
    print(f"  output = {out_path}")
    print_source_summary(h5_path)

    result = compute_absorption_post(
        spec,
        wmin_cm=omega_min,
        wmax_cm=omega_max,
        strict_ground_w0=not w0_override,
    )

    if result.consistency_max_error > COMPLETENESS_WARN_TOL:
        warnings.warn(
            f"completeness error {result.consistency_max_error:.3e} exceeds "
            f"{COMPLETENESS_WARN_TOL:.1e}; output is written for inspection"
        )

    write_absorption_h5(result, out_path)

    print("[peaks]")
    if result.peaks_total:
        for x, y in result.peaks_total:
            print(f"   omega = {x:9.2f} cm^-1    I_norm = {y:.4e}")
    else:
        print("   (no peaks above prominence threshold)")

    print("[stick spectrum]")
    for k, st in enumerate(result.sticks, 1):
        print(
            f"   exc {k}: omega = {st.omega_cm:9.2f} cm^-1  "
            f"|mu|^2 = {st.oscillator_strength:.4e}  "
            f"site {st.dominant_site_index + 1} ({st.dominant_site_fraction * 100:.1f}%)"
        )

    if experiments is None:
        experiments = discover_experiment_spectra_auto(plot_dir)
    if print_experiments:
        _print_experiment_overlays(experiments)

    return result, out_path, plot_dir, experiments


def _run_one_maybe_plot(
    args: argparse.Namespace,
    result: AbsorptionPostResult,
    out_path: str,
    plot_dir: str,
    experiments: list[ExperimentSpectrum],
) -> int:
    if args.no_plots:
        if args.show:
            warnings.warn("--show ignored when --no-plots is set", UserWarning)
        print(f"[las-post] wrote {out_path} (plots skipped by --no-plots)")
        return 0

    exp_overlays = experiments if experiments else None
    try:
        write_plots(
            result,
            plot_dir=plot_dir,
            base=basename_noext(result.spec.filename),
            experiments=exp_overlays,
            normalize=True,
            show_sticks=True,
        )
    except ImportError as e:
        print(
            "[las-post] matplotlib/import failed — PNG figures were not written.\n"
            f"  HDF5 output is intact: {out_path}\n"
            '  Fix A: upgrade numpy: python3 -m pip install -U "numpy>=1.23"\n'
            "  Fix B: matplotlib<3.9 with older numpy, or use a dedicated venv.\n"
            "  Fix C: rerun with --no-plots for HDF5-only.",
            file=sys.stderr,
        )
        warnings.warn(f"plotting skipped: {e}", UserWarning)
        return 0

    if args.show:
        try:
            print("[las-post] close all plot windows to continue …")
            show_absorption_plots_interactive(
                result,
                experiments=exp_overlays,
            )
        except ImportError as e:
            print(
                "[las-post] could not open interactive windows (--show).\n"
                "  Omit --show if no GUI backend.\n",
                file=sys.stderr,
            )
            warnings.warn(f"--show failed: {e}", UserWarning)

    return 0


def run_one(
    h5_path: str,
    args: argparse.Namespace,
    *,
    experiments: Optional[list[ExperimentSpectrum]] = None,
    print_experiments: bool = True,
) -> int:
    try:
        result, out_path, plot_dir, experiments = _run_one_compute_and_write(
            h5_path,
            args,
            experiments=experiments,
            print_experiments=print_experiments,
        )
        return _run_one_maybe_plot(args, result, out_path, plot_dir, experiments)
    except _PROCESSING_ERRORS:
        print(f"[las-post] FAILED: {h5_path}", file=sys.stderr)
        traceback.print_exc()
        return 1


def run_auto(args: argparse.Namespace) -> int:
    cwd = os.getcwd()
    fold_rank_groups(cwd)
    pending = discover_unprocessed_absorption_h5_flat(cwd)
    print("[las-post auto]")
    print(f"  cwd = {cwd}")
    print(f"  unprocessed ({len(pending)}):")
    for pth in pending:
        print(f"    {pth}")
    if not pending:
        n_dyn = count_dynamics_h5_flat(cwd)
        if n_dyn:
            print(
                "[las-post auto] no raw absorption .h5 to process.",
                flush=True,
            )
            print(
                f"  found {n_dyn} dynamics HDF5 file(s) in this directory "
                f"(task=dynamics).",
                flush=True,
            )
            print(
                "  For population plots:  pop-plot --batch .",
                flush=True,
            )
            print(
                "  For absorption post:   run `heom_solver absorption ...` first, "
                "then las-post here.",
                flush=True,
            )
        return 0
    batch_experiments = discover_experiment_spectra_auto(cwd)
    _print_experiment_overlays(batch_experiments)
    n_fail = 0
    for i, pth in enumerate(pending):
        if i > 0:
            print()
        n_fail += run_one(
            pth,
            args,
            experiments=batch_experiments,
            print_experiments=False,
        )
    return 1 if n_fail else 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.h5 is None:
        return run_auto(args)
    h5 = ensure_base_merged(args.h5) if is_rank_shard_filename(args.h5) else args.h5
    return run_one(h5, args)
