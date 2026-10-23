"""flu-post: post-process a `heom_solver fluorescence` HDF5 file."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings
from dataclasses import replace
from typing import Optional

from heom_tools.absorption.post.core import COMPLETENESS_WARN_TOL, compute_absorption_post
from heom_tools.absorption.post.io import is_rank_shard_filename, print_source_summary
from heom_tools.common.spectrum import (
    basename_noext,
    load_spectrum_h5,
    processed_absorption_h5_path,
)

from .core import FluorescencePostResult, compute_fluorescence_post, write_fluorescence_h5
from .io import (
    discover_unprocessed_fluorescence_h5_flat,
    ensure_base_merged,
    fold_rank_groups,
    resolve_frequency_window,
)
from .kms import fit_kms, require_same_physics

_PROCESSING_ERRORS = (OSError, ValueError, RuntimeError, KeyError, TypeError)

_DOC = """Post-processing for `heom_solver fluorescence ...` HDF5 output.

Default output: <input_dir>/<input_base>_spectrum.h5 (+ PNGs).

With no H5 argument, scans the current directory for raw fluorescence *.h5
without a sibling *_spectrum.h5 (MC rank shards are merged first).
--absorption <abs.h5> adds the KMS check: ln(I_F/I_A) fitted on the overlap
gives T_eff (the two files must share system/bath/filter).

Run: flu-post
     flu-post <fl.h5> [--absorption <abs.h5>] [--omega3]
"""


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=_DOC)
    p.add_argument("h5", nargs="?", default=None, metavar="H5",
                   help=".h5 written by `heom_solver fluorescence ...` (omit to scan cwd)")
    p.add_argument("--absorption", metavar="ABS_H5", default=None,
                   help="raw `heom_solver absorption` .h5 of the same system: KMS fit")
    p.add_argument("--omega3", action="store_true",
                   help="also write S_em = (omega/omega_ref)^3 I_F (absolute omega)")
    p.add_argument("--wmin", type=float, default=None, help="lower frequency bound (cm^-1)")
    p.add_argument("--wmax", type=float, default=None, help="upper frequency bound (cm^-1)")
    p.add_argument("--w0", type=float, default=None, metavar="CM",
                   help="override w0_cm for FFT frequency axis only")
    p.add_argument("--no-plots", action="store_true", help="skip PNG / Tk plotting")
    p.add_argument("--show", action="store_true", help="open Tk windows after saving PNGs")
    args = p.parse_args(argv)
    if args.absorption and args.h5 is None:
        p.error("--absorption pairs with one fluorescence H5; pass it explicitly")
    return args


def _load(path: str, task: str, w0: Optional[float]):
    spec = load_spectrum_h5(path, task=task)
    return replace(spec, w0_cm=float(w0)) if w0 is not None else spec


def _print_summary(result: FluorescencePostResult) -> None:
    lin, pops, spec = result.linear, result.populations, result.linear.spec
    if spec.equilibration is not None:
        eq = spec.equilibration
        print(f"[equilibration] t_eq = {eq.t_eq_fs:.1f} fs (tolerance {eq.tolerance:.1e}, "
              f"{eq.delta.size} checks)")
    print("[peaks]")
    for x, y in lin.peaks_total or []:
        print(f"   omega = {x:9.2f} cm^-1    I_norm = {y:.4e}")
    if not lin.peaks_total:
        print("   (no peaks above threshold)")
    print(f"[populations] (stick weights: {pops.stick_weights})")
    print("   exc   omega_cm     HEOM      Boltz(H_eff)  Boltz(H_S)")
    for k, st in enumerate(lin.sticks):
        heom = f"{pops.heom[k]:.5f}" if pops.heom is not None else "   —   "
        print(f"   {k + 1:3d}  {st.omega_cm:9.2f}   {heom}   {pops.boltz_heff[k]:.5f}"
              f"       {pops.boltz_hs[k]:.5f}")
    if result.kms is not None:
        k = result.kms
        print(f"[kms] T_eff = {k.T_eff_K:.3f} K  (T = {k.T_K:g} K, rel. error "
              f"{k.rel_error:.3e}, {k.omega_cm.size} points above {k.floor:g})")


def _plots(args, result: FluorescencePostResult, plot_dir: str, out_path: str) -> None:
    if args.no_plots:
        print(f"[flu-post] wrote {out_path} (plots skipped by --no-plots)")
        return
    from heom_tools.absorption.post.plot import write_plots
    from heom_tools.fluorescence.plot.plot import (
        EmissionView,
        show_emission_plot,
        write_emission_plot,
    )
    base = basename_noext(result.linear.spec.filename)
    try:
        write_plots(result.linear, plot_dir=plot_dir, base=base, experiments=None)
        view = EmissionView.from_result(result, base)
        write_emission_plot(view, os.path.join(plot_dir, f"{base}_fl.png"))
    except ImportError as e:
        print(f"[flu-post] matplotlib import failed; HDF5 output is intact: {out_path}",
              file=sys.stderr)
        warnings.warn(f"plotting skipped: {e}", UserWarning)
        return
    if args.show:
        show_emission_plot(view)


def run_one(h5_path: str, args: argparse.Namespace) -> int:
    try:
        spec = _load(h5_path, "fluorescence", args.w0)
        wmin, wmax = resolve_frequency_window(args.wmin, args.wmax, spec)
        out_path = processed_absorption_h5_path(h5_path)
        print(f"[source] {os.path.basename(h5_path)}")
        print(f"  input = {os.path.abspath(h5_path)}")
        print(f"  output = {out_path}")
        print_source_summary(h5_path)

        result = compute_fluorescence_post(spec, wmin, wmax, omega3=args.omega3,
                                           strict_ground_w0=args.w0 is None)
        if args.absorption:
            require_same_physics(h5_path, args.absorption)
            a = compute_absorption_post(_load(args.absorption, "absorption", args.w0),
                                        wmin, wmax, strict_ground_w0=args.w0 is None)
            result.kms = fit_kms(result.linear.omega_cm, result.linear.intensity_total,
                                 a.omega_cm, a.intensity_total, spec.temperature_K)
        if result.linear.consistency_max_error > COMPLETENESS_WARN_TOL:
            warnings.warn(f"completeness error {result.linear.consistency_max_error:.3e} "
                          f"exceeds {COMPLETENESS_WARN_TOL:.1e}; output written for inspection")
        write_fluorescence_h5(result, out_path)
        _print_summary(result)
        _plots(args, result, os.path.dirname(os.path.abspath(h5_path)), out_path)
        return 0
    except _PROCESSING_ERRORS:
        print(f"[flu-post] FAILED: {h5_path}", file=sys.stderr)
        traceback.print_exc()
        return 1


def run_auto(args: argparse.Namespace) -> int:
    cwd = os.getcwd()
    fold_rank_groups(cwd)
    pending = discover_unprocessed_fluorescence_h5_flat(cwd)
    print("[flu-post auto]")
    print(f"  cwd = {cwd}")
    print(f"  unprocessed ({len(pending)}):")
    for pth in pending:
        print(f"    {pth}")
    n_fail = 0
    for i, pth in enumerate(pending):
        if i:
            print()
        n_fail += run_one(pth, args)
    return 1 if n_fail else 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.h5 is None:
        return run_auto(args)
    h5 = ensure_base_merged(args.h5) if is_rank_shard_filename(args.h5) else args.h5
    return run_one(h5, args)
