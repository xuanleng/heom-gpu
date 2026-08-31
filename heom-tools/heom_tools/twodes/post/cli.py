"""CLI orchestration for 2DES post-processing."""
from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback
from typing import Dict, Optional, Tuple

from .cache import precheck, resolve_process_needs, write_sidecar
from .cmap import diverging_cmap, positive_cmap
from .config import PostConfig, find_config, load_config, parse_window, print_runtime_config
from .core import (
    build_tw_slices,
    fft_grid,
    pathway_plan,
    polarization_weights,
    stack_spectra,
)
from .io import (
    ensure_base_merged,
    fold_rank_groups,
    is_rank_shard_filename,
    is_solver_2des_h5,
    load_axes,
    write_post_h5,
)
from .panels import prune_orphan_plots, render_plots
from .types import OutputPlan, PlotCtx, PrecheckResult, TwSlice


def _import_plot_backend() -> Optional[Tuple[object, object, object, object]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        print("twodes-post: matplotlib not available — cannot plot", file=sys.stderr)
        return None
    cmap = diverging_cmap()
    return mpl, plt, cmap, positive_cmap(cmap)


def execute(
    h5: str,
    cfg: PostConfig,
    pre: PrecheckResult,
    *,
    config_path: Optional[str] = None,
    show_config: bool = True,
) -> int:
    """Run FFT/plot/H5 writes for one file (skip decision already in *pre*)."""
    if show_config:
        print_runtime_config(cfg, config_path=config_path, input_h5=h5)

    if pre.skip_msg:
        print(f"twodes-post: {pre.skip_msg}")

    if not cfg.write_h5 and not cfg.plot:
        print("twodes-post: write_h5 and plot are both false — nothing to do")
        return 0

    plot_backend = None
    if cfg.plot:
        plot_backend = _import_plot_backend()
        if plot_backend is None:
            return 1

    stale = pre.stale
    out_ctx = pre.out_ctx
    os.makedirs(out_ctx.outdir, exist_ok=True)

    pol = cfg.pol
    w0 = float(cfg.w0)
    dw = float(cfg.dw)
    window = parse_window(cfg.window)
    fft_method = cfg.fft_method
    weights = polarization_weights(pol)

    tau, tw, t, signals, pathways = load_axes(h5)
    pplan = pathway_plan(pathways)
    plan = OutputPlan.from_cfg(cfg, pathways=pathways)
    needs = resolve_process_needs(plan, stale, out_ctx)

    if not needs.tw_compute:
        print("twodes-post: all requested outputs already exist")
        return 0

    sn = plan.spectra_needs
    print(
        f"  FFT plan    : pathway_ffts={len(sn.pathway_fft)} "
        f"total={sorted(sn.total_fft)} compute_2des={sn.compute_2des}"
    )
    if cfg.write_h5:
        print(f"  H5 spectra  : {cfg.h5_spectra}")

    pidx = pplan.pidx
    n_tau, n_t, w_tau, w_t, dtau, dt, w_tau_nat, w_t_nat = fft_grid(
        tau, t, dw=dw, w0=w0,
    )

    if stale or needs.need_h5 or needs.need_plot:
        prune_orphan_plots(
            needs.outdir, needs.stem, needs.ptag, tw, needs.tw_indices, td=plan.need_time,
        )

    slices: Dict[int, TwSlice] = build_tw_slices(
        h5, weights, signals, pathways, pidx, needs.tw_compute,
        n_tau, n_t, plan, fft_method, dtau=dtau, dt=dt,
        w_tau_src=w_tau_nat, w_t_src=w_t_nat,
        w_tau_dst=w_tau, w_t_dst=w_t,
    )
    if not slices:
        raise RuntimeError("internal error: no Tw slices built")

    wrote_outputs = False
    if needs.need_h5:
        spec_by_p, spec_total = stack_spectra(
            slices, needs.tw_indices, signals,
        )
        write_post_h5(
            needs.out_h5, w_tau=w_tau, w_t=w_t, tw_vals=tw[needs.tw_indices],
            spec_by_pathway=spec_by_p, spec_total=spec_total,
            cfg=cfg, source_h5=h5, pathway=pplan.pathway_sum,
            signals=signals, pathways=pathways,
        )
        print(f"wrote {needs.out_h5}")
        wrote_outputs = True

    if needs.need_plot:
        mpl, plt, cmap, cmap_pos = plot_backend
        plot_ctx = PlotCtx(
            w_tau=w_tau, w_t=w_t, window=window,
            levels=int(cfg.levels), lines=cfg.lines or [],
            tau=tau, t=t,
        )
        render_plots(
            outdir=needs.outdir, stem=needs.stem, ptag=needs.ptag, tw=tw,
            tw_indices=needs.tw_compute, pol=pol, pathways=pathways, slices=slices,
            stale=stale, plan=plan, ctx=plot_ctx,
            mpl=mpl, plt_mod=plt, cmap=cmap, cmap_pos=cmap_pos,
        )
        wrote_outputs = True

    if wrote_outputs:
        write_sidecar(needs.outdir, needs.stem, needs.ptag, cfg, h5)
    return 0


def process(
    h5: str,
    cfg: PostConfig,
    *,
    config_path: Optional[str] = None,
    show_config: bool = True,
) -> int:
    """Precheck one file, then execute if outputs are missing or stale."""
    if is_rank_shard_filename(h5):
        h5 = ensure_base_merged(h5)
    if not is_solver_2des_h5(h5):
        print(f"twodes-post: not a solver 2DES HDF5: {h5}", file=sys.stderr)
        return 1
    pre = precheck(h5, cfg)
    if pre.should_skip:
        print(f"skip {os.path.basename(h5)}: {pre.skip_msg}")
        return 0
    try:
        return execute(h5, cfg, pre, config_path=config_path, show_config=show_config)
    except Exception as e:
        print(f"twodes-post: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


def run_auto(cfg: PostConfig, *, config_path: Optional[str] = None) -> int:
    scan = cfg.scan_dir or "."
    print_runtime_config(
        cfg,
        config_path=config_path,
        mode=f"auto (scan {os.path.abspath(scan)})",
    )
    fold_rank_groups(scan)
    candidates = sorted(glob.glob(os.path.join(scan, "*.h5")))
    candidates = [f for f in candidates if not f.endswith("_post.h5")]
    files = []
    ignored = 0
    shard_skipped = 0
    for path in candidates:
        if is_rank_shard_filename(path):
            shard_skipped += 1
            continue
        if is_solver_2des_h5(path):
            files.append(path)
        else:
            ignored += 1
    if shard_skipped:
        print(f"auto: skipped {shard_skipped} MC rank shard(s) (folded into merged files)")
    if ignored:
        print(f"auto: ignored {ignored} non-2DES .h5 file(s)")
    if not files:
        print(f"auto: no solver 2DES .h5 files in {os.path.abspath(scan)}")
        return 0
    done = skipped = failed = 0
    for h5 in files:
        pre = precheck(h5, cfg)
        if pre.should_skip:
            print(f"auto: skip {os.path.basename(h5)}: {pre.skip_msg}")
            skipped += 1
            continue
        print(f"auto: process {os.path.basename(h5)}")
        rc = process(h5, cfg, config_path=config_path, show_config=False)
        if rc != 0:
            failed += 1
        else:
            done += 1
    print(f"auto: {done} processed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


_DESCRIPTION = """\
2DES post-processing: FFT, polarization mixing, PNG plots, optional post HDF5.

Input: solver 2DES HDF5 from `heom_solver 2des ...` (results.resp, tau/t/tw axes).
Config: twodes_post.yaml (or .yml / .json) in the same directory as the data.

With no H5 argument, scans scan_dir from config (blank -> current directory)
for solver *.h5 files and post-processes each one (skips stale outputs).
Run from the data directory, or pass an explicit H5 path.
"""

_EPILOG = """\
examples:
  cd run_dir && %(prog)s              # scan cwd (twodes_post.yaml must be here)
  %(prog)s run_dir/sim_2des.h5        # config must be in run_dir/

  python -m heom_tools.twodes.post sim_2des.h5

outputs (beside input, or under config `out:`):
  <stem>_<pol>_post.h5                 when write_h5: true
  <stem>_<pol>_2d_tw<T>fs.png          when plot: true
  .<stem>_<pol>_twodes_post.json       sidecar fingerprint (skip cache)
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="twodes-post",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument(
        "h5",
        nargs="?",
        default=None,
        metavar="H5",
        help="solver 2DES HDF5 path (omit to scan scan_dir for all *.h5)",
    )
    return p


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _build_parser().parse_args(argv)
    target = args.h5
    try:
        if target is None:
            config_path = find_config(os.getcwd())
            cfg = load_config(config_path)
            return run_auto(cfg, config_path=config_path)
        if not os.path.exists(target):
            print(f"twodes-post: not found: {target}", file=sys.stderr)
            return 1
        config_path = find_config(os.path.dirname(os.path.abspath(target)))
        cfg = load_config(config_path)
        return process(target, cfg, config_path=config_path)
    except ValueError as e:
        print(f"twodes-post: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        name = os.path.basename(target) if target else "auto"
        print(f"twodes-post: FAILED {name}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
