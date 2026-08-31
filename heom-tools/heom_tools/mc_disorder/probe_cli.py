"""mc-probe CLI (argparse subcommands)."""
import argparse
import os
import sys
import tempfile

from .probe_absorption import run_probe
from .probe_arena import run_arena
from .probe_converge import run_converge
from .probe_rqmc import run_rqmc

def _add_ref_file(sp):
    sp.add_argument("--ref-file", default=None, dest="ref_file",
                    help="external gold-reference spectrum .h5 (e.g. disorder-in-"
                         "HEOM L20); cross-scheme → peak-normalised shape compare")


def _add_iid_ref(sp):
    sp.add_argument("--ref-n", type=int, default=512, dest="ref_n",
                    help="INDEPENDENT iid reference node count. rqmc --obs 2des: "
                         ">0 = independent iid ref (expensive, honest); <=0 = SELF "
                         "combined grand-mean reference (free de-risk)")
    sp.add_argument("--ref-seed", type=int, default=99999, dest="ref_seed",
                    help="reference iid seed (must differ from --seed)")


def _add_sweep(sp):
    sp.add_argument("--sweep", type=int, nargs="+", default=[8, 16, 32, 64, 128],
                    metavar="N", help="node counts to sweep")


def _add_cv_width(sp):
    sp.add_argument("--cv-width", type=float, default=60.0, dest="cv_width",
                    help="stick-surrogate Gaussian width (cm⁻¹); efficiency knob "
                         "only, CV is unbiased for any width")


def _add_pool_file(sp):
    sp.add_argument("--pool-file", default=None, dest="pool_file",
                    help="HDF5 pool cache; load if it exists (free re-analysis), "
                         "else run the solver once and save it")


def _build_parser():
    # Parent: physics/grid/precision knobs every subcommand needs.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("config", help="base absorption config .yaml (FMO)")
    common.add_argument("--fwhm", type=float, required=True,
                        help="global per-site disorder FWHM (cm⁻¹)")
    common.add_argument("--per-site", default="", dest="per_site",
                        help='1-based FWHM overrides matching the reference model, '
                             'e.g. "3:50" for FMO (site3=50, rest=--fwhm)')
    common.add_argument("--workers", type=int, default=1,
                        help="GPUs for mc-disorder-run (one worker each; "
                             "per-node runs always use 1)")
    common.add_argument("--precision", choices=("fp64", "mixed"), default="fp64",
                        help="build to run; FIX fp64 for a fair sampling metric")
    common.add_argument("--wmin", type=float, default=None)
    common.add_argument("--wmax", type=float, default=None)
    common.add_argument("--w0", type=float, default=None,
                        help="override w0_cm for the FFT axis only")
    common.add_argument("--seed", type=int, default=20260704, help="master_seed")
    common.add_argument("--out", default="p3.5-smoothness-probe.csv",
                        help="output CSV path")
    common.add_argument("--work-dir", default=None,
                        help="scratch dir for node files / shards (default: temp)")

    p = argparse.ArgumentParser(
        description="MC sampling probe: smoothness and variance reduction")
    sub = p.add_subparsers(dest="task", required=True,
                           metavar="{absorption,arena,converge,rqmc}")

    a = sub.add_parser("absorption", parents=[common],
                       help="sparse_gh vs iid smoothness probe")
    a.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="sparse_gh relative Smolyak levels to sweep")
    a.add_argument("--iid", type=int, nargs="+", default=[1, 8, 27, 64],
                   metavar="N", help="iid node counts to sweep")

    ar = sub.add_parser("arena", parents=[common],
                        help="iid/cv/antithetic variance-reduction arena")
    _add_sweep(ar)
    _add_ref_file(ar)
    _add_iid_ref(ar)
    _add_cv_width(ar)
    _add_pool_file(ar)
    ar.add_argument("--no-antithetic", action="store_true",
                    help="skip the antithetic curve (iid/cv only)")
    ar.add_argument("--stack", action="store_true",
                    help="run antithetic per-node too → derive antithetic AND "
                         "antithetic_cv, testing whether they stack or overlap")

    cv = sub.add_parser("converge", parents=[common],
                        help="closed-form minimum evaluations to converge (N*)")
    cv.add_argument("--pool-size", type=int, default=256, dest="pool_size",
                    help="per-node pool size M (even; iid + antithetic each get M)")
    cv.add_argument("--tol", type=float, nargs="+", default=[0.02, 0.01, 0.005],
                    help="self-convergence tolerances for the N* table")
    _add_pool_file(cv)
    _add_ref_file(cv)
    _add_cv_width(cv)

    rq = sub.add_parser("rqmc", parents=[common],
                        help="Sobol convergence-rate de-risk probe (slope test)")
    _add_sweep(rq)
    _add_pool_file(rq)
    _add_ref_file(rq)
    _add_iid_ref(rq)
    rq.add_argument("--n-replica", type=int, default=4, dest="n_replica",
                    help="number of independent Sobol scrambles (RMS + across-"
                         "replica error)")
    rq.add_argument("--iid-pool", default=None, dest="iid_pool",
                    help="converge pool .h5 to read the FREE iid baseline from")
    rq.add_argument("--obs", choices=("absorption", "2des"), default="absorption",
                    help="observable/solver task — absorption (1-D, needs --ref-"
                         "file gold) or 2des (2-D map at T=0; iid ref if no gold)")
    rq.add_argument("--dw", type=float, default=10.0,
                    help="2des: FFT frequency bin (cm⁻¹); consistent across nodes")
    rq.add_argument("--pol", default="ap",
                    help="2des: polarization preset/angles for the map (default ap)")
    rq.add_argument("--tw-index", type=int, default=0, dest="tw_index",
                    help="2des: waiting-time index into the config tw list (0=T=0)")
    return p


def main(argv=None):
    p = _build_parser()
    args = p.parse_args(argv)

    if args.precision != "fp64":
        print("[mc-probe] WARNING: --precision != fp64; the FP32 arithmetic floor "
              "(~1e-4) will pollute the sampling-error metric.",
              file=sys.stderr)

    work_root = args.work_dir or tempfile.mkdtemp(prefix="mc_probe_")
    try:
        if args.task == "arena":
            if not args.ref_file and args.ref_seed == args.seed:
                p.error("--ref-seed must differ from --seed (independent reference)")
            run_arena(base_cfg=args.config, sweep_ns=args.sweep, fwhm=args.fwhm,
                      workers=args.workers, precision=args.precision,
                      wmin=args.wmin, wmax=args.wmax, w0=args.w0,
                      cv_width=args.cv_width, ref_n=args.ref_n,
                      ref_seed=args.ref_seed, out_csv=args.out,
                      work_root=work_root, seed=args.seed,
                      do_antithetic=not args.no_antithetic, stack=args.stack,
                      ref_file=args.ref_file, per_site=args.per_site,
                      pool_file=args.pool_file)
        elif args.task == "converge":
            run_converge(base_cfg=args.config, pool_size=args.pool_size,
                         tols=args.tol, fwhm=args.fwhm, per_site=args.per_site,
                         cv_width=args.cv_width, precision=args.precision,
                         wmin=args.wmin, wmax=args.wmax, w0=args.w0,
                         ref_file=args.ref_file, seed=args.seed, out_csv=args.out,
                         work_root=work_root, pool_file=args.pool_file)
        elif args.task == "rqmc":
            run_rqmc(base_cfg=args.config, sweep_ns=args.sweep,
                     n_replica=args.n_replica, fwhm=args.fwhm,
                     per_site=args.per_site, precision=args.precision,
                     wmin=args.wmin, wmax=args.wmax, w0=args.w0,
                     ref_file=args.ref_file, seed=args.seed, out_csv=args.out,
                     work_root=work_root, iid_pool=args.iid_pool,
                     pool_file=args.pool_file, task=args.obs, dw=args.dw,
                     pol=args.pol, tw_index=args.tw_index, ref_n=args.ref_n,
                     ref_seed=args.ref_seed)
        else:  # absorption
            run_probe(base_cfg=args.config, levels=args.levels, iid_ns=args.iid,
                      fwhm=args.fwhm, workers=args.workers,
                      precision=args.precision, wmin=args.wmin, wmax=args.wmax,
                      w0=args.w0, out_csv=args.out, work_root=work_root,
                      seed=args.seed)
    except (RuntimeError, ValueError, OSError) as e:
        p.error(str(e))
    return 0
