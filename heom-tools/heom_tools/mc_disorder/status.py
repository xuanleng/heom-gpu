#!/usr/bin/env python3
"""MC convergence status — "am I done, how much more?"

Points at a live or finished MC run and reports, per shard group / merged file,
the current L² RELATIVE error by the CORRECT per-sampler criterion and the
evaluations still needed to reach a tolerance:

  * mean / antithetic   √N SE (from /mc, count-weighted) → N* = N·(rel/tol)²
  * rqmc                across-replica SE (combine_replicas output) → N* under
                        the empirical slope (--slope), else a conservative √N
  * quadrature          no √N SE — status is undefined (use level-diff refine)

Parseval makes the relative error identical in time/frequency, so this reads the
raw observable (rho_xyz / resp) and needs NO FFT — the running SPECTRUM itself is
already available via ``las-post`` on the current merged file, so this tool is
purely the verdict half.

Exit code is the composable stop/continue signal (no worker-killing here — that
is the launcher): 0 = all converged, 3 = at least one not converged, 2 =
undefined (quadrature / bad input). So a driver can loop::

    until mc-disorder-status ./run --tol 0.01; do  mc-disorder-run ... ; done

Usage:
    mc-disorder-status ./data/fmo_las --tol 0.01
    mc-disorder-status rqmc_combined.h5 --tol 0.01 --slope -0.91
    mc-disorder-status ./data/fmo7_2des --dataset resp --tol 0.02
"""
import argparse
import os
import sys

import numpy as np
import h5py

from heom_tools.common.mc_merge import convergence_status, merge_mc_shards
from heom_tools.common import mc_naming
from .merge import find_shards, group_by_base


def _read_attr(group, name):
    v = group.attrs.get(name)
    return v.decode() if isinstance(v, bytes) else v


def _estimate_from_file(path, dataset):
    """Read (mean, se_l2, count, estimator, n_replica) from a merged/combined
    single-shot file (has /mc but is not a _rank shard)."""
    with h5py.File(path, "r") as f:
        if "/mc" not in f:
            raise ValueError(
                f"'{path}' has no /mc group — not an MC merged file (run "
                "mc-disorder-merge first, or pass the shard directory)")
        mc = f["/mc"]
        estimator = _read_attr(mc, "estimator") or "mean"
        mean = np.asarray(f[f"/results/{dataset}"][...], dtype=np.complex128)
        count = int(np.asarray(mc["count"]).reshape(-1)[0])
        n_replica = mc.attrs.get("n_replica")
        n_replica = int(n_replica) if n_replica is not None else None
        se_l2 = mc.attrs.get("se_l2")
        se_l2 = float(se_l2) if se_l2 is not None else None
    return mean, se_l2, count, estimator, n_replica


def _estimate_from_merged_dict(merged, dataset):
    return (np.asarray(merged["mean"]), merged.get("se_l2"),
            int(merged["count"]), merged["estimator"],
            merged.get("n_replica"))


def _report(label, mean, se_l2, count, estimator, n_replica, tol, slope):
    """Print one estimate's status; return True if converged, None if undefined."""
    print(f"[status] {label}")
    if estimator == "quadrature":
        print("  estimator : quadrature — no √N SE; refine by level-to-level "
              "difference (status undefined)")
        return None
    if se_l2 is None:
        print(f"  estimator : {estimator} — file has no se_l2 (not a merged MC "
              "estimate?); cannot judge convergence")
        return None
    s = convergence_status(mean, se_l2, count, estimator, tol,
                           slope=slope, n_replica=n_replica)
    nrep = f", {n_replica} replicas" if n_replica else ""
    print(f"  estimator : {estimator}   N = {count}{nrep}")
    print(f"  rel error : {s['rel_err']:.3e}   (‖SE‖₂/‖μ‖₂)")
    if s["converged"]:
        print(f"  target tol: {tol:.2e}  → CONVERGED ✓")
    else:
        law = (f"slope {s['rate']:+.2f}" if not s["conservative"]
               else "conservative 1/√N; real RQMC steeper → pass --slope")
        print(f"  target tol: {tol:.2e}  → NOT converged")
        print(f"  projected : N* ≈ {s['n_star']} ({law}), "
              f"~{s['remaining']} more evals")
    return s["converged"]


def main(argv=None):
    p = argparse.ArgumentParser(
        description="MC convergence status: current relative error + evals to "
                    "reach --tol (exit 0=converged, 3=not, 2=undefined)")
    p.add_argument("inputs", nargs="*",
                   help="shard directories/files (merged on the fly) and/or "
                        "already-merged .h5 (default: current directory)")
    p.add_argument("--dataset", default="rho_xyz",
                   help="/results dataset (rho_xyz=absorption, resp=2des)")
    p.add_argument("--tol", type=float, default=0.01,
                   help="target L² relative self-convergence tolerance")
    p.add_argument("--slope", type=float, default=None,
                   help="rqmc: empirical log-log convergence slope (e.g. -0.91 "
                        "from mc-probe rqmc) for a tight N*; default is a "
                        "conservative 1/√N")
    args = p.parse_args(argv)

    inputs = args.inputs or ["."]
    shard_inputs, merged_files = [], []
    for item in inputs:
        if os.path.isdir(item):
            shard_inputs.append(item)
        elif mc_naming.is_rank_shard(os.path.basename(item)):
            shard_inputs.append(item)
        else:
            merged_files.append(item)

    estimates = []  # (label, mean, se_l2, count, estimator, n_replica)
    shards = find_shards(shard_inputs) if shard_inputs else []
    if shards:
        try:
            groups = group_by_base(shards)
        except ValueError as e:
            p.error(str(e))
        for _key, group in sorted(groups.items()):
            merged = merge_mc_shards(sorted(group), args.dataset)
            label = mc_naming.shard_group_key(os.path.basename(sorted(group)[0]))
            estimates.append((label, *_estimate_from_merged_dict(merged,
                                                                 args.dataset)))
    for f in merged_files:
        try:
            estimates.append((os.path.basename(f),
                              *_estimate_from_file(f, args.dataset)))
        except (OSError, ValueError, KeyError) as e:
            p.error(str(e))

    if not estimates:
        p.error("no MC shards or merged files found "
                "(pass a run directory or a merged .h5)")

    verdicts = []
    for label, mean, se_l2, count, estimator, n_replica in estimates:
        verdicts.append(_report(label, mean, se_l2, count, estimator,
                                n_replica, args.tol, args.slope))

    if any(v is None for v in verdicts):
        return 2
    return 0 if all(verdicts) else 3


if __name__ == "__main__":
    sys.exit(main())
