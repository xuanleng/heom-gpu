#!/usr/bin/env python3
"""MC static-disorder shard merge (component 4) — thin CLI over the shared
``heom_tools.common.mc_merge`` primitive.

Each MC worker writes a **single-shot-format** spectrum file (its disorder mean
in ``/results`` + a ``/mc`` provenance group), named
``<base>_<key>_<TS>_r<a>-<b>_rank{N}.h5``. The 16-hex parameter ``key`` is the
grouping identity, so shards from different timestamps and intervals (incremental
batches) merge together; the TS/interval are info bits only. Merging is a
count-weighted average of the means (guarded by r_done/config/disorder
invariants) written back as the ordinary single-shot file
``<base>_<key>_r<lo>-<hi>.h5`` (the covered 1-based range in the name keeps
successive incremental merges side by side) — so ``las-post`` reads it with zero
special-casing. No ref-swap.

Note: ``las-post <base>_..._rank1.h5`` already auto-merges the group and makes
the spectrum in one step; this CLI stays for explicit/scripted merges.

Usage:
    mc-disorder-merge ./data/fmo_las               # merge every key group there
    mc-disorder-merge ./data/fmo_las --out avg.h5  # single group -> explicit path
    mc-disorder-merge a_..._rank1.h5 a_..._rank2.h5 # explicit shard files
    mc-disorder-merge ./data/fmo7_2des --dataset resp

RQMC error path (slice 4b): each scramble replica is a normal mean run in its
OWN dir; merge each, then combine the R replica means into the rqmc estimator
(mean-of-means + across-replica SE — never a √N SEM on one Sobol set):
    mc-disorder-merge --combine-replicas --out rqmc.h5 rep0.h5 rep1.h5 rep2.h5
"""
import argparse
import glob
import os
import sys

from heom_tools.common.mc_merge import (
    merge_mc_shards, write_merged_spectrum, combine_replicas)
from heom_tools.common import mc_naming


def find_shards(inputs):
    """Resolve CLI inputs (dirs and/or files) to a sorted list of shard paths."""
    if not inputs:
        inputs = ["."]
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(glob.glob(os.path.join(item, "*_rank*.h5")))
        else:
            paths.append(item)
    return sorted(set(paths))


def group_by_base(paths):
    """Map each ``<dir>/<base>_<key>`` group key to its shard list (spanning
    timestamps/intervals). Non-shard inputs raise."""
    groups = {}
    for p in paths:
        try:
            group_stem = mc_naming.shard_group_key(os.path.basename(p))
        except ValueError as e:
            raise ValueError(
                f"'{p}' is not a key-tagged rank shard "
                "(<base>_<key>_<TS>_r<a>-<b>_rank{N}.h5); MC shards are written "
                "by `mc-disorder-run`") from e
        directory = os.path.dirname(os.path.abspath(p))
        key = os.path.join(directory, group_stem)
        groups.setdefault(key, []).append(p)
    return groups


def merge_group(shards, dataset, out=None):
    """Merge one shard group (single-shot format). Writes to `out` if given, else
    to ``<base>_<key>_r<lo>-<hi>.h5`` next to the shards — the covered range in the
    name keeps successive incremental merges (r1-100, r1-200, …) side by side
    instead of overwriting. Returns (out_path, merge summary)."""
    ordered = sorted(shards)
    merged = merge_mc_shards(ordered, dataset)
    if out is None:
        lo, hi = mc_naming.coverage_1based(merged["r_done"])
        group_stem = mc_naming.shard_group_key(os.path.basename(ordered[0]))
        directory = os.path.dirname(os.path.abspath(ordered[0]))
        out = os.path.join(directory, mc_naming.merged_name(group_stem, lo, hi))
    write_merged_spectrum(ordered[0], out, merged, dataset)
    return out, merged


def _combine_replicas_main(p, args):
    """RQMC replica combine (slice 4b): inputs are R merged replica spectra."""
    reps = [x for x in args.inputs if os.path.isfile(x)]
    if len(reps) < 2:
        p.error("--combine-replicas needs >= 2 merged replica files as inputs")
    if not args.out:
        p.error("--combine-replicas requires --out")
    combined = combine_replicas(sorted(reps), args.dataset)
    write_merged_spectrum(sorted(reps)[0], args.out, combined, args.dataset)
    print(f"combined {combined['n_replica']} RQMC replicas -> {args.out}")
    print(f"  total node evals   : {combined['count']}")
    print(f"  across-replica SE L2: {combined['se_l2']:.6e}   "
          "(honest RQMC error; NOT a √N SEM)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Merge MC static-disorder rank shards into single-shot files")
    p.add_argument("inputs", nargs="*",
                   help="directories (glob *_rank*.h5) and/or shard files "
                        "(default: current directory)")
    p.add_argument("--dataset", default="rho_xyz",
                   help="/results dataset to average (rho_xyz=absorption, "
                        "resp=2des; default: rho_xyz)")
    p.add_argument("--out", default=None,
                   help="output path; valid only when inputs are one shard group "
                        "(default: <base>_<key>_r<lo>-<hi>.h5 next to the shards)")
    p.add_argument("--combine-replicas", action="store_true",
                   help="RQMC error path (slice 4b): treat each input as a merged "
                        "replica MEAN (one scramble each) and combine into the "
                        "rqmc estimator (mean-of-means + across-replica SE). "
                        "Requires >=2 inputs and --out")
    args = p.parse_args(argv)

    if args.combine_replicas:
        return _combine_replicas_main(p, args)

    shards = find_shards(args.inputs)
    if not shards:
        p.error("no *_rank*.h5 found (pass a directory or shard files)")

    try:
        groups = group_by_base(shards)
    except ValueError as e:
        p.error(str(e))

    if args.out and len(groups) > 1:
        p.error(f"--out is valid for a single shard group, but inputs resolve to "
                f"{len(groups)} groups; drop --out to write each "
                "<base>_<key>_r<lo>-<hi>.h5")

    for _key, group in sorted(groups.items()):
        out_path, merged = merge_group(group, args.dataset, args.out)
        rd = merged["r_done"]
        print(f"merged {merged['n_shards']} shard(s) -> {out_path}")
        print(f"  realizations (count): {merged['count']}")
        if rd.size:
            print(f"  r covered          : {rd.size} "
                  f"(r in [{rd.min()}..{rd.max()}])")
        print(f"  SE L2 norm         : {merged['se_l2']:.6e}   "
              "(convergence proxy; smaller = tighter)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
