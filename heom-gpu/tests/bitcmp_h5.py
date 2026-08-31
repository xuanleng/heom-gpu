#!/usr/bin/env python3
"""Bitwise HDF5 comparison for the "this change must not move a single bit" gate.

compare_benchmark.py answers a different question: "is a freshly built binary
close enough to a committed baseline", and its --tol 1e-12 is right for that
(cross-build codegen residual is real). It cannot answer ours, because a
bookkeeping error in the manifold blocking work lands well inside 1e-12 — and
`--tol 0` does not help either, its predicate is `max_diff < tol`, false even
for identical arrays.

So: compare the raw bytes of every dataset. Two consequences worth knowing.
  * +0.0 and -0.0 compare as DIFFERENT. That is deliberate. The blocking design
    registers a ±0 corner case; this is how it surfaces instead of hiding.
  * NaN compares as equal to itself (same payload bytes), unlike ==.

We compare dataset CONTENT, never file hashes: output names carry timestamps
and HDF5 embeds creation metadata, so file digests always differ.

Usage
  # 1. produce a run's outputs into a directory
  python3 bitcmp_h5.py --emit before --solver build/heom_solver
  #    ... apply the change, rebuild ...
  python3 bitcmp_h5.py --emit after  --solver build/heom_solver
  # 2. compare
  python3 bitcmp_h5.py before after

Both steps must be repeated for build/heom_solver_mp: the `regression_all`
ctest target only ever drives the FP64 binary, so nobody collects the mixed
build's bit evidence for you.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import yaml

# (config stem, solver subcommand). benchmark_fmo7_n29 is the ONLY one of these
# carrying a doubly-excited (f) manifold — do not drop it when trimming this
# list, it is the sole coverage of the block structure the design targets.
BENCHMARKS = [
    ("benchmark_fmo7", "dynamics"),                # EXCITED_ONLY   N=7
    ("benchmark_fmo7_n29", "dynamics"),            # DOUBLE_EXCITED N=29
    ("benchmark_fmo7_with_ground", "dynamics"),    # WITH_GROUND    N=8
    ("benchmark_absorption_fmo7", "absorption"),
    # Two 2DES entries, and both are load-bearing for the manifold-blocking A/B
    # gate: the first is filter-on / flat-walk / no T gap, the second is
    # filter-off / batched / T gap. Neither alone exercises the three walks and
    # the three segment blocks. There is no committed baseline for the second —
    # this gate compares two runs of the same session, not a run against history.
    ("benchmark_3sites_2des", "2des"),
    ("benchmark_3sites_2des_blocking", "2des"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "configs")


def datasets(h5file):
    """(path, dataset) for every dataset in the file, recursively."""
    out = []
    h5file.visititems(
        lambda name, obj: out.append((name, obj))
        if isinstance(obj, h5py.Dataset) else None)
    return out


def compare_pair(ref_path, test_path):
    """True iff every dataset matches byte for byte. Prints what differs."""
    with h5py.File(ref_path, "r") as a, h5py.File(test_path, "r") as b:
        a_sets = dict(datasets(a))
        b_sets = dict(datasets(b))

        only_a = sorted(set(a_sets) - set(b_sets))
        only_b = sorted(set(b_sets) - set(a_sets))
        bad = [f"missing in test: {n}" for n in only_a]
        bad += [f"missing in ref: {n}" for n in only_b]

        for name in sorted(set(a_sets) & set(b_sets)):
            ref = np.asarray(a_sets[name])
            test = np.asarray(b_sets[name])
            if ref.shape != test.shape or ref.dtype != test.dtype:
                bad.append(f"{name}: shape/dtype {ref.shape}{ref.dtype} vs "
                           f"{test.shape}{test.dtype}")
            elif ref.dtype.kind == "O":
                # Variable-length strings. numpy stores POINTERS here, so
                # tobytes() would compare heap addresses and disagree on every
                # run — compare what the pointers point at.
                if ref.tolist() != test.tolist():
                    bad.append(f"{name}: text differs")
            elif ref.tobytes() != test.tobytes():
                # Report the magnitude too: a ulp-scale scatter points at a
                # codegen/FMA-contraction shift, a large or structured delta
                # points at a real indexing bug.
                if ref.dtype.kind in "fciu":
                    delta = float(np.max(np.abs(ref - test)))
                    scale = float(np.max(np.abs(ref))) or 1.0
                    bad.append(f"{name}: max|Δ|={delta:.3e} "
                               f"(rel {delta / scale:.3e})")
                elif ref.dtype.names:  # HDF5 compound, e.g. {r, i} complex
                    fields = [f for f in ref.dtype.names
                              if not np.array_equal(ref[f], test[f])]
                    delta = max(float(np.max(np.abs(ref[f] - test[f])))
                                for f in fields)
                    scale = max(float(np.max(np.abs(ref[f]))) for f in fields) or 1.0
                    bad.append(f"{name}: max|Δ|={delta:.3e} "
                               f"(rel {delta / scale:.3e}) in {','.join(fields)}")
                else:
                    bad.append(f"{name}: bytes differ (non-numeric)")

    label = f"{os.path.basename(ref_path)}"
    if bad:
        print(f"BITDIFF {label}")
        for line in bad:
            print(f"    {line}")
        return False
    print(f"BITSAME {label}")
    return True


# Every run writes here first and the results are moved out afterwards. The
# solver embeds the resolved config in parameters/config_yaml, so writing
# straight into the before/after directory would put those two directory names
# inside the compared bytes and report a difference on every single file.
STAGE_DIR = "/tmp/heom_bitcmp_stage"


def emit(outdir, solver):
    """Run every benchmark, writing outputs into outdir.

    The output path lives in each config's own `<task>.output` sub-block, so we
    rewrite that block in a temp copy — the committed configs are never touched
    (same approach as compare_benchmark.py).
    """
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(STAGE_DIR, exist_ok=True)
    ok = True
    for stem, task in BENCHMARKS:
        config_path = os.path.join(CONFIG_DIR, f"{stem}.yaml")
        if not os.path.exists(config_path):
            print(f"[SKIP] {stem}.yaml not found")
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        out = cfg.setdefault(task, {}).setdefault("output", {})
        out["save_path"] = STAGE_DIR + "/"
        out["filename"] = stem
        out["use_timestamp"] = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as tmp:
            yaml.safe_dump(cfg, tmp)
            tmp_path = tmp.name
        try:
            print(f">>> {solver} {task} {stem}.yaml")
            ret = subprocess.call([solver, task, tmp_path],
                                  stdout=subprocess.DEVNULL)
        finally:
            os.unlink(tmp_path)

        if ret != 0:
            print(f"[FAIL] solver returned {ret} for {stem}")
            ok = False
            continue

        produced = os.path.join(STAGE_DIR, f"{stem}.h5")
        if not os.path.exists(produced):
            print(f"[FAIL] {stem}: no output at {produced}")
            ok = False
            continue
        os.replace(produced, os.path.join(outdir, f"{stem}.h5"))
    return ok


def compare_dirs(ref_dir, test_dir):
    names = sorted(n for n in os.listdir(ref_dir) if n.endswith(".h5"))
    if not names:
        print(f"[FAIL] no .h5 files in {ref_dir}")
        return False
    ok = True
    for name in names:
        test_path = os.path.join(test_dir, name)
        if not os.path.exists(test_path):
            print(f"BITDIFF {name}: absent from {test_dir}")
            ok = False
            continue
        ok &= compare_pair(os.path.join(ref_dir, name), test_path)
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*",
                   help="two .h5 files, or two directories of them")
    p.add_argument("--emit", metavar="OUTDIR",
                   help="run the benchmark set into OUTDIR instead of comparing")
    p.add_argument("--solver", default="build/heom_solver",
                   help="solver binary for --emit (use build/heom_solver_mp too)")
    args = p.parse_args()

    if args.emit:
        sys.exit(0 if emit(args.emit, args.solver) else 1)

    if len(args.paths) != 2:
        p.error("need exactly two paths to compare (or --emit)")
    ref, test = args.paths
    if os.path.isdir(ref) != os.path.isdir(test):
        p.error("compare two files or two directories, not a mix")

    ok = (compare_dirs(ref, test) if os.path.isdir(ref)
          else compare_pair(ref, test))
    print("ALL BITSAME" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
