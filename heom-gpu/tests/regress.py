#!/usr/bin/env python3
"""The one entry point for HDF5 regression work.

    regress.py check   [--tol 1e-12]        run the baseline set, compare each
                                            run against its committed baseline
                                            within --tol  (ctest: regression_all)
    regress.py diff    REF.h5 TEST.h5 [--tol]   one pair, same tolerance policy
    regress.py inspect FILE.h5              self-check: shape, trace, populations
    regress.py bitwise --emit DIR           run the bitwise set into DIR
    regress.py bitwise A B                  two dirs (or two files): byte-exact
    regress.py regen   [STEM ...]           rewrite committed baselines from
                                            tests/configs (all, or the named stems)
    regress.py sites                        (re)generate the 14 FMO-7 per-site
                                            dynamics runs (filtered / nofilter)

Two comparison policies, on purpose:

  * `check` / `diff` are TOLERANCE checks (|Δ| < tol, default 1e-12). Baselines
    are committed files from another build, and cross-build codegen carries a
    ulp-scale residual — so `--tol 0` can never pass and is not what you want.
  * `bitwise` compares dataset BYTES. It answers "did this refactor move a
    single bit" between two emits of the same session, not against history.
    +0.0 vs -0.0 is a difference (deliberate: the manifold-block design has a
    ±0 corner and this is where it surfaces); NaN equals NaN (same payload).

One benchmark table feeds every subcommand; one `run_solver` writes every temp
config. The solver binary is $HEOM_SOLVER or build/heom_solver. ctest runs
`check` twice (regression_all: FP64 at 1e-12; regression_mp: the MP binary
against the same FP64 baselines at 1e-5); repeat `bitwise --emit` with
build/heom_solver_mp yourself.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np

try:
    import h5py
    import yaml
except ImportError as e:  # pragma: no cover
    print(f"ERROR: {e.name} required (pip install h5py pyyaml)")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(HERE, "configs")
BASELINE_DIR = os.path.join(HERE, "benchmark_data")
DEFAULT_SOLVER = os.environ.get("HEOM_SOLVER", os.path.join(ROOT, "build", "heom_solver"))

# Scratch for `bitwise --emit`: the solver embeds the resolved config (incl.
# save_path) in parameters/config_yaml, so emitting straight into before/ and
# after/ would put those two names into the compared bytes.
STAGE_DIR = "/tmp/heom_regress_stage"


@dataclass(frozen=True)
class Bench:
    stem: str        # tests/configs/<stem>.yaml ; output <stem>.h5
    task: str        # solver subcommand
    baseline: bool   # has a committed tests/benchmark_data/<stem>.h5


# benchmark_fmo7_n29 is the ONLY entry with a doubly-excited (f) manifold — it
# is the sole coverage of the manifold block structure; never trim it.
# The two 2DES entries are both load-bearing: 3sites_2des is filter-on / reuse
# walk (B=1) / no T gap; 3sites_2des_blocking is filter-off / batched (B=21) /
# T gap. The blocking one has no committed baseline: bitwise only.
BENCHMARKS = (
    Bench("benchmark_fmo7",                "dynamics",   True),   # EXCITED_ONLY   N=7
    Bench("benchmark_fmo7_n29",            "dynamics",   True),   # DOUBLE_EXCITED N=29
    Bench("benchmark_fmo7_with_ground",    "dynamics",   True),   # WITH_GROUND    N=8
    Bench("benchmark_absorption_fmo7",     "absorption", True),   # disorder-off absorption
    Bench("benchmark_fluorescence_fmo7",   "fluorescence", True), # equilibrate + emit, filter off
    Bench("benchmark_3sites_2des",         "2des",       True),
    Bench("benchmark_3sites_2des_blocking", "2des",      False),
)


# --------------------------------------------------------------------------
# running the solver
# --------------------------------------------------------------------------

def run_solver(bench, out_dir, filename, solver, quiet=True):
    """Run one benchmark, output to <out_dir>/<filename>.h5. Returns the path.

    The committed config is never touched: its `<task>.output` sub-block is
    rewritten in a temp copy. Raises RuntimeError on solver failure.
    """
    config_path = os.path.join(CONFIG_DIR, f"{bench.stem}.yaml")
    if not os.path.exists(config_path):
        # Tracked fixture; absence is a broken checkout, not a skip.
        raise RuntimeError(f"config {bench.stem}.yaml not found (tracked file — broken checkout?)")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    out = cfg.setdefault(bench.task, {}).setdefault("output", {})
    out["save_path"] = out_dir.rstrip("/") + "/"
    out["filename"] = filename
    out["use_timestamp"] = False

    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = tmp.name
    try:
        print(f">>> {os.path.basename(solver)} {bench.task} {bench.stem}.yaml")
        ret = subprocess.call([solver, bench.task, tmp_path],
                              stdout=subprocess.DEVNULL if quiet else None)
    finally:
        os.unlink(tmp_path)
    if ret != 0:
        raise RuntimeError(f"solver returned {ret} for {bench.stem}")
    produced = os.path.join(out_dir, f"{filename}.h5")
    if not os.path.exists(produced):
        raise RuntimeError(f"{bench.stem}: no output at {produced}")
    return produced


# --------------------------------------------------------------------------
# tolerance comparison (check / diff / inspect)
# --------------------------------------------------------------------------

def _decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except (ValueError, TypeError):
            return v
    return v


def load_hdf5(path):
    """(data, attrs, task). Complex datasets split into *_real/_imag; attrs of
    /parameters and /derived keyed "<group>/<name>"."""
    with h5py.File(path, "r") as f:
        task = _decode(f["/"].attrs.get("task", b"dynamics"))
        if task == "dynamics":
            rho = np.asarray(f["/results/density_matrix"][...], dtype=np.complex128)
            data = {"time": np.asarray(f["/results/time"][...], dtype=np.float64),
                    "rho_real": rho.real, "rho_imag": rho.imag}
        elif task in ("absorption", "fluorescence"):
            rx = np.asarray(f["/results/rho_xyz"][...], dtype=np.complex128)
            data = {"time": np.asarray(f["/results/time"][...], dtype=np.float64),
                    "rho_xyz_real": rx.real, "rho_xyz_imag": rx.imag}
            if task == "fluorescence":
                eq = np.asarray(f["/results/rho_eq"][...], dtype=np.complex128)
                data.update(rho_eq_real=eq.real, rho_eq_imag=eq.imag)
        elif task == "2des":
            resp = np.asarray(f["/results/resp"][...], dtype=np.complex128)
            data = {"tau_fs": np.asarray(f["/results/tau_fs"][...], dtype=np.float64),
                    "tw_fs": np.asarray(f["/results/tw_fs"][...], dtype=np.float64),
                    "t_fs": np.asarray(f["/results/t_fs"][...], dtype=np.float64),
                    "resp_real": resp.real, "resp_imag": resp.imag}
        else:
            raise ValueError(f"unknown task attr: {task!r}")
        attrs = {f"{g}/{k}": _decode(f[g].attrs[k])
                 for g in ("parameters", "derived") if g in f for k in f[g].attrs}
    return data, attrs, task


def compare_arrays(name, ref, test, tol):
    if ref.shape != test.shape:
        print(f"  [FAIL] {name}: shape mismatch {ref.shape} vs {test.shape}")
        return False
    diff = np.abs(ref - test)
    max_diff = float(np.max(diff))
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    passed = max_diff < tol
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"         max |Δ| = {max_diff:.2e}  (at index {idx})")
    print(f"        mean |Δ| = {float(np.mean(diff)):.2e}")
    return passed


def compare_trace(ref_real, test_real, tol):
    tr_ref = np.einsum("tii->t", ref_real)
    tr_test = np.einsum("tii->t", test_real)
    max_diff = float(np.max(np.abs(tr_ref - tr_test)))
    passed = max_diff < tol
    print(f"  [{'PASS' if passed else 'FAIL'}] Tr(rho) comparison")
    print(f"         max |Δ Tr| = {max_diff:.2e}")
    print(f"         ref  max|Tr-1| = {float(np.max(np.abs(tr_ref - 1.0))):.2e}")
    print(f"         test max|Tr-1| = {float(np.max(np.abs(tr_test - 1.0))):.2e}")
    return passed


def compare_metadata(ref_attrs, test_attrs):
    # Keys absent from the reference are tolerated (older baselines).
    keys = [f"parameters/{k}" for k in ("n_sites", "n_steps", "dt", "temperature",
                                        "n_extra_poles", "L_cutoff", "filter_threshold")]
    keys.append("derived/M_total")
    ok = True
    for k in keys:
        if k not in ref_attrs:
            continue
        if k not in test_attrs:
            print(f"  [WARN] metadata '{k}' missing in test file")
            continue
        rv, tv = ref_attrs[k], test_attrs[k]
        match = (np.isclose(rv, tv, rtol=1e-10) if isinstance(rv, (float, np.floating))
                 else rv == tv)
        if not match:
            print(f"  [FAIL] metadata '{k}': {rv} vs {tv}")
            ok = False
    if ok:
        print("  [PASS] metadata consistency")
    return ok


def compare_files(ref_path, test_path, tol):
    """Tolerance comparison of two outputs of the same task. True iff all pass."""
    print(f"\n{'=' * 60}\nReference: {ref_path}\nTest:      {test_path}\n"
          f"Tolerance: {tol:.0e}\n{'=' * 60}")
    ref, ref_attrs, ref_task = load_hdf5(ref_path)
    test, test_attrs, test_task = load_hdf5(test_path)
    if ref_task != test_task:
        print(f"\n  [FAIL] task mismatch: {ref_task!r} vs {test_task!r}")
        print(f"\n{'=' * 60}\nRESULT: SOME FAILED\n{'=' * 60}\n")
        return False

    results = []
    print("\n--- Metadata ---")
    results.append(compare_metadata(ref_attrs, test_attrs))
    print(f"\n--- Data Arrays (task={ref_task}) ---")
    axes = {"dynamics": ("time",), "absorption": ("time",), "fluorescence": ("time",),
            "2des": ("tau_fs", "tw_fs", "t_fs")}[ref_task]
    for a in axes:
        results.append(compare_arrays(a, ref[a], test[a], tol=1e-15))
    payload = {"dynamics": ("rho_real", "rho_imag"),
               "absorption": ("rho_xyz_real", "rho_xyz_imag"),
               "fluorescence": ("rho_eq_real", "rho_eq_imag", "rho_xyz_real", "rho_xyz_imag"),
               "2des": ("resp_real", "resp_imag")}[ref_task]
    for k in payload:
        results.append(compare_arrays(k, ref[k], test[k], tol))
    if ref_task == "dynamics":
        print("\n--- Trace Conservation ---")
        results.append(compare_trace(ref["rho_real"], test["rho_real"], tol))
        print("\n--- Population Summary (first & last timestep) ---")
        N = ref["rho_real"].shape[1]
        for label, d in (("ref", ref), ("test", test)):
            first = np.diagonal(d["rho_real"][0])
            last = np.diagonal(d["rho_real"][-1])[:min(N, 10)]
            print(f"  {label} t=0:     " + "  ".join(f"{p:.6f}" for p in first))
            print(f"  {label} t={d['time'][-1]:.0f}fs: " + "  ".join(f"{p:.6f}" for p in last))

    ok = all(results)
    print(f"\n{'=' * 60}\nRESULT: {'ALL PASSED' if ok else 'SOME FAILED'}\n{'=' * 60}\n")
    return ok


def inspect_file(path):
    data, _, task = load_hdf5(path)
    print(f"  File: {path}\n  task: {task}")
    if task == "dynamics":
        n_steps, N = data["rho_real"].shape[:2]
        print(f"  n_sites={N}, timesteps={n_steps}, "
              f"t=[{data['time'][0]:.1f}, {data['time'][-1]:.1f}] fs")
        tr = np.einsum("tii->t", data["rho_real"])
        print(f"  Tr(rho): min={tr.min():.10f}, max={tr.max():.10f}")
        print(f"  max|Tr-1| = {float(np.max(np.abs(tr - 1.0))):.2e}")
        last = np.diagonal(data["rho_real"][-1])
        print("  Final populations: " + "  ".join(f"{p:.6f}" for p in last[:10]))
        if N > 10:
            print(f"    ... ({N - 10} more)")
    elif task in ("absorption", "fluorescence"):
        n_dir, n_steps, N, _ = data["rho_xyz_real"].shape
        print(f"  n_sites={N}, timesteps={n_steps}, dirs={n_dir}, "
              f"t=[{data['time'][0]:.1f}, {data['time'][-1]:.1f}] fs")
        if task == "fluorescence":
            print("  rho_eq populations: " + "  ".join(
                f"{p:.6f}" for p in np.diagonal(data["rho_eq_real"])))
    else:
        print(f"  resp shape: {data['resp_real'].shape}")
        print(f"  n_tau={len(data['tau_fs'])}, n_tw={len(data['tw_fs'])}, "
              f"n_t={len(data['t_fs'])}")


# --------------------------------------------------------------------------
# bitwise comparison
# --------------------------------------------------------------------------

def _datasets(h5file):
    out = []
    h5file.visititems(lambda name, obj: out.append((name, obj))
                      if isinstance(obj, h5py.Dataset) else None)
    return dict(out)


def bitwise_pair(ref_path, test_path):
    """True iff every dataset matches byte for byte. Prints what differs."""
    with h5py.File(ref_path, "r") as a, h5py.File(test_path, "r") as b:
        a_sets, b_sets = _datasets(a), _datasets(b)
        bad = [f"missing in test: {n}" for n in sorted(set(a_sets) - set(b_sets))]
        bad += [f"missing in ref: {n}" for n in sorted(set(b_sets) - set(a_sets))]
        for name in sorted(set(a_sets) & set(b_sets)):
            ref = np.asarray(a_sets[name])
            test = np.asarray(b_sets[name])
            if ref.shape != test.shape or ref.dtype != test.dtype:
                bad.append(f"{name}: shape/dtype {ref.shape}{ref.dtype} vs "
                           f"{test.shape}{test.dtype}")
            elif ref.dtype.kind == "O":
                # vlen strings: numpy holds pointers here, compare the text.
                if ref.tolist() != test.tolist():
                    bad.append(f"{name}: text differs")
            elif ref.tobytes() != test.tobytes():
                # Magnitude tells ulp-scale codegen drift from a real indexing bug.
                if ref.dtype.kind in "fciu":
                    delta = float(np.max(np.abs(ref - test)))
                    scale = float(np.max(np.abs(ref))) or 1.0
                    bad.append(f"{name}: max|Δ|={delta:.3e} (rel {delta / scale:.3e})")
                elif ref.dtype.names:  # compound {r, i}
                    # Bytes differ yet values compare equal: ±0 (or a NaN payload).
                    fields = [f for f in ref.dtype.names if not np.array_equal(ref[f], test[f])]
                    if not fields:
                        bad.append(f"{name}: bytes differ, values equal (±0 or NaN payload)")
                        continue
                    delta = max(float(np.max(np.abs(ref[f] - test[f]))) for f in fields)
                    scale = max(float(np.max(np.abs(ref[f]))) for f in fields) or 1.0
                    bad.append(f"{name}: max|Δ|={delta:.3e} (rel {delta / scale:.3e}) "
                               f"in {','.join(fields)}")
                else:
                    bad.append(f"{name}: bytes differ (non-numeric)")
    label = os.path.basename(ref_path)
    if bad:
        print(f"BITDIFF {label}")
        for line in bad:
            print(f"    {line}")
        return False
    print(f"BITSAME {label}")
    return True


def bitwise_dirs(ref_dir, test_dir):
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
        ok &= bitwise_pair(os.path.join(ref_dir, name), test_path)
    return ok


def bitwise_emit(outdir, solver):
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    ok = True
    for bench in BENCHMARKS:
        try:
            produced = run_solver(bench, STAGE_DIR, bench.stem, solver)
        except RuntimeError as e:
            print(f"[FAIL] {e}")
            ok = False
            continue
        shutil.move(produced, os.path.join(outdir, f"{bench.stem}.h5"))   # STAGE_DIR may be tmpfs
    return ok


# --------------------------------------------------------------------------
# baselines: check / regen
# --------------------------------------------------------------------------

def check_baselines(tol, solver):
    """Run every baseline benchmark, compare against its committed file."""
    ok = True
    compared = 0
    with tempfile.TemporaryDirectory(prefix="heom_regress_") as tmp:
        for bench in (b for b in BENCHMARKS if b.baseline):
            ref = os.path.join(BASELINE_DIR, f"{bench.stem}.h5")
            if not os.path.exists(ref):
                print(f"[FAIL] baseline {bench.stem}.h5 not found (tracked file — broken checkout?)")
                ok = False
                continue
            try:
                test = run_solver(bench, tmp, f"test_{bench.stem}", solver, quiet=False)
            except RuntimeError as e:
                print(f"[FAIL] {e}")
                ok = False
                continue
            ok &= compare_files(ref, test, tol)
            compared += 1
    # Guards against someone emptying BENCHMARKS and getting a green gate.
    if compared == 0:
        print("[FAIL] zero benchmarks compared")
        return False
    return ok


def regen_baselines(stems, solver):
    """Overwrite committed baselines. Only for a deliberate re-baseline commit."""
    wanted = {b.stem: b for b in BENCHMARKS if b.baseline}
    unknown = [s for s in stems if s not in wanted]
    if unknown:
        print(f"ERROR: not baseline stems: {unknown}; choose from {sorted(wanted)}")
        return False
    ok = True
    for stem in (stems or sorted(wanted)):
        try:
            run_solver(wanted[stem], BASELINE_DIR, stem, solver)
            print(f"  [OK] rewrote benchmark_data/{stem}.h5")
        except RuntimeError as e:
            print(f"[FAIL] {e}")
            ok = False
    print("\nBaselines are tracked: review `git diff --stat tests/benchmark_data` and\n"
          "state in the commit why the physics moved before committing them.")
    return ok


# --------------------------------------------------------------------------
# FMO-7 per-site set (14 dynamics runs; schema canary for benchmark_infra)
# --------------------------------------------------------------------------

FMO7_SITE_TEMPLATE = """\
system:
  n_states: 7
  basis: "excited_only"
  temperature: 77.0
  hamiltonian:
    - [12400.0, -106.0,    8.0,   -5.0,    6.0,   -8.0,   -4.0]
    - [-106.0, 12540.0,   28.0,    6.0,    2.0,   13.0,    1.0]
    - [    8.0,   28.0, 12120.0,  -62.0,   -1.0,   -9.0,   17.0]
    - [   -5.0,    6.0,  -62.0, 12295.0,  -70.0,  -19.0,  -57.0]
    - [    6.0,    2.0,   -1.0,  -70.0, 12440.0,   40.0,   -2.0]
    - [   -8.0,   13.0,   -9.0,  -19.0,   40.0, 12480.0,   32.0]
    - [   -4.0,    1.0,   17.0,  -57.0,   -2.0,   32.0, 12380.0]
  w0: 12000.0

bath:
  spectral_density: "Drude-Lorentz"
  eta: 70.0
  gamma_inverse: 100.0
  expansion_method: matsubara
  n_extra_poles: 5
  L_cutoff: 3

filter:
  enabled: {filter_enabled}
  tolerance: {tolerance}

dynamics:
  integrator: rk4
  dt: 2.0
  n_steps: 500
  output_interval: 10
  print_interval: {print_interval}
  sink:
    enabled: false
    target_site: 3
    rate: 0.001
  initial_state:
    type: "site_excitation"
    excited_site: {site}
  output:
    save_path: "{output_dir}"
    filename: "{output_name}"
    use_timestamp: false
"""


def fmo7_site_config(site, use_filter, output_dir, output_name):
    # filter.enabled is the switch; when false the knobs are ignored but must
    # still parse.
    return FMO7_SITE_TEMPLATE.format(
        filter_enabled="true" if use_filter else "false",
        tolerance="1.0e-7",
        print_interval="500" if use_filter else "0",
        site=site, output_dir=output_dir.rstrip("/") + "/", output_name=output_name)


def generate_sites(solver):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    failed = []
    for use_filter in (True, False):
        tag = "filtered" if use_filter else "nofilter"
        print(f"=== FMO-7 {tag} ===")
        for site in range(1, 8):
            name = f"benchmark_fmo7_{tag}_site{site}"
            if os.path.exists(os.path.join(BASELINE_DIR, f"{name}.h5")):
                print(f"  [SKIP] {name}.h5 already exists")
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                f.write(fmo7_site_config(site, use_filter, BASELINE_DIR, name))
                path = f.name
            try:
                r = subprocess.run([solver, "dynamics", path], capture_output=True,
                                   text=True, timeout=600)
            finally:
                os.unlink(path)
            if r.returncode != 0:
                print(f"  [FAIL] {name} (exit={r.returncode}) {r.stderr[:200]}")
                failed.append(name)
            else:
                print(f"  [OK]   {name}.h5")
    if failed:
        print(f"FAILED: {failed}")
    return not failed


# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solver", default=DEFAULT_SOLVER,
                   help="solver binary (default: $HEOM_SOLVER or build/heom_solver)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check", help="baseline set vs committed baselines, within --tol")
    s.add_argument("--tol", type=float, default=1e-12)

    s = sub.add_parser("diff", help="tolerance-compare two files")
    s.add_argument("ref"); s.add_argument("test")
    s.add_argument("--tol", type=float, default=1e-12)

    s = sub.add_parser("inspect", help="self-check one file")
    s.add_argument("file")

    s = sub.add_parser("bitwise", help="byte-exact gate: --emit DIR, or A B")
    s.add_argument("paths", nargs="*")
    s.add_argument("--emit", metavar="OUTDIR")

    s = sub.add_parser("regen", help="rewrite committed baselines (all or STEM...)")
    s.add_argument("stems", nargs="*")

    sub.add_parser("sites", help="generate the 14 FMO-7 per-site runs")

    a = p.parse_args(argv)
    if a.cmd == "check":
        return check_baselines(a.tol, a.solver)
    if a.cmd == "diff":
        return compare_files(a.ref, a.test, a.tol)
    if a.cmd == "inspect":
        inspect_file(a.file)
        return True
    if a.cmd == "bitwise":
        if a.emit:
            return bitwise_emit(a.emit, a.solver)
        if len(a.paths) != 2:
            p.error("bitwise needs --emit DIR, or exactly two paths")
        ref, test = a.paths
        if os.path.isdir(ref) != os.path.isdir(test):
            p.error("compare two files or two directories, not a mix")
        ok = bitwise_dirs(ref, test) if os.path.isdir(ref) else bitwise_pair(ref, test)
        print("ALL BITSAME" if ok else "DIFFERENCES FOUND")
        return ok
    if a.cmd == "regen":
        return regen_baselines(a.stems, a.solver)
    if a.cmd == "sites":
        return generate_sites(a.solver)
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
