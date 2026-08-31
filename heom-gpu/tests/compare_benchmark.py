#!/usr/bin/env python3
"""
Regression benchmark comparison for HEOM solver.

Compares two HDF5 output files (group layout, 2026-05-21+) element by
element:
  - /results/density_matrix (complex compound)
  - /results/time
  - Tr(rho) at each timestep
  - /parameters attributes

Usage:
  python compare_benchmark.py <reference.h5> <test.h5> [--tol 1e-12]
  python compare_benchmark.py --run-all [--tol 1e-12]
"""

import sys
import argparse
import os
import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py required. Install: pip install h5py")
    sys.exit(1)


def _decode(v):
    """Match heom_h5.decode_attr without taking a runtime dep on tools/."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except (ValueError, TypeError):
            return v
    return v


def load_hdf5(path):
    """Open ``path`` and return ``(data, parameters_attrs, task, file_handle)``.

    The caller closes the file handle. The root ``task`` attr selects which
    ``/results`` datasets to load (dynamics/absorption/2des have different
    layouts). Complex datasets are split into ``*_real``/``*_imag`` so the
    array-comparison helper stays free of complex-dtype branching.
    """
    f = h5py.File(path, "r")
    task = _decode(f["/"].attrs.get("task", b"dynamics"))
    if task == "dynamics":
        rho = np.asarray(f["/results/density_matrix"][...], dtype=np.complex128)
        data = {"time":     np.asarray(f["/results/time"][...], dtype=np.float64),
                "rho_real": rho.real,
                "rho_imag": rho.imag}
    elif task == "absorption":
        rx = np.asarray(f["/results/rho_xyz"][...], dtype=np.complex128)
        data = {"time":         np.asarray(f["/results/time"][...], dtype=np.float64),
                "rho_xyz_real": rx.real,
                "rho_xyz_imag": rx.imag}
    elif task == "2des":
        resp = np.asarray(f["/results/resp"][...], dtype=np.complex128)
        data = {"tau_fs":    np.asarray(f["/results/tau_fs"][...], dtype=np.float64),
                "tw_fs":     np.asarray(f["/results/tw_fs"][...],  dtype=np.float64),
                "t_fs":      np.asarray(f["/results/t_fs"][...],   dtype=np.float64),
                "resp_real": resp.real,
                "resp_imag": resp.imag}
    else:
        raise ValueError(f"unknown task attr: {task!r}")
    params_grp = f["/parameters"]
    attrs = {k: _decode(params_grp.attrs[k]) for k in params_grp.attrs}
    return data, attrs, task, f


def compare_arrays(name, ref, test, tol):
    if ref.shape != test.shape:
        return False, f"shape mismatch: {ref.shape} vs {test.shape}"

    diff = np.abs(ref - test)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    idx = np.unravel_index(np.argmax(diff), diff.shape)

    passed = max_diff < tol
    status = "PASS" if passed else "FAIL"

    print(f"  [{status}] {name}")
    print(f"         max |Δ| = {max_diff:.2e}  (at index {idx})")
    print(f"        mean |Δ| = {mean_diff:.2e}")

    return passed, None


def compare_trace(ref_real, test_real, tol):
    n_steps = ref_real.shape[0]
    N = ref_real.shape[1]

    traces_ref = np.array([np.trace(ref_real[t]) for t in range(n_steps)])
    traces_test = np.array([np.trace(test_real[t]) for t in range(n_steps)])

    trace_diff = np.abs(traces_ref - traces_test)
    max_trace_diff = np.max(trace_diff)

    trace_deviation_ref = np.max(np.abs(traces_ref - 1.0))
    trace_deviation_test = np.max(np.abs(traces_test - 1.0))

    passed = max_trace_diff < tol
    status = "PASS" if passed else "FAIL"

    print(f"  [{status}] Tr(rho) comparison")
    print(f"         max |Δ Tr| = {max_trace_diff:.2e}")
    print(f"         ref  max|Tr-1| = {trace_deviation_ref:.2e}")
    print(f"         test max|Tr-1| = {trace_deviation_test:.2e}")

    return passed


def compare_metadata(ref_attrs, test_attrs):
    # The `if k not in ref_attrs: continue` skip below tolerates benchmark
    # files that predate any given attribute.
    keys_to_check = ["n_sites", "n_steps", "dt", "temperature", "n_extra_poles",
                     "L_cutoff", "M_total", "filter_threshold"]
    all_ok = True
    for k in keys_to_check:
        if k not in ref_attrs:
            continue
        if k not in test_attrs:
            print(f"  [WARN] metadata '{k}' missing in test file")
            continue
        rv, tv = ref_attrs[k], test_attrs[k]
        if isinstance(rv, (float, np.floating)):
            match = np.isclose(rv, tv, rtol=1e-10)
        else:
            match = (rv == tv)
        if not match:
            print(f"  [FAIL] metadata '{k}': {rv} vs {tv}")
            all_ok = False
    if all_ok:
        print("  [PASS] metadata consistency")
    return all_ok


def compare_files(ref_path, test_path, tol):
    print(f"\n{'='*60}")
    print(f"Reference: {ref_path}")
    print(f"Test:      {test_path}")
    print(f"Tolerance: {tol:.0e}")
    print(f"{'='*60}")

    ref_data, ref_attrs, ref_task, ref_f = load_hdf5(ref_path)
    test_data, test_attrs, test_task, test_f = load_hdf5(test_path)

    results = []

    if ref_task != test_task:
        print(f"\n  [FAIL] task mismatch: {ref_task!r} vs {test_task!r}")
        ref_f.close()
        test_f.close()
        print(f"\n{'='*60}\nRESULT: SOME FAILED\n{'='*60}\n")
        return False

    print("\n--- Metadata ---")
    results.append(compare_metadata(ref_attrs, test_attrs))

    print(f"\n--- Data Arrays (task={ref_task}) ---")
    if ref_task == "dynamics":
        ok, _ = compare_arrays("time", ref_data["time"], test_data["time"], tol=1e-15)
        results.append(ok)
        ok, _ = compare_arrays("density_matrix_real",
                               ref_data["rho_real"], test_data["rho_real"], tol)
        results.append(ok)
        ok, _ = compare_arrays("density_matrix_imag",
                               ref_data["rho_imag"], test_data["rho_imag"], tol)
        results.append(ok)

        print("\n--- Trace Conservation ---")
        results.append(compare_trace(ref_data["rho_real"], test_data["rho_real"], tol))

        print("\n--- Population Summary (first & last timestep) ---")
        N = ref_data["rho_real"].shape[1]
        for label, data in [("ref", ref_data), ("test", test_data)]:
            pop_first = np.array([data["rho_real"][0, i, i] for i in range(N)])
            pop_last = np.array([data["rho_real"][-1, i, i] for i in range(N)])
            t_last = data["time"][-1]
            print(f"  {label} t=0:     " + "  ".join(f"{p:.6f}" for p in pop_first))
            print(f"  {label} t={t_last:.0f}fs: " + "  ".join(f"{p:.6f}" for p in pop_last[:min(N,10)]))

    elif ref_task == "absorption":
        ok, _ = compare_arrays("time", ref_data["time"], test_data["time"], tol=1e-15)
        results.append(ok)
        ok, _ = compare_arrays("rho_xyz_real",
                               ref_data["rho_xyz_real"], test_data["rho_xyz_real"], tol)
        results.append(ok)
        ok, _ = compare_arrays("rho_xyz_imag",
                               ref_data["rho_xyz_imag"], test_data["rho_xyz_imag"], tol)
        results.append(ok)

    elif ref_task == "2des":
        for axis in ("tau_fs", "tw_fs", "t_fs"):
            ok, _ = compare_arrays(axis, ref_data[axis], test_data[axis], tol=1e-15)
            results.append(ok)
        ok, _ = compare_arrays("resp_real",
                               ref_data["resp_real"], test_data["resp_real"], tol)
        results.append(ok)
        ok, _ = compare_arrays("resp_imag",
                               ref_data["resp_imag"], test_data["resp_imag"], tol)
        results.append(ok)

    ref_f.close()
    test_f.close()

    all_passed = all(results)
    print(f"\n{'='*60}")
    print(f"RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'='*60}\n")

    return all_passed


def run_all(tol):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.join(script_dir, "benchmark_data")

    # (ref_h5, config_yaml, task): the task drives both the solver subcommand
    # and which output.<task> block gets the temp save path. Same-build reruns
    # are bit-identical, but comparing committed baselines against a freshly
    # built binary is cross-build and carries ~1e-15 float/codegen residual, so
    # the gate is a tolerance check (--tol), NOT bitwise (max|Δ|==0).
    benchmarks = [
        ("benchmark_fmo7.h5",             "benchmark_fmo7.yaml",             "dynamics"),    # EXCITED_ONLY  (N=7)
        ("benchmark_fmo7_n29.h5",         "benchmark_fmo7_n29.yaml",         "dynamics"),    # DOUBLE_EXCITED (N=29)
        ("benchmark_fmo7_with_ground.h5", "benchmark_fmo7_with_ground.yaml", "dynamics"),    # WITH_GROUND   (N=8)
        ("benchmark_absorption_fmo7.h5",  "benchmark_absorption_fmo7.yaml",  "absorption"),  # disorder-OFF absorption (N=8)
        ("benchmark_3sites_2des.h5",      "benchmark_3sites_2des.yaml",      "2des"),        # 3-site 2DES
    ]

    all_passed = True
    for ref_name, config_name, task in benchmarks:
        ref_path = os.path.join(benchmark_dir, ref_name)
        if not os.path.exists(ref_path):
            print(f"[SKIP] {ref_name} not found")
            continue

        test_path = os.path.join(benchmark_dir, f"test_{ref_name}")
        config_path = os.path.join(script_dir, "configs", config_name)

        if not os.path.exists(config_path):
            print(f"[SKIP] config {config_name} not found")
            continue

        print(f"\n>>> Running solver with {config_name}...")
        solver = os.environ.get("HEOM_SOLVER", os.path.join(script_dir, "..", "build", "heom_solver"))

        import tempfile, yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        # Schema: each task block carries its own nested `output:` sub-block.
        # Redirect only this task's block so the temp run writes to test_<ref>
        # without touching other blocks.
        out = cfg.setdefault(task, {}).setdefault("output", {})
        out["save_path"] = os.path.dirname(test_path) + "/"
        out["filename"] = os.path.splitext(os.path.basename(test_path))[0]
        out["use_timestamp"] = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name

        # heom_solver now requires a subcommand (no implicit
        # 'dynamics' is the default task.
        ret = os.system(f"{solver} {task} {tmp_path}")
        os.unlink(tmp_path)

        if ret != 0:
            print(f"[FAIL] solver returned {ret}")
            all_passed = False
            continue

        passed = compare_files(ref_path, test_path, tol)
        if not passed:
            all_passed = False

        if os.path.exists(test_path):
            os.unlink(test_path)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="HEOM benchmark comparison")
    parser.add_argument("files", nargs="*", help="reference.h5 test.h5")
    parser.add_argument("--tol", type=float, default=1e-12,
                        help="tolerance for element-wise comparison (default: 1e-12)")
    parser.add_argument("--run-all", action="store_true",
                        help="run all benchmarks and compare")

    args = parser.parse_args()

    if args.run_all:
        passed = run_all(args.tol)
        sys.exit(0 if passed else 1)
    elif len(args.files) == 2:
        passed = compare_files(args.files[0], args.files[1], args.tol)
        sys.exit(0 if passed else 1)
    elif len(args.files) == 1:
        print("Self-check mode: verifying single file integrity")
        data, attrs, task, f = load_hdf5(args.files[0])
        print(f"  File: {args.files[0]}")
        print(f"  task: {task}")
        if task == "dynamics":
            N = data["rho_real"].shape[1]
            n_steps = data["rho_real"].shape[0]
            print(f"  n_sites={N}, timesteps={n_steps}, t=[{data['time'][0]:.1f}, {data['time'][-1]:.1f}] fs")

            traces = np.array([np.trace(data["rho_real"][t]) for t in range(n_steps)])
            print(f"  Tr(rho): min={traces.min():.10f}, max={traces.max():.10f}")
            print(f"  max|Tr-1| = {np.max(np.abs(traces - 1.0)):.2e}")

            pop_last = [data["rho_real"][-1, i, i] for i in range(N)]
            print(f"  Final populations: " + "  ".join(f"{p:.6f}" for p in pop_last[:10]))
            if N > 10:
                print(f"    ... ({N - 10} more)")
        elif task == "absorption":
            n_dir, n_steps, N, _ = data["rho_xyz_real"].shape
            print(f"  n_sites={N}, timesteps={n_steps}, dirs={n_dir}, "
                  f"t=[{data['time'][0]:.1f}, {data['time'][-1]:.1f}] fs")
        elif task == "2des":
            print(f"  resp shape: {data['resp_real'].shape}")
            print(f"  n_tau={len(data['tau_fs'])}, n_tw={len(data['tw_fs'])}, "
                  f"n_t={len(data['t_fs'])}")
        f.close()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
