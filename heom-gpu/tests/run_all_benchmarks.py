#!/usr/bin/env python3
"""
Generate and run all FMO-7 benchmark cases:
  - site 1-7 with ADO filtering
  - site 1-7 without filtering
"""

import os
import sys
import time
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SOLVER = os.path.join(PROJECT_DIR, "build", "heom_solver")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "benchmark_data")

TEMPLATE = """\
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
  start_step: {filter_start_step}
  warmup_factor: {filter_warmup_factor}

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


def run_case(site, use_filter, label):
    tag = "filtered" if use_filter else "nofilter"
    h5_name = f"benchmark_fmo7_{tag}_site{site}.h5"
    h5_path = os.path.join(OUTPUT_DIR, h5_name)

    if os.path.exists(h5_path):
        print(f"  [SKIP] {h5_name} already exists")
        return h5_name, 0.0, True

    # filter.enabled is the explicit on/off switch; when false the knobs below
    # are ignored (still valid values so the parser accepts them).
    config_content = TEMPLATE.format(
        filter_enabled="true" if use_filter else "false",
        tolerance="1.0e-7",
        print_interval="500" if use_filter else "0",
        filter_start_step="10",
        filter_warmup_factor="1.0e-3",
        site=site,
        output_dir=os.path.dirname(h5_path) + "/",
        output_name=os.path.splitext(os.path.basename(h5_path))[0],
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=OUTPUT_DIR
    ) as f:
        f.write(config_content)
        tmp_path = f.name

    try:
        t0 = time.time()
        # heom_solver requires an explicit subcommand.
        # benchmark_*.yaml configs are dynamics runs; spectroscopy benches
        # would call ['absorption', tmp_path] on a different config family.
        result = subprocess.run(
            [SOLVER, "dynamics", tmp_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  [FAIL] {label} (exit={result.returncode})")
            if result.stderr:
                print(f"         {result.stderr[:200]}")
            return h5_name, elapsed, False

        for line in result.stdout.splitlines():
            if "Final trace:" in line:
                trace_val = line.split(":")[-1].strip()
                print(f"  [OK]   {h5_name}  ({elapsed:.1f}s, Tr={trace_val})")
                break
        else:
            print(f"  [OK]   {h5_name}  ({elapsed:.1f}s)")

        return h5_name, elapsed, True

    finally:
        os.unlink(tmp_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(SOLVER):
        print(f"ERROR: solver not found at {SOLVER}")
        sys.exit(1)

    results = []

    print("=== FMO-7 with ADO filtering (tol=1e-7) ===")
    for site in range(1, 8):
        name, t, ok = run_case(site, use_filter=True,
                               label=f"site {site} filtered")
        results.append((name, t, ok))

    print("\n=== FMO-7 without filtering (tol=0) ===")
    for site in range(1, 8):
        name, t, ok = run_case(site, use_filter=False,
                               label=f"site {site} nofilter")
        results.append((name, t, ok))

    print(f"\n{'='*50}")
    total_time = sum(r[1] for r in results)
    n_ok = sum(1 for r in results if r[2])
    n_skip = sum(1 for r in results if r[1] == 0.0 and r[2])
    print(f"Done: {n_ok}/{len(results)} passed, {n_skip} skipped")
    print(f"Total time: {total_time:.1f}s")

    failed = [r for r in results if not r[2]]
    if failed:
        print(f"FAILED: {[r[0] for r in failed]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
