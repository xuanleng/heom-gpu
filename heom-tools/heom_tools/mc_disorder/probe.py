#!/usr/bin/env python3
"""Compare sparse-grid quadrature with iid Monte Carlo on absorption.

Goal (de-risk before expensive 2des): measure how many solver evaluations each
sampler needs to reach a given accuracy on the SAME FMO absorption. If the
Gaussian-smoothed line shape is smooth in the disorder coordinates, Smolyak
quadrature (sparse_gh) hits target accuracy with far fewer nodes than iid MC —
the "quadrature dividend". If it is NOT smooth (large σ / resonances crossing),
the curves cross and the report recommends falling back to RQMC/antithetic
Both outcomes are a successful probe.

Pipeline per node set (reuses the whole existing stack, nothing re-implemented):
  1. generate a disorder node file (sparse grid or iid)
  2. mc-disorder-run absorption  → per-worker shards         [NEEDS GPU]
  3. merge shards                → single-shot .h5 (⟨ρ_xyz⟩)
  4. las-post core               → (ω, intensity) spectrum
  5. L² relative diff vs a reference spectrum → one (evals, error) point

Run every set with the same build and
fix it to fp64 — mixed-FP32 has a ~1e-4 arithmetic floor that would pollute the
sampling-error metric. The probe measures SAMPLING error vs evaluation count,
so arithmetic noise must be out of the picture.

The solver run (step 2) needs a GPU:
    mc-probe absorption config/fmo.yaml --levels 0 1 2 3 --iid 1 8 27 64 \\
             --workers 4 --out smoothness-probe.csv

-----------------------------------------------------------------------------
Variance-reduction arena (task "arena"):

If the line shape is not smooth in the disorder, the arena
tests variance-reduction methods that DON'T assume smoothness — antithetic
variates and a control variate built from the cheap stick surrogate — against
an independent large-N iid reference. It runs
a per-node iid sweep ONCE (N single-node solver runs, solver untouched) and
DERIVES the iid and CV convergence curves from the same per-node spectra, so the
comparison is maximally fair. The killer metric ρ = corr(O_HEOM, c_stick) is
printed first — it caps the CV dividend at 1/(1−ρ²), telling you the ceiling
before you read the whole curve.

    mc-probe arena config/fmo.yaml --fwhm 100 --sweep 8 16 32 64 128 \\
             --ref-n 512 --workers 4 --out arena.csv

Invoke as ``mc-probe``, ``python -m heom_tools.mc_disorder.probe``, or by path.
"""
from .probe_core import (  # noqa: F401
    l2_relative_diff, eff_variance, eff_variance_antithetic, fixed_beta,
    cv_residual, conv_amplitude, n_star, analyze_pool)
from .probe_absorption import run_probe  # noqa: F401
from .probe_arena import run_arena  # noqa: F401
from .probe_converge import run_converge  # noqa: F401
from .probe_rqmc import run_rqmc  # noqa: F401
from .probe_cli import _build_parser, main  # noqa: F401

if __name__ == "__main__":
    import sys
    sys.exit(main())
