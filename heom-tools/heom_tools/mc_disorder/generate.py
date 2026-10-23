#!/usr/bin/env python3
"""MC static-disorder node generator.

Writes a `disorder_realizations.h5` weighted-node file that the solver reads
row-by-row.

`iid` is Gaussian (estimator="mean"). Each row r is a pure function
of (master_seed, r) via SeedSequence([seed, r]) -> the file can be appended /
refined without changing any existing row (row r is identical whether N=8 or
N=16), which is what makes parallel claim + incremental convergence safe.

Invoke as the console script ``mc-disorder-gen``, as ``python -m
heom_tools.mc_disorder``, or by file path (self-contained, needs only
numpy + h5py) -- the last form is how the solver test suite calls it.

Usage:
    # Generate:
    mc-disorder-gen \\
        --out disorder_realizations.h5 --n-node 64 --n-site 7 \\
        --fwhm 100.0 [--per-site "3:140,5:120"] --seed 20260625 --method iid

    # Extend (inherit seed/fwhm/n_site from FILE; only --n-node needed). Defaults --out to the extended file:
    mc-disorder-gen --extend disorder_realizations.h5 --n-node 128

    # RQMC: one command, R replica files + manifest (no shell for-loop):
    mc-disorder-gen --out config/nodes_rqmc --method rqmc --n-replicas 4 \\
        --n-node 256 --n-site 7 --fwhm 100.0 --seed 1
    # -> config/nodes_rqmc_r0.h5 ... _r3.h5 + config/nodes_rqmc.manifest.yaml

    # Print an existing file's sampling params and exit:
    mc-disorder-gen --print-params disorder_realizations.h5
"""
import argparse
import os
import sys
from itertools import product
from math import comb

import numpy as np
import h5py


def _manifest_module():
    """Sibling import so ``python generate.py`` works without package install."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.py")
    spec = importlib.util.spec_from_file_location("mc_disorder_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Gaussian FWHM -> sigma conversion (single source of truth; cross-checked in
# this package's tests/ and the solver C++ side).
SIGMA_PER_FWHM = 1.0 / 2.354820045031


def parse_per_site(spec, n_site, default_fwhm):
    """Return fwhm_cm[n_site] with 1-based pigment overrides from "3:140,5:120"."""
    fwhm = np.full(n_site, float(default_fwhm), dtype=np.float64)
    if not spec:
        return fwhm
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        idx_str, val_str = item.split(":")
        idx1 = int(idx_str)  # 1-based pigment number
        if not (1 <= idx1 <= n_site):
            raise ValueError(f"--per-site index {idx1} out of range [1, {n_site}]")
        fwhm[idx1 - 1] = float(val_str)
    return fwhm


def generate_iid(n_node, n_site, sigma_cm, master_seed):
    """delta_eps[n_node, n_site], per-row-independent iid Gaussian (cm^-1)."""
    delta = np.empty((n_node, n_site), dtype=np.float64)
    for r in range(n_node):
        # Per-row RNG so row r is invariant to n_node (append/refine safe).
        rng_r = np.random.default_rng(np.random.SeedSequence([int(master_seed), r]))
        delta[r, :] = sigma_cm * rng_r.standard_normal(n_site)
    return delta


def generate_antithetic(n_node, n_site, sigma_cm, master_seed):
    """delta_eps[n_node, n_site] as N/2 antithetic pairs (ξ, -ξ) (cm^-1).

    Row 2i = ξ_i, row 2i+1 = -ξ_i, with ξ_i a PURE FUNCTION of (master_seed,
    pair i) via SeedSequence([seed, i]) -> append-safe at PAIR granularity (the
    first N/2 pairs are bit-identical whether the file holds N or N' nodes).
    Gaussian symmetry makes the odd (linear-in-ξ) part of the response cancel
    within each pair, killing the largest variance block. Still estimator="mean",
    w = 1/N: the solver just averages all N rows, unaware of the pairing."""
    if n_node % 2 != 0:
        raise ValueError("antithetic requires an even --n-node (N/2 ± pairs)")
    delta = np.empty((n_node, n_site), dtype=np.float64)
    for i in range(n_node // 2):
        rng_i = np.random.default_rng(np.random.SeedSequence([int(master_seed), i]))
        xi = sigma_cm * rng_i.standard_normal(n_site)
        delta[2 * i, :] = xi
        delta[2 * i + 1, :] = -xi
    return delta


def generate_rqmc(n_node, n_site, sigma_cm, master_seed, replica=0):
    """delta_eps[n_node, n_site] from a SCRAMBLED Sobol sequence (cm^-1).

    Randomized quasi-Monte Carlo: a low-discrepancy Sobol point set in [0,1)^d,
    Owen-scrambled with seed=(master_seed, replica), mapped through the inverse
    normal CDF and scaled by σ. For bounded-variation integrands this converges
    toward 1/N (vs iid's 1/√N). Each `replica` is an INDEPENDENT scramble — the
    honest RQMC error is the spread across replicas, not a 1/sqrt(N) error
    on one set. estimator stays "mean", w=1/N. The solver and the merge are unchanged;
    `replica` rides in the `group` column so merge can average per replica.

    Sobol is balanced at powers of two; n_node need not be 2^m but a 2^m prefix
    is the low-discrepancy sweet spot (the probe sweeps 2^k)."""
    try:
        from scipy.stats import qmc, norm
    except ImportError as e:
        raise ImportError(
            "heom-tools requires scipy; reinstall the package"
        ) from e
    rng = np.random.default_rng(
        np.random.SeedSequence([int(master_seed), int(replica)]))
    engine = qmc.Sobol(d=int(n_site), scramble=True, seed=rng)
    u = engine.random(int(n_node))                  # [n_node, n_site] in (0,1)
    z = norm.ppf(u)                                 # standard normal
    return sigma_cm * z


def hermegauss_1d(n):
    """1D probabilist-Hermite (Gauss) rule for the standard normal N(0,1).

    Returns (nodes_std, weights_norm) with Σ weights_norm = 1, EXACT for
    polynomials up to degree 2n-1 under the N(0,1) measure. nodes_std are the
    standard-normal abscissae x_i; the physical offset is ξ = σ·x_i. numpy's
    hermegauss integrates against e^{-x²/2} (Σ w_i = √(2π)), so we divide by
    √(2π) to normalise to a probability measure."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / np.sqrt(2.0 * np.pi)


def _smolyak_indices(d, q):
    """Multi-indices k (each k_j ≥ 1, |k|₁ ≤ q) that survive the Smolyak
    combination, paired with coefficient (-1)^(q-|k|)·C(d-1, q-|k|). Range
    q-d+1 ≤ |k|₁ ≤ q; the k_j ≥ 1 floor makes |k|₁ ≥ d automatic."""
    lo = q - d + 1
    for k in product(range(1, q - d + 2), repeat=d):
        s = sum(k)
        if lo <= s <= q:
            yield k, (-1) ** (q - s) * comb(d - 1, q - s)


def sparse_gh(n_site, level, sigma_cm):
    """Smolyak sparse Gauss–Hermite nodes for `n_site` independent N(0, σ_i²)
    dims (no cross-correlation). `level` is the RELATIVE level ℓ ≥ 0 (ℓ=0 → a
    single centre node with w=1); the internal absolute level is q = n_site + ℓ.
    The 1D rule at index k has m(k)=k points (non-nested, simplest correct).

    Returns (delta_eps[N, n_site], weight[N]) with Σ weight = 1 and UNIQUE nodes
    (duplicate coordinates merged, weights summed). Pure function of
    (n_site, level, sigma_cm) — no RNG, bit-reproducible."""
    d = int(n_site)
    q = d + int(level)
    sigma_cm = np.asarray(sigma_cm, dtype=np.float64)
    rule = {m: hermegauss_1d(m) for m in range(1, q - d + 2)}  # cache 1D rules

    acc = {}  # quantised std-normal coord -> (coord tuple, summed weight)
    for k, coef in _smolyak_indices(d, q):
        axes = [rule[m] for m in k]
        for idx in product(*(range(len(a[0])) for a in axes)):
            coord = tuple(axes[j][0][idx[j]] for j in range(d))
            w = float(coef)
            for j in range(d):
                w *= axes[j][1][idx[j]]
            # Quantise the key so the shared x=0 node (present in every odd rule,
            # bit-identical) merges; non-zero nodes never collide across rules.
            key = tuple(round(c, 12) for c in coord)
            if key in acc:
                acc[key][1] += w
            else:
                acc[key] = [coord, w]

    coords = np.array([v[0] for v in acc.values()], dtype=np.float64)
    weights = np.array([v[1] for v in acc.values()], dtype=np.float64)
    order = np.lexsort(coords.T[::-1])  # canonical lexicographic node order
    coords, weights = coords[order], weights[order]
    delta_eps = coords * sigma_cm[None, :]  # ξ = σ·x, per-site σ
    return delta_eps, weights


def write_file(out, delta_eps, weight, group, estimator, generator,
               master_seed, fwhm_cm, sigma_cm, sg_level=None):
    with h5py.File(out, "w") as h:
        h.create_dataset("delta_eps", data=delta_eps)
        h.create_dataset("weight", data=weight)
        h.create_dataset("group", data=group)
        h.attrs["estimator"] = estimator
        h.attrs["generator"] = generator
        h.attrs["master_seed"] = np.uint64(master_seed)
        h.attrs["fwhm_cm"] = fwhm_cm
        h.attrs["sigma_cm"] = sigma_cm
        if sg_level is not None:
            h.attrs["sg_level"] = np.int32(sg_level)


def read_params(path):
    """Read an existing node file's sampling params back into a dict."""
    with h5py.File(path, "r") as h:
        return {
            "generator": str(h.attrs["generator"]),
            "estimator": str(h.attrs["estimator"]),
            "master_seed": int(h.attrs["master_seed"]),
            "fwhm_cm": np.array(h.attrs["fwhm_cm"], dtype=np.float64),
            "sigma_cm": np.array(h.attrs["sigma_cm"], dtype=np.float64),
            "n_node": int(h["delta_eps"].shape[0]),
            "n_site": int(h["delta_eps"].shape[1]),
        }


def print_params(path):
    p = read_params(path)
    print(f"file        : {path}")
    print(f"generator   : {p['generator']}")
    print(f"estimator   : {p['estimator']}")
    print(f"master_seed : {p['master_seed']}")
    print(f"n_node      : {p['n_node']}")
    print(f"n_site      : {p['n_site']}")
    print(f"fwhm_cm     : {p['fwhm_cm'].tolist()}")
    print(f"sigma_cm    : {p['sigma_cm'].tolist()}")
    with h5py.File(path, "r") as h:
        if "sg_level" in h.attrs:
            print(f"sg_level    : {int(h.attrs['sg_level'])}")


def out_stem(path):
    """Strip a trailing .h5 so --out can name a multi-replica stem."""
    if path.lower().endswith(".h5"):
        return path[:-3]
    return path


def replica_path(stem, replica):
    """<stem>_r<repl>.h5 naming for RQMC multi-replica bundles."""
    return f"{stem}_r{replica}.h5"


def _write_rqmc_replica(out_path, n_node, n_site, sigma_cm, fwhm_cm, seed, replica):
    delta_eps = generate_rqmc(n_node, n_site, sigma_cm, seed, replica=replica)
    generator = f"sobol_gauss@v1|repl={replica}"
    group = np.full(n_node, replica, dtype=np.int32)
    weight = np.full(n_node, 1.0 / n_node, dtype=np.float64)
    write_file(out_path, delta_eps, weight, group,
               estimator="mean", generator=generator,
               master_seed=seed, fwhm_cm=fwhm_cm, sigma_cm=sigma_cm)
    return out_path


def _write_rqmc_manifest(manifest_path, **kwargs):
    try:
        from heom_tools.mc_disorder.manifest import write_rqmc_manifest
    except ImportError:
        write_rqmc_manifest = _manifest_module().write_rqmc_manifest
    write_rqmc_manifest(manifest_path, **kwargs)


def _rollback_rqmc_bundle(paths, manifest):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
    if manifest:
        try:
            os.remove(manifest)
        except OSError:
            pass


def _cmd_rqmc_bundle(p, args, out, n_site, sigma_cm, fwhm_cm, seed):
    """Write R replica files + optional manifest; rollback partial output on error."""
    stem = out_stem(out)
    paths = []
    manifest = None
    try:
        for r in range(args.n_replicas):
            op = replica_path(stem, r)
            _write_rqmc_replica(op, args.n_node, n_site, sigma_cm, fwhm_cm, seed, r)
            paths.append(op)
            print(f"wrote {op}: method=rqmc, repl={r}, n_node={args.n_node}, "
                  f"n_site={n_site}, seed={seed}")
        if not args.no_manifest:
            manifest = f"{stem}.manifest.yaml"
            _write_rqmc_manifest(
                manifest,
                master_seed=seed,
                n_node=args.n_node,
                n_site=n_site,
                fwhm_cm=fwhm_cm,
                sigma_cm=sigma_cm,
                n_replicas=args.n_replicas,
                draws_files=paths,
            )
            print(f"manifest: {manifest}")
    except (OSError, ImportError, RuntimeError, ValueError):
        _rollback_rqmc_bundle(paths, manifest)
        print("[mc-disorder-gen] rqmc bundle write failed; "
              "partial files removed — retry after fixing the error",
              file=sys.stderr)
        raise
    print(f"wrote {args.n_replicas} RQMC replica files (stem={stem})")
    return 0


def _cmd_sparse_gh(p, args):
    if args.extend:
        p.error("--extend does not apply to sparse_gh: quadrature refinement "
                "is a NEW node set (different disorder_identity) — regenerate "
                "with a higher --level instead")
    if args.level is None or args.level < 0:
        p.error("sparse_gh requires --level L >= 0 (relative Smolyak level; "
                "0 = single centre node)")
    missing = [f for f, v in (("--out", args.out), ("--n-site", args.n_site),
                              ("--fwhm", args.fwhm)) if v is None]
    if missing:
        p.error("missing required args for sparse_gh: " + ", ".join(missing))
    if args.n_site < 1:
        p.error("--n-site must be >= 1")
    fwhm_cm = parse_per_site(args.per_site, args.n_site, args.fwhm)
    sigma_cm = fwhm_cm * SIGMA_PER_FWHM
    delta_eps, weight = sparse_gh(args.n_site, args.level, sigma_cm)
    group = np.zeros(len(weight), dtype=np.int32)
    write_file(args.out, delta_eps, weight, group,
               estimator="quadrature", generator="sparse_gh@v1",
               master_seed=0, fwhm_cm=fwhm_cm, sigma_cm=sigma_cm,
               sg_level=args.level)
    print(f"wrote {args.out}: sparse_gh level={args.level}, "
          f"n_node={len(weight)}, n_site={args.n_site}")
    return 0


def _resolve_generation_params(p, args):
    """Return (n_site, fwhm_cm, sigma_cm, seed, out) for iid/antithetic/rqmc."""
    if args.method in ("antithetic", "rqmc") and args.extend:
        p.error(f"--extend does not apply to {args.method}: regenerate with a "
                "larger --n-node (deterministic prefix preserved)")

    if args.extend:
        src = read_params(args.extend)
        if src["generator"] != "iid_gauss@v1":
            p.error(f"--extend only supports iid files (got generator="
                    f"'{src['generator']}')")
        if args.n_node is None or args.n_node < 1:
            p.error("--extend requires --n-node >= 1 (the new total node count)")
        out = args.out if args.out else args.extend
        if args.n_node < src["n_node"]:
            print(f"note: --n-node {args.n_node} < existing {src['n_node']}; "
                  "shrinking (prefix kept)", file=sys.stderr)
        return (src["n_site"], src["fwhm_cm"], src["sigma_cm"],
                src["master_seed"], out)

    missing = [f for f, v in (("--out", args.out), ("--n-node", args.n_node),
                              ("--n-site", args.n_site), ("--fwhm", args.fwhm))
               if v is None]
    if missing:
        p.error("missing required args for generation: " + ", ".join(missing) +
                " (or use --extend FILE / --print-params FILE)")
    if args.n_node < 1 or args.n_site < 1:
        p.error("--n-node and --n-site must be >= 1")
    fwhm_cm = parse_per_site(args.per_site, args.n_site, args.fwhm)
    sigma_cm = fwhm_cm * SIGMA_PER_FWHM
    return args.n_site, fwhm_cm, sigma_cm, args.seed, args.out


def _cmd_mean_sampler(p, args):
    """iid / antithetic / rqmc (single file or multi-replica bundle)."""
    n_site, fwhm_cm, sigma_cm, seed, out = _resolve_generation_params(p, args)

    if args.method == "rqmc" and args.n_replicas > 1:
        return _cmd_rqmc_bundle(p, args, out, n_site, sigma_cm, fwhm_cm, seed)

    if args.method == "iid" and not args.extend:
        print("[mc-disorder-gen] note: production absorption usually uses "
              "--method rqmc --n-replicas 4; iid is the CLI default / baseline",
              file=sys.stderr)

    if args.method == "antithetic":
        if args.n_node % 2 != 0:
            p.error("antithetic requires an even --n-node (N/2 ± pairs)")
        delta_eps = generate_antithetic(args.n_node, n_site, sigma_cm, seed)
        generator = "antithetic_gauss@v1"
        group = np.zeros(args.n_node, dtype=np.int32)
    elif args.method == "rqmc":
        replica = args.replica if args.replica is not None else 0
        delta_eps = generate_rqmc(args.n_node, n_site, sigma_cm, seed,
                                  replica=replica)
        generator = f"sobol_gauss@v1|repl={replica}"
        group = np.full(args.n_node, replica, dtype=np.int32)
    else:
        delta_eps = generate_iid(args.n_node, n_site, sigma_cm, seed)
        generator = "iid_gauss@v1"
        group = np.zeros(args.n_node, dtype=np.int32)

    weight = np.full(args.n_node, 1.0 / args.n_node, dtype=np.float64)
    write_file(out, delta_eps, weight, group,
               estimator="mean", generator=generator,
               master_seed=seed, fwhm_cm=fwhm_cm, sigma_cm=sigma_cm)
    print(f"wrote {out}: method={args.method}, n_node={args.n_node}, "
          f"n_site={n_site}, seed={seed}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="MC static-disorder node generator")
    p.add_argument("--out", help="output HDF5 path (extend defaults to the extended file)")
    p.add_argument("--n-node", type=int, help="number of nodes (rows)")
    p.add_argument("--n-site", type=int, help="pigments (= n_states); normal mode only")
    p.add_argument("--fwhm", type=float, help="global per-site FWHM (cm^-1); normal mode only")
    p.add_argument("--per-site", default="", help='1-based overrides, e.g. "3:140,5:120"')
    p.add_argument("--seed", type=int, default=0, help="master_seed (iid/RQMC)")
    p.add_argument("--method", default="iid",
                   choices=("iid", "antithetic", "rqmc", "sparse_gh"),
                   help="sampler: iid (mean estimator) | antithetic (±ξ pairs, "
                        "mean estimator) | rqmc (scrambled Sobol, mean "
                        "estimator; --replica for the scramble id) | sparse_gh "
                        "(Smolyak Gauss-Hermite quadrature)")
    p.add_argument("--replica", type=int, default=None,
                   help="rqmc single-file mode: scramble/replica id (default 0). "
                        "Mutually exclusive with --n-replicas > 1.")
    p.add_argument("--n-replicas", type=int, default=1, dest="n_replicas",
                   metavar="R",
                   help="rqmc only: write R files <stem>_r0.h5.._r{R-1}.h5 plus "
                        "a manifest (default 1 = single --out file)")
    p.add_argument("--no-manifest", action="store_true",
                   help="with --n-replicas > 1, skip writing <stem>.manifest.yaml")
    p.add_argument("--level", type=int,
                   help="sparse_gh relative Smolyak level ℓ >= 0 (0 = single "
                        "centre node); node count grows with ℓ. Ignored by iid.")
    p.add_argument("--extend", metavar="FILE",
                   help="inherit seed/fwhm/n_site from FILE; only --n-node needed. "
                        "The shared prefix stays bit-identical.")
    p.add_argument("--print-params", metavar="FILE", dest="print_params",
                   help="print an existing node file's sampling params and exit")
    args = p.parse_args(argv)

    if args.n_replicas < 1:
        p.error("--n-replicas must be >= 1")
    if args.n_replicas > 1:
        if args.method != "rqmc":
            p.error("--n-replicas > 1 requires --method rqmc")
        if args.replica is not None:
            p.error("--replica and --n-replicas > 1 are mutually exclusive")

    # Mode 1: inspect only.
    if args.print_params:
        print_params(args.print_params)
        return 0

    if args.method == "sparse_gh":
        return _cmd_sparse_gh(p, args)

    return _cmd_mean_sampler(p, args)


if __name__ == "__main__":
    sys.exit(main())
