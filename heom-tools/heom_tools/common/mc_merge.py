"""MC static-disorder shard merge (task-agnostic).

Each MC worker writes a single-shot-format spectrum file with a ``/mc``
provenance group. Two estimators share the file shape but NOT the data flow;
the ``/mc/estimator`` attr selects the branch:

  mean (iid MC)::
    /results/<ds>    this worker's disorder MEAN  (Σ_r O_r / count)
    /mc/count        int64    this worker's realization count
    /mc/sum_obs_sq   float64  Σ_r |O_r|²  (same shape as /results/<ds>)
  → merge = count-weighted average of per-worker means + a SEM from pooled Σ|O|².

  quadrature (sparse_gh)::
    /results/<ds>    partial weighted sum PLACEHOLDER (not normalised)
    /mc/count        int64    claimed-node count (bookkeeping only)
    /mc/sum_obs      complex128  Σ_r w_r·O_r  (absolute weights, Σ_global=1)
  → merge = plain SUM of per-shard /mc/sum_obs (NO count divide, NO SEM).

Both branches are guarded by four invariants: no realization id counted twice,
one physical config, one disorder set, one precision mode — plus one estimator
(mean and quadrature shards never mix). The merged file is written back in the
SAME single-shot format (dropping the ``_rank`` tag) so ``las-post`` reads it
unchanged. Legacy shards (no ``estimator`` attr) are treated as ``mean``.

RQMC (slice 4b) reuses the ``mean`` estimator per replica (solver/merge
untouched for the point estimate), but its honest error is the ACROSS-REPLICA
spread, not a √N SEM. So ``combine_replicas`` takes R independent merged replica
means (each run in its own dir) and produces the ``rqmc`` estimator: mean-of-
means + a per-ω across-replica SE. This is a file-level combine, not an intra-
shard group branch — the C++ shard never learns about replicas, keeping the
solver dumb (the group column is only a bookkeeping id).
"""
from __future__ import annotations

import glob
import os
import shutil
import warnings

import numpy as np
import h5py

from heom_tools.common import mc_naming

MC_GROUP = "/mc"


def _read_attr(group, name):
    v = group.attrs.get(name)
    return v.decode() if isinstance(v, bytes) else v


def merge_mc_shards(paths, results_dataset):
    """Merge MC shards. `results_dataset` is the /results dataset name
    ("rho_xyz" for absorption, "resp" for 2des). Branches on the /mc estimator:

      mean       count-weighted average of per-worker means + SEM. Returns
                 mean / sem / sum_obs_sq / se_l2 (+ common keys).
      quadrature plain sum of per-shard /mc/sum_obs (weights absolute, Σ=1).
                 Returns mean (= the estimator ⟨O⟩) / sum_obs (+ common keys);
                 no sem/sum_obs_sq (quadrature has no √N SE semantics).

    Common keys always present: estimator / count / disorder_identity /
    config_fingerprint / precision_mode / r_done / n_shards."""
    if not paths:
        raise ValueError("no MC shards to merge")
    ds_path = f"/results/{results_dataset}"

    identity = fingerprint = precision = estimator = None
    total_count = 0
    acc_mean = None      # mean: Σ_i mean_i·count_i ; quadrature: Σ_i sum_obs_i
    acc_sq = None        # mean only: Σ_i sum_obs_sq_i
    r_done = []

    for f in paths:
        with h5py.File(f, "r") as p:
            if MC_GROUP not in p:
                raise ValueError(
                    f"'{f}' has no /mc group; not an MC shard (looks like a "
                    "plain single-shot file, or the wrong input)")
            mc = p[MC_GROUP]
            this_id = _read_attr(mc, "disorder_identity")
            this_fp = _read_attr(mc, "config_fingerprint")
            # precision_mode is a ROOT attr (a build choice, not physics, so NOT
            # in the merge key). Legacy shards predate it and were all
            # FP64, so a missing attr means "fp64". Guard #4: never average FP64
            # and mixed-FP32 shards together (silent-mix defence).
            this_prec = _read_attr(p, "precision_mode") or "fp64"
            # estimator attr predates only legacy mean shards → default "mean".
            this_est = _read_attr(mc, "estimator") or "mean"
            if identity is None:
                identity, fingerprint = this_id, this_fp
                precision, estimator = this_prec, this_est
            else:
                if this_id != identity:
                    raise ValueError(
                        f"disorder_identity mismatch: '{f}' has '{this_id}', "
                        f"expected '{identity}' (shards from a different "
                        "disorder set cannot be merged)")
                if this_fp != fingerprint:
                    raise ValueError(
                        f"config_fingerprint mismatch: '{f}' has '{this_fp}', "
                        f"expected '{fingerprint}' (physical config changed -> "
                        "rerun from scratch)")
                if this_prec != precision:
                    raise ValueError(
                        f"precision_mode mismatch: '{f}' is '{this_prec}', "
                        f"expected '{precision}' (FP64 and mixed-FP32 shards "
                        "cannot be averaged together -- they are different "
                        "numerical pipelines; re-run the odd batch with the "
                        "same solver binary)")
                if this_est != estimator:
                    raise ValueError(
                        f"estimator mismatch: '{f}' is '{this_est}', expected "
                        f"'{estimator}' (mean and quadrature are different "
                        "estimators and must never be merged together)")

            count = int(np.asarray(mc["count"]).reshape(-1)[0])
            total_count += count
            if estimator == "quadrature":
                # Absolute weights: the merged estimator is the plain sum of each
                # shard's partial weighted sum. No count divide.
                so = np.asarray(mc["sum_obs"][...], dtype=np.complex128)
                acc_mean = so if acc_mean is None else acc_mean + so
            else:
                mean_i = np.asarray(p[ds_path][...], dtype=np.complex128)
                contrib = mean_i * count
                acc_mean = contrib if acc_mean is None else acc_mean + contrib
                sq_i = np.asarray(mc["sum_obs_sq"][...], dtype=np.float64)
                acc_sq = sq_i if acc_sq is None else acc_sq + sq_i
            r_done.extend(int(x) for x in np.asarray(mc.attrs["r_done"]).reshape(-1))

    dup = len(r_done) - len(set(r_done))
    if dup:
        raise ValueError(
            f"{dup} realization id(s) counted by more than one shard "
            "(shards overlap -> double counting). Usual cause: stale shards from "
            "an earlier run of the SAME --range left next to the new ones (a "
            "re-run in a new minute makes a fresh timestamp, not an overwrite) -- "
            "delete the old <...>_r<a>-<b>_rank*.h5 and re-merge. If ranges truly "
            "overlap, the claim queue mis-partitioned r across workers.")
    if total_count <= 0:
        raise ValueError("total_count == 0: no realizations to average")

    common = {
        "estimator": estimator,
        "count": total_count,
        "disorder_identity": identity,
        "config_fingerprint": fingerprint,
        "precision_mode": precision,
        "r_done": np.array(sorted(r_done), dtype=np.int64),
        "n_shards": len(paths),
    }

    if estimator == "quadrature":
        # acc_mean already IS ⟨O⟩ (Σ w_r·O_r, Σw=1). No count divide, no SEM.
        return {**common, "mean": acc_mean, "sum_obs": acc_mean}

    mean = acc_mean / total_count
    var = np.maximum(acc_sq / total_count - np.abs(mean) ** 2, 0.0)
    sem = np.sqrt(var / total_count)
    return {
        **common,
        "mean": mean,
        "sem": sem,
        "sum_obs_sq": acc_sq,
        "se_l2": float(np.sqrt((sem ** 2).sum())),
    }


def combine_replicas(paths, results_dataset):
    """Combine R independent RQMC replica means into the RQMC estimator with the
    ONLY honest RQMC error — the ACROSS-REPLICA spread (never a √N SE on one
    scrambled Sobol set). RQMC uses estimator=mean
    (solver/merge untouched for the point estimate), so each replica is a normal
    mean-merge; the error semantics differ, so it lives here as a separate
    file-level combine rather than a √N SEM.

    Workflow: run each scramble replica in its OWN output dir → merge each
    (existing mean branch) → pass the R merged spectra here.

        point estimate = mean of the R replica means  (equal-M RQMC replicas)
        error SE(ω)    = sqrt( Σ_r |O_r(ω) − Ō(ω)|² / (R(R−1)) )   (per ω-bin)
        se_l2          = ‖SE‖₂   (convergence proxy)

    Guards: ≥2 replicas, one physical config, one precision, each a `mean` shard,
    and no two replica means bit-identical (same file passed twice / a copy)."""
    if len(paths) < 2:
        raise ValueError("combine_replicas needs >= 2 replica means "
                         "(RQMC error is the across-replica spread)")
    ds_path = f"/results/{results_dataset}"
    fingerprint = precision = None
    means, counts = [], []
    for f in paths:
        with h5py.File(f, "r") as p:
            if MC_GROUP not in p:
                raise ValueError(f"'{f}' has no /mc group; not an MC merged file")
            mc = p[MC_GROUP]
            this_fp = _read_attr(mc, "config_fingerprint")
            this_prec = _read_attr(p, "precision_mode") or "fp64"
            this_est = _read_attr(mc, "estimator") or "mean"
            if this_est != "mean":
                raise ValueError(
                    f"'{f}' is estimator='{this_est}'; replica combine expects "
                    "'mean' replica merges (RQMC uses the mean estimator)")
            if fingerprint is None:
                fingerprint, precision = this_fp, this_prec
            else:
                if this_fp != fingerprint:
                    raise ValueError(
                        f"config_fingerprint mismatch: '{f}' differs (replicas "
                        "must share one physical config)")
                if this_prec != precision:
                    raise ValueError(
                        f"precision_mode mismatch: '{f}' is '{this_prec}', "
                        f"expected '{precision}' (don't mix FP64/mixed-FP32)")
            means.append(np.asarray(p[ds_path][...], dtype=np.complex128))
            counts.append(int(np.asarray(mc["count"]).reshape(-1)[0]))

    stack = np.stack(means, axis=0)                       # [R, ...]
    R = stack.shape[0]
    for i in range(R):
        for j in range(i + 1, R):
            if np.array_equal(stack[i], stack[j]):
                raise ValueError(
                    f"replica {i} and {j} have identical means — same replica "
                    "passed twice (each --combine-replicas input must be a "
                    "DISTINCT scramble)")
    if len(set(counts)) != 1:
        import warnings
        warnings.warn(
            f"replica node counts differ ({sorted(set(counts))}); RQMC expects "
            "equal M per replica — using the unweighted mean-of-means and its "
            "across-replica SE anyway", stacklevel=2)

    mean = stack.mean(axis=0)
    # Across-replica SE per ω-bin: sample std of complex replica means / √R.
    dev2 = np.abs(stack - mean) ** 2
    replica_se = np.sqrt(dev2.sum(axis=0) / (R * (R - 1)))
    return {
        "estimator": "rqmc",
        "count": int(sum(counts)),
        "n_replica": R,
        "config_fingerprint": fingerprint,
        "precision_mode": precision,
        "mean": mean,
        "replica_se": replica_se,
        "se_l2": float(np.sqrt((replica_se ** 2).sum())),
        "r_done": np.array([], dtype=np.int64),
        "n_shards": R,
    }


def convergence_status(mean, se_l2, count, estimator, tol,
                       slope=None, n_replica=None):
    """Convergence verdict for a running/finished MC estimate. The
    self-convergence metric is the L² RELATIVE error ``rel = ‖SE‖₂ / ‖μ‖₂``;
    Parseval makes it identical in time/frequency, so no FFT is needed — the raw
    observable (``rho_xyz`` time series / ``resp`` map) is enough.

    Projects the evaluations needed to reach ``tol`` under the sampler's error
    law ``err(N) = rel · (N/count)^rate``:
      * mean / antithetic      rate = −0.5  (√N law)          → N* = count·(rel/tol)²
      * rqmc                   rate = ``slope`` if given (empirical, e.g. −0.91),
                               else a CONSERVATIVE −0.5 upper bound (real RQMC is
                               steeper ⇒ true N* is smaller).
      * quadrature             no √N SE semantics → status is undefined here.

    Returns: rel_err / converged / n_current / n_star / remaining / rate.
    `count` is the evaluation count (mean: N; rqmc: total R·M)."""
    if estimator == "quadrature":
        raise ValueError(
            "convergence_status is undefined for quadrature (no √N SE; use the "
            "level-to-level difference as the refinement criterion instead)")
    mean = np.asarray(mean, dtype=np.complex128)
    mean_l2 = float(np.sqrt(np.sum(np.abs(mean) ** 2)))
    if mean_l2 == 0.0:
        raise ValueError("estimate has zero L² norm; cannot form a relative error")
    if count <= 0:
        raise ValueError("count must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    rel = float(se_l2) / mean_l2
    converged = rel <= tol
    rate = slope if (estimator == "rqmc" and slope is not None) else -0.5
    if rate >= 0.0:
        raise ValueError(f"convergence rate must be negative, got {rate}")
    conservative = estimator == "rqmc" and slope is None
    if converged:
        n_star = int(count)
    else:
        # tol = rel · (N*/count)^rate  ⇒  N* = count · (tol/rel)^(1/rate).
        # N* is an estimate; shave a relative epsilon so float roundoff on an
        # exact integer (e.g. 100^2 = 10000.0000002) doesn't bump the ceil.
        raw = count * (tol / rel) ** (1.0 / rate)
        n_star = int(np.ceil(raw * (1.0 - 1e-9)))
    return {
        "estimator": estimator,
        "rel_err": rel,
        "tol": float(tol),
        "converged": converged,
        "n_current": int(count),
        "n_star": n_star,
        "remaining": max(n_star - int(count), 0),
        "rate": float(rate),
        "conservative": conservative,
        "n_replica": n_replica,
    }


def _shard_coverage_1based(shards: list[str]) -> tuple[int, int]:
    """(min, max) realization index over all shards' /mc r_done, as 1-based
    inclusive display bounds. Reads only the tiny r_done attr (no bulk data)."""
    r_done: list[int] = []
    for shard in shards:
        with h5py.File(shard, "r") as f:
            rd = np.asarray(f[MC_GROUP].attrs["r_done"]).reshape(-1)
        r_done.extend(int(x) for x in rd)
    return mc_naming.coverage_1based(r_done)


def ensure_shard_group_merged(
    shard_path: str,
    results_dataset: str,
    *,
    log_prefix: str = "[mc-merge]",
) -> str:
    """Given one shard, merge ALL sibling shards sharing the same
    ``<base>_<key>`` (across timestamps and intervals) into
    ``<dir>/<base>_<key>_r<lo>-<hi>.h5`` and return that merged path. Already-
    merged coverage is a no-op. The merged file is ordinary single-shot format,
    so callers process it as-is."""
    directory = os.path.dirname(os.path.abspath(shard_path))
    base_key = mc_naming.shard_group_key(os.path.basename(shard_path))
    shards = sorted(
        p for p in glob.glob(
            os.path.join(directory, mc_naming.shard_group_glob(base_key)))
        if mc_naming.SHARD_RE.match(os.path.basename(p))
    )
    if not shards:
        return os.path.join(directory, f"{base_key}.h5")
    lo, hi = _shard_coverage_1based(shards)
    base_path = os.path.join(directory, mc_naming.merged_name(base_key, lo, hi))
    if os.path.isfile(base_path):
        return base_path
    merged = merge_mc_shards(shards, results_dataset)
    write_merged_spectrum(shards[0], base_path, merged, results_dataset)
    print(
        f"{log_prefix} merged {len(shards)} MC shard(s) -> "
        f"{os.path.basename(base_path)} (count={merged['count']}, "
        f"SE L2={merged['se_l2']:.3e})",
    )
    return base_path


def fold_shard_groups(
    cwd: str,
    results_dataset: str,
    *,
    log_prefix: str = "[mc-merge]",
) -> None:
    """Ensure every ``<base>_<key>`` shard group in `cwd` has its merged
    ``<base>_<key>_r<lo>-<hi>.h5`` for the current coverage. Called before flat
    discovery so the shards themselves stay out of the processing list."""
    if not os.path.isdir(cwd):
        return
    seen: set[str] = set()
    for fn in sorted(os.listdir(cwd)):
        if not mc_naming.is_rank_shard(fn):
            continue
        try:
            base_key = mc_naming.shard_group_key(fn)
        except ValueError as e:
            warnings.warn(
                f"skipping unparseable MC shard {fn!r}: {e}",
                UserWarning,
                stacklevel=2,
            )
            continue
        if base_key in seen:
            continue
        seen.add(base_key)
        try:
            ensure_shard_group_merged(
                os.path.join(cwd, fn),
                results_dataset,
                log_prefix=log_prefix,
            )
        except (OSError, ValueError) as e:
            warnings.warn(
                f"could not merge shard group {base_key!r}: {e}",
                UserWarning,
                stacklevel=2,
            )


def write_merged_spectrum(shard_ref, out_path, merged, results_dataset):
    """Copy one shard (single-shot format + /mc) to `out_path` and overwrite
    /results/<ds> with the merged estimator ⟨O⟩ and /mc with the merged totals.
    The result is an ordinary single-shot-format file (plus a `disorder_averaged`
    marker) that las-post reads with zero special-casing. For quadrature this
    also normalises the /results placeholder into the true disorder average."""
    ds_path = f"/results/{results_dataset}"
    shutil.copyfile(shard_ref, out_path)
    with h5py.File(out_path, "r+") as f:
        f[ds_path][...] = merged["mean"].astype(np.complex128)
        f[ds_path].attrs["disorder_averaged"] = np.int64(1)
        mc = f[MC_GROUP]
        mc["count"][...] = np.int64(merged["count"])
        mc.attrs["r_done"] = merged["r_done"]
        mc.attrs["n_shards"] = np.int64(merged["n_shards"])
        if merged["estimator"] == "rqmc":
            # RQMC: across-replica SE replaces the (forbidden) √N SEM. Drop the
            # mean-branch datasets that would misrepresent the error.
            mc.attrs["n_replica"] = np.int64(merged["n_replica"])
            for stale in ("sum_obs_sq", "sem"):
                if stale in mc:
                    del mc[stale]
            mc.create_dataset("replica_se",
                              data=merged["replica_se"].astype(np.float64))
            mc.attrs["se_l2"] = float(merged["se_l2"])
            if "estimator" in mc.attrs:
                del mc.attrs["estimator"]
            mc.attrs["estimator"] = "rqmc"
        elif merged["estimator"] == "quadrature":
            # sum_obs IS the estimator; no SEM (absolute weights, no √N SE).
            mc["sum_obs"][...] = merged["sum_obs"].astype(np.complex128)
        else:
            mc["sum_obs_sq"][...] = merged["sum_obs_sq"].astype(np.float64)
            if "sem" in mc:
                del mc["sem"]
            mc.create_dataset("sem", data=merged["sem"].astype(np.float64))
            mc.attrs["se_l2"] = float(merged["se_l2"])
    return out_path
