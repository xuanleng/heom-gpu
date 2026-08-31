#!/usr/bin/env python3
"""MC static-disorder multi-GPU launcher (orchestrator).

Spawns one ``heom_solver`` worker per GPU (``--rank 1..N``). Workers share ONE
flock queue so realizations split with no double-count. The solver's batch mode
fills a single card — do **not** pin multiple workers to one GPU.

Low-level plumbing: ``heom_solver <task> <cfg> --rank N``. This launcher:

1. Locates the solver
2. Mints a shared ``--run-id`` and resets the queue
3. Launches one worker per GPU (or one unpinned worker when ``--gpus`` omitted)
4. Prints the ``las-post`` step

Injected flags: (1) ``--run-id <TS>`` shared by every worker; (2) ``--r-begin/
--r-end`` from ``--range`` for incremental runs.

Usage:
    mc-disorder-run absorption fmo.yaml
    mc-disorder-run absorption fmo.yaml --range 1 200
    mc-disorder-run absorption fmo.yaml --gpus 0,1
    mc-disorder-run 2des fmo.yaml --gpus 0,1 --range 1 200 --resume
    mc-disorder-run absorption fmo.yaml --precision fp64 --gpus 0
    mc-disorder-run absorption fmo.yaml --gpus 0,1 \\
        --manifest config/nodes_rqmc.manifest.yaml --range 1 200

``--gpus 0,1`` pins each worker via ``CUDA_VISIBLE_DEVICES`` (one id per process;
the solver sees device 0). The shared flock queue balances work across cards.
Omit ``--gpus`` for a single worker that inherits the current CUDA env.

Precision defaults to mixed (``heom_solver_mp``); pass ``--precision fp64`` for
the bit-exact build. ``--solver`` / ``$HEOM_SOLVER`` override binary lookup.
"""
import argparse
import fcntl
import glob
import os
import shutil
import struct
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import yaml

from heom_tools.common.mc_identity import disorder_identity_from_draws
from heom_tools.common import mc_naming
from heom_tools.mc_disorder.config_io import (
    read_n_states,
    read_run_paths,
    read_save_path,
)
from heom_tools.mc_disorder.manifest import read_rqmc_manifest

# MC applies to spectroscopy only; dynamics disorder averaging is future work.
MC_TASKS = ("absorption", "2des")

# How often to refresh the terminal progress line (seconds). A node is many
# seconds of GPU work, so polling an 8-byte counter this often is free.
PROGRESS_POLL_S = 5.0


def parse_gpus(spec):
    """'0,1,3' -> [0,1,3]; None/'' -> None (no pinning == current single-GPU).
    Rejects duplicate or negative ids so a typo can't silently overload one card."""
    if not spec:
        return None
    ids = [int(x) for x in spec.split(",") if x.strip() != ""]
    if not ids:
        return None
    if any(g < 0 for g in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"--gpus must be distinct non-negative ids (got '{spec}')")
    return ids


def plan_gpu_assignment(gpus):
    """CUDA_VISIBLE_DEVICES value for each worker (index 0 == rank 1).

    One worker per GPU. ``gpus=None`` -> a single unpinned worker (inherit env).
    """
    if gpus:
        return [str(g) for g in gpus]
    return [None]


def missing_subintervals(done0, r_begin, r_end):
    """1-based inclusive [r_begin,r_end] minus the done set (0-based r_done from
    shards) -> list of (lo,hi) contiguous 1-based inclusive gaps to re-run. Empty
    list == every realization in the range is already on disk."""
    gaps, lo = [], None
    for r in range(r_begin, r_end + 1):
        if (r - 1) in done0:
            if lo is not None:
                gaps.append((lo, r - 1))
                lo = None
        elif lo is None:
            lo = r
    if lo is not None:
        gaps.append((lo, r_end))
    return gaps


def _read_shard_disorder_identity(h5_path):
    """Read /mc disorder_identity from a shard; None if absent or unreadable."""
    import h5py
    try:
        with h5py.File(h5_path, "r") as f:
            mc = f.get("mc")
            if mc is None:
                return None
            sid = mc.attrs.get("disorder_identity")
            return None if sid is None else str(sid)
    except OSError:
        return None


def read_done_r(save_path, disorder_identity=None):
    """Union of /mc r_done across rank shards in save_path (0-based).

    Only ``*_rank*.h5`` files are scanned — merged products (``*_rlo-hi.h5``)
    are ignored so resume does not treat a partial merge as complete coverage.
    Missing/unreadable shards are skipped; a corrupt shard just re-runs its
    realizations (safe: r is idempotent).

    When ``disorder_identity`` is given (RQMC manifest mode), only shards whose
    ``/mc/disorder_identity`` matches are counted so resume does not see
    siblings' r_done."""
    import h5py
    done = set()
    for h5 in glob.glob(os.path.join(save_path, "*.h5")):
        if not mc_naming.is_rank_shard(os.path.basename(h5)):
            continue
        try:
            with h5py.File(h5, "r") as f:
                mc = f.get("mc")
                if mc is None or "r_done" not in mc.attrs:
                    continue
                if disorder_identity is not None:
                    sid = mc.attrs.get("disorder_identity")
                    if sid is None or str(sid) != disorder_identity:
                        continue
                done.update(int(x) for x in mc.attrs["r_done"])
        except OSError:
            pass
    return done


def solver_basename(precision):
    """PATH name for a precision: fp64 -> heom_solver, mixed -> heom_solver_mp."""
    return "heom_solver_mp" if precision == "mixed" else "heom_solver"


def precision_display(precision: str) -> str:
    """Human-readable precision mode for startup banners."""
    if precision == "mixed":
        return "mixed precision"
    return "fp64 (double precision)"


def format_precision_line(precision: str, solver: str) -> str:
    """One-line precision provenance for mc-disorder-run banners."""
    return (f"  precision: {precision_display(precision)} "
            f"[{os.path.basename(solver)}]")


def find_solver(explicit=None, precision="fp64"):
    """Resolve the solver binary.

    An explicit full path (--solver, then $HEOM_SOLVER) always wins. Otherwise
    look up the binary on PATH by name, chosen by precision: fp64 ->
    'heom_solver', mixed -> 'heom_solver_mp'. This is the common case once both
    builds are symlinked onto PATH — flip precision, not a whole path."""
    for cand in (explicit, os.environ.get("HEOM_SOLVER")):
        if cand:
            if not os.path.isfile(cand) or not os.access(cand, os.X_OK):
                raise FileNotFoundError(
                    f"solver not executable at '{cand}'")
            return cand
    name = solver_basename(precision)
    found = shutil.which(name)
    if not found:
        hint = ""
        if precision == "mixed":
            hint = (" (default mixed; install/link heom_solver_mp on PATH or "
                    "pass --precision fp64)")
        raise FileNotFoundError(
            f"'{name}' not found on PATH{hint}: pass --solver PATH, set "
            f"$HEOM_SOLVER, or put '{name}' on PATH")
    return found


def validate_manifest_against_config(manifest, config):
    """Fail fast if manifest node geometry disagrees with the HEOM config."""
    n_states = read_n_states(config)
    n_site = manifest.get("n_site")
    if n_site is not None and int(n_site) != n_states:
        raise ValueError(
            f"manifest n_site={n_site} != config system.n_states={n_states}")
    if manifest.get("draws_files"):
        import h5py
        with h5py.File(manifest["draws_files"][0], "r") as h:
            cols = int(h["delta_eps"].shape[1])
        if cols != n_states:
            raise ValueError(
                f"draws file n_site={cols} != config system.n_states={n_states}")
    fwhm = manifest.get("fwhm_cm")
    if fwhm is not None and len(np.asarray(fwhm).reshape(-1)) not in (0, n_states):
        raise ValueError(
            f"manifest fwhm_cm length {len(np.asarray(fwhm))} != n_states={n_states}")


@contextmanager
def save_path_lock(save_path):
    """Exclusive flock on save_path/.mc_run.lock (non-blocking).

    Prevents two mc-disorder-run processes from sharing one queue/shard dir."""
    os.makedirs(save_path, exist_ok=True)
    lock_path = os.path.join(save_path, ".mc_run.lock")
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise RuntimeError(
            f"another mc-disorder-run holds the lock on '{save_path}' "
            "(shared save_path / mc.queue); wait or stop the other run")
    try:
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
        try:
            os.remove(lock_path)
        except OSError:
            pass


def write_replica_cfg(base_config, save_path, replica, draws_abs):
    """Patch static_disorder.draws_file into a traceable per-replica cfg."""
    with open(base_config) as fh:
        doc = yaml.safe_load(fh) or {}
    doc.setdefault("static_disorder", {})["draws_file"] = draws_abs
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f".mc_run_r{replica}.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
    return path


def read_n_node(draws):
    """Total node count = rows of delta_eps in the disorder file. Returns None if the
    file can't be read yet (progress then shows a bare claimed count)."""
    try:
        import h5py
        with h5py.File(draws, "r") as h:
            return int(h["delta_eps"].shape[0])
    except Exception:
        return None


def read_queue_claimed(queue_path):
    """The flock counter holds the next-to-claim index == nodes handed out so
    far (int64 LE). This is the live progress signal; absent/empty => 0."""
    try:
        with open(queue_path, "rb") as f:
            b = f.read(8)
        return struct.unpack("<q", b)[0] if len(b) == 8 else 0
    except OSError:
        return 0


def _fmt_dur(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _emit_progress(claimed, n_total, elapsed, n_alive, tty):
    """One progress line. 'claimed' (nodes handed to workers) leads 'done' by at
    most #workers in flight; honest label, monotonic, zero log-scraping."""
    total = f"/{n_total}" if n_total else ""
    if claimed > 0:
        per = elapsed / claimed
        eta = _fmt_dur((n_total - claimed) * per) if n_total else "?"
        rate = f"~{per:.1f}s/node, eta {eta}"
    else:
        rate = "warming up"
    tail = "" if n_alive else "  (all workers exited)"
    msg = (f"[mc-disorder-run] claimed {claimed}{total} | "
           f"elapsed {_fmt_dur(elapsed)} | {rate}{tail}")
    if tty:
        print("\r" + msg.ljust(90), end="", flush=True)
    else:
        print(msg, flush=True)


def reset_queue(queue_path):
    """Remove the shared counter so a fresh run starts realizations at 0. A
    stale counter (== N from a finished run) would make every worker a no-op."""
    try:
        os.remove(queue_path)
    except FileNotFoundError:
        pass


def clean_orphan_interval_shard_logs(save_path, r_begin, r_end) -> int:
    """Remove ``*_r{a}-{b}_rank*.log`` files with no matching ``.h5`` shard."""
    removed = 0
    pat = os.path.join(save_path, f"*_r{r_begin}-{r_end}_rank*.log")
    for log_path in glob.glob(pat):
        h5_path = log_path[:-4] + ".h5"
        if os.path.isfile(h5_path):
            continue
        try:
            os.remove(log_path)
            removed += 1
        except OSError:
            pass
    return removed


def clean_stale_interval_shards(save_path, r_begin, r_end, disorder_identity=None):
    """Delete shards (and their renamed logs) from a PREVIOUS run of THIS exact
    interval before relaunching. Re-running ``--range a b`` in a new minute would
    otherwise leave the old ``..._<TS>_r{a}-{b}_rank*.h5`` behind next to the new
    ones; both cover the same realizations, so the later merge would trip the
    overlap guard. A fresh run of an interval replaces that interval's shards;
    OTHER intervals (incremental batches) are matched by their own r{a}-{b} and
    left untouched. Merged products (``*_r*-*.h5`` without ``_rank``) are also
    untouched.

    When ``disorder_identity`` is set (RQMC manifest / multi-key save_path),
    only shards whose ``/mc/disorder_identity`` matches are removed — sibling
    replicas' shards for the same interval are preserved. Returns files removed."""
    removed = 0
    for suffix in ("h5", "log"):
        pat = os.path.join(save_path, f"*_r{r_begin}-{r_end}_rank*.{suffix}")
        for old in glob.glob(pat):
            if disorder_identity is not None:
                h5 = old if old.endswith(".h5") else old[:-4] + ".h5"
                ident = _read_shard_disorder_identity(h5) if os.path.isfile(h5) else None
                if ident != disorder_identity:
                    continue
            try:
                os.remove(old)
                removed += 1
            except OSError:
                pass
    removed += clean_orphan_interval_shard_logs(save_path, r_begin, r_end)
    return removed


def _rename_log_to_shard(save_path, run_id, r_begin, r_end, rank, log):
    """Rename a successful worker's temp log to match its data shard: <stem>.log.
    The SOLVER names the shard <base>_<key>_<run_id>_r<begin>-<end>_rank{N}.h5;
    we learn the stem from the produced file (single source of truth) instead of
    recomputing the key in Python. No/ambiguous match => leave the temp log."""
    pat = os.path.join(save_path, f"*_{run_id}_r{r_begin}-{r_end}_rank{rank}.h5")
    hits = glob.glob(pat)
    if len(hits) != 1:
        return
    stem = os.path.splitext(os.path.basename(hits[0]))[0]
    try:
        os.replace(log, os.path.join(save_path, stem + ".log"))
    except OSError:
        pass


def launch(solver, task, config, save_path, queue_path,
           run_id, r_begin, r_end, gpu_of_rank):
    """Spawn one worker per entry in ``gpu_of_rank`` over [r_begin, r_end]
    (1-based inclusive) and poll the shared queue for progress.

    Rank is 1..len(gpu_of_rank). Each worker's stdout+stderr goes to a temp
    <save_path>/worker_rank{N}.log during the run (live-tail with `tail -f`);
    on success it is renamed to <shard>.log. All workers get the SAME --run-id
    and the SAME --r-begin/--r-end (the solver does the 1-based->0-based
    conversion).

    ``gpu_of_rank[i]`` is the CUDA_VISIBLE_DEVICES string for rank i+1, or None
    to inherit the parent env. Returns [(rank, exit, log)] for failures."""
    os.makedirs(save_path, exist_ok=True)
    n_total = r_end - r_begin + 1
    begin0 = r_begin - 1                     # queue counter base (0-based)
    n_workers = len(gpu_of_rank)
    workers = []
    for rank in range(1, n_workers + 1):
        log = os.path.join(save_path, f"worker_rank{rank}.log")
        fh = open(log, "w")
        try:
            env = os.environ.copy()
            gpu = gpu_of_rank[rank - 1]
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            proc = subprocess.Popen(
                [solver, task, config, "--rank", str(rank), "--run-id", run_id,
                 "--r-begin", str(r_begin), "--r-end", str(r_end)],
                stdout=fh, stderr=subprocess.STDOUT, env=env)
        except Exception:
            fh.close()
            raise
        workers.append([rank, proc, fh, log])

    tty = sys.stdout.isatty()
    start = time.time()
    last = None
    while True:
        n_alive = sum(1 for w in workers if w[1].poll() is None)
        # Counter holds the next 0-based index; nodes done THIS batch = counter
        # minus the interval base, clamped to [0, n_total].
        claimed = min(max(0, read_queue_claimed(queue_path) - begin0), n_total)
        if claimed != last or n_alive == 0:
            _emit_progress(claimed, n_total, time.time() - start, n_alive, tty)
            last = claimed
        if n_alive == 0:
            break
        time.sleep(PROGRESS_POLL_S)
    if tty:
        print()  # terminate the in-place \r line

    failed = []
    for rank, proc, fh, log in workers:
        fh.close()
        if proc.returncode != 0:
            failed.append((rank, proc.returncode, log))
            continue
        _rename_log_to_shard(save_path, run_id, r_begin, r_end, rank, log)
    return failed


def _log_tail(path, n=15):
    """Last n lines of a worker log, for surfacing a failure without hunting."""
    try:
        with open(path) as fh:
            return "".join(fh.readlines()[-n:]).rstrip()
    except OSError:
        return "(log unavailable)"


def _resolve_range(p, draws, n_file, range_args):
    """Return validated (r_begin, r_end, n_total) for one node file."""
    if range_args is not None:
        r_begin, r_end = range_args
    else:
        if n_file is None:
            p.error(f"could not read node count from '{draws}' to default "
                    "--range; pass --range START END explicitly")
        r_begin, r_end = 1, n_file
    if r_begin < 1 or r_end < r_begin:
        p.error(f"--range must satisfy 1 <= START <= END (got {r_begin} {r_end})")
    if n_file is not None and r_end > n_file:
        p.error(f"--range END={r_end} exceeds node file rows={n_file} in "
                f"'{draws}'; extend the file (mc-disorder-gen --extend) or "
                "lower END")
    return r_begin, r_end, r_end - r_begin + 1


def _run_batch(*, solver, task, config, save_path, draws, gpu_of_rank,
               gpu_line, run_id, r_begin, r_end, resume, precision,
               disorder_identity=None, replica_banner=None):
    """Run one MC interval (fresh or resume) for a single draws_file. Returns 0/1."""
    queue_path = os.path.join(save_path, "mc.queue")
    n_file = read_n_node(draws)
    n_total = r_end - r_begin + 1
    n_workers = len(gpu_of_rank)
    banner = replica_banner or f"{task} {config}"

    if resume:
        done0 = read_done_r(save_path, disorder_identity)
        gaps = missing_subintervals(done0, r_begin, r_end)
        if not gaps:
            print(f"[mc-disorder-run] resume: [{r_begin},{r_end}] already "
                  f"complete ({len(done0)} r on disk); nothing to do.")
            return 0
        print(f"[mc-disorder-run] {banner}  (resume)")
        print(format_precision_line(precision, solver))
        print(f"  solver   : {solver}")
        print(gpu_line)
        print(f"  draws    : {draws}")
        print(f"  on disk  : {len(done0)} r; re-running gaps {gaps}")
        rc = 0
        for lo, hi in gaps:
            reset_queue(queue_path)
            clean_stale_interval_shards(save_path, lo, hi, disorder_identity)
            failed = launch(solver, task, config,
                            save_path, queue_path, run_id, lo, hi, gpu_of_rank)
            if failed:
                rc = 1
                for rank, code, log in failed:
                    print(f"\n[mc-disorder-run] gap [{lo},{hi}] worker {rank} "
                          f"FAILED (exit {code}); tail of {log}:", file=sys.stderr)
                    print(_log_tail(log), file=sys.stderr)
        if rc == 0:
            print(f"[mc-disorder-run] resume done. Next:\n  las-post "
                  f"{save_path}*_rank1.h5   # merges shards for this replica by key")
        return rc

    reset_queue(queue_path)
    n_stale = clean_stale_interval_shards(save_path, r_begin, r_end,
                                          disorder_identity)

    print(f"[mc-disorder-run] {banner}")
    print(format_precision_line(precision, solver))
    print(f"  solver   : {solver}")
    print(gpu_line)
    print(f"  draws    : {draws}")
    print(f"  range    : [{r_begin},{r_end}] (1-based inclusive, {n_total} nodes "
          f"of {n_file if n_file is not None else '?'} in {draws})")
    print(f"  queue    : {queue_path} (reset)")
    if n_stale:
        print(f"  cleaned  : {n_stale} stale r{r_begin}-{r_end} shard/log file(s) "
              "from a previous run of this interval")
    print(f"  run-id   : {run_id} (shared by all workers; info bit, not group key)")
    print(f"  shards   : {save_path}*_{run_id}_r{r_begin}-{r_end}_rank1..{n_workers}.h5")
    print(f"  logs     : {save_path}worker_rank1..{n_workers}.log "
          f"(temp; renamed to <shard>.log on success)")

    failed = launch(solver, task, config,
                    save_path, queue_path, run_id, r_begin, r_end, gpu_of_rank)

    if failed:
        for rank, code, log in failed:
            print(f"\n[mc-disorder-run] worker rank {rank} FAILED (exit {code}); "
                  f"tail of {log}:", file=sys.stderr)
            print(_log_tail(log), file=sys.stderr)
        return 1

    print(f"[mc-disorder-run] all {n_workers} worker(s) done. Next:")
    print(f"  las-post {save_path}*_{run_id}_r{r_begin}-{r_end}_rank1.h5   "
          "# auto-merges by param key (across batches), then makes the spectrum")
    return 0


def _main_manifest(p, args, solver, save_path, gpu_of_rank, gpu_line, precision):
    """Serial RQMC replica loop driven by mc-disorder-gen manifest."""
    try:
        manifest = read_rqmc_manifest(args.manifest)
        validate_manifest_against_config(manifest, args.config)
    except (ValueError, RuntimeError) as e:
        p.error(str(e))

    with open(args.config) as fh:
        base_doc = yaml.safe_load(fh) or {}
    cfg_draws = (base_doc.get("static_disorder") or {}).get("draws_file")
    if cfg_draws:
        print("[mc-disorder-run] note: static_disorder.draws_file in config is "
              "ignored; --manifest drives draws_file per replica",
              file=sys.stderr)

    r_begin, r_end, _ = _resolve_range(
        p, manifest["draws_files"][0], manifest["n_node"], args.range)

    n_rep = manifest["n_replicas"]
    print(f"[mc-disorder-run] manifest: {args.manifest} ({n_rep} rqmc replicas)")
    print(f"  task     : {args.task} {args.config}")
    print(format_precision_line(precision, solver))
    print(f"  solver   : {solver}")
    print(f"  save_path: {save_path}")
    print(f"  range    : [{r_begin},{r_end}] (applied to every replica)")

    for repl, draws in enumerate(manifest["draws_files"]):
        identity = disorder_identity_from_draws(draws)
        cfg = write_replica_cfg(args.config, save_path, repl, draws)
        banner = (f"{args.task} replica {repl}/{n_rep - 1} "
                  f"(cfg {os.path.basename(cfg)})")
        # Per-replica run_id: info bit only (merge keys on disorder_identity),
        # but distinct timestamps make shard dirs easier to debug.
        repl_run_id = f"{time.strftime('%Y%m%d_%H%M')}_repl{repl}"
        rc = _run_batch(
            solver=solver, task=args.task, config=cfg, save_path=save_path,
            draws=draws, gpu_of_rank=gpu_of_rank,
            gpu_line=gpu_line, run_id=repl_run_id, precision=precision,
            r_begin=r_begin, r_end=r_end, resume=args.resume,
            disorder_identity=identity, replica_banner=banner,
        )
        if rc != 0:
            print(f"[mc-disorder-run] stopped at replica {repl}/{n_rep - 1}",
                  file=sys.stderr)
            return rc

    print(f"[mc-disorder-run] all {n_rep} replica(s) done. Next:")
    print("  las-post <save_path>/*_rank1.h5   # per replica (distinct merge keys)")
    print("  mc-disorder-merge --combine-replicas --out rqmc.h5 rep0.h5 ...")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Launch one heom_solver MC worker per GPU over a shared queue")
    p.add_argument("task", choices=MC_TASKS,
                   help="spectroscopy task (MC does not apply to dynamics yet)")
    p.add_argument("config", help="config .yaml (must have a static_disorder block)")
    p.add_argument("--gpus", metavar="IDS",
                   help="comma list of GPU ids, one worker each, e.g. '0,1'. "
                        "Pinned via CUDA_VISIBLE_DEVICES; shared queue balances "
                        "realizations. Omit for a single unpinned worker.")
    p.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                   help="1-based INCLUSIVE realization interval to run (default: "
                        "the whole node file). Incremental: --range 1 200 then "
                        "--range 201 300 adds samples without recomputing 1..200.")
    p.add_argument("--resume", action="store_true",
                   help="run only realizations in --range that are NOT already on "
                        "disk (union of shard r_done). Re-runs contiguous gaps "
                        "left by crashed workers instead of recomputing the batch.")
    p.add_argument("--precision", choices=("fp64", "mixed"), default=None,
                   help="fp64 -> heom_solver (reference/regression); default mixed "
                        "-> heom_solver_mp (production). fp64 and mixed shards "
                        "must not share one save_path. Ignored when --solver/"
                        "$HEOM_SOLVER gives a full path.")
    p.add_argument("--solver", metavar="PATH",
                   help="explicit solver binary path (overrides --precision; "
                        "default: $HEOM_SOLVER, else looked up on PATH)")
    p.add_argument("--manifest", metavar="PATH",
                   help="rqmc only: mc-disorder-gen manifest; run every "
                        "draws_file serially (overrides config draws_file)")
    args = p.parse_args(argv)

    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as e:
        p.error(str(e))
    gpu_of_rank = plan_gpu_assignment(gpus)
    n_workers = len(gpu_of_rank)

    # Default mixed (production); always print precision so provenance isn't
    # silent. fp64 vs mixed shards can't be merged — never mix in one save_path.
    precision = args.precision or "mixed"
    try:
        solver = find_solver(args.solver, precision)
        if args.manifest:
            save_path = read_save_path(args.config, args.task)
            draws = None
        else:
            save_path, draws = read_run_paths(args.config, args.task)
    except (FileNotFoundError, ValueError) as e:
        p.error(str(e))

    print(f"[mc-disorder-run] precision: "
          f"{precision_display(precision)} [{os.path.basename(solver)}]",
          file=sys.stderr)

    # Only nag on a genuine contradiction the user typed: --precision AND an
    # explicit full path that wins over it. Silent when they align or default.
    if args.precision is not None and (args.solver or os.environ.get("HEOM_SOLVER")):
        print(f"[mc-disorder-run] note: --precision {args.precision} ignored; "
              f"explicit solver path in effect ({solver})", file=sys.stderr)

    # One timestamp for the whole launch, handed to every worker so their shards
    # share one prefix (a per-worker now() would split across a minute). Format
    # matches the solver's resolve_output_path (%Y%m%d_%H%M). It is an info bit in
    # the filename only — grouping/merging keys on the parameter key, not the TS.
    run_id = time.strftime("%Y%m%d_%H%M")

    if gpus is not None:
        gpu_line = (f"  gpus     : {gpus}  ({n_workers} worker(s), one per card; "
                    "CUDA_VISIBLE_DEVICES pinned; queue balances across cards)")
    else:
        gpu_line = "  workers  : 1 (unpinned; inherits CUDA_VISIBLE_DEVICES)"

    try:
        with save_path_lock(save_path):
            if args.manifest:
                return _main_manifest(p, args, solver, save_path,
                                      gpu_of_rank, gpu_line, precision)

            n_file = read_n_node(draws)
            r_begin, r_end, _ = _resolve_range(p, draws, n_file, args.range)
            identity = disorder_identity_from_draws(draws)

            return _run_batch(
                solver=solver, task=args.task, config=args.config,
                save_path=save_path, draws=draws, gpu_of_rank=gpu_of_rank,
                gpu_line=gpu_line, run_id=run_id,
                r_begin=r_begin, r_end=r_end, resume=args.resume,
                precision=precision,
                disorder_identity=identity,
            )
    except RuntimeError as e:
        p.error(str(e))


if __name__ == "__main__":
    sys.exit(main())
