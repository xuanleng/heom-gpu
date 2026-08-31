#!/usr/bin/env python3
"""mc-disorder-calibrate (obsolete).

Solver batch mode fills one GPU; one worker per card is fixed.
``main()`` prints the replacement ``mc-disorder-run --gpus …`` command and exits.
Helper functions remain for unit tests / archaeology.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Sequence

from heom_tools.mc_disorder import config_io as cfg_io
from . import run as run_mod

_RUN_MODULE = "heom_tools.mc_disorder.run"


@dataclass
class SweepRow:
    k: int
    wall_s: float
    n_realizations: int
    realizations_per_s: float
    mean_sm_pct: float | None
    peak_vram_mb: int | None
    peak_rss_mb: int
    ok: bool
    rc: int

    @property
    def vram_per_worker_mb(self) -> float | None:
        if self.peak_vram_mb is None or self.k < 1:
            return None
        return self.peak_vram_mb / self.k

    @property
    def rss_per_worker_mb(self) -> float:
        return self.peak_rss_mb / max(self.k, 1)


def write_calibration_cfg(base_config: str, task: str, scratch_save_path: str,
                        *, cfg_path: str) -> str:
    """Patch save_path to scratch; resolve draws_file to absolute (base cfg dir)."""
    return cfg_io.write_calibration_cfg(
        base_config, task, scratch_save_path, cfg_path=cfg_path)


def validate_draws_for_samples(config: str, task: str, samples: int) -> str:
    """Fail fast if draws_file has fewer than ``samples`` rows. Returns draws path."""
    _, draws = run_mod.read_run_paths(config, task)
    n_node = run_mod.read_n_node(draws)
    if n_node is None:
        raise ValueError(f"cannot read node count from draws_file '{draws}'")
    if samples > n_node:
        raise ValueError(
            f"--samples {samples} exceeds draws_file rows={n_node} in '{draws}'; "
            "regenerate with mc-disorder-gen or lower --samples")
    return draws


def _read_child_pids(pid: int) -> list[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children", encoding="utf-8") as fh:
            return [int(x) for x in fh.read().split() if x.isdigit()]
    except OSError:
        return []


def process_tree_pids(roots: Sequence[int]) -> set[int]:
    """All PIDs in the subprocess trees rooted at ``roots`` (Linux /proc)."""
    seen: set[int] = set()
    stack = [int(r) for r in roots if r > 0]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(_read_child_pids(pid))
    return seen


def process_tree_rss_kb(roots: Sequence[int]) -> int:
    """Sum VmRSS over the process trees rooted at ``roots``."""
    total = 0
    for pid in process_tree_pids(roots):
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
                for row in fh:
                    if row.startswith("VmRSS:"):
                        total += int(row.split()[1])
                        break
        except OSError:
            continue
    return total


def parse_dmon_stats(log_path: str, gpu_id: int) -> tuple[float | None, int | None]:
    """Mean SM% and peak framebuffer MB from nvidia-smi dmon -s um log."""
    mean_sm = 0.0
    peak_fb = 0
    n = 0
    gid = str(gpu_id)
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 8 or parts[0] != gid:
                    continue
                try:
                    sm = float(parts[1])
                    fb = int(float(parts[7]))
                except ValueError:
                    continue
                n += 1
                mean_sm += sm
                peak_fb = max(peak_fb, fb)
    except OSError:
        return None, None
    if n == 0:
        return None, None
    return mean_sm / n, peak_fb


def mem_available_mb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def query_gpu_total_mb(gpu_id: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return int(out.strip().split()[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        return None


def query_visible_gpu_ids() -> list[int]:
    """All GPU indices visible to nvidia-smi (empty if unavailable)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        ids: list[int] = []
        for line in out.splitlines():
            token = line.strip().split(",")[0].strip()
            if token.isdigit():
                ids.append(int(token))
        return ids
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []


def pick_k_compute(rows: Sequence[SweepRow], *, min_gain: float = 0.05) -> int | None:
    """Largest K with near-peak realizations/s; tie → smaller K (conservative)."""
    ok = [r for r in rows if r.ok and r.realizations_per_s > 0]
    if not ok:
        return None
    best_rate = max(r.realizations_per_s for r in ok)
    threshold = best_rate * (1.0 - min_gain)
    candidates = [r for r in ok if r.realizations_per_s >= threshold]
    return min(r.k for r in candidates)


def pick_k_gpumem(
    rows: Sequence[SweepRow],
    gpu_total_mb: int | None,
    *,
    headroom: float = 0.85,
) -> int | None:
    if not gpu_total_mb:
        return None
    budget = gpu_total_mb * headroom
    per_worker = [
        r.vram_per_worker_mb for r in rows
        if r.ok and r.vram_per_worker_mb and r.vram_per_worker_mb > 0
    ]
    if not per_worker:
        return None
    worst = max(per_worker)
    return max(1, int(budget // worst))


def pick_k_hostmem(
    rows: Sequence[SweepRow],
    mem_avail_mb: int | None,
    *,
    headroom: float = 0.85,
) -> int | None:
    if not mem_avail_mb:
        return None
    budget = mem_avail_mb * headroom
    per_worker = [r.rss_per_worker_mb for r in rows if r.ok and r.rss_per_worker_mb > 0]
    if not per_worker:
        return None
    worst = max(per_worker)
    return max(1, int(budget // worst))


def recommend_k(
    rows: Sequence[SweepRow],
    *,
    gpu_total_mb: int | None,
    mem_avail_mb: int | None,
) -> dict:
    k_compute = pick_k_compute(rows)
    k_gpumem = pick_k_gpumem(rows, gpu_total_mb)
    k_hostmem = pick_k_hostmem(rows, mem_avail_mb)
    limits = {
        "K_compute": k_compute,
        "K_gpumem": k_gpumem,
        "K_hostmem": k_hostmem,
    }
    vals = [v for v in limits.values() if v is not None]
    k_star = min(vals) if vals else None
    return {**limits, "K_star": k_star}


class _DmonMonitor:
    def __init__(self, gpu_ids: str, log_path: str):
        self._gpu_ids = gpu_ids
        self._log_path = log_path
        self._proc = None

    def __enter__(self):
        with open(self._log_path, "w", encoding="utf-8"):
            pass
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                self._proc = subprocess.Popen(
                    ["nvidia-smi", "dmon", "-i", self._gpu_ids, "-s", "um", "-d", "1"],
                    stdout=fh,
                    stderr=subprocess.DEVNULL,
                )
        except FileNotFoundError:
            self._proc = None
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


class _RssMonitor:
    def __init__(self, peak_path: str, roots: list[int]):
        self._peak_path = peak_path
        self._roots = roots
        self._stop = False

    def __enter__(self):
        with open(self._peak_path, "w", encoding="utf-8") as fh:
            fh.write("0")
        self._stop = False
        import threading

        def loop():
            peak = 0
            while not self._stop:
                peak = max(peak, process_tree_rss_kb(self._roots))
                try:
                    with open(self._peak_path, "w", encoding="utf-8") as fh:
                        fh.write(str(peak))
                except OSError:
                    pass
                time.sleep(1.0)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._thread.join(timeout=3)


def _read_peak_rss_mb(path: str) -> int:
    try:
        kb = int(open(path, encoding="utf-8").read().strip() or "0")
    except (OSError, ValueError):
        kb = 0
    return kb // 1024


def run_mc_disorder(
    task: str,
    config: str,
    *,
    gpus: str,
    n_realizations: int,
    precision: str,
    log_path: str,
    roots: list[int] | None = None,
) -> int:
    """Run mc-disorder-run via the current interpreter (not PATH)."""
    cmd = [
        sys.executable, "-m", _RUN_MODULE,
        task, config,
        "--gpus", gpus,
        "--range", "1", str(n_realizations),
        "--precision", precision,
    ]
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        if roots is not None:
            roots.append(proc.pid)
        rc = proc.wait()
    return rc


def sweep_per_gpu(
    task: str,
    config: str,
    *,
    gpu0: int,
    kmax: int,
    n_realizations: int,
    precision: str,
    out_dir: str,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    gpu_s = str(gpu0)
    for k in range(1, kmax + 1):
        dmon_log = os.path.join(out_dir, f"dmon_k{k}.log")
        rss_path = os.path.join(out_dir, f"rss_k{k}.txt")
        run_log = os.path.join(out_dir, f"run_k{k}.log")
        print(f"  [K={k}] mc-disorder-run {n_realizations} realizations "
              f"({run_mod.precision_display(precision)}, "
              f"--gpus {gpu_s})...", flush=True)
        roots: list[int] = []
        t0 = time.time()
        with _DmonMonitor(gpu_s, dmon_log), _RssMonitor(rss_path, roots):
            rc = run_mc_disorder(
                task, config, gpus=gpu_s,
                n_realizations=n_realizations, precision=precision,
                log_path=run_log, roots=roots,
            )
        wall = max(time.time() - t0, 1.0)
        mean_sm, peak_fb = parse_dmon_stats(dmon_log, gpu0)
        rss_mb = _read_peak_rss_mb(rss_path)
        rate = n_realizations / wall
        rows.append(SweepRow(
            k=k, wall_s=wall, n_realizations=n_realizations,
            realizations_per_s=rate,
            mean_sm_pct=mean_sm, peak_vram_mb=peak_fb,
            peak_rss_mb=rss_mb, ok=(rc == 0), rc=rc,
        ))
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        sm_s = f"{mean_sm:.0f}" if mean_sm is not None else "NA"
        fb_s = str(peak_fb) if peak_fb is not None else "NA"
        print(f"      K={k}  wall={wall:.0f}s  realizations/s={rate:.2f}  "
              f"meanSM={sm_s}%  peakVRAM={fb_s}MB  RSS={rss_mb}MB  {status}")
        if rc != 0:
            try:
                with open(run_log, encoding="utf-8", errors="replace") as fh:
                    tail = fh.readlines()[-8:]
                print("      --- tail run log ---", file=sys.stderr)
                for line in tail:
                    print(f"      {line.rstrip()}", file=sys.stderr)
            except OSError:
                pass
    return rows


def dual_card_smoke(
    task: str,
    config: str,
    *,
    gpus: str,
    n_realizations: int,
    precision: str,
    out_dir: str,
) -> tuple[dict[int, tuple[float | None, int | None]], int]:
    dmon_log = os.path.join(out_dir, "dmon_dual.log")
    run_log = os.path.join(out_dir, "run_dual.log")
    print(f"== dual-card smoke: --gpus {gpus} "
          f"({n_realizations} realizations, "
          f"{run_mod.precision_display(precision)}) ==")
    t0 = time.time()
    with _DmonMonitor(gpus, dmon_log):
        rc = run_mc_disorder(
            task, config, gpus=gpus,
            n_realizations=n_realizations, precision=precision,
            log_path=run_log,
        )
    wall = max(time.time() - t0, 1.0)
    print(f"  wall={wall:.0f}s  rc={rc}")
    stats = {}
    for gid in run_mod.parse_gpus(gpus) or []:
        sm, fb = parse_dmon_stats(dmon_log, gid)
        stats[gid] = (sm, fb)
        sm_s = f"{sm:.0f}" if sm is not None else "NA"
        fb_s = str(fb) if fb is not None else "NA"
        print(f"  GPU {gid}: meanSM={sm_s}%  peakVRAM={fb_s}MB")
    if rc != 0:
        print("[mc-disorder-calibrate] dual-card smoke FAILED", file=sys.stderr)
    return stats, rc


def _format_gpus(gpus: list[int]) -> str:
    return ",".join(str(g) for g in gpus)


def production_gpu_ids(sweep_gpus: list[int]) -> tuple[list[int], str | None]:
    """GPU list for the production example.

    Sweep always uses ``sweep_gpus[0]`` only. When the user passed a single id
    but the machine exposes multiple GPUs, recommend all visible cards for
    production (pass ``--gpus 0,1`` explicitly to limit which cards to use).
    """
    if len(sweep_gpus) > 1:
        return sweep_gpus, None
    visible = query_visible_gpu_ids()
    if len(sweep_gpus) == 1 and len(visible) > 1:
        note = (f"sweep used GPU {sweep_gpus[0]} only; "
                f"production uses all {len(visible)} visible GPU(s) "
                f"({_format_gpus(visible)})")
        return visible, note
    return sweep_gpus, None


def print_oversubscription_hint(rows: Sequence[SweepRow]) -> None:
    """Explain when higher K shows higher SM% but lower realizations/s."""
    ok = [r for r in rows if r.ok and r.realizations_per_s > 0]
    if len(ok) < 2:
        return
    best = max(ok, key=lambda r: r.realizations_per_s)
    for r in ok:
        if r.k <= best.k:
            continue
        if r.mean_sm_pct is None or best.mean_sm_pct is None:
            continue
        if (r.mean_sm_pct > best.mean_sm_pct
                and r.realizations_per_s < best.realizations_per_s * 0.95):
            print(
                f"[mc-disorder-calibrate] note: K={r.k} mean SM% "
                f"({r.mean_sm_pct:.0f}) > K={best.k} ({best.mean_sm_pct:.0f}) "
                f"but realizations/s is lower "
                f"({r.realizations_per_s:.2f} vs {best.realizations_per_s:.2f}) — "
                "GPU oversubscription; higher occupancy ≠ faster throughput.",
            )
            break


def _print_table(rows: Sequence[SweepRow], summary_fh):
    hdr = (f"{'K':<4} {'wall_s':<9} {'realiz':<8} {'realiz/s':<10} "
           f"{'meanSM%':<10} {'peakVRAM':<10} {'RSS_MB':<8}")
    print(hdr)
    print(hdr, file=summary_fh)
    for r in rows:
        sm_s = f"{r.mean_sm_pct:.0f}" if r.mean_sm_pct is not None else "NA"
        fb_s = str(r.peak_vram_mb) if r.peak_vram_mb is not None else "NA"
        st = "ok" if r.ok else f"FAIL({r.rc})"
        line = (f"{r.k:<4} {r.wall_s:<9.0f} {r.n_realizations:<8} "
                f"{r.realizations_per_s:<10.2f} {sm_s:<10} {fb_s:<10} "
                f"{r.peak_rss_mb:<8}  {st}")
        print(line)
        print(line, file=summary_fh)


def main(argv=None):
    """Obsolete: solver batch fills one GPU; one worker per card is fixed."""
    del argv  # argparse unused
    print(
        "mc-disorder-calibrate is obsolete.\n"
        "heom_solver batch mode fills one GPU — do not pin multiple workers\n"
        "to one card. Use one worker per GPU:\n"
        "\n"
        "  mc-disorder-run <task> <cfg.yaml> --gpus 0,1\n"
        "  mc-disorder-run <task> <cfg.yaml> --gpus 0,1 --range 1 200 --resume\n",
        file=sys.stderr,
    )
    return 2

if __name__ == "__main__":
    sys.exit(main())
