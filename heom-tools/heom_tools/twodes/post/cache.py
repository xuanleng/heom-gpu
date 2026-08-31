"""Output cache: sidecar fingerprint is the single source of truth for skip/regenerate."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    PostConfig,
    cfg_pol_str,
    effective_fft_method,
    normalize_tw_index,
    parse_window,
    plot_png_path,
    pol_tag,
    post_h5_path,
    resolve_tw_by_config_index,
    tw_tag,
)
from .h5_spectra import h5_spectra_fingerprint
from .io import is_solver_2des_h5, load_meta
from .types import OutputCtx, OutputPlan, PrecheckResult, ProcessNeeds


FINGERPRINT_KEYS = (
    "dw", "w0", "window", "fft_method", "pol", "tw_index",
    "time_domain", "levels", "lines", "plot", "write_h5", "h5_spectra",
    "pathways", "source_mtime", "source_size",
)


def source_fingerprint(source_h5: str) -> Dict[str, int]:
    st = os.stat(source_h5)
    return {"source_mtime": int(st.st_mtime_ns), "source_size": int(st.st_size)}


def solver_pathways_label(source_h5: str) -> str:
    if not os.path.exists(source_h5):
        return ""
    try:
        if not is_solver_2des_h5(source_h5):
            return ""
        _tw, _signals, pathways = load_meta(source_h5)
        return ",".join(pathways)
    except (OSError, ValueError):
        return ""


def cfg_fingerprint(cfg: PostConfig, source_h5: str) -> Dict[str, Any]:
    win = parse_window(cfg.window)
    fp = {
        "dw": float(cfg.dw),
        "w0": float(cfg.w0),
        "window": win,
        "fft_method": effective_fft_method(str(cfg.fft_method)),
        "pol": cfg_pol_str(cfg.pol),
        "tw_index": normalize_tw_index(cfg.tw_index),
        "time_domain": bool(cfg.time_domain),
        "plot": bool(cfg.plot),
        "write_h5": bool(cfg.write_h5),
        "h5_spectra": h5_spectra_fingerprint(
            cfg.h5_spectra, write_h5=bool(cfg.write_h5),
        ),
        "levels": int(cfg.levels),
        "lines": list(cfg.lines or []),
        "pathways": solver_pathways_label(source_h5),
    }
    fp.update(source_fingerprint(source_h5))
    return fp


def sidecar_path(outdir: str, stem: str, ptag: str) -> str:
    return os.path.join(outdir, f".{stem}_{ptag}_twodes_post.json")


def write_sidecar(outdir: str, stem: str, ptag: str, cfg: PostConfig, source_h5: str) -> None:
    path = sidecar_path(outdir, stem, ptag)
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=outdir)
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(cfg_fingerprint(cfg, source_h5), f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def meta_mismatch(
    stored: Dict[str, Any],
    current: Dict[str, Any],
    keys: Tuple[str, ...],
    *,
    stored_label: str,
) -> Optional[str]:
    for key in keys:
        if stored.get(key) != current[key]:
            return f"{key}: {stored_label}={stored.get(key)!r} cfg={current[key]!r}"
    return None


def fingerprint_diff(stored: Dict[str, Any], cfg: PostConfig, source_h5: str) -> Optional[str]:
    if "pathways" not in stored and "source_pathways" in stored:
        stored = {**stored, "pathways": stored["source_pathways"]}
    return meta_mismatch(
        stored, cfg_fingerprint(cfg, source_h5), FINGERPRINT_KEYS, stored_label="stored",
    )


def sidecar_meta_diff(sidecar_path: str, cfg: PostConfig, source_h5: str) -> Optional[str]:
    if not os.path.exists(sidecar_path):
        return None
    with open(sidecar_path) as f:
        stored = json.load(f)
    return fingerprint_diff(stored, cfg, source_h5)


def plot_targets(stem, ptag, outdir, tw, tw_indices, td):
    paths = []
    for k in tw_indices:
        tw_fs = float(tw[k])
        paths.append(plot_png_path(outdir, stem, ptag, "2d", tw_fs))
        if td:
            paths.append(plot_png_path(outdir, stem, ptag, "td", tw_fs))
    return paths


def resolve_output_ctx(h5: str, cfg: PostConfig) -> OutputCtx:
    tw, _signals, _pathways = load_meta(h5)
    stem = os.path.splitext(os.path.basename(h5))[0]
    outdir = cfg.out or os.path.dirname(os.path.abspath(h5))
    ptag = pol_tag(cfg.pol)
    tw_indices = resolve_tw_by_config_index(cfg.tw_index, len(tw))
    td = bool(cfg.time_domain)
    post_path = post_h5_path(h5, cfg)
    plot_paths = plot_targets(stem, ptag, outdir, tw, tw_indices, td) if cfg.plot else []
    sc = sidecar_path(outdir, stem, ptag)
    return OutputCtx(
        tw=tw, stem=stem, outdir=outdir, ptag=ptag, post_path=post_path,
        plot_paths=plot_paths, sidecar=sc, tw_indices=tw_indices,
    )


def empty_output_ctx(h5: str, cfg: PostConfig) -> OutputCtx:
    stem = os.path.splitext(os.path.basename(h5))[0]
    outdir = cfg.out or os.path.dirname(os.path.abspath(h5)) or "."
    ptag = pol_tag(cfg.pol)
    return OutputCtx(
        tw=np.array([]), stem=stem, outdir=outdir, ptag=ptag,
        post_path=post_h5_path(h5, cfg), plot_paths=[],
        sidecar=sidecar_path(outdir, stem, ptag), tw_indices=[],
    )


def precheck(h5: str, cfg: PostConfig) -> PrecheckResult:
    """Skip/stale decision and output layout for one input file."""
    should_skip, skip_msg, stale, out_ctx = check_skip_outputs(h5, cfg)
    return PrecheckResult(
        should_skip=should_skip, skip_msg=skip_msg, stale=stale, out_ctx=out_ctx,
    )


def check_skip_outputs(h5: str, cfg: PostConfig) -> Tuple[bool, str, bool, OutputCtx]:
    """Return (should_skip, message, stale, output_ctx).

    Skip/regenerate decisions use the sidecar fingerprint only.
    """
    if bool(cfg.force) or (not cfg.write_h5 and not cfg.plot):
        ctx = resolve_output_ctx(h5, cfg) if os.path.exists(h5) else empty_output_ctx(h5, cfg)
        return False, "", False, ctx

    ctx = resolve_output_ctx(h5, cfg)
    write_h5, do_plot = bool(cfg.write_h5), bool(cfg.plot)

    if (write_h5 and not os.path.exists(ctx.post_path)) or (
        do_plot and any(not os.path.exists(p) for p in ctx.plot_paths)
    ):
        return False, "", False, ctx

    if not os.path.exists(ctx.sidecar):
        return False, "outputs lack sidecar; regenerating", True, ctx

    sidecar_diff = sidecar_meta_diff(ctx.sidecar, cfg, h5)
    if sidecar_diff:
        return False, f"sidecar metadata differs ({sidecar_diff}); regenerating", True, ctx

    return True, "outputs exist (verified via sidecar)", False, ctx


def tw_indices_needing_plots(
    *,
    stem: str,
    outdir: str,
    ptag: str,
    tw,
    tw_indices: List[int],
    plan: OutputPlan,
) -> List[int]:
    """Incremental plot path: only Tw indices with missing PNGs."""
    missing: set[int] = set()
    for k in tw_indices:
        tw_fs = float(tw[k])
        if not os.path.exists(plot_png_path(outdir, stem, ptag, "2d", tw_fs)):
            missing.add(k)
        if plan.need_time:
            if not os.path.exists(plot_png_path(outdir, stem, ptag, "td", tw_fs)):
                missing.add(k)
    return sorted(missing)


def _tw_indices_to_compute(
    *,
    stale: bool,
    need_h5: bool,
    need_plot: bool,
    plan: OutputPlan,
    ctx: OutputCtx,
) -> List[int]:
    """Full Tw stack when H5 or stale; else only Tw indices missing PNGs."""
    if stale or need_h5:
        return list(ctx.tw_indices)
    if not need_plot:
        return []
    return tw_indices_needing_plots(
        stem=ctx.stem,
        outdir=ctx.outdir,
        ptag=ctx.ptag,
        tw=ctx.tw,
        tw_indices=ctx.tw_indices,
        plan=plan,
    )


def resolve_process_needs(
    plan: OutputPlan, stale: bool, ctx: OutputCtx,
) -> ProcessNeeds:
    need_h5 = plan.write_h5 and (not os.path.exists(ctx.post_path) or stale)
    need_plot = plan.do_plot and (
        any(not os.path.exists(p) for p in ctx.plot_paths) or stale
    )
    tw_compute = _tw_indices_to_compute(
        stale=stale, need_h5=need_h5, need_plot=need_plot, plan=plan, ctx=ctx,
    )
    return ProcessNeeds(
        stem=ctx.stem, outdir=ctx.outdir, ptag=ctx.ptag, out_h5=ctx.post_path,
        sidecar=ctx.sidecar, tw_indices=ctx.tw_indices, plot_paths=ctx.plot_paths,
        need_h5=need_h5, need_plot=need_plot, tw_compute=tw_compute,
    )
