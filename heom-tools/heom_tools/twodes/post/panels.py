"""Matplotlib diagnostic panels for twodes-post."""
from __future__ import annotations

import glob
import os
import re
from typing import Any, List

import numpy as np

from .config import plot_png_path, tw_tag
from heom_tools.twodes.viz import cut_window, draw_freq_panel, make_norm

from .core import (
    combined_freq_rows,
    combined_time_rows,
    time_domain_combo,
)
from .types import OutputPlan, PlotCtx


def prune_orphan_plots(
    outdir: str, stem: str, ptag: str, tw, tw_indices: List[int], *, td: bool,
) -> None:
    """Remove PNGs for waiting times no longer in cfg.tw_index."""
    keep = {tw_tag(float(tw[k])) for k in tw_indices}
    for kind in ("2d", "td") if td else ("2d",):
        prefix = f"{stem}_{ptag}_{kind}_tw"
        pattern = re.compile(rf"^{re.escape(prefix)}(.+)\.png$")
        for fp in glob.glob(os.path.join(outdir, f"{prefix}*.png")):
            m = pattern.match(os.path.basename(fp))
            if m and m.group(1) not in keep:
                os.remove(fp)
                print(f"removed orphan {fp}")


def freq_panel(
    ax,
    data2d,
    title,
    ctx: PlotCtx,
    *,
    mpl: Any,
    plt_mod: Any,
    cmap: Any,
    cmap_pos: Any,
    nonnegative: bool = False,
) -> None:
    x, y, d = cut_window(ctx.w_tau, ctx.w_t, data2d, ctx.window)
    cm = cmap_pos if nonnegative else cmap
    draw_freq_panel(
        ax, x, y, d, title=title, mpl=mpl, plt_mod=plt_mod,
        cmap=cm, levels=ctx.levels, lines=ctx.lines, window=ctx.window,
        nonnegative=nonnegative,
    )


def time_panel(
    ax,
    data,
    title,
    ctx: PlotCtx,
    *,
    mpl: Any,
    plt_mod: Any,
    cmap: Any,
    x_fs: np.ndarray,
) -> None:
    norm = make_norm(data, mpl)
    ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, norm=norm,
              interpolation="spline36",
              extent=[x_fs.min(), x_fs.max(), ctx.t.min(), ctx.t.max()])
    ax.contour(x_fs, ctx.t, data, levels=ctx.levels, colors="k", linewidths=0.4)
    ax.axvline(0.0, color="k", ls="--", lw=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\tau,\ t_1$ (fs)")
    ax.set_ylabel(r"$t,\ t_3$ (fs)")
    ax.set_title(title)
    cb = plt_mod.colorbar(ax.images[-1], ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)


def plot_combined_freq(
    outdir, stem, ptag, tw_val, pol, rows, ctx: PlotCtx, *, plt_mod: Any,
    cmap: Any, cmap_pos: Any, mpl: Any,
) -> None:
    head = f"{stem}  T={tw_val:.0f} fs  {pol}"
    n = len(rows)
    fig, axes = plt_mod.subplots(n, 3, figsize=(14, 3.2 * n), squeeze=False)
    for r, (name, spec) in enumerate(rows):
        freq_panel(
            axes[r][0], spec.real, f"Re {name}", ctx,
            mpl=mpl, plt_mod=plt_mod, cmap=cmap, cmap_pos=cmap_pos,
        )
        freq_panel(
            axes[r][1], spec.imag, f"Im {name}", ctx,
            mpl=mpl, plt_mod=plt_mod, cmap=cmap, cmap_pos=cmap_pos,
        )
        freq_panel(
            axes[r][2], np.abs(spec), f"|{name}|", ctx,
            mpl=mpl, plt_mod=plt_mod, cmap=cmap, cmap_pos=cmap_pos, nonnegative=True,
        )
    fig.suptitle(head, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fp = plot_png_path(outdir, stem, ptag, "2d", tw_val)
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    plt_mod.close(fig)
    print(f"wrote {fp}")


def plot_combined_time(
    outdir, stem, ptag, tw_val, pol, rows, ctx: PlotCtx, *,
    plt_mod: Any, cmap: Any, mpl: Any,
) -> None:
    head = f"{stem}  T={tw_val:.0f} fs  {pol}"
    n = len(rows)
    fig, axes = plt_mod.subplots(n, 2, figsize=(10, 3.2 * n), squeeze=False)
    for r, (name, slabs) in enumerate(rows):
        x_fs, combo = time_domain_combo(slabs, ctx.tau)
        time_panel(
            axes[r][0], combo.real, f"Re {name}", ctx, x_fs=x_fs,
            mpl=mpl, plt_mod=plt_mod, cmap=cmap,
        )
        time_panel(
            axes[r][1], combo.imag, f"Im {name}", ctx, x_fs=x_fs,
            mpl=mpl, plt_mod=plt_mod, cmap=cmap,
        )
    fig.suptitle(head, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fp = plot_png_path(outdir, stem, ptag, "td", tw_val)
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    plt_mod.close(fig)
    print(f"wrote {fp}")


def render_plots(
    *,
    outdir,
    stem,
    ptag,
    tw,
    tw_indices,
    pol,
    pathways,
    slices,
    stale,
    plan: OutputPlan,
    ctx: PlotCtx,
    mpl: Any,
    plt_mod: Any,
    cmap: Any,
    cmap_pos: Any,
) -> None:
    td = plan.need_time
    for k in tw_indices:
        sl = slices[k]
        tw_val = float(tw[k])
        png_f = plot_png_path(outdir, stem, ptag, "2d", tw_val)
        if stale or not os.path.exists(png_f):
            freq_rows = combined_freq_rows(
                sl.spec_path, sl.spec_total, sl.spec_2des, pathways,
            )
            plot_combined_freq(
                outdir, stem, ptag, tw_val, pol, freq_rows, ctx,
                plt_mod=plt_mod, cmap=cmap, cmap_pos=cmap_pos, mpl=mpl,
            )
        if td and sl.time_total is not None:
            png_t = plot_png_path(outdir, stem, ptag, "td", tw_val)
            if stale or not os.path.exists(png_t):
                time_rows = combined_time_rows(
                    sl.time_path or {}, sl.time_total, sl.spec_2des is not None, pathways,
                )
                plot_combined_time(
                    outdir, stem, ptag, tw_val, pol, time_rows, ctx,
                    plt_mod=plt_mod, cmap=cmap, mpl=mpl,
                )
