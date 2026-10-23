"""Shared LAS plot/compare pipeline (entry points: ``las-plot``, ``las-compare``)."""
from __future__ import annotations

import hashlib
import os
import re
import sys
import warnings
from typing import Optional

from heom_tools.common.config_diff import load_config_diff_summary, print_config_yaml_diffs
from heom_tools.common.spectrum import ExperimentSpectrum, discover_experiment_spectra_auto

from .align import align_las_compare, load_theory_from_spectrum_h5, print_metrics_table
from .plot import plot_las_compare_png, show_las_compare_interactive


def sanitize_filename_stem(stem: str, max_len: int = 200) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem.strip())
    s = re.sub(r"\s+", "_", s)
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "spectrum"


def inputs_path_hash(h5_paths: list[str], n_hex: int = 8) -> str:
    """Stable short tag from the set of input *paths* (not file contents)."""
    h = hashlib.sha256()
    for p in sorted(os.path.normpath(os.path.abspath(x)) for x in h5_paths):
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:n_hex]


def allocate_output_png_path(
    out_dir: str,
    h5_paths: list[str],
    *,
    prefix: str = "las_compare",
) -> str:
    if not h5_paths:
        raise ValueError("need at least one h5 path")
    first_stem = os.path.splitext(os.path.basename(h5_paths[0]))[0]
    safe = sanitize_filename_stem(first_stem)
    tag = inputs_path_hash(h5_paths)
    base = f"{prefix}_{safe}_{tag}"
    for n in range(1, 10_000):
        name = f"{base}.png" if n == 1 else f"{base}_{n}.png"
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            return path
    raise RuntimeError("could not allocate a free output filename in out_dir")


def interactive_display_available() -> bool:
    if os.environ.get("LAS_PLOT_NO_GUI", "").strip().lower() in ("1", "true", "yes"):
        return False
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def resolve_experiment(
    out_dir: str,
    experiment_path: Optional[str],
    *,
    auto: bool,
) -> Optional[ExperimentSpectrum]:
    if experiment_path:
        return ExperimentSpectrum.load(experiment_path)
    if not auto:
        return None
    found = discover_experiment_spectra_auto(out_dir)
    if not found:
        return None
    if len(found) > 1:
        warnings.warn(
            f"multiple experimental files in {out_dir!r}; "
            f"using first only for residuals: {found[0].label!r}",
            UserWarning,
        )
    return found[0]


def run_las_pipeline(
    h5_paths: list[str],
    *,
    cmd_name: str,
    output_prefix: str,
    title: str,
    experiment_path: Optional[str] = None,
    auto_experiment: bool = True,
    no_config_diff: bool = False,
) -> int:
    """Load theories, align, print metrics, write PNG, optional GUI."""
    out_dir = os.path.dirname(os.path.abspath(h5_paths[0]))
    theories = [load_theory_from_spectrum_h5(pth) for pth in h5_paths]
    experiment = resolve_experiment(
        out_dir, experiment_path, auto=auto_experiment,
    )

    h5_abs = [os.path.abspath(os.path.normpath(p)) for p in h5_paths]
    out = allocate_output_png_path(out_dir, h5_abs, prefix=output_prefix)

    print(f"[{cmd_name}]", flush=True)
    print(f"  theory files ({len(theories)}):", flush=True)
    for t in theories:
        print(f"    - {t.label}", flush=True)
    if len(theories) >= 2:
        print(f"  baseline theory = {theories[0].label!r} (first .h5)", flush=True)
    print(f"  output_dir = {os.path.abspath(out_dir)}  (first .h5 directory)", flush=True)
    print(f"  output_png = {os.path.basename(out)}", flush=True)
    print(
        f"  experiment = {experiment.label!r}" if experiment else "  experiment = (none)",
        flush=True,
    )

    config_diff_text = None
    if len(h5_abs) >= 2:
        config_summary = load_config_diff_summary(h5_abs, baseline_idx=0)
        if config_summary is not None:
            print_config_yaml_diffs(summary=config_summary)
            if not no_config_diff:
                config_diff_text = config_summary.overlay_text

    try:
        aligned = align_las_compare(theories, experiment)
        if aligned.residuals:
            print_metrics_table(aligned.x_residual, aligned.residuals)
    except ValueError as e:
        print(f"{cmd_name}: {e}", file=sys.stderr)
        return 1

    try:
        plot_las_compare_png(
            aligned, out, config_diff_text=config_diff_text, title=title,
        )
    except ImportError as e:
        print(
            f"[{cmd_name}] matplotlib import failed — PNG not written "
            "(metrics above are still valid). "
            "Use heom-tools/environment-heom-post.yml / upgrade numpy.\n"
            f"  {e}",
            file=sys.stderr,
        )
        return 1
    except (OSError, PermissionError, ValueError, RuntimeError) as e:
        print(
            f"[{cmd_name}] PNG save failed (metrics above are still valid).\n"
            f"  {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1

    print(f"[{cmd_name}] wrote {os.path.abspath(out)}")
    if interactive_display_available():
        try:
            print(f"[{cmd_name}] close the plot window to exit …")
            show_las_compare_interactive(
                aligned, config_diff_text=config_diff_text, title=title,
            )
        except (ImportError, OSError, RuntimeError):
            warnings.warn(
                "interactive plot window unavailable (matplotlib / GUI backend)",
                UserWarning,
            )
    else:
        print(
            f"[{cmd_name}] no GUI display "
            "(set DISPLAY/WAYLAND or unset LAS_PLOT_NO_GUI); skip window",
            flush=True,
        )

    return 0
