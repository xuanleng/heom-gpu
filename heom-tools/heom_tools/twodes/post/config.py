"""Config loading and runtime reporting for 2DES post-processing."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Union

from .constants import CONFIG_CANDIDATES, DEFAULTS, DW_DEFAULT_CM, FFT_RUNTIME_NOTES
from .fft import CONFIG_FFT_METHODS, _FFT_DEPRECATED_ALIASES, dw_phys_cm
from .h5_spectra import normalize_h5_spectra

PolType = Union[str, tuple]


@dataclass
class PostConfig:
    """Resolved twodes-post configuration."""

    out: Optional[str] = None
    scan_dir: Optional[str] = None
    pol: PolType = "ap"
    tw_index: Any = "all"
    w0: float = 0.0
    dw: float = DW_DEFAULT_CM
    window: Optional[list] = None
    levels: int = 25
    lines: List[float] = field(default_factory=list)
    plot: bool = True
    time_domain: bool = True
    write_h5: bool = False
    h5_spectra: Optional[List[str]] = None
    force: bool = False
    fft_method: str = "axis"

    @classmethod
    def defaults(cls) -> PostConfig:
        return cls(**DEFAULTS)

    def with_updates(self, **kwargs) -> PostConfig:
        return replace(self, **kwargs)


def effective_fft_method(method: str) -> str:
    """Spectra math uses axis when method is check."""
    return "axis" if method == "check" else method


def parse_window(window, *, validate: bool = False):
    if window and all(v is not None for v in window):
        lo, hi = float(window[0]), float(window[1])
        if validate and lo >= hi:
            raise ValueError(f"window lo must be < hi, got {lo} {hi}")
        return [lo, hi]
    return None


def normalize_tw_index(val) -> Any:
    """Comparable tw_index for cfg attrs, sidecar JSON, and post H5 metadata."""
    if isinstance(val, str):
        text = val.strip()
        if text.lower() == "all":
            return "all"
        if text.startswith("["):
            return json.loads(text)
        return int(text)
    if isinstance(val, (list, tuple)):
        return [int(i) for i in val]
    return int(val)


def tw_index_to_attr(val) -> str:
    norm = normalize_tw_index(val)
    if norm == "all":
        return "all"
    if isinstance(norm, list):
        return json.dumps(norm)
    return str(norm)


def cfg_pol_str(pol) -> str:
    return pol if isinstance(pol, str) else ",".join(str(x) for x in pol)


def pol_tag(pol):
    s = pol if isinstance(pol, str) else "_".join(str(a) for a in pol)
    return s.replace(",", "_")


def tw_tag(tw_fs: float) -> str:
    """Filesystem label for waiting time, e.g. 0fs, 1000fs."""
    v = float(tw_fs)
    if abs(v - round(v)) < 1e-6:
        return f"{int(round(v))}fs"
    return f"{v:g}fs"


def plot_png_path(outdir: str, stem: str, ptag: str, kind: str, tw_fs: float) -> str:
    """PNG path for a 2d or td plot at one waiting time."""
    return os.path.join(outdir, f"{stem}_{ptag}_{kind}_tw{tw_tag(tw_fs)}.png")


def resolve_tw_by_config_index(tw_index, n_tw):
    norm = normalize_tw_index(tw_index)
    if norm == "all":
        indices = list(range(n_tw))
    elif isinstance(norm, list):
        indices = norm
    else:
        indices = [norm]
    bad = [k for k in indices if not (0 <= k < n_tw)]
    if bad:
        raise ValueError(f"tw_index {bad} out of range [0,{n_tw})")
    return indices


def post_h5_path(h5: str, cfg: PostConfig) -> str:
    stem = os.path.splitext(os.path.basename(h5))[0]
    outdir = cfg.out or os.path.dirname(os.path.abspath(h5))
    return os.path.join(outdir, f"{stem}_{pol_tag(cfg.pol)}_post.h5")


def _format_cfg_value(key: str, value) -> str:
    if value is None:
        if key == "out":
            return "(blank → next to input .h5)"
        if key == "scan_dir":
            return "(blank → current directory)"
        if key == "window":
            return "(blank → full frequency range)"
        return "null"
    if value == "":
        return "(blank)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return repr(list(value))
    return str(value)


def print_runtime_config(
    cfg: PostConfig,
    *,
    config_path: Optional[str] = None,
    input_h5: Optional[str] = None,
    mode: Optional[str] = None,
) -> None:
    """Print resolved config at run start so the user can verify the right file."""
    width = 52
    bar = "=" * width
    lines = [bar, "twodes-post runtime config"]
    if config_path:
        lines.append(f"  config file : {os.path.abspath(config_path)}")
    if mode:
        lines.append(f"  run mode    : {mode}")
    if input_h5:
        lines.append(f"  input h5    : {os.path.abspath(input_h5)}")
        outdir = cfg.out or os.path.dirname(os.path.abspath(input_h5))
        lines.append(f"  output dir  : {outdir}")
    elif cfg.out:
        lines.append(f"  output dir  : {os.path.abspath(cfg.out)}")
    lines.append("  ---")
    for key in DEFAULTS:
        lines.append(f"  {key:14}: {_format_cfg_value(key, getattr(cfg, key))}")
    fft = cfg.fft_method
    note = FFT_RUNTIME_NOTES.get(fft, fft)
    lines.append(f"  {'→ fft active':14}: {fft}  ({note})")
    lines.append(bar)
    print("\n".join(lines))


def _fft_axis_line(axis: str, dt_key: str, n_data: int, n_fft: int, dt_fs: float) -> str:
    if n_fft > n_data:
        pad = n_fft - n_data
        return f"{axis} n={n_data}→{n_fft} ({dt_key}={dt_fs:g} fs, +{pad} pad)"
    return f"{axis} n={n_data} ({dt_key}={dt_fs:g} fs)"


def print_fft_grid_info(
    *,
    dtau: float,
    dt: float,
    n_tau_data: int,
    n_tau_fft: int,
    n_t_data: int,
    n_t_fft: int,
    dw: float,
    w0: float,
    n_w_tau: Optional[int] = None,
    n_w_t: Optional[int] = None,
):
    """Resolved FFT padding, published ``w0+k·dw`` lattice, and phys resolve."""
    tau = _fft_axis_line("τ", "dtau", n_tau_data, n_tau_fft, dtau)
    t_ax = _fft_axis_line("t", "dt", n_t_data, n_t_fft, dt)
    print(f"  FFT grid    : {tau}  {t_ax}  dw={dw:g} cm^-1  w0={w0:g}")
    if n_w_tau is not None and n_w_t is not None:
        print(
            f"  ω lattice   : nτ={n_w_tau} nt={n_w_t}  "
            f"output Δω={dw:g} cm^-1 (w0 + k·dw)"
        )
    dwp_tau = dw_phys_cm(n_tau_data, dtau)
    dwp_t = dw_phys_cm(n_t_data, dt)
    print(
        f"  ω resolve   : Δω_phys(τ)≈{dwp_tau:.3g} cm^-1  "
        f"Δω_phys(t)≈{dwp_t:.3g} cm^-1  "
        f"(FS_TO_CM/(n·dt); pad densifies, does not resolve)"
    )


def _validate_config(cfg: PostConfig) -> PostConfig:
    method = str(cfg.fft_method).lower()
    if method in _FFT_DEPRECATED_ALIASES:
        new = _FFT_DEPRECATED_ALIASES[method]
        raise ValueError(
            f"fft_method {method!r} is deprecated and maps to {new!r}, which is "
            f"not a production setting (spectra ≠ axis). Use one of "
            f"{CONFIG_FFT_METHODS}."
        )
    if method == "legacy_mode3":
        raise ValueError(
            "fft_method 'legacy_mode3' is not a production setting "
            "(refs fft_2d_data mode 3; spectra ≠ axis). Use 'axis' or 'check'. "
            "For tests, call two_d_fft(..., method='legacy_mode3') or "
            "two_d_fft_legacy_mode3 directly."
        )
    if method not in CONFIG_FFT_METHODS:
        raise ValueError(
            f"fft_method must be one of {CONFIG_FFT_METHODS}, got {cfg.fft_method!r}"
        )
    if float(cfg.dw) <= 0.0:
        raise ValueError(f"dw must be positive, got {cfg.dw!r}")
    if int(cfg.levels) < 1:
        raise ValueError(f"levels must be >= 1, got {cfg.levels!r}")
    h5_spectra = normalize_h5_spectra(cfg.h5_spectra, write_h5=bool(cfg.write_h5))
    return replace(cfg, fft_method=method, h5_spectra=h5_spectra)


def load_config(path: str) -> PostConfig:
    if not os.path.exists(path):
        raise ValueError(f"config not found: {path}")
    with open(path) as f:
        text = f.read()
    if path.endswith(".json"):
        user = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            raise ValueError("PyYAML required for .yaml config (pip install pyyaml)") from None
        user = yaml.safe_load(text)
    user = user or {}
    if not isinstance(user, dict):
        raise ValueError("config root must be a mapping")
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config keys {unknown}")
    merged = {**DEFAULTS, **user}
    return _validate_config(PostConfig(**merged))


def find_config(data_dir: str) -> str:
    """Return twodes_post config path in the same directory as the data."""
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise ValueError(f"data directory not found: {data_dir}")
    for name in CONFIG_CANDIDATES:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    names = ", ".join(CONFIG_CANDIDATES)
    raise ValueError(f"no config file ({names}) next to data in {data_dir}")
