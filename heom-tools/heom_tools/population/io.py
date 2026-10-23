"""Load population dynamics from HDF5 and legacy text exports."""
from __future__ import annotations

import os
import warnings
from typing import Any, Optional

import h5py
import numpy as np

from heom_tools.common import heom_h5

from .h5_report import print_hdf5_parameters, print_sink_exciton_analysis
from .models import PopulationData
from .util import MAX_METADATA_LINES


def extract_metadata(line: str) -> tuple[Optional[str], Optional[Any]]:
    line = line.strip()

    if "# Number of sites:" in line:
        try:
            return "n_sites", int(line.split("=")[1].strip().split()[0])
        except (ValueError, IndexError) as e:
            warnings.warn(f"无法解析位点数: {e}")
            return None, None

    if "# Temperature:" in line:
        return "temperature", line.split(":")[1].strip()

    if "# Total time:" in line:
        try:
            return "total_time", float(line.split(":")[1].strip().split()[0])
        except (ValueError, IndexError) as e:
            warnings.warn(f"无法解析总时间: {e}")
            return None, None

    return None, None


def parse_csv_data(lines: list[str], start_idx: int) -> tuple[np.ndarray, np.ndarray]:
    time: list[float] = []
    populations: list[list[float]] = []

    for line in lines[start_idx + 1 :]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(",")
        if len(parts) < 2:
            continue

        try:
            t = float(parts[0])
            pops = [float(x) for x in parts[1:]]
            time.append(t)
            populations.append(pops)
        except ValueError:
            continue

    if not time:
        raise ValueError("没有有效的数据行")

    return np.array(time), np.array(populations)


def is_text_dual_basis(filename: str) -> bool:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for _ in range(MAX_METADATA_LINES):
                line = f.readline()
                if not line:
                    break
                if (
                    "DUAL BASIS OUTPUT ENABLED" in line
                    or "[Population Dynamics - Site Basis]" in line
                    or "[Population Dynamics - Exciton Basis]" in line
                ):
                    return True
    except (OSError, UnicodeDecodeError) as e:
        warnings.warn(f"Failed to detect dual basis in {filename}: {e}")
    return False


def compute_sink_capture(
    time: np.ndarray,
    populations: np.ndarray,
    sink_site: int,
    gamma_sink_fs: float,
) -> tuple[np.ndarray, float, float]:
    rho_sink = populations[:, sink_site]
    integrand = gamma_sink_fs * rho_sink
    dt = np.diff(time)
    N_captured = np.zeros_like(time)
    N_captured[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dt)
    Phi = float(N_captured[-1])
    t50 = float(time[np.argmin(np.abs(N_captured - Phi / 2))]) if Phi > 0 else 0.0
    return N_captured, Phi, t50


def compute_exciton_populations(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    rho_exc = np.einsum("ij,tjk,kl->til", U.conj().T, rho, U)
    return np.real(np.diagonal(rho_exc, axis1=1, axis2=2))


def read_sink_params(f: h5py.File) -> tuple[bool, int, float]:
    sink_enabled = bool(heom_h5.param(f, "sink_enabled", False))
    if not sink_enabled:
        return False, -1, 0.0
    sink_site = int(heom_h5.param(f, "sink_site", -1))
    gamma_sink_fs = float(heom_h5.param(f, "gamma_sink_fs", 0.0))
    return sink_enabled, sink_site, gamma_sink_fs


def extract_hdf5_params(f: h5py.File) -> tuple[Optional[np.ndarray], float, dict]:
    H_cm = (
        f["/parameters/hamiltonian_cm"][...]
        if "/parameters/hamiltonian_cm" in f
        else None
    )
    w0_cm = float(heom_h5.param(f, "w0_cm", 0.0))

    dt = heom_h5.param(f, "dt", None)
    n_steps = heom_h5.param(f, "n_steps", None)
    total_time = dt * n_steps if (dt is not None and n_steps is not None) else 0.0

    init_type = heom_h5.param(f, "initial_state_type", "unknown")
    if init_type == "site_excitation":
        initial_state = f"Site {heom_h5.param(f, 'excited_site', 1)}"
    elif init_type == "exciton_excitation":
        initial_state = (
            f"Exciton {int(heom_h5.param(f, 'selected_exciton_index', 0)) + 1}"
        )
    else:
        initial_state = str(init_type)

    params = {
        "n_sites": heom_h5.param(f, "n_sites"),
        "temperature": heom_h5.param(f, "temperature"),
        "eta_cm": heom_h5.param(f, "eta_cm"),
        "gamma_cm": heom_h5.param(f, "gamma_cm"),
        "n_extra_poles": heom_h5.param(f, "n_extra_poles"),
        "L_cutoff": heom_h5.param(f, "L_cutoff"),
        "M_total": heom_h5.derived(f, "M_total"),
        "dt": dt,
        "total_time": total_time,
        "initial_state": initial_state,
        "sink_enabled": bool(heom_h5.param(f, "sink_enabled", False)),
        "sink_site": int(heom_h5.param(f, "sink_site", -1)),
        "gamma_sink_fs": float(heom_h5.param(f, "gamma_sink_fs", 0.0)),
    }
    return H_cm, w0_cm, params


def parse_hdf5_dual_basis(filename: str, verbose: bool = False) -> PopulationData:
    with h5py.File(filename, "r") as f:
        heom_h5.require_complete(f)
        if verbose:
            print_hdf5_parameters(f, filename)

        time = np.asarray(f["/results/time"][...], dtype=np.float64)
        sink_enabled, sink_site, gamma_sink_fs = read_sink_params(f)
        hamiltonian_cm, w0_cm, params = extract_hdf5_params(f)

        if "/results/density_matrix" not in f:
            raise ValueError(
                f"missing /results/density_matrix in {filename}; "
                "file pre-dates the group-layout refactor"
            )

        rho = np.asarray(f["/results/density_matrix"][...], dtype=np.complex128)
        pops_site = np.real(np.diagonal(rho, axis1=1, axis2=2))

        init_type = heom_h5.param(f, "initial_state_type", "site_excitation")
        primary = "exciton" if init_type == "exciton_excitation" else "site"

        U = None
        pops_exciton = None
        if "/derived/eigenvectors" in f:
            U = np.asarray(f["/derived/eigenvectors"][...], dtype=np.complex128)
            pops_exciton = compute_exciton_populations(rho, U)
            if verbose:
                print("  [INFO] Exciton populations computed from /derived/eigenvectors")

        n_sites = pops_site.shape[1]
        site_labels = [f"Site{i + 1}" for i in range(n_sites)]
        exciton_labels = [f"Exciton{i + 1}" for i in range(n_sites)] if pops_exciton is not None else None

        N_captured, Phi, t50 = None, 0.0, 0.0
        if sink_enabled and 0 <= sink_site < n_sites:
            N_captured, Phi, t50 = compute_sink_capture(
                time, pops_site, sink_site, gamma_sink_fs
            )
            if verbose:
                print(f"  [INFO] Sink capture computed: Φ = {Phi:.4f}, t50 = {t50:.1f} fs")
                if U is not None:
                    print_sink_exciton_analysis(
                        U, sink_site, heom_h5.eigenvalues_absolute_cm(f)
                    )

        data = PopulationData(
            filename=filename,
            time=time,
            populations_site=pops_site,
            site_labels=site_labels,
            populations_exciton=pops_exciton,
            exciton_labels=exciton_labels,
            primary_basis=primary,
            sink_enabled=sink_enabled,
            sink_site=sink_site,
            gamma_sink_fs=gamma_sink_fs,
            N_captured=N_captured,
            Phi=Phi,
            t50=t50,
            hamiltonian_cm=hamiltonian_cm,
            w0_cm=w0_cm,
            eigenvectors=U,
            params=params,
        )
        data.ensure_labels()
        return data


def parse_csv_section(
    lines: list[str], start_idx: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    header_line = lines[start_idx].strip()
    col_names = [s.strip() for s in header_line.split(",")[1:]]

    time_data: list[float] = []
    pop_data: list[list[float]] = []
    for line in lines[start_idx + 1 :]:
        line = line.strip()
        if not line or line.startswith("#") or "[Population Dynamics" in line:
            break
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                time_data.append(float(parts[0]))
                pop_data.append([float(x) for x in parts[1:]])
            except ValueError:
                continue

    return np.array(time_data), np.array(pop_data), col_names


def parse_text_dual_basis(filename: str) -> PopulationData:
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    primary_basis = "site"
    for line in lines:
        if "Primary basis (initial state):" in line:
            if "Exciton" in line:
                primary_basis = "exciton"
            break

    section_starts: list[int] = []
    for i, line in enumerate(lines):
        if "[Population Dynamics -" in line:
            section_starts.append(i + 1)

    if len(section_starts) < 2:
        raise ValueError("Dual basis file should have 2 data sections")

    time1, pops1, names1 = parse_csv_section(lines, section_starts[0])
    time2, pops2, names2 = parse_csv_section(lines, section_starts[1])

    if "Exciton" in names1[0]:
        pops_exciton, pops_site = pops1, pops2
        exciton_labels, site_labels = names1, names2
    else:
        pops_site, pops_exciton = pops1, pops2
        site_labels, exciton_labels = names1, names2

    data = PopulationData(
        filename=filename,
        time=time1,
        populations_site=pops_site,
        site_labels=site_labels,
        populations_exciton=pops_exciton,
        exciton_labels=exciton_labels,
        primary_basis=primary_basis,
    )
    data.ensure_labels()
    return data


def parse_population_file(filename: str, verbose: bool = False) -> PopulationData:
    """Load site (always) and optional exciton populations.

    Cross-run exciton comparison requires HDF5 with matching physical H and
    ``/derived/eigenvectors``. Text dual-basis exports have populations only
    (no H/U), so exciton compare is skipped.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")

    if filename.endswith(".h5") or filename.endswith(".hdf5"):
        return parse_hdf5_dual_basis(filename, verbose=verbose)

    if is_text_dual_basis(filename):
        return parse_text_dual_basis(filename)

    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        raise OSError(f"无法读取文件 {filename}: {e}") from e

    if not lines:
        raise ValueError(f"文件为空: {filename}")

    metadata: dict[str, Any] = {}
    for line in lines:
        if line.strip().startswith("#"):
            key, val = extract_metadata(line)
            if key:
                metadata[key] = val

    data_start_idx = None
    site_labels: list[str] = []
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#"):
            continue
        if "," in line_stripped:
            data_start_idx = i
            site_labels = [s.strip() for s in line_stripped.split(",")[1:]]
            break

    if data_start_idx is None:
        raise ValueError(f"无法找到数据表头在文件 {filename}")

    time, populations = parse_csv_data(lines, data_start_idx)

    data = PopulationData(
        filename=filename,
        time=time,
        populations_site=populations,
        site_labels=site_labels,
        populations_exciton=None,
        exciton_labels=None,
        primary_basis="site",
        temperature=metadata.get("temperature", ""),
        total_time=metadata.get("total_time", 0.0),
    )
    data.ensure_labels()
    return data
