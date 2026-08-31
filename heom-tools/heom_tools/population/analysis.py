"""Terminal comparison reports and interpolated comparison data."""
from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np

from heom_tools.common.spectrum_metrics import l2_relative_percent_on_grid

from .models import BasisSeries, ComparisonResult, PopulationData
from .util import (
    apply_time_limit,
    basename_noext,
    common_time_grid,
    interp_populations,
    require_nonempty_time,
)


def population_l2_rel_percent(
    pops_ref: np.ndarray,
    pops_other: np.ndarray,
) -> float:
    """Global L² rel. error (%): 100 × ‖pops_other − pops_ref‖₂ / ‖pops_ref‖₂.

    Flattens the full ``(n_times, n_sites)`` arrays into one vector — not a
    per-site metric. Matches las-plot's ‖Δ‖₂ / ‖ref‖₂ convention with baseline
    = file 1.
    """
    ref = np.asarray(pops_ref, dtype=np.float64)
    other = np.asarray(pops_other, dtype=np.float64)
    if ref.shape != other.shape:
        raise ValueError(
            f"population shape mismatch for L²: {ref.shape} vs {other.shape}"
        )
    return l2_relative_percent_on_grid(ref, other - ref)


def format_compare_l2_label(l2_percent: float) -> str:
    """Legend/annotation text matching las-plot: ``1 vs 2 (L² 0.3%)``."""
    return f"1 vs 2 (L² {l2_percent:.1f}%)"


def physical_hamiltonian_cm(data: PopulationData) -> Optional[np.ndarray]:
    """H in cm⁻¹ with ``w0_cm`` on the diagonal, or ``None`` if missing."""
    if data.hamiltonian_cm is None:
        return None
    H = np.asarray(data.hamiltonian_cm, dtype=np.float64).copy()
    diag = np.diag(H).copy()
    np.fill_diagonal(H, diag + float(data.w0_cm))
    return H


def hamiltonians_compatible(data1: PopulationData, data2: PopulationData) -> bool:
    """True only when both runs expose the same physical Hamiltonian."""
    H1 = physical_hamiltonian_cm(data1)
    H2 = physical_hamiltonian_cm(data2)
    if H1 is None or H2 is None:
        return False
    if H1.shape != H2.shape:
        return False
    return bool(np.allclose(H1, H2, rtol=0.0, atol=1e-6))


def eigenvectors_compatible(
    U1: np.ndarray,
    U2: np.ndarray,
    *,
    atol: float = 1e-5,
) -> bool:
    """True when U1 and U2 are the same ordered basis (column phase OK).

    Requires ``|U1† U2| ≈ I``. Column phases are allowed; column permutation
    or a rotation inside a degenerate subspace is rejected — those would
    mislabel exciton indices across runs.
    """
    A = np.asarray(U1, dtype=np.complex128)
    B = np.asarray(U2, dtype=np.complex128)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    overlap = np.abs(A.conj().T @ B)
    return bool(np.allclose(overlap, np.eye(A.shape[0]), rtol=0.0, atol=atol))


def exciton_skip_reason(
    data1: PopulationData,
    data2: PopulationData,
) -> Optional[str]:
    """Why exciton comparison is unavailable, or ``None`` if bases match.

    ``None`` means H and ordered U are compatible — exciton series may be built.
    """
    if not data1.is_dual_basis or not data2.is_dual_basis:
        if data1.is_dual_basis or data2.is_dual_basis:
            return "only one run has exciton populations"
        return (
            "no exciton populations (need HDF5 with density_matrix + eigenvectors)"
        )
    if not hamiltonians_compatible(data1, data2):
        return "Hamiltonians differ or missing"
    if data1.eigenvectors is None or data2.eigenvectors is None:
        return (
            "eigenvectors missing (exciton compare requires matching U from "
            "HDF5 /derived/eigenvectors; text exports are not supported)"
        )
    if not eigenvectors_compatible(data1.eigenvectors, data2.eigenvectors):
        return (
            "eigenvectors differ "
            "(column phase OK; permutation / degenerate subspace not)"
        )
    return None


def _basis_series(
    time_common: np.ndarray,
    time1: np.ndarray,
    pops1: np.ndarray,
    time2: np.ndarray,
    pops2: np.ndarray,
    n_sites: int,
) -> BasisSeries:
    p1 = interp_populations(time_common, time1, pops1, n_sites)
    p2 = interp_populations(time_common, time2, pops2, n_sites)
    return BasisSeries(pops1=p1, pops2=p2, diff=p2 - p1)


def create_comparison_result(
    data1: PopulationData,
    data2: PopulationData,
) -> ComparisonResult:
    """Build site (always) and exciton (only if H and U match) comparisons."""
    if data1.n_sites != data2.n_sites:
        raise ValueError(
            f"site count mismatch: {data1.n_sites} vs {data2.n_sites}"
        )

    n_sites = data1.n_sites
    time_common = common_time_grid(data1.time, data2.time)
    if time_common.size == 0:
        raise ValueError("no overlapping time range between the two runs")

    site = _basis_series(
        time_common,
        data1.time,
        data1.populations_site,
        data2.time,
        data2.populations_site,
        n_sites,
    )

    exciton: Optional[BasisSeries] = None
    reason = exciton_skip_reason(data1, data2)
    if reason is None:
        assert data1.populations_exciton is not None
        assert data2.populations_exciton is not None
        exciton = _basis_series(
            time_common,
            data1.time,
            data1.populations_exciton,
            data2.time,
            data2.populations_exciton,
            n_sites,
        )
    elif data1.is_dual_basis and data2.is_dual_basis:
        warnings.warn(
            f"exciton comparison skipped: {reason}",
            UserWarning,
            stacklevel=2,
        )

    return ComparisonResult(
        data1=data1,
        data2=data2,
        time_common=time_common,
        site=site,
        exciton=exciton,
    )


def windowed_series(
    result: ComparisonResult,
    time_limit: Optional[float] = None,
) -> tuple[np.ndarray, BasisSeries, Optional[BasisSeries]]:
    """Apply the same time window used by compare figures (for L² / reports)."""
    t, p1, p2, d = apply_time_limit(
        result.time_common,
        time_limit,
        result.site.pops1,
        result.site.pops2,
        result.site.diff,
    )
    require_nonempty_time(t, time_limit, data_time=result.time_common)
    site = BasisSeries(pops1=p1, pops2=p2, diff=d)
    exciton = None
    if result.exciton is not None:
        _, e1, e2, ed = apply_time_limit(
            result.time_common,
            time_limit,
            result.exciton.pops1,
            result.exciton.pops2,
            result.exciton.diff,
        )
        exciton = BasisSeries(pops1=e1, pops2=e2, diff=ed)
    return t, site, exciton


def print_file_info(data: PopulationData, label: str) -> None:
    print(f"\n{label}: {os.path.basename(data.filename)}")
    print(f"  - 时间范围: {data.time[0]:.1f} - {data.time[-1]:.1f} fs")
    print(f"  - 时间点数: {data.n_times}")
    print(f"  - 位点数: {data.n_sites}")
    if data.temperature:
        print(f"  - 温度: {data.temperature}")


def print_comparison_table(
    title: str,
    col_labels: list[str],
    row_labels: list[str],
    data1: np.ndarray,
    data2: np.ndarray,
) -> None:
    print(f"\n{title}:")
    label1, label2 = col_labels[0][:15], col_labels[1][:15]
    print(f"{'Site':<10} {label1:<15} {label2:<15} {'Difference':<15}")
    print("-" * 55)

    for i, site in enumerate(row_labels):
        diff = data2[i] - data1[i]
        print(f"{site:<10} {data1[i]:>14.6f} {data2[i]:>14.6f} {diff:>14.6f}")


def print_statistics(
    result: ComparisonResult,
    site: BasisSeries,
    time: np.ndarray,
) -> None:
    print("\n" + "-" * 70)
    print("STATISTICAL ANALYSIS")
    print("-" * 70)

    n_sites = site.pops1.shape[1]
    labels = result.data1.site_labels[:n_sites]
    file1_name = basename_noext(result.data1.filename)
    file2_name = basename_noext(result.data2.filename)

    print_comparison_table(
        "初始布居 (window start)",
        [file1_name, file2_name],
        labels,
        site.pops1[0, :n_sites],
        site.pops2[0, :n_sites],
    )
    print_comparison_table(
        "最终布居 (window end)",
        [file1_name, file2_name],
        labels,
        site.pops1[-1, :n_sites],
        site.pops2[-1, :n_sites],
    )

    print("\n平均绝对差值 (Mean Absolute Difference):")
    print(f"{'Site':<10} {'MAD':<15} {'Max Diff':<15} {'Min Diff':<15}")
    print("-" * 55)
    for i in range(n_sites):
        mad = np.mean(np.abs(site.diff[:, i]))
        max_diff = np.max(site.diff[:, i])
        min_diff = np.min(site.diff[:, i])
        print(f"{labels[i]:<10} {mad:>14.6f} {max_diff:>14.6f} {min_diff:>14.6f}")

    print(f"\n(window: {time[0]:.1f} - {time[-1]:.1f} fs, {len(time)} points)")


def print_transfer_analysis(result: ComparisonResult, site: BasisSeries) -> None:
    print("\n" + "-" * 70)
    print("POPULATION TRANSFER ANALYSIS")
    print("-" * 70)

    n_sites = site.pops1.shape[1]
    labels = result.data1.site_labels[:n_sites]
    file1_name = basename_noext(result.data1.filename)
    file2_name = basename_noext(result.data2.filename)

    for i in range(n_sites):
        change1 = site.pops1[-1, i] - site.pops1[0, i]
        change2 = site.pops2[-1, i] - site.pops2[0, i]
        print(
            f"{labels[i]:<10}: {file1_name}变化 = {change1:>8.6f}, "
            f"{file2_name}变化 = {change2:>8.6f}, 差异 = {change2 - change1:>8.6f}"
        )


def print_correlation(site: BasisSeries, labels: list[str], time: np.ndarray) -> None:
    print("\n" + "-" * 70)
    print("CORRELATION ANALYSIS")
    print("-" * 70)
    print(f"(window: {time[0]:.1f} - {time[-1]:.1f} fs, site basis)")

    n_sites = site.pops1.shape[1]
    for i in range(n_sites):
        corr = np.corrcoef(site.pops1[:, i], site.pops2[:, i])[0, 1]
        print(f"{labels[i]:<10}: 相关系数 = {corr:>8.6f}")


def print_analysis(
    result: ComparisonResult,
    *,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> None:
    """Print comparison summary.

    L² uses the same time window as compare figures when ``time_limit`` is set.
    Detailed per-site tables only when ``verbose`` is true.
    """
    time, site, exciton = windowed_series(result, time_limit)

    print("\n" + "=" * 70)
    print("POPULATION DYNAMICS COMPARISON ANALYSIS")
    print("=" * 70)

    print_file_info(result.data1, "文件1")
    print_file_info(result.data2, "文件2")

    print("\n" + "-" * 70)
    print("GLOBAL L² RELATIVE ERROR (all times × sites/excitons in window)")
    print("-" * 70)
    print(f"  window: {time[0]:.1f} - {time[-1]:.1f} fs ({len(time)} points)")
    print(
        f"  site basis:    "
        f"{format_compare_l2_label(population_l2_rel_percent(site.pops1, site.pops2))}"
    )
    if exciton is not None:
        print(
            f"  exciton basis: "
            f"{format_compare_l2_label(population_l2_rel_percent(exciton.pops1, exciton.pops2))}"
        )
    elif result.data1.is_dual_basis or result.data2.is_dual_basis:
        reason = exciton_skip_reason(result.data1, result.data2) or "unknown"
        print(f"  exciton basis: skipped ({reason})")

    if verbose:
        print_statistics(result, site, time)
        print_transfer_analysis(result, site)
        n_sites = site.pops1.shape[1]
        print_correlation(site, result.data1.site_labels[:n_sites], time)

    print("\n" + "=" * 70 + "\n")
