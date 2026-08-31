"""HDF5 parameter reporting for heom_solver dynamics runs."""
from __future__ import annotations

import os
from typing import Optional

import h5py
import numpy as np

from heom_tools.common import heom_h5


def _print_hardware_info(f: h5py.File) -> None:
    print("\n[Hardware Information]")
    print(f"  CPU: {heom_h5.provenance(f, 'cpu_model', 'N/A')}")
    print(f"  GPU: {heom_h5.provenance(f, 'gpu_model', 'N/A')}")
    print(f"  OS:  {heom_h5.provenance(f, 'os', 'N/A')}")


def _print_software_info(f: h5py.File) -> None:
    print("\n[Software Information]")
    print(f"  CUDA Runtime: {heom_h5.provenance(f, 'cuda_version', 'N/A')}")
    print(f"  Compiler: {heom_h5.provenance(f, 'compiler', 'N/A')}")


def _print_system_params(f: h5py.File) -> None:
    print("\n[System Parameters]")
    print(f"  Number of sites: N = {heom_h5.param(f, 'n_sites', 0)}")
    print(f"  Temperature: T = {heom_h5.param(f, 'temperature', 0.0):.1f} K")
    print(f"  Number of Matsubara terms: K = {heom_h5.param(f, 'n_extra_poles', 0)}")


def _print_hamiltonian(f: h5py.File) -> None:
    if "/parameters/hamiltonian_cm" not in f:
        return
    H_cm = f["/parameters/hamiltonian_cm"][...]
    w0_cm = float(heom_h5.param(f, "w0_cm", 0.0))
    print("\n[Hamiltonian Matrix] (in cm^-1)")
    for i in range(H_cm.shape[0]):
        row_str = f"  H[{i},:] = "
        for j in range(H_cm.shape[1]):
            val = H_cm[i, j] + (w0_cm if i == j else 0.0)
            row_str += f"{val:>9.1f}"
        print(row_str)


def _print_spectral_density(f: h5py.File) -> None:
    eta = heom_h5.param_array(f, "eta_cm", None)
    if eta is None:
        return
    gamma_cm = heom_h5.param_array(f, "gamma_cm", None)
    ginv = heom_h5.param_array(f, "gamma_inverse_fs", None)
    fwhm = heom_h5.param_array(f, "static_disorder_fwhm_cm", None)
    homogeneous = np.allclose(eta, eta[0]) and (
        ginv is None or np.allclose(ginv, ginv[0])
    )

    print("\n[Spectral Density - Drude-Lorentz]")
    if homogeneous:
        print(
            f"  Drude coupling η (J): {eta[0]:.1f} cm^-1  "
            f"(λ_reorg = η/2 = {eta[0] / 2:.1f})  [all sites]"
        )
        if gamma_cm is not None:
            print(f"  Relaxation rate: γ = {gamma_cm[0]:.1f} cm^-1")
        if ginv is not None:
            print(f"  Relaxation time: 1/γ = {ginv[0]:.1f} fs")
    else:
        print("  Per-site (1-based pigment):")
        print("    site   eta(cm^-1)   1/gamma(fs)   FWHM(cm^-1)")
        for n in range(eta.size):
            g = ginv[n] if ginv is not None else float("nan")
            w = fwhm[n] if fwhm is not None else 0.0
            print(f"    {n + 1:>4d}   {eta[n]:>10.1f}   {g:>11.1f}   {w:>11.1f}")


def _print_matsubara(f: h5py.File) -> None:
    if "/derived/bath_c_cm2" not in f or "/derived/bath_nu_cm" not in f:
        return
    c = np.asarray(f["/derived/bath_c_cm2"][...], dtype=np.complex128)
    nu = np.asarray(f["/derived/bath_nu_cm"][...], dtype=np.float64)
    FS_TO_AU, CM_TO_AU = 41.341374575751, 4.556335e-6

    c = np.atleast_2d(c)
    nu = np.atleast_2d(nu)
    n_bath = c.shape[0]

    print("\n[Matsubara Expansion Parameters]")
    print("# site  k    c_k (real)       c_k (imag)       ν_k          1/ν_k")
    print("#                     (cm^-1)         (cm^-1)         (cm^-1)      (fs)")
    for n in range(n_bath):
        for k in range(c.shape[1]):
            nu_inv = (
                1.0 / (nu[n, k] * CM_TO_AU) / FS_TO_AU if nu[n, k] > 0 else 0.0
            )
            print(
                f"#  {n + 1:2d} {k:2d}   {c[n, k].real:14.6f}   {c[n, k].imag:14.6f}"
                f"   {nu[n, k]:14.6f}   {nu_inv:10.4f}"
            )


def _print_hierarchy(f: h5py.File) -> None:
    L_cutoff = heom_h5.param(f, "L_cutoff", None)
    if L_cutoff is None:
        return
    print("\n[Hierarchy Truncation]")
    print("  Scheme: Simple truncation (Σα_{nk} ≤ L)")
    print(f"  Cutoff: L = {L_cutoff}")
    L_total = heom_h5.param(f, "L_total", None)
    npa = heom_h5.param(f, "npa", None)
    if L_total is not None:
        print(f"  Joint budget: L_total = {L_total} (npa = {npa})")
    M_total = heom_h5.derived(f, "M_total", None)
    if M_total is not None:
        print(f"  Total auxiliary matrices: M = {M_total}")
    use_filtering = heom_h5.param(f, "use_filtering", None)
    filter_threshold = heom_h5.param(f, "filter_threshold", None)
    if use_filtering is not None and filter_threshold is not None:
        status = "enabled" if use_filtering else "disabled"
        print(f"  ADO filtering: {status} (threshold = {filter_threshold:.6e})")


def _print_time_evolution(f: h5py.File) -> None:
    dt = heom_h5.param(f, "dt", None)
    n_steps = heom_h5.param(f, "n_steps", None)
    if dt is None or n_steps is None:
        return
    print("\n[Time Evolution]")
    print(f"  Time step: dt = {dt:.6f} fs")
    print(f"  Number of steps: {n_steps}")
    print(f"  Total time: {dt * n_steps:.6f} fs")
    output_interval = heom_h5.param(f, "output_interval", None)
    if output_interval is not None:
        print(f"  Output interval: every {output_interval} steps")


def _print_initial_state(f: h5py.File) -> None:
    init_type = heom_h5.param(f, "initial_state_type", None)
    if init_type is None:
        return
    print("\n[Initial State]")

    if init_type == "site_excitation":
        site = heom_h5.param(f, "excited_site", None)
        if site is not None:
            print("  Type: Site excitation")
            print(f"  Excited pigment: {site} (1-based physical index, basis-aware)")
    elif init_type == "exciton_excitation":
        exciton = heom_h5.param(f, "selected_exciton_index", None)
        if exciton is not None:
            print("  Type: Exciton excitation")
            print(f"  Selected exciton: {exciton + 1}")
            if "/derived/eigenvalues_relative_cm" in f:
                eig_rel = np.asarray(
                    f["/derived/eigenvalues_relative_cm"][...], dtype=np.float64
                )
                eig_abs = heom_h5.eigenvalues_absolute_cm(f)
                print("\n  Exciton Energies:")
                print(f"  {'Index':<8} {'E_abs (cm^-1)':<18} {'E_rel (cm^-1)':<18}")
                for i in range(eig_abs.size):
                    marker = " <--" if i == exciton else ""
                    print(
                        f"  {i + 1:<8} {eig_abs[i]:>16.2f}   {eig_rel[i]:>16.2f}{marker}"
                    )

    if "/parameters/initial_density_matrix" in f:
        rho0 = np.asarray(
            f["/parameters/initial_density_matrix"][...], dtype=np.complex128
        )
        print("\n  Density matrix ρ(0):")
        for i in range(rho0.shape[0]):
            row = "".join(
                f"({rho0[i, j].real:>8.6f}{rho0[i, j].imag:+.6f}j)"
                for j in range(rho0.shape[1])
            )
            print(f"    ρ[{i},:] = {row}")


def _print_output_settings(f: h5py.File) -> None:
    print("\n[Output Settings]")
    output_basis = heom_h5.param(f, "output_basis", None)
    if output_basis is not None:
        print(f"  Output basis: {output_basis}")
    if "/results/density_matrix" in f:
        print("  Format: density_matrix (complex compound under /results)")
        if "/derived/eigenvectors" in f:
            print("  Exciton transform: available (eigenvectors in /derived)")


def print_hdf5_parameters(f: h5py.File, filename: str) -> None:
    print("\n" + "=" * 70)
    print(f"HDF5 Model Parameters: {os.path.basename(filename)}")
    print("=" * 70)

    _print_hardware_info(f)
    _print_software_info(f)
    _print_system_params(f)
    _print_hamiltonian(f)
    _print_spectral_density(f)
    _print_matsubara(f)
    _print_hierarchy(f)
    _print_time_evolution(f)
    _print_initial_state(f)
    _print_output_settings(f)

    print("\n" + "=" * 70 + "\n")


def print_sink_exciton_analysis(
    U: np.ndarray,
    sink_site: int,
    eigenvalues_cm: Optional[np.ndarray] = None,
) -> None:
    n_sites = U.shape[0]
    weights = np.abs(U[sink_site, :]) ** 2

    print("\n" + "-" * 60)
    print(f"Sink-Exciton Coupling Analysis (Sink: Site {sink_site + 1})")
    print("-" * 60)
    print("激子态在 Sink 位点的权重 |U_sink,n|²：")
    print("(权重越高 → 该激子态越容易被 Sink 捕获)")
    print()

    if eigenvalues_cm is not None:
        print(f"{'Exciton':<10} {'Energy (cm⁻¹)':<16} {'|U_sink|²':<12} {'Bar'}")
        print("-" * 55)
        for n in range(n_sites):
            bar = "█" * int(weights[n] * 40)
            print(
                f"  {n + 1:<8} {eigenvalues_cm[n]:>14.1f}   {weights[n]:>10.4f}   {bar}"
            )
    else:
        print(f"{'Exciton':<10} {'|U_sink|²':<12} {'Bar'}")
        print("-" * 40)
        for n in range(n_sites):
            bar = "█" * int(weights[n] * 40)
            print(f"  {n + 1:<8} {weights[n]:>10.4f}   {bar}")

    max_idx = np.argmax(weights)
    print()
    print(f"最强耦合: Exciton {max_idx + 1} (|U|² = {weights[max_idx]:.4f})")
    print("-" * 60 + "\n")
