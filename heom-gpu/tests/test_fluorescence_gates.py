"""Fluorescence gates that need the solver binary plus NumPy.

G4a: one pigment with two underdamped modes; the solver's absorption and
fluorescence X_0(t) equal the NumPy hierarchy (same expansion read from the
file, same L, dt, RK4).
G4b: pure-Drude pigment; max_t |C − C_cum| against the cumulant closed form
(same expansion) falls with L, for absorption and fluorescence separately.
G5: the Stokes shift of the first output step, ω_A − ω_F with ω = −arg C(Δ)/Δ,
matches the closed form on the same Δ (the one-step difference is itself
biased against 2λ by O(Δ); 2λ exactly is G3's first-tier sum).
G6: FMO-7, 77 K: T_eff from the flu-post KMS fit; Padé K = 2 is within G6_TOL
and well below K = 1 (whose expansion caps T_eff at ~ −4 % for any L).
FP64 only: the MP build's error is checked by the C++ gates.
"""
import os
import subprocess

import h5py
import numpy as np
import pytest
import yaml

from fl_monomer_ref import CM_FS, monomer_x0
from heom_tools.common.cumulant import monomer_correlations

ROOT = os.environ.get("HEOM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOLVER = os.environ.get("HEOM_SOLVER", os.path.join(ROOT, "build", "heom_solver"))
MONO_VIB = os.path.join(ROOT, "tests", "configs", "fl_monomer_vib.yaml")
MONO = os.path.join(ROOT, "tests", "configs", "fl_monomer.yaml")
G4A_TOL = 1e-10
G4B_L = (2, 4, 6)
G4B_TOL = {"absorption": 3e-3, "fluorescence": 9e-3}   # at L = 6; measured 1.5e-3 / 4.5e-3
G5_TOL = 1e-6                                           # measured 3.7e-8
FMO7 = os.path.join(ROOT, "tests", "configs", "benchmark_fluorescence_fmo7.yaml")
G6_TOL = 0.015                                          # K=2 L=3 +0.71 %, K=1 L=3 −3.1 %

fp64_only = pytest.mark.skipif(os.path.basename(SOLVER).endswith("_mp"),
                               reason="FP64 gate")


def _run(task, cfg_path, tmp_path, L=None, K=None):
    cfg = yaml.safe_load(open(cfg_path))
    tag = f"g_{task}" + (f"_L{L}" if L is not None else "") + (f"_K{K}" if K is not None else "")
    if L is not None:
        cfg["bath"]["L_cutoff"] = L
    if K is not None:
        cfg["bath"]["n_extra_poles"] = K
    cfg[task]["output"] = {"save_path": str(tmp_path) + "/", "filename": tag,
                           "use_timestamp": False}
    y = tmp_path / f"{tag}.yaml"
    y.write_text(yaml.safe_dump(cfg))
    r = subprocess.run([SOLVER, task, str(y)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{task} failed:\n{r.stderr[-1500:]}"
    return str(tmp_path / f"{tag}.h5")


def _cumulant(h5, task):
    with h5py.File(h5, "r") as f:
        args = (f["/derived/bath_c_cm2"][...], f["/derived/bath_cbar_cm2"][...],
                f["/derived/bath_nu_cm"][...], float(f["/derived/markov_residue_cm"][0]),
                float(f["/parameters/hamiltonian_cm"][1, 1]), f["/results/time"][...])
        lam = float(f["/parameters/reorg_cm"][0])
    C_A, C_F = monomer_correlations(*args, cm_fs=CM_FS)
    return (C_A if task == "absorption" else C_F), lam


def _x0(h5):
    """Normalised (e,g) coherence ρ_eg(t)/ρ_eg(0) of the x direction."""
    with h5py.File(h5, "r") as f:
        r = f["/results/rho_xyz"][0, :, 1, 0]
        t = f["/results/time"][...]
    return t, r / r[0]


@fp64_only
@pytest.mark.parametrize("task", ["absorption", "fluorescence"])
def test_g4a_monomer_matches_numpy_hierarchy(task, tmp_path):
    h5 = _run(task, MONO_VIB, tmp_path)
    t, heom = _x0(h5)
    t_ref, ref = monomer_x0(h5, task)
    np.testing.assert_array_equal(t, t_ref)
    err = float(np.max(np.abs(heom - ref)))
    print(f"G4a {task}: max |X0_heom − X0_numpy| = {err:.3e} over {t[-1]:.0f} fs")
    assert err < G4A_TOL
    assert float(np.min(np.abs(ref))) > 1e-3       # the signal is still there to compare


@pytest.fixture(scope="module")
def drude_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("g4b")
    return {(task, L): _run(task, MONO, tmp, L)
            for task in ("absorption", "fluorescence") for L in G4B_L}


@fp64_only
@pytest.mark.parametrize("task", ["absorption", "fluorescence"])
def test_g4b_cumulant_error_falls_with_L(task, drude_runs):
    errs = []
    for L in G4B_L:
        h5 = drude_runs[task, L]
        _, heom = _x0(h5)
        ref, _ = _cumulant(h5, task)
        errs.append(float(np.max(np.abs(heom - ref))))
    print(f"G4b {task}: " + "  ".join(f"L={L} {e:.3e}" for L, e in zip(G4B_L, errs)))
    assert all(b < 0.5 * a for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < G4B_TOL[task]


@fp64_only
def test_g5_stokes_shift_matches_closed_form(drude_runs):
    shift, shift_ref = 0.0, 0.0
    for task, sign in (("absorption", 1.0), ("fluorescence", -1.0)):
        h5 = drude_runs[task, G4B_L[1]]
        t, heom = _x0(h5)
        ref, lam = _cumulant(h5, task)
        h = (t[1] - t[0]) * CM_FS
        shift += sign * -np.angle(heom[1]) / h
        shift_ref += sign * -np.angle(ref[1]) / h
    rel = abs(shift - shift_ref) / (2 * lam)
    print(f"G5: ω_A − ω_F = {shift:.6f} (closed form {shift_ref:.6f}, 2λ {2 * lam:.4f}), "
          f"rel {rel:.2e}")
    assert rel < G5_TOL


def _t_eff_rel(tmp_path, K):
    from heom_tools.absorption.post.core import compute_absorption_post
    from heom_tools.common.spectrum import load_spectrum_h5
    from heom_tools.fluorescence.post.core import compute_fluorescence_post
    from heom_tools.fluorescence.post.io import resolve_frequency_window
    from heom_tools.fluorescence.post.kms import fit_kms

    fl = load_spectrum_h5(_run("fluorescence", FMO7, tmp_path, K=K), task="fluorescence")
    ab = load_spectrum_h5(_run("absorption", FMO7, tmp_path, K=K), task="absorption")
    lo, hi = resolve_frequency_window(None, None, fl)
    f = compute_fluorescence_post(fl, lo, hi).linear
    a = compute_absorption_post(ab, lo, hi)
    k = fit_kms(f.omega_cm, f.intensity_total, a.omega_cm, a.intensity_total, fl.temperature_K)
    return (k.T_eff_K - k.T_K) / k.T_K


@fp64_only
def test_g6_fmo7_kms_temperature(tmp_path):
    rel2, rel1 = _t_eff_rel(tmp_path, 2), _t_eff_rel(tmp_path, 1)
    print(f"G6 FMO-7 77 K, L=3: (T_eff − T)/T = {rel2:+.3e} (K=2), {rel1:+.3e} (K=1)")
    assert abs(rel2) < G6_TOL
    assert abs(rel2) < 0.5 * abs(rel1)
