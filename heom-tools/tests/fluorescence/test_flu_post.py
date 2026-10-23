"""flu-post core, HDF5 layout and CLI on synthetic solver output."""
import os
import warnings
from dataclasses import replace

import h5py
import numpy as np
import pytest

from heom_tools.absorption.post.core import compute_absorption_post
from heom_tools.fluorescence.post.kms import fit_kms
from heom_tools.common.spectrum import (
    DW_DEFAULT_CM,
    _fft_grid_length,
    build_stick_spectrum,
    default_absorption_frequency_window_cm,
    linear_h5_skip_reason,
    load_spectrum_h5,
)
from heom_tools.fluorescence.plot.plot import EmissionView, write_emission_plot
from heom_tools.fluorescence.post.cli import main as fl_main
from heom_tools.fluorescence.post.core import (
    compute_fluorescence_post,
    compute_populations,
    default_fluorescence_frequency_window_cm,
)
from tests.fluorescence._synth import (
    H_EFF,
    KB_CM,
    REORG,
    T_K,
    boltzmann,
    dimer_eigen,
    write_raw,
)

POPS = np.array([0.9, 0.1])     # deliberately not Boltzmann


@pytest.fixture
def fl_spec(tmp_path):
    path = str(tmp_path / "dimer_fluorescence.h5")
    write_raw(path, "fluorescence", n_outputs=400, pops=POPS)
    return load_spectrum_h5(path, task="fluorescence")


def test_loader_reads_fluorescence_fields(fl_spec):
    assert fl_spec.spectrum_type == "fluorescence"
    np.testing.assert_array_equal(fl_spec.reorg_cm, REORG)
    assert fl_spec.rho_eq.shape == (3, 3)
    assert fl_spec.equilibration.t_eq_fs == 100.0
    assert fl_spec.markov_residue_cm.shape == (2,)


def test_task_guards(tmp_path, fl_spec):
    with pytest.raises(ValueError, match="need 'absorption'"):
        load_spectrum_h5(fl_spec.filename, task="absorption")
    assert "task='fluorescence'" in linear_h5_skip_reason(fl_spec.filename, "absorption")
    assert linear_h5_skip_reason(fl_spec.filename, "fluorescence") is None


def test_populations_three_sources(fl_spec):
    pops = compute_populations(fl_spec)
    E = dimer_eigen()[0][1:]
    np.testing.assert_allclose(pops.heom, POPS, atol=1e-14)
    np.testing.assert_allclose(pops.boltz_heff, boltzmann(E), atol=1e-14)
    w, V = np.linalg.eigh(H_EFF - np.diag(REORG))
    p = np.exp(-(w - w.min()) / (KB_CM * T_K))
    rho_s = (V * (p / p.sum())) @ V.T
    U = dimer_eigen()[1][1:, 1:]
    np.testing.assert_allclose(pops.boltz_hs, np.diag(U.T @ rho_s @ U), atol=1e-14)
    for x in (pops.heom, pops.boltz_heff, pops.boltz_hs):
        assert x.sum() == pytest.approx(1.0, abs=1e-14)


def test_sticks_are_population_weighted(fl_spec):
    lo, hi = default_fluorescence_frequency_window_cm(fl_spec)
    res = compute_fluorescence_post(fl_spec, lo, hi)
    f = build_stick_spectrum(fl_spec.eigenvalues_absolute_cm, fl_spec.eigenvectors,
                             fl_spec.dipoles, fl_spec.strengths)
    for st, a, p in zip(res.linear.sticks, f, POPS):
        assert st.oscillator_strength == pytest.approx(p * a.oscillator_strength, rel=1e-14)
        assert st.omega_cm == a.omega_cm
    assert res.populations.stick_weights == "heom"

    bare = replace(fl_spec, rho_eq=None)
    res_b = compute_fluorescence_post(bare, lo, hi)
    assert res_b.populations.stick_weights == "boltzmann_heff"
    np.testing.assert_array_equal(res_b.populations.weights, res_b.populations.boltz_heff)


def test_linear_part_is_the_absorption_code(fl_spec):
    lo, hi = default_fluorescence_frequency_window_cm(fl_spec)
    fl = compute_fluorescence_post(fl_spec, lo, hi).linear
    ab = compute_absorption_post(fl_spec, lo, hi)
    np.testing.assert_array_equal(fl.intensity_total, ab.intensity_total)
    np.testing.assert_array_equal(fl.ct_iso_per_exciton, ab.ct_iso_per_exciton)
    assert fl.consistency_max_error < 1e-12


def test_window_extends_red_edge_by_two_lambda(fl_spec):
    lo_a, hi_a = default_absorption_frequency_window_cm(fl_spec.eigenvalues_absolute_cm,
                                                        eigenvectors=fl_spec.eigenvectors)
    assert default_fluorescence_frequency_window_cm(fl_spec) == (lo_a - 70.0, hi_a)


def test_omega3_factor(fl_spec):
    lo, hi = default_fluorescence_frequency_window_cm(fl_spec)
    res = compute_fluorescence_post(fl_spec, lo, hi, omega3=True)
    lin = res.linear
    k = int(np.argmax(lin.intensity_total))
    assert res.omega_ref_cm == lin.omega_cm[k]
    assert res.omega3[k] == lin.intensity_total[k]
    np.testing.assert_allclose(res.omega3,
                               (lin.omega_cm / res.omega_ref_cm) ** 3 * lin.intensity_total,
                               rtol=1e-15)


def test_missing_reorg_is_an_error(fl_spec):
    with pytest.raises(ValueError, match="reorg_cm"):
        compute_populations(replace(fl_spec, reorg_cm=None))


def _pair(tmp_path, **kw):
    fl, ab = str(tmp_path / "dimer_fluorescence.h5"), str(tmp_path / "dimer_absorption.h5")
    write_raw(fl, "fluorescence", **kw)
    write_raw(ab, "absorption")
    return fl, ab


def test_cli_end_to_end_kms_recovers_temperature(tmp_path, capsys):
    fl, ab = _pair(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fl_main([fl, "--absorption", ab, "--omega3", "--no-plots"]) == 0
    out = str(tmp_path / "dimer_fluorescence_spectrum.h5")
    with h5py.File(out, "r") as f:
        assert f.attrs["spectrum_type"] == "fluorescence"
        assert f.attrs["entry_point"] == "flu-post"
        assert f["sticks"].attrs["stick_weights"] == "heom"
        for k in ("heom", "boltzmann_heff", "boltzmann_hs"):
            assert f[f"populations/{k}"].shape == (2,)
        assert "intensity_omega3" in f and "source/rho_eq" in f
        assert abs(f["kms"].attrs["T_eff_K"] / T_K - 1.0) < 1e-10
    assert "[kms] T_eff" in capsys.readouterr().out

    view = EmissionView.from_processed_h5(out)
    assert view.kms is not None and view.stick_weights == "heom"
    png = write_emission_plot(view, str(tmp_path / "fl.png"))
    assert os.path.getsize(png) > 0


def test_kms_needs_half_weight_at_t0(tmp_path):
    fl, ab = _pair(tmp_path)
    F = load_spectrum_h5(fl, task="fluorescence")
    lo, hi = default_fluorescence_frequency_window_cm(F)
    f = compute_fluorescence_post(F, lo, hi).linear
    a = compute_absorption_post(load_spectrum_h5(ab, task="absorption"), lo, hi)
    exact = fit_kms(f.omega_cm, f.intensity_total, a.omega_cm, a.intensity_total, T_K)

    def full_weight(r):   # the pre-2.1.1 (Fortran) rectangle: + Re C(0)/(2·nfft)
        nfft = _fft_grid_length(r.spec.n_outputs, r.spec.dt_fs, DW_DEFAULT_CM)
        return r.intensity_total + r.ct_iso_total[0].real / (2.0 * nfft)

    biased = fit_kms(f.omega_cm, full_weight(f), a.omega_cm, full_weight(a), T_K)
    assert exact.rel_error < 1e-10
    assert biased.rel_error > 1e3 * max(exact.rel_error, 1e-12)


def test_cli_rejects_mismatched_physics(tmp_path):
    fl, ab = _pair(tmp_path)
    with h5py.File(ab, "a") as f:
        del f["parameters/config_yaml"]
        f["parameters"].create_dataset("config_yaml", data="system:\n  n_states: 3\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fl_main([fl, "--absorption", ab, "--no-plots"]) == 1


def test_auto_mode_processes_only_fluorescence(tmp_path, monkeypatch):
    _pair(tmp_path, n_outputs=200)
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert fl_main(["--no-plots"]) == 0
    assert os.path.isfile(tmp_path / "dimer_fluorescence_spectrum.h5")
    assert not os.path.exists(tmp_path / "dimer_absorption_spectrum.h5")
    assert any("task='absorption'" in str(x.message) for x in w)
