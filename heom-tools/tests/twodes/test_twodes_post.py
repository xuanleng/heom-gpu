"""Unit tests for heom_tools.twodes.post/ (FFT signs, window cut, polarization)."""
import json
import os
import numpy as np
import pytest

from heom_tools.twodes.post import (
    DEFAULTS,
    FS_TO_CM,
    PostConfig,
    TwSlice,
    check_skip_outputs,
    compute_tw_slice,
    cut_window,
    freq_axis_cm,
    from_h5_complex,
    load_config,
    load_meta,
    make_norm,
    nfft_len,
    polarization_weights,
    time_domain_combo,
    two_d_fft,
    print_runtime_config,
    write_post_h5,
)
from heom_tools.twodes.post.fft import (
    cross_check_two_d_fft,
    two_d_fft_axis,
    two_d_fft_separable,
)
from heom_tools.twodes.post.fft_legacy import two_d_fft_fft2, two_d_fft_legacy_mode3
from heom_tools.twodes.post.cache import (
    fingerprint_diff,
    resolve_output_ctx,
    resolve_process_needs,
    tw_indices_needing_plots,
    write_sidecar,
)
from heom_tools.twodes.post.config import print_fft_grid_info
from heom_tools.twodes.post.core import (
    combined_freq_rows,
    combined_time_rows,
    fft_grid,
    pathway_plan,
    pathways_complete,
    require_fft_time_axes,
    spec_2des_total,
)
from heom_tools.twodes.post.io import to_h5_complex, is_solver_2des_h5
from heom_tools.twodes.post.h5_spectra import SpectraComputeNeeds
from heom_tools.twodes.post.panels import prune_orphan_plots
from heom_tools.twodes.post.types import OutputPlan


def _slice_needs(
    pathways,
    signals,
    *,
    compute_2des=True,
    pathway_fft=None,
    total_fft=None,
):
    if pathway_fft is None:
        pathway_fft = frozenset((pi, s) for pi in range(len(pathways)) for s in signals)
    if total_fft is None:
        total_fft = frozenset(signals)
    return SpectraComputeNeeds(
        pathway_fft=pathway_fft,
        total_fft=total_fft,
        compute_2des=compute_2des,
    )


def test_combined_freq_row_order_all_pathways():
    z = np.zeros((4, 4), dtype=complex)
    spec_p = {p: {"rephasing": z, "nonrephasing": z} for p in ("SE", "GSB", "ESA")}
    spec_t = {"rephasing": z, "nonrephasing": z}
    labels = [r[0] for r in combined_freq_rows(
        spec_p, spec_t, z, ["SE", "GSB", "ESA"])]
    assert labels == [
        "2des",
        "rephasing", "rephasing GSB", "rephasing SE", "rephasing ESA",
        "nonrephasing", "nonrephasing GSB", "nonrephasing SE", "nonrephasing ESA",
    ]


def test_combined_freq_row_order_single_pathway():
    z = np.zeros((4, 4), dtype=complex)
    spec_p = {"GSB": {"rephasing": z, "nonrephasing": z}}
    spec_t = {"rephasing": z, "nonrephasing": z}
    labels = [r[0] for r in combined_freq_rows(
        spec_p, spec_t, z, ["GSB"])]
    assert labels == ["2des", "rephasing", "rephasing GSB", "nonrephasing", "nonrephasing GSB"]


def test_combined_time_row_order():
    z = np.zeros((4, 4), dtype=complex)
    time_p = {p: {"rephasing": z, "nonrephasing": z} for p in ("SE", "GSB", "ESA")}
    time_t = {"rephasing": z, "nonrephasing": z}
    labels = [r[0] for r in combined_time_rows(
        time_p, time_t, True, ["SE", "GSB", "ESA"])]
    assert labels == ["2des", "GSB", "SE", "ESA"]


def test_combined_time_row_single_pathway():
    z = np.zeros((4, 4), dtype=complex)
    time_p = {"GSB": {"rephasing": z, "nonrephasing": z}}
    time_t = {"rephasing": z, "nonrephasing": z}
    labels = [r[0] for r in combined_time_rows(
        time_p, time_t, True, ["GSB"])]
    assert labels == ["2des", "GSB"]


def test_pathway_plan_complete():
    plan = pathway_plan(["SE", "GSB", "ESA"])
    assert plan.pidx == [0, 1, 2]
    assert plan.pathway_sum == "all"


def test_pathway_plan_incomplete():
    plan = pathway_plan(["SE", "GSB"])
    assert plan.pathway_sum == "GSB,SE"


def test_pathways_complete():
    assert pathways_complete(["SE", "GSB", "ESA"]) is True
    assert pathways_complete(["SE"]) is False


def test_is_solver_2des_h5(tmp_path):
    raw = tmp_path / "sim.h5"
    _make_solver_h5(raw)
    assert is_solver_2des_h5(str(raw)) is True
    (tmp_path / "junk.h5").write_bytes(b"not hdf5")
    assert is_solver_2des_h5(str(tmp_path / "junk.h5")) is False


def test_process_rejects_non_solver_h5(tmp_path):
    import heom_tools.twodes.post as tp

    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not hdf5")
    cfg = PostConfig.defaults()
    assert tp.process(str(bad), cfg, show_config=False) == 1


def test_sidecar_fingerprint_includes_pathways(tmp_path):
    raw = tmp_path / "raw.h5"
    _make_solver_h5(raw, pathways=["SE", "GSB"])
    cfg = PostConfig.defaults()
    from heom_tools.twodes.post.cache import cfg_fingerprint
    fp = cfg_fingerprint(cfg, str(raw))
    assert fp["pathways"] == "SE,GSB"


def test_compute_tw_slice_skips_time_when_not_needed():
    S = np.zeros((2, 3, 4, 5), dtype=complex)
    signals = ["rephasing", "nonrephasing"]
    pathways = ["SE", "GSB", "ESA"]
    needs = _slice_needs(pathways, signals)
    sl = compute_tw_slice(
        S, signals, pathways, list(range(3)), 0, 8, 8,
        needs=needs,
        need_time=False, need_time_path=False,
    )
    assert sl.spec_path
    assert sl.time_path is None
    assert sl.time_total is None


def test_compute_tw_slice_skips_spec_when_not_needed():
    S = np.zeros((2, 3, 4, 5), dtype=complex)
    signals = ["rephasing", "nonrephasing"]
    pathways = ["SE", "GSB", "ESA"]
    needs = _slice_needs(
        pathways, signals, compute_2des=False,
        pathway_fft=frozenset(), total_fft=frozenset(),
    )
    sl = compute_tw_slice(
        S, signals, pathways, list(range(3)), 0, 8, 8,
        needs=needs,
        need_time=True, need_time_path=True,
    )
    assert not sl.spec_path
    assert sl.time_path is not None


def test_spec_2des_total_explicit_sum():
    z = np.ones((4, 4), dtype=complex)
    w = np.full((4, 4), 2.0, dtype=complex)
    out = spec_2des_total({"rephasing": z, "nonrephasing": w})
    assert np.allclose(out, z + w)


def test_spec_2des_total_missing_signal_raises():
    with pytest.raises(ValueError, match="rephasing and nonrephasing"):
        spec_2des_total({"rephasing": np.zeros((2, 2))})


def testcombined_time_rows_single_pathway_no_time_path():
    """pathway≠all: 2des row from time_total only (time_path empty)."""
    z = np.zeros((4, 4), dtype=complex)
    time_t = {"rephasing": z, "nonrephasing": z}
    labels = [r[0] for r in combined_time_rows(
        {}, time_t, True, ["GSB"])]
    assert labels == ["2des"]


def test_check_skip_outputs_force_disables_skip(tmp_path):
    import heom_tools.twodes.post as tp

    cfg = PostConfig.defaults()
    cfg.force = True
    cfg.plot = True
    skip, _msg, stale, _ctx = check_skip_outputs("missing.h5", cfg)
    assert skip is False
    assert stale is False


def test_compute_tw_slice_plot_complete_pathways_includes_path_fft():
    """GSB+SE+ESA complete: total + per-pathway spectra for component plots."""
    S = np.zeros((2, 3, 4, 5), dtype=complex)
    signals = ["rephasing", "nonrephasing"]
    pathways = ["SE", "GSB", "ESA"]
    needs = _slice_needs(pathways, signals)
    sl = compute_tw_slice(
        S, signals, pathways, list(range(3)), 0, 8, 8,
        needs=needs,
        need_time=True, need_time_path=True,
    )
    assert sl.spec_path
    assert sl.time_path is not None
    assert sl.time_total is not None


def test_process_skips_before_load(monkeypatch, tmp_path):
    import heom_tools.twodes.post as tp
    from heom_tools.twodes.post.cache import empty_output_ctx
    from heom_tools.twodes.post.types import PrecheckResult

    h5 = tmp_path / "tiny.h5"
    import h5py

    n_sig, n_path, n_tau, n_tw, n_t = 2, 3, 4, 1, 5
    with h5py.File(h5, "w") as f:
        dt = np.dtype([("r", "f8"), ("i", "f8")])
        resp = np.zeros((3, n_sig, n_path, n_tau, n_tw, n_t), dtype=dt)
        f.create_group("results").create_dataset("resp", data=resp)
        res = f["results"]
        res.create_dataset("tau_fs", data=np.linspace(0, 12, n_tau))
        res.create_dataset("tw_fs", data=[0.0])
        res.create_dataset("t_fs", data=np.linspace(0, 20, n_t))
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = "SE,GSB,ESA"

    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = False
    cfg.write_h5 = False

    def boom(*_a, **_kw):
        raise AssertionError("load_axes must not run when skipping")

    monkeypatch.setattr(tp, "load_axes", boom)
    monkeypatch.setattr(
        tp,
        "precheck",
        lambda _h, _c: PrecheckResult(
            True, "outputs exist (verified via sidecar)", False, empty_output_ctx(str(h5), cfg),
        ),
    )
    assert tp.process(str(h5), cfg) == 0


def test_time_domain_combo_empty_raises():
    with pytest.raises(ValueError, match="at least one signal"):
        time_domain_combo({}, np.linspace(0, 1, 4))


def test_write_h5_defaults_false():
    assert PostConfig.defaults().write_h5 is False


def test_polarization_ap_weights():
    w = polarization_weights("ap")
    assert w.shape == (3,)
    assert np.allclose(w, np.full(3, 2.0 / 30.0))


def _solver_invariants(R):
    """(I1,I2,I3) via the bucket rule of heom-gpu/src/spectra/two_dim.cpp:190-192,
    with m[0..3] = 1st/2nd/3rd interaction and detection (two_dim.cpp:902-905)."""
    import itertools

    I = np.zeros(3)
    for m in itertools.product(range(3), repeat=4):
        m1, m2, m3, m4 = m
        for ip, hit in enumerate(((m1 == m2 and m3 == m4),
                                  (m1 == m3 and m2 == m4),
                                  (m1 == m4 and m2 == m3))):
            if hit:
                I[ip] += R[m]
    return I


def _planar(angles_deg):
    t = np.radians(np.asarray(angles_deg, dtype=float))
    return np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)


@pytest.mark.parametrize("angles", [
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 90.0, 90.0),
    (45.0, -45.0, 90.0, 0.0),
    (13.0, 71.0, -29.0, 50.0),
])
def test_polarization_pairing_matches_exact_isotropic_average(angles):
    """The three delta-pairing probes have an exact lab-frame average --
    a rotation preserves inner products, so <P_k^Lab> = u_k with no averaging
    formula involved. Their invariant triples (9,3,3)/(3,9,3)/(3,3,9) are
    independent, so matching all three pins the whole u <-> I pairing. This is
    the test that catches an I2 <-> I3 transposition, which flips DC and leaves
    AP untouched (paper checklist A1)."""
    d = np.eye(3)
    probes = {
        (0, 1, 2, 3): np.einsum("ab,cd->abcd", d, d),   # u1 = (e1.e2)(e3.e4)
        (0, 2, 1, 3): np.einsum("ac,bd->abcd", d, d),   # u2 = (e1.e3)(e2.e4)
        (0, 3, 1, 2): np.einsum("ad,bc->abcd", d, d),   # u3 = (e1.e4)(e2.e3)
    }
    e = _planar(angles)
    w = polarization_weights(tuple(angles))
    for (i, j, k, l), R in probes.items():
        got = float(w @ _solver_invariants(R))
        want = float((e[i] @ e[j]) * (e[k] @ e[l]))
        assert got == pytest.approx(want, abs=1e-13)


def test_double_crossed_weights_are_half_difference_of_i2_i3():
    """DC contracts to (I2-I3)/12 exactly; the leading weight must vanish."""
    assert np.allclose(polarization_weights("double_crossed"),
                       [0.0, 1.0 / 12.0, -1.0 / 12.0], atol=1e-15)


def test_i2_i3_transposition_flips_dc_and_spares_ap():
    """Why the DC sign is the only observable sensitive to the pairing."""
    I = np.array([1.0, 0.35, -0.20])
    swap = I[[0, 2, 1]]
    w_ap = polarization_weights("ap")
    w_dc = polarization_weights("double_crossed")
    assert float(w_ap @ swap) == pytest.approx(float(w_ap @ I), abs=1e-15)
    assert float(w_dc @ swap) == pytest.approx(-float(w_dc @ I), abs=1e-15)


def test_reversed_pump_order_negates_double_crossed():
    """<-45,45,90,0> is the same physics with the opposite overall sign, which
    is the documented origin of the -1 against the published DC maps."""
    assert np.allclose(polarization_weights((-45.0, 45.0, 90.0, 0.0)),
                       -polarization_weights("double_crossed"), atol=1e-15)


def test_monomer_has_no_double_crossed_signal():
    """A single transition dipole gives R = mu mu mu mu, hence I1 = I2 = I3 and
    a strictly vanishing DC signal, while AP reduces to |mu|^4/5."""
    mu = np.array([0.967169, 1.6009, 0.218504])
    R = np.einsum("a,b,c,d->abcd", mu, mu, mu, mu)
    I = _solver_invariants(R)
    assert np.ptp(I) == pytest.approx(0.0, abs=1e-12)
    assert float(polarization_weights("double_crossed") @ I) == pytest.approx(0.0, abs=1e-13)
    assert float(polarization_weights("ap") @ I) == pytest.approx(float(mu @ mu) ** 2 / 5.0)


def test_two_d_fft_rephasing_tau_sign():
    """τ-axis fft (rp) vs ifft (nr) differ off τ=0."""
    n_tau, n_t = 32, 32
    S = np.zeros((n_tau, n_t), dtype=complex)
    S[1, 0] = 1.0
    rp = two_d_fft(S, "rephasing", n_tau, n_t)
    nr = two_d_fft(S, "nonrephasing", n_tau, n_t)
    assert not np.allclose(rp, nr)


def test_two_d_fft_nr_rp_same_impulse_scale():
    """nr has an extra τ ifft (÷n_tau); code multiplies nr by n_tau to match rp."""
    n_tau, n_t = 64, 64
    S = np.zeros((n_tau, n_t), dtype=complex)
    S[0, 0] = 1.0
    rp = two_d_fft(S, "rephasing", n_tau, n_t)
    nr = two_d_fft(S, "nonrephasing", n_tau, n_t)
    assert np.abs(rp).max() == pytest.approx(np.abs(nr).max(), rel=1e-12)


def test_fft_dtau_dt_weight_absolute_amplitude():
    """Same continuous signal on two grids: unweighted peaks scale with Δτ·Δt;
    weighted peaks agree (L6 / IF-report absolute-amplitude fix)."""
    from heom_tools.twodes.post.fft import fft_time_weight

    def sample(dtau, dt, n=48):
        tau = np.arange(n) * dtau
        t = np.arange(n) * dt
        # Smooth decaying oscillation — continuous f(τ,t).
        TT, tt = np.meshgrid(tau, t, indexing="ij")
        return np.exp(-(TT + tt) / 80.0) * np.exp(1j * 0.05 * (TT + tt))

    d1, d2 = 8.0, 12.0
    S1, S2 = sample(d1, d1), sample(d2, d2)
    n_pad = 128  # shared pad so ifft 1/n factor matches
    # Unweighted (default dtau=dt=1): peaks differ by the measure ratio.
    p1 = np.abs(two_d_fft(S1, "rephasing", n_pad, n_pad)).max()
    p2 = np.abs(two_d_fft(S2, "rephasing", n_pad, n_pad)).max()
    # Weighted: absolute amplitudes comparable across grids.
    w1 = np.abs(two_d_fft(S1, "rephasing", n_pad, n_pad, dtau=d1, dt=d1)).max()
    w2 = np.abs(two_d_fft(S2, "rephasing", n_pad, n_pad, dtau=d2, dt=d2)).max()
    assert w1 == pytest.approx(p1 * fft_time_weight(d1, d1), rel=1e-12)
    assert w2 == pytest.approx(p2 * fft_time_weight(d2, d2), rel=1e-12)
    # Without weight the denser grid systematically differs; with weight the
    # residual is sampling/Nyquist, not a global Δτ·Δt scale factor.
    rel_raw = abs(p1 - p2) / max(p1, p2)
    rel_w = abs(w1 - w2) / max(w1, w2)
    assert rel_raw > 0.15
    assert rel_w < 0.10
    assert rel_w < 0.5 * rel_raw


def test_axis_matches_separable_square():
    rng = np.random.default_rng(11)
    S = rng.standard_normal((12, 12)) + 1j * rng.standard_normal((12, 12))
    for sig in ("rephasing", "nonrephasing"):
        a = two_d_fft_axis(S, sig, 32, 32)
        b = two_d_fft_separable(S, sig, 32, 32)
        assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_axis_matches_separable_rectangular():
    rng = np.random.default_rng(12)
    S = rng.standard_normal((10, 14)) + 1j * rng.standard_normal((10, 14))
    for sig in ("rephasing", "nonrephasing"):
        a = two_d_fft_axis(S, sig, 40, 32)
        b = two_d_fft_separable(S, sig, 40, 32)
        assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_cross_check_passes_random():
    rng = np.random.default_rng(13)
    S = rng.standard_normal((8, 9)) + 1j * rng.standard_normal((8, 9))
    for sig in ("rephasing", "nonrephasing"):
        spec, stats = cross_check_two_d_fft(S, sig, 36, 28)
        assert stats["axis_vs_separable"] < 1e-12
        assert np.allclose(spec, two_d_fft(S, sig, 36, 28))


def test_two_d_fft_trapezoid_endpoint_weights():
    """τ=0 row and t=0 column at ½, corner ¼, interior 1; the input is untouched."""
    from heom_tools.twodes.post.fft import trapezoid_endpoints

    for (i, j), w in (((0, 0), 0.25), ((0, 3), 0.5), ((2, 0), 0.5), ((2, 3), 1.0)):
        S = np.zeros((6, 6), dtype=complex)
        S[i, j] = 1.0
        for sig in ("rephasing", "nonrephasing"):
            np.testing.assert_allclose(two_d_fft(S, sig, 16, 16),
                                       w * two_d_fft_axis(S, sig, 16, 16), rtol=0, atol=1e-15)
        assert S[i, j] == 1.0
    S = np.ones((3, 3))
    np.testing.assert_array_equal(trapezoid_endpoints(S).real,
                                  [[0.25, 0.5, 0.5], [0.5, 1, 1], [0.5, 1, 1]])


def test_2des_total_matches_continuum_lorentzian():
    """One Lorentzian peak, rp + nr on the production 8 fs grid against the
    closed-form absorptive map L(ω_τ)·L(ω_t). Trapezoid: O(Δ²) (6e-4 measured);
    full weight at τ=0 / t=0 leaves stripes Γ·Δ/4 ≈ 4 % (3.7e-2 measured)."""
    from heom_tools.twodes.post.fft import freq_axis_cm, nfft_len

    fwhm, d = 100.0, 8.0
    g = fwhm * 2 * np.pi * 2.99792458e-5
    x = np.arange(0.0, 1200.0 + d / 2, d)
    S = np.exp(-g * x[:, None] / 2) * np.exp(-g * x[None, :] / 2)
    n = nfft_len(len(x), d, 10.0)
    w = freq_axis_cm(n, d, 0.0)
    i0 = int(np.argmin(np.abs(w)))
    win = np.abs(w) < 1500.0
    h2 = (fwhm / 2) ** 2
    lor = h2 / (h2 + w[win] ** 2)
    exact = np.outer(lor, lor)

    def err(fft):
        spec = sum(fft(S, s, n, n, dtau=d, dt=d) for s in ("rephasing", "nonrephasing")).real
        return np.max(np.abs(spec[np.ix_(win, win)] / spec[i0, i0] - exact))

    assert err(two_d_fft) < 2e-3
    assert err(two_d_fft_axis) > 2e-2


def test_two_d_fft_legacy_mode3_runs():
    S = np.zeros((4, 4), dtype=complex)
    S[1, 2] = 1.0
    out = two_d_fft_legacy_mode3(S, "rephasing", 16, 16)
    assert out.shape == (16, 16)
    assert np.allclose(out, two_d_fft_fft2(S, "rephasing", 16, 16))


def test_two_d_fft_dispatch_legacy_mode3():
    S = np.zeros((4, 4), dtype=complex)
    S[2, 1] = 1.0
    assert np.allclose(
        two_d_fft(S, "nonrephasing", 16, 16, method="legacy_mode3"),
        two_d_fft_legacy_mode3(S, "nonrephasing", 16, 16),
    )


def test_print_runtime_config_lists_all_keys(capsys, tmp_path):
    cfg = PostConfig.defaults()
    cfg.fft_method = "check"
    cfg_path = tmp_path / "twodes_post.yaml"
    print_runtime_config(
        cfg, config_path=str(cfg_path), input_h5=str(tmp_path / "in.h5"),
    )
    out = capsys.readouterr().out
    assert "twodes-post runtime config" in out
    assert str(cfg_path) in out
    assert "in.h5" in out
    for key in DEFAULTS:
        assert key in out
    assert "→ fft active" in out
    assert "check" in out
    assert "diagnostic" in out


def test_print_fft_grid_info_padded(capsys):
    print_fft_grid_info(
        dtau=8.0,
        dt=8.0,
        n_tau_data=151,
        n_tau_fft=417,
        n_t_data=151,
        n_t_fft=417,
        dw=10.0,
        w0=12300.0,
        n_w_tau=333,
        n_w_t=333,
    )
    out = capsys.readouterr().out
    assert "τ n=151→417 (dtau=8 fs, +266 pad)" in out
    assert "t n=151→417 (dt=8 fs, +266 pad)" in out
    assert "dw=10 cm^-1  w0=12300" in out
    assert "ω lattice   : nτ=333 nt=333  output Δω=10 cm^-1 (w0 + k·dw)" in out
    assert "ω resolve   :" in out
    assert "Δω_phys(τ)≈27.6 cm^-1" in out
    assert "Δω_phys(t)≈27.6 cm^-1" in out
    assert "pad densifies, does not resolve" in out


def test_print_fft_grid_info_no_pad(capsys):
    print_fft_grid_info(
        dtau=8.0,
        dt=8.0,
        n_tau_data=417,
        n_tau_fft=417,
        n_t_data=417,
        n_t_fft=417,
        dw=10.0,
        w0=12300.0,
    )
    out = capsys.readouterr().out
    assert "τ n=417 (dtau=8 fs)" in out
    assert "t n=417 (dt=8 fs)" in out
    assert "→" not in out
    assert "pad" not in out.split("ω resolve")[0]  # no pad on FFT grid line
    assert "ω lattice" not in out
    assert "Δω_phys(τ)≈10 cm^-1" in out
    assert "pad densifies, does not resolve" in out


def test_fft_grid_prints_data_and_fft_lengths(capsys):
    tau = np.arange(151, dtype=float) * 8.0
    t = np.arange(151, dtype=float) * 8.0
    n_tau, n_t, w_tau, w_t, _, _, w_nat_tau, w_nat_t = fft_grid(
        tau, t, dw=10.0, w0=12300.0,
    )
    assert n_tau == 417
    assert n_t == 417
    assert np.allclose(np.diff(w_tau), 10.0)
    assert np.allclose(np.diff(w_t), 10.0)
    assert 12300.0 in w_tau
    assert w_tau[0] >= w_nat_tau[0] - 1e-9
    assert w_tau[-1] <= w_nat_tau[-1] + 1e-9
    out = capsys.readouterr().out
    assert "151→417" in out
    assert "+266 pad" in out
    assert "output Δω=10" in out
    assert "Δω_phys(τ)≈27.6" in out


def test_dw_phys_cm_unpadded():
    from heom_tools.twodes.post.fft import dw_phys_cm

    # FS_TO_CM / (151 * 8) ≈ 27.613
    assert dw_phys_cm(151, 8.0) == pytest.approx(33356.40952 / (151 * 8.0))
    assert dw_phys_cm(417, 8.0) == pytest.approx(10.0, rel=1e-3)


def test_freq_axis_dw_common_across_dt():
    """Same (dw, w0) → shared ω pixels for dt=8 and dt=10 (no interp)."""
    from heom_tools.twodes.post.fft import freq_axis_dw_cm

    dw, w0 = 10.0, 12300.0
    n8 = nfft_len(64, 8.0, dw)
    n10 = nfft_len(64, 10.0, dw)
    a8 = freq_axis_dw_cm(dw, w0, n8, 8.0)
    a10 = freq_axis_dw_cm(dw, w0, n10, 10.0)
    shared = np.intersect1d(a8, a10)
    assert len(shared) > 50
    assert np.allclose(np.diff(a8), dw)
    assert np.allclose(np.diff(a10), dw)
    # every shared point is exactly w0 + k·dw
    k = np.round((shared - w0) / dw)
    assert np.allclose(shared, w0 + k * dw)


def test_project_spec_2d_exact_on_native_node():
    from heom_tools.twodes.post.fft import freq_axis_dw_cm, project_spec_2d

    dw, w0, dt = 10.0, 12300.0, 8.0
    n = nfft_len(64, dt, dw)
    src = freq_axis_cm(n, dt, w0)
    dst = freq_axis_dw_cm(dw, w0, n, dt)
    # identity peak on a native bin that lands on the dw lattice
    mid = len(src) // 2
    assert abs(src[mid] - w0) < 1e-9  # odd n → centre bin is w0
    assert w0 in dst
    spec = np.zeros((n, n), dtype=np.complex128)
    spec[mid, mid] = 3.0 + 4.0j
    out = project_spec_2d(spec, src, src, dst, dst)
    i = int(np.where(dst == w0)[0][0])
    assert out[i, i] == pytest.approx(3.0 + 4.0j)
    assert np.max(np.abs(out)) == pytest.approx(5.0)


def test_load_config_fft2_deprecated_alias_rejected(tmp_path):
    cfg_path = tmp_path / "twodes_post.yaml"
    cfg_path.write_text("fft_method: fft2\n")
    with pytest.raises(ValueError, match="not a production setting"):
        load_config(str(cfg_path))


def test_load_config_legacy_mode3_rejected(tmp_path):
    cfg_path = tmp_path / "twodes_post.yaml"
    cfg_path.write_text("fft_method: legacy_mode3\n")
    with pytest.raises(ValueError, match="not a production setting"):
        load_config(str(cfg_path))


def test_find_config_in_data_dir(tmp_path):
    from heom_tools.twodes.post.config import find_config

    cfg_path = tmp_path / "twodes_post.yaml"
    cfg_path.write_text("pol: ap\n")
    assert find_config(str(tmp_path)) == str(cfg_path)


def test_main_no_args_runs_auto(tmp_path, capsys):
    from heom_tools.twodes.post.cli import main

    cfg_path = tmp_path / "twodes_post.yaml"
    cfg_path.write_text("pol: ap\nplot: false\nwrite_h5: false\n")
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert main([]) == 0
    finally:
        os.chdir(old)
    assert "auto: no solver 2DES" in capsys.readouterr().out


def test_main_single_file_uses_config_beside_h5(tmp_path, capsys):
    from heom_tools.twodes.post.cli import main

    data = tmp_path / "data"
    data.mkdir()
    (data / "twodes_post.yaml").write_text("pol: ap\nplot: false\nwrite_h5: false\n")
    h5 = data / "sim.h5"
    _make_solver_h5(h5)
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert main([str(h5)]) == 0
    finally:
        os.chdir(old)
    out = capsys.readouterr().out
    assert str(data / "twodes_post.yaml") in out or "twodes_post.yaml" in out


def test_main_rejects_unknown_args():
    from heom_tools.twodes.post.cli import main

    with pytest.raises(SystemExit):
        main(["--not-a-flag"])


def test_tw_tag_formats_waiting_time_fs():
    from heom_tools.twodes.post.config import tw_tag

    assert tw_tag(0.0) == "0fs"
    assert tw_tag(1000.0) == "1000fs"
    assert tw_tag(500.0) == "500fs"
    assert tw_tag(12.5) == "12.5fs"


def test_polarization_perpendicular_alias():
    assert np.allclose(polarization_weights("perpendicular"), polarization_weights("perp"))


def _minimal_post_h5_attrs(par, *, source_h5=None, **overrides):
    """Minimal post H5 parameters attrs for skip/regenerate tests."""
    base = {
        "fft_method": "axis",
        "dw": 10.0,
        "w0_cm": 0.0,
        "pol": "ap",
        "pathway_sum": "all",
        "pathways": "",
        "window": "null",
        "tw_index": "all",
        "time_domain": 1,
        "h5_spectra": json.dumps(["2des"]),
        "levels": 25,
        "lines": "[]",
    }
    if source_h5 is not None:
        st = os.stat(source_h5)
        base["source_mtime"] = int(st.st_mtime_ns)
        base["source_size"] = int(st.st_size)
    base.update(overrides)
    for k, v in base.items():
        par.attrs[k] = v


def test_sidecar_fingerprint_diff_dw_mismatch(tmp_path):
    raw = tmp_path / "raw.h5"
    raw.write_bytes(b"x")
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    write_sidecar(str(tmp_path), "raw", "ap", cfg, str(raw))
    cfg.dw = 5.0
    with open(tmp_path / ".raw_ap_twodes_post.json") as f:
        stored = json.load(f)
    diff = fingerprint_diff(stored, cfg, str(raw))
    assert diff is not None
    assert "dw" in diff


def test_tw_indices_needing_plots_only_missing_png(tmp_path):
    raw = tmp_path / "raw.h5"
    _make_solver_h5(raw, n_tw=2)
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.time_domain = False
    ctx = resolve_output_ctx(str(raw), cfg)
    plan = OutputPlan.from_cfg(cfg, pathways=["SE", "GSB", "ESA"])
    (tmp_path / "raw_ap_2d_tw0fs.png").write_bytes(b"png")
    missing = tw_indices_needing_plots(
        stem=ctx.stem,
        outdir=ctx.outdir,
        ptag=ctx.ptag,
        tw=ctx.tw,
        tw_indices=ctx.tw_indices,
        plan=plan,
    )
    assert missing == [1]


def test_resolve_process_needs_incremental_tw_only(tmp_path):
    raw = tmp_path / "raw.h5"
    _make_solver_h5(raw, n_tw=2)
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = True
    cfg.write_h5 = False
    cfg.time_domain = False
    ctx = resolve_output_ctx(str(raw), cfg)
    plan = OutputPlan.from_cfg(cfg, pathways=["SE", "GSB", "ESA"])
    write_sidecar(str(tmp_path), ctx.stem, ctx.ptag, cfg, str(raw))
    (tmp_path / "raw_ap_2d_tw0fs.png").write_bytes(b"png")
    (tmp_path / "raw_ap_2d_tw500fs.png").write_bytes(b"png")
    needs = resolve_process_needs(plan, stale=False, ctx=ctx)
    assert needs.need_plot is False
    assert needs.tw_compute == []


def test_resolve_process_needs_incremental_tw_partial(tmp_path):
    raw = tmp_path / "raw.h5"
    _make_solver_h5(raw, n_tw=2)
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = True
    cfg.write_h5 = False
    cfg.time_domain = False
    ctx = resolve_output_ctx(str(raw), cfg)
    plan = OutputPlan.from_cfg(cfg, pathways=["SE", "GSB", "ESA"])
    write_sidecar(str(tmp_path), ctx.stem, ctx.ptag, cfg, str(raw))
    (tmp_path / "raw_ap_2d_tw0fs.png").write_bytes(b"png")
    needs = resolve_process_needs(plan, stale=False, ctx=ctx)
    assert needs.need_plot is True
    assert needs.tw_compute == [1]


def test_check_skip_tw_index_mismatch_regenerates(tmp_path):
    import h5py
    import heom_tools.twodes.post as tp

    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as f:
        res = f.create_group("results")
        res.create_dataset("tw_fs", data=[0.0, 500.0])
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = "SE,GSB,ESA"

    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = True
    cfg.write_h5 = True
    cfg.h5_spectra = ["2des"]
    cfg.time_domain = False
    cfg.tw_index = "all"
    post = tp.post_h5_path(str(raw), cfg)
    with h5py.File(post, "w") as f:
        _minimal_post_h5_attrs(f.create_group("parameters"), source_h5=str(raw), tw_index="all")
    (tmp_path / "raw_ap_2d_tw0fs.png").write_bytes(b"png")
    (tmp_path / "raw_ap_2d_tw500fs.png").write_bytes(b"png")
    write_sidecar(str(tmp_path), "raw", "ap", cfg, str(raw))

    cfg.tw_index = [0]
    skip, msg, stale, _ctx = check_skip_outputs(str(raw), cfg)
    assert skip is False
    assert stale is True
    assert "tw_index" in msg


def test_process_skip_prints_reason(capsys, monkeypatch, tmp_path):
    import heom_tools.twodes.post as tp

    raw = tmp_path / "raw.h5"
    _make_solver_h5(raw, n_tw=1)

    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = True
    cfg.write_h5 = False
    cfg.time_domain = False
    (tmp_path / "raw_ap_2d_tw0fs.png").write_bytes(b"png")
    write_sidecar(str(tmp_path), "raw", "ap", cfg, str(raw))

    def boom(*_a, **_kw):
        raise AssertionError("load_axes must not run when skipping")

    monkeypatch.setattr(tp, "load_axes", boom)
    tp.process(str(raw), cfg, show_config=False)
    out = capsys.readouterr().out
    assert "skip raw.h5:" in out
    assert "sidecar" in out


def test_check_skip_outputs_meta_mismatch_regenerates(tmp_path):
    import h5py
    import heom_tools.twodes.post as tp

    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as f:
        res = f.create_group("results")
        res.create_dataset("tw_fs", data=[0.0])
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = "SE,GSB,ESA"

    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = True
    cfg.write_h5 = True
    cfg.h5_spectra = ["2des"]
    cfg.time_domain = False
    post = tp.post_h5_path(str(raw), cfg)
    with h5py.File(post, "w") as f:
        _minimal_post_h5_attrs(f.create_group("parameters"), source_h5=str(raw))
    stem = "raw"
    ptag = "ap"
    png = tmp_path / f"{stem}_{ptag}_2d_tw0fs.png"
    png.write_bytes(b"png")
    write_sidecar(str(tmp_path), stem, ptag, cfg, str(raw))

    cfg.dw = 5.0
    skip, msg, stale, _ctx = check_skip_outputs(str(raw), cfg)
    assert skip is False
    assert stale is True
    assert "regenerating" in msg
    assert "dw" in msg


def test_check_skip_outputs_png_only_without_sidecar_regenerates(tmp_path):
    import h5py
    import heom_tools.twodes.post as tp

    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as f:
        res = f.create_group("results")
        res.create_dataset("tw_fs", data=[0.0])
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = "SE,GSB,ESA"

    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = True
    cfg.write_h5 = False
    cfg.time_domain = False
    png = tmp_path / "raw_ap_2d_tw0fs.png"
    png.write_bytes(b"png")

    skip, msg, stale, _ctx = check_skip_outputs(str(raw), cfg)
    assert skip is False
    assert stale is True
    assert "regenerating" in msg
    assert "sidecar" in msg


def test_check_skip_outputs_sidecar_verified(tmp_path):
    import h5py
    import heom_tools.twodes.post as tp

    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as f:
        res = f.create_group("results")
        res.create_dataset("tw_fs", data=[0.0])
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = "SE,GSB,ESA"

    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = True
    cfg.write_h5 = False
    cfg.time_domain = False
    (tmp_path / "raw_ap_2d_tw0fs.png").write_bytes(b"png")
    write_sidecar(str(tmp_path), "raw", "ap", cfg, str(raw))

    skip, msg, stale, _ctx = check_skip_outputs(str(raw), cfg)
    assert skip is True
    assert stale is False
    assert "sidecar" in msg


def test_load_meta_missing_tw_fs(tmp_path):
    import h5py

    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f.create_group("parameters").attrs["signals"] = "rephasing,nonrephasing"
    with pytest.raises(ValueError, match="tw_fs"):
        load_meta(str(path))


def test_process_png_without_sidecar_regenerates(tmp_path):
    import heom_tools.twodes.post as tp

    raw = tmp_path / "sim.h5"
    _make_solver_h5(raw, n_tau=8, n_t=8)
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = True
    cfg.write_h5 = False
    cfg.time_domain = False
    stale_png = tmp_path / "sim_ap_2d_tw0fs.png"
    stale_png.write_bytes(b"stale")

    assert tp.process(str(raw), cfg, show_config=False) == 0
    assert (tmp_path / ".sim_ap_twodes_post.json").exists()
    assert stale_png.read_bytes() != b"stale"


def test_check_skip_source_solver_change_regenerates(tmp_path):
    import h5py
    import time
    import heom_tools.twodes.post as tp

    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as f:
        res = f.create_group("results")
        res.create_dataset("tw_fs", data=[0.0])
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = "SE,GSB,ESA"

    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = True
    cfg.write_h5 = False
    cfg.time_domain = False
    (tmp_path / "raw_ap_2d_tw0fs.png").write_bytes(b"png")
    write_sidecar(str(tmp_path), "raw", "ap", cfg, str(raw))

    time.sleep(0.02)
    os.utime(raw, None)

    skip, msg, stale, _ctx = check_skip_outputs(str(raw), cfg)
    assert skip is False
    assert stale is True
    assert "regenerating" in msg


def test_require_fft_time_axes_single_point():
    import heom_tools.twodes.post as tp

    with pytest.raises(ValueError, match="tau_fs"):
        require_fft_time_axes(np.array([0.0]), np.linspace(0, 1, 4))


def test_require_fft_time_axes_nonuniform_tau():
    import heom_tools.twodes.post as tp

    tau = np.array([0.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="uniformly spaced"):
        require_fft_time_axes(tau, np.linspace(0, 1, 4))


def test_run_auto_counts_process_failure(monkeypatch, tmp_path, capsys):
    import heom_tools.twodes.post as tp

    raw = tmp_path / "sim.h5"
    _make_solver_h5(raw)
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = False
    cfg.write_h5 = False

    def fail_execute(*_a, **_kw):
        return 1

    def no_skip(h, c):
        return False, "", False, resolve_output_ctx(h, c)

    monkeypatch.setattr(tp.cli, "precheck", lambda h, c: tp.PrecheckResult(*no_skip(h, c)))
    monkeypatch.setattr(tp.cli, "execute", fail_execute)
    monkeypatch.setattr(tp.cli.glob, "glob", lambda _p: [str(raw)])

    assert tp.run_auto(cfg) == 1
    assert "0 processed, 0 skipped, 1 failed" in capsys.readouterr().out


def test_process_h5_incomplete_pathways(tmp_path):
    import h5py
    import heom_tools.twodes.post as tp

    raw = tmp_path / "sim.h5"
    _make_solver_h5(raw, pathways=["SE"])
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = False
    cfg.write_h5 = True
    cfg.h5_spectra = ["rp-se"]
    cfg.time_domain = False

    assert tp.process(str(raw), cfg, show_config=False) == 0
    post = tmp_path / "sim_ap_post.h5"
    with h5py.File(post, "r") as f:
        assert list(f["results/spectra"].keys()) == ["SE"]
        assert f["parameters"].attrs["pathway_sum"] == "SE"
        assert f["parameters"].attrs["pathways"] == "SE"
        assert "source_pathways" not in f["parameters"].attrs
        assert "GSB" not in f["results/spectra"]
        assert "ESA" not in f["results/spectra"]


def test_prune_orphan_plots(tmp_path):
    import numpy as np

    outdir = tmp_path
    stem, ptag = "sim", "ap"
    tw = np.array([0.0, 1000.0])
    (outdir / f"{stem}_{ptag}_2d_tw0fs.png").write_bytes(b"png")
    (outdir / f"{stem}_{ptag}_2d_tw1000fs.png").write_bytes(b"png")
    prune_orphan_plots(str(outdir), stem, ptag, tw, [0], td=False)
    assert (outdir / f"{stem}_{ptag}_2d_tw0fs.png").exists()
    assert not (outdir / f"{stem}_{ptag}_2d_tw1000fs.png").exists()


def test_compute_tw_slice_check_one_summary(capsys):
    S = np.zeros((2, 3, 4, 5), dtype=complex)
    S[0, 0, 2, 3] = 1.0 + 0.5j
    pathways = ["SE", "GSB", "ESA"]
    signals = ["rephasing", "nonrephasing"]
    needs = _slice_needs(pathways, signals)
    compute_tw_slice(
        S, signals, pathways, list(range(3)),
        0, 8, 8,
        needs=needs,
        need_time=False, need_time_path=False,
        fft_method="check",
    )
    out = capsys.readouterr().out
    assert out.count("fft check tw0:") == 1
    assert "n=8 separable_max=" in out
    assert "legacy_mode3_max=" in out


def test_cut_window_includes_bin_beyond_hi():
    dt = 4.0
    dw = 5.0
    w0 = 12000.0
    n = nfft_len(64, dt, dw)
    axis = freq_axis_cm(n, dt, w0)
    data = np.random.randn(n, n)
    x, y, d = cut_window(axis, axis, data, [12000.0, 12800.0])
    assert x.max() >= 12800.0
    assert y.max() >= 12800.0
    assert d.shape == (len(x), len(y))


def test_cut_window_rejects_too_narrow():
    axis = np.linspace(12000.0, 12200.0, 5)
    data = np.zeros((5, 5))
    with pytest.raises(ValueError, match="fewer than 2 bins"):
        cut_window(axis, axis, data, [12100.0, 12105.0])


def test_make_norm_signed_is_symmetric_about_zero():
    import matplotlib as mpl

    z = np.array([[-1.0, 2.0], [-0.5, 1.5]])
    norm = make_norm(z, mpl, nonnegative=False)
    assert norm.vmin == -2.0
    assert norm.vmax == 2.0


def test_make_norm_nonneg_starts_at_zero():
    import matplotlib as mpl

    z = np.abs(np.random.randn(4, 4))
    norm = make_norm(z, mpl, nonnegative=True)
    assert norm.vmin == 0.0
    assert norm.vmax == pytest.approx(float(z.max()))


def test_h5_complex_roundtrip():
    arr = np.random.randn(2, 8, 8) + 1j * np.random.randn(2, 8, 8)
    stored = to_h5_complex(arr)
    assert np.allclose(from_h5_complex(stored), arr)


def test_write_post_h5_freq_only(tmp_path):
    import h5py

    w = np.linspace(-100, 100, 16)
    spec = {
        "SE": {"rephasing": np.ones((1, 16, 16), dtype=complex)},
        "GSB": {"rephasing": np.zeros((1, 16, 16), dtype=complex)},
        "ESA": {"rephasing": np.zeros((1, 16, 16), dtype=complex)},
    }
    total = {"rephasing": spec["SE"]["rephasing"]}
    path = tmp_path / "out_post.h5"
    source = tmp_path / "in.h5"
    source.write_bytes(b"source")
    cfg = PostConfig.defaults().with_updates(
        plot=False,
        write_h5=True,
        h5_spectra=["rp-se"],
    )
    write_post_h5(
        str(path),
        w_tau=w,
        w_t=w,
        tw_vals=np.array([0.0]),
        spec_by_pathway=spec,
        spec_total=total,
        cfg=cfg,
        source_h5=str(source),
        pathway="all",
        signals=["rephasing"],
        pathways=["SE", "GSB", "ESA"],
    )
    with h5py.File(path, "r") as f:
        assert "results/spectra/SE/rephasing" in f
        assert "results/axes/w_tau_cm" in f
        assert "time_domain" not in f["results"]
        assert f["parameters"].attrs["time_domain"] == 1
        assert f["parameters"].attrs["pathways"] == "SE,GSB,ESA"
        raw = f["results/spectra/SE/rephasing"][...]
        assert np.allclose(from_h5_complex(raw), spec["SE"]["rephasing"])


def _make_solver_h5(path, *, n_tau=8, n_tw=1, n_t=8, pathways=None):
    import h5py

    pathways = list(pathways or ["SE", "GSB", "ESA"])
    n_sig, n_path = 2, len(pathways)
    with h5py.File(path, "w") as f:
        dt = np.dtype([("r", "f8"), ("i", "f8")])
        resp = np.zeros((3, n_sig, n_path, n_tau, n_tw, n_t), dtype=dt)
        resp["r"][0, 0, 0, 2, 0, 3] = 1.0
        res = f.create_group("results")
        res.create_dataset("resp", data=resp)
        res.create_dataset("tau_fs", data=np.linspace(0, 12, n_tau))
        res.create_dataset("tw_fs", data=np.linspace(0, 500, n_tw))
        res.create_dataset("t_fs", data=np.linspace(0, 20, n_t))
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = ",".join(pathways)


def test_process_e2e_writes_post_h5(tmp_path):
    import h5py
    import heom_tools.twodes.post as tp

    raw = tmp_path / "sim.h5"
    _make_solver_h5(raw)
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = False
    cfg.write_h5 = True
    cfg.h5_spectra = ["2des"]
    cfg.time_domain = False

    assert tp.process(str(raw), cfg, show_config=False) == 0

    post = tmp_path / "sim_ap_post.h5"
    sidecar = tmp_path / ".sim_ap_twodes_post.json"
    assert post.exists()
    assert sidecar.exists()
    with h5py.File(post, "r") as f:
        assert "results/spectra/total/2des" in f
        assert f["parameters"].attrs["time_domain"] == 0
        assert f["results/axes/w_tau_cm"].shape[0] > 8


def test_process_rewrites_stale_post_h5(tmp_path):
    import h5py
    import heom_tools.twodes.post as tp

    raw = tmp_path / "sim.h5"
    _make_solver_h5(raw, n_tw=2)
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = False
    cfg.write_h5 = True
    cfg.h5_spectra = ["2des"]
    cfg.time_domain = False
    cfg.tw_index = "all"

    assert tp.process(str(raw), cfg, show_config=False) == 0
    post = tmp_path / "sim_ap_post.h5"
    with h5py.File(post, "r") as f:
        n_tw_old = f["results/axes/tw_fs"].shape[0]
    assert n_tw_old == 2

    cfg.tw_index = [0]
    assert tp.process(str(raw), cfg, show_config=False) == 0
    with h5py.File(post, "r") as f:
        assert f["results/axes/tw_fs"].shape[0] == 1
        assert f["parameters"].attrs["tw_index"] == "[0]"


def test_load_meta_minimal_h5(tmp_path):
    import h5py

    path = tmp_path / "mini.h5"
    with h5py.File(path, "w") as f:
        f.create_group("results").create_dataset("tw_fs", data=[0.0, 500.0])
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = "SE,GSB,ESA"
    tw, signals, pathways = load_meta(str(path))
    assert len(tw) == 2
    assert signals == ["rephasing", "nonrephasing"]
    assert pathways == ["SE", "GSB", "ESA"]


def test_load_config_write_h5_requires_h5_spectra(tmp_path):
    cfg_path = tmp_path / "twodes_post.yaml"
    cfg_path.write_text("write_h5: true\n")
    with pytest.raises(ValueError, match="h5_spectra is missing"):
        load_config(str(cfg_path))


def test_load_config_rejects_empty_h5_spectra(tmp_path):
    cfg_path = tmp_path / "twodes_post.yaml"
    cfg_path.write_text("write_h5: true\nh5_spectra: []\n")
    with pytest.raises(ValueError, match="must not be empty"):
        load_config(str(cfg_path))


def test_load_config_rejects_unknown_h5_spectra_token(tmp_path):
    cfg_path = tmp_path / "twodes_post.yaml"
    cfg_path.write_text("write_h5: true\nh5_spectra: [2des, foo]\n")
    with pytest.raises(ValueError, match="unknown h5_spectra"):
        load_config(str(cfg_path))


def test_load_config_ignores_h5_spectra_when_write_h5_false(tmp_path):
    cfg_path = tmp_path / "twodes_post.yaml"
    cfg_path.write_text("write_h5: false\nh5_spectra: [2des, not-a-token]\n")
    cfg = load_config(str(cfg_path))
    assert cfg.h5_spectra is None


def test_output_plan_ignores_h5_spectra_when_write_h5_false():
    from heom_tools.twodes.post.types import OutputPlan

    cfg = PostConfig.defaults()
    cfg.write_h5 = False
    cfg.plot = False
    cfg.h5_spectra = ["2des"]
    plan = OutputPlan.from_cfg(cfg, pathways=["SE", "GSB", "ESA"])
    assert plan.h5_spectra is None
    assert plan.spectra_needs.total_fft == frozenset()
    assert plan.spectra_needs.compute_2des is False


def test_process_h5_pathways_attr_matches_solver(tmp_path):
    """parameters.pathways always lists solver pathways, not only written spectra."""
    import h5py
    import heom_tools.twodes.post as tp

    raw = tmp_path / "sim.h5"
    _make_solver_h5(raw, pathways=["SE", "GSB", "ESA"])
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = False
    cfg.write_h5 = True
    cfg.h5_spectra = ["2des"]
    cfg.time_domain = False

    assert tp.process(str(raw), cfg, show_config=False) == 0
    with h5py.File(tmp_path / "sim_ap_post.h5", "r") as f:
        assert f["parameters"].attrs["pathways"] == "SE,GSB,ESA"
        assert "source_pathways" not in f["parameters"].attrs
        assert list(f["results/spectra/total"].keys()) == ["2des"]


def test_process_h5_spectra_2des_only(tmp_path):
    import h5py
    import heom_tools.twodes.post as tp

    raw = tmp_path / "sim.h5"
    _make_solver_h5(raw)
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = False
    cfg.write_h5 = True
    cfg.h5_spectra = ["2des"]
    cfg.time_domain = False

    assert tp.process(str(raw), cfg, show_config=False) == 0
    with h5py.File(tmp_path / "sim_ap_post.h5", "r") as f:
        assert list(f["results/spectra/total"].keys()) == ["2des"]
        assert "rephasing" not in f["results/spectra/total"]
        assert "GSB" not in f["results/spectra"]


def test_process_h5_spectra_missing_pathway_errors(tmp_path):
    import heom_tools.twodes.post as tp

    raw = tmp_path / "sim.h5"
    _make_solver_h5(raw, pathways=["SE"])
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = False
    cfg.write_h5 = True
    cfg.h5_spectra = ["rp-gsb"]
    cfg.time_domain = False

    assert tp.process(str(raw), cfg, show_config=False) == 1


def test_check_skip_h5_spectra_mismatch_regenerates(tmp_path):
    import h5py
    import heom_tools.twodes.post as tp

    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as f:
        res = f.create_group("results")
        res.create_dataset("tw_fs", data=[0.0])
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = "SE,GSB,ESA"

    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = False
    cfg.write_h5 = True
    cfg.h5_spectra = ["2des"]
    cfg.time_domain = False
    post = tp.post_h5_path(str(raw), cfg)
    with h5py.File(post, "w") as f:
        _minimal_post_h5_attrs(f.create_group("parameters"), source_h5=str(raw))
    write_sidecar(str(tmp_path), "raw", "ap", cfg, str(raw))

    cfg.h5_spectra = ["rp", "nr"]
    skip, msg, stale, _ctx = check_skip_outputs(str(raw), cfg)
    assert skip is False
    assert stale is True
    assert "h5_spectra" in msg


def test_source_precision_mode_propagation(tmp_path) -> None:
    import h5py

    from heom_tools.twodes.post.io import _source_precision_mode

    mixed = tmp_path / "mixed.h5"
    with h5py.File(mixed, "w") as f:
        f.attrs["precision_mode"] = "mixed_fp32"
    assert _source_precision_mode(str(mixed)) == "mixed_fp32"

    legacy = tmp_path / "legacy.h5"
    with h5py.File(legacy, "w") as f:
        f.attrs["task"] = "2des"
    assert _source_precision_mode(str(legacy)) == "fp64"

    assert _source_precision_mode(str(tmp_path / "nope.h5")) == "fp64"
