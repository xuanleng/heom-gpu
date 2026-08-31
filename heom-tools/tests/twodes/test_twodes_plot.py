"""Unit tests for heom_tools.twodes.plot."""
import json
import os
from unittest import mock

import h5py
import numpy as np
import pytest

from heom_tools.twodes.plot.core import (
    ComparePlotRequest,
    PlotNormConfig,
    PlotSlice,
    SinglePlotRequest,
    SliceSeriesRequest,
    build_animation_request,
    build_compare_request,
    build_plot_request,
    build_series_request,
    extract_real_slice,
    materialize_slices,
    norm_figure_caption,
    prepare_compare,
    prepare_plot_normalization,
    resample_2d,
    resolve_diagonal_reference,
    resolve_tw_by_fs,
    resolve_tw_list,
    resolve_tw_selection,
    shared_re_range,
    warn_compare_metadata_mismatch,
)
from heom_tools.twodes.plot.io import (
    is_post_2des_h5,
    load_post_2des_h5,
    read_spectrum_planes,
)
from heom_tools.twodes.post.h5_spectra import token_spec
from heom_tools.twodes.post.io import to_h5_complex


def _make_post_h5(
    path, *, n_tw=None, n=16, tw_vals=None, w0=12000.0, window=None,
    spectra=("2des",), pol="ap", dw=10.0, w0_cm=None,
):
    tw_vals = np.asarray(tw_vals if tw_vals is not None else [0.0, 500.0, 1000.0])
    if n_tw is None:
        n_tw = int(tw_vals.size)
    if w0_cm is None:
        w0_cm = float(w0)
    w = np.linspace(w0 - 40, w0 + 40, n)
    z = np.random.randn(n_tw, n, n) + 1j * np.random.randn(n_tw, n, n)
    with h5py.File(path, "w") as f:
        res = f.create_group("results")
        ax = res.create_group("axes")
        ax.create_dataset("w_tau_cm", data=w)
        ax.create_dataset("w_t_cm", data=w)
        ax.create_dataset("tw_fs", data=tw_vals)
        spec_root = res.create_group("spectra")
        for tok in spectra:
            spec = token_spec(tok)
            pg = spec_root.require_group(spec.h5_group)
            ds = pg.create_dataset(spec.h5_dataset, shape=z.shape, dtype=to_h5_complex(z).dtype)
            ds[...] = to_h5_complex(z)
        par = f.create_group("parameters")
        par.attrs["spectrum_type"] = "2des_post"
        par.attrs["pol"] = pol
        par.attrs["levels"] = 10
        par.attrs["lines"] = "[]"
        par.attrs["window"] = json.dumps(window)
        par.attrs["h5_spectra"] = json.dumps(list(spectra))
        par.attrs["dw"] = float(dw)
        par.attrs["w0_cm"] = float(w0_cm)


def test_load_post_spectrum_missing(tmp_path):
    p = tmp_path / "a_post.h5"
    _make_post_h5(p, spectra=("2des",))
    with pytest.raises(ValueError, match="spectrum 'rp' not stored"):
        load_post_2des_h5(str(p), spectrum="rp")


def test_read_spectrum_planes_rp(tmp_path):
    p = tmp_path / "a_post.h5"
    _make_post_h5(p, spectra=("2des", "rp"))
    ds = load_post_2des_h5(str(p), spectrum="rp")
    assert ds.spectrum == "rp"
    assert ds.available_spectra == ("2des", "rp")
    planes = read_spectrum_planes(ds, [0])
    assert len(planes) == 1
    assert planes[0].shape == (16, 16)


def test_build_plot_request_spectrum(tmp_path):
    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, tw_vals=[0.0, 500.0], spectra=("2des",))
    with pytest.raises(ValueError, match="spectrum 'rp' not stored"):
        build_plot_request(str(p), [0.0, 500.0], None, spectrum="rp")
    _make_post_h5(p, tw_vals=[0.0, 500.0], spectra=("2des", "rp"))
    req = build_plot_request(str(p), [0.0, 500.0], None, spectrum="rp")
    assert isinstance(req, SliceSeriesRequest)
    assert req.meta.spectrum == "rp"


def test_is_post_2des_h5(tmp_path):
    p = tmp_path / "x_post.h5"
    _make_post_h5(p)
    assert is_post_2des_h5(str(p)) is True
    assert is_post_2des_h5(str(tmp_path / "missing.h5")) is False


def test_is_post_2des_h5_missing_parameters_returns_false(tmp_path):
    """Probe must not raise AttributeError when /parameters is absent."""
    p = tmp_path / "bare.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("/results/axes/w_tau_cm", data=[1.0, 2.0])
    assert is_post_2des_h5(str(p)) is False


def test_is_post_2des_h5_bytes_spectrum_type(tmp_path):
    p = tmp_path / "bytes_post.h5"
    _make_post_h5(p)
    with h5py.File(p, "a") as f:
        f["parameters"].attrs["spectrum_type"] = np.bytes_(b"2des_post")
    assert is_post_2des_h5(str(p)) is True


def test_resolve_tw_by_fs_exact():
    tw = np.array([0.0, 500.0, 1000.0])
    assert resolve_tw_by_fs(tw, 500.0) == 1
    assert resolve_tw_by_fs(tw, 0.0) == 0


def test_resolve_tw_by_fs_rejects_nearest():
    tw = np.array([0.0, 500.0, 1000.0])
    with pytest.raises(ValueError, match="must match exactly"):
        resolve_tw_by_fs(tw, 480.0)


def test_resolve_tw_selection_rejects_duplicate():
    tw = np.array([0.0, 500.0, 1000.0])
    with pytest.raises(ValueError, match="duplicate"):
        resolve_tw_selection(tw, [500.0, 500.0])


def test_extract_real_slice_window(tmp_path):
    p = tmp_path / "a_post.h5"
    _make_post_h5(p, window=[12000.0, 12020.0])
    ds = load_post_2des_h5(str(p))
    sl = extract_real_slice(ds, tw_fs=500.0, window=None)
    assert sl.tw_fs == 500.0
    assert sl.data.ndim == 2
    assert sl.w_tau.min() >= 11960.0
    assert sl.w_tau.max() <= 12060.0


def test_read_spectrum_planes_one_open(tmp_path):
    p = tmp_path / "a_post.h5"
    _make_post_h5(p, tw_vals=[0.0, 500.0, 1000.0])
    ds = load_post_2des_h5(str(p))
    with mock.patch("heom_tools.twodes.plot.io.h5py.File", wraps=h5py.File) as fopen:
        planes = read_spectrum_planes(ds, [0, 1, 2])
    assert len(planes) == 3
    assert fopen.call_count == 1


def test_resample_2d_identity():
    x = np.linspace(0, 1, 5)
    z = np.outer(x, x)
    out = resample_2d(x, x, z, x, x)
    assert np.allclose(out, z)


def test_resample_2d_rejects_extrapolation():
    x = np.linspace(0, 1, 5)
    z = np.outer(x, x)
    x_out = np.linspace(-0.1, 1.1, 5)
    with pytest.raises(ValueError, match="outside"):
        resample_2d(x, x, z, x_out, x)


def test_prepare_compare_same_grid(tmp_path):
    a = tmp_path / "a_post.h5"
    b = tmp_path / "b_post.h5"
    _make_post_h5(a)
    _make_post_h5(b)
    ds_a = load_post_2des_h5(str(a))
    ds_b = load_post_2des_h5(str(b))
    cmp = prepare_compare(ds_a, ds_b, tw_fs=500.0, window=None)
    # raw data, no per-panel normalization
    assert cmp.a.data.shape == cmp.b.data.shape
    assert cmp.delta.shape == cmp.a.data.shape
    assert np.allclose(cmp.delta, cmp.a.data - cmp.b.data)
    assert cmp.resampled_b is False


def test_prepare_compare_resample_warns(tmp_path):
    a = tmp_path / "a_post.h5"
    b = tmp_path / "b_post.h5"
    _make_post_h5(a, n=16)
    _make_post_h5(b, n=20)
    ds_a = load_post_2des_h5(str(a))
    ds_b = load_post_2des_h5(str(b))
    with pytest.warns(UserWarning, match="frequency grids differ"):
        cmp = prepare_compare(ds_a, ds_b, tw_fs=500.0, window=None)
    assert cmp.resampled_b is True
    assert cmp.delta.shape == cmp.a.data.shape


def test_warn_compare_metadata_mismatch(tmp_path):
    a = tmp_path / "a_post.h5"
    b = tmp_path / "b_post.h5"
    _make_post_h5(a, window=[12000.0, 12800.0])
    _make_post_h5(b, window=None)
    ds_a = load_post_2des_h5(str(a))
    ds_b = load_post_2des_h5(str(b))
    with pytest.warns(UserWarning, match="window differs"):
        warn_compare_metadata_mismatch(ds_a, ds_b)


def test_warn_compare_pol_mismatch(tmp_path):
    a = tmp_path / "a_post.h5"
    b = tmp_path / "b_post.h5"
    _make_post_h5(a, pol="ap")
    _make_post_h5(b, pol="perp")
    ds_a = load_post_2des_h5(str(a))
    ds_b = load_post_2des_h5(str(b))
    assert ds_a.dw == pytest.approx(10.0)
    assert ds_a.w0_cm == pytest.approx(12000.0)
    with pytest.warns(UserWarning, match="pol differs"):
        warn_compare_metadata_mismatch(ds_a, ds_b)


def test_warn_compare_dw_w0_mismatch(tmp_path):
    a = tmp_path / "a_post.h5"
    b = tmp_path / "b_post.h5"
    _make_post_h5(a, dw=10.0, w0_cm=12300.0)
    _make_post_h5(b, dw=5.0, w0_cm=12300.0)
    with pytest.warns(UserWarning, match="dw differs"):
        warn_compare_metadata_mismatch(
            load_post_2des_h5(str(a)), load_post_2des_h5(str(b)),
        )
    _make_post_h5(b, dw=10.0, w0_cm=12000.0)
    with pytest.warns(UserWarning, match="w0_cm differs"):
        warn_compare_metadata_mismatch(
            load_post_2des_h5(str(a)), load_post_2des_h5(str(b)),
        )


def test_resolve_tw_list_rejects_duplicate():
    tw = np.array([0.0, 500.0, 1000.0])
    with pytest.raises(ValueError, match="duplicate"):
        resolve_tw_list(tw, [500.0, 500.0])


def test_extract_real_slice_series(tmp_path):
    p = tmp_path / "a_post.h5"
    _make_post_h5(p)
    ds = load_post_2des_h5(str(p))
    with mock.patch("heom_tools.twodes.plot.io.h5py.File", wraps=h5py.File) as fopen:
        sl0 = extract_real_slice(ds, tw_fs=0.0, window=None)
        sl1 = extract_real_slice(ds, tw_fs=500.0, window=None)
    assert fopen.call_count == 2
    assert sl0.tw_fs == 0.0
    assert sl1.tw_fs == 500.0
    assert sl0.data.shape == sl1.data.shape


def test_shared_re_range():
    from heom_tools.twodes.plot.core import PlotSlice

    lo = PlotSlice("a", 0.0, np.array([0.0]), np.array([0.0]), np.array([[1.0]]))
    hi = PlotSlice("a", 1.0, np.array([0.0]), np.array([0.0]), np.array([[3.0]]))
    assert shared_re_range([lo, hi]) == pytest.approx((1.0, 3.0))
    neg = PlotSlice("b", 0.0, np.array([0.0]), np.array([0.0]), np.array([[-8.0, 2.0]]))
    assert shared_re_range([neg, hi]) == pytest.approx((-8.0, 3.0))


def test_diagonal_norm_by_frequency():
    from heom_tools.twodes.plot.core import normalize_slice_at_grid_ref

    w = np.array([12000.0, 12100.0, 12200.0])
    data = np.array([
        [0.2, 0.1, 0.0],
        [0.1, 2.0, 0.1],
        [0.0, 0.1, 0.3],
    ])
    ref = resolve_diagonal_reference(w, w, data, target_freq=12100.0)
    assert ref.i_tau == 1 and ref.i_t == 1
    assert ref.re_value == pytest.approx(2.0)
    sl = PlotSlice("a", 0.0, w, w, data)
    out, used = normalize_slice_at_grid_ref(sl, ref.i_tau, ref.i_t, method=ref.method)
    assert used.re_value == pytest.approx(2.0)
    assert out.data[1, 1] == pytest.approx(1.0)


def test_diagonal_norm_auto_peak():
    w = np.linspace(12000.0, 12200.0, 3)
    data = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, -3.0],
    ])
    ref = resolve_diagonal_reference(w, w, data)
    assert ref.re_value == pytest.approx(-3.0)
    assert ref.i_tau == 2 and ref.i_t == 2


def test_diagonal_norm_shared_grid_from_first_panel():
    w = np.linspace(12000.0, 12200.0, 3)
    data0 = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 3.0],
    ])
    data1 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    sl0 = PlotSlice("a", 0.0, w, w, data0)
    sl1 = PlotSlice("b", 500.0, w, w, data1)
    prepared = prepare_plot_normalization(
        [sl0, sl1], PlotNormConfig(diagonal=True),
    )
    assert prepared.slices[0].data[2, 2] == pytest.approx(1.0)
    assert prepared.slices[1].data[2, 2] == pytest.approx(1.0)
    assert prepared.slices[1].data[1, 1] == pytest.approx(5.0)
    assert prepared.vmax == pytest.approx(5.0)
    assert any("first panel" in line for line in prepared.norm_log)


def test_prepare_plot_default_raw_shared_limits():
    w = np.array([12000.0, 12100.0, 12200.0])
    sl0 = PlotSlice("a", 0.0, w, w, np.array([[-2.0, 1.0], [0.0, 3.0], [0.0, 0.0]]))
    sl1 = PlotSlice("b", 500.0, w, w, np.array([[-1.0, 0.5], [0.0, 2.0], [0.0, 0.0]]))
    prepared = prepare_plot_normalization([sl0, sl1], PlotNormConfig())
    assert prepared.slices[0].data[0, 0] == pytest.approx(-2.0)
    assert prepared.slices[1].data[0, 1] == pytest.approx(0.5)
    assert prepared.vmin == pytest.approx(-2.0)
    assert prepared.vmax == pytest.approx(3.0)
    assert any("raw Re" in line for line in prepared.norm_log)


def test_norm_config_rejects_ref_freq_without_diagonal():
    with pytest.raises(ValueError, match="--norm-ref-freq requires --norm diagonal"):
        PlotNormConfig(ref_freq=12000.0).validate()


def test_norm_figure_caption():
    assert "min/max" in norm_figure_caption(PlotNormConfig())
    cap = norm_figure_caption(PlotNormConfig(diagonal=True))
    assert "first-panel" in cap
    cap = norm_figure_caption(PlotNormConfig(diagonal=True, ref_freq=12350.0))
    assert "12350" in cap
    w = np.array([12000.0, 12100.0, 12200.0])
    sl = PlotSlice(
        "a", 0.0, w, w,
        np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    norm = PlotNormConfig(diagonal=True, ref_freq=12100.0)
    prepared = prepare_plot_normalization([sl], norm)
    assert prepared.slices[0].data[1, 1] == pytest.approx(1.0)
    assert any("T=0 fs" in line for line in prepared.norm_log)
    assert any("ω_τ=12100" in line for line in prepared.norm_log)


def test_shared_re_range_mixed():
    a = PlotSlice("a", 0.0, np.array([0.0]), np.array([0.0]), np.array([[-2.0, 1.0]]))
    b = PlotSlice("a", 1.0, np.array([0.0]), np.array([0.0]), np.array([[-6.0, 3.0]]))
    assert shared_re_range([a, b]) == pytest.approx((-6.0, 3.0))


def test_series_request_defers_h5_read(tmp_path):
    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, tw_vals=[0.0, 500.0, 1000.0])
    with mock.patch("heom_tools.twodes.plot.io.h5py.File", wraps=h5py.File) as fopen:
        series = build_animation_request(str(p), None, None)
        assert len(series.selection) == 3
        assert fopen.call_count == 1  # metadata load only
    slices = materialize_slices(series, None)
    assert len(slices) == 3
    lo, hi = shared_re_range(slices)
    assert hi > 0.0


def test_build_animation_request(tmp_path):
    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, tw_vals=[0.0, 500.0, 1000.0])
    series = build_animation_request(str(p), None, None)
    assert len(series.selection) == 3
    assert series.selection[0][1] == 0.0
    assert series.selection[-1][1] == 1000.0


def test_build_animation_request_requires_two_tw(tmp_path):
    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, n_tw=1, tw_vals=[0.0])
    series = build_animation_request(str(p), None, None)
    assert len(series.selection) == 1


def test_resolve_animation_timing_interval():
    from heom_tools.twodes.plot.cli_common import resolve_animation_timing

    fps, ms = resolve_animation_timing(None, 300.0)
    assert ms == 300
    assert fps == pytest.approx(1000.0 / 300.0)


def test_resolve_animation_timing_fps_default():
    from heom_tools.twodes.plot.cli_common import resolve_animation_timing

    fps, ms = resolve_animation_timing(None, None)
    assert fps == 5.0
    assert ms == 200


def test_cli_animate_gif(tmp_path):
    import heom_tools.twodes.animate.cli as cli

    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, tw_vals=[0.0, 500.0, 1000.0])
    assert cli.main([str(p), "--format", "gif", "--fps", "10"]) == 0
    out = tmp_path / "sim_ap_post_2des_anim.gif"
    assert out.is_file() and out.stat().st_size > 0


def test_cli_animate_default_norm(tmp_path, capsys):
    import heom_tools.twodes.animate.cli as cli

    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, tw_vals=[0.0, 500.0, 1000.0])
    assert cli.main([str(p), "--format", "gif"]) == 0
    out = capsys.readouterr().out
    assert "display: raw Re 2des" in out
    assert (tmp_path / "sim_ap_post_2des_anim.gif").is_file()


def test_cli_animate_diagonal_norm(tmp_path, capsys):
    import heom_tools.twodes.animate.cli as cli

    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, tw_vals=[0.0, 500.0, 1000.0])
    assert cli.main([
        str(p), "--format", "gif", "--norm", "diagonal", "--norm-ref-freq", "12040",
    ]) == 0
    out = capsys.readouterr().out
    assert "normalization: diagonal" in out
    assert "divide by Re=" in out
    assert (tmp_path / "sim_ap_post_2des_anim.gif").is_file()


def test_write_animation_diagonal_norm(tmp_path):
    from unittest.mock import MagicMock, patch

    from heom_tools.twodes.plot.plot import write_animation

    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, n_tw=2, tw_vals=[0.0, 500.0])
    req = build_animation_request(str(p), None, None)
    out = tmp_path / "sim_ap_post_2des_anim.gif"
    with patch("matplotlib.animation.FuncAnimation") as anim_cls:
        anim_cls.return_value.save = MagicMock()
        norm_log = write_animation(
            req, str(out), fmt="gif", fps=5,
            norm_config=PlotNormConfig(diagonal=True),
        )
    assert any("first panel" in line for line in norm_log)
    anim_cls.return_value.save.assert_called_once()



def test_build_plot_request_single_and_multi(tmp_path):
    p_single = tmp_path / "one_tw_post.h5"
    p_multi = tmp_path / "multi_tw_post.h5"
    _make_post_h5(p_single, n_tw=1, tw_vals=[0.0])
    _make_post_h5(p_multi, tw_vals=[0.0, 500.0, 1000.0])
    single = build_plot_request(str(p_single), None, None)
    assert isinstance(single, SinglePlotRequest)
    multi = build_plot_request(str(p_multi), [0.0, 500.0], None)
    assert isinstance(multi, SliceSeriesRequest)
    assert len(multi.selection) == 2


def test_build_compare_request(tmp_path):
    a = tmp_path / "a_post.h5"
    b = tmp_path / "b_post.h5"
    _make_post_h5(a, n_tw=1, tw_vals=[0.0])
    _make_post_h5(b, n_tw=1, tw_vals=[0.0])
    req = build_compare_request(str(a), str(b), [0.0], None)
    assert isinstance(req, ComparePlotRequest)


def test_cli_multi_tw(tmp_path):
    import heom_tools.twodes.plot.cli as cli

    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, tw_vals=[0.0, 500.0, 1000.0])
    assert cli.main([str(p), "--tw", "0", "500", "1000"]) == 0
    out = tmp_path / "sim_ap_post_2des_0fs_500fs_1000fs.png"
    assert out.is_file() and out.stat().st_size > 0


def test_cli_compare(tmp_path):
    import heom_tools.twodes.compare.cli as cli

    a = tmp_path / "a_post.h5"
    b = tmp_path / "b_post.h5"
    _make_post_h5(a, n_tw=1, tw_vals=[0.0])
    _make_post_h5(b, n_tw=1, tw_vals=[0.0])
    assert cli.main([str(a), str(b)]) == 0
    out = tmp_path / "twodes_compare_2des_a_post_vs_b_post_0fs.png"
    assert out.is_file() and out.stat().st_size > 0


def test_cli_compare_rejects_multi_tw(tmp_path):
    import heom_tools.twodes.compare.cli as cli

    a = tmp_path / "a_post.h5"
    b = tmp_path / "b_post.h5"
    _make_post_h5(a, n_tw=1, tw_vals=[0.0])
    _make_post_h5(b, n_tw=1, tw_vals=[0.0])
    assert cli.main([str(a), str(b), "--tw", "0", "500"]) == 1


def test_multi_x_tick_step():
    import matplotlib

    from heom_tools.twodes.plot.plot import (
        _colorbar_ticks,
        _shared_color_scale,
        _multi_x_tick_step,
    )

    assert _multi_x_tick_step(800.0, 5) == 200.0
    assert _multi_x_tick_step(800.0, 2) == 200.0
    assert _multi_x_tick_step(400.0, 5) == 100.0

    ticks = _colorbar_ticks(-0.25, 1.0)
    assert list(ticks) == pytest.approx([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ticks = _colorbar_ticks(-0.26, 1.0)
    assert list(ticks) == pytest.approx([-0.26, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ticks = _colorbar_ticks(-0.26, 1.01)
    assert ticks[-1] == pytest.approx(1.01)
    assert ticks[0] == pytest.approx(-0.26)

    cmap, norm = _shared_color_scale(-0.25, 1.0, matplotlib)
    assert isinstance(norm, matplotlib.colors.Normalize)
    assert norm.vmin == pytest.approx(-0.25)
    assert norm.vmax == pytest.approx(1.0)
    white = cmap(norm(0.0))
    assert white[0] > 0.95 and white[1] > 0.95 and white[2] > 0.95

    cmap_pos, norm_pos = _shared_color_scale(0.15, 1.0, matplotlib)
    assert norm_pos.vmin == pytest.approx(0.15)
    assert norm_pos.vmax == pytest.approx(1.0)


def test_cli_single(tmp_path):
    import heom_tools.twodes.plot.cli as cli

    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, n_tw=1, tw_vals=[0.0])
    assert cli.main([str(p)]) == 0
    out = tmp_path / "sim_ap_post_2des_0fs.png"
    assert out.is_file() and out.stat().st_size > 0


def test_cli_rejects_inexact_tw(tmp_path):
    import heom_tools.twodes.plot.cli as cli

    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, tw_vals=[0.0, 500.0, 1000.0])
    assert cli.main([str(p), "--tw", "480"]) == 1


def test_cli_diagonal_norm(tmp_path, capsys):
    import heom_tools.twodes.plot.cli as cli

    p = tmp_path / "sim_ap_post.h5"
    _make_post_h5(p, n_tw=2, tw_vals=[0.0, 500.0])
    assert cli.main([
        str(p), "--tw", "0", "500", "--norm", "diagonal", "--norm-ref-freq", "12040",
    ]) == 0
    out = capsys.readouterr().out
    assert "normalization: diagonal" in out
    assert "divide by Re=" in out


def test_render_plot_rejects_unknown_request():
    from heom_tools.twodes.plot.core import PlotNormConfig
    from heom_tools.twodes.plot.plot import render_plot

    plt = __import__("matplotlib.pyplot")
    with pytest.raises(TypeError, match="unsupported"):
        render_plot(
            object(), plt, levels=10, lines=[], window=None,
            norm_config=PlotNormConfig(),
        )


def test_multi_panel_title_includes_spectrum():
    import matplotlib.pyplot as plt
    from heom_tools.twodes.plot.core import PlotNormConfig, PlotSlice
    from heom_tools.twodes.plot.plot import render_multi_figure

    axis = np.linspace(12000, 12800, 8)
    z = np.zeros((8, 8))
    slices = [
        PlotSlice(label="sim", tw_fs=0.0, w_tau=axis, w_t=axis, data=z),
        PlotSlice(label="sim", tw_fs=1000.0, w_tau=axis, w_t=axis, data=z),
    ]
    fig, _ = render_multi_figure(
        slices, plt, pol="ap", spectrum="rp-gsb",
        levels=5, lines=[], window=None, norm_config=PlotNormConfig(),
    )
    titles = [ax.get_title() for ax in fig.axes[:2]]
    plt.close(fig)
    assert titles == [
        "T=0 fs  ap  Re rp-gsb",
        "T=1000 fs  ap  Re rp-gsb",
    ]
