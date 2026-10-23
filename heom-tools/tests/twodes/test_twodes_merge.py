"""Tests for heom_tools.twodes.merge."""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import pytest

from heom_tools.twodes.merge import merge_solver_2des
from heom_tools.twodes.merge.core import update_config_yaml_tw
from heom_tools.twodes.merge.fingerprint import diff_fingerprints, load_fingerprint
from heom_tools.twodes.post.io import (
    from_h5_complex,
    is_h5_complex_compound,
    is_solver_2des_h5,
    to_h5_complex,
)


def _yaml_template(tw_line: str) -> str:
    return f"""\
system:
  n_states: 7
  temperature: 77.0
2des:
  tau_max: 600.0
  dtau: 8.0
  {tw_line}
  t_max: 600.0
  pathways: ["SE", "GSB", "ESA"]
  signals: ["rephasing", "nonrephasing"]
  orientation: "isotropic"
"""


def _make_mergeable_h5(
    path,
    *,
    tw_values,
    marker,
    hamiltonian_diag=12415.0,
    yaml_tw_line=None,
):
    tw = np.asarray(tw_values, dtype=np.float64)
    n_tau, n_t, n_tw = 6, 6, len(tw)
    n_sig, n_path = 2, 3
    resp = np.zeros((3, n_sig, n_path, n_tau, n_tw, n_t), dtype=np.complex128)
    for ki, tv in enumerate(tw):
        resp[0, 0, 0, 0, ki, 0] = marker + tv

    with h5py.File(path, "w") as f:
        f.attrs["task"] = "2des"
        f.attrs["config_hash"] = "deadbeef"
        res = f.create_group("results")
        res.create_dataset("resp", data=to_h5_complex(resp))
        res.create_dataset("tau_fs", data=np.arange(n_tau, dtype=float) * 8.0)
        res.create_dataset("t_fs", data=np.arange(n_t, dtype=float) * 8.0)
        res.create_dataset("tw_fs", data=tw)
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = "SE,GSB,ESA"
        par.attrs["orientation"] = "isotropic"
        par.attrs["spectrum_type"] = "2des"
        par.attrs["n_sites"] = 7
        par.attrs["w0_cm"] = 12300.0
        par.attrs["basis"] = "double_excited"
        par.attrs["temperature"] = 77.0
        par.attrs["dt"] = 8.0
        h = np.full((7, 7), hamiltonian_diag)
        np.fill_diagonal(h, hamiltonian_diag)
        par.create_dataset("hamiltonian_cm", data=h)
        tw_line = yaml_tw_line or f'tw: ["{int(tw_values[0])}"]'
        par.create_dataset(
            "config_yaml",
            data=np.bytes_(_yaml_template(tw_line).encode("utf-8")),
        )
        der = f.create_group("derived")
        der.create_dataset("eigenvalues_relative_cm", data=np.linspace(0, 100, 7))


def test_update_config_yaml_tw():
    text = _yaml_template('tw: ["0:1000:1000"]')
    out = update_config_yaml_tw(text, [0.0, 104.0, 504.0, 1000.0])
    doc = __import__("yaml").safe_load(out)
    assert doc["2des"]["tw"] == ["0", "104", "504", "1000"]


def test_merge_combines_tw(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _make_mergeable_h5(a, tw_values=[0.0], marker=1.0)
    _make_mergeable_h5(b, tw_values=[500.0], marker=2.0)
    out = tmp_path / "merged.h5"
    summary = merge_solver_2des([str(a), str(b)], str(out))
    assert summary["tw_fs"] == [0.0, 500.0]
    assert is_solver_2des_h5(str(out))
    with h5py.File(out, "r") as f:
        tw = f["/results/tw_fs"][...]
        resp_ds = f["/results/resp"]
        assert is_h5_complex_compound(resp_ds)
        resp = from_h5_complex(resp_ds[...])
        assert list(tw) == [0.0, 500.0]
        assert resp.shape[4] == 2
        assert resp[0, 0, 0, 0, 0, 0] == pytest.approx(1.0 + 0j)
        assert resp[0, 0, 0, 0, 1, 0] == pytest.approx(502.0 + 0j)
        yaml = f["/parameters/config_yaml"][()].decode()
        import yaml as yaml_mod
        assert yaml_mod.safe_load(yaml)["2des"]["tw"] == ["0", "500"]
        sources = json.loads(f["/parameters"].attrs["merge_sources"])
        assert len(sources) == 2


def test_merge_writes_compound_even_from_native_input(tmp_path):
    """Inputs presented as complex128 still emit on-disk compound {r,i}."""
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _make_mergeable_h5(a, tw_values=[0.0], marker=1.0)
    _make_mergeable_h5(b, tw_values=[500.0], marker=2.0)
    out = tmp_path / "merged.h5"
    merge_solver_2des([str(a), str(b)], str(out))
    with h5py.File(out, "r") as f:
        assert is_h5_complex_compound(f["/results/resp"])
        resp = from_h5_complex(f["/results/resp"][...])
        assert resp[0, 0, 0, 0, 0, 0] == pytest.approx(1.0 + 0j)


def test_merge_rejects_physics_mismatch(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _make_mergeable_h5(a, tw_values=[0.0], marker=1.0)
    _make_mergeable_h5(b, tw_values=[100.0], marker=2.0, hamiltonian_diag=99999.0)
    with pytest.raises(ValueError, match="mismatch"):
        merge_solver_2des([str(a), str(b)], str(tmp_path / "out.h5"))


def test_overlap_fail_reports_conflict(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _make_mergeable_h5(a, tw_values=[0.0], marker=1.0)
    _make_mergeable_h5(b, tw_values=[0.0], marker=99.0)
    with pytest.raises(ValueError, match="overlapping waiting times"):
        merge_solver_2des([str(a), str(b)], str(tmp_path / "out.h5"))


def test_overlap_keep_first(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _make_mergeable_h5(a, tw_values=[0.0], marker=1.0)
    _make_mergeable_h5(b, tw_values=[0.0], marker=99.0)
    out = tmp_path / "out.h5"
    summary = merge_solver_2des(
        [str(a), str(b)], str(out), on_overlap="keep-first",
    )
    assert summary["tw_fs"] == [0.0]
    with h5py.File(out, "r") as f:
        assert is_h5_complex_compound(f["/results/resp"])
        resp = from_h5_complex(f["/results/resp"][...])
        assert resp[0, 0, 0, 0, 0, 0] == pytest.approx(1.0 + 0j)


def test_fingerprint_no_diff_for_compatible(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _make_mergeable_h5(a, tw_values=[0.0], marker=1.0)
    _make_mergeable_h5(b, tw_values=[100.0], marker=2.0)
    ref = load_fingerprint(str(a))
    other = load_fingerprint(str(b))
    assert not diff_fingerprints(ref, other)


def test_fingerprint_value_mismatch_reports_max_delta(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _make_mergeable_h5(a, tw_values=[0.0], marker=1.0, hamiltonian_diag=12415.0)
    _make_mergeable_h5(b, tw_values=[100.0], marker=2.0, hamiltonian_diag=12415.0)
    # Corner-only change — must not hide behind ravel()[:4] samples.
    with h5py.File(b, "a") as f:
        h = f["/parameters/hamiltonian_cm"][...]
        h[-1, -1] = 12415.5
        f["/parameters/hamiltonian_cm"][...] = h
    diffs = diff_fingerprints(load_fingerprint(str(a)), load_fingerprint(str(b)))
    assert diffs
    ham = [d for d in diffs if d.field == "parameters/hamiltonian_cm"]
    assert len(ham) == 1
    assert "max|Δ|" in str(ham[0].ref_value)
    assert "at=(6, 6)" in str(ham[0].ref_value)
    assert ham[0].ref_value != ham[0].other_value
    assert "12415.5" in str(ham[0].other_value) or "1.24155e" in str(ham[0].other_value).lower()


def test_fingerprint_tau_mismatch_uses_same_report(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _make_mergeable_h5(a, tw_values=[0.0], marker=1.0)
    _make_mergeable_h5(b, tw_values=[100.0], marker=2.0)
    with h5py.File(b, "a") as f:
        tau = f["/results/tau_fs"][...]
        tau[-1] += 1.0
        f["/results/tau_fs"][...] = tau
    diffs = diff_fingerprints(load_fingerprint(str(a)), load_fingerprint(str(b)))
    tau_diffs = [d for d in diffs if d.field == "results/tau_fs"]
    assert len(tau_diffs) == 1
    assert "max|Δ|" in str(tau_diffs[0].ref_value)
    assert tau_diffs[0].ref_value != tau_diffs[0].other_value


def test_group_by_tw_clusters_float_noise():
    from heom_tools.twodes.merge.core import TwSliceRef, _group_by_tw

    refs = [
        TwSliceRef(500.0, 0, 0, "a.h5"),
        TwSliceRef(500.0 + 1e-9, 1, 0, "b.h5"),
        TwSliceRef(1000.0, 0, 1, "a.h5"),
    ]
    groups = _group_by_tw(refs)
    assert len(groups) == 2
    assert len(groups[500.0]) == 2


def test_overlap_near_equal_tw_is_detected(tmp_path):
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _make_mergeable_h5(a, tw_values=[500.0], marker=1.0)
    _make_mergeable_h5(b, tw_values=[500.0 + 1e-9], marker=99.0)
    with pytest.raises(ValueError, match="overlapping waiting times"):
        merge_solver_2des([str(a), str(b)], str(tmp_path / "out.h5"))


@pytest.mark.skip(reason="requires local FMO cluster data")
def test_fmo_cluster_merge(tmp_path):
    base = os.environ.get("HEOM_FMO_CLUSTER_DATA", "")
    paths = [
        os.path.join(base, name)
        for name in (
            "fmo_2des_20260626_0129.h5",
            "fmo_2des_20260628_0139.h5",
            "fmo_2des_20260628_1137.h5",
        )
    ]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("FMO cluster data not present")
    out = tmp_path / "fmo_merged.h5"
    summary = merge_solver_2des(paths, str(out))
    assert summary["tw_fs"] == [0.0, 104.0, 504.0, 1000.0]
