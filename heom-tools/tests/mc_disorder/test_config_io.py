"""Unit tests for config_io (task-scoped vs legacy yaml schema)."""
import yaml

import pytest

from heom_tools.mc_disorder import config_io as cfg


def _fmo_like_cfg(tmp_path):
    """Minimal task-scoped cfg matching config/fmo.yaml layout."""
    p = tmp_path / "fmo.yaml"
    p.write_text(yaml.safe_dump({
        "system": {
            "n_states": 3,
            "hamiltonian": [[1.0, 0.1, 0.0], [0.1, 2.0, 0.0], [0.0, 0.0, 3.0]],
            "dipoles": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "strengths": [1.0, 1.0, 1.0],
        },
        "static_disorder": {"draws_file": "nodes.h5"},
        "absorption": {
            "dt": 2,
            "n_steps": 10,
            "output_interval": 1,
            "output": {"save_path": "data/fmo_las_mp/"},
        },
        "2des": {
            "dt": 8,
            "tau_max": 100.0,
            "tw": ["0", "50"],
            "output": {"save_path": "data/fmo7_2des_mp/"},
        },
    }))
    return str(p)


def test_read_save_path_task_scoped(tmp_path):
    p = _fmo_like_cfg(tmp_path)
    assert cfg.read_save_path(p, "absorption") == "data/fmo_las_mp/"


def test_read_run_paths_single_load(tmp_path):
    draws = tmp_path / "nodes.h5"
    draws.write_bytes(b"")
    p = _fmo_like_cfg(tmp_path)
    sp, dp = cfg.read_run_paths(p, "absorption")
    assert sp == "data/fmo_las_mp/"
    assert dp == str(draws)


def test_patch_save_path_task_scoped(tmp_path):
    p = _fmo_like_cfg(tmp_path)
    doc = yaml.safe_load(open(p, encoding="utf-8").read())
    cfg.patch_save_path_in_doc(doc, "absorption", "/scratch/run/")
    assert doc["absorption"]["output"]["save_path"] == "/scratch/run/"
    assert "output" not in doc or "absorption" not in (doc.get("output") or {})


def test_patch_save_path_legacy(tmp_path):
    doc = {"output": {"absorption": {"save_path": "data/prod/"}}}
    cfg.patch_save_path_in_doc(doc, "absorption", "/scratch/")
    assert doc["output"]["absorption"]["save_path"] == "/scratch/"


def test_write_probe_cfg_absorption_task_scoped(tmp_path):
    base = _fmo_like_cfg(tmp_path)
    work = tmp_path / "work"
    out = cfg.write_probe_cfg(
        base,
        draws_path=str(tmp_path / "probe_nodes.h5"),
        save_path=str(work / "out"),
        work_dir=str(work),
        task="absorption",
    )
    doc = cfg.load_doc(out)
    assert doc["absorption"]["output"]["save_path"].endswith("/out/")
    assert doc["static_disorder"]["draws_file"].endswith("probe_nodes.h5")
    assert "output" not in doc or "absorption" not in (doc.get("output") or {})


def test_write_probe_cfg_2des_forces_tw_zero(tmp_path):
    base = _fmo_like_cfg(tmp_path)
    work = tmp_path / "work2"
    out = cfg.write_probe_cfg(
        base,
        draws_path=str(tmp_path / "n.h5"),
        save_path=str(work / "out"),
        work_dir=str(work),
        task="2des",
    )
    doc = cfg.load_doc(out)
    assert doc["2des"]["tw"] == ["0"]
    assert doc["2des"]["output"]["save_path"].endswith("/out/")


def test_load_h0_mu_from_system_dipoles(tmp_path):
    doc = yaml.safe_load(open(_fmo_like_cfg(tmp_path), encoding="utf-8").read())
    H0, mu = cfg.load_h0_mu(doc)
    assert H0.shape == (3, 3)
    assert mu.shape == (3, 3)


def test_load_h0_mu_legacy_spectroscopy_block(tmp_path):
    doc = {
        "system": {"hamiltonian": [[1.0]]},
        "spectroscopy": {"dipoles": [[0.0, 1.0, 0.0]], "strengths": [2.0]},
    }
    H0, mu = cfg.load_h0_mu(doc)
    assert H0.shape == (1, 1)
    assert mu[0, 1] == pytest.approx(2.0)


def test_write_calibration_cfg_absolves_draws(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    draws = cfg_dir / "nodes.h5"
    draws.write_bytes(b"")
    base = cfg_dir / "fmo.yaml"
    base.write_text(yaml.safe_dump({
        "static_disorder": {"draws_file": "nodes.h5"},
        "absorption": {"dt": 2, "n_steps": 10, "output_interval": 1,
                       "output": {"save_path": "data/fmo_las_mp/"}},
    }))
    out = tmp_path / "mccal" / "calibration.yaml"
    cfg.write_calibration_cfg(
        str(base), "absorption", str(tmp_path / "scratch"), cfg_path=str(out))
    doc = cfg.load_doc(str(out))
    assert doc["static_disorder"]["draws_file"] == str(draws.resolve())


def test_real_fmo_yaml_if_present():
    import os
    p = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "heom-gpu", "config", "fmo.yaml",
    ))
    if not os.path.isfile(p):
        pytest.skip("solver fmo.yaml not in tree")
    # fmo.yaml is a local, gitignored production config: its save_path value is
    # user-owned and changes freely. Pin the loader contract (non-empty path),
    # not a specific path string.
    sp = cfg.read_save_path(p, "absorption")
    assert isinstance(sp, str) and sp
    H0, mu = cfg.load_h0_mu(cfg.load_doc(p))
    assert H0.shape == (7, 7)
    assert mu.shape == (7, 3)
