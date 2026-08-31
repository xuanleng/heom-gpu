"""probe_orch config wiring (no GPU)."""
import yaml

from heom_tools.mc_disorder import config_io as cfg
from heom_tools.mc_disorder import probe_orch as orch


def _fmo_like(tmp_path):
    base = tmp_path / "fmo.yaml"
    base.write_text(yaml.safe_dump({
        "system": {
            "n_states": 2,
            "hamiltonian": [[1.0, 0.0], [0.0, 2.0]],
            "dipoles": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "strengths": [1.0, 1.0],
        },
        "static_disorder": {"draws_file": "nodes.h5"},
        "absorption": {
            "dt": 2, "n_steps": 4, "output_interval": 1,
            "output": {"save_path": "data/las/"},
        },
        "2des": {
            "dt": 8, "tau_max": 50.0, "tw": ["0", "100"],
            "output": {"save_path": "data/2des/"},
        },
    }))
    return str(base)


def test_cfg_for_nodes_absorption_uses_task_scoped_save_path(tmp_path):
    work = tmp_path / "run"
    out = orch._cfg_for_nodes(
        _fmo_like(tmp_path), str(tmp_path / "d.h5"), str(work / "out"),
        str(work), task="absorption",
    )
    doc = cfg.load_doc(out)
    assert doc["absorption"]["output"]["save_path"].endswith("/out/")
    assert "output" not in doc or "absorption" not in (doc.get("output") or {})


def test_cfg_for_nodes_2des_sets_tw_on_task_block(tmp_path):
    work = tmp_path / "run"
    out = orch._cfg_for_nodes(
        _fmo_like(tmp_path), str(tmp_path / "d.h5"), str(work / "out"),
        str(work), task="2des",
    )
    doc = cfg.load_doc(out)
    assert doc["2des"]["tw"] == ["0"]
    assert doc["2des"]["output"]["save_path"].endswith("/out/")


def test_n_site_and_h0_mu_from_task_scoped(tmp_path):
    base = _fmo_like(tmp_path)
    assert orch._n_site_of(base) == 2
    H0, mu = orch._load_h0_mu(base)
    assert H0.shape == (2, 2)
    assert mu.shape == (2, 3)
