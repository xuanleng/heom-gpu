"""Unit tests for mc-disorder-calibrate (no GPU required)."""
import textwrap

import os

import pytest
import yaml

from heom_tools.mc_disorder import calibrate as cal


@pytest.fixture
def dmon_log(tmp_path):
    p = tmp_path / "dmon.log"
    p.write_text(textwrap.dedent("""\
        # Idx sm mem ...
        0 56 0 0 0 0 0 320 0 0
        0 58 0 0 0 0 0 400 0 0
        1 10 0 0 0 0 0 100 0 0
    """))
    return str(p)


def test_parse_dmon_stats_gpu0(dmon_log):
    sm, fb = cal.parse_dmon_stats(dmon_log, 0)
    assert sm == pytest.approx(57.0)
    assert fb == 400


def test_parse_dmon_stats_missing_gpu(dmon_log):
    sm, fb = cal.parse_dmon_stats(dmon_log, 9)
    assert sm is None and fb is None


def test_pick_k_compute_fmo_v100_pattern():
    """FMO absorption: K=1 fastest; higher K over-subscribes."""
    rows = [
        cal.SweepRow(1, 10, 32, 3.20, 56, 322, 112, True, 0),
        cal.SweepRow(2, 25, 32, 1.28, 85, 644, 225, True, 0),
        cal.SweepRow(3, 25, 32, 1.28, 88, 942, 337, True, 0),
    ]
    assert cal.pick_k_compute(rows) == 1


def test_pick_k_compute_monotonic():
    rows = [
        cal.SweepRow(1, 10, 16, 1.0, 40, 200, 100, True, 0),
        cal.SweepRow(2, 10, 16, 1.9, 70, 380, 190, True, 0),
        cal.SweepRow(3, 10, 16, 2.5, 80, 550, 280, True, 0),
    ]
    assert cal.pick_k_compute(rows) == 3


def test_pick_k_gpumem_and_hostmem():
    rows = [
        cal.SweepRow(1, 10, 32, 3.0, 50, 320, 110, True, 0),
        cal.SweepRow(2, 10, 32, 1.2, 80, 640, 220, True, 0),
    ]
    assert cal.pick_k_gpumem(rows, 32768) >= 50
    assert cal.pick_k_hostmem(rows, 64000) >= 100


def test_recommend_k_min_of_three():
    rows = [
        cal.SweepRow(1, 10, 32, 3.20, 56, 322, 112, True, 0),
        cal.SweepRow(2, 25, 32, 1.28, 85, 644, 225, True, 0),
    ]
    rec = cal.recommend_k(rows, gpu_total_mb=32768, mem_avail_mb=64000)
    assert rec["K_compute"] == 1
    assert rec["K_star"] == 1


def test_print_oversubscription_hint(capsys):
    rows = [
        cal.SweepRow(1, 16, 32, 1.97, 62, 734, 157, True, 0),
        cal.SweepRow(2, 26, 32, 1.22, 83, 850, 277, True, 0),
    ]
    cal.print_oversubscription_hint(rows)
    out = capsys.readouterr().out
    assert "oversubscription" in out
    assert "K=2" in out


def test_print_oversubscription_hint_silent_when_monotonic(capsys):
    rows = [
        cal.SweepRow(1, 10, 16, 1.0, 40, 200, 100, True, 0),
        cal.SweepRow(2, 10, 16, 1.9, 70, 380, 190, True, 0),
    ]
    cal.print_oversubscription_hint(rows)
    assert capsys.readouterr().out == ""


def test_precision_display():
    assert cal.run_mod.precision_display("mixed") == "mixed precision"
    assert cal.run_mod.precision_display("fp64") == "fp64 (double precision)"


def test_main_obsolete(capsys):
    assert cal.main(["absorption", "cfg.yaml"]) == 2
    err = capsys.readouterr().err
    assert "obsolete" in err
    assert "mc-disorder-run" in err
    assert "--gpus" in err


def test_write_calibration_cfg_patches_save_path(tmp_path):
    base = tmp_path / "fmo.yaml"
    base.write_text(yaml.safe_dump({
        "output": {"absorption": {"save_path": "data/prod/"}},
        "static_disorder": {"draws_file": "nodes.h5"},
    }))
    scratch = tmp_path / "mccal" / "save"
    cfg_path = tmp_path / "mccal" / "calibration.yaml"
    out = cal.write_calibration_cfg(
        str(base), "absorption", str(scratch), cfg_path=str(cfg_path))
    assert out == str(cfg_path)
    with open(out, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    assert doc["output"]["absorption"]["save_path"].endswith("/save/")


def test_write_calibration_cfg_resolves_relative_draws(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    draws = cfg_dir / "nodes.h5"
    draws.write_bytes(b"")
    base = cfg_dir / "fmo.yaml"
    base.write_text(yaml.safe_dump({
        "static_disorder": {"draws_file": "nodes.h5"},
        "absorption": {"output": {"save_path": "data/prod/"}},
    }))
    cal_cfg = tmp_path / "mccal" / "calibration.yaml"
    cal.write_calibration_cfg(
        str(base), "absorption", str(tmp_path / "scratch"),
        cfg_path=str(cal_cfg),
    )
    doc = yaml.safe_load(open(cal_cfg, encoding="utf-8"))
    assert os.path.isabs(doc["static_disorder"]["draws_file"])
    assert doc["static_disorder"]["draws_file"] == str(draws.resolve())


def test_write_calibration_cfg_task_scoped_schema(tmp_path):
    base = tmp_path / "fmo.yaml"
    base.write_text(yaml.safe_dump({
        "absorption": {"dt": 2, "n_steps": 10, "output_interval": 1,
                       "output": {"save_path": "data/fmo_las_mp/"}},
        "static_disorder": {"draws_file": "nodes.h5"},
    }))
    scratch = tmp_path / "mccal" / "save"
    cfg_path = tmp_path / "mccal" / "calibration.yaml"
    cal.write_calibration_cfg(
        str(base), "absorption", str(scratch), cfg_path=str(cfg_path))
    with open(cfg_path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    assert doc["absorption"]["output"]["save_path"].endswith("/save/")
    assert "output" not in doc or "absorption" not in (doc.get("output") or {})


def test_validate_draws_for_samples_too_few(tmp_path, monkeypatch):
    import h5py

    draws = tmp_path / "nodes.h5"
    with h5py.File(draws, "w") as h:
        h.create_dataset("delta_eps", data=[[1.0, 2.0]] * 8)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "output": {"absorption": {"save_path": "out/"}},
        "static_disorder": {"draws_file": str(draws)},
        "system": {"n_states": 2},
    }))
    with pytest.raises(ValueError, match="exceeds draws_file"):
        cal.validate_draws_for_samples(str(cfg), "absorption", 32)


def test_run_mc_disorder_uses_current_interpreter(monkeypatch, tmp_path):
    calls = []

    class FakeProc:
        pid = 4242

        def wait(self):
            return 0

    def fake_popen(cmd, stdout=None, stderr=None):
        calls.append(cmd)
        return FakeProc()

    monkeypatch.setattr(cal.subprocess, "Popen", fake_popen)
    log = tmp_path / "run.log"
    rc = cal.run_mc_disorder(
        "absorption", "cfg.yaml", gpus="0",
        n_realizations=4, precision="mixed", log_path=str(log))
    assert rc == 0
    assert calls[0][0] == cal.sys.executable
    assert calls[0][1:4] == ["-m", cal._RUN_MODULE, "absorption"]


def test_process_tree_pids_includes_children(monkeypatch):
    tree = {100: [101, 102], 101: [103], 102: [], 103: []}

    def fake_children(pid):
        return tree.get(pid, [])

    monkeypatch.setattr(cal, "_read_child_pids", fake_children)
    assert cal.process_tree_pids([100]) == {100, 101, 102, 103}


def test_dual_card_smoke_returns_rc(monkeypatch, tmp_path):
    monkeypatch.setattr(cal, "run_mc_disorder", lambda *a, **k: 7)
    monkeypatch.setattr(cal, "parse_dmon_stats", lambda *a: (None, None))
    stats, rc = cal.dual_card_smoke(
        "absorption", "cfg.yaml", gpus="0,1",
        n_realizations=4, precision="mixed", out_dir=str(tmp_path))
    assert rc == 7
    assert stats == {0: (None, None), 1: (None, None)}


def test_query_visible_gpu_ids(monkeypatch):
    monkeypatch.setattr(
        cal.subprocess, "check_output",
        lambda *a, **k: "0\n1\n",
    )
    assert cal.query_visible_gpu_ids() == [0, 1]


def test_production_gpu_ids_single_user_multi_visible(monkeypatch):
    monkeypatch.setattr(cal, "query_visible_gpu_ids", lambda: [0, 1])
    gpus, note = cal.production_gpu_ids([0])
    assert gpus == [0, 1]
    assert note is not None
    assert "sweep used GPU 0 only" in note


def test_production_gpu_ids_explicit_list():
    gpus, note = cal.production_gpu_ids([0, 1])
    assert gpus == [0, 1]
    assert note is None


def test_production_gpu_ids_single_card_machine(monkeypatch):
    monkeypatch.setattr(cal, "query_visible_gpu_ids", lambda: [0])
    gpus, note = cal.production_gpu_ids([0])
    assert gpus == [0]
    assert note is None
