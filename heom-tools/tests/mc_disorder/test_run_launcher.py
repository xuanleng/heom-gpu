"""Launcher wiring test: `launch` must hand every worker the SAME --run-id and
the SAME 1-based inclusive interval (--r-begin/--r-end), plus a unique 1-based
rank; on success it renames each worker's temp log to match its data shard
(<stem>.log). Uses a fake solver script (no real heom_solver / GPU)."""
import os

import numpy as np
import pytest

from heom_tools.mc_disorder import run

# Fake solver: logs its argv, then (if it got an interval) writes a shard named
# exactly as the real solver would — <base>_<key>_<run_id>_r<b>-<e>_rank{N}.h5 —
# into $SAVE, so the launcher's post-run log rename has a file to learn from.
_FAKE_SOLVER = (
    "#!/usr/bin/env python3\n"
    "import os, sys\n"
    "args = sys.argv[1:]\n"
    "with open(os.environ['ARGV_LOG'], 'a') as f:\n"
    "    f.write(' '.join(args) + '\\n')\n"
    "def val(flag):\n"
    "    return args[args.index(flag) + 1]\n"
    "rank, ts = val('--rank'), val('--run-id')\n"
    "b, e = val('--r-begin'), val('--r-end')\n"
    "save = os.environ['SAVE']\n"
    "os.makedirs(save, exist_ok=True)\n"
    "name = f'fmo_absorption_0123456789abcdef_{ts}_r{b}-{e}_rank{rank}.h5'\n"
    "open(os.path.join(save, name), 'w').close()\n"
)


def _write_fake_solver(tmp_path, body=_FAKE_SOLVER):
    solver = tmp_path / "fake_solver.py"
    solver.write_text(body)
    solver.chmod(0o755)
    return solver


# Fake solver that records the GPU it was pinned to (rank -> CUDA_VISIBLE_DEVICES).
_FAKE_SOLVER_ENV = (
    "#!/usr/bin/env python3\n"
    "import os, sys\n"
    "args = sys.argv[1:]\n"
    "cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')\n"
    "with open(os.environ['ARGV_LOG'], 'a') as f:\n"
    "    f.write(args[args.index('--rank')+1] + ' ' + cvd + '\\n')\n"
)


def test_parse_gpus_and_assignment():
    assert run.parse_gpus(None) is None
    assert run.parse_gpus("") is None
    assert run.parse_gpus("0,1,3") == [0, 1, 3]
    # no --gpus -> one unpinned worker
    assert run.plan_gpu_assignment(None) == [None]
    # one worker per card
    assert run.plan_gpu_assignment([0, 1]) == ["0", "1"]
    assert run.plan_gpu_assignment([2]) == ["2"]


def test_parse_gpus_rejects_dups_and_negatives():
    with pytest.raises(ValueError):
        run.parse_gpus("0,0")
    with pytest.raises(ValueError):
        run.parse_gpus("-1,0")


def test_read_save_path_task_scoped_and_legacy(tmp_path):
    import yaml
    task_scoped = tmp_path / "task.yaml"
    task_scoped.write_text(yaml.safe_dump({
        "absorption": {"output": {"save_path": "data/fmo_las/"}},
    }))
    assert run.read_save_path(str(task_scoped), "absorption") == "data/fmo_las/"

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(yaml.safe_dump({
        "output": {"absorption": {"save_path": "data/prod"}},
    }))
    assert run.read_save_path(str(legacy), "absorption") == "data/prod/"


def test_launch_pins_each_worker_to_its_gpu(tmp_path, monkeypatch):
    save = os.path.join(str(tmp_path), "out") + os.sep
    argv_log = tmp_path / "argv.log"
    solver = _write_fake_solver(tmp_path, _FAKE_SOLVER_ENV)
    monkeypatch.setenv("ARGV_LOG", str(argv_log))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    gpu_of_rank = run.plan_gpu_assignment([0, 1])
    run.launch(str(solver), "absorption", "cfg.yaml",
               save_path=save, queue_path=str(tmp_path / "mc.queue"),
               run_id="20260710_0000", r_begin=1, r_end=8,
               gpu_of_rank=gpu_of_rank)

    seen = dict(ln.split() for ln in argv_log.read_text().splitlines())
    assert seen == {"1": "0", "2": "1"}


def test_launch_without_gpus_leaves_env_untouched(tmp_path, monkeypatch):
    save = os.path.join(str(tmp_path), "out") + os.sep
    argv_log = tmp_path / "argv.log"
    solver = _write_fake_solver(tmp_path, _FAKE_SOLVER_ENV)
    monkeypatch.setenv("ARGV_LOG", str(argv_log))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    run.launch(str(solver), "absorption", "cfg.yaml",
               save_path=save, queue_path=str(tmp_path / "mc.queue"),
               run_id="20260710_0000", r_begin=1, r_end=8,
               gpu_of_rank=[None])
    seen = dict(ln.split() for ln in argv_log.read_text().splitlines())
    assert seen == {"1": "<unset>"}


def test_launch_shares_run_id_interval_and_renames_logs(tmp_path, monkeypatch):
    save = os.path.join(str(tmp_path), "out") + os.sep
    argv_log = tmp_path / "argv.log"
    solver = _write_fake_solver(tmp_path)
    monkeypatch.setenv("ARGV_LOG", str(argv_log))
    monkeypatch.setenv("SAVE", save)

    ts = "20260706_1122"
    failed = run.launch(str(solver), "absorption", "cfg.yaml",
                        save_path=save, queue_path=str(tmp_path / "mc.queue"),
                        run_id=ts, r_begin=201, r_end=300,
                        gpu_of_rank=["0", "1", "2"])
    assert failed == []

    lines = argv_log.read_text().splitlines()
    assert len(lines) == 3
    ranks = []
    for ln in lines:
        toks = ln.split()
        assert toks[0] == "absorption" and toks[1] == "cfg.yaml"
        assert toks[toks.index("--run-id") + 1] == ts
        # 1-based inclusive interval passed through verbatim (driver does the -1).
        assert toks[toks.index("--r-begin") + 1] == "201"
        assert toks[toks.index("--r-end") + 1] == "300"
        ranks.append(toks[toks.index("--rank") + 1])
    assert sorted(ranks) == ["1", "2", "3"]

    # Each temp log renamed to its shard stem; no worker_rank*.log left behind.
    for n in (1, 2, 3):
        stem = f"fmo_absorption_0123456789abcdef_{ts}_r201-300_rank{n}"
        assert os.path.isfile(os.path.join(save, stem + ".log"))
        assert not os.path.exists(os.path.join(save, f"worker_rank{n}.log"))


def test_launch_default_interval_is_one_based(tmp_path, monkeypatch):
    """--range omitted upstream ⇒ launcher passes [1, n]; verify 1-based start."""
    save = os.path.join(str(tmp_path), "out") + os.sep
    argv_log = tmp_path / "argv.log"
    solver = _write_fake_solver(tmp_path)
    monkeypatch.setenv("ARGV_LOG", str(argv_log))
    monkeypatch.setenv("SAVE", save)

    run.launch(str(solver), "absorption", "cfg.yaml",
               save_path=save, queue_path=str(tmp_path / "mc.queue"),
               run_id="20260706_0000", r_begin=1, r_end=50,
               gpu_of_rank=[None])
    toks = argv_log.read_text().split()
    assert toks[toks.index("--r-begin") + 1] == "1"
    assert toks[toks.index("--r-end") + 1] == "50"


def test_clean_stale_interval_shards_scoped_to_same_interval(tmp_path):
    """A re-run of --range a b must drop ONLY that interval's old shards/logs;
    other intervals (incremental batches) and merged products stay."""
    save = str(tmp_path) + os.sep
    key = "fmo_absorption_0123456789abcdef"
    # Stale first-attempt shards+logs for range 1-100 (old timestamp).
    stale = [f"{key}_20260706_1100_r1-100_rank{n}.h5" for n in (1, 2)]
    stale += [f"{key}_20260706_1100_r1-100_rank{n}.log" for n in (1, 2)]
    # A different interval (101-200) that must survive.
    keep_other = f"{key}_20260706_1200_r101-200_rank1.h5"
    # A merged product for 1-100 that must survive (no _rank).
    keep_merged = f"{key}_r1-100.h5"
    for fn in stale + [keep_other, keep_merged]:
        open(os.path.join(save, fn), "w").close()

    removed = run.clean_stale_interval_shards(save, 1, 100)

    assert removed == 4
    for fn in stale:
        assert not os.path.exists(os.path.join(save, fn))
    assert os.path.isfile(os.path.join(save, keep_other))
    assert os.path.isfile(os.path.join(save, keep_merged))


def test_clean_stale_interval_shards_scoped_by_disorder_identity(tmp_path):
    """Manifest mode: cleaning replica B must not delete replica A shards."""
    import h5py
    save = str(tmp_path) + os.sep
    key_a = "fmo_absorption_aaaaaaaabbbbbbbb"
    key_b = "fmo_absorption_ccccccccdddddddd"
    id_a = "sobol_gauss@v1|repl=0|seed=1|sigma=30"
    id_b = "sobol_gauss@v1|repl=1|seed=1|sigma=30"
    fa = f"{key_a}_20260706_1200_r1-100_rank1.h5"
    fb = f"{key_b}_20260706_1200_r1-100_rank1.h5"
    for fn, ident in ((fa, id_a), (fb, id_b)):
        with h5py.File(os.path.join(save, fn), "w") as h:
            h.create_group("mc").attrs["disorder_identity"] = ident
    removed = run.clean_stale_interval_shards(save, 1, 100, id_a)
    assert removed == 1
    assert not os.path.exists(os.path.join(save, fa))
    assert os.path.isfile(os.path.join(save, fb))


def test_clean_orphan_interval_shard_logs(tmp_path):
    save = str(tmp_path) + os.sep
    key = "fmo_absorption_0123456789abcdef"
    orphan = f"{key}_20260706_1100_r1-100_rank1.log"
    paired_h5 = f"{key}_20260706_1200_r1-100_rank1.h5"
    paired_log = f"{key}_20260706_1200_r1-100_rank1.log"
    for fn in (orphan, paired_h5, paired_log):
        open(os.path.join(save, fn), "w").close()

    removed = run.clean_orphan_interval_shard_logs(save, 1, 100)

    assert removed == 1
    assert not os.path.exists(os.path.join(save, orphan))
    assert os.path.isfile(os.path.join(save, paired_log))


def test_clean_stale_interval_shards_removes_orphan_logs(tmp_path):
    save = str(tmp_path) + os.sep
    key = "fmo_absorption_0123456789abcdef"
    orphan = f"{key}_20260706_1100_r1-100_rank1.log"
    open(os.path.join(save, orphan), "w").close()

    removed = run.clean_stale_interval_shards(save, 1, 100)

    assert removed == 1
    assert not os.path.exists(os.path.join(save, orphan))


def test_find_solver_precision_picks_binary_by_name(monkeypatch):
    """--precision only chooses the PATH name: fp64 -> heom_solver, mixed ->
    heom_solver_mp. No explicit path in play."""
    monkeypatch.delenv("HEOM_SOLVER", raising=False)
    seen = []
    monkeypatch.setattr(run.shutil, "which",
                        lambda name: seen.append(name) or f"/usr/bin/{name}")
    assert run.find_solver(None, "fp64") == "/usr/bin/heom_solver"
    assert run.find_solver(None, "mixed") == "/usr/bin/heom_solver_mp"
    assert seen == ["heom_solver", "heom_solver_mp"]


def test_find_solver_explicit_path_overrides_precision(tmp_path, monkeypatch):
    """A full --solver path wins even when precision=mixed asks for _mp."""
    monkeypatch.delenv("HEOM_SOLVER", raising=False)
    exe = tmp_path / "my_solver"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert run.find_solver(str(exe), "mixed") == str(exe)


def test_find_solver_missing_mp_names_the_binary(monkeypatch):
    """The not-found error must name the precise binary it looked for."""
    monkeypatch.delenv("HEOM_SOLVER", raising=False)
    monkeypatch.setattr(run.shutil, "which", lambda name: None)
    try:
        run.find_solver(None, "mixed")
    except FileNotFoundError as e:
        msg = str(e)
        assert "heom_solver_mp" in msg
        assert "--precision fp64" in msg
    else:
        raise AssertionError("expected FileNotFoundError")


def test_main_defaults_to_mixed_precision(tmp_path, monkeypatch, capsys):
    """Omitted --precision -> mixed + stderr banner (production default)."""
    import yaml
    monkeypatch.delenv("HEOM_SOLVER", raising=False)
    save = tmp_path / "out"
    save.mkdir()
    draws = tmp_path / "nodes.h5"
    _write_minimal_draws(draws, generator="iid@v1")
    cfg = tmp_path / "fmo.yaml"
    cfg.write_text(yaml.safe_dump({
        "system": {"n_states": 3},
        "output": {"absorption": {"save_path": str(save)}},
        "static_disorder": {"draws_file": draws.name},
    }))
    seen = []

    def fake_find_solver(explicit, precision):
        seen.append(precision)
        return str(_write_fake_solver(tmp_path))

    monkeypatch.setattr(run, "find_solver", fake_find_solver)
    monkeypatch.setattr(run, "_run_batch", lambda **kw: 0)

    rc = run.main(["absorption", str(cfg), "--range", "1", "4"])
    assert rc == 0
    assert seen == ["mixed"]
    err = capsys.readouterr().err
    assert "precision: mixed precision" in err


def test_main_explicit_fp64_shows_precision_banner(tmp_path, monkeypatch, capsys):
    """User-passed --precision fp64 prints double-precision banner."""
    import yaml
    monkeypatch.delenv("HEOM_SOLVER", raising=False)
    save = tmp_path / "out"
    save.mkdir()
    draws = tmp_path / "nodes.h5"
    _write_minimal_draws(draws, generator="iid@v1")
    cfg = tmp_path / "fmo.yaml"
    cfg.write_text(yaml.safe_dump({
        "system": {"n_states": 3},
        "output": {"absorption": {"save_path": str(save)}},
        "static_disorder": {"draws_file": draws.name},
    }))
    seen = []

    def fake_find_solver(explicit, precision):
        seen.append(precision)
        return str(_write_fake_solver(tmp_path))

    monkeypatch.setattr(run, "find_solver", fake_find_solver)
    monkeypatch.setattr(run, "_run_batch", lambda **kw: 0)

    rc = run.main([
        "absorption", str(cfg), "--precision", "fp64",
        "--range", "1", "4",
    ])
    assert rc == 0
    assert seen == ["fp64"]
    err = capsys.readouterr().err
    assert "precision: fp64 (double precision)" in err
    assert "mixed precision" not in err


def test_precision_display_and_format_line():
    assert run.precision_display("mixed") == "mixed precision"
    assert run.precision_display("fp64") == "fp64 (double precision)"
    line = run.format_precision_line("mixed", "/usr/bin/heom_solver_mp")
    assert "mixed precision" in line
    assert "heom_solver_mp" in line


def test_launch_failure_keeps_worker_log(tmp_path, monkeypatch):
    """A worker that fails (no shard produced) keeps its worker_rank{N}.log so the
    tail is still surfaceable — no rename to a non-existent shard."""
    save = os.path.join(str(tmp_path), "out") + os.sep
    argv_log = tmp_path / "argv.log"
    # Fake solver that logs, prints an error, and exits non-zero (no file).
    body = (
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "with open(os.environ['ARGV_LOG'], 'a') as f:\n"
        "    f.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "sys.stderr.write('boom\\n')\n"
        "sys.exit(1)\n"
    )
    solver = _write_fake_solver(tmp_path, body)
    monkeypatch.setenv("ARGV_LOG", str(argv_log))
    monkeypatch.setenv("SAVE", save)

    failed = run.launch(str(solver), "absorption", "cfg.yaml",
                        save_path=save, queue_path=str(tmp_path / "mc.queue"),
                        run_id="20260706_0000", r_begin=1, r_end=10,
                        gpu_of_rank=["0", "1"])
    assert sorted(r for r, _, _ in failed) == [1, 2]
    for n in (1, 2):
        assert os.path.isfile(os.path.join(save, f"worker_rank{n}.log"))


# --- resume gap detection (pure functions) ---

def test_missing_subintervals():
    # done0 is 0-based r_done (as stored in shards)
    assert run.missing_subintervals(set(), 1, 5) == [(1, 5)]
    assert run.missing_subintervals({0, 1, 2, 3, 4}, 1, 5) == []
    # tail gap: first 3 done, crashed on the 4th
    assert run.missing_subintervals({0, 1, 2}, 1, 6) == [(4, 6)]
    # a hole in the middle (r=3 claimed but not finished)
    assert run.missing_subintervals({0, 1, 3, 4}, 1, 5) == [(3, 3)]
    # only gaps INSIDE the requested range; out-of-range done ignored
    assert run.missing_subintervals({0, 1, 2}, 3, 5) == [(4, 5)]


def test_read_done_r_unions_shards(tmp_path):
    import h5py
    save = str(tmp_path) + os.sep
    for name, done in (("a_rank1.h5", [0, 1, 2]), ("a_rank2.h5", [3, 4])):
        with h5py.File(os.path.join(save, name), "w") as h:
            h.create_group("mc").attrs["r_done"] = done
    assert run.read_done_r(save) == {0, 1, 2, 3, 4}


def test_read_done_r_skips_files_without_mc(tmp_path):
    import h5py
    save = str(tmp_path) + os.sep
    with h5py.File(os.path.join(save, "shard_rank1.h5"), "w") as h:
        h.create_group("mc").attrs["r_done"] = [0, 1]
    with h5py.File(os.path.join(save, "not_mc.h5"), "w") as h:
        h.create_dataset("junk", data=[1, 2, 3])
    assert run.read_done_r(save) == {0, 1}


# --- RQMC manifest mode (--manifest) ---

def _write_minimal_draws(path, *, generator, seed=9, n_node=8, sigma=42.0):
    import h5py
    n_site = 3
    with h5py.File(path, "w") as h:
        h.create_dataset("delta_eps", data=np.zeros((n_node, n_site)))
        h.create_dataset("weight", data=np.full(n_node, 1.0 / n_node))
        h.create_dataset("group", data=np.zeros(n_node, dtype=np.int32))
        h.attrs["estimator"] = "mean"
        h.attrs["generator"] = generator
        h.attrs["master_seed"] = seed
        h.attrs["fwhm_cm"] = np.full(n_site, 100.0)
        h.attrs["sigma_cm"] = np.full(n_site, sigma)


def _write_cfg(tmp_path, save_path, n_states=3):
    import yaml
    cfg = tmp_path / "fmo.yaml"
    doc = {
        "system": {"n_states": n_states},
        "output": {"absorption": {"save_path": str(save_path)}},
        "static_disorder": {"draws_file": "unused.h5"},
    }
    cfg.write_text(yaml.safe_dump(doc))
    return cfg


def test_disorder_identity_from_draws_matches_cpp_rule(tmp_path):
    p = tmp_path / "d.h5"
    _write_minimal_draws(p, generator="sobol_gauss@v1|repl=2", seed=11, sigma=30.0)
    assert run.disorder_identity_from_draws(str(p)) == (
        "sobol_gauss@v1|repl=2|seed=11|sigma=30,30,30")


def test_read_rqmc_manifest_resolves_relative_paths(tmp_path):
    import yaml
    from heom_tools.mc_disorder.manifest import read_rqmc_manifest

    d0 = tmp_path / "nodes_r0.h5"
    d1 = tmp_path / "nodes_r1.h5"
    _write_minimal_draws(d0, generator="sobol_gauss@v1|repl=0")
    _write_minimal_draws(d1, generator="sobol_gauss@v1|repl=1")
    manifest = tmp_path / "nodes.manifest.yaml"
    manifest.write_text(yaml.safe_dump({
        "method": "rqmc",
        "n_node": 8,
        "n_replicas": 2,
        "draws_files": ["nodes_r0.h5", "nodes_r1.h5"],
    }))
    got = read_rqmc_manifest(str(manifest))
    assert got["n_replicas"] == 2
    assert got["draws_files"] == [str(d0), str(d1)]


def test_read_rqmc_manifest_rejects_non_rqmc_draws(tmp_path):
    import yaml
    from heom_tools.mc_disorder.manifest import read_rqmc_manifest

    d = tmp_path / "iid.h5"
    _write_minimal_draws(d, generator="iid_gauss@v1")
    manifest = tmp_path / "nodes.manifest.yaml"
    manifest.write_text(yaml.safe_dump({
        "method": "rqmc",
        "n_node": 8,
        "n_replicas": 1,
        "draws_files": ["iid.h5"],
    }))
    with pytest.raises(ValueError, match="not an rqmc node file"):
        read_rqmc_manifest(str(manifest))


def test_read_done_r_filters_by_disorder_identity(tmp_path):
    import h5py
    save = str(tmp_path) + os.sep
    for name, ident, done in (
        ("a_rank1.h5", "sobol_gauss@v1|repl=0|seed=1|sigma=1.0", [0, 1]),
        ("b_rank1.h5", "sobol_gauss@v1|repl=1|seed=1|sigma=1.0", [2, 3]),
    ):
        with h5py.File(os.path.join(save, name), "w") as h:
            mc = h.create_group("mc")
            mc.attrs["r_done"] = done
            mc.attrs["disorder_identity"] = ident
    id0 = "sobol_gauss@v1|repl=0|seed=1|sigma=1.0"
    assert run.read_done_r(save, id0) == {0, 1}
    assert run.read_done_r(save) == {0, 1, 2, 3}


def test_write_replica_cfg_patches_draws_file(tmp_path):
    import yaml
    cfg = _write_cfg(tmp_path, tmp_path / "out")
    draws = tmp_path / "nodes_r2.h5"
    _write_minimal_draws(draws, generator="sobol_gauss@v1|repl=2")
    patched = run.write_replica_cfg(str(cfg), str(tmp_path / "out") + os.sep,
                                    2, str(draws))
    with open(patched) as fh:
        doc = yaml.safe_load(fh)
    assert doc["static_disorder"]["draws_file"] == str(draws)


def test_main_manifest_launches_each_replica(tmp_path, monkeypatch):
    import yaml
    save = tmp_path / "out"
    save.mkdir()
    cfg = _write_cfg(tmp_path, save)
    draws = []
    for repl in range(2):
        p = tmp_path / f"nodes_r{repl}.h5"
        _write_minimal_draws(p, generator=f"sobol_gauss@v1|repl={repl}")
        draws.append(str(p))
    manifest = tmp_path / "nodes.manifest.yaml"
    manifest.write_text(yaml.safe_dump({
        "method": "rqmc",
        "n_node": 8,
        "n_replicas": 2,
        "draws_files": [os.path.basename(d) for d in draws],
    }))

    argv_log = tmp_path / "argv.log"
    solver = _write_fake_solver(tmp_path)
    monkeypatch.setenv("ARGV_LOG", str(argv_log))
    monkeypatch.setenv("SAVE", str(save) + os.sep)
    monkeypatch.setattr(run, "find_solver", lambda *a, **k: str(solver))
    monkeypatch.setattr(run.time, "strftime", lambda *a: "20260706_1200")

    rc = run.main([
        "absorption", str(cfg), "--precision", "mixed",
        "--manifest", str(manifest), "--range", "1", "4",
    ])
    assert rc == 0
    lines = argv_log.read_text().splitlines()
    assert len(lines) == 2
    run_ids = [ln.split()[ln.split().index("--run-id") + 1] for ln in lines]
    assert run_ids[0] != run_ids[1]
    assert run_ids[0].endswith("_repl0")
    assert run_ids[1].endswith("_repl1")
    cfgs = [ln.split()[1] for ln in lines]
    assert all(".mc_run_r" in c for c in cfgs)
    assert cfgs[0] != cfgs[1]


def test_save_path_lock_exclusive(tmp_path):
    save = str(tmp_path) + os.sep
    lock = os.path.join(save, ".mc_run.lock")
    with run.save_path_lock(save):
        assert os.path.isfile(lock)
        with pytest.raises(RuntimeError, match="lock"):
            with run.save_path_lock(save):
                pass
    assert not os.path.exists(lock)


def test_validate_manifest_rejects_n_site_mismatch(tmp_path):
    save = tmp_path / "out"
    save.mkdir()
    cfg = _write_cfg(tmp_path, save, n_states=7)
    draws = tmp_path / "nodes_r0.h5"
    _write_minimal_draws(draws, generator="sobol_gauss@v1|repl=0")
    manifest = {
        "method": "rqmc",
        "n_node": 8,
        "n_site": 3,
        "n_replicas": 1,
        "draws_files": [os.path.basename(str(draws))],
    }
    import yaml
    mpath = tmp_path / "bad.manifest.yaml"
    mpath.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="n_site"):
        run.validate_manifest_against_config(
            run.read_rqmc_manifest(str(mpath)), str(cfg))
