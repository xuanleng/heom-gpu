import os, shutil, subprocess, sys
import pytest
import regress  # tests/ is on sys.path

ROOT = os.environ.get("HEOM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOLVER = os.environ.get("HEOM_SOLVER", os.path.join(ROOT, "build", "heom_solver"))
# ctest passes the CMake-validated checkout, or "" for a solver-only tree.
TOOLS = os.environ.get("HEOM_TOOLS_DIR", os.path.normpath(os.path.join(ROOT, "..", "heom-tools")))
GEN = os.path.join(TOOLS, "heom_tools", "mc_disorder", "generate.py") if TOOLS else None
needs_tools = pytest.mark.skipif(GEN is None, reason="no heom-tools (HEOM_TOOLS_DIR empty)")


def test_template_parses_under_current_schema(tmp_path):
    """The FMO-7 per-site template parses and runs under the current solver."""
    path = tmp_path / "probe.yaml"
    path.write_text(regress.fmo7_site_config(site=1, use_filter=False,
                                             output_dir=str(tmp_path), output_name="probe"))
    r = subprocess.run([SOLVER, "dynamics", str(path)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"solver rejected TEMPLATE:\n{r.stderr[-800:]}"


@pytest.mark.parametrize("argv, msg", [
    (["--rank", "1x"], "expects an integer, got '1x'"),
    (["--rank", "1", "--r-begin", "2", "--r-end", "3e"], "expects an integer, got '3e'"),
    (["--rank", ""], "expects an integer"),
    (["--run-id", "20260101_0000"], "only apply to an MC worker"),
    (["--r-begin", "1", "--r-end", "4"], "only apply to an MC worker"),
])
def test_worker_flags_are_strict(argv, msg):
    """Worker flags are parsed before the YAML is read, so no config is needed:
    a mistyped integer or a worker flag without --rank must fail, not be ignored."""
    r = subprocess.run([SOLVER, "absorption", "/nonexistent.yaml", *argv],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and msg in r.stderr, r.stderr


def _run_task(task, cfg_dict, out_name, out_dir):
    import yaml
    cfg = dict(cfg_dict)
    cfg.setdefault(task, {})["output"] = {"save_path": out_dir + "/", "filename": out_name, "use_timestamp": False}
    ypath = os.path.join(out_dir, out_name + ".yaml")
    open(ypath, "w").write(yaml.dump(cfg))
    r = subprocess.run([SOLVER, task, ypath], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{task} solver failed:\n{r.stderr[-800:]}"
    return os.path.join(out_dir, out_name + ".h5")


def _run_mc_single(task, cfg_dict, out_name, out_dir, draws):
    """New CLI: config carries static_disorder.draws_file; worker rank 1 writes a
    shard named <base>_<key>_<TS>_r..._rank1.h5 (single-shot schema in /results +
    a /mc provenance group). Returns that shard path (globbed, base-name agnostic)."""
    import glob, yaml
    cfg = dict(cfg_dict)
    cfg.setdefault(task, {})["output"] = {"save_path": out_dir + "/", "filename": out_name, "use_timestamp": False}
    cfg["static_disorder"] = {"draws_file": os.path.abspath(draws)}
    ypath = os.path.join(out_dir, out_name + "_mc.yaml")
    open(ypath, "w").write(yaml.dump(cfg))
    r = subprocess.run([SOLVER, task, ypath, "--rank", "1"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"MC {task} failed:\n{r.stderr[-800:]}"
    shards = glob.glob(os.path.join(out_dir, "*_rank1.h5"))
    assert len(shards) == 1, f"expected one rank1 shard, got {shards}"
    return shards[0]


def _assert_compare_catches_perturbation(h5, dset, tmp_path):
    """compare_files must accept the file itself and reject a copy with one
    payload element moved by 1e-9 — a self-compare alone proves nothing."""
    import h5py
    bad = os.path.join(tmp_path, "perturbed.h5")
    shutil.copy(h5, bad)
    with h5py.File(bad, "a") as f:
        v = f[dset][...]
        v.flat[v.size // 2] += 1e-9
        f[dset][...] = v
    assert regress.compare_files(h5, h5, tol=1e-15) is True
    assert regress.compare_files(h5, bad, tol=1e-12) is False


def test_compare_absorption_catches_perturbation(tmp_path):
    import yaml
    # Tracked CI fixture: FMO7 absorption, disorder OFF, output_interval>0.
    cfg = yaml.safe_load(open(os.path.join(ROOT, "tests", "configs", "benchmark_absorption_fmo7.yaml")))
    h5 = _run_task("absorption", cfg, "_abs_probe", str(tmp_path))
    _assert_compare_catches_perturbation(h5, "/results/rho_xyz", tmp_path)


def test_compare_2des_catches_perturbation(tmp_path):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "tests", "configs", "dimer_2des.yaml")))
    h5 = _run_task("2des", cfg, "_2des_probe", str(tmp_path))
    _assert_compare_catches_perturbation(h5, "/results/resp", tmp_path)


@needs_tools
def test_absorption_single_node_bit_identical(tmp_path):
    """sigma = 0, one node: the MC partial matches a single-shot rho_xyz
    from the same build, bit for bit. A committed baseline can differ by
    a build, so it is not the reference here.
    """
    import h5py, numpy as np, yaml
    # 1) same-build single-shot absorption, the reference rho_xyz.
    cfg = yaml.safe_load(open(os.path.join(ROOT, "tests", "configs", "benchmark_absorption_fmo7.yaml")))
    ref = _run_task("absorption", cfg, "_abs_ref", str(tmp_path))

    # 2) sigma = 0, one node. n_site must match n_states = 7.
    draws = os.path.join(tmp_path, "d.h5")
    subprocess.run([sys.executable, GEN, "--out", draws, "--n-node", "1",
                    "--n-site", "7", "--fwhm", "0.0", "--seed", "1",
                    "--method", "iid"], check=True)

    # 3) MC run: draws_file plus --rank 1.
    part = _run_mc_single("absorption", cfg, "_abs_ref", str(tmp_path), draws)

    # 4) bit compare, plus count / r_done. One node => shard equals single-shot.
    with h5py.File(part) as p, h5py.File(ref) as rf:
        assert np.array_equal(p["/results/rho_xyz"][...], rf["/results/rho_xyz"][...]), \
            "shard /results/rho_xyz != single-shot rho_xyz"
        assert p["/mc/count"][()] == 1
        assert list(p["/mc"].attrs["r_done"]) == [0]


@needs_tools
def test_fluorescence_single_node_bit_identical(tmp_path):
    """sigma = 0, one node: the fluorescence shard matches the single-shot
    rho_xyz. The shard omits rho_eq and equilibration."""
    import h5py, numpy as np, yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "tests", "configs", "dimer_fluorescence.yaml")))
    ref = _run_task("fluorescence", cfg, "_fl_ref", str(tmp_path))

    draws = os.path.join(tmp_path, "dfl.h5")
    subprocess.run([sys.executable, GEN, "--out", draws, "--n-node", "1",
                    "--n-site", str(cfg["system"]["n_states"]), "--fwhm", "0.0",
                    "--seed", "1", "--method", "iid"], check=True)

    part = _run_mc_single("fluorescence", cfg, "_fl_ref", str(tmp_path), draws)

    with h5py.File(part) as p, h5py.File(ref) as rf:
        assert np.array_equal(p["/results/rho_xyz"][...], rf["/results/rho_xyz"][...]), \
            "shard /results/rho_xyz != single-shot rho_xyz"
        assert "/results/rho_eq" in rf and "/results/rho_eq" not in p
        assert "/results/equilibration" not in p
        assert p["/mc/count"][()] == 1


@needs_tools
def test_2des_single_node_bit_identical(tmp_path):
    """sigma = 0, one node: the 2DES MC partial matches the single-shot response."""
    import h5py, numpy as np, yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "tests", "configs", "benchmark_3sites_2des.yaml")))
    ref = _run_task("2des", cfg, "_2des_ref", str(tmp_path))
    n_states = cfg["system"]["n_states"]

    draws = os.path.join(tmp_path, "d2.h5")
    subprocess.run([sys.executable, GEN, "--out", draws, "--n-node", "1",
                    "--n-site", str(n_states), "--fwhm", "0.0", "--seed", "1",
                    "--method", "iid"], check=True)

    part = _run_mc_single("2des", cfg, "_2des_ref", str(tmp_path), draws)

    with h5py.File(part) as p, h5py.File(ref) as rf:
        assert np.array_equal(p["/results/resp"][...], rf["/results/resp"][...]), \
            "shard /results/resp != single-shot resp"
        assert p["/mc/count"][()] == 1
        assert list(p["/mc"].attrs["r_done"]) == [0]
