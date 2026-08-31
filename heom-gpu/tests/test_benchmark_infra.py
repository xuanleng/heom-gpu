import os, subprocess, sys, tempfile
import pytest
import run_all_benchmarks as rab  # tests/ is on sys.path

ROOT = os.environ.get("HEOM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOLVER = os.environ.get("HEOM_SOLVER", os.path.join(ROOT, "build", "heom_solver"))
PROBE_DIR = os.path.join(ROOT, "tests", "benchmark_data")
TOOLS = os.environ.get("HEOM_TOOLS_DIR", os.path.normpath(os.path.join(ROOT, "..", "heom-tools")))
GEN = os.path.join(TOOLS, "heom_tools", "mc_disorder", "generate.py") if TOOLS else None
needs_tools = pytest.mark.skipif(
    not GEN or not os.path.isfile(GEN),
    reason="no heom-tools (HEOM_TOOLS_DIR empty or missing generate.py)",
)

def test_template_parses_under_current_schema():
    """The benchmark template parses and runs under the current solver."""
    os.makedirs(PROBE_DIR, exist_ok=True)
    content = rab.TEMPLATE.format(filter_enabled="false", tolerance="1.0e-7",
                                  filter_start_step=10, filter_warmup_factor="1.0e-3",
                                  print_interval=0, site=1,
                                  output_dir=PROBE_DIR + "/",
                                  output_name="_infra_probe")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(content); path = f.name
    try:
        r = subprocess.run([SOLVER, "dynamics", path],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, f"solver rejected TEMPLATE:\n{r.stderr[-800:]}"
    finally:
        os.unlink(path)
        probe = os.path.join(PROBE_DIR, "_infra_probe.h5")
        if os.path.exists(probe): os.unlink(probe)


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


def test_compare_absorption_self_identical(tmp_path):
    import yaml, compare_benchmark as cb
    # Tracked CI fixture: FMO7 absorption, disorder OFF, output_interval>0.
    cfg = yaml.safe_load(open(os.path.join(ROOT, "tests", "configs", "benchmark_absorption_fmo7.yaml")))
    h5 = _run_task("absorption", cfg, "_abs_probe", str(tmp_path))
    assert cb.compare_files(h5, h5, tol=1e-15) is True


def test_compare_2des_self_identical(tmp_path):
    import yaml, compare_benchmark as cb
    cfg = yaml.safe_load(open(os.path.join(ROOT, "tests", "configs", "dimer_2des.yaml")))
    h5 = _run_task("2des", cfg, "_2des_probe", str(tmp_path))
    assert cb.compare_files(h5, h5, tol=1e-15) is True


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
