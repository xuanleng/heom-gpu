"""Solver-dependent MC tests (need the heom_solver binary + GPU).

The pure generator tests (schema / reproducible / append / sigma_zero /
extend / print-params) live with the package at
heom-tools/tests/mc_disorder/test_generate.py. Here we only
keep the tests that actually run the solver.
"""
import glob, subprocess, sys, os, h5py, numpy as np, yaml
ROOT = os.environ.get("HEOM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Generator lives in the distributable package (single source of truth); ctest
# passes the CMake-validated checkout.
TOOLS = os.environ.get("HEOM_TOOLS_DIR", os.path.normpath(os.path.join(ROOT, "..", "heom-tools")))
GEN = os.path.join(TOOLS, "heom_tools", "mc_disorder", "generate.py")
SOLVER = os.environ.get("HEOM_SOLVER", os.path.join(ROOT, "build", "heom_solver"))
ABS_CFG = os.path.join(ROOT, "tests", "configs", "benchmark_absorption_fmo7.yaml")


def _gen(tmp, **kw):
    os.makedirs(tmp, exist_ok=True)
    out = os.path.join(tmp, "d.h5")
    cmd = [sys.executable, GEN, "--out", out, "--n-node", str(kw.get("n", 8)),
           "--n-site", str(kw.get("s", 7)), "--fwhm", str(kw.get("fwhm", 100.0)),
           "--seed", str(kw.get("seed", 42)), "--method", "iid"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"generator failed:\n{r.stderr}"
    return out


def _make_cfg(draws, run_dir):
    """Synthesise a per-run config from the benchmark with static_disorder.draws_file
    + a private save_path. use_timestamp off so the
    single-shot filename is deterministic. Returns the config path."""
    os.makedirs(run_dir, exist_ok=True)
    with open(ABS_CFG) as fh:
        doc = yaml.safe_load(fh)
    doc["static_disorder"] = {"draws_file": os.path.abspath(draws)}
    # save_path is string-concatenated with the filename (save_path + name.h5),
    # so it MUST end with a separator or the single-shot file lands in the parent
    # dir with a run-dir-name prefix instead of inside run_dir.
    doc["absorption"]["output"]["save_path"] = run_dir.rstrip("/") + "/"
    doc["absorption"]["output"]["use_timestamp"] = False
    cfg = os.path.join(run_dir, "cfg.yaml")
    with open(cfg, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    return cfg


def _run_absorption_mc(draws, run_dir, run_id="20260705_0000"):
    """Config-driven MC worker rank 1. Rank files now use the single-shot format
    (mean in /results/rho_xyz) + a /mc provenance group, named
    <base>_<run_id>_rank1.h5. Returns the shard path (globbed to stay agnostic
    to the synthesised base name)."""
    cfg = _make_cfg(draws, run_dir)
    r = subprocess.run([SOLVER, "absorption", cfg, "--rank", "1",
                        "--run-id", run_id],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"MC absorption failed:\n{r.stderr[-800:]}"
    shards = glob.glob(os.path.join(run_dir, "*_rank1.h5"))
    assert len(shards) == 1, f"expected one rank1 shard, got {shards}"
    # The shard was built under .partial/ and renamed out once /mc was
    # appended; a finished worker leaves nothing behind there, and the shard
    # that IS visible must already carry /mc.
    leftovers = glob.glob(os.path.join(run_dir, ".partial", "*.h5"))
    assert leftovers == [], f"half-written shards left in .partial/: {leftovers}"
    with h5py.File(shards[0], "r") as h:
        assert "mc" in h, "published shard has no /mc group"
    return shards[0]


def _run_absorption_single_shot(draws, run_dir):
    """Same config, no --rank => ordinary single-shot absorption. Returns the
    <base>.h5 path (use_timestamp off ⇒ deterministic, and it is the only
    non-rank absorption file in the fresh dir)."""
    cfg = _make_cfg(draws, run_dir)
    r = subprocess.run([SOLVER, "absorption", cfg],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"single-shot absorption failed:\n{r.stderr[-800:]}"
    # Base-name agnostic: the fresh run_dir holds one non-rank .h5 (the auto stem
    # no longer carries a task suffix). MC shards end in _rankN.h5 and are excluded.
    files = [f for f in glob.glob(os.path.join(run_dir, "*.h5"))
             if "_rank" not in os.path.basename(f)]
    assert len(files) == 1, f"expected one single-shot file, got {files}"
    return files[0]


def _se_l2(part):
    """Aggregate standard error of the mean over all elements.

    mean lives in /results/rho_xyz; count and Σ|O|² in the /mc group.
    se_k = sqrt(max(sum_obs_sq_k/count - |mean_k|², 0)) / sqrt(count)
    Returns L2 norm over k = sqrt(Σ_k var_k / count); the explicit 1/sqrt(count)
    makes it robustly shrink with count (deterministic, not a flaky fluctuation).
    """
    with h5py.File(part) as p:
        count = int(p["/mc/count"][()])
        mean = p["/results/rho_xyz"][...]
        var = np.maximum(p["/mc/sum_obs_sq"][...] / count - np.abs(mean) ** 2, 0.0)
        return float(np.sqrt(var.sum() / count))


def test_sigma_positive_reproducible(tmp_path):
    """The same draws file (sigma > 0), run twice on one worker of one build,
    reproduces /results/rho_xyz bit for bit."""
    draws = _gen(str(tmp_path), n=4, s=7, fwhm=100.0, seed=123)
    p1 = _run_absorption_mc(draws, os.path.join(tmp_path, "run1"))
    p2 = _run_absorption_mc(draws, os.path.join(tmp_path, "run2"))
    with h5py.File(p1) as a, h5py.File(p2) as b:
        assert a["/mc/count"][()] == 4 and b["/mc/count"][()] == 4
        maxabs = float(np.max(np.abs(a["/results/rho_xyz"][...]
                                     - b["/results/rho_xyz"][...])))
        assert maxabs == 0.0, f"σ>0 MC not bit-reproducible: maxabs={maxabs}"


def test_sigma_zero_matches_single_shot(tmp_path):
    """sigma = 0, one node: the MC rank file matches the single-shot rho_xyz
    of the same build."""
    draws = _gen(str(tmp_path), n=1, s=7, fwhm=0.0, seed=7)
    shard = _run_absorption_mc(draws, os.path.join(tmp_path, "mc"))
    single = _run_absorption_single_shot(draws, os.path.join(tmp_path, "single"))
    with h5py.File(shard) as a, h5py.File(single) as b:
        assert a["/mc/count"][()] == 1
        assert "/mc" not in b, "single-shot file must not carry a /mc group"
        maxabs = float(np.max(np.abs(a["/results/rho_xyz"][...]
                                     - b["/results/rho_xyz"][...])))
        assert maxabs == 0.0, f"σ=0 rank file != single-shot: maxabs={maxabs}"


def test_queue_is_per_batch(tmp_path):
    """Direct --rank workers in ONE save_path: a new run-id is a new batch and
    must claim every node even though the previous batch drained its counter;
    the same run-id and range joins the drained batch and claims none."""
    draws = _gen(str(tmp_path), n=2, s=7, fwhm=100.0, seed=5)
    run_dir = os.path.join(tmp_path, "run")
    cfg = _make_cfg(draws, run_dir)

    def worker(run_id):
        r = subprocess.run([SOLVER, "absorption", cfg, "--rank", "1", "--run-id", run_id],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-800:]
        return glob.glob(os.path.join(run_dir, f"*_{run_id}_r1-2_rank1.h5")), r.stdout

    first, _ = worker("20260101_000001")
    second, _ = worker("20260101_000002")
    assert len(first) == 1 and len(second) == 1
    with h5py.File(second[0]) as h:
        assert h["/mc/count"][()] == 2
    assert sorted(os.path.basename(q) for q in glob.glob(os.path.join(run_dir, "mc.*.queue"))) \
        == ["mc.20260101_000001_r1-2.queue", "mc.20260101_000002_r1-2.queue"]

    rejoin, out = worker("20260101_000002")
    assert rejoin == second and "claimed 0 nodes" in out


def test_error_estimator_wellformed(tmp_path):
    """Same seed, 8 nodes versus 16 (the 16-node file extends the 8-node file).
    The standard error is finite, non-negative, and shrinks as the count grows."""
    d8 = _gen(str(tmp_path / "a"), n=8, s=7, fwhm=100.0, seed=77)
    d16 = _gen(str(tmp_path / "b"), n=16, s=7, fwhm=100.0, seed=77)
    se8 = _se_l2(_run_absorption_mc(d8, os.path.join(tmp_path, "run8")))
    se16 = _se_l2(_run_absorption_mc(d16, os.path.join(tmp_path, "run16")))
    assert np.isfinite(se8) and se8 >= 0.0
    assert np.isfinite(se16) and se16 >= 0.0
    assert se16 <= se8 + 1e-12, f"se did not shrink: se8={se8}, se16={se16}"
