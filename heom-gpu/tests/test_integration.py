#!/usr/bin/env python3
"""Integration tests (pytest):

  - test_fmo7_integration : FMO-7 dynamics vs reference (skips if no ref file).
  - test_2des_dimer_smoke : carry-through 2DES dimer smoke.

Both functions stay importable without side effects (no top-level solver runs),
so `python -m pytest tests/test_integration.py -k 2des` collects cleanly.
Also runnable as a plain script: `python tests/test_integration.py`.
"""
import os, sys, subprocess, tempfile

root = os.environ.get("HEOM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
solver = os.environ.get("HEOM_SOLVER", os.path.join(root, "build", "heom_solver"))


# ctest reads this as "skipped", not "passed" (SKIP_RETURN_CODE in CMakeLists).
# Reserved for genuine environment gaps (missing Python module); a missing
# TRACKED fixture is a broken checkout and must assert, not skip.
SKIP_EXIT_CODE = 77


def _skip(msg):
    # "pytest is importable" != "pytest is running us". Under ctest this is a
    # plain script; pytest.skip() there is an uncaught exception (exit 1).
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg, allow_module_level=True)
    print(f"SKIP: {msg}")
    sys.exit(SKIP_EXIT_CODE)


try:
    import numpy as np
    import yaml
except ImportError as e:
    _skip(f"{e.name} not installed")


def test_fmo7_integration():
    regress = os.path.join(root, "tests", "regress.py")
    cfg_path = os.path.join(root, "tests", "configs", "benchmark_fmo7.yaml")
    ref_path = os.path.join(root, "tests", "benchmark_data", "benchmark_fmo7.h5")

    assert os.path.exists(ref_path), \
        f"reference file {ref_path} not found (tracked file — broken checkout?)"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_h5:
        out_path = tmp_h5.name
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as tmp_yaml:
        # Schema: each task block carries its own nested `output:` sub-block.
        cfg["dynamics"]["output"]["save_path"] = os.path.dirname(out_path) + "/"
        cfg["dynamics"]["output"]["filename"] = os.path.splitext(os.path.basename(out_path))[0]
        cfg["dynamics"]["output"]["use_timestamp"] = False
        yaml.dump(cfg, tmp_yaml)
        yaml_path = tmp_yaml.name

    try:
        r = subprocess.run([solver, "dynamics", yaml_path],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"solver exited {r.returncode}\n{r.stderr[-500:]}"

        r2 = subprocess.run([sys.executable, regress, "diff", ref_path, out_path, "--tol", "1e-12"],
                            capture_output=True, text=True)
        print(r2.stdout)
        assert r2.returncode == 0, "benchmark comparison failed"
    finally:
        for p in [out_path, yaml_path]:
            if os.path.exists(p):
                os.unlink(p)


def test_2des_dimer_smoke():
    """Run the dimer 2DES smoke. The HDF5 product is finite and shaped correctly,
    and carry-through GSB depends on the waiting time T."""
    try:
        import h5py
    except ImportError:
        _skip("h5py not installed")
        return

    cfg_path = os.path.join(root, "tests", "configs", "dimer_2des.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = tempfile.mkdtemp(prefix="dimer_2des_")
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w",
                                     dir=out_dir) as tmp_yaml:
        # Schema: each task block carries its own nested `output:` sub-block.
        cfg["2des"]["output"]["save_path"] = out_dir + "/"
        cfg["2des"]["output"]["filename"] = "smoke"
        cfg["2des"]["output"]["use_timestamp"] = False
        yaml.dump(cfg, tmp_yaml)
        yaml_path = tmp_yaml.name

    try:
        r = subprocess.run([solver, "2des", yaml_path],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"solver exited {r.returncode}\n{r.stderr[-800:]}"

        # Explicit (non-"auto") filename is used verbatim (no bath suffix).
        out_path = os.path.join(out_dir, "smoke.h5")
        assert os.path.exists(out_path), f"missing output {out_path}\n{r.stdout[-400:]}"

        with h5py.File(out_path, "r") as h:
            resp = h["/results/resp"][...]
            tau = h["/results/tau_fs"][...]
            tw = h["/results/tw_fs"][...]
            t = h["/results/t_fs"][...]
            sig = h["/parameters"].attrs["signals"]
            pth = h["/parameters"].attrs["pathways"]

        sig = sig.decode() if isinstance(sig, bytes) else sig
        pth = pth.decode() if isinstance(pth, bytes) else pth
        signals = sig.split(",")
        pathways = pth.split(",")

        # isotropic → 3 invariants (I1, I2, I3); shape
        # [n_component, n_signal, n_pathway, n_tau, n_tw, n_t].
        assert resp.shape == (3, len(signals), len(pathways), len(tau), len(tw), len(t)), \
            f"unexpected resp shape {resp.shape}"
        assert resp.shape == (3, 2, 3, 8, 2, 8), f"smoke grid changed: {resp.shape}"

        v = np.asarray(resp)
        assert np.isfinite(v).all(), "resp contains NaN/Inf"
        for ip, nm in enumerate(["I1", "I2", "I3"]):
            assert np.abs(v[ip]).max() > 0, f"{nm} is identically zero"

        # carry-through signature: GSB rephasing must depend on the waiting time
        # T (reset factorization would make it T-independent). Use the I1
        # invariant of the GSB rephasing component, tw[0] vs tw[1].
        si = signals.index("rephasing")
        pi = pathways.index("GSB")
        gsb_tw0 = v[0, si, pi, :, 0, :]
        gsb_tw1 = v[0, si, pi, :, 1, :]
        rel = np.abs(gsb_tw0 - gsb_tw1).max() / max(np.abs(gsb_tw0).max(), 1e-30)
        assert rel > 1e-6, \
            f"GSB rephasing is T-independent (rel diff {rel:.2e}) — carry-through broken"

        print(f"PASS: 2des dimer smoke (resp {resp.shape}, GSB T-dep rel {rel:.2e})")
    finally:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    test_fmo7_integration()
    test_2des_dimer_smoke()
    print("PASS: integration tests")
