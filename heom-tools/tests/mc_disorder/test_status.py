"""mc-disorder-status thin-shell tests (numpy + h5py only; no solver / GPU).

The convergence math is unit-tested in heom_tools.common's test_mc_merge.py
(convergence_status); here we lock the CLI's EXIT-CODE contract — the composable
stop/continue signal a driver loops on: 0=converged, 3=not, 2=undefined.
"""
import numpy as np
import h5py
import pytest

from heom_tools.mc_disorder import status as st

_KEY = "0123456789abcdef"


def _shard(path, mean, count, ssq, r_done):
    with h5py.File(path, "w") as f:
        f.create_group("results").create_dataset(
            "rho_xyz", data=np.asarray(mean, np.complex128))
        mc = f.create_group("mc")
        mc.create_dataset("count", data=np.int64(count))
        mc.create_dataset("sum_obs_sq", data=np.asarray(ssq, np.float64))
        mc.attrs["r_done"] = np.asarray(r_done, np.int64)
        mc.attrs["config_fingerprint"] = "cfg#abc"
        mc.attrs["disorder_identity"] = "iid_gauss@v1|seed=1|sigma=42.0"
    return str(path)


def _merged(path, mean, count, se_l2, estimator="mean", n_replica=None):
    with h5py.File(path, "w") as f:
        f.create_group("results").create_dataset(
            "rho_xyz", data=np.asarray(mean, np.complex128))
        mc = f.create_group("mc")
        mc.create_dataset("count", data=np.int64(count))
        mc.attrs["estimator"] = estimator
        mc.attrs["se_l2"] = float(se_l2)
        if n_replica is not None:
            mc.attrs["n_replica"] = np.int64(n_replica)
    return str(path)


def test_exit3_when_not_converged_from_shards(tmp_path):
    # mean=[3,4] ‖μ‖=5; huge variance -> rel error >> tol -> exit 3.
    name = f"fmo_{_KEY}_20260101_1200_r1-4_rank1.h5"
    _shard(tmp_path / name, [3, 4], 4, [200.0, 300.0], [0, 1, 2, 3])
    assert st.main([str(tmp_path), "--tol", "0.01"]) == 3


def test_exit0_when_converged_merged_mean(tmp_path):
    f = _merged(tmp_path / "m.h5", [3, 4], 4000, se_l2=0.02)  # rel=0.004
    assert st.main([f, "--tol", "0.01"]) == 0


def test_exit0_rqmc_with_slope(tmp_path):
    f = _merged(tmp_path / "rq.h5", [3, 4], 1024, se_l2=0.0141,
                estimator="rqmc", n_replica=4)   # rel≈0.0028
    assert st.main([f, "--tol", "0.01", "--slope", "-0.91"]) == 0


def test_exit3_rqmc_conservative_no_slope(tmp_path):
    f = _merged(tmp_path / "rq.h5", [3, 4], 1024, se_l2=0.5,
                estimator="rqmc", n_replica=4)   # rel=0.1 > tol
    assert st.main([f, "--tol", "0.01"]) == 3


def test_exit2_quadrature_undefined(tmp_path):
    with h5py.File(tmp_path / "q.h5", "w") as f:
        f.create_group("results").create_dataset(
            "rho_xyz", data=np.array([1.0 + 0j, 2.0]))
        mc = f.create_group("mc")
        mc.create_dataset("count", data=np.int64(10))
        mc.attrs["estimator"] = "quadrature"
    assert st.main([str(tmp_path / "q.h5"), "--tol", "0.01"]) == 2


def test_error_on_empty_inputs(tmp_path):
    with pytest.raises(SystemExit):
        st.main([str(tmp_path), "--tol", "0.01"])   # empty dir -> p.error


def test_multiple_all_converged_exit0(tmp_path):
    a = _merged(tmp_path / "a.h5", [3, 4], 4000, se_l2=0.02)
    b = _merged(tmp_path / "b.h5", [3, 4], 4000, se_l2=0.01)
    assert st.main([a, b, "--tol", "0.01"]) == 0
