"""Unit tests for the task-agnostic MC shard merge primitive.

These build tiny synthetic single-shot-format files (just /results/<ds> + a /mc
group) — no solver, no GPU — and check the count-weighted average, the SEM, the
round-trip write, and the three merge guards.
"""
import numpy as np
import h5py
import pytest

from heom_tools.common.mc_merge import (
    merge_mc_shards, write_merged_spectrum, combine_replicas,
    convergence_status)

DS = "rho_xyz"


def _make_shard(path, mean, count, sum_obs_sq, r_done,
                identity="iid_gauss@v1|seed=1|sigma=42.0", fingerprint="cfg#abc",
                precision=None):
    mean = np.asarray(mean, dtype=np.complex128)
    with h5py.File(path, "w") as f:
        # precision_mode is a ROOT attr (not /mc): a build choice, not physics.
        # None => omit it, mimicking legacy pre-mixed-precision shards.
        if precision is not None:
            f.attrs["precision_mode"] = precision
        res = f.create_group("results")
        res.create_dataset(DS, data=mean)          # h5py stores complex natively
        mc = f.create_group("mc")
        mc.create_dataset("count", data=np.int64(count))
        mc.create_dataset("sum_obs_sq",
                          data=np.asarray(sum_obs_sq, dtype=np.float64))
        mc.attrs["r_done"] = np.asarray(r_done, dtype=np.int64)
        mc.attrs["config_fingerprint"] = fingerprint
        mc.attrs["disorder_identity"] = identity
    return str(path)


def test_weighted_average(tmp_path):
    shape = (2, 2)
    a = _make_shard(tmp_path / "a_rank1.h5", np.full(shape, 2.0), 2,
                    np.full(shape, 8.0), [0, 1])
    b = _make_shard(tmp_path / "a_rank2.h5", np.full(shape, 4.0), 6,
                    np.full(shape, 96.0), [2, 3, 4, 5, 6, 7])
    m = merge_mc_shards([a, b], DS)
    # (2*2 + 4*6) / 8 = 28/8 = 3.5, element-wise.
    assert m["count"] == 8
    assert np.allclose(m["mean"], 3.5)
    assert m["r_done"].tolist() == list(range(8))
    assert m["n_shards"] == 2
    # var = ΣΣ|O|²/N - |mean|² = (8+96)/8 - 3.5² = 13 - 12.25 = 0.75; sem=sqrt(0.75/8)
    assert np.allclose(m["sem"], np.sqrt(0.75 / 8))


def test_roundtrip_write(tmp_path):
    shape = (3,)
    a = _make_shard(tmp_path / "x_rank1.h5", np.full(shape, 1.0), 1,
                    np.full(shape, 1.0), [0])
    b = _make_shard(tmp_path / "x_rank2.h5", np.full(shape, 3.0), 3,
                    np.full(shape, 27.0), [1, 2, 3])
    m = merge_mc_shards([a, b], DS)
    out = write_merged_spectrum(a, str(tmp_path / "x.h5"), m, DS)
    with h5py.File(out, "r") as f:
        got = f[f"/results/{DS}"][...]
        assert np.allclose(got, 2.5)   # (1*1 + 3*3)/4 = 10/4
        assert f["/results/rho_xyz"].attrs["disorder_averaged"] == 1
        assert int(f["/mc/count"][()]) == 4
        assert np.asarray(f["/mc"].attrs["r_done"]).shape[0] == 4
        assert "sem" in f["/mc"]


def test_r_done_overlap_raises(tmp_path):
    a = _make_shard(tmp_path / "a_rank1.h5", [1.0], 1, [1.0], [0])
    b = _make_shard(tmp_path / "a_rank2.h5", [1.0], 1, [1.0], [0])  # dup r=0
    with pytest.raises(ValueError, match="double counting"):
        merge_mc_shards([a, b], DS)


def test_identity_mismatch_raises(tmp_path):
    a = _make_shard(tmp_path / "a_rank1.h5", [1.0], 1, [1.0], [0])
    b = _make_shard(tmp_path / "a_rank2.h5", [1.0], 1, [1.0], [1],
                    identity="iid_gauss@v1|seed=999|sigma=42.0")
    with pytest.raises(ValueError, match="disorder_identity mismatch"):
        merge_mc_shards([a, b], DS)


def test_fingerprint_mismatch_raises(tmp_path):
    a = _make_shard(tmp_path / "a_rank1.h5", [1.0], 1, [1.0], [0])
    b = _make_shard(tmp_path / "a_rank2.h5", [1.0], 1, [1.0], [1],
                    fingerprint="cfg#different")
    with pytest.raises(ValueError, match="config_fingerprint mismatch"):
        merge_mc_shards([a, b], DS)


def test_precision_mode_mismatch_raises(tmp_path):
    # FP64 and mixed-FP32 shards are different numerical pipelines — averaging
    # them silently would be a subtle correctness bug (guard #4).
    a = _make_shard(tmp_path / "a_rank1.h5", [1.0], 1, [1.0], [0],
                    precision="fp64")
    b = _make_shard(tmp_path / "a_rank2.h5", [1.0], 1, [1.0], [1],
                    precision="mixed_fp32")
    with pytest.raises(ValueError, match="precision_mode mismatch"):
        merge_mc_shards([a, b], DS)


def test_precision_mode_legacy_missing_treated_as_fp64(tmp_path):
    # Pre-mixed-precision shards have no precision_mode attr; they were all FP64,
    # so a missing attr must merge cleanly with an explicit "fp64" shard.
    a = _make_shard(tmp_path / "a_rank1.h5", [1.0], 1, [1.0], [0])  # no attr
    b = _make_shard(tmp_path / "a_rank2.h5", [3.0], 3, [27.0], [1, 2, 3],
                    precision="fp64")
    m = merge_mc_shards([a, b], DS)
    assert m["count"] == 4
    assert m["precision_mode"] == "fp64"


def test_missing_mc_group_raises(tmp_path):
    path = str(tmp_path / "plain.h5")
    with h5py.File(path, "w") as f:
        f.create_group("results").create_dataset(DS, data=np.zeros((2,)))
    with pytest.raises(ValueError, match="no /mc group"):
        merge_mc_shards([path], DS)


# --------------------------------------------------------------------------
# Quadrature merge: sum /mc/sum_obs, no count divide, no SEM.
# --------------------------------------------------------------------------

def _make_quad_shard(path, sum_obs, count, r_done,
                     identity="sparse_gh@v1|level=2|sigma=42.0",
                     fingerprint="cfg#abc", precision=None):
    """A quadrature shard: /mc/sum_obs (complex, the payload) + estimator attr.
    /results/<ds> holds the same partial weighted sum as a placeholder."""
    sum_obs = np.asarray(sum_obs, dtype=np.complex128)
    with h5py.File(path, "w") as f:
        if precision is not None:
            f.attrs["precision_mode"] = precision
        f.create_group("results").create_dataset(DS, data=sum_obs)
        mc = f.create_group("mc")
        mc.create_dataset("count", data=np.int64(count))
        mc.create_dataset("sum_obs", data=sum_obs)
        mc.attrs["estimator"] = "quadrature"
        mc.attrs["r_done"] = np.asarray(r_done, dtype=np.int64)
        mc.attrs["config_fingerprint"] = fingerprint
        mc.attrs["disorder_identity"] = identity
    return str(path)


def test_quadrature_merge_sums_directly(tmp_path):
    shape = (2, 2)
    a = _make_quad_shard(tmp_path / "q_rank1.h5", np.full(shape, 0.3 + 0j), 3,
                         [0, 1, 2])
    b = _make_quad_shard(tmp_path / "q_rank2.h5", np.full(shape, 0.7 + 0j), 5,
                         [3, 4, 5, 6, 7])
    m = merge_mc_shards([a, b], DS)
    assert m["estimator"] == "quadrature"
    assert np.allclose(m["mean"], 1.0)          # 0.3 + 0.7, NO count divide
    assert np.allclose(m["sum_obs"], 1.0)
    assert "sem" not in m and "sum_obs_sq" not in m
    assert m["count"] == 8                       # bookkeeping only
    assert m["r_done"].tolist() == list(range(8))
    assert m["n_shards"] == 2


def test_quadrature_roundtrip_write(tmp_path):
    shape = (3,)
    a = _make_quad_shard(tmp_path / "q_rank1.h5", np.full(shape, 0.25 + 0j), 2,
                         [0, 1])
    b = _make_quad_shard(tmp_path / "q_rank2.h5", np.full(shape, 0.75 + 0j), 2,
                         [2, 3])
    m = merge_mc_shards([a, b], DS)
    out = write_merged_spectrum(a, str(tmp_path / "q.h5"), m, DS)
    with h5py.File(out, "r") as f:
        assert np.allclose(f[f"/results/{DS}"][...], 1.0)   # normalised placeholder
        assert f[f"/results/{DS}"].attrs["disorder_averaged"] == 1
        assert np.allclose(f["/mc/sum_obs"][...], 1.0)
        assert "sem" not in f["/mc"]                        # no SEM for quadrature


def test_estimator_mismatch_raises(tmp_path):
    # A mean shard (no estimator attr => "mean") must never merge with a
    # quadrature shard, even when identity/fingerprint/precision all match.
    a = _make_shard(tmp_path / "m_rank1.h5", [1.0], 1, [1.0], [0],
                    identity="shared|sigma=1")
    b = _make_quad_shard(tmp_path / "q_rank2.h5", [1.0 + 0j], 1, [1],
                         identity="shared|sigma=1")
    with pytest.raises(ValueError, match="estimator mismatch"):
        merge_mc_shards([a, b], DS)


# --------------------------------------------------------------------------
# RQMC replica combine: across-replica standard error, not a 1/sqrt(N) SEM.
# Each input is a MERGED replica mean (estimator=mean); combine_replicas gives
# the rqmc estimator: mean-of-means + per-ω across-replica SE.
# --------------------------------------------------------------------------

def _make_replica(path, mean, count=256, fingerprint="cfg#abc", precision=None,
                  estimator="mean"):
    """A merged replica mean file (one scramble): /results + /mc/count/estimator.
    sum_obs_sq is included to prove combine drops it (it's meaningless for RQMC)."""
    mean = np.asarray(mean, dtype=np.complex128)
    with h5py.File(path, "w") as f:
        if precision is not None:
            f.attrs["precision_mode"] = precision
        f.create_group("results").create_dataset(DS, data=mean)
        mc = f.create_group("mc")
        mc.create_dataset("count", data=np.int64(count))
        mc.create_dataset("sum_obs_sq",
                          data=np.abs(mean) ** 2 * count)
        mc.attrs["estimator"] = estimator
        mc.attrs["config_fingerprint"] = fingerprint
        mc.attrs["disorder_identity"] = "sobol_gauss@v1|seed=7|sigma=42.0"
    return str(path)


def test_combine_replicas_meanofmeans_and_se(tmp_path):
    # R=2, O0=1, O1=3 -> mean-of-means = 2; per-bin dev=1 each ->
    # SE = sqrt((1+1)/(2*(2-1))) = 1.
    a = _make_replica(tmp_path / "r0.h5", np.full((2, 2), 1.0))
    b = _make_replica(tmp_path / "r1.h5", np.full((2, 2), 3.0))
    c = combine_replicas([a, b], DS)
    assert c["estimator"] == "rqmc"
    assert c["n_replica"] == 2
    assert c["count"] == 512
    assert np.allclose(c["mean"], 2.0)
    assert np.allclose(c["replica_se"], 1.0)
    assert np.isclose(c["se_l2"], np.sqrt(4 * 1.0 ** 2))  # 4 bins, each se=1


def test_combine_replicas_write_drops_sem(tmp_path):
    a = _make_replica(tmp_path / "r0.h5", np.full((3,), 1.0 + 0j))
    b = _make_replica(tmp_path / "r1.h5", np.full((3,), 3.0 + 0j))
    c = combine_replicas([a, b], DS)
    out = write_merged_spectrum(a, str(tmp_path / "rqmc.h5"), c, DS)
    with h5py.File(out, "r") as f:
        assert np.allclose(f[f"/results/{DS}"][...], 2.0)
        assert f[f"/results/{DS}"].attrs["disorder_averaged"] == 1
        mc = f["/mc"]
        est = mc.attrs["estimator"]
        assert (est.decode() if isinstance(est, bytes) else est) == "rqmc"
        assert "replica_se" in mc
        assert "sem" not in mc and "sum_obs_sq" not in mc  # √N error is forbidden
        assert int(mc.attrs["n_replica"]) == 2


def test_combine_replicas_needs_two(tmp_path):
    a = _make_replica(tmp_path / "r0.h5", np.full((2,), 1.0))
    with pytest.raises(ValueError, match="needs >= 2"):
        combine_replicas([a], DS)


def test_combine_replicas_identical_means_raises(tmp_path):
    # Same replica passed twice: SE would be a fake 0 -> must be caught.
    a = _make_replica(tmp_path / "r0.h5", np.full((2,), 2.0))
    b = _make_replica(tmp_path / "r1.h5", np.full((2,), 2.0))
    with pytest.raises(ValueError, match="identical means"):
        combine_replicas([a, b], DS)


def test_combine_replicas_fingerprint_mismatch_raises(tmp_path):
    a = _make_replica(tmp_path / "r0.h5", np.full((2,), 1.0))
    b = _make_replica(tmp_path / "r1.h5", np.full((2,), 3.0),
                      fingerprint="cfg#other")
    with pytest.raises(ValueError, match="config_fingerprint mismatch"):
        combine_replicas([a, b], DS)


def test_combine_replicas_precision_mismatch_raises(tmp_path):
    a = _make_replica(tmp_path / "r0.h5", np.full((2,), 1.0), precision="fp64")
    b = _make_replica(tmp_path / "r1.h5", np.full((2,), 3.0),
                      precision="mixed_fp32")
    with pytest.raises(ValueError, match="precision_mode mismatch"):
        combine_replicas([a, b], DS)


def test_combine_replicas_rejects_non_mean_input(tmp_path):
    a = _make_replica(tmp_path / "r0.h5", np.full((2,), 1.0))
    b = _make_replica(tmp_path / "r1.h5", np.full((2,), 3.0),
                      estimator="quadrature")
    with pytest.raises(ValueError, match="expects 'mean'"):
        combine_replicas([a, b], DS)


def test_combine_replicas_unequal_counts_warns(tmp_path):
    a = _make_replica(tmp_path / "r0.h5", np.full((2,), 1.0), count=256)
    b = _make_replica(tmp_path / "r1.h5", np.full((2,), 3.0), count=128)
    with pytest.warns(UserWarning, match="node counts differ"):
        c = combine_replicas([a, b], DS)
    assert np.allclose(c["mean"], 2.0)   # still unweighted mean-of-means


# --------------------------------------------------------------------------
# convergence_status — verdict + evals-to-tol under the sampler law.
# --------------------------------------------------------------------------

def test_convergence_status_mean_sqrtN_projection():
    # mean=[3,4] -> ‖μ‖=5; se_l2=0.5 -> rel=0.1 at N=100.
    # N*(tol=0.01) under 1/√N = N·(rel/tol)² = 100·100 = 10000.
    s = convergence_status([3.0, 4.0], se_l2=0.5, count=100,
                           estimator="mean", tol=0.01)
    assert np.isclose(s["rel_err"], 0.1)
    assert not s["converged"]
    assert s["rate"] == -0.5
    assert s["n_star"] == 10000
    assert s["remaining"] == 9900


def test_convergence_status_converged_zero_remaining():
    s = convergence_status([3.0, 4.0], se_l2=0.02, count=1000,
                           estimator="mean", tol=0.01)  # rel=0.004 <= 0.01
    assert s["converged"]
    assert s["n_star"] == 1000
    assert s["remaining"] == 0


def test_convergence_status_rqmc_uses_empirical_slope():
    # rel=0.1 at N=100, tol=0.01, slope=-1 (ideal RQMC) -> N* = 100·(tol/rel)^(1/-1)
    # = 100·(0.1)^(-1) = 1000  (vs iid's 10000 -> RQMC needs far fewer).
    s = convergence_status([3.0, 4.0], se_l2=0.5, count=100,
                           estimator="rqmc", tol=0.01, slope=-1.0)
    assert s["rate"] == -1.0
    assert not s["conservative"]
    assert s["n_star"] == 1000


def test_convergence_status_rqmc_without_slope_is_conservative():
    # No slope -> falls back to -0.5 (upper bound); flags conservative.
    s = convergence_status([3.0, 4.0], se_l2=0.5, count=100,
                           estimator="rqmc", tol=0.01)
    assert s["rate"] == -0.5
    assert s["conservative"]
    assert s["n_star"] == 10000


def test_convergence_status_quadrature_rejected():
    with pytest.raises(ValueError, match="undefined for quadrature"):
        convergence_status([1.0], se_l2=0.1, count=10,
                           estimator="quadrature", tol=0.01)


def test_convergence_status_zero_norm_rejected():
    with pytest.raises(ValueError, match="zero L"):
        convergence_status([0.0, 0.0], se_l2=0.1, count=10,
                           estimator="mean", tol=0.01)
