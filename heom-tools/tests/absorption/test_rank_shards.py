"""las-post MC rank-shard grouping: fold <base>_<key>_<TS>_r<a>-<b>_rank*.h5 ->
<base>_<key>.h5 (grouping by the 16-hex parameter KEY, across timestamps and
intervals) and keep shards out of flat discovery. Uses synthetic
single-shot-format + /mc files (no solver / GPU)."""
import numpy as np
import h5py

from heom_tools.absorption.post.io import (
    ensure_base_merged,
    fold_rank_groups,
    is_rank_shard_filename,
    discover_unprocessed_absorption_h5_flat,
)

_KEY = "0123456789abcdef"


def _name(base, ts, r_begin, r_end, rank, key=_KEY):
    return f"{base}_{key}_{ts}_r{r_begin}-{r_end}_rank{rank}.h5"


def _shard(path, mean, count, r_done, ident="iid_gauss@v1|seed=1|sigma=42.0"):
    mean = np.asarray(mean, dtype=np.complex128)
    with h5py.File(path, "w") as f:
        f.create_group("results").create_dataset("rho_xyz", data=mean)
        mc = f.create_group("mc")
        mc.create_dataset("count", data=np.int64(count))
        mc.create_dataset("sum_obs_sq",
                          data=np.abs(mean).astype(np.float64) ** 2 * count)
        mc.attrs["r_done"] = np.asarray(r_done, dtype=np.int64)
        mc.attrs["config_fingerprint"] = "cfg#1"
        mc.attrs["disorder_identity"] = ident
    return str(path)


def test_is_rank_shard_filename():
    assert is_rank_shard_filename(_name("fmo_absorption", "20260706_1122", 1, 200, 1))
    assert is_rank_shard_filename("/a/b/x_rank12.h5")
    # Merged file (no _rank) and processed sibling are NOT shards.
    assert not is_rank_shard_filename("fmo_absorption_0123456789abcdef.h5")
    assert not is_rank_shard_filename("fmo_absorption_spectrum.h5")


def test_ensure_base_merged_creates_weighted_base(tmp_path):
    base, ts = "fmo_absorption", "20260706_1122"
    n1 = _name(base, ts, 1, 8, 1)
    n2 = _name(base, ts, 1, 8, 2)
    _shard(tmp_path / n1, np.full((2, 2), 2.0), 2, [0, 1])
    _shard(tmp_path / n2, np.full((2, 2), 4.0), 6, [2, 3, 4, 5, 6, 7])
    out = ensure_base_merged(str(tmp_path / n1))
    # Covered range (1-based) is baked into the merged name: r_done 0..7 -> r1-8.
    assert out == str(tmp_path / f"{base}_{_KEY}_r1-8.h5")
    with h5py.File(out) as f:
        assert np.allclose(f["/results/rho_xyz"][...], 3.5)  # (2*2+4*6)/8
        assert int(f["/mc/count"][()]) == 8
        assert f["/results/rho_xyz"].attrs["disorder_averaged"] == 1
    # Idempotent: same coverage returns the same file without re-merging.
    assert ensure_base_merged(str(tmp_path / n1)) == out


def test_fold_across_timestamps_and_intervals_same_key(tmp_path):
    """The incremental case: batch [1,4] at one TS and batch [5,8] at a LATER TS,
    same parameter key, fold into ONE <base>_<key>.h5 with the r_done union."""
    base = "fmo_absorption"
    # Batch 1: TS ...1122, range [1,4] (r=0..3), two workers.
    _shard(tmp_path / _name(base, "20260706_1122", 1, 4, 1),
           np.full((2, 2), 2.0), 2, [0, 1])
    _shard(tmp_path / _name(base, "20260706_1122", 1, 4, 2),
           np.full((2, 2), 4.0), 2, [2, 3])
    # Batch 2: DIFFERENT TS ...1500, range [5,8] (r=4..7), two workers.
    _shard(tmp_path / _name(base, "20260706_1500", 5, 8, 1),
           np.full((2, 2), 6.0), 2, [4, 5])
    _shard(tmp_path / _name(base, "20260706_1500", 5, 8, 2),
           np.full((2, 2), 8.0), 2, [6, 7])

    fold_rank_groups(str(tmp_path))
    merged = tmp_path / f"{base}_{_KEY}_r1-8.h5"                   # r_done 0..7 -> r1-8
    assert merged.is_file()
    with h5py.File(merged) as f:
        assert int(f["/mc/count"][()]) == 8                       # 2+2+2+2
        assert np.allclose(f["/results/rho_xyz"][...], 5.0)       # (4+8+12+16)/8
        r_done = np.asarray(f["/mc"].attrs["r_done"])
        assert sorted(r_done.tolist()) == list(range(8))          # union 0..7

    # Shards must never appear as standalone processing targets.
    pending = discover_unprocessed_absorption_h5_flat(str(tmp_path))
    assert all("_rank" not in p for p in pending)


def test_incremental_remerge_keeps_history(tmp_path):
    """A wider incremental coverage writes a NEW merged file (r1-4 -> r1-8),
    leaving the earlier convergence point in place instead of overwriting it."""
    base = "fmo_absorption"
    _shard(tmp_path / _name(base, "20260706_1122", 1, 4, 1), np.full((2, 2), 2.0), 2, [0, 1])
    _shard(tmp_path / _name(base, "20260706_1122", 1, 4, 2), np.full((2, 2), 4.0), 2, [2, 3])
    fold_rank_groups(str(tmp_path))
    assert (tmp_path / f"{base}_{_KEY}_r1-4.h5").is_file()

    # Second batch arrives; folding now covers 1..8 and must NOT clobber r1-4.
    _shard(tmp_path / _name(base, "20260706_1500", 5, 8, 1), np.full((2, 2), 6.0), 2, [4, 5])
    _shard(tmp_path / _name(base, "20260706_1500", 5, 8, 2), np.full((2, 2), 8.0), 2, [6, 7])
    fold_rank_groups(str(tmp_path))
    assert (tmp_path / f"{base}_{_KEY}_r1-4.h5").is_file()        # history preserved
    assert (tmp_path / f"{base}_{_KEY}_r1-8.h5").is_file()        # new coverage
    with h5py.File(tmp_path / f"{base}_{_KEY}_r1-8.h5") as f:
        assert int(f["/mc/count"][()]) == 8
