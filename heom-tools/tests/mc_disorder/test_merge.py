"""mc-disorder-merge thin-shell tests (numpy + h5py only; no solver / GPU).

The numeric merge + guards live in heom_tools.common's test_mc_merge.py; here we
only check the CLI shell: shard discovery, grouping by the parameter KEY (across
timestamps/intervals), and that a group folds into a single-shot-format
<base>_<key>.h5 with the count-weighted mean.
"""
import os

import h5py
import numpy as np
import pytest

from heom_tools.mc_disorder import merge

_KEY = "0123456789abcdef"


def _name(base, ts, r_begin, r_end, rank, key=_KEY):
    return f"{base}_{key}_{ts}_r{r_begin}-{r_end}_rank{rank}.h5"


def _shard(path, mean, count, r_done, dataset="rho_xyz",
           ident="iid_gauss@v1|seed=1|sigma=42.0"):
    """Write a single-shot-format shard (/results/<ds> mean + /mc provenance)."""
    mean = np.asarray(mean, dtype=np.complex128)
    with h5py.File(path, "w") as f:
        f.create_group("results").create_dataset(dataset, data=mean)
        mc = f.create_group("mc")
        mc.create_dataset("count", data=np.int64(count))
        mc.create_dataset("sum_obs_sq",
                          data=np.abs(mean).astype(np.float64) ** 2 * count)
        mc.attrs["r_done"] = np.asarray(r_done, dtype=np.int64)
        mc.attrs["config_fingerprint"] = "cfg#abc"
        mc.attrs["disorder_identity"] = ident
    return str(path)


def test_find_shards_globs_rank_files(tmp_path):
    _shard(tmp_path / _name("fmo_absorption", "20260706_1122", 1, 2, 1), [1.0], 1, [0])
    _shard(tmp_path / _name("fmo_absorption", "20260706_1122", 1, 2, 2), [3.0], 3, [1, 2, 3])
    (tmp_path / "note.txt").write_text("ignore me")
    found = merge.find_shards([str(tmp_path)])
    assert len(found) == 2
    assert all(p.endswith("_rank1.h5") or p.endswith("_rank2.h5") for p in found)


def test_group_by_base_rejects_non_shard(tmp_path):
    plain = str(tmp_path / f"fmo_absorption_{_KEY}.h5")  # merged file, no _rank
    with h5py.File(plain, "w") as f:
        f.create_group("results").create_dataset("rho_xyz", data=np.zeros(1))
    with pytest.raises(ValueError, match="not a key-tagged rank shard"):
        merge.group_by_base([plain])


def test_end_to_end_group_merge_across_intervals(tmp_path):
    """Two incremental batches (different TS + interval, same key) fold into one
    <base>_<key>.h5 with the count-weighted mean and r_done union."""
    base = "fmo_absorption"
    _shard(tmp_path / _name(base, "20260706_1122", 1, 4, 1), np.full((2, 2), 2.0), 2, [0, 1])
    _shard(tmp_path / _name(base, "20260706_1122", 1, 4, 2), np.full((2, 2), 4.0), 2, [2, 3])
    _shard(tmp_path / _name(base, "20260706_1500", 5, 8, 1), np.full((2, 2), 6.0), 2, [4, 5])
    _shard(tmp_path / _name(base, "20260706_1500", 5, 8, 2), np.full((2, 2), 8.0), 2, [6, 7])
    rc = merge.main([str(tmp_path)])
    assert rc == 0
    out = tmp_path / f"{base}_{_KEY}_r1-8.h5"   # r_done 0..7 -> r1-8 baked in name
    assert out.is_file()
    with h5py.File(out) as f:
        assert np.allclose(f["/results/rho_xyz"][...], 5.0)   # (4+8+12+16)/8
        assert int(f["/mc/count"][()]) == 8
        r_done = np.asarray(f["/mc"].attrs["r_done"])
        assert sorted(r_done.tolist()) == list(range(8))


def test_out_rejected_for_multiple_groups(tmp_path):
    _shard(tmp_path / _name("a", "20260706_1122", 1, 1, 1, key="a" * 16), [1.0], 1, [0])
    _shard(tmp_path / _name("b", "20260706_1122", 1, 1, 1, key="b" * 16), [1.0], 1, [0])
    with pytest.raises(SystemExit):
        merge.main([str(tmp_path), "--out", str(tmp_path / "x.h5")])
