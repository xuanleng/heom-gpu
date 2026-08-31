"""twodes-post MC rank-shard folding: merge <base>_<key>_..._rank*.h5 before post."""
import numpy as np
import h5py
import pytest

from heom_tools.twodes.post.io import (
    ensure_base_merged,
    fold_rank_groups,
    is_rank_shard_filename,
    is_solver_2des_h5,
)

_KEY = "a469ee2f2eb48cd6"


def _name(base, ts, r_begin, r_end, rank, key=_KEY):
    return f"{base}_{key}_{ts}_r{r_begin}-{r_end}_rank{rank}.h5"


def _shard(path, mean, count, r_done, ident="sobol_gauss@v1|repl=0|seed=1|sigma=42.0"):
    mean = np.asarray(mean, dtype=np.complex128)
    with h5py.File(path, "w") as f:
        res = f.create_group("results")
        res.create_dataset("resp", data=mean)
        res.create_dataset("tau_fs", data=np.linspace(0, 12, mean.shape[3]))
        res.create_dataset("tw_fs", data=np.linspace(0, 500, mean.shape[4]))
        res.create_dataset("t_fs", data=np.linspace(0, 20, mean.shape[5]))
        par = f.create_group("parameters")
        par.attrs["signals"] = "rephasing,nonrephasing"
        par.attrs["pathways"] = "SE,GSB,ESA"
        mc = f.create_group("mc")
        mc.create_dataset("count", data=np.int64(count))
        mc.create_dataset(
            "sum_obs_sq",
            data=np.abs(mean).astype(np.float64) ** 2 * count,
        )
        mc.attrs["r_done"] = np.asarray(r_done, dtype=np.int64)
        mc.attrs["config_fingerprint"] = "cfg#1"
        mc.attrs["disorder_identity"] = ident
    return str(path)


def test_is_rank_shard_filename():
    assert is_rank_shard_filename(_name("fmo_2des", "20260715_1713", 1, 64, 1))
    assert not is_rank_shard_filename(f"fmo_2des_{_KEY}_r1-64.h5")


def test_ensure_base_merged_creates_weighted_base(tmp_path):
    base, ts = "fmo_2des", "20260715_1713"
    shape = (3, 2, 3, 4, 1, 4)
    n1 = _name(base, ts, 1, 64, 1)
    n2 = _name(base, ts, 1, 64, 2)
    _shard(tmp_path / n1, np.full(shape, 2.0), 2, [0, 1])
    _shard(tmp_path / n2, np.full(shape, 4.0), 6, [2, 3, 4, 5, 6, 7])
    out = ensure_base_merged(str(tmp_path / n1))
    assert out == str(tmp_path / f"{base}_{_KEY}_r1-8.h5")
    with h5py.File(out) as f:
        assert np.allclose(f["/results/resp"][...], 3.5)
        assert int(f["/mc/count"][()]) == 8
        assert is_solver_2des_h5(out)
    assert ensure_base_merged(str(tmp_path / n1)) == out


def test_fold_rank_groups_keeps_shards_out_of_auto_targets(tmp_path):
    base = "fmo_2des"
    shape = (3, 2, 3, 4, 1, 4)
    _shard(tmp_path / _name(base, "20260715_1713", 1, 64, 1), np.full(shape, 2.0), 2, [0, 1])
    _shard(tmp_path / _name(base, "20260715_1713", 1, 64, 2), np.full(shape, 4.0), 2, [2, 3])
    fold_rank_groups(str(tmp_path))
    merged = tmp_path / f"{base}_{_KEY}_r1-4.h5"
    assert merged.is_file()
    pending = [
        fn for fn in sorted(tmp_path.glob("*.h5"))
        if not fn.name.endswith("_post.h5") and not is_rank_shard_filename(fn.name)
    ]
    assert [p.name for p in pending] == [merged.name]


def test_process_merges_rank_shard_before_post(tmp_path):
    import heom_tools.twodes.post as tp
    from heom_tools.twodes.post import PostConfig

    base, ts = "fmo_2des", "20260715_1713"
    shape = (3, 2, 3, 8, 1, 8)
    n1 = _name(base, ts, 1, 64, 1)
    n2 = _name(base, ts, 1, 64, 2)
    _shard(tmp_path / n1, np.zeros(shape, dtype=np.complex128), 1, [0])
    _shard(tmp_path / n2, np.zeros(shape, dtype=np.complex128), 1, [1])
    (tmp_path / "twodes_post.yaml").write_text(
        "pol: ap\nplot: false\nwrite_h5: true\nh5_spectra: [2des]\n",
    )
    cfg = PostConfig.defaults()
    cfg.out = str(tmp_path)
    cfg.plot = False
    cfg.write_h5 = True
    cfg.h5_spectra = ["2des"]
    assert tp.process(str(tmp_path / n1), cfg, show_config=False) == 0
    post = tmp_path / f"{base}_{_KEY}_r1-2_ap_post.h5"
    assert post.is_file()
    assert not (tmp_path / f"{base}_{ts}_r1-64_rank1_ap_post.h5").exists()
