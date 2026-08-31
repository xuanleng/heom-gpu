"""Unit tests for the single-source-of-truth MC naming module.

The shard/merged name grammar used to be hand-copied into las-post (io.py) and
mc-disorder-merge (merge.py). These tests pin the one owner so a drift there is
caught here, not in a silently-empty merge.
"""
import pytest

from heom_tools.common import mc_naming

_GOOD = "fmo_absorption_0123456789abcdef_20260706_1122_r201-300_rank3.h5"


def test_is_rank_shard_loose_suffix():
    assert mc_naming.is_rank_shard(_GOOD)
    assert mc_naming.is_rank_shard("anything_weird_rank1.h5")
    assert not mc_naming.is_rank_shard("fmo_absorption_0123456789abcdef_r1-8.h5")
    assert not mc_naming.is_rank_shard("plain.h5")


def test_group_key_strips_ts_interval_rank():
    assert (mc_naming.shard_group_key(_GOOD)
            == "fmo_absorption_0123456789abcdef")


def test_group_key_ignores_ts_and_interval():
    # Same key, different TS + interval ⇒ SAME group (incremental batches merge).
    a = "fmo_absorption_0123456789abcdef_20260706_1122_r1-100_rank1.h5"
    b = "fmo_absorption_0123456789abcdef_20260706_1200_r101-200_rank2.h5"
    assert mc_naming.shard_group_key(a) == mc_naming.shard_group_key(b)


def test_group_key_rejects_non_shard():
    with pytest.raises(ValueError, match="key-tagged rank shard"):
        mc_naming.shard_group_key("plain.h5")
    # Missing the 16-hex key (a raw single-shot file) is NOT a shard.
    with pytest.raises(ValueError):
        mc_naming.shard_group_key("fmo_absorption_20260706_1122_rank1.h5")


def test_group_glob_matches_all_ts_intervals():
    from fnmatch import fnmatch
    key = "fmo_absorption_0123456789abcdef"
    pat = mc_naming.shard_group_glob(key)
    assert fnmatch(_GOOD, pat)
    assert fnmatch(f"{key}_20260706_1200_r1-100_rank1.h5", pat)
    assert not fnmatch(f"{key}_r1-8.h5", pat)   # merged product, not a shard


def test_merged_name_bakes_coverage():
    assert (mc_naming.merged_name("fmo_absorption_0123456789abcdef", 1, 200)
            == "fmo_absorption_0123456789abcdef_r1-200.h5")


def test_coverage_1based_from_zero_based():
    assert mc_naming.coverage_1based([0, 1, 2, 7]) == (1, 8)
    assert mc_naming.coverage_1based([200, 201, 299]) == (201, 300)
    assert mc_naming.coverage_1based([]) == (0, 0)
