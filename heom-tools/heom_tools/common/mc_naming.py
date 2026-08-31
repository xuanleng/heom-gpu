"""MC shard / merged-file naming — the ONE place that knows the layout.

The solver (C++ ``mc_driver.cpp::shard_stem``) writes worker shards as::

    <base>_<key>_<TS>_r<a>-<b>_rank<N>.h5

where ``<key>`` is a 16-hex parameter hash (same physics ⇒ same key). The key
is the merge/grouping identity; the timestamp ``<TS>`` and the 1-based interval
``r<a>-<b>`` are info bits only, so incremental batches (different TS/interval,
same key) fold together.

Both readers — ``las-post`` (absorption ``post/io.py``) and the standalone
``mc-disorder-merge`` — used to hand-copy this regex and the merged-name rule.
That is two (three, counting C++) sources of truth for one data structure. This
module is the single Python owner; the C++ side is cross-checked by the golden
shard-name test.
"""
from __future__ import annotations

import re

import numpy as np

__all__ = [
    "SHARD_RE",
    "is_rank_shard",
    "shard_group_key",
    "shard_group_glob",
    "merged_name",
    "coverage_1based",
]

# Strict shard name: captures the group identity (base + key). TS/interval/rank
# are matched but discarded — they do not enter the group key.
# TS is YYYYMMDD_HHMM (legacy / single-shot) or YYYYMMDD_HHMMSS (standalone
# launcher, second resolution for same-minute re-runs).
SHARD_RE = re.compile(
    r"^(?P<base>.+)_(?P<key>[0-9a-f]{16})_\d{8}_\d{4,6}_r\d+-\d+_rank\d+\.h5$")

# Loose test: ANY *_rank{N}.h5 is a worker shard (kept out of plain-file
# discovery), even if it does not parse into a group (malformed ⇒ warned later).
_RANK_SUFFIX_RE = re.compile(r"^.+_rank\d+\.h5$")


def is_rank_shard(name: str) -> bool:
    """True for a worker-shard basename (ends ``_rank{N}.h5``)."""
    return _RANK_SUFFIX_RE.match(name) is not None


def shard_group_key(name: str) -> str:
    """The group stem ``<base>_<key>`` for a shard basename (no TS/interval/rank,
    no ``.h5``). This is also the merged file's stem prefix. Raises ``ValueError``
    when ``name`` is not a parseable key-tagged shard."""
    m = SHARD_RE.match(name)
    if not m:
        raise ValueError(
            f"not a key-tagged rank shard "
            f"(<base>_<key>_<TS>_r<a>-<b>_rank{{N}}.h5): {name!r}")
    return f"{m.group('base')}_{m.group('key')}"


def shard_group_glob(group_key: str) -> str:
    """Glob (relative to the shard directory) matching every TS/interval shard of
    one group: ``<base>_<key>_*_r*-*_rank*.h5``."""
    return f"{group_key}_*_r*-*_rank*.h5"


def merged_name(group_key: str, lo: int, hi: int) -> str:
    """Merged file basename ``<base>_<key>_r<lo>-<hi>.h5``. Baking the covered
    1-based range into the name keeps successive incremental merges (r1-100,
    r1-200, …) side by side instead of overwriting the previous convergence
    point."""
    return f"{group_key}_r{lo}-{hi}.h5"


def coverage_1based(r_done) -> tuple[int, int]:
    """Covered realization range of a 0-based ``r_done`` index set, returned as
    1-based INCLUSIVE display bounds ``(lo, hi)``. Empty ⇒ ``(0, 0)``."""
    rd = np.asarray(r_done).reshape(-1)
    if rd.size == 0:
        return 0, 0
    return int(rd.min()) + 1, int(rd.max()) + 1
