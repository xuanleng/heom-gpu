"""Config identity hashing — the Python mirror of the solver's C++ hashing.

Two hashes over a config yaml, kept byte-for-byte compatible with
``heom-gpu/src/heom_config.cpp``:

* :func:`fnv1a64_hex` — FNV-1a 64-bit of a UTF-8 string (matches
  ``include/util/fnv.hpp``).
* :func:`physics_hash` — the "same physics?" fingerprint the solver stores as
  ``config_hash``: canonicalise the PARSED yaml (drop comments / whitespace /
  key-order) with the disorder-owned ``static_disorder`` top-level block and the
  non-physical ``output`` sub-block inside each task block (``dynamics`` /
  ``absorption`` / ``2des``) removed, then FNV-1a. Stable across comment/format
  edits so an incremental MC batch (or a 2des re-merge) after a harmless yaml
  edit does not fork the group.

The C++ ``canonical_yaml`` walks the PARSED node emitting scalars verbatim, maps
with sorted keys, sequences in order. We do the SAME by composing the yaml into
its node tree (``yaml.compose`` preserves the raw scalar text, unlike
``safe_load`` which types scalars) and serialising identically. A golden vector
shared with the C++ test suite pins the two implementations together.
"""
from __future__ import annotations

import yaml

__all__ = ["fnv1a64_hex", "physics_hash"]

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF

# Disorder-owned top-level block excluded from the physics identity, plus the
# non-physical `output` sub-block stripped from each task block below (mirrors
# heom_config.cpp::physics_hash).
_DROP_TOP_LEVEL = ("static_disorder",)
_TASK_BLOCKS = ("dynamics", "absorption", "2des")


def fnv1a64_hex(text: str) -> str:
    """FNV-1a 64-bit hex digest of ``text`` (matches heom::fnv1a64_hex)."""
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK64
    return f"{h:016x}"


def _canonical(node) -> str:
    """Serialise a composed yaml node exactly as C++ canonical_yaml does."""
    if node is None:
        return "~"
    if isinstance(node, yaml.ScalarNode):
        # yaml-cpp represents an empty/`~`/`null` value as a Null node -> "~".
        if node.tag.endswith(":null"):
            return "~"
        return node.value
    if isinstance(node, yaml.SequenceNode):
        return "[" + ",".join(_canonical(c) for c in node.value) + "]"
    if isinstance(node, yaml.MappingNode):
        items = sorted(((k.value, v) for k, v in node.value), key=lambda kv: kv[0])
        return "{" + ",".join(f"{k}:{_canonical(v)}" for k, v in items) + "}"
    raise ValueError(f"unexpected yaml node type: {type(node).__name__}")


def physics_hash(yaml_text: str) -> str:
    """Physics-identity hash of a config yaml string (== solver ``config_hash``)."""
    root = yaml.compose(yaml_text)
    if root is None:
        return fnv1a64_hex("~")
    if isinstance(root, yaml.MappingNode):
        # Drop the disorder-owned top-level block.
        root.value = [(k, v) for k, v in root.value
                      if k.value not in _DROP_TOP_LEVEL]
        # Drop the non-physical `output` sub-block inside each task block.
        for k, v in root.value:
            if k.value in _TASK_BLOCKS and isinstance(v, yaml.MappingNode):
                v.value = [(kk, vv) for kk, vv in v.value if kk.value != "output"]
    return fnv1a64_hex(_canonical(root))
