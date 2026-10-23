"""Config identity hashing — the Python mirror of the solver's C++ hashing.

Two hashes over a config yaml, kept byte-for-byte compatible with
``heom-gpu/src/heom_config.cpp``:

* :func:`fnv1a64_hex` — FNV-1a 64-bit of a UTF-8 string (matches
  ``include/util/fnv.hpp``).
* :func:`physics_hash` — the "same physics?" fingerprint the solver stores as
  ``config_hash``: keep exactly what changes the task's numbers — ``system``,
  ``bath``, ``filter`` (minus ``log_per_slot``) and the ONE task block being run
  (minus ``output`` / ``print_interval`` / ``batch``) — canonicalise the PARSED
  yaml (drop comments / whitespace / key-order) and FNV-1a it. The other task
  blocks and ``static_disorder`` are not in it, so an incremental MC batch (or a
  2des re-merge) after a harmless yaml edit, or an edit to another task's dt,
  does not fork the group. ``task=None`` mirrors ``load_yaml`` before the task is
  known (shared core only).

The C++ ``canonical_yaml`` walks the PARSED node emitting scalars verbatim, maps
with sorted keys, sequences in order. We do the SAME by composing the yaml into
its node tree (``yaml.compose`` preserves the raw scalar text, unlike
``safe_load`` which types scalars) and serialising identically. A golden vector
shared with the C++ test suite pins the two implementations together.
"""
from __future__ import annotations

import yaml

__all__ = ["TASKS", "fnv1a64_hex", "physics_hash"]

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF

# Mirrors heom_config.cpp::physics_hash: shared core blocks kept for every task,
# print knob dropped from `filter`, non-physical knobs dropped from the task block.
_CORE_BLOCKS = ("system", "bath", "filter")
_FILTER_DROP = ("log_per_slot",)
_TASK_DROP = ("output", "print_interval", "batch")
TASKS = ("dynamics", "absorption", "fluorescence", "2des")


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


def _without_keys(node, drop):
    if isinstance(node, yaml.MappingNode):
        node.value = [(k, v) for k, v in node.value if k.value not in drop]
    return node


def physics_hash(yaml_text: str, task: str | None = None) -> str:
    """Physics-identity hash of a config yaml string (== solver ``config_hash``).

    ``task`` is one of :data:`TASKS`; ``None`` gives the task-less core hash the
    solver holds between ``load_yaml`` and ``resolve_<task>_config``.
    """
    if task is not None and task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    root = yaml.compose(yaml_text)
    if not isinstance(root, yaml.MappingNode):
        return fnv1a64_hex("~")
    keep = []
    for k, v in root.value:
        if k.value in _CORE_BLOCKS:
            keep.append((k, _without_keys(v, _FILTER_DROP) if k.value == "filter" else v))
        elif task is not None and k.value == task:
            keep.append((k, _without_keys(v, _TASK_DROP)))
    if not keep:
        # yaml-cpp: a node that was never assigned to is Null -> "~".
        return fnv1a64_hex("~")
    root.value = keep
    return fnv1a64_hex(_canonical(root))
