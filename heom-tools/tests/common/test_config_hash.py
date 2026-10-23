"""Config-hash tests, incl. the cross-language golden that pins the Python
physics_hash to the C++ solver's config_hash.

If the golden constant here ever disagrees with the C++ test
(``tests/test_core.cpp`` kPhysicsHashVectorExpected), the two canonicalisers
have drifted — that is the bug this test exists to catch.
"""
from heom_tools.common.config_hash import fnv1a64_hex, physics_hash

# Byte-for-byte identical to kPhysicsHashVectorYaml in test_core.cpp.
_VECTOR = """system:
  n_states: 2
  basis: with_ground
  temperature: 77.0
  hamiltonian:
    - [12100.0, -50.0]
    - [-50.0, 12300.0]
  w0: 12000.0
  dipoles:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
  strengths: [4.5, 4.5]
bath:
  spectral_density: "Drude-Lorentz"
  eta: 35.0
  gamma_inverse: 100.0
  expansion_method: matsubara
  n_extra_poles: 1
  L_cutoff: 3
filter:
  enabled: true
  tolerance: 1.0e-7
absorption:
  integrator: rk4
  dt: 1.0
  n_steps: 200
  output_interval: 1
  print_interval: 200
  output:
    save_path: "/tmp/"
    filename: "auto"
    use_timestamp: false
fluorescence:
  integrator: rk4
  dt: 1.0
  n_steps: 200
  output_interval: 1
  batch: true
  output:
    save_path: "/tmp/"
  equilibration:
    t_max: 5000.0
    check_interval: 50.0
    tolerance: 1.0e-6
static_disorder:
  draws_file: "nodes.h5"
"""

# Must equal kPhysicsHashVectorExpected in test_core.cpp.
# Task-scoped (absorption) — the value the solver writes into an absorption file.
_GOLDEN = "bef65eb0a9c0f82b"
# Task-less core (system+bath+filter) — what load_yaml holds before resolve.
_GOLDEN_CORE = "93fab135bac4ec03"
# Must equal kPhysicsHashVectorExpectedFl: pins the nested `equilibration` map.
# Fluorescence rejects the ADO filter, so its vector has the filter off.
_GOLDEN_FL = "b36d1b90e6bca5c1"
_VECTOR_FL = _VECTOR.replace("  enabled: true\n  tolerance: 1.0e-7\n", "  enabled: false\n")


def test_fnv1a64_hex_known_vectors():
    # FNV-1a 64 of the empty string is the offset basis.
    assert fnv1a64_hex("") == "cbf29ce484222325"


def test_physics_hash_matches_cpp_golden():
    assert physics_hash(_VECTOR, "absorption") == _GOLDEN
    assert physics_hash(_VECTOR) == _GOLDEN_CORE
    assert physics_hash(_VECTOR_FL, "fluorescence") == _GOLDEN_FL
    assert _VECTOR_FL != _VECTOR
    assert _GOLDEN != _GOLDEN_CORE


def test_fluorescence_hash_tracks_equilibration_not_batch():
    ref = physics_hash(_VECTOR_FL, "fluorescence")
    assert physics_hash(_VECTOR_FL.replace("batch: true", "batch: false"), "fluorescence") == ref
    assert physics_hash(_VECTOR_FL.replace("tolerance: 1.0e-6", "tolerance: 1.0e-8"),
                        "fluorescence") != ref
    assert physics_hash(_VECTOR_FL.replace("t_max: 5000.0", "t_max: 9000.0"),
                        "fluorescence") != ref


def test_physics_hash_ignores_comments_whitespace_and_dropped_blocks():
    ref = physics_hash(_VECTOR, "absorption")
    # Leading/trailing comments + blank line: no change.
    commented = "# header\n" + _VECTOR + "\n# trailing r-range note\n"
    assert physics_hash(commented, "absorption") == ref
    # Editing the non-physical output block: no change.
    assert physics_hash(_VECTOR.replace("/tmp/", "/other/dir/"), "absorption") == ref
    # Editing static_disorder (disorder identity is tracked separately): no change.
    assert physics_hash(_VECTOR.replace("nodes.h5", "other_nodes.h5"), "absorption") == ref
    # Print / batch knobs move bytes, not physics: no change.
    assert physics_hash(_VECTOR.replace("print_interval: 200", "print_interval: 7"),
                        "absorption") == ref
    assert physics_hash(_VECTOR.replace("integrator: rk4", "integrator: rk4\n  batch: true"),
                        "absorption") == ref
    assert physics_hash(_VECTOR.replace("tolerance: 1.0e-7",
                                        "tolerance: 1.0e-7\n  log_per_slot: true"),
                        "absorption") == ref


def test_physics_hash_scoped_to_own_task_block():
    ref = physics_hash(_VECTOR, "absorption")
    # Another task's block appearing / changing must not fork this task's group.
    with_dyn = _VECTOR + "dynamics:\n  dt: 0.5\n  n_steps: 10\n"
    assert physics_hash(with_dyn, "absorption") == ref
    assert physics_hash(with_dyn.replace("dt: 0.5", "dt: 0.25"), "absorption") == ref
    # But the block IS the identity of its own task.
    assert physics_hash(with_dyn, "dynamics") != physics_hash(
        with_dyn.replace("dt: 0.5", "dt: 0.25"), "dynamics")
    # And a task whose block is absent hashes to the core.
    assert physics_hash(_VECTOR, "2des") == _GOLDEN_CORE


def test_physics_hash_changes_on_real_physics_edit():
    ref = physics_hash(_VECTOR, "absorption")
    assert physics_hash(_VECTOR.replace("temperature: 77.0",
                                        "temperature: 100.0"), "absorption") != ref
    # Waiting-time / spectroscopy IS physics.
    assert physics_hash(_VECTOR.replace("strengths: [4.5, 4.5]",
                                        "strengths: [4.5, 5.0]"), "absorption") != ref
    # Time grid of the task being run IS physics.
    assert physics_hash(_VECTOR.replace("n_steps: 200", "n_steps: 400"),
                        "absorption") != ref
    # Filter tolerance changes the numbers: physics.
    assert physics_hash(_VECTOR.replace("tolerance: 1.0e-7", "tolerance: 1.0e-5"),
                        "absorption") != ref


def test_physics_hash_stable_under_key_reorder():
    # Move the whole bath block above system: same physics, same hash.
    lines = _VECTOR.splitlines(keepends=True)
    bath_start = next(i for i, l in enumerate(lines) if l.startswith("bath:"))
    bath_end = next(i for i, l in enumerate(lines) if l.startswith("filter:"))
    reordered = ("".join(lines[bath_start:bath_end])
                 + "".join(lines[:bath_start])
                 + "".join(lines[bath_end:]))
    assert physics_hash(reordered, "absorption") == physics_hash(_VECTOR, "absorption")
