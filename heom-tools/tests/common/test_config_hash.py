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
  start_step: 10
  warmup_factor: 0.001
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
static_disorder:
  draws_file: "nodes.h5"
"""

# Must equal kPhysicsHashVectorExpected in test_core.cpp.
_GOLDEN = "7e9a4ebcf3c349b7"


def test_fnv1a64_hex_known_vectors():
    # FNV-1a 64 of the empty string is the offset basis.
    assert fnv1a64_hex("") == "cbf29ce484222325"


def test_physics_hash_matches_cpp_golden():
    assert physics_hash(_VECTOR) == _GOLDEN


def test_physics_hash_ignores_comments_whitespace_and_dropped_blocks():
    ref = physics_hash(_VECTOR)
    # Leading/trailing comments + blank line: no change.
    commented = "# header\n" + _VECTOR + "\n# trailing r-range note\n"
    assert physics_hash(commented) == ref
    # Editing the non-physical output block: no change.
    assert physics_hash(_VECTOR.replace("/tmp/", "/other/dir/")) == ref
    # Editing static_disorder (disorder identity is tracked separately): no change.
    assert physics_hash(_VECTOR.replace("nodes.h5", "other_nodes.h5")) == ref


def test_physics_hash_changes_on_real_physics_edit():
    ref = physics_hash(_VECTOR)
    assert physics_hash(_VECTOR.replace("temperature: 77.0",
                                        "temperature: 100.0")) != ref
    # Waiting-time / spectroscopy IS physics.
    assert physics_hash(_VECTOR.replace("strengths: [4.5, 4.5]",
                                        "strengths: [4.5, 5.0]")) != ref


def test_physics_hash_stable_under_key_reorder():
    # Move the whole bath block above system: same physics, same hash.
    lines = _VECTOR.splitlines(keepends=True)
    bath_start = next(i for i, l in enumerate(lines) if l.startswith("bath:"))
    bath_end = next(i for i, l in enumerate(lines) if l.startswith("filter:"))
    reordered = ("".join(lines[bath_start:bath_end])
                 + "".join(lines[:bath_start])
                 + "".join(lines[bath_end:]))
    assert physics_hash(reordered) == physics_hash(_VECTOR)
