#ifndef HEOM_CONFIG_HPP
#define HEOM_CONFIG_HPP

#include "types.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief Simple configuration loader for HEOM simulations
 *
 * Replaces the complex ParameterManager with direct YAML loading.
 * All configuration is loaded into standard types.hpp structures.
 */
namespace HEOMConfig {

/**
 * @brief 2DES task input — coherence/waiting/detection axes + pathway selection.
 *
 * YAML: the `2des:` task block. Dipoles/strengths are NOT here — they are a
 * system property (PhysicalParameters::dipoles/strengths, YAML system.dipoles).
 * Only the 2des task carries these axes; absorption needs no extra struct
 * (its only inputs are the shared dipoles + time grid).
 *
 * resolve_2des_config validates presence + integer-multiple-of-time.dt for the
 * spacings and defaults pathways/signals to "all" when omitted. Field name is
 * dt_detect (NOT dt) to avoid clashing with time.dt.
 */
struct TwoDESConfig {
    double tau_max = 0.0, dtau = 0.0;     // coherence axis (fs)
    // Waiting axis (fs): non-uniform explicit sample times, expanded from the
    // yaml `tw` list ("T" or "start:step:stop"). Empty ⇒ resolve defaults to
    // a single T=0 point (n_tw=1).
    std::vector<double> tw_fs;
    double t_max   = 0.0, dt_detect = 0.0;// detection axis (fs)
    std::vector<std::string> pathways;    // subset of {"SE","GSB","ESA"}
    std::vector<std::string> signals;     // subset of {"rephasing","nonrephasing"}
    // Orientation is always the isotropic average (I1,I2,I3). Kept as a constant
    // for output/MC-merge provenance; the YAML `orientation` key is validated to
    // reject anything but "isotropic" (the non-physical "none" shortcut was removed).
    std::string orientation = "isotropic";
    // YAML 2des.batch: walk width B_t ∈ {1, 7, 21}, or kTwoDESBatchAuto (0) for
    // "auto" (try 21, then 7, then 1). 1 / false (default):
    // B=1 reuse walk. 21 / true: lockstep 3/9/21. 7: per-m1 subtree 1/3/7.
    // An explicit width that exceeds the GPU budget is an error, never a
    // silent switch; auto only when the user wrote "auto".
    int batch = 1;
};

/// Sentinel for YAML `2des.batch: auto`. Resolved to 21/7/1 inside compute_2des
/// before the walk; never written to HDF5 as-is (write_2des records the chosen
/// width from TwoDESResult::batch_width).
inline constexpr int kTwoDESBatchAuto = 0;


/**
 * @brief ADO filtering knobs (YAML: the `filter:` block).
 *
 * `enabled` is the single explicit on/off switch. When false, every other field
 * is ignored (left at its default) — no "set four knobs to 0" ceremony. When
 * true, parse_filter validates tolerance > 0, ulist_tau ∈ [0,1].
 */
struct FilterConfig {
    bool   enabled = false;         // master on/off
    double tolerance = 0.0;         // ADO max-norm threshold (> 0 when enabled)
    double ulist_tau = 0.0;         // U-list whole-step launch (YAML: filter.ulist).
                                    // 0 = off; (0,1] enables the compact derivative
                                    // + in-place RK4 over the active closure AND is
                                    // the adaptive gate τ (|E|/M > τ ⇒ full grid).
    bool   log_per_slot = false;    // Batched diagnostic (YAML: filter.log_per_slot).
                                    // When on, the stats line appends per-slot active
                                    // counts slots[|A_0|,|A_1|,...] beside the union
                                    // |A|. Default off ⇒ zero hot-path cost (the union
                                    // norm kernel emits one max norm, no per-slot pass).
};

/**
 * @brief Excited-state equilibration before fluorescence emission (YAML:
 * `fluorescence.equilibration`). Stop when max(δ, ε) ≤ tolerance, δ = relative
 * hierarchy change over one check_interval, ε its geometric-tail estimate.
 */
struct EquilibrationConfig {
    double t_max_fs = 0.0;
    double check_interval_fs = 0.0;
    double tolerance = 0.0;
};

/**
 * @brief Complete configuration for a HEOM simulation
 *
 * Task-driven schema: the shared blocks (system / bath / filter /
 * static_disorder) are parsed by `load_yaml()`; the per-task blocks
 * (dynamics / absorption / 2des) are parsed by the matching
 * `resolve_<task>_config()`, which fills `time`, `output`, and (for their
 * task) `initial` / `twodes`. A run only ever executes ONE task, so there is
 * no per-task map — one block, one parse. `Config::validate()` only enforces
 * shared-schema invariants; task-level checks live in each resolver.
 */
struct Config {
    PhysicalParameters physics;

    // Which task this config was resolved for. Set by resolve_<task>_config();
    // a config straight out of load_yaml() reads Dynamics. That default is
    // deliberate and safe: Dynamics stores the full N×N ρ, the largest block
    // any task needs (rho_capacity), so a config that never went through a
    // resolver over-allocates rather than under-allocates.
    enum class Task { Dynamics, Absorption, TwoDES, Fluorescence };
    Task task = Task::Dynamics;

    // Active time grid + output. Empty after load_yaml(); filled by
    // resolve_<task>_config() from the running task's YAML block.
    TimeParameters time;
    OutputConfig output;

    std::optional<InitialState> initial;   // dynamics only (dynamics.initial_state)
    std::optional<TwoDESConfig> twodes;    // 2des only (the `2des:` axes block)

    std::string yaml_path;          // Config file path (for auto-naming)
    std::string config_yaml;        // Exact YAML text used to build this config
    // "Same physics?" fingerprint: FNV-1a of the PARSED config, canonicalised
    // (comments / whitespace / key-order stripped) and scoped to what changes
    // this task's numbers: `system` + `bath` + `filter` (minus log_per_slot)
    // + the ONE task block being run (minus output / print_interval / batch).
    // The other task blocks and `static_disorder` are not in it. Editing a
    // comment, an output path, or another task's dt does NOT change it (so an
    // MC incremental batch still merges); a real physical edit does. Final
    // after resolve_<task>_config(); load_yaml alone yields the task-less core
    // hash. The exact input text is preserved in `config_yaml`.
    std::string config_hash;
    // MC static-disorder node file, from the optional top-level
    // `static_disorder: { draws_file: ... }` block. Empty ⇒ block absent.
    // Resolved at load time relative to the config file's directory (a bare
    // filename lives next to the yaml). Only consulted on the MC path
    // (heom_solver --rank N / mc-disorder-run); the single-shot path ignores it.
    std::string disorder_draws_file;
    // Hierarchy occupation cutoff: Σ_{n,k} α_{nk} ≤ L_cutoff. Kept generic so
    // future Padé / other pole expansions reuse it. REQUIRED key in `bath:`.
    int L_cutoff = 0;
    // ADO filtering knobs (the `filter:` block). enabled is the on/off switch.
    FilterConfig filter;

    // Linear-response 3-polarization (x/y/z) batching switch (YAML:
    // absorption.batch / fluorescence.batch — one run has one task block).
    // true ⇒ pack the ≤3 non-zero directions into one B-wide propagation;
    // false (default) ⇒ historical serial 3-direction loop, bit-identical.
    // 2DES batching is its own switch (TwoDESConfig::batch).
    bool linear_batch = false;
    EquilibrationConfig equilibration;   // fluorescence only

    /**
     * @brief Load configuration from YAML file.
     *
     * Historically an unconditional `resolve_auto_filename(cfg)` call at
     * the bottom of this function has moved into `resolve_dynamics_config()`
     * (the only task whose auto-naming rule reads `cfg.initial`). Spectro-
     * scopy tasks resolve `output.filename == "auto"` in their own
     * `resolve_<task>_config()` entry (e.g. `<system>_absorption`).
     *
     * @param yaml_path Path to YAML configuration file
     * @return Loaded configuration
     * @throws std::runtime_error on validation failure
     */
    static Config load_yaml(const std::string& yaml_path);

    /**
     * @brief Print configuration summary
     */
    void print_summary() const;

    /**
     * @brief Validate configuration (schema-level only).
     *
     * Optional-aware: initial-related branches are gated on
     * `initial.has_value()`. Task-required-section checks live in the
     * matching `resolve_<task>_config()` entry, NOT here.
     */
    void validate() const;
};

// ---------------------------------------------------------------------------
// Filename helpers (exposed for main.cpp dispatch).
// ---------------------------------------------------------------------------

/// Strip directory + extension from `yaml_path`, then take the leading token
/// before the first '_' (e.g. `config/fmo_benkmarch.yaml` -> "fmo",
/// `config/abs_dimer.yaml` -> "abs"). Used by every `resolve_<task>_config()`
/// when synthesising an auto filename.
std::string extract_system_name(const std::string& yaml_path);

/// Build the dynamics auto filename (was `resolve_auto_filename` inside
/// `Config::load_yaml`). Requires `cfg.initial.has_value()` and
/// `cfg.initial->type ∈ {SITE_EXCITATION, EXCITON_EXCITATION}`. Throws
/// otherwise with a message pointing at the offending init type — coherent /
/// custom_matrix users must set `output.filename` explicitly.
std::string resolve_dynamics_auto_filename(const Config& cfg);

// ---------------------------------------------------------------------------
// Task-level resolvers (run BEFORE SystemContext construction).
// ---------------------------------------------------------------------------
//
// Each `resolve_<task>_config()` is responsible for:
//   - asserting that the task-required YAML sections exist on `cfg`,
//   - asserting any task-level physics preconditions (e.g. basis),
//   - resolving `output.filename == "auto"` per the task's naming rule.
// All mutations happen here; `run_<task>(sys, cfg)` is read-only on cfg.
// SystemContext holds `const Config&`, so writing to `cfg` after
// `SystemContext sys(cfg)` would NOT be visible to the writer (plan
//).

/// Dynamics task entry. Throws std::invalid_argument if
/// `cfg.initial` is empty; resolves `output.filename` via
/// `resolve_dynamics_auto_filename(cfg)` when set to "auto".
void resolve_dynamics_config(Config& cfg);

/// Absorption task entry: parses the `absorption:` block (time + output),
/// forces the with_ground basis via apply_basis(), and requires
/// system.dipoles/strengths to be present. Resolves `output.filename` to
/// `<system>_absorption` when "auto".
void resolve_absorption_config(Config& cfg);

/// Fluorescence task entry: like absorption (with_ground basis, dipoles
/// required, `batch`) plus the required `equilibration:` sub-block, validated
/// here. Resolves `output.filename` to `<system>_fluorescence` when "auto".
void resolve_fluorescence_config(Config& cfg);

/// 2des task entry: parses the `2des:` block (dt/print_interval + axes +
/// output). Throws if system.dipoles/strengths are absent or an axis spacing
/// (dtau/dt_detect) or any `tw` sample is not an integer multiple of time.dt
/// (also rejects non-increasing/negative `tw`). Forces the double_excited basis
/// (ESA needs it; the driver builds a second with_ground context for SE/GSB).
/// Defaults pathways/signals to all when the yaml omits them; resolves
/// `<system>_2des` auto filename.
void resolve_2des_config(Config& cfg);

/// Apply a task-chosen Hilbert-space basis: sets physics.basis, recomputes
/// n_sites, and re-expands the stored site Hamiltonian (physics.hamiltonian_site)
/// into the full space. Called by each resolve_<task>_config before
/// SystemContext is built; 2des calls it twice (on cloned configs) to obtain
/// the with_ground (SE/GSB) and double_excited (ESA) contexts.
void apply_basis(Config& cfg, BasisType basis);

} // namespace HEOMConfig

#endif // HEOM_CONFIG_HPP
