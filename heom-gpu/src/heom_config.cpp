#include "heom_config.hpp"
#include "util/stream_format.hpp"
#include "core/hamiltonian_expansion.hpp"
#include "util/fnv.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <vector>

namespace HEOMConfig {

static std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open YAML file: " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// Canonical serialisation of a parsed YAML node: maps emit their keys in sorted
// order, sequences keep their order (list order is semantic), scalars emit their
// text verbatim. Comments and whitespace are already gone (the node is parsed),
// so two files that differ only in comments/formatting/key-order serialise
// identically. This is the input to the physics fingerprint.
static void canonical_yaml(const YAML::Node& n, std::string& out) {
    switch (n.Type()) {
        case YAML::NodeType::Null:
        case YAML::NodeType::Undefined:
            out += "~";
            break;
        case YAML::NodeType::Scalar:
            out += n.Scalar();
            break;
        case YAML::NodeType::Sequence:
            out += "[";
            for (std::size_t i = 0; i < n.size(); ++i) {
                if (i) out += ",";
                canonical_yaml(n[i], out);
            }
            out += "]";
            break;
        case YAML::NodeType::Map: {
            std::vector<std::string> keys;
            keys.reserve(n.size());
            for (auto it = n.begin(); it != n.end(); ++it)
                keys.push_back(it->first.Scalar());
            std::sort(keys.begin(), keys.end());
            out += "{";
            for (std::size_t i = 0; i < keys.size(); ++i) {
                if (i) out += ",";
                out += keys[i];
                out += ":";
                canonical_yaml(n[keys[i]], out);
            }
            out += "}";
            break;
        }
    }
}

// "Same physics?" fingerprint: canonicalise the parsed config with the
// non-physical `output` sub-blocks and disorder-owned `static_disorder` block
// dropped, then FNV-1a. Stable across comment/whitespace/key-order edits;
// changes only when a physically-meaningful input changes. Used for the MC
// merge key so an incremental batch run after a harmless yaml edit still merges.
// `output` now lives inside each task block, so strip it there.
static std::string physics_hash(const YAML::Node& yaml) {
    YAML::Node phys = YAML::Clone(yaml);
    phys.remove("static_disorder");
    for (const char* task : {"dynamics", "absorption", "2des"}) {
        if (phys[task] && phys[task].IsMap()) phys[task].remove("output");
    }
    std::string canon;
    canonical_yaml(phys, canon);
    return heom::fnv1a64_hex(canon);
}

/**
 * @brief Validate density matrix properties (Hermiticity, normalization, positivity)
 * @param rho Density matrix to validate
 * @param n_sites Number of sites
 * @param tolerance Numerical tolerance for checks
 * @return true if all checks pass, false otherwise
 * 
 * Prints warnings for violations but does not modify the matrix.
 */
static bool validate_density_matrix(const Eigen::MatrixXcd& rho, int n_sites, double tolerance = 1e-10) {
    const heom::StreamFormat fmt_guard(std::cerr);
    bool all_ok = true;
    
    // Check 1: Hermiticity (ρ_{ij} = ρ*_{ji})
    bool is_hermitian = true;
    double max_hermiticity_error = 0.0;
    for (int i = 0; i < n_sites; ++i) {
        for (int j = i; j < n_sites; ++j) {
            std::complex<double> diff = rho(i, j) - std::conj(rho(j, i));
            double error = std::abs(diff);
            if (error > max_hermiticity_error) {
                max_hermiticity_error = error;
            }
            if (error > tolerance) {
                is_hermitian = false;
            }
        }
    }
    
    if (!is_hermitian) {
        std::cerr << "  WARNING: Density matrix is not Hermitian (max error: " 
                  << std::scientific << std::setprecision(3) << max_hermiticity_error << ")\n";
        all_ok = false;
    }
    
    // Check 2: Normalization (Tr(ρ) = 1)
    std::complex<double> trace = rho.trace();
    double trace_error = std::abs(trace - 1.0);
    if (trace_error > tolerance) {
        std::cerr << "  WARNING: Density matrix is not normalized. Tr(rho) = " 
                  << std::fixed << std::setprecision(10) << trace.real();
        if (std::abs(trace.imag()) > tolerance) {
            std::cerr << " + " << trace.imag() << "i";
        }
        std::cerr << " (expected: 1.0, error: " << std::scientific << trace_error << ")\n";
        all_ok = false;
    }
    
    // Check 3: Positivity (all eigenvalues ≥ 0)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigensolver(rho);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "  WARNING: Failed to compute eigenvalues of density matrix\n";
        all_ok = false;
    } else {
        const Eigen::VectorXd& eigenvalues = eigensolver.eigenvalues().real();
        double min_eigenvalue = eigenvalues.minCoeff();
        if (min_eigenvalue < -tolerance) {
            std::cerr << "  WARNING: Density matrix is not positive semidefinite. "
                      << "Minimum eigenvalue: " << std::scientific << std::setprecision(3) 
                      << min_eigenvalue << "\n";
            all_ok = false;
        } else if (min_eigenvalue < tolerance) {
            // Near-zero negative eigenvalue (numerical noise)
            std::cerr << "  NOTE: Minimum eigenvalue is near zero: " 
                      << std::scientific << std::setprecision(3) << min_eigenvalue << "\n";
        }
    }
    
    return all_ok;
}

// ============================================================================
// YAML section parsers
// ============================================================================

// Parse the optional system.dipoles / system.strengths lists into
// cfg.physics.dipoles / strengths. Absent ⇒ both stay empty (dynamics never
// reads them; the absorption/2des resolvers require them non-empty). When
// present, dipoles must be n_states unit 3-vectors (non-unit ⇒ warn + auto-
// normalise, magnitude belongs in strengths) and strengths n_states values
// ≥ 0 that are not all zero.
static void parse_dipoles(const YAML::Node& sys, Config& cfg) {
    const heom::StreamFormat fmt_guard(std::cerr);
    const bool has_d = static_cast<bool>(sys["dipoles"]);
    const bool has_s = static_cast<bool>(sys["strengths"]);
    if (!has_d && !has_s) return;                 // block omitted entirely
    if (has_d != has_s)
        throw std::runtime_error(
            "system.dipoles and system.strengths must be given together "
            "(one is present, the other missing)");

    auto dipoles_yaml = sys["dipoles"];
    auto strengths_yaml = sys["strengths"];
    if (!dipoles_yaml.IsSequence())
        throw std::runtime_error("system.dipoles must be a list of 3-vectors");
    if (!strengths_yaml.IsSequence())
        throw std::runtime_error("system.strengths must be a list of doubles");

    const int n_states = cfg.physics.n_states;
    if (static_cast<int>(dipoles_yaml.size()) != n_states)
        throw std::runtime_error("system.dipoles size (" +
            std::to_string(dipoles_yaml.size()) + ") does not match n_states (" +
            std::to_string(n_states) + ")");
    if (static_cast<int>(strengths_yaml.size()) != n_states)
        throw std::runtime_error("system.strengths size (" +
            std::to_string(strengths_yaml.size()) + ") does not match n_states (" +
            std::to_string(n_states) + ")");

    cfg.physics.dipoles.resize(n_states);
    cfg.physics.strengths.resize(n_states);
    double strength_sum = 0.0;
    for (int i = 0; i < n_states; ++i) {
        auto d = dipoles_yaml[i];
        if (!d.IsSequence() || d.size() != 3)
            throw std::runtime_error("system.dipoles[" + std::to_string(i) +
                                     "] must be a 3-vector [dx, dy, dz]");
        std::array<double, 3> v = {
            d[0].as<double>(), d[1].as<double>(), d[2].as<double>()
        };
        const double norm = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (norm < 1e-12)
            throw std::runtime_error("system.dipoles[" + std::to_string(i) +
                                     "] has zero length");
        if (std::abs(norm - 1.0) >= 1e-6) {
            std::cerr << "  WARNING: system.dipoles[" << i
                      << "] is not unit length (||d|| = "
                      << std::scientific << std::setprecision(6) << norm
                      << "); auto-normalising. Put magnitude into 'strengths'.\n";
            v[0] /= norm; v[1] /= norm; v[2] /= norm;
        }
        cfg.physics.dipoles[i]   = v;
        cfg.physics.strengths[i] = strengths_yaml[i].as<double>();
        if (cfg.physics.strengths[i] < 0.0)
            throw std::runtime_error("system.strengths[" + std::to_string(i) +
                                     "] must be >= 0");
        strength_sum += cfg.physics.strengths[i];
    }
    if (strength_sum < 1e-12)
        throw std::runtime_error("system.strengths cannot all be zero");
}

// Parses the system block and returns the BARE site-basis Hamiltonian (cm^-1):
// NOT reorg-shifted, NOT w0-shifted, NOT basis-expanded. The reorganization
// energy needs per-site eta (parsed later by parse_bath), so the final
// H_eff = H_S + H_reorg and the w0 shift + basis expansion are deferred to
// build_effective_hamiltonian(), called after parse_bath. w0_cm is still
// captured here.
static void parse_system(const YAML::Node& sys, Config& cfg,
                         Eigen::MatrixXcd& H_site_cm_out) {
    if (!sys["n_states"])
        throw std::runtime_error("Missing required field: system.n_states");
    cfg.physics.n_states = sys["n_states"].as<int>();

    // basis is task-driven: production yamls omit it and each task picks one
    // (resolve_<task>_config -> apply_basis; 2des builds two). An explicit
    // system.basis: is still honored as the initial/default basis (handy for
    // tests and direct SystemContext use); a task resolver may override it.
    cfg.physics.basis = BasisType::EXCITED_ONLY;
    if (sys["basis"]) {
        std::string basis_str = sys["basis"].as<std::string>();
        if (basis_str == "excited_only")        cfg.physics.basis = BasisType::EXCITED_ONLY;
        else if (basis_str == "with_ground")    cfg.physics.basis = BasisType::WITH_GROUND;
        else if (basis_str == "double_excited") cfg.physics.basis = BasisType::DOUBLE_EXCITED;
        else throw std::runtime_error("Invalid basis type: " + basis_str);
    }

    cfg.physics.temperature_K = sys["temperature"].as<double>();
    cfg.physics.temperature = cfg.physics.temperature_K * BOLTZMANN;

    int n_states = cfg.physics.n_states;
    Eigen::MatrixXcd H_excited_cm = Eigen::MatrixXcd::Zero(n_states, n_states);
    auto H_yaml = sys["hamiltonian"];

    if (!H_yaml || !H_yaml.IsSequence() || H_yaml.size() != static_cast<size_t>(n_states))
        throw std::runtime_error("Hamiltonian must be " + std::to_string(n_states) + "x" + std::to_string(n_states));

    for (int i = 0; i < n_states; ++i) {
        auto row = H_yaml[i];
        if (!row.IsSequence() || row.size() != static_cast<size_t>(n_states))
            throw std::runtime_error("Hamiltonian row " + std::to_string(i) + " size mismatch");
        for (int j = 0; j < n_states; ++j)
            H_excited_cm(i, j) = row[j].as<double>();
    }

    // w0 captured here, but applied later in build_effective_hamiltonian so it
    // references the reorg-shifted (H_eff) diagonal, not the bare H_S.
    cfg.physics.w0_cm = sys["w0"] ? sys["w0"].as<double>() : 0.0;

    // Optional transition dipoles (system property; spectroscopy tasks require them).
    parse_dipoles(sys, cfg);

    H_site_cm_out = std::move(H_excited_cm);  // bare site-basis (cm^-1); finalized later
}

// Builds the effective system Hamiltonian and commits it to cfg.physics.hamiltonian.
//
//   H_eff = H_S + H_reorg ,   H_reorg = Σ_n λ_n |n⟩⟨n| ,   λ_n = η_n / 2
//
// H_reorg is diagonal in the site basis, so λ_n is just added to site n's energy
// BEFORE the w0 shift (w0 now references the reorg-shifted physical energy).
// expand_hamiltonian then propagates λ correctly into every basis with no special
// cases: ground +0, |n⟩ +λ_n, |m,p⟩ +(λ_m+λ_p) (the double-excited diagonal is
// V(i,i)+V(j,j)). Homogeneous baths (λ_n≡λ) reduce H_reorg to λ·I on the single-
// excitation manifold — a trivial energy shift that leaves dynamics unchanged and
// only rigidly translates spectra. Heterogeneous baths shift the site-energy gaps
// and DO change dynamics and spectra. MUST run after parse_bath (needs per-site η).
static void build_effective_hamiltonian(Config& cfg, Eigen::MatrixXcd& H_site_cm) {
    const int n_states = cfg.physics.n_states;
    for (int n = 0; n < n_states; ++n)
        H_site_cm(n, n) += 0.5 * cfg.physics.bath_sites[n].eta_cm;  // + λ_n  (reorg)
    for (int i = 0; i < n_states; ++i)
        H_site_cm(i, i) -= cfg.physics.w0_cm;                       // − w0   (vs H_eff)
    // Store the finalized site-basis H (a.u.) but DON'T expand yet — the basis
    // is task-chosen and applied later by apply_basis().
    cfg.physics.hamiltonian_site = H_site_cm * CM_TO_AU;
}

// Build / override one BathSite from raw (cm/fs) inputs, recomputing a.u.
static void set_bath_site(BathSite& s, double eta_cm, double gamma_inverse_fs) {
    if (gamma_inverse_fs <= 0.0)
        throw std::runtime_error("bath.gamma_inverse must be > 0 (got " +
                                 std::to_string(gamma_inverse_fs) + ")");
    s.eta_cm            = eta_cm;
    s.eta_au            = eta_cm * CM_TO_AU;
    s.gamma_inverse_fs  = gamma_inverse_fs;
    s.gamma_au          = 1.0 / (gamma_inverse_fs * FS_TO_AU);
}

static void parse_bath(const YAML::Node& bath, Config& cfg) {
    // --- GLOBAL (numerical) knobs: expansion scheme + pole count ---
    // REQUIRED field — no implicit default. Forcing every yaml to spell out the
    // bath-correlation scheme eliminates the "old yaml silently switches scheme"
    // failure mode entirely (there is no fall-through, no opinion on a default).
    if (!bath["expansion_method"]) {
        throw std::runtime_error(
            "bath.expansion_method is required (legal values: 'pade' | "
            "'matsubara'). No implicit default is provided — every yaml must "
            "declare which numerical bath-correlation scheme it intends. See "
            "config/fmo.yaml for an example.");
    }
    std::string method = bath["expansion_method"].as<std::string>();
    if (method == "pade") {
        cfg.physics.expansion = BathExpansion::PADE;
    } else if (method == "matsubara") {
        cfg.physics.expansion = BathExpansion::MATSUBARA;
    } else {
        throw std::runtime_error(
            "bath.expansion_method must be 'pade' or 'matsubara' (got '"
            + method + "').");
    }

    if (!bath["n_extra_poles"]) {
        throw std::runtime_error("bath.n_extra_poles is required");
    }
    cfg.physics.n_extra_poles = bath["n_extra_poles"].as<int>();
    if (cfg.physics.n_extra_poles < 0 || cfg.physics.n_extra_poles > 50) {
        throw std::runtime_error("bath.n_extra_poles must be in [0, 50]");
    }

    // Hierarchy occupation cutoff Σ_{n,k} α_{nk} ≤ L_cutoff. Moved from the old
    // `truncation:` block into `bath:` (it is a bath-expansion convergence knob,
    // living next to K = n_extra_poles). Range check in Config::validate.
    if (!bath["L_cutoff"])
        throw std::runtime_error(
            "bath.L_cutoff is required (occupation cutoff, "
            "Σα := Σ_{n,k} α_{n,k} ≤ L_cutoff). Note: this key moved from the "
            "removed 'truncation:' block into 'bath:'.");
    cfg.L_cutoff = bath["L_cutoff"].as<int>();
    // Snapshot; compute_bath_correlation will reset n_bath_modes_total to its
    // final value (== n_extra_poles; there is no extra slot). Set here so a
    // build that skips compute (none currently) still has a sane runtime mode
    // count.
    cfg.physics.n_bath_modes_total = cfg.physics.n_extra_poles;

    // static_disorder_fwhm is no longer supported — reject on presence (any value).
    if (bath["static_disorder_fwhm"])
        throw std::runtime_error(
            "bath.static_disorder_fwhm is no longer supported: disorder-in-HEOM "
            "(ν=0 Matsubara mode) has been removed. Static disorder is now sampled "
            "via Monte Carlo on the Hamiltonian diagonal. Delete this key.");

    // --- Default per-site physics (applied to every site without an override) ---
    const double def_eta   = bath["eta"].as<double>();
    const double def_ginv  = bath["gamma_inverse"].as<double>();

    const int n_bath = cfg.physics.n_states;
    cfg.physics.bath_sites.assign(n_bath, BathSite{});
    for (auto& s : cfg.physics.bath_sites)
        set_bath_site(s, def_eta, def_ginv);

    // --- Optional per-site overrides: sites: { <1-based idx>: {fields...} } ---
    if (bath["sites"]) {
        const YAML::Node& sites = bath["sites"];
        if (!sites.IsMap())
            throw std::runtime_error("bath.sites must be a map keyed by 1-based "
                                     "pigment index, e.g. `3: { eta: 35.0 }`.");
        for (const auto& kv : sites) {
            const int idx1 = kv.first.as<int>();
            if (idx1 < 1 || idx1 > n_bath)
                throw std::runtime_error(
                    "bath.sites key " + std::to_string(idx1) +
                    " out of range [1, " + std::to_string(n_bath) + "].");
            const YAML::Node& ov = kv.second;
            // Only per-site physics fields allowed; the numerical knobs are global.
            for (const auto& f : ov) {
                const std::string key = f.first.as<std::string>();
                if (key != "eta" && key != "gamma_inverse")
                    throw std::runtime_error(
                        "bath.sites[" + std::to_string(idx1) + "] has unknown "
                        "field '" + key + "'. Only eta / gamma_inverse "
                        "may be overridden per site "
                        "(expansion_method / n_extra_poles are global).");
            }
            BathSite& s = cfg.physics.bath_sites[idx1 - 1];
            const double eta  = ov["eta"] ? ov["eta"].as<double>() : s.eta_cm;
            const double ginv = ov["gamma_inverse"]
                                    ? ov["gamma_inverse"].as<double>()
                                    : s.gamma_inverse_fs;
            set_bath_site(s, eta, ginv);
        }
    }

    compute_bath_correlation(cfg.physics);
}

// Parse the dynamics-only sink block (dynamics.sink). Called from
// resolve_dynamics_config; absorption/2des never parse it, so the sink stays
// disabled for those tasks (no cross-task leakage from a shared yaml).
static void parse_sink(const YAML::Node& dyn, Config& cfg) {
    // Parse, don't print. Sink configuration is shown later in
    // print_summary() like every other parameter.
    auto& sink_cfg = cfg.physics.sink;

    if (!dyn || !dyn["sink"]) {
        sink_cfg.enabled = false;
        sink_cfg.gamma = 0.0;
        sink_cfg.gamma_fs = 0.0;
        sink_cfg.site = -1;
        return;
    }

    auto sink = dyn["sink"];
    sink_cfg.enabled = sink["enabled"].as<bool>(false);

    if (!sink_cfg.enabled) {
        sink_cfg.gamma = 0.0;
        sink_cfg.gamma_fs = 0.0;
        sink_cfg.site = -1;
        return;
    }

    int user_site = sink["target_site"].as<int>();
    if (user_site < 1 || user_site > cfg.physics.n_states)
        throw std::runtime_error("target_site (" + std::to_string(user_site) +
                                 ") out of range [1, " + std::to_string(cfg.physics.n_states) + "]");
    sink_cfg.site = (cfg.physics.basis == BasisType::EXCITED_ONLY)
        ? user_site - 1
        : user_site;
    sink_cfg.gamma_fs = sink["rate"].as<double>();
    if (sink_cfg.gamma_fs <= 0.0) {
        throw std::runtime_error("sink.rate must be > 0 when sink.enabled is true (got "
                                 + std::to_string(sink_cfg.gamma_fs) + ")");
    }
    sink_cfg.gamma = sink_cfg.gamma_fs / FS_TO_AU;
}

// Parse the `filter:` block. `enabled` is the explicit on/off switch: when
// false, every knob is left at its default (no "set four things to 0"
// ceremony). When true, tolerance/start_step/warmup_factor/ulist are validated
// inline (fail loud at parse). The ulist sub-feature still uses 0 = off.
static void parse_filter(const YAML::Node& f, Config& cfg) {
    if (!f)
        throw std::runtime_error(
            "config is missing the required 'filter:' section "
            "(filter.enabled is the on/off switch for ADO filtering)");
    if (!f["enabled"])
        throw std::runtime_error(
            "filter.enabled is required (true/false). It is the single on/off "
            "switch for ADO filtering; when false the other filter knobs are "
            "ignored.");
    cfg.filter.enabled = f["enabled"].as<bool>();
    if (!cfg.filter.enabled) return;   // off ⇒ ignore all knobs, keep defaults

    if (!f["tolerance"])
        throw std::runtime_error(
            "filter.tolerance is required when filter.enabled is true "
            "(ADO max-norm threshold, e.g. 1e-6)");
    cfg.filter.tolerance     = f["tolerance"].as<double>();
    cfg.filter.start_step    = f["start_step"].as<int>(10);
    cfg.filter.warmup_factor = f["warmup_factor"].as<double>(0.1);
    cfg.filter.ulist_tau     = f["ulist"].as<double>(0.0);
    cfg.filter.log_per_slot  = f["log_per_slot"].as<bool>(false);

    if (cfg.filter.tolerance <= 0.0)
        throw std::runtime_error("filter.tolerance must be > 0 when enabled");
    if (cfg.filter.start_step <= 0)
        throw std::runtime_error("filter.start_step must be > 0 when enabled");
    if (cfg.filter.warmup_factor <= 0.0 || cfg.filter.warmup_factor > 1.0)
        throw std::runtime_error("filter.warmup_factor must be in (0, 1] when enabled");
    if (cfg.filter.ulist_tau < 0.0 || cfg.filter.ulist_tau > 1.0)
        throw std::runtime_error("filter.ulist must be in [0, 1] (0 = off)");
}

// Helpers below take the destination InitialState by reference
// and read from `cfg` only as a const context (basis / n_states / n_sites).
// `parse_initial_state` writes into a local `InitialState init` first and the
// load_yaml() top-level then commits via `cfg.initial = std::move(init)`. This
// keeps `cfg.initial == std::nullopt` until parsing succeeds end-to-end, so a
// failure half-way through never leaves a half-built optional behind.
static void parse_site_excitation(const YAML::Node& node,
                                  const Config& cfg,
                                  InitialState& init) {
    // User input is 1-based PHYSICAL pigment index (consistent with sink.target_site).
    // Mapping to internal 0-based Hilbert index depends on basis:
    //   EXCITED_ONLY            : pigment k → Hilbert k-1
    //   WITH_GROUND / DOUBLE_*  : ground at index 0, pigment k → Hilbert k
    int user_site = node["excited_site"].as<int>();
    if (user_site < 1 || user_site > cfg.physics.n_states)
        throw std::runtime_error("excited_site (" + std::to_string(user_site) +
            ") out of range [1, " + std::to_string(cfg.physics.n_states) + "]");
    init.excited_site = (cfg.physics.basis == BasisType::EXCITED_ONLY)
        ? user_site - 1
        : user_site;
}

static void parse_exciton_excitation(const YAML::Node& node,
                                     const Config& cfg,
                                     InitialState& init) {
    int user_idx = node["exciton_index"].as<int>();
    if (cfg.physics.basis == BasisType::DOUBLE_EXCITED) {
        throw std::runtime_error(
            "exciton_excitation is not supported for basis=double_excited; "
            "single-exciton block mapping is not implemented");
    }
    if (user_idx < 1 || user_idx > cfg.physics.n_states)
        throw std::runtime_error("exciton_index (" + std::to_string(user_idx) +
            ") out of single-exciton range [1, " +
            std::to_string(cfg.physics.n_states) + "]");
    init.exciton_index = (cfg.physics.basis == BasisType::EXCITED_ONLY)
        ? user_idx - 1
        : user_idx;  // WITH_GROUND: skip ground at eigenvector/Hilbert index 0.
}

static void parse_coherent(const YAML::Node& node,
                           const Config& cfg,
                           InitialState& init) {
    auto coeffs = node["coherent_coefficients"];
    if (!coeffs.IsSequence())
        throw std::runtime_error("coherent_coefficients must be a list");

    size_t n = coeffs.size();
    init.coherent_coeffs.resize(n);

    // Determine format: explicit "coherent_format" field takes priority,
    // otherwise auto-detect by checking if first element is a sequence.
    bool complex_format;
    if (node["coherent_format"]) {
        std::string fmt = node["coherent_format"].as<std::string>();
        if (fmt == "complex")          complex_format = true;
        else if (fmt == "probability") complex_format = false;
        else throw std::runtime_error("coherent_format must be 'complex' or 'probability', got: " + fmt);
    } else {
        complex_format = coeffs[0].IsSequence();
    }

    if (complex_format) {
        // New format: [[re, im], [re, im], ...] — direct complex amplitudes
        double norm_sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (!coeffs[i].IsSequence() || coeffs[i].size() != 2)
                throw std::runtime_error("coherent_coefficients[" + std::to_string(i) + "] must be [re, im]");
            double re = coeffs[i][0].as<double>();
            double im = coeffs[i][1].as<double>();
            init.coherent_coeffs[i] = std::complex<double>(re, im);
            norm_sq += re * re + im * im;
        }
        if (norm_sq < 1e-10)
            throw std::runtime_error("coherent_coefficients norm is too small");
        double scale = 1.0 / std::sqrt(norm_sq);
        for (size_t i = 0; i < n; ++i)
            init.coherent_coeffs[i] *= scale;
    } else {
        // Legacy format: [p1, p2, ...] — real probabilities, convert to amplitudes
        double sum_p = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double p = coeffs[i].as<double>();
            if (p < 0.0) throw std::runtime_error("coherent_coefficients must be non-negative");
            sum_p += p;
            init.coherent_coeffs[i] = std::complex<double>(p, 0.0);
        }
        if (sum_p < 1e-10)
            throw std::runtime_error("coherent_coefficients sum is too small");
        for (size_t i = 0; i < n; ++i)
            init.coherent_coeffs[i] = std::complex<double>(
                std::sqrt(init.coherent_coeffs[i].real() / sum_p), 0.0);
    }

    if (static_cast<int>(n) != cfg.physics.n_sites)
        throw std::runtime_error("coherent_coefficients size (" + std::to_string(n) +
                                 ") does not match n_sites (" + std::to_string(cfg.physics.n_sites) + ")");
}

static void parse_custom_matrix(const YAML::Node& node,
                                const Config& cfg,
                                InitialState& init) {
    int N = cfg.physics.n_sites;
    init.custom_matrix = Eigen::MatrixXcd::Zero(N, N);

    auto elems = node["matrix_elements"];
    if (!elems || !elems.IsSequence())
        throw std::runtime_error("custom_matrix type requires matrix_elements list");

    for (size_t idx = 0; idx < elems.size(); ++idx) {
        auto e = elems[idx];
        if (!e.IsSequence() || e.size() < 2)
            throw std::runtime_error("matrix_elements[" + std::to_string(idx) + "] format error");

        int i = e[0].as<int>() - 1;
        int j = e[1].as<int>() - 1;

        if (i < 0 || i >= N || j < 0 || j >= N)
            throw std::runtime_error("matrix_elements[" + std::to_string(idx) + "] index out of range");

        if (e.size() == 2) continue;

        std::complex<double> val;
        if (e[2].IsScalar()) {
            val = std::complex<double>(e[2].as<double>(), 0.0);
        } else if (e[2].IsSequence() && e[2].size() == 2) {
            val = std::complex<double>(e[2][0].as<double>(), e[2][1].as<double>());
        } else {
            throw std::runtime_error("matrix_elements[" + std::to_string(idx) + "] value format error");
        }
        init.custom_matrix(i, j) = val;
    }

    // custom_matrix is a physical density matrix in dynamics YAML. Invalid
    // Hermiticity, trace, or positivity is not a warning; it changes the
    // problem being solved.
    if (!validate_density_matrix(init.custom_matrix, N)) {
        throw std::runtime_error(
            "custom_matrix is not a valid density matrix");
    }
}

static void parse_initial_state(const YAML::Node& node,
                                const Config& cfg,
                                InitialState& init) {
    std::string type_str = node["type"].as<std::string>();

    if (type_str == "site_excitation") {
        init.type = InitialState::Type::SITE_EXCITATION;
        parse_site_excitation(node, cfg, init);
    } else if (type_str == "exciton_excitation") {
        init.type = InitialState::Type::EXCITON_EXCITATION;
        parse_exciton_excitation(node, cfg, init);
    } else if (type_str == "coherent") {
        init.type = InitialState::Type::COHERENT;
        parse_coherent(node, cfg, init);
    } else if (type_str == "custom_matrix") {
        init.type = InitialState::Type::CUSTOM_MATRIX;
        parse_custom_matrix(node, cfg, init);
    } else {
        throw std::runtime_error("Unknown initial state type: " + type_str);
    }
}

// ---------------------------------------------------------------------------
// Spectroscopy YAML section (absorption first).
//
// Direction normalisation contract:
//   - ‖d‖ < 1e-12   -> hard error (zero-dipole pigment is meaningless input;
//                      a separate check handles a *runtime-time* zero direction in
//                      compute_linear_response, not a per-pigment zero in YAML)
//   - |‖d‖ - 1| > 1e-6 -> warn + auto-normalise (user wrote a non-unit vector;
//                         strengths are meant to carry the magnitude)
// ---------------------------------------------------------------------------
// Expand one 2DES waiting-time entry: either a single point "T" or a closed
// range "start:step:stop" (step > 0, must divide stop-start exactly; both
// endpoints included). Cross-entry monotonicity and time.dt divisibility are
// checked later in resolve_2des_config (dt is not known here).
static void expand_tw_entry(const std::string& s, std::vector<double>& out) {
    auto bad = [&](const char* why) {
        throw std::runtime_error("spectroscopy.tw entry \"" + s + "\": " + why);
    };
    std::vector<std::string> tok;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, ':')) tok.push_back(cur);
    try {
        if (tok.size() == 1) {
            out.push_back(std::stod(tok[0]));
        } else if (tok.size() == 3) {
            const double start = std::stod(tok[0]);
            const double step  = std::stod(tok[1]);
            const double stop  = std::stod(tok[2]);
            if (step <= 0.0)   bad("step must be > 0");
            if (stop < start)  bad("stop must be >= start");
            const double n = (stop - start) / step;
            if (std::abs(n - std::round(n)) > 1e-6)
                bad("step must divide (stop - start) exactly");
            const int cnt = static_cast<int>(std::round(n));
            for (int i = 0; i <= cnt; ++i) out.push_back(start + i * step);
        } else {
            bad("expected \"T\" or \"start:step:stop\"");
        }
    } catch (const std::invalid_argument&) {
        bad("non-numeric value");
    }
}

// Parse the 2DES `tw` schedule node (a single entry scalar or a sequence of
// entries) into an ordered list of waiting times (fs).
static std::vector<double> expand_tw_schedule(const YAML::Node& tw) {
    std::vector<double> out;
    if (tw.IsSequence())
        for (const auto& it : tw) expand_tw_entry(it.as<std::string>(), out);
    else
        expand_tw_entry(tw.as<std::string>(), out);
    return out;
}

// Parse the 2DES axis parameters from the `2des:` block into a TwoDESConfig.
// Dipoles/strengths are NOT here (they are system.dipoles, parsed by
// parse_dipoles). resolve_2des_config validates spacing/divisibility and fills
// defaults for pathways/signals/tw afterwards.
static void parse_2des_axes(const YAML::Node& node, TwoDESConfig& spec) {
    auto rd = [&](const char* k, double& dst) {
        if (node[k]) dst = node[k].as<double>();
    };
    rd("tau_max", spec.tau_max);  rd("dtau", spec.dtau);
    rd("t_max",   spec.t_max);    rd("dt_detect", spec.dt_detect);
    if (node["tw"]) spec.tw_fs = expand_tw_schedule(node["tw"]);
    if (node["pathways"])
        for (const auto& p : node["pathways"]) spec.pathways.push_back(p.as<std::string>());
    if (node["signals"])
        for (const auto& s : node["signals"])  spec.signals.push_back(s.as<std::string>());
    // 'isotropic' is the only supported orientation model. The old 'none'
    // (single lab=mol, μx+μy+μz) was a non-physical debug shortcut — removed;
    // smoke-test with a small tau/t grid instead.
    if (node["orientation"]) {
        const auto o = node["orientation"].as<std::string>();
        if (o != "isotropic")
            throw std::runtime_error(
                "2des.orientation: only 'isotropic' is supported (got '" + o +
                "'); the 'none' shortcut was removed — use a small grid to smoke-test");
    }
    if (node["orient_reuse"]) spec.orient_reuse = node["orient_reuse"].as<bool>();
    if (node["batch"]) spec.batch = node["batch"].as<bool>();
}

// Parse one `output.<task>` subblock (save_path / filename / use_timestamp /
// verbose_memory) into an OutputConfig.
static OutputConfig parse_output_block(const YAML::Node& blk) {
    OutputConfig oc;
    oc.save_path = blk["save_path"].as<std::string>();
    if (!oc.save_path.empty() && oc.save_path.back() != '/')
        oc.save_path += '/';
    oc.filename = blk["filename"].as<std::string>("auto");
    oc.use_timestamp = blk["use_timestamp"].as<bool>(true);
    // Optional, default false (compact budget print).
    oc.verbose_memory = blk["verbose_memory"].as<bool>(false);
    return oc;
}

// Parse the `output:` sub-block of a task block into cfg.output. Every task
// block (dynamics/absorption/2des) is self-contained and MUST carry its own
// output. Hard-fail (no silent default) when absent.
static void parse_task_output(const YAML::Node& task_blk, const char* task,
                              Config& cfg) {
    if (!task_blk["output"])
        throw std::runtime_error(
            std::string("'") + task + "' task requires an 'output:' sub-block "
            "inside the '" + task + ":' section (save_path is required)");
    cfg.output = parse_output_block(task_blk["output"]);
}

// Parse the REQUIRED per-task `integrator` key. "rk4" = plain historical path
// (bit-identical); "if-rk4" = integrating-factor RK4 (Term 2 exact, larger dt at
// an accuracy cost). No silent default: a missing key forces the user to make an
// explicit choice (no "commented-out pseudo-option"). Only field-free
// propagation tasks are permitted (all current tasks qualify).
static void parse_integrator(const YAML::Node& blk, const char* task,
                             Config& cfg) {
    if (!blk["integrator"])
        throw std::runtime_error(std::string("'") + task +
            "' task requires " + task +
            ".integrator (explicit choice, no default): 'rk4' or 'if-rk4'");
    const std::string v = blk["integrator"].as<std::string>();
    if (v == "rk4") {
        cfg.time.integrator = Integrator::Rk4;
    } else if (v == "if-rk4") {
        cfg.time.integrator = Integrator::IfRk4;
    } else {
        throw std::runtime_error(std::string("'") + task +
            "'.integrator must be 'rk4' or 'if-rk4', got '" + v + "'");
    }
}

// Parse the per-task time grid for a STEPPED task (dynamics / absorption): dt,
// n_steps, output_interval are required; print_interval defaults to 0 (off).
// Validated inline so a bad grid fails at parse, not deep in the solver.
static void parse_stepped_time(const YAML::Node& blk, const char* task,
                               Config& cfg) {
    for (const char* k : {"dt", "n_steps", "output_interval"}) {
        if (!blk[k])
            throw std::runtime_error(
                std::string("'") + task + "' task requires " + task + "." + k);
    }
    cfg.time.dt = blk["dt"].as<double>();
    cfg.time.dt_au = cfg.time.dt * FS_TO_AU;
    cfg.time.n_steps = blk["n_steps"].as<int>();
    cfg.time.output_interval = blk["output_interval"].as<int>();
    cfg.time.print_interval = blk["print_interval"].as<int>(0);

    if (cfg.time.dt <= 0.0)
        throw std::runtime_error(std::string("'") + task + "'.dt must be > 0");
    if (cfg.time.n_steps <= 0)
        throw std::runtime_error(std::string("'") + task + "'.n_steps must be > 0");
    if (cfg.time.output_interval <= 0 ||
        cfg.time.output_interval > cfg.time.n_steps)
        throw std::runtime_error(std::string("'") + task +
            "'.output_interval must be in [1, n_steps]");
    if (cfg.time.print_interval < 0)
        throw std::runtime_error(std::string("'") + task +
            "'.print_interval must be >= 0 (0 disables printing)");
    if (cfg.filter.enabled && cfg.filter.start_step > cfg.time.n_steps)
        throw std::runtime_error(std::string("'") + task +
            "': filter.start_step must be <= n_steps");
    parse_integrator(blk, task, cfg);
}

// Parse the 2DES time grid: only dt / print_interval are meaningful (the driver
// derives step counts from the τ/t grid). n_steps / output_interval stay 0.
static void parse_2des_time(const YAML::Node& blk, Config& cfg) {
    if (!blk["dt"])
        throw std::runtime_error("'2des' task requires 2des.dt");
    cfg.time.dt = blk["dt"].as<double>();
    cfg.time.dt_au = cfg.time.dt * FS_TO_AU;
    cfg.time.print_interval = blk["print_interval"].as<int>(0);
    cfg.time.n_steps = 0;
    cfg.time.output_interval = 0;
    if (cfg.time.dt <= 0.0)
        throw std::runtime_error("'2des'.dt must be > 0");
    if (cfg.time.print_interval < 0)
        throw std::runtime_error("'2des'.print_interval must be >= 0");
    parse_integrator(blk, "2des", cfg);
}

// Parse the optional top-level `static_disorder:` block (MC static disorder).
// Only key is `draws_file` — the node file the MC path reads. A bare
// filename resolves next to the config; an absolute path is used verbatim. The
// block carries NO on/off switch: the run mode is chosen by the invocation
// (single-shot `heom_solver <task> <cfg>` vs MC `--rank N` / mc-disorder-run).
static void parse_static_disorder(const YAML::Node& node, Config& cfg) {
    if (!node) return;                       // block absent ⇒ single-shot only
    const YAML::Node df = node["draws_file"];
    if (!df) return;                         // block present but no file ⇒ inert
    const std::string raw = df.as<std::string>();
    if (raw.empty()) return;
    std::filesystem::path p(raw);
    cfg.disorder_draws_file = p.is_absolute()
        ? raw
        : (std::filesystem::path(cfg.yaml_path).parent_path() / p).string();
    // How many / which realizations to consume is a run-time knob (`--range`,
    // 1-based inclusive), not config — the queue interval lives on the CLI.
}

// ---------------------------------------------------------------------------
// Filename helpers: used to be `static` inside this
// TU and called unconditionally from `Config::load_yaml`. They are now
// non-static so main.cpp's task-table dispatch can call them at the
// right point — AFTER load_yaml() but BEFORE SystemContext construction —
// per task. Spectroscopy tasks have their own naming rule and do NOT call
// `resolve_dynamics_auto_filename`.
// ---------------------------------------------------------------------------

std::string extract_system_name(const std::string& yaml_path) {
    size_t last_slash = yaml_path.find_last_of("/");
    std::string fname = (last_slash != std::string::npos)
        ? yaml_path.substr(last_slash + 1) : yaml_path;
    size_t dot = fname.find_last_of(".");
    if (dot != std::string::npos) fname = fname.substr(0, dot);
    size_t underscore = fname.find('_');
    return (underscore != std::string::npos) ? fname.substr(0, underscore) : fname;
}

// Auto-filename method marker. Padé is the project's primary run mode and gets
// NO marker; Matsubara auto-names carry a trailing `_matsubara` so a later
// Matsubara re-run never silently overwrites a Padé output (and vice versa).
// Custom (non-"auto") filenames are the user's responsibility and untouched.
static std::string bath_method_suffix(const Config& cfg) {
    return cfg.physics.expansion == BathExpansion::MATSUBARA ? "_matsubara" : "";
}

std::string resolve_dynamics_auto_filename(const Config& cfg) {
    if (!cfg.initial.has_value()) {
        // Defence-in-depth — `resolve_dynamics_config()` already gates on
        // this. Phrase the message so the user knows which task they were
        // asking for when load_yaml claimed to succeed.
        throw std::runtime_error(
            "resolve_dynamics_auto_filename: cfg.initial is empty. "
            "'dynamics' task requires 'initial_state:' in yaml.");
    }
    const InitialState& init = *cfg.initial;

    if (init.type != InitialState::Type::SITE_EXCITATION &&
        init.type != InitialState::Type::EXCITON_EXCITATION) {
        throw std::runtime_error(
            "Auto filename only supports site_excitation and exciton_excitation. "
            "For coherent/custom_matrix, set filename to a custom string.");
    }

    std::string sys = extract_system_name(cfg.yaml_path);

    std::string excitation;
    if (init.type == InitialState::Type::SITE_EXCITATION) {
        excitation = "site" + std::to_string(
            hilbert_to_pigment(init.excited_site, cfg.physics.basis));
    } else {
        excitation = "exciton" + std::to_string(
            single_exciton_user_index_1based(init.exciton_index,
                                             cfg.physics.basis));
    }

    // Auto-name uses site 0 as the representative (Default for homogeneous runs;
    // for per-site runs the filename can only carry one value — site 0 stands in,
    // the full per-site table lives in HDF5 metadata / print_summary).
    const BathSite& s0 = cfg.physics.bath_sites.at(0);
    int temp = static_cast<int>(std::round(cfg.physics.temperature_K));
    int eta  = static_cast<int>(std::round(s0.eta_cm));
    int ginv = static_cast<int>(std::round(s0.gamma_inverse_fs));

    std::string prefix;
    if (cfg.physics.sink.enabled) prefix += "sink_";

    return prefix + sys + "_" + excitation + "_"
        + std::to_string(temp) + "K_eta" + std::to_string(eta)
        + "_ginv" + std::to_string(ginv) + bath_method_suffix(cfg);
}

// ============================================================================
// Main loader
// ============================================================================

Config Config::load_yaml(const std::string& yaml_path) {
    Config cfg;
    cfg.yaml_path = yaml_path;

    try {
        cfg.config_yaml = read_text_file(yaml_path);

        YAML::Node yaml = YAML::Load(cfg.config_yaml);
        // Physics identity, NOT a raw-text hash: stable across comment/format
        // edits (see Config::config_hash). config_yaml keeps the exact input.
        cfg.config_hash = physics_hash(yaml);

        Eigen::MatrixXcd H_site_cm;                 // bare site-basis H_S (cm^-1)
        parse_system(yaml["system"], cfg, H_site_cm);
        parse_bath(yaml["bath"], cfg);              // fills per-site η → λ_n = η_n/2
        build_effective_hamiltonian(cfg, H_site_cm);  // H_eff = H_S + H_reorg, then − w0

        // Expand H into the initial/default basis (parse_system set it from an
        // optional system.basis:, else EXCITED_ONLY) so load-time consumers
        // (sink site index, initial_state parsing, validate) and direct
        // SystemContext callers (tests) see a consistent expanded H.
        // resolve_<task>_config may override via apply_basis() before
        // SystemContext is built (absorption->with_ground, 2des->double_excited).
        apply_basis(cfg, cfg.physics.basis);

        parse_filter(yaml["filter"], cfg);
        parse_static_disorder(yaml["static_disorder"], cfg);

        // Task blocks (dynamics / absorption / 2des) are NOT parsed here: a
        // single yaml can serve all three, and load_yaml does not know which
        // task will run. Each resolve_<task>_config() (called from main.cpp
        // BEFORE SystemContext) re-loads cfg.config_yaml and parses only its
        // own block — filling cfg.time / cfg.output / cfg.initial / cfg.twodes
        // and its task-specific validation + auto-filename. One task, one block,
        // no per-task map.

    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML error in '" + yaml_path + "': " + e.what());
    } catch (const std::exception& e) {
        throw std::runtime_error("Config error: " + std::string(e.what()));
    }

    cfg.validate();
    return cfg;
}

void Config::validate() const {
    // Validate Hamiltonian (expanded space)
    int N_full = physics.n_sites;
    if (physics.hamiltonian.rows() != N_full || physics.hamiltonian.cols() != N_full) {
        throw std::runtime_error(
            "Hamiltonian size mismatch: expected " + std::to_string(N_full) + "x" + 
            std::to_string(N_full) + ", got " + std::to_string(physics.hamiltonian.rows()) + 
            "x" + std::to_string(physics.hamiltonian.cols())
        );
    }
    
    // Check Hermiticity
    for (int i = 0; i < N_full; ++i) {
        for (int j = i; j < N_full; ++j) {
            auto diff = physics.hamiltonian(i, j) - std::conj(physics.hamiltonian(j, i));
            if (std::abs(diff) > 1e-10) {
                throw std::runtime_error("Hamiltonian not Hermitian at (" + 
                    std::to_string(i) + "," + std::to_string(j) + ")");
            }
        }
    }
    
    // Validate other parameters
    if (physics.n_states <= 0) throw std::runtime_error("n_states must be > 0");
    if (physics.n_sites <= 0) throw std::runtime_error("n_sites must be > 0");
    if (physics.temperature_K <= 0) throw std::runtime_error("temperature must be > 0");
    if (static_cast<int>(physics.bath_sites.size()) != physics.n_states)
        throw std::runtime_error("bath_sites must have one entry per pigment "
                                 "(expected " + std::to_string(physics.n_states) + ")");
    {
        // Per-site sanity: eta / gamma must be positive on every site.
        for (const auto& s : physics.bath_sites) {
            if (s.eta_au <= 0) throw std::runtime_error("bath eta must be > 0 (per site)");
            if (s.gamma_au <= 0) throw std::runtime_error("bath gamma must be > 0 (per site)");
        }
    }
    // Defensive backstop matching parse_bath's range gate (same field, so no
    // "parse ok / validate misses" window opens).
    if (physics.n_extra_poles < 0 || physics.n_extra_poles > 50)
        throw std::runtime_error("n_extra_poles must be in [0, 50]");
    // Occupation cutoff range check.
    if (L_cutoff < 0)
        throw std::runtime_error("L_cutoff must be ≥ 0 (got "
            + std::to_string(L_cutoff) + ")");
    if (L_cutoff >= 127)
        throw std::runtime_error("L_cutoff (" + std::to_string(L_cutoff)
            + ") must be < 127 (int8_t storage limit in HierarchyIndex)");

    // NOTE: time grid, filter knobs, and initial-state checks are NOT here.
    // Time + output + initial/twodes are parsed per-task by resolve_<task>_config
    // (validated inline there); the filter block is validated inline in
    // parse_filter. validate() only guards the shared physics schema.
}

// ---------------------------------------------------------------------------
// Shared formatting helpers (used by both error messages and print_summary).
// ---------------------------------------------------------------------------

static const char* basis_to_string(BasisType b) {
    switch (b) {
        case BasisType::EXCITED_ONLY:    return "excited_only";
        case BasisType::WITH_GROUND:     return "with_ground";
        case BasisType::DOUBLE_EXCITED:  return "double_excited";
        default: return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Task-level config resolvers. Called from
// src/main.cpp's task-table dispatch BEFORE SystemContext sys(cfg). The
// SystemContext keeps a `const Config&`; mutating cfg AFTER the ctor would
// have no effect on the writer. Any cfg.* mutation a task needs
// — task-required-section checks, "auto" filename resolution — happens here.
// ---------------------------------------------------------------------------

void apply_basis(Config& cfg, BasisType basis) {
    cfg.physics.basis   = basis;
    cfg.physics.n_sites = compute_hilbert_dim(cfg.physics.n_states, basis);
    cfg.physics.hamiltonian =
        expand_hamiltonian(cfg.physics.hamiltonian_site, basis);
}

// Re-parse the running task's block from cfg.config_yaml. A single yaml can
// carry all three task blocks; load_yaml parsed only the shared blocks, so each
// resolver pulls exactly its own block here (one task, one parse — no per-task
// map). Throws if the block is absent.
static YAML::Node task_block(const Config& cfg, const char* task) {
    YAML::Node blk = YAML::Load(cfg.config_yaml)[task];
    if (!blk || !blk.IsMap())
        throw std::invalid_argument(
            std::string("'") + task + "' task requires a '" + task +
            ":' section in " + cfg.yaml_path);
    return blk;
}

void resolve_dynamics_config(Config& cfg) {
    YAML::Node blk = task_block(cfg, "dynamics");
    parse_stepped_time(blk, "dynamics", cfg);
    parse_task_output(blk, "dynamics", cfg);
    // Sink site index is basis-aware; the load-time (default) basis is already
    // applied, and dynamics keeps it (no override).
    parse_sink(blk, cfg);

    if (!blk["initial_state"])
        throw std::invalid_argument(
            "'dynamics' task requires an 'initial_state:' sub-block "
            "inside the 'dynamics:' section");
    InitialState init{};
    parse_initial_state(blk["initial_state"], cfg, init);
    cfg.initial = std::move(init);

    if (cfg.output.filename == "auto")
        cfg.output.filename = resolve_dynamics_auto_filename(cfg);
}

void resolve_absorption_config(Config& cfg) {
    YAML::Node blk = task_block(cfg, "absorption");
    parse_stepped_time(blk, "absorption", cfg);
    parse_task_output(blk, "absorption", cfg);
    // Batch the ≤3 non-zero polarization directions into one propagation.
    // Default false ⇒ historical serial 3-direction loop, bit-identical.
    if (blk["batch"])
        cfg.absorption_batch = blk["batch"].as<bool>();
    if (cfg.physics.dipoles.empty())
        throw std::invalid_argument(
            "'absorption' task requires system.dipoles / system.strengths");
    // Linear response needs |g><g| as the equilibrium reference, so absorption
    // always runs in the with_ground basis (re-expands H accordingly).
    apply_basis(cfg, BasisType::WITH_GROUND);

    if (cfg.output.filename == "auto")
        cfg.output.filename = extract_system_name(cfg.yaml_path) + "_absorption"
            + bath_method_suffix(cfg);
}

void resolve_2des_config(Config& cfg) {
    YAML::Node blk = task_block(cfg, "2des");
    parse_2des_time(blk, cfg);            // dt + print_interval (no n_steps grid)
    parse_task_output(blk, "2des", cfg);
    if (cfg.physics.dipoles.empty())
        throw std::invalid_argument(
            "'2des' task requires system.dipoles / system.strengths");

    TwoDESConfig spec{};
    parse_2des_axes(blk, spec);

    // ESA enters the |f><e| (double-excited) manifold, so main builds the
    // double_excited context; the 2des driver builds a second with_ground
    // context for SE/GSB (whose {g,e} sub-block is element-identical).
    apply_basis(cfg, BasisType::DOUBLE_EXCITED);

    const double dt = cfg.time.dt;  // fs
    // Axis sampling spacing: must be > 0, >= the HEOM step dt (you cannot
    // sample finer than you integrate), and an integer multiple of dt (so a
    // whole number of internal steps lands on every sample).
    auto check_mult = [&](const char* name, double step) {
        if (step <= 0.0)
            throw std::invalid_argument(
                std::string("'2des' task: 2des.") + name +
                " must be > 0 (got " + std::to_string(step) + ")");
        if (step < dt - 1e-9)
            throw std::invalid_argument(
                std::string("'2des' task: 2des.") + name + " (" +
                std::to_string(step) + " fs) must be >= dt (" +
                std::to_string(dt) + " fs)");
        const double ratio = step / dt;
        if (std::abs(ratio - std::round(ratio)) > 1e-6)
            throw std::invalid_argument(
                std::string("'2des' task: 2des.") + name + " (" +
                std::to_string(step) + " fs) must be an integer multiple of "
                "2des.dt (" + std::to_string(dt) + " fs)");
    };
    check_mult("dtau", spec.dtau);
    check_mult("dt_detect", spec.dt_detect);

    // Axis upper bound: must be a non-negative integer multiple of its sampling
    // spacing, so the last sample lands exactly on it (n = max/step + 1 points).
    auto check_axis_max = [&](const char* max_name, double max_val,
                              const char* step_name, double step) {
        if (max_val < 0.0)
            throw std::invalid_argument(
                std::string("'2des' task: 2des.") + max_name +
                " must be >= 0 (got " + std::to_string(max_val) + ")");
        const double ratio = max_val / step;
        if (std::abs(ratio - std::round(ratio)) > 1e-6)
            throw std::invalid_argument(
                std::string("'2des' task: 2des.") + max_name + " (" +
                std::to_string(max_val) + " fs) must be an integer multiple of "
                "2des." + step_name + " (" + std::to_string(step) + " fs)");
    };
    check_axis_max("tau_max", spec.tau_max, "dtau", spec.dtau);
    check_axis_max("t_max",   spec.t_max,   "dt_detect", spec.dt_detect);

    // Waiting-time schedule: default to a single T=0 point if unspecified.
    // Each sample must be >= 0, strictly increasing (no duplicates/reordering),
    // and an integer multiple of dt (carry-through propagates whole steps).
    if (spec.tw_fs.empty()) spec.tw_fs = {0.0};
    double prev_tw = -1.0;
    for (double tw : spec.tw_fs) {
        if (tw < 0.0)
            throw std::invalid_argument(
                "'2des' task: 2des.tw values must be >= 0 (got " +
                std::to_string(tw) + ")");
        if (tw <= prev_tw)
            throw std::invalid_argument(
                "'2des' task: 2des.tw must be strictly increasing "
                "(no duplicates/reordering); offending value " +
                std::to_string(tw) + " fs");
        const double ratio = tw / dt;
        if (tw > 0.0 && std::abs(ratio - std::round(ratio)) > 1e-6)
            throw std::invalid_argument(
                "'2des' task: 2des.tw value (" + std::to_string(tw) +
                " fs) must be an integer multiple of 2des.dt (" +
                std::to_string(dt) + " fs)");
        prev_tw = tw;
    }

    if (spec.pathways.empty()) spec.pathways = {"SE", "GSB", "ESA"};
    if (spec.signals.empty())  spec.signals  = {"rephasing", "nonrephasing"};

    cfg.twodes = std::move(spec);

    if (cfg.output.filename == "auto")
        cfg.output.filename =
            extract_system_name(cfg.yaml_path) + "_2des" + bath_method_suffix(cfg);
}

static void print_kv(const std::string& key, const std::string& val) {
    // The trailing `<< std::right` this used to end with restored an ASSUMED
    // default rather than the caller's actual adjustfield. The guard restores
    // what was really there.
    const heom::StreamFormat fmt_guard(std::cout);
    std::cout << "  " << std::left << std::setw(28) << key << " " << val << "\n";
}

void Config::print_summary() const {
    auto fmt_d = [](double v, int prec = 1) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(prec) << v;
        return s.str();
    };

    auto fmt_sci = [](double v, int prec = 1) {
        std::ostringstream s;
        s << std::scientific << std::setprecision(prec) << v;
        return s.str();
    };

    std::cout << "\n=== System Configuration ===\n\n";

    std::cout << "[Physical System]\n";
    print_kv("Excited states (n_states):", std::to_string(physics.n_states));
    print_kv("Basis type:", basis_to_string(physics.basis));
    print_kv("Total dimensions (n_sites):", std::to_string(physics.n_sites));

    if (physics.basis == BasisType::WITH_GROUND) {
        std::cout << "    1 ground + " << physics.n_states << " excited\n";
    } else if (physics.basis == BasisType::DOUBLE_EXCITED) {
        int n_double = physics.n_states * (physics.n_states - 1) / 2;
        std::cout << "    1 ground + " << physics.n_states << " single + " << n_double << " double\n";
    }

    std::cout << "\n[Bath Parameters]\n";
    print_kv("Expansion method:",
             physics.expansion == BathExpansion::PADE ? "pade [K-1/K] PSD" : "matsubara");
    // Directly print the immutable user input. n_extra_poles is the
    // mode count as entered (== n_bath_modes_total); no reverse-engineering of
    // a mutated field.
    print_kv("Extra bath poles (K):", std::to_string(physics.n_extra_poles));

    // Homogeneous run → one compact block (mirrors legacy output). Heterogeneous
    // → a per-site table. Homogeneity is "all sites equal site 0".
    const auto& sites = physics.bath_sites;
    auto same = [](const BathSite& a, const BathSite& b) {
        return a.eta_cm == b.eta_cm && a.gamma_inverse_fs == b.gamma_inverse_fs;
    };
    bool homogeneous = true;
    for (const auto& s : sites) if (!same(s, sites[0])) { homogeneous = false; break; }

    if (homogeneous) {
        const BathSite& s = sites[0];
        print_kv("Drude coupling eta (J):", fmt_d(s.eta_cm) + " cm^-1 (all sites)");
        print_kv("  -> lambda_reorg (= eta/2):", fmt_d(s.eta_cm * 0.5) + " cm^-1");
        print_kv("Relaxation time (1/γ):", fmt_d(s.gamma_inverse_fs) + " fs");
        print_kv("Markov residue (a.u.):", fmt_sci(physics.markov_residue[0], 3));
    } else {
        std::cout << "  Per-site bath (1-based pigment):\n";
        std::cout << "    site   eta(cm^-1)   1/gamma(fs)   markov(a.u.)\n";
        for (int n = 0; n < static_cast<int>(sites.size()); ++n) {
            const BathSite& s = sites[n];
            std::ostringstream row;
            row << "    " << std::setw(4) << (n + 1)
                << std::setw(13) << fmt_d(s.eta_cm)
                << std::setw(14) << fmt_d(s.gamma_inverse_fs)
                << std::setw(15) << fmt_sci(physics.markov_residue[n], 3);
            std::cout << row.str() << "\n";
        }
    }

    if (physics.sink.enabled) {
        // Sink dump moved here from parse_sink (constructor cleanup).
        std::cout << "\n[Sink/Trap]\n";
        // Echo the user-facing 1-based pigment index, mirroring the YAML field.
        const int pigment_1b = (physics.basis == BasisType::EXCITED_ONLY)
                             ? physics.sink.site + 1
                             : physics.sink.site;
        print_kv("Target pigment (1-based):", std::to_string(pigment_1b));
        print_kv("Rate gamma_sink:", fmt_d(physics.sink.gamma_fs, 4) + " fs^-1");
        print_kv("tau_sink:", fmt_d(1.0 / physics.sink.gamma_fs) + " fs");
    } else {
        std::cout << "\n[Sink/Trap]            disabled\n";
    }

    std::cout << "\n[Hierarchy / Filter]\n";
    print_kv("Occupation cutoff (L):", std::to_string(L_cutoff)
             + "  (Σ_{n,k} α_{nk} ≤ L)");
    if (filter.enabled) {
        print_kv("ADO filtering:", "enabled");
        print_kv("  tolerance:", fmt_sci(filter.tolerance));
        print_kv("  warmup start_step:", std::to_string(filter.start_step));
        print_kv("  warmup factor:", fmt_d(filter.warmup_factor, 3));
        print_kv("  ulist gate:",
                 filter.ulist_tau > 0.0 ? fmt_d(filter.ulist_tau, 3) : "off");
    } else {
        print_kv("ADO filtering:", "disabled");
    }

    std::cout << "\n[Time Evolution]\n";
    print_kv("Time step (dt):", fmt_d(time.dt) + " fs");
    if (time.n_steps > 0) {
        print_kv("Number of steps:", std::to_string(time.n_steps));
        print_kv("Total time:", fmt_d(time.dt * time.n_steps) + " fs");
    } else {
        print_kv("Number of steps:", "(task-defined grid)");
    }
    print_kv("Temperature:", fmt_d(physics.temperature_K) + " K");

    std::cout << "\n[Output]\n";
    print_kv("Format:", "hdf5");
    print_kv("Directory:", output.save_path);
    print_kv("Filename:", output.filename);
    print_kv("Timestamp:", output.use_timestamp ? "yes" : "no");

    std::cout << "\n";
}

} // namespace HEOMConfig

