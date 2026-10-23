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
#include <initializer_list>
#include <type_traits>
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

// ---------------------------------------------------------------------------
// Scalar keys — one policy. Every value goes through decode(): a value of the
// wrong type fails with the yaml path ("bath.eta: cannot read 'abc' as a
// number"), never with yaml-cpp's "bad conversion" and a node nobody can
// locate. A number must also be finite: NaN passes every `x <= 0` range check
// downstream, and filter.tolerance = NaN silently dropped the whole hierarchy.
// require() fails on a missing key; get_or() returns the default for a missing
// key only. yaml-cpp's own `as<T>(default)` is not used anywhere: it returns
// the default on a decode failure too, so `enabled: ture` meant `false`.
// ---------------------------------------------------------------------------
template <class T> static const char* yaml_type_name();
template <> const char* yaml_type_name<int>()         { return "an integer"; }
template <> const char* yaml_type_name<double>()      { return "a number"; }
template <> const char* yaml_type_name<bool>()        { return "true/false"; }
template <> const char* yaml_type_name<std::string>() { return "a string"; }

template <class T>
static T decode(const YAML::Node& n, const std::string& path) {
    T v;
    try {
        v = n.as<T>();
    } catch (const YAML::Exception&) {
        throw std::runtime_error(path + ": cannot read '" + n.Scalar() + "' as "
                                 + yaml_type_name<T>());
    }
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::isfinite(v))
            throw std::runtime_error(path + " must be finite (got '" + n.Scalar() + "')");
    }
    return v;
}

template <class T>
static T require(const YAML::Node& parent, const char* key, const std::string& path) {
    const YAML::Node n = parent ? parent[key] : YAML::Node();
    if (!n) throw std::runtime_error(path + " is required");
    return decode<T>(n, path);
}

template <class T>
static T get_or(const YAML::Node& parent, const char* key, const std::string& path,
                T fallback) {
    const YAML::Node n = parent ? parent[key] : YAML::Node();
    return n ? decode<T>(n, path) : fallback;
}

// Every key of a block must be one its parser reads. Called after the block's
// dedicated checks, so a removed or moved key gets its own message first. A typo ('sink.enable', '2des.pathway') used to be
// dropped and the run went on with the default — a different problem from the
// one the user wrote down.
static void check_keys(const YAML::Node& blk, const std::string& where,
                       std::initializer_list<const char*> known) {
    if (!blk) return;
    if (!blk.IsMap()) throw std::runtime_error(where + " must be a map");
    for (const auto& kv : blk) {
        const std::string key = kv.first.as<std::string>();
        auto same = [&](const char* k) { return key == k; };
        if (std::any_of(known.begin(), known.end(), same)) continue;
        std::string list;
        for (const char* k : known) list += (list.empty() ? "" : ", ") + std::string(k);
        throw std::runtime_error(where + ": unknown key '" + key + "' (known: " + list + ")");
    }
}

static std::string at(const std::string& path, size_t i) {
    return path + "[" + std::to_string(i) + "]";
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

// "Same physics?" fingerprint. Keeps exactly what changes the numbers this
// task produces — `system`, `bath`, `filter` (minus its `log_per_slot` print
// knob) and the ONE task block being run (minus `output`, `print_interval`,
// `batch`) — canonicalises it and FNV-1a's the result. Everything else is
// dropped by construction: the other two task blocks (editing `dynamics.dt`
// must not fork an absorption MC group), the disorder-owned `static_disorder`
// block (tracked by disorder_identity), and print/batch/output knobs (they
// move bytes around, not physics). Stable across comment/whitespace/key-order
// edits. `task == nullptr` (load_yaml, task not yet known) hashes the shared
// core only; resolve_<task>_config() replaces it with the task-scoped hash.
// Mirrored byte-for-byte by heom-tools/heom_tools/common/config_hash.py; the two
// are pinned together by a shared golden vector in both test suites.
static std::string physics_hash(const YAML::Node& yaml, const char* task) {
    YAML::Node phys;
    for (const char* key : {"system", "bath", "filter"}) {
        if (yaml[key]) phys[key] = YAML::Clone(yaml[key]);
    }
    if (phys["filter"] && phys["filter"].IsMap()) phys["filter"].remove("log_per_slot");
    if (task && yaml[task]) {
        YAML::Node blk = YAML::Clone(yaml[task]);
        if (blk.IsMap()) {
            for (const char* knob : {"output", "print_interval", "batch"}) blk.remove(knob);
        }
        phys[task] = blk;
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
        const std::string where = at("system.dipoles", i);
        if (!d.IsSequence() || d.size() != 3)
            throw std::runtime_error(where + " must be a 3-vector [dx, dy, dz]");
        std::array<double, 3> v = {
            decode<double>(d[0], at(where, 0)), decode<double>(d[1], at(where, 1)),
            decode<double>(d[2], at(where, 2))
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
        cfg.physics.strengths[i] = decode<double>(strengths_yaml[i], at("system.strengths", i));
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
    cfg.physics.n_states = require<int>(sys, "n_states", "system.n_states");

    // basis is task-driven: production yamls omit it and each task picks one
    // (resolve_<task>_config -> apply_basis; 2des builds two). An explicit
    // system.basis: is still honored as the initial/default basis (handy for
    // tests and direct SystemContext use); a task resolver may override it.
    const std::string basis_str =
        get_or<std::string>(sys, "basis", "system.basis", "excited_only");
    if (basis_str == "excited_only")        cfg.physics.basis = BasisType::EXCITED_ONLY;
    else if (basis_str == "with_ground")    cfg.physics.basis = BasisType::WITH_GROUND;
    else if (basis_str == "double_excited") cfg.physics.basis = BasisType::DOUBLE_EXCITED;
    else throw std::runtime_error("Invalid basis type: " + basis_str);

    cfg.physics.temperature_K = require<double>(sys, "temperature", "system.temperature");
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
            H_excited_cm(i, j) = decode<double>(row[j], at(at("system.hamiltonian", i), j));
    }

    // w0 captured here, but applied later in build_effective_hamiltonian so it
    // references the reorg-shifted (H_eff) diagonal, not the bare H_S.
    cfg.physics.w0_cm = get_or<double>(sys, "w0", "system.w0", 0.0);

    // Optional transition dipoles (system property; spectroscopy tasks require them).
    parse_dipoles(sys, cfg);
    check_keys(sys, "system", {"n_states", "basis", "temperature", "hamiltonian", "w0",
                               "dipoles", "strengths"});

    H_site_cm_out = std::move(H_excited_cm);  // bare site-basis (cm^-1); finalized later
}

// λ_n = η_n/2 + Σ_m S_m ω₀m  (cm⁻¹). Reorganization of the WHOLE J_n, used
// both to shift H_S and to print. One formula; print_summary cannot drift.
static double site_reorg_cm(const BathSite& s) {
    double vib = 0.0;
    for (const VibMode& v : s.modes) vib += v.huang_rhys * v.omega_cm;
    return 0.5 * s.eta_cm + vib;
}

// Builds the effective system Hamiltonian and commits it to cfg.physics.hamiltonian.
//
//   H_eff = H_S + H_reorg ,   H_reorg = Σ_n λ_n |n⟩⟨n| ,   λ_n = η_n/2 + Σ_m S_m ω₀m
//
// λ_n is the reorganization energy of the WHOLE J_n(ω) = (1/π)∫J_n/ω: the Drude
// half-eta plus one S*omega_0 per underdamped vibrational mode. Without modes
// the second sum is exactly 0.0 (bit-identical energies).
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
    cfg.physics.reorg_cm.resize(n_states);
    for (int n = 0; n < n_states; ++n) {
        cfg.physics.reorg_cm[n] = site_reorg_cm(cfg.physics.bath_sites[n]);
        H_site_cm(n, n) += cfg.physics.reorg_cm[n];
    }
    for (int i = 0; i < n_states; ++i)
        H_site_cm(i, i) -= cfg.physics.w0_cm;                       // − w0   (vs H_eff)
    // Store the finalized site-basis H (a.u.) but DON'T expand yet — the basis
    // is task-chosen and applied later by apply_basis().
    cfg.physics.hamiltonian_site = H_site_cm * CM_TO_AU;
}

// Build / override one BathSite from raw (cm/fs) inputs, recomputing a.u.
static void set_bath_site(BathSite& s, double eta_cm, double gamma_inverse_fs,
                          int n_extra_poles) {
    if (gamma_inverse_fs <= 0.0)
        throw std::runtime_error("bath.gamma_inverse must be > 0 (got " +
                                 std::to_string(gamma_inverse_fs) + ")");
    if (n_extra_poles < 0 || n_extra_poles > 50)
        throw std::runtime_error("bath.n_extra_poles must be in [0, 50] (got " +
                                 std::to_string(n_extra_poles) + ")");
    s.eta_cm            = eta_cm;
    s.eta_au            = eta_cm * CM_TO_AU;
    s.gamma_inverse_fs  = gamma_inverse_fs;
    s.gamma_au          = 1.0 / (gamma_inverse_fs * FS_TO_AU);
    s.n_extra_poles     = n_extra_poles;
}

// One `vibrational_modes:` list entry → VibMode (units only; the physical
// gates γ < ω₀, ω₁/ω₀ ≥ 1e-3, S > 0 live in physics_utils::set_bo_pair so a
// PhysicalParameters built without the parser is checked the same way).
// gamma_inverse is 1/γ in fs with γ the C(t) decay rate — the same meaning as
// the Drude key. The mode's Lorentzian FWHM is 2γ.
static VibMode parse_vib_mode(const YAML::Node& node, const std::string& where) {
    if (!node.IsMap())
        throw std::runtime_error(where + ": each vibrational mode must be a map "
                                 "{omega_cm, huang_rhys, gamma_inverse}");
    for (const auto& f : node) {
        const std::string key = f.first.as<std::string>();
        if (key != "omega_cm" && key != "huang_rhys" && key != "gamma_inverse")
            throw std::runtime_error(where + ": unknown vibrational mode field '" + key
                + "'. Legal keys: omega_cm / huang_rhys / gamma_inverse (C(t) decay "
                  "time 1/γ in fs; FWHM_cm = 2γ_cm = 2·5308.8 / gamma_inverse).");
    }
    VibMode v;
    v.omega_cm         = require<double>(node, "omega_cm",      where + ".omega_cm");
    v.huang_rhys       = require<double>(node, "huang_rhys",    where + ".huang_rhys");
    v.gamma_inverse_fs = require<double>(node, "gamma_inverse", where + ".gamma_inverse");
    if (v.omega_cm <= 0.0)
        throw std::runtime_error(where + ": omega_cm must be > 0");
    if (v.huang_rhys <= 0.0)
        throw std::runtime_error(where + ": huang_rhys must be > 0 (got "
                                 + std::to_string(v.huang_rhys) + ")");
    if (v.gamma_inverse_fs <= 0.0)
        throw std::runtime_error(where + ": gamma_inverse must be > 0 (got "
                                 + std::to_string(v.gamma_inverse_fs) + ")");
    v.omega_au  = v.omega_cm * CM_TO_AU;
    v.lambda_au = v.huang_rhys * v.omega_au;
    v.gamma_au  = 1.0 / (v.gamma_inverse_fs * FS_TO_AU);
    return v;
}

static std::vector<VibMode> parse_vib_modes(const YAML::Node& list, const std::string& where) {
    if (!list.IsSequence())
        throw std::runtime_error(where + ": vibrational_modes must be a list");
    std::vector<VibMode> out;
    for (size_t i = 0; i < list.size(); ++i)
        out.push_back(parse_vib_mode(list[i], where + "[" + std::to_string(i) + "]"));
    return out;
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
    const std::string method = require<std::string>(bath, "expansion_method",
                                                    "bath.expansion_method");
    if (method == "pade") {
        cfg.physics.expansion = BathExpansion::PADE;
    } else if (method == "matsubara") {
        cfg.physics.expansion = BathExpansion::MATSUBARA;
    } else {
        throw std::runtime_error(
            "bath.expansion_method must be 'pade' or 'matsubara' (got '"
            + method + "').");
    }

    cfg.physics.n_extra_poles = require<int>(bath, "n_extra_poles", "bath.n_extra_poles");
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
    cfg.L_cutoff = require<int>(bath, "L_cutoff", "bath.L_cutoff");

    // The Drude-Lorentz form is the only overdamped J(ω) implemented; the key
    // is optional, but a different value would be a model this code ignores.
    const std::string jw = get_or<std::string>(bath, "spectral_density",
                                               "bath.spectral_density", "Drude-Lorentz");
    if (jw != "Drude-Lorentz")
        throw std::runtime_error(
            "bath.spectral_density: only 'Drude-Lorentz' is implemented (got '" + jw +
            "'); underdamped modes go in bath.vibrational_modes");

    // static_disorder_fwhm is no longer supported — reject on presence (any value).
    if (bath["static_disorder_fwhm"])
        throw std::runtime_error(
            "bath.static_disorder_fwhm is no longer supported: disorder-in-HEOM "
            "(ν=0 Matsubara mode) has been removed. Static disorder is now sampled "
            "via Monte Carlo on the Hamiltonian diagonal. Delete this key.");

    // --- Default per-site physics (applied to every site without an override) ---
    const double def_eta   = require<double>(bath, "eta",           "bath.eta");
    const double def_ginv  = require<double>(bath, "gamma_inverse", "bath.gamma_inverse");
    const std::vector<VibMode> def_modes = bath["vibrational_modes"]
        ? parse_vib_modes(bath["vibrational_modes"], "bath.vibrational_modes")
        : std::vector<VibMode>{};

    const int n_bath = cfg.physics.n_states;
    cfg.physics.bath_sites.assign(n_bath, BathSite{});
    for (auto& s : cfg.physics.bath_sites) {
        set_bath_site(s, def_eta, def_ginv, cfg.physics.n_extra_poles);
        s.modes = def_modes;
    }

    // --- Optional per-site overrides: sites: { <1-based idx>: {fields...} } ---
    // A key that is absent inherits the bath-level default; a present
    // vibrational_modes list REPLACES the default list (it does not append).
    if (bath["sites"]) {
        const YAML::Node& sites = bath["sites"];
        if (!sites.IsMap())
            throw std::runtime_error("bath.sites must be a map keyed by 1-based "
                                     "pigment index, e.g. `3: { eta: 35.0 }`.");
        for (const auto& kv : sites) {
            const int idx1 = decode<int>(kv.first, "bath.sites key");
            if (idx1 < 1 || idx1 > n_bath)
                throw std::runtime_error(
                    "bath.sites key " + std::to_string(idx1) +
                    " out of range [1, " + std::to_string(n_bath) + "].");
            const YAML::Node& ov = kv.second;
            const std::string where = "bath.sites[" + std::to_string(idx1) + "]";
            for (const auto& f : ov) {
                const std::string key = f.first.as<std::string>();
                if (key != "eta" && key != "gamma_inverse" && key != "n_extra_poles"
                    && key != "vibrational_modes")
                    throw std::runtime_error(
                        where + " has unknown field '" + key + "'. Only eta / "
                        "gamma_inverse / n_extra_poles / vibrational_modes may be "
                        "overridden per site (expansion_method / L_cutoff are global).");
            }
            BathSite& s = cfg.physics.bath_sites[idx1 - 1];
            const double eta  = get_or<double>(ov, "eta", where + ".eta", s.eta_cm);
            const double ginv = get_or<double>(ov, "gamma_inverse", where + ".gamma_inverse",
                                               s.gamma_inverse_fs);
            const int K_n = get_or<int>(ov, "n_extra_poles", where + ".n_extra_poles",
                                        s.n_extra_poles);
            set_bath_site(s, eta, ginv, K_n);
            if (ov["vibrational_modes"])
                s.modes = parse_vib_modes(ov["vibrational_modes"],
                                          where + ".vibrational_modes");
        }
    }
    check_keys(bath, "bath", {"expansion_method", "n_extra_poles", "L_cutoff",
                              "spectral_density", "eta", "gamma_inverse",
                              "vibrational_modes", "sites"});

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
    check_keys(sink, "dynamics.sink", {"enabled", "target_site", "rate"});
    sink_cfg.enabled = get_or<bool>(sink, "enabled", "dynamics.sink.enabled", false);

    if (!sink_cfg.enabled) {
        sink_cfg.gamma = 0.0;
        sink_cfg.gamma_fs = 0.0;
        sink_cfg.site = -1;
        return;
    }

    int user_site = require<int>(sink, "target_site", "dynamics.sink.target_site");
    if (user_site < 1 || user_site > cfg.physics.n_states)
        throw std::runtime_error("target_site (" + std::to_string(user_site) +
                                 ") out of range [1, " + std::to_string(cfg.physics.n_states) + "]");
    sink_cfg.site = pigment_to_hilbert(user_site, cfg.physics.basis);
    sink_cfg.gamma_fs = require<double>(sink, "rate", "dynamics.sink.rate");
    if (sink_cfg.gamma_fs <= 0.0) {
        throw std::runtime_error("sink.rate must be > 0 when sink.enabled is true (got "
                                 + std::to_string(sink_cfg.gamma_fs) + ")");
    }
    sink_cfg.gamma = sink_cfg.gamma_fs / FS_TO_AU;
}

// Parse the `filter:` block. `enabled` is the explicit on/off switch: when
// false, every knob is left at its default (no "set four things to 0"
// ceremony). When true, tolerance/ulist are validated inline (fail loud at
// parse). The ulist sub-feature still uses 0 = off.
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
    // Removed keys, rejected on presence even with the filter off: a config
    // that asks for a warmup must not run without one unnoticed.
    for (const char* key : {"start_step", "warmup_factor"})
        if (f[key])
            throw std::runtime_error(
                std::string("filter.") + key + " was removed: the filter has one "
                "threshold for the whole run. Delete the key; to recover the "
                "accuracy a warmup gave, lower filter.tolerance (about 10x for "
                "population dynamics). Configs written for heom-gpu <= 2.0.0 keep "
                "their warmup there. See "
                "the filter warmup-removal study notes.");
    check_keys(f, "filter", {"enabled", "tolerance", "ulist", "log_per_slot"});
    cfg.filter.enabled = require<bool>(f, "enabled", "filter.enabled");
    if (!cfg.filter.enabled) return;   // off ⇒ ignore all knobs, keep defaults

    cfg.filter.tolerance     = require<double>(f, "tolerance",
        "filter.tolerance (ADO max-norm threshold, e.g. 1e-6; needed when filter.enabled)");
    cfg.filter.ulist_tau     = get_or<double>(f, "ulist", "filter.ulist", 0.0);
    cfg.filter.log_per_slot  = get_or<bool>(f, "log_per_slot", "filter.log_per_slot", false);

    if (cfg.filter.tolerance <= 0.0)
        throw std::runtime_error("filter.tolerance must be > 0 when enabled");
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
    check_keys(node, "dynamics.initial_state", {"type", "excited_site"});
    // User input is 1-based PHYSICAL pigment index (consistent with sink.target_site).
    int user_site = require<int>(node, "excited_site", "dynamics.initial_state.excited_site");
    if (user_site < 1 || user_site > cfg.physics.n_states)
        throw std::runtime_error("excited_site (" + std::to_string(user_site) +
            ") out of range [1, " + std::to_string(cfg.physics.n_states) + "]");
    init.excited_site = pigment_to_hilbert(user_site, cfg.physics.basis);
}

static void parse_exciton_excitation(const YAML::Node& node,
                                     const Config& cfg,
                                     InitialState& init) {
    check_keys(node, "dynamics.initial_state", {"type", "exciton_index"});
    int user_idx = require<int>(node, "exciton_index", "dynamics.initial_state.exciton_index");
    if (cfg.physics.basis == BasisType::DOUBLE_EXCITED) {
        throw std::runtime_error(
            "exciton_excitation is not supported for basis=double_excited; "
            "single-exciton block mapping is not implemented");
    }
    if (user_idx < 1 || user_idx > cfg.physics.n_states)
        throw std::runtime_error("exciton_index (" + std::to_string(user_idx) +
            ") out of single-exciton range [1, " +
            std::to_string(cfg.physics.n_states) + "]");
    init.exciton_index = pigment_to_hilbert(user_idx, cfg.physics.basis);
}

static void parse_coherent(const YAML::Node& node,
                           const Config& cfg,
                           InitialState& init) {
    check_keys(node, "dynamics.initial_state",
               {"type", "coherent_coefficients", "coherent_format"});
    const std::string where = "dynamics.initial_state.coherent_coefficients";
    auto coeffs = node["coherent_coefficients"];
    if (!coeffs.IsSequence())
        throw std::runtime_error("coherent_coefficients must be a list");

    size_t n = coeffs.size();
    init.coherent_coeffs.resize(n);

    // Determine format: explicit "coherent_format" field takes priority,
    // otherwise auto-detect by checking if first element is a sequence.
    bool complex_format;
    if (node["coherent_format"]) {
        const std::string fmt = decode<std::string>(node["coherent_format"],
                                                    "dynamics.initial_state.coherent_format");
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
            const double re = decode<double>(coeffs[i][0], at(at(where, i), 0));
            const double im = decode<double>(coeffs[i][1], at(at(where, i), 1));
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
            const double p = decode<double>(coeffs[i], at(where, i));
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
    check_keys(node, "dynamics.initial_state", {"type", "matrix_elements"});
    int N = cfg.physics.n_sites;
    init.custom_matrix = Eigen::MatrixXcd::Zero(N, N);

    auto elems = node["matrix_elements"];
    if (!elems || !elems.IsSequence())
        throw std::runtime_error("custom_matrix type requires matrix_elements list");

    for (size_t idx = 0; idx < elems.size(); ++idx) {
        auto e = elems[idx];
        const std::string where = at("dynamics.initial_state.matrix_elements", idx);
        if (!e.IsSequence() || e.size() < 2)
            throw std::runtime_error("matrix_elements[" + std::to_string(idx) + "] format error");

        int i = decode<int>(e[0], at(where, 0)) - 1;
        int j = decode<int>(e[1], at(where, 1)) - 1;

        if (i < 0 || i >= N || j < 0 || j >= N)
            throw std::runtime_error("matrix_elements[" + std::to_string(idx) + "] index out of range");

        if (e.size() == 2) continue;

        std::complex<double> val;
        if (e[2].IsScalar()) {
            val = std::complex<double>(decode<double>(e[2], at(where, 2)), 0.0);
        } else if (e[2].IsSequence() && e[2].size() == 2) {
            val = std::complex<double>(decode<double>(e[2][0], at(at(where, 2), 0)),
                                       decode<double>(e[2][1], at(at(where, 2), 1)));
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
    std::string type_str = require<std::string>(node, "type", "dynamics.initial_state.type");

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
//                      A runtime-time zero direction in
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
        throw std::runtime_error("2des.tw entry \"" + s + "\": " + why);
    };
    std::vector<std::string> tok;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, ':')) tok.push_back(cur);
    auto num = [&](const std::string& t) {
        const double v = std::stod(t);
        if (!std::isfinite(v)) bad("values must be finite");
        return v;
    };
    try {
        if (tok.size() == 1) {
            out.push_back(num(tok[0]));
        } else if (tok.size() == 3) {
            const double start = num(tok[0]);
            const double step  = num(tok[1]);
            const double stop  = num(tok[2]);
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
        for (size_t i = 0; i < tw.size(); ++i)
            expand_tw_entry(decode<std::string>(tw[i], at("2des.tw", i)), out);
    else
        expand_tw_entry(decode<std::string>(tw, "2des.tw"), out);
    return out;
}

static std::vector<std::string> string_list(const YAML::Node& list, const std::string& path) {
    if (!list.IsSequence()) throw std::runtime_error(path + " must be a list");
    std::vector<std::string> out;
    for (size_t i = 0; i < list.size(); ++i)
        out.push_back(decode<std::string>(list[i], at(path, i)));
    return out;
}

// Parse the 2DES axis parameters from the `2des:` block into a TwoDESConfig.
// Dipoles/strengths are NOT here (they are system.dipoles, parsed by
// parse_dipoles). resolve_2des_config validates spacing/divisibility and fills
// defaults for pathways/signals/tw afterwards.
// The four axis parameters are required: a missing tau_max used to default to
// 0 and the run "succeeded" with a one-point axis.
static void parse_2des_axes(const YAML::Node& node, TwoDESConfig& spec) {
    spec.tau_max   = require<double>(node, "tau_max",   "2des.tau_max");
    spec.dtau      = require<double>(node, "dtau",      "2des.dtau");
    spec.t_max     = require<double>(node, "t_max",     "2des.t_max");
    spec.dt_detect = require<double>(node, "dt_detect", "2des.dt_detect");
    if (node["tw"]) spec.tw_fs = expand_tw_schedule(node["tw"]);
    if (node["pathways"]) spec.pathways = string_list(node["pathways"], "2des.pathways");
    if (node["signals"])  spec.signals  = string_list(node["signals"],  "2des.signals");
    // 'isotropic' is the only supported orientation model. The old 'none'
    // (single lab=mol, μx+μy+μz) was a non-physical debug shortcut — removed;
    // smoke-test with a small tau/t grid instead.
    const auto o = get_or<std::string>(node, "orientation", "2des.orientation", "isotropic");
    if (o != "isotropic")
        throw std::runtime_error(
            "2des.orientation: only 'isotropic' is supported (got '" + o +
            "'); the 'none' shortcut was removed — use a small grid to smoke-test");
    // Removed key: the B=1 path is always the reuse walk now,
    // and a key that used to pick the walk but no longer does must not be
    // silently accepted — the user would believe it still means something.
    if (node["orient_reuse"])
        throw std::runtime_error(
            "2des.orient_reuse was removed: the B=1 walk is always the 3->9->21 "
            "reuse walk (the flat 21-tree is gone). Delete the key; use "
            "2des.batch: true for the B=21 lockstep walk");
    // Walk width: true/false keep their meaning (21 / 1);
    // auto → kTwoDESBatchAuto (resolved in compute_2des). Not in config_hash;
    // write_2des records the resolved width from TwoDESResult.
    if (const YAML::Node b = node["batch"]) {
        if (b.IsScalar() && b.Scalar() == "auto") {
            spec.batch = kTwoDESBatchAuto;
        } else {
            int w = 0;
            try { w = b.as<int>(); }
            catch (const YAML::Exception&) {
                w = decode<bool>(b, "2des.batch") ? 21 : 1;
            }
            if (w != 1 && w != 7 && w != 21)
                throw std::runtime_error(
                    "2des.batch: " + b.Scalar() +
                    " is not a walk width (want true|false|21|7|1|auto)");
            spec.batch = w;
        }
    }
}

// Parse one `output.<task>` subblock (save_path / filename / use_timestamp /
// verbose_memory) into an OutputConfig.
static OutputConfig parse_output_block(const YAML::Node& blk, const char* task) {
    const std::string where = std::string(task) + ".output";
    OutputConfig oc;
    oc.save_path = require<std::string>(blk, "save_path", where + ".save_path");
    if (!oc.save_path.empty() && oc.save_path.back() != '/')
        oc.save_path += '/';
    oc.filename = get_or<std::string>(blk, "filename", where + ".filename", "auto");
    oc.use_timestamp = get_or<bool>(blk, "use_timestamp", where + ".use_timestamp", true);
    // Optional, default false (compact budget print).
    oc.verbose_memory = get_or<bool>(blk, "verbose_memory", where + ".verbose_memory", false);
    check_keys(blk, where, {"save_path", "filename", "use_timestamp", "verbose_memory"});
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
    cfg.output = parse_output_block(task_blk["output"], task);
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
    const std::string v = require<std::string>(blk, "integrator",
                                               std::string(task) + ".integrator");
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
// n_steps must be a whole number of records: the steps after the last record
// were propagated, never written, and never checked for divergence, so the
// run could report success on a state nobody looked at.
static void parse_stepped_time(const YAML::Node& blk, const char* task,
                               Config& cfg) {
    const std::string t(task);
    cfg.time.dt = require<double>(blk, "dt", t + ".dt");
    cfg.time.dt_au = cfg.time.dt * FS_TO_AU;
    cfg.time.n_steps = require<int>(blk, "n_steps", t + ".n_steps");
    cfg.time.output_interval = require<int>(blk, "output_interval", t + ".output_interval");
    cfg.time.print_interval = get_or<int>(blk, "print_interval", t + ".print_interval", 0);

    if (cfg.time.dt <= 0.0)
        throw std::runtime_error(std::string("'") + task + "'.dt must be > 0");
    if (cfg.time.n_steps <= 0)
        throw std::runtime_error(std::string("'") + task + "'.n_steps must be > 0");
    if (cfg.time.output_interval <= 0 ||
        cfg.time.output_interval > cfg.time.n_steps)
        throw std::runtime_error(std::string("'") + task +
            "'.output_interval must be in [1, n_steps]");
    if (cfg.time.n_steps % cfg.time.output_interval != 0)
        throw std::runtime_error(std::string("'") + task + "'.n_steps (" +
            std::to_string(cfg.time.n_steps) + ") must be a multiple of output_interval (" +
            std::to_string(cfg.time.output_interval) + ")");
    if (cfg.time.print_interval < 0)
        throw std::runtime_error(std::string("'") + task +
            "'.print_interval must be >= 0 (0 disables printing)");
    parse_integrator(blk, task, cfg);
}

// Parse the 2DES time grid: only dt / print_interval are meaningful (the driver
// derives step counts from the τ/t grid). n_steps / output_interval stay 0.
static void parse_2des_time(const YAML::Node& blk, Config& cfg) {
    cfg.time.dt = require<double>(blk, "dt", "2des.dt");
    cfg.time.dt_au = cfg.time.dt * FS_TO_AU;
    cfg.time.print_interval = get_or<int>(blk, "print_interval", "2des.print_interval", 0);
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
    check_keys(node, "static_disorder", {"draws_file"});
    const std::string raw = get_or<std::string>(node, "draws_file",
                                                "static_disorder.draws_file", "");
    if (raw.empty()) return;                 // block present but no file ⇒ inert
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
            hilbert_to_pigment(init.exciton_index, cfg.physics.basis));
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
        // Task not known yet ⇒ shared core (system/bath/filter) only;
        // resolve_<task>_config() replaces it with the task-scoped hash.
        cfg.config_hash = physics_hash(yaml, nullptr);

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
        check_keys(yaml, "config", {"system", "bath", "filter", "static_disorder",
                                    "dynamics", "absorption", "fluorescence", "2des"});

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
            if (s.n_extra_poles < 0 || s.n_extra_poles > 50)
                throw std::runtime_error("n_extra_poles must be in [0, 50] (per site)");
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
// Also the one place the task-scoped config_hash is set: every resolver comes
// through here, so no resolver can forget it.
static YAML::Node task_block(Config& cfg, const char* task) {
    YAML::Node yaml = YAML::Load(cfg.config_yaml);
    YAML::Node blk = yaml[task];
    if (!blk || !blk.IsMap())
        throw std::invalid_argument(
            std::string("'") + task + "' task requires a '" + task +
            ":' section in " + cfg.yaml_path);
    cfg.config_hash = physics_hash(yaml, task);
    return blk;
}

void resolve_dynamics_config(Config& cfg) {
    cfg.task = Config::Task::Dynamics;
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
    check_keys(blk, "dynamics", {"integrator", "dt", "n_steps", "output_interval",
                                 "print_interval", "output", "sink", "initial_state"});
    cfg.initial = std::move(init);

    if (cfg.output.filename == "auto")
        cfg.output.filename = resolve_dynamics_auto_filename(cfg);
}

void resolve_absorption_config(Config& cfg) {
    cfg.task = Config::Task::Absorption;
    YAML::Node blk = task_block(cfg, "absorption");
    parse_stepped_time(blk, "absorption", cfg);
    parse_task_output(blk, "absorption", cfg);
    // Batch the ≤3 non-zero polarization directions into one propagation.
    // Default false ⇒ historical serial 3-direction loop, bit-identical.
    cfg.linear_batch = get_or<bool>(blk, "batch", "absorption.batch", false);
    check_keys(blk, "absorption", {"integrator", "dt", "n_steps", "output_interval",
                                   "print_interval", "output", "batch"});
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

// fluorescence.equilibration: every bad value is reported with its key, value
// and legal range; the intervals must fit the dt grid (equilibrate re-checks).
static EquilibrationConfig parse_equilibration(const YAML::Node& blk, double dt) {
    const std::string p = "fluorescence.equilibration";
    if (!blk || !blk.IsMap())
        throw std::invalid_argument(
            "'fluorescence' task requires an 'equilibration:' sub-block (t_max, "
            "check_interval, tolerance)");
    EquilibrationConfig eq;
    eq.t_max_fs          = require<double>(blk, "t_max", p + ".t_max");
    eq.check_interval_fs = get_or<double>(blk, "check_interval", p + ".check_interval", 50.0);
    eq.tolerance         = get_or<double>(blk, "tolerance", p + ".tolerance", 1e-6);
    check_keys(blk, p, {"t_max", "check_interval", "tolerance"});

    auto bad = [&](const char* key, double v, const std::string& rule) {
        std::ostringstream os;
        os << p << "." << key << " = " << v << ": " << rule;
        throw std::invalid_argument(os.str());
    };
    if (!(eq.t_max_fs > 0.0)) bad("t_max", eq.t_max_fs, "must be > 0 fs");
    if (!(eq.check_interval_fs > 0.0)) bad("check_interval", eq.check_interval_fs, "must be > 0 fs");
    const double n = eq.check_interval_fs / dt;
    if (std::abs(n - std::round(n)) > 1e-9 * n)
        bad("check_interval", eq.check_interval_fs,
            "must be an integer multiple of fluorescence.dt = " + std::to_string(dt) + " fs");
    if (eq.t_max_fs < 2.0 * eq.check_interval_fs)
        bad("t_max", eq.t_max_fs, "must be ≥ 2·check_interval (two checks give the first ratio)");
    if (!(eq.tolerance > 0.0 && eq.tolerance < 1.0)) bad("tolerance", eq.tolerance, "must be in (0, 1)");
    return eq;
}

void resolve_fluorescence_config(Config& cfg) {
    cfg.task = Config::Task::Fluorescence;
    YAML::Node blk = task_block(cfg, "fluorescence");
    parse_stepped_time(blk, "fluorescence", cfg);
    // IF-RK4's fixed point is not the steady state: the bias grows as dt^5
    // (monomer 3.5e-2 at dt = 2 fs), exactly where IF-RK4 would pay off.
    if (cfg.time.integrator == Integrator::IfRk4)
        throw std::invalid_argument(
            "fluorescence.integrator = 'if-rk4' is rejected: its fixed point is biased "
            "against the steady state by O(dt^5) (3.5e-2 at dt = 2 fs on one pigment). "
            "Use 'rk4'.");
    // The filter pins δ at a floor whose ratio to filter.tolerance is system-specific
    // (FMO ~6x, LHC-II ~100x), so the equilibration criterion cannot be trusted.
    if (cfg.filter.enabled)
        throw std::invalid_argument(
            "'fluorescence' task requires filter.enabled = false: the ADO filter pins the "
            "equilibration distance δ at a floor 0.2-100x filter.tolerance depending on the "
            "system (LHC-II: ~1e-4 at filter.tolerance 1e-6), so equilibration cannot "
            "converge reliably.");
    parse_task_output(blk, "fluorescence", cfg);
    cfg.linear_batch = get_or<bool>(blk, "batch", "fluorescence.batch", false);
    cfg.equilibration = parse_equilibration(blk["equilibration"], cfg.time.dt);
    check_keys(blk, "fluorescence", {"integrator", "dt", "n_steps", "output_interval",
                                     "print_interval", "output", "batch", "equilibration"});
    if (cfg.physics.dipoles.empty())
        throw std::invalid_argument(
            "'fluorescence' task requires system.dipoles / system.strengths");
    // Emission ends in the (e,g) coherence, measured against |g><g|.
    apply_basis(cfg, BasisType::WITH_GROUND);

    if (cfg.output.filename == "auto")
        cfg.output.filename = extract_system_name(cfg.yaml_path) + "_fluorescence"
            + bath_method_suffix(cfg);
}

void resolve_2des_config(Config& cfg) {
    cfg.task = Config::Task::TwoDES;
    YAML::Node blk = task_block(cfg, "2des");
    parse_2des_time(blk, cfg);            // dt + print_interval (no n_steps grid)
    parse_task_output(blk, "2des", cfg);
    if (cfg.physics.dipoles.empty())
        throw std::invalid_argument(
            "'2des' task requires system.dipoles / system.strengths");

    TwoDESConfig spec{};
    parse_2des_axes(blk, spec);
    check_keys(blk, "2des", {"integrator", "dt", "print_interval", "output", "tau_max",
                             "dtau", "t_max", "dt_detect", "tw", "pathways", "signals",
                             "orientation", "batch"});

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
    // Directly print the immutable user input (the per-site default).
    print_kv("Extra bath poles (K):", std::to_string(physics.n_extra_poles) + " (default)");
    {
        std::ostringstream per;
        for (int n = 0; n < physics.n_bath(); ++n)
            per << (n ? "," : "") << physics.mode_layout().count(n);
        print_kv("Ladders D:", std::to_string(physics.n_ladders())
                 + "  (per site: " + per.str() + ")");
    }

    // Homogeneous run → one compact block (mirrors legacy output). Heterogeneous
    // → a per-site table. Homogeneity is "all sites equal site 0".
    const auto& sites = physics.bath_sites;
    auto same_mode = [](const VibMode& a, const VibMode& b) {
        return a.omega_cm == b.omega_cm && a.huang_rhys == b.huang_rhys
            && a.gamma_inverse_fs == b.gamma_inverse_fs;
    };
    auto same = [&](const BathSite& a, const BathSite& b) {
        if (a.eta_cm != b.eta_cm || a.gamma_inverse_fs != b.gamma_inverse_fs
            || a.n_extra_poles != b.n_extra_poles || a.modes.size() != b.modes.size())
            return false;
        for (size_t m = 0; m < a.modes.size(); ++m)
            if (!same_mode(a.modes[m], b.modes[m])) return false;
        return true;
    };
    bool homogeneous = true;
    for (const auto& s : sites) if (!same(s, sites[0])) { homogeneous = false; break; }

    // γ is the C(t) envelope decay rate (as for Drude); the mode's Lorentzian
    // FWHM is 2γ. Both printed so the log reads against either convention.
    auto fmt_g = [](double v) {   // shortest faithful form: 0.02, 1e-06, 0.5
        std::ostringstream s;
        s << std::setprecision(6) << v;
        return s.str();
    };
    auto print_modes = [&](const BathSite& s, const char* indent) {
        for (const VibMode& v : s.modes) {
            const double gamma_cm = v.gamma_au / CM_TO_AU;
            const double lam_m = v.huang_rhys * v.omega_cm;
            std::cout << indent << "mode: ω₀=" << fmt_d(v.omega_cm, 0) << " cm⁻¹  S="
                      << fmt_g(v.huang_rhys) << "  λ=Sω₀=" << fmt_d(lam_m)
                      << " cm⁻¹  γ=" << fmt_d(gamma_cm) << " cm⁻¹ (C(t) decay 1/γ="
                      << fmt_d(v.gamma_inverse_fs, 0) << " fs, FWHM 2γ="
                      << fmt_d(2.0 * gamma_cm) << " cm⁻¹)\n";
        }
    };

    if (homogeneous) {
        const BathSite& s = sites[0];
        const double vib = site_reorg_cm(s) - 0.5 * s.eta_cm;
        print_kv("Drude coupling eta (J):", fmt_d(s.eta_cm) + " cm^-1 (all sites)");
        print_kv("  -> lambda_reorg:",
                 fmt_d(site_reorg_cm(s)) + " cm^-1  (η/2=" + fmt_d(0.5 * s.eta_cm)
                 + " + ΣSω₀=" + fmt_d(vib) + ")");
        print_kv("Relaxation time (1/γ):", fmt_d(s.gamma_inverse_fs) + " fs");
        print_kv("Markov residue (a.u.):", fmt_sci(physics.markov_residue[0], 3));
        print_modes(s, "  ");
    } else {
        std::cout << "  Per-site bath (1-based pigment):\n";
        std::cout << "    site   eta(cm^-1)   1/gamma(fs)   K    lambda   markov(a.u.)\n";
        for (int n = 0; n < static_cast<int>(sites.size()); ++n) {
            const BathSite& s = sites[n];
            std::ostringstream row;
            row << "    " << std::setw(4) << (n + 1)
                << std::setw(13) << fmt_d(s.eta_cm)
                << std::setw(14) << fmt_d(s.gamma_inverse_fs)
                << std::setw(4)  << s.n_extra_poles
                << std::setw(10) << fmt_d(site_reorg_cm(s))
                << std::setw(15) << fmt_sci(physics.markov_residue[n], 3);
            std::cout << row.str() << "\n";
            print_modes(s, "         ");
        }
    }

    if (physics.sink.enabled) {
        // Sink dump moved here from parse_sink (constructor cleanup).
        std::cout << "\n[Sink/Trap]\n";
        // Echo the user-facing 1-based pigment index, mirroring the YAML field.
        print_kv("Target pigment (1-based):",
                 std::to_string(hilbert_to_pigment(physics.sink.site, physics.basis)));
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

