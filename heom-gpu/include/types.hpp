#ifndef HEOM_TYPES_HPP
#define HEOM_TYPES_HPP

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <complex>
#include <Eigen/Dense>

// Physical constants (atomic units)
constexpr double BOLTZMANN = 3.166811563e-6;    // a.u./K (k_B in atomic units)
constexpr double CM_TO_AU = 4.556335e-6;        // cm^-1 to a.u. conversion
constexpr double FS_TO_AU = 41.341374;          // fs to a.u. conversion
constexpr double PI = 3.14159265358979323846;

/**
 * @brief Hilbert space basis type for system Hamiltonian
 * 
 * Determines how the system Hamiltonian is expanded from the input
 * excited-state Hamiltonian:
 * 
 * - EXCITED_ONLY: Use only excited states (n_sites = n_states)
 * - WITH_GROUND: Add ground state at index 0 (n_sites = n_states + 1)
 * - DOUBLE_EXCITED: Add ground + double-excited states 
 *   (n_sites = 1 + n_states + n_states*(n_states-1)/2)
 */
enum class BasisType {
    EXCITED_ONLY,       // Only excited states
    WITH_GROUND,        // Ground + single-excited
    DOUBLE_EXCITED      // Ground + single-excited + double-excited
};

/**
 * @brief Bath-correlation exponential-expansion scheme.
 *
 * Both expand C(t)=Σ_{k=0}^K c_k e^{-ν_k t} with the SAME k=0 Drude pole and
 * the SAME downstream hierarchy/kernel layout (K+1 modes). Only the k=1..K
 * pole positions ν_k and weights c_k differ:
 *   - MATSUBARA: ν_k = 2πkT/ℏ (equally spaced; slow ~1/k² convergence).
 *   - PADE     : Padé [K-1/K] PSD poles (optimised distribution; same K
 *                converges far faster, shrinking the ADO count M).
 */
enum class BathExpansion {
    MATSUBARA,
    PADE
};

// Inverse mapping: convert internal 0-based Hilbert index back to the
// user-facing 1-based PHYSICAL pigment index. This is the inverse of
// parse_site_excitation() in heom_config.cpp and is the canonical
// representation for console output, HDF5 attributes, and any downstream
// post-processing that should mirror the YAML input.
//
//   EXCITED_ONLY            : pigment k → Hilbert k-1   ⇒ inverse k = h + 1
//   WITH_GROUND / DOUBLE_*  : pigment k → Hilbert k     ⇒ inverse k = h
inline int hilbert_to_pigment(int hilbert_idx, BasisType basis) {
    return (basis == BasisType::EXCITED_ONLY) ? hilbert_idx + 1 : hilbert_idx;
}

// Convert the internal eigenvector/Hilbert index used for single-exciton
// excitation back to the YAML-facing 1-based single-exciton index.
// EXCITED_ONLY has no ground slot; WITH_GROUND stores ground at index 0, so
// single excitons start at Hilbert/eigenvector index 1.
inline int single_exciton_user_index_1based(int internal_idx, BasisType basis) {
    return (basis == BasisType::EXCITED_ONLY) ? internal_idx + 1 : internal_idx;
}

// Hierarchy index alpha = (alpha_{n,k}) for n in [0, n_bath), k in [0, K]:
// a fixed-width row of D = n_bath*(K+1) occupation counts. int8_t is
// intentional: HEOMConfig::validate enforces L_cutoff < 127, so every entry
// fits in one byte and lies in [0, 126] — which is why the compares below can
// use memcmp (unsigned byte order) and still reproduce signed int8_t ordering.
//
// This is a *view*: the rows live in one contiguous HierarchyTable buffer.
// Ordering + std::lower_bound over that table replaces the old
// unordered_map<hash,int>, so there is no hash and no per-index cache.
class HierarchyIndex {
public:
    HierarchyIndex() = default;
    HierarchyIndex(const int8_t* row, int width) : row_(row), width_(width) {}

    int8_t operator[](int i) const {
        assert(i >= 0 && i < width_);
        return row_[i];
    }

    int get_size() const { return width_; }
    const int8_t* data() const { return row_; }

    int sum() const {
        int total = 0;
        for (int i = 0; i < width_; ++i) total += row_[i];
        return total;
    }

    bool operator==(const HierarchyIndex& o) const {
        return width_ == o.width_ && std::memcmp(row_, o.row_, width_) == 0;
    }
    bool operator<(const HierarchyIndex& o) const {
        assert(width_ == o.width_);
        return std::memcmp(row_, o.row_, width_) < 0;
    }

private:
    const int8_t* row_ = nullptr;
    int width_ = 0;
};

// Contiguous [M][width] table of hierarchy indices.
//
// One allocation for the whole hierarchy instead of one per index. At LHC-II
// scale (M ~ 9e6) the previous vector-of-vectors paid M malloc calls plus 32 B
// of std::vector header per row, and scattered the rows across the heap; the
// lower_bound in build_neighbor_maps then chased a pointer per probe. Here the
// same probe walks cache lines.
class HierarchyTable {
public:
    HierarchyTable() = default;
    explicit HierarchyTable(int width) : width_(width) {}

    int width() const { return width_; }
    size_t size() const {
        return width_ ? flat_.size() / static_cast<size_t>(width_) : 0;
    }
    bool empty() const { return flat_.empty(); }

    const int8_t* row(size_t i) const {
        return flat_.data() + i * static_cast<size_t>(width_);
    }
    HierarchyIndex operator[](size_t i) const { return {row(i), width_}; }
    const int8_t* data() const { return flat_.data(); }

    void append(const int8_t* row_data) {
        flat_.insert(flat_.end(), row_data, row_data + width_);
    }

    // Move row i to position dest[i], in place by cycle-following swaps.
    // dest is the *destination* permutation, not the source order: reordering
    // M rows must not need a second M*width buffer, which is the whole point
    // of the flat layout. dest is consumed (left as the identity).
    void permute_rows(std::vector<int>& dest) {
        assert(dest.size() == size());
        for (size_t i = 0; i < dest.size(); ++i) {
            while (dest[i] != static_cast<int>(i)) {
                const size_t j = static_cast<size_t>(dest[i]);
                std::swap_ranges(flat_.begin() + i * width_,
                                 flat_.begin() + (i + 1) * width_,
                                 flat_.begin() + j * width_);
                std::swap(dest[i], dest[j]);
            }
        }
    }

    class const_iterator {
    public:
        const_iterator(const int8_t* p, int width) : p_(p), width_(width) {}
        HierarchyIndex operator*() const { return {p_, width_}; }
        const_iterator& operator++() { p_ += width_; return *this; }
        bool operator!=(const const_iterator& o) const { return p_ != o.p_; }
    private:
        const int8_t* p_;
        int width_;
    };
    const_iterator begin() const { return {flat_.data(), width_}; }
    const_iterator end() const { return {flat_.data() + flat_.size(), width_}; }

private:
    std::vector<int8_t> flat_;
    int width_ = 0;
};

/**
 * @brief Physical parameters of the system
 * 
 * LINUS REFACTOR: All internal values stored in ATOMIC UNITS.
 * - GPU kernels only use a.u., so we store them directly
 * - Use conversion functions for cm^-1 output (if needed)
 * - Eliminates redundant data and sync issues
 * 
 * NEW DESIGN (basis expansion):
 * - n_states: Physical system dimension (excited states only)
 * - basis: Hilbert space type (excited_only/with_ground/double_excited)
 * - n_sites: Total Hilbert space dimension (auto-computed from n_states + basis)
 * - hamiltonian: Always full expanded Hamiltonian (n_sites × n_sites)
 */
struct SinkConfig {
    bool enabled = false;
    int site = -1;                                  // Hilbert space index (0-based)
    double gamma = 0.0;                             // Sink rate (a.u.)
    double gamma_fs = 0.0;                          // Sink rate in fs^-1 (for reference)
};

/**
 * @brief Per-site bath input parameters (Drude-Lorentz).
 *
 * Single source of truth for the site-dependent bath physics. One entry per
 * physical pigment n ∈ [0, n_bath). Both raw input units (cm⁻¹ / fs) and
 * atomic units are stored: the cm/fs forms feed print_summary / HDF5 metadata,
 * the a.u. forms feed compute_*_drude. The YAML "Default" + per-site `sites:`
 * overrides are merged at parse time into one BathSite per pigment; there is
 * NO separate global scalar (eliminates the homogeneous-vs-heterogeneous
 * special case — homogeneous is just "all rows equal").
 */
struct BathSite {
    double eta_cm = 0.0;            // Drude coupling η (cm⁻¹, input); λ_reorg = η/2
    double eta_au = 0.0;            // η (a.u.)
    double gamma_inverse_fs = 0.0;  // 1/γ (fs, input)
    double gamma_au = 0.0;          // γ (a.u.)
};

struct PhysicalParameters {
    // Internal storage: ATOMIC UNITS (for GPU)
    Eigen::MatrixXcd hamiltonian;                   // Full expanded Hamiltonian (n_sites × n_sites, a.u.)
    // Finalized site-basis Hamiltonian (n_states × n_states, a.u., reorg + w0
    // applied, pre-expansion). Kept so a task can (re)expand into its chosen
    // basis via HEOMConfig::apply_basis() without re-reading the yaml.
    Eigen::MatrixXcd hamiltonian_site;
    // Per-site bath-correlation coefficients, flat row-major [n][k], length
    // n_bath*(n_bath_modes_total+1). Index via mode_idx(n,k). c_{nk} = entry
    // (n, k); ν_{nk} likewise. Every physical bath n carries its own
    // correlation function C_n(t) = Σ_k c_{nk} e^{-ν_{nk} t}.
    std::vector<std::complex<double>> c_bath;       // c_{nk} (a.u.), [n_bath*(K+1)]
    std::vector<double> nu_bath;                    // ν_{nk} (a.u.), [n_bath*(K+1)]
    std::vector<double> markov_residue;             // per-site Markov term Σ_{k>K} c_{nk}/ν_{nk} (a.u.), [n_bath]

    // Which exponential-expansion scheme generated c_bath / nu_bath. Set from
    // the required YAML field bath.expansion_method (no implicit default).
    BathExpansion expansion = BathExpansion::MATSUBARA;

    // Padé-only reference caches (length n_extra_poles + 1, index [0] is the
    // Drude placeholder = 0, [1..K] the PSD poles/weights). Filled by
    // compute_pade_drude; left empty on the Matsubara path. Written to HDF5
    // /derived for post-processing cross-checks; never read by the kernel.
    std::vector<double> pade_zeta;                  // dimensionless pole positions ζ_l
    std::vector<double> pade_chi;                   // Padé rational weights χ_l
    
    double temperature;                             // Temperature (a.u.) — GLOBAL (all sites share)

    // Per-site bath input (η_n, γ_n) in both raw and a.u. forms. Single source
    // of truth; the old global scalars eta/gamma/eta_cm/gamma_inverse_fs have
    // been folded in here (one entry per pigment).
    std::vector<BathSite> bath_sites;               // [n_bath]

    // Input parameters (original units, for HDF5 metadata)
    double temperature_K;                           // Temperature (K)
    double w0_cm;                                   // Reference energy w0 (cm^-1)
    
    // Hilbert space configuration
    int n_states;                                   // Physical system dimension (excited states)
    BasisType basis;                                // Hilbert space basis type
    int n_sites;                                    // Total Hilbert space dimension (auto-computed)

    // Two bath-mode counts kept as separate fields for their distinct
    // conceptual roles (user-input snapshot vs runtime last index). They now
    // coincide numerically: n_bath_modes_total == n_extra_poles.
    //
    //   n_extra_poles       : USER input from YAML, immutable after parse_bath.
    //                         Number of non-Drude bath poles k=1..K (Matsubara
    //                         frequencies or Padé poles). The k=0 Drude mode is
    //                         implicit (always present).
    //   n_bath_modes_total  : RUNTIME index of the last entry in c_bath /
    //                         nu_bath arrays. Equals n_extra_poles. All kernel
    //                         loops `for k in 0..n_bath_modes_total` use this
    //                         field.
    int n_bath_modes_total = 0;
    int n_extra_poles = 0;

    int n_modes_per_bath() const { return n_bath_modes_total + 1; }

    // Flat row-major index into c_bath/nu_bath: entry (n, k).
    int mode_idx(int n, int k) const { return n * (n_bath_modes_total + 1) + k; }
    
    // Sink/Trap configuration (single source of truth)
    SinkConfig sink;

    // Transition dipole moments per pigment (system property; YAML: system.dipoles
    // / system.strengths). Empty on dynamics runs (which never read them); the
    // absorption / 2des resolvers require them non-empty. dipoles[i] is a unit
    // vector (parse-time normalised), strengths[i] the magnitude in Debye.
    // Length n_states when present.
    std::vector<std::array<double, 3>> dipoles;   // [n_states][3], unit vectors
    std::vector<double>                strengths; // [n_states], Debye

    int n_bath() const { return n_states; }
    
    Eigen::MatrixXcd hamiltonian_in_cm() const {
        return hamiltonian / CM_TO_AU;
    }
    
    std::complex<double> c_bath_in_cm(int n, int k) const {
        return c_bath[mode_idx(n, k)] / (CM_TO_AU * CM_TO_AU);
    }
    
    double nu_bath_in_cm(int n, int k) const {
        return nu_bath[mode_idx(n, k)] / CM_TO_AU;
    }
};

// Free functions (defined in physics_utils.cpp).
// Keeping them outside PhysicalParameters avoids putting non-trivial logic in
// every translation unit that includes types.hpp, and prepares for future
// multi-bath models where these helpers may take additional parameters.
// Bath-correlation expansions. compute_bath_correlation dispatches on
// phys.expansion; the two compute_*_drude builders share the Drude k=0 pole
// and the Markov-residue formula.
// boson_pade_NmN exposes the raw Padé [K-1/K] poles/weights for unit tests
// (anchors ζ_1²=60, χ_1=5/2). Algorithm is transcribed from the authoritative
// reference bath_correlation Padé poles helper — not rederived.
void compute_matsubara_drude(PhysicalParameters& phys);
void compute_pade_drude(PhysicalParameters& phys);
void compute_bath_correlation(PhysicalParameters& phys);
void boson_pade_NmN(int K, std::vector<double>& zeta, std::vector<double>& chi);
std::vector<int8_t> build_d_coupling(const PhysicalParameters& phys);

/**
 * @brief Time evolution parameters.
 *
 * Pure time-grid knobs, parsed per-task (YAML: dynamics/absorption/2des each
 * carry their own dt / n_steps / output_interval / print_interval). The 2des
 * task derives its step counts from the τ/t grid, so it leaves n_steps and
 * output_interval at 0 (only dt / print_interval are meaningful there).
 *
 * NOTE: there is deliberately NO filter_interval. ADO filtering runs every
 * internal step on both the full-grid and compact (ulist) paths — one cadence,
 * no knob. All filter
 * knobs live in FilterConfig (heom_config.hpp), not here.
 */
// Time integrator selection (per-task YAML key `integrator`, default rk4).
//   Rk4   : plain 4th-order RK4 (historical, bit-identical baseline).
//   IfRk4 : integrating-factor RK4 — Term 2 (hierarchy decay) is integrated
//           EXACTLY per-ADO, RK4 handles only the non-stiff coupling. Lets the
//           step escape the Term-2 stability wall (IF-RK4 integrator).
enum class Integrator { Rk4, IfRk4 };

struct TimeParameters {
    double dt = 0.0;                    // Time step in fs
    double dt_au = 0.0;                 // Time step in a.u. (= dt × FS_TO_AU, for RK4)
    int n_steps = 0;                    // Total number of output steps (0 for 2des)
    std::vector<double> output_times;   // Times to output (fs)
    int output_interval = 0;            // Output every N steps (write to file; 0 for 2des)
    int print_interval = 0;             // Print statistics every N steps; 0 disables printing
    Integrator integrator = Integrator::Rk4;  // time integrator (default: plain RK4)
};

/**
 * @brief Number of output records produced by a HEOM run with this time config.
 *
 * Single source of truth used by
 *   - HEOMTrajectory::n_output_records()
 *   - HDF5Writer::open() (replaces the hardcoded `n_steps/output_interval + 1`
 *     formula that previously lived at hdf5_writer.cpp:112)
 *   - heom::spectra::compute_linear_response() (avoids constructing a dummy
 *     HEOMTrajectory just to read this int — a HEOMTrajectory would otherwise
 *     do 3+ cudaMalloc calls inside its constructor)
 *
 * Record 0 corresponds to t=0; one extra record per `output_interval` HEOM steps
 * thereafter. The `+1` is for the t=0 entry.
 */
constexpr int output_record_count(const TimeParameters& t) {
    return t.n_steps / t.output_interval + 1;
}

/**
 * @brief Initial state specification — USER INPUT ONLY.
 *
 * Stripped of all runtime-computed fields (eigenvalues, eigenvectors,
 * selected_exciton_idx, and the dual-purpose custom_matrix that previously
 * held both user input AND solver-resolved exciton ρ(0)).
 *
 *   - eigenvalues / eigenvectors → moved to SystemContext (computed once,
 *     shared across trajectories)
 *   - selected_exciton_idx → use exciton_index directly (it's the same value)
 *   - custom_matrix → ONLY for CUSTOM_MATRIX type (user-provided ρ(0)).
 *     Exciton ρ(0) is now built on demand via build_initial_state().
 *
 * The corresponding ρ(0) construction lives in build_initial_state() in
 * heom_trajectory.hpp (free function), not as a member of this struct.
 */
struct InitialState {
    enum class Type {
        SITE_EXCITATION,    // Single site excitation |n⟩⟨n|
        EXCITON_EXCITATION, // Single exciton (eigenstate) excitation
        COHERENT,           // Coherent superposition
        CUSTOM_MATRIX       // User-provided density matrix
    };

    Type type;
    int excited_site = -1;                      // For SITE_EXCITATION (0-based Hilbert)
    int exciton_index = -1;                     // For EXCITON_EXCITATION: internal Hilbert/eigenvector index
    Eigen::MatrixXcd custom_matrix;             // ONLY for CUSTOM_MATRIX (user input)
    std::vector<std::complex<double>> coherent_coeffs;  // For COHERENT
};

/**
 * @brief Output configuration
 * 
 * Output: HDF5 format (full density matrix + populations)
 * - save_path: directory path (e.g. "data/fmo_benkmarch/")
 * - filename: "auto" for auto-naming from config params, or custom string
 *   Auto-naming only works for site_excitation and exciton_excitation.
 *   For coherent/custom_matrix, user must provide a custom filename.
 * - use_timestamp: append _YYYYMMDD_HHMM to filename
 */
struct OutputConfig {
    std::string save_path;              // Directory path
    std::string filename = "auto";      // "auto" → auto-generated, else custom name
    bool use_timestamp = true;          // true: append _YYYYMMDD_HHMM
    // When false (default), the pre-allocation memory budget prints
    // a single compact line; full per-bucket detail is only printed when the
    // estimate exceeds 50% of free GPU memory. When true, always full detail.
    bool verbose_memory = false;
};

/**
 * @brief ADO filtering statistics record
 */
struct FilteringRecord {
    int step;
    double time_fs;
    int active_ados;
    int total_ados;
    std::vector<int> level_counts;  // Active ADOs per level [L+1]
};

#endif // HEOM_TYPES_HPP

