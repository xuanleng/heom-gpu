#ifndef HEOM_MEMORY_BUDGET_HPP
#define HEOM_MEMORY_BUDGET_HPP

// GPU + host memory budget estimator.
//
// Purpose: given a fully-loaded Config and the hierarchy size M (known only
// after HierarchyTruncation::generate_hierarchy_indices()), predict every
// cudaMalloc + every persistent host allocation that HEOMSolver will issue,
// and refuse to start if the result would not fit on the selected device.
//
// This complements the existing "X MB allocated" line in HEOMSolver::initialize()
// (heom_solver.cpp:55-61) which fires AFTER cudaMalloc and therefore cannot
// avoid OOM, only report it. The functions below are pure (no GPU side-effects)
// so unit tests can drive them on any host.
//
// All formulas trace 1:1 to allocate_gpu_memory() in heom_solver.cpp; if that
// function gains/loses a buffer, update estimate_budget() in lockstep.

#include "heom_config.hpp"
#include <cstddef>
#include <string>

namespace heom {

/**
 * @brief Estimated memory footprint, broken down by purpose.
 *
 * Sizes are in bytes. The GPU buckets sum exactly to the post-allocation
 * total printed by HEOMSolver::initialize(); the host buckets cover persistent
 * std::vectors that survive past initialization (transient buffers like the
 * h_active_flags upload staging are excluded — they are dwarfed by the
 * persistent ones for any non-trivial M).
 */
struct MemoryBudget {
    // ---- GPU buckets (bytes) ----
    size_t rho_tensor;          // d_rho_                         = 1 · M · N² · 16
    size_t rk4_workspace;       // d_workspace_                   = 3 · M · N² · 16
    size_t hierarchy_maps;      // d_plus_map + d_minus_map + d_level_indices
                                //   = M · n_bath · (K+1) · (4+4+1)
    size_t filter_buffers;      // d_ado_norms + d_active_flags + d_level_counts
                                //   (zero if !use_filtering)
    size_t auxiliary;           // d_hamiltonian + d_c_bath + d_markov_residue
                                //   + d_total_decay + d_d_coupling
    size_t gpu_total;           // sum of the five buckets above

    // ---- Host buckets (bytes) ----
    // host_hierarchy and host_neighbor_maps are TRANSIENT: built by
    // SystemContext::generate_hierarchy_and_maps(), used to populate the GPU
    // copies, then released when the constructor returns. host_density_buffer
    // (h_rho0_ in HEOMTrajectory) is the only persistent host allocation.
    // host_total is therefore peak host RAM during SystemContext init, not
    // steady-state usage.
    size_t host_hierarchy;      // HierarchyTable hh.indices (+ sort permutation)
    size_t host_neighbor_maps;  // hh.plus_map + hh.minus_map (std::vector<int>)
    size_t host_density_buffer; // h_rho0_ (rho_0 only)        N² · 16
    size_t host_total;          // sum of the three buckets above

    // Additional transient peak inside upload_gpu_readonly itself:
    // h_level_indices (M · n_bath · (K+1) · 1B) coexists briefly with
    // host_hierarchy + host_neighbor_maps. For LHC-II this overlap is the
    // difference between fitting in 16 GB and a mysterious resize() OOM.
    size_t host_init_peak;

    // ---- Echo of derived parameters (for printing / debugging) ----
    size_t M;                   // number of ADOs
    int    N;                   // Hilbert dimension
    int    n_bath;              // bath count (= n_states)
    int    K;                   // n_bath_modes_total
    int    L;                   // L_cutoff
    size_t batch;               // batch width B (d_rho / d_workspace scale by B)
    bool   use_filtering;
};

/**
 * @brief Compute the budget for a given config and known hierarchy size.
 *
 * Pure function. Does not touch CUDA, can be unit-tested on a CPU-only host.
 *
 * @param cfg Fully-loaded HEOM configuration.
 * @param M   Number of ADOs after HierarchyTruncation::generate_hierarchy_indices().
 *            Pass 0 to estimate everything except the M-dependent terms (rare).
 * @param batch Batch width B (default 1). Scales the two O(M·N²) mutable
 *            buffers (d_rho, d_workspace) and the mixed Kahan carry; read-only
 *            structure is unaffected. B==1 reproduces the pre-batch estimate.
 */
MemoryBudget estimate_budget(const HEOMConfig::Config& cfg, size_t M,
                             size_t batch = 1);

/**
 * @brief Pretty-print a full per-bucket budget to std::cout.
 *
 * Output uses IEC binary units (KiB / MiB / GiB) consistently. If gpu_total
 * is 0 the device-fit summary line is suppressed (useful when called outside
 * a CUDA context, e.g. from unit tests).
 *
 * @param b           Budget produced by estimate_budget().
 * @param gpu_free    Bytes free on the active CUDA device. 0 = unknown.
 * @param gpu_total   Total bytes on the active CUDA device. 0 = unknown.
 */
void print_budget(const MemoryBudget& b, size_t gpu_free, size_t gpu_total);

/**
 * @brief One-line compact budget print (default for small jobs).
 *
 * Used for FMO-7 / regression configs where the full ~25-line per-bucket
 * dump is just CI log noise. Prints something like:
 *     [MEMORY BUDGET]  GPU: 79.61 MiB  (0.5% of free)  M=14495 N=7
 */
void print_budget_compact(const MemoryBudget& b, size_t gpu_free);

/**
 * @brief Tri-state outcome of a pre-allocation budget check.
 *
 * - Fits     : projected GPU usage fits within (gpu_free * safety_margin).
 * - Exceeds  : would OOM; caller should abort and surface `err` to the user.
 * - Unknown  : caller could not query device memory (gpu_free == 0). The
 *              check is *skipped* — caller decides whether to proceed
 *              (typical: print warning, continue, accept OOM risk).
 *
 * Returning a single bool here was a footgun: `validate_budget(...) == true`
 * conflated "fits" and "could not check", so callers blindly proceeded into
 * a poisoned cudaMalloc on hosts where cudaMemGetInfo failed.
 */
enum class BudgetCheck { Fits, Exceeds, Unknown };

/**
 * @brief Pre-allocation budget check (replaces validate_budget).
 *
 * Applies a safety_margin (default 0.95) because the CUDA driver itself,
 * cuBLAS handles, and per-kernel scratch consume a few hundred MB that
 * cudaMemGetInfo does not report as "free". 5% headroom is conservative
 * for HEOM (no cuBLAS, single stream); tune downward only with profiling.
 *
 * @param b              Budget produced by estimate_budget().
 * @param cfg            Loaded config (used only to tailor mitigation hints).
 * @param gpu_free       Bytes free on the active CUDA device. 0 ⇒ Unknown.
 * @param err            On Exceeds, populated with a multi-line, actionable
 *                       message (cleared on Fits/Unknown).
 * @param safety_margin  Fraction of gpu_free treated as usable. Must be in
 *                       (0, 1]; out-of-range values throw std::invalid_argument
 *                       (no silent clamp — that hid unit-test bugs).
 *
 * @return Fits | Exceeds | Unknown — see enum docs.
 * @throws std::invalid_argument if safety_margin ∉ (0, 1].
 */
BudgetCheck check_budget(const MemoryBudget& b,
                         const HEOMConfig::Config& cfg,
                         size_t gpu_free,
                         std::string& err,
                         double safety_margin = 0.95);

/**
 * @brief Convenience wrapper around cudaMemGetInfo.
 *
 * @return true on cudaSuccess; on failure free=total=0 and the caller should
 *         treat the device as "unknown capacity" (skip the fit check, but
 *         still print the budget for diagnostic value).
 */
bool query_gpu_memory(size_t& free, size_t& total);

}  // namespace heom

#endif  // HEOM_MEMORY_BUDGET_HPP
