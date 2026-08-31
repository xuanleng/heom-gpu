#ifndef HEOM_GPU_ALLOC_PLAN_HPP
#define HEOM_GPU_ALLOC_PLAN_HPP

// Single source of truth for "which GPU buffers does HEOMSolver allocate".
//
// Both estimate_budget() in memory_budget.cpp and allocate_gpu_memory() in
// heom_solver.cpp consume this list. If you add or remove a GPU buffer,
// update plan_gpu_buffers() — and only here. The pre-allocation budget
// check and the actual cudaMalloc walk the same list, so they cannot
// drift out of sync.
//
// SCOPE: the plan covers SystemContext/HEOMTrajectory-OWNED buffers only.
// Driver-owned allocations live outside it and their callers must account
// for them on top of estimate_budget():
//   - 2DES DeviceHierarchySnapshot blocks (tau_snap B=3 + tw_snap B=9,
//     worst case n_active = M ⇒ 12·M·N² elements) and the [21·tree]
//     detection grid — both added to the budget by compute_2des's batch
//     guard (src/spectra/two_dim.cpp) before deciding B=21 vs fallback.
//   - Small lazy scratch (d_mu_det_, d_mu_scratch_): noise, not modeled.
//
// Naming convention: each entry's `name` matches the corresponding
// HEOMSolver member variable (e.g. "d_rho", "d_workspace"). The allocator
// uses these names to look up element counts; the budget summer ignores
// names and aggregates by `Bucket`.

#include "gpu_memory.hpp"         // GPUMemory<T> (typed_alloc helper)
#include "core/precision.hpp"     // heom::complex_t (hier precision buffers)
#include <cstddef>
#include <cstdint>
#include <cuComplex.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace heom {

struct GPUBufferSpec {
    enum class Bucket {
        RhoTensor,      // d_rho                             — 1 · M · N² · 16
        RK4Workspace,   // d_workspace (k_curr+temp+acc)     — 3 · M · N² · 16
        HierarchyMap,   // d_plus_map, d_minus_map, d_level_indices
        FilterBuf,      // d_ado_norms, d_active_flags, d_level_counts
        Auxiliary       // H, bath poles, total_decay, d_coupling
    };

    const char* name;       // diagnostic; matches HEOMSolver member name
    size_t      count;      // element count (NOT bytes)
    size_t      elem_bytes; // sizeof(T)
    Bucket      bucket;

    size_t bytes() const { return count * elem_bytes; }
};

/**
 * @brief Build the canonical list of GPU buffers that HEOMSolver allocates.
 *
 * Pure function: no GPU side-effects, can be called from unit tests on a
 * CPU-only host. The returned list is ordered to match the declaration
 * order in heom_solver.hpp for grep-friendliness.
 *
 * @param M               Number of ADOs (after hierarchy generation).
 * @param N               Hilbert dimension (= cfg.physics.n_sites).
 * @param K               Last bath-pole index (= n_bath_modes_total ==
 *                        n_extra_poles); every per-pole array is sized K+1.
 * @param n_bath          Number of physical baths (= cfg.physics.n_states).
 * @param L               Hierarchy truncation level (for d_level_counts size).
 * @param use_filtering   When false, the four FilterBuf entries are omitted.
 * @param batch           Batch width B (default 1). ONLY the two O(M·N²)
 *                        mutable buffers scale: d_rho ×B and d_workspace ×B.
 *                        Read-only structure (maps, coeffs, H, filter flags)
 *                        is shared across slots and stays ×1. B==1 reproduces
 *                        the pre-batch plan exactly.
 */
inline std::vector<GPUBufferSpec>
plan_gpu_buffers(size_t M, int N, int K, int n_bath, int L, bool use_filtering,
                 size_t batch = 1)
{
    using B = GPUBufferSpec::Bucket;
    const size_t rho  = M * static_cast<size_t>(N) * N;   // one slot's ρ tensor
    const size_t brho = batch * rho;                      // batched ρ / workspace
    const size_t map = M * static_cast<size_t>(n_bath) * (K + 1);

    // Design note (d_level_indices, dense [M * n_bath * (K+1)] int8):
    //   The hierarchy guarantees Sigma alpha_{n,k} <= L, so on average each
    //   ADO has at most L nonzero entries out of n_bath*(K+1) slots. A
    //   sparse (n, k, value) per-ADO layout would shrink the buffer by
    //     - FMO-7  (M=14k,  n_bath=7,  K+1=6,  L=3) :  600 KB ->  ~150 KB
    //     - LHC-II (M~9M,   n_bath=14, K+1=3,  L=10):  ~378 MB -> ~270 MB
    //   but the kernel hot path
    //     for n in n_bath: for k in 0..K: a_nk = level_indices[alpha,n,k]
    //   becomes a per-(n,k) search through the sparse list (O(nnz) per
    //   access vs O(1) today), and Term 4/5 would have to be restructured
    //   to iterate the sparse entries directly. For both shipping configs
    //   d_level_indices is far below 1 GiB and is dwarfed by d_rho /
    //   d_workspace / d_plus_map. Keeping the dense layout. Revisit if a
    //   future config makes hierarchy_maps the GPU memory bottleneck.
    // No d_rho_next: rk4_step's final combine is in-place (rho += dt/6·acc),
    // identical arithmetic to the old ping-pong+swap but one full M·N²·16B
    // buffer cheaper. Do not re-add a second rho tensor.
    // d_rho / d_workspace follow the HIER precision (FP32 in mixed): they are
    // the O(M·N²) bandwidth-bound big arrays.
    // "Downcast once on read" is a statement about WHERE the FP64→FP32
    // rounding happens, and the only affordable place is outside the kernel's
    // inner loops — FP64 runs at 1/64 rate on the consumer cards this build
    // targets, so a handful of per-thread FP64 ops out-cost everything else.
    // Every array the derivative kernel reads is therefore stored pre-rounded
    // in hier precision, with the rounding done once at upload (SystemContext)
    // where the FP64 source values are still exact. That includes d_hamiltonian:
    // its downcast is "once per block", but a block stages the whole N×N matrix
    // while producing only blockDim outputs, so for N² > blockDim it costs more
    // FP64 converts per thread than the inner loop ever did.
    std::vector<GPUBufferSpec> p = {
        {"d_rho",           brho,                                 sizeof(heom::complex_t), B::RhoTensor},
        {"d_workspace",     3 * brho,                             sizeof(heom::complex_t), B::RK4Workspace},
        {"d_hamiltonian",   static_cast<size_t>(N) * N,           sizeof(heom::complex_t), B::Auxiliary},
        {"d_c_bath",        static_cast<size_t>(n_bath) * (K + 1), sizeof(heom::complex_t), B::Auxiliary},
        // No d_nu_bath / d_abs_c_bath: bath frequencies ν_{nk} and |c_{nk}| are
        // host-only. ν_{nk} feeds the total_decay precompute; |c_{nk}| feeds the
        // coeff_plus/coeff_minus precompute below. The derivative kernel reads
        // neither directly — it consumes the precomputed per-ADO arrays instead.
        {"d_markov_residue", static_cast<size_t>(n_bath),         sizeof(heom::real_t),    B::Auxiliary},
        {"d_plus_map",      map,                                  sizeof(int),             B::HierarchyMap},
        {"d_minus_map",     map,                                  sizeof(int),             B::HierarchyMap},
        {"d_level_indices", map,                                  sizeof(int8_t),          B::HierarchyMap},
        // Per-ADO Term 4/5 magnitudes (√[(α+1)|c|], √[α/|c|]); same map layout.
        {"d_coeff_plus",    map,                                  sizeof(heom::real_t),    B::HierarchyMap},
        {"d_coeff_minus",   map,                                  sizeof(heom::real_t),    B::HierarchyMap},
        {"d_total_decay",   M,                                    sizeof(double),          B::Auxiliary},
        {"d_d_coupling",    static_cast<size_t>(n_bath) * N,      sizeof(int8_t),          B::Auxiliary},
    };
    if (use_filtering) {
        p.push_back({"d_ado_norms",    M,                                sizeof(double), B::FilterBuf});
        p.push_back({"d_active_flags", M,                                sizeof(int8_t), B::FilterBuf});
        p.push_back({"d_level_counts", static_cast<size_t>(L + 1),       sizeof(int),    B::FilterBuf});
    }
    return p;
}

/**
 * @brief Allocate a `GPUMemory<T>` for the named entry in the plan, with
 *        compile-T vs plan-elem_bytes mismatch detection.
 *
 * Replaces the previous pattern
 *
 *     auto find_count = [&](sv name) { ... return s.count; };
 *     d_active_flags_ = GPUMemory<int8_t>(find_count("d_active_flags"));
 *
 * with the single line
 *
 *     d_active_flags_ = heom::typed_alloc<int8_t>(plan, "d_active_flags");
 *
 * If the plan entry's `elem_bytes` doesn't match `sizeof(T)`, throws a
 * `std::logic_error` *before* hitting cudaMalloc — converts the previous
 * "silent under/over-allocation if someone changes one side" failure mode
 * into a fail-loud-at-startup invariant. The string lookup itself remains
 * (vector is ~13 entries, walked once at trajectory ctor; not a hot path).
 */
template<typename T>
GPUMemory<T> typed_alloc(const std::vector<GPUBufferSpec>& plan,
                         std::string_view name)
{
    for (const auto& s : plan) {
        if (std::string_view(s.name) == name) {
            if (s.elem_bytes != sizeof(T)) {
                throw std::logic_error(
                    "gpu_alloc_plan type mismatch for '" + std::string(name)
                    + "': plan.elem_bytes=" + std::to_string(s.elem_bytes)
                    + " vs sizeof(T)=" + std::to_string(sizeof(T))
                    + " — caller's GPUMemory<T> template arg disagrees with"
                    + " plan_gpu_buffers(); fix the spec or the call site.");
            }
            return GPUMemory<T>(s.count);
        }
    }
    throw std::logic_error(
        std::string("gpu_alloc_plan missing buffer: ") + std::string(name)
        + " — caller asked for a buffer name that plan_gpu_buffers() did not"
        + " produce; either add it to the spec list or fix the call site.");
}

}  // namespace heom

#endif  // HEOM_GPU_ALLOC_PLAN_HPP
