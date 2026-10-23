#ifndef HEOM_GPU_ALLOC_PLAN_HPP
#define HEOM_GPU_ALLOC_PLAN_HPP

// Single source of truth for "which GPU buffers do SystemContext and
// HEOMTrajectory allocate".
//
// Both estimate_budget() in memory_budget.cpp and the two owners' allocators
// consume this list. If you add or remove a GPU buffer,
// update plan_gpu_buffers() — and only here. The pre-allocation budget
// check and the actual cudaMalloc walk the same list, so they cannot
// drift out of sync.
//
// SCOPE: the plan covers SystemContext/HEOMTrajectory-OWNED buffers only.
// Driver-owned allocations live outside it and their callers must account
// for them on top of estimate_budget():
//   - 2DES DeviceHierarchySnapshot ρ copies (tau_snap B=3 + tw_snap B=9,
//     each sized batch·M·Nb_cap ⇒ 12·M·Nb_cap elements; twice that in the
//     mixed build, which also snapshots the Kahan carry) and the [21·tree]
//     detection grid — both added to the budget by compute_2des's walk
//     guard (check_walk_budget in src/spectra/two_dim.cpp), which refuses
//     the run if they do not fit.
//   - Small lazy scratch (d_mu_scratch_, Detector's μ_det/grid): noise, not modeled.
//
// Naming convention: each entry's `name` matches the corresponding owner
// member variable (e.g. "d_rho", "d_workspace"). The allocator
// uses these names to look up element counts; the budget summer ignores
// names and aggregates by `Bucket`.

#include "gpu_memory.hpp"         // GPUMemory<T> (typed_alloc helper)
#include "core/precision.hpp"     // heom::complex_t (hier precision buffers)
#include "core/hierarchy_distance.cuh"   // kDistMaxBlocks
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
        RhoTensor,      // d_rho                             — 1 · M · Nb · 16
        RK4Workspace,   // d_workspace (k_curr+temp+acc)     — 3 · M · Nb · 16
        HierarchyMap,   // d_plus_map, d_minus_map, d_level_indices
        FilterBuf,      // d_ado_norms, d_active_flags, d_level_counts
        Auxiliary       // H, bath poles, total_decay, d_coupling
    };

    const char* name;       // diagnostic; matches the owning member name
    size_t      count;      // element count (NOT bytes)
    size_t      elem_bytes; // sizeof(T)
    Bucket      bucket;

    size_t bytes() const { return count * elem_bytes; }
};

/**
 * @brief Build the canonical list of GPU buffers the solver allocates
 *        (SystemContext-owned first, then HEOMTrajectory-owned).
 *
 * Pure function: no GPU side-effects, can be called from unit tests on a
 * CPU-only host.
 *
 * @param M               Number of ADOs (after hierarchy generation).
 * @param N               Hilbert dimension (= cfg.physics.n_sites).
 * @param D               Total ladder count Σ_n P_n (= phys.n_ladders()):
 *                        the row width of every per-ADO mode table and the
 *                        length of the per-slot coefficient arrays.
 * @param n_bath          Number of physical baths (= cfg.physics.n_states).
 * @param L               Hierarchy truncation level (for d_level_counts size).
 * @param use_filtering   When false, the four FilterBuf entries are omitted.
 * @param batch           Batch width B. The two O(M·Nb) mutable buffers
 *                        scale (d_rho ×B, d_workspace ×B) and so do the O(M)
 *                        per-slot filter rows (d_ado_norms, d_active_flags).
 *                        Read-only structure (maps, coeffs, H) is shared
 *                        across slots and stays ×1.
 * @param rho_elems       Elements stored per ADO, Nb (= heom::rho_capacity(cfg)):
 *                        N² for dynamics, the largest manifold block the task
 *                        will ever hold otherwise. Sizes d_rho / d_workspace;
 *                        d_hamiltonian and d_d_coupling still go by N.
 */
inline std::vector<GPUBufferSpec>
plan_gpu_buffers(size_t M, int N, int D, int n_bath, int L, bool use_filtering,
                 size_t batch, size_t rho_elems)
{
    using B = GPUBufferSpec::Bucket;
    if (rho_elems == 0 || rho_elems > static_cast<size_t>(N) * N) {
        throw std::invalid_argument(
            "plan_gpu_buffers: rho_elems must be in [1, N^2], got "
            + std::to_string(rho_elems) + " with N=" + std::to_string(N));
    }
    const size_t rho  = M * rho_elems;                    // one slot's ρ tensor
    const size_t brho = batch * rho;                      // batched ρ / workspace
    const size_t channels = static_cast<size_t>(D);
    const size_t map = M * channels;

    // Design note (d_level_indices, dense [M * D] int8):
    //   The hierarchy guarantees Sigma alpha_{n,k} <= L, so on average each
    //   ADO has at most L nonzero entries out of D slots. A
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
    // identical arithmetic to the old ping-pong+swap but one full M·Nb·16B
    // buffer cheaper. Do not re-add a second rho tensor.
    // d_rho / d_workspace follow the HIER precision (FP32 in mixed): they are
    // the O(M·Nb) bandwidth-bound big arrays.
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
        {"d_c_bath",        channels,                             sizeof(heom::complex_t), B::Auxiliary},
        {"d_cbar_bath",     channels,                             sizeof(heom::complex_t), B::Auxiliary},
        // No d_nu_bath / d_abs_c_bath: bath frequencies ν_{nk} and the Shi scale
        // s_{nk} = √(|c||c̄|) are host-only. ν_{nk} feeds the total_decay
        // precompute; s_{nk} feeds the coeff_plus/coeff_minus precompute below. The derivative kernel reads
        // neither directly — it consumes the precomputed per-ADO arrays instead.
        {"d_markov_residue", static_cast<size_t>(n_bath),         sizeof(heom::real_t),    B::Auxiliary},
        {"d_plus_map",      map,                                  sizeof(int),             B::HierarchyMap},
        {"d_minus_map",     map,                                  sizeof(int),             B::HierarchyMap},
        {"d_level_indices", map,                                  sizeof(int8_t),          B::HierarchyMap},
        // Per-ADO Term 4/5 magnitudes (√[(α+1)s], √[α/s]); same map layout.
        {"d_coeff_plus",    map,                                  sizeof(heom::real_t),    B::HierarchyMap},
        {"d_coeff_minus",   map,                                  sizeof(heom::real_t),    B::HierarchyMap},
        {"d_total_decay",   M,                                    sizeof(cuDoubleComplex), B::Auxiliary},
        {"d_d_coupling",    static_cast<size_t>(n_bath) * N,      sizeof(int8_t),          B::Auxiliary},
        // Ladder prefix sums [n_bath+1]; the kernel's Term 4/5 bath loop reads
        // mode_off[n], mode_off[n+1] to find its slot range (core/mode_layout.hpp).
        {"d_mode_off",      static_cast<size_t>(n_bath) + 1,      sizeof(int),             B::Auxiliary},
        // relative_distance's per-block (distance², norm²) partial sums, 16 KB.
        {"d_dist_partials", 2 * static_cast<size_t>(kDistMaxBlocks), sizeof(double),     B::Auxiliary},
    };
    if (use_filtering) {
        // One row of M per slot: per-slot active sets (2026-10-22-batch-per-slot).
        p.push_back({"d_ado_norms",    batch * M,                        sizeof(double), B::FilterBuf});
        p.push_back({"d_active_flags", batch * M,                        sizeof(int8_t), B::FilterBuf});
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
