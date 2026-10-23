#include "ado_filter.cuh"
#include "ado_norm.cuh"
#include "cuda_common.cuh"
#include "core/precision.hpp"
#include <cuComplex.h>
#include <stdexcept>

using heom::complex_t;

// ADO filtering by ||rho||_max. alpha 0 (the physical density matrix) stays active.

/**
 * @brief Per-thread ADO norm kernel (small-block path, Nb < 64).
 *
 * One thread scans one ADO's compact block: Nb = n_row·n_col contiguous
 * elements (the block IS the ADO, there is nothing else).
 * Compare |z|² and sqrt() the winner once. Matches the Fortran reference
 * maxval(abs(rho)).
 *
 * U-local domain: scan_list != nullptr maps slot → α = scan_list[slot]
 * (same indirection as the derivative kernel); n_scan is then the launch count.
 * scan_list == nullptr keeps the historical full-M sweep (α = slot).
 */
/**
 * max(cur, x) where a NaN, once seen, is never displaced.
 *
 * For finite x this is exactly the old `if (x > cur) cur = x;` — same
 * predicate, same order, so every finite-valued run stays bit-identical.
 * For NaN x the second clause fires and cur becomes NaN; afterwards both
 * clauses are false for any finite x (x > NaN is false, x != x is false), so
 * the NaN survives the rest of the scan and the warp reduction.
 *
 * Not fmax(): CUDA fmax returns the non-NaN operand by definition — it is
 * built to hide exactly what we need to surface. Not !(x <= cur) either: it
 * lets a NaN cur be overwritten by the next finite x.
 */
__device__ __forceinline__ double nan_sticky_max(double cur, double x) {
    return (x > cur || x != x) ? x : cur;
}

__global__ void compute_ado_norms_kernel(
    const complex_t* rho_in,
    double* norms,
    int n_scan,
    int Nb,
    int M,
    const int* __restrict__ scan_list,
    const int* __restrict__ scan_off
)
{
    // Slot b = grid.y owns its own [M][Nb] block of ρ and its own [M] row of
    // norms: every slot is filtered on its own state (per-slot active sets,
    // 2026-10-22-batch-per-slot). scan_off != nullptr gives each slot its own
    // segment [scan_off[b], scan_off[b+1]) of scan_list (the slot's previous
    // launch set); nullptr ⇒ the one scan_list (or full M) for every slot.
    // batch == 1 ⇒ b == 0, the historical single-state scan.
    const int b = blockIdx.y;
    const int n = scan_off ? scan_off[b + 1] - scan_off[b] : n_scan;
    int slot = blockIdx.x * blockDim.x + threadIdx.x;

    if (slot >= n) return;
    const int* list = (scan_list && scan_off) ? scan_list + scan_off[b] : scan_list;
    const int alpha = list ? list[slot] : slot;

    // Norm math stays FP64: max-norm has no accumulation, and the
    // hier element (float in mixed) promotes cleanly into the double compare.
    //
    // NaN must STICK (see nan_sticky_max). A plain `>` skips NaN elements, so
    // an ADO that has blown up to NaN reports the max of its finite remainder
    // (0 if fully NaN), gets marked inactive, is frozen out of the derivative,
    // and the run finishes with a finite, wrong answer that ‖ρ₀‖_F never sees.
    double max_abs_sq = 0.0;

    const complex_t* rho_alpha =
        &rho_in[(static_cast<size_t>(b) * M + alpha) * Nb];
    for (int e = 0; e < Nb; ++e)
        max_abs_sq = nan_sticky_max(max_abs_sq, heom::ado_abs_sq(rho_alpha[e]));

    norms[static_cast<size_t>(b) * M + alpha] = sqrt(max_abs_sq);
}

/**
 * @brief Warp-cooperative ADO norm kernel (large-block path, Nb ≥ 64).
 *
 * Layout: blockDim = (32, WARPS_PER_BLOCK), one warp per ADO. Lane l strides
 * over the ADO's Nb contiguous elements with step 32; the per-lane max is
 * reduced across the warp via __shfl_xor_sync and lane 0 writes the sqrt. At
 * large Nb this turns one thread's serial scan into 32 coalesced lanes, and it
 * saturates the SMs at far smaller M (important once filtering shrinks M).
 * The ρ offset (b·M + alpha)·Nb is computed in size_t; only the ADO slot
 * index is int, and n_scan ≤ M is far below that range.
 */
__global__ void compute_ado_norms_warp_kernel(
    const complex_t* __restrict__ rho_in,
    double* __restrict__ norms,
    int n_scan,
    int Nb,
    int M,
    const int* __restrict__ scan_list,
    const int* __restrict__ scan_off)
{
    // Per-slot (grid.y = b), same domain rule as compute_ado_norms_kernel.
    const int b = blockIdx.y;
    const int n = scan_off ? scan_off[b + 1] - scan_off[b] : n_scan;
    int slot = blockIdx.x * blockDim.y + threadIdx.y;
    if (slot >= n) return;
    const int* list = (scan_list && scan_off) ? scan_list + scan_off[b] : scan_list;
    const int alpha = list ? list[slot] : slot;

    const int lane = threadIdx.x;
    double local_max = 0.0;

    const complex_t* rho_alpha =
        &rho_in[(static_cast<size_t>(b) * M + alpha) * Nb];
    #pragma unroll 4
    for (int e = lane; e < Nb; e += 32)
        local_max = nan_sticky_max(local_max, heom::ado_abs_sq(rho_alpha[e]));

    // Warp shuffle max-reduction (32 lanes → 1)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_xor_sync(0xFFFFFFFFu, local_max, offset);
        local_max = nan_sticky_max(local_max, other);
    }

    if (lane == 0) norms[static_cast<size_t>(b) * M + alpha] = sqrt(local_max);
}

void compute_ado_norms(
    const complex_t* rho_in,
    double* norms,
    int M,
    int N,
    const heom::ManifoldBlock& blk,
    const int* scan_list,
    int scan_count,
    int batch,
    const int* scan_off
)
{
    // Same contract as HEOMContext, same single rule. The block is the storage
    // shape of `rho_in`; a wrong one here is a wrong stride, so fail loud.
    blk.validate(N, "compute_ado_norms");
    const int Nb = blk.elems();

    // U-local: launch over the compact scan set instead of full M.
    // With per-slot segments (scan_off) scan_count is the LONGEST segment —
    // grid.x covers it and shorter slots return early.
    const int n_scan = scan_list ? scan_count : M;
    if (n_scan <= 0 || batch <= 0) return;
    // norms[] is [batch][M]: one row per slot (per-slot active sets).

    // Nb >= 64: warp kernel; smaller blocks stay on the per-thread path.
    // Both agree bitwise. Do not retune the cutoff without a measurement.
    if (Nb >= 64) {
        constexpr int WARPS_PER_BLOCK = 4;          // 4 warps × 32 lanes = 128 threads/block
        dim3 block(32, WARPS_PER_BLOCK);
        dim3 grid((n_scan + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batch);
        compute_ado_norms_warp_kernel<<<grid, block>>>(
            rho_in, norms, n_scan, Nb, M, scan_list, scan_off);
    } else {
        dim3 grid((n_scan + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE, batch);
        compute_ado_norms_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
            rho_in, norms, n_scan, Nb, M, scan_list, scan_off);
    }
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// Mark active ADOs
// ============================================================================

/**
 * @brief Threshold-based active/inactive marker for ADOs.
 *
 *   active_flags[alpha] = (||rho_alpha||_max > threshold) for alpha >= 1.
 *
 * Physical invariant: rho_0 is the only ADO whose entries are directly
 * observable (populations, coherences). Filtering it out would zero the
 * HDF5 output even when the rest of the hierarchy is well-converged, so it
 * stays active for life. That invariant is enforced HERE, in one place: the
 * alpha == 0 lane writes 1 unconditionally. The constructor's initial
 * active_flags[0] = 1 is then merely a pre-first-filter default, not a
 * correctness dependency — no caller has to "remember" to keep index 0 set.
 *
 * NaN norm (the norm kernel lets it through on purpose): the ADO stays ACTIVE
 * and nan_flag is raised. Active so the NaN keeps propagating toward ρ₀
 * instead of being frozen behind an inactive flag; the flag so the host can
 * abort at its next divergence check without waiting for that. The atomic
 * only fires on the NaN path — the finite hot path is unchanged.
 */
__global__ void filter_ados_kernel(
    const double* __restrict__ norms,
    int8_t* __restrict__ active_flags,
    int* __restrict__ nan_flag,
    double threshold,
    int n_scan,
    int M,
    const int* __restrict__ scan_list,
    const int* __restrict__ scan_off,
    bool squared
)
{
    // Per-slot (grid.y = b): slot b's flags [b·M, +M) from its own norms,
    // over its own scan segment (see compute_ado_norms_kernel).
    const int b = blockIdx.y;
    const int n_b = scan_off ? scan_off[b + 1] - scan_off[b] : n_scan;
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_b) return;
    // U-local domain: flags outside scan_list keep their value. Callers pass
    // the previous step's WRITE SET, so any α outside it is inactive (A ⊆ U)
    // with a frozen sub-threshold norm — its flag is already 0 and stays 0.
    const int* list = (scan_list && scan_off) ? scan_list + scan_off[b] : scan_list;
    const int alpha = list ? list[slot] : slot;
    const size_t at = static_cast<size_t>(b) * M + alpha;
    // squared: norms holds max|ρ|² from the fused kFinal epilogue; the sqrt
    // the standalone kernel would have taken is taken here (same operand).
    const double n = squared ? sqrt(norms[at]) : norms[at];
    const bool is_nan = (n != n);
    if (is_nan) atomicOr(nan_flag, 1);
    active_flags[at] = (alpha == 0 || is_nan || n > threshold) ? 1 : 0;
}

void filter_ados_mark_only(
    const double* norms,
    int8_t* active_flags,
    int* nan_flag,
    double threshold,
    int M,
    const int* scan_list,
    int scan_count,
    int batch,
    const int* scan_off,
    bool norms_squared
)
{
    if (nan_flag == nullptr) {
        throw std::invalid_argument(
            "filter_ados_mark_only: nan_flag must be a device int (the NaN "
            "hazard this guards against is silent; no opt-out)");
    }
    const int n_scan = scan_list ? scan_count : M;
    if (n_scan <= 0 || batch <= 0) return;
    const dim3 grid((n_scan + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE, batch);
    filter_ados_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
        norms, active_flags, nan_flag, threshold, n_scan, M, scan_list, scan_off,
        norms_squared
    );
    CUDA_CHECK_KERNEL();
    // No count here: |A| is derived on demand from the level distribution
    // (see emit_ado_stats). This drops one D2H sync per filter step.
}

__global__ void count_zero_norms_kernel(const double* __restrict__ norms,
                                        int M, int* __restrict__ count)
{
    // grid.y = slot b: count[b] = #{α : norms[b·M + α] == 0}.
    // One atomic per block (block-wide count).
    const int b = blockIdx.y;
    const int alpha = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = __syncthreads_count(
        alpha < M && norms[static_cast<size_t>(b) * M + alpha] == 0.0);
    if (threadIdx.x == 0 && c > 0) atomicAdd(&count[b], c);
}

void count_zero_norms(const double* norms, int M, int batch, int* d_count,
                      int* h_count)
{
    CUDA_CHECK(cudaMemset(d_count, 0, batch * sizeof(int)));
    const dim3 grid((M + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE, batch);
    count_zero_norms_kernel<<<grid, HEOM_BLOCK_SIZE>>>(norms, M, d_count);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_count, d_count, batch * sizeof(int),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Active-ADO level histogram
// ============================================================================

/**
 * @brief Count active ADOs broken down by hierarchy level (sum of alpha_{nk}).
 *
 * Each block reduces into a (max_level + 1)-element shared counter via
 * shared-memory atomics, then flushes (max_level + 1) global atomicAdds per
 * block instead of one per active ADO — cuts global-atomic contention sharply.
 * Shared footprint is (L + 1) * sizeof(int), never enough to bump occupancy.
 */
__global__ void count_active_by_level_kernel(
    const int8_t* __restrict__ active_flags,
    const int8_t* __restrict__ level_indices,
    int* __restrict__ level_counts,
    int M, int D, int max_level
)
{
    extern __shared__ int s_level[];   // [max_level + 1]

    for (int p = threadIdx.x; p <= max_level; p += blockDim.x) {
        s_level[p] = 0;
    }
    __syncthreads();

    int alpha = blockIdx.x * blockDim.x + threadIdx.x;
    if (alpha < M && active_flags[alpha]) {
        const int level = filter_stats_level_for_ado(level_indices, alpha, D);
        // Bounded by L_cutoff < 127 (HEOMConfig::validate enforces this);
        // s_level[level] is always in range.
        atomicAdd(&s_level[level], 1);
    }
    __syncthreads();

    for (int p = threadIdx.x; p <= max_level; p += blockDim.x) {
        const int v = s_level[p];
        if (v > 0) atomicAdd(&level_counts[p], v);
    }
}

void compute_active_level_distribution(
    const int8_t* active_flags,
    const int8_t* level_indices,
    int M, int D,
    int max_level,
    int* d_level_counts,
    int* h_level_counts
)
{
    CUDA_CHECK(cudaMemset(d_level_counts, 0, (max_level + 1) * sizeof(int)));

    const int num_blocks = (M + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;
    const size_t smem_bytes =
        static_cast<size_t>(max_level + 1) * sizeof(int);

    count_active_by_level_kernel<<<num_blocks, HEOM_BLOCK_SIZE, smem_bytes>>>(
        active_flags, level_indices, d_level_counts, M, D, max_level
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_level_counts, d_level_counts,
                          (max_level + 1) * sizeof(int), cudaMemcpyDeviceToHost));
}

