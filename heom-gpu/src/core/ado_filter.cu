#include "ado_filter.cuh"
#include "cuda_common.cuh"
#include "core/precision.hpp"
#include <cuComplex.h>
#include <stdexcept>

using heom::complex_t;

// ADO filtering by ||rho||_max. alpha 0 (the physical density matrix) stays active.

/**
 * @brief Per-thread ADO norm kernel (small-N path, N*N < 64).
 *
 * One thread scans a full N×N block. Compare |z|² and sqrt() the winner once.
 * Matches the Fortran reference maxval(abs(rho)).
 *
 * U-local domain: scan_list != nullptr maps slot → α = scan_list[slot]
 * (same indirection as the derivative kernel); n_scan is then the launch count.
 * scan_list == nullptr keeps the historical full-M sweep (α = slot).
 */
__global__ void compute_ado_norms_kernel(
    const complex_t* rho_in,
    double* norms,
    int n_scan,
    int N,
    int M,
    int batch,
    int row_off,
    int n_row,
    int col_off,
    int n_col,
    const int* __restrict__ scan_list
)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;

    if (slot >= n_scan) return;
    const int alpha = scan_list ? scan_list[slot] : slot;

    // Union norm: ‖ρ_α‖ := max over batch slots b of ‖ρ_α^(b)‖_max. One flag
    // set gates all B states (union closure), so an ADO active in ANY slot must
    // survive the filter. batch==1 ⇒ single slot, boff==0 ⇒ historical path.
    const size_t N2 = static_cast<size_t>(N) * N;
    // Norm math stays FP64: max-norm has no accumulation, and the
    // hier element (float in mixed) promotes cleanly into the double compare.
    double max_abs_sq = 0.0;

    // One thread owns the whole sub-block, so walk it as nested row/col loops:
    // no per-element division, and each row is a contiguous run.
    for (int b = 0; b < batch; ++b) {
        const complex_t* rho_alpha =
            &rho_in[static_cast<size_t>(b) * M * N2 + static_cast<size_t>(alpha) * N2];
        for (int r = 0; r < n_row; ++r) {
            const complex_t* row =
                rho_alpha + static_cast<size_t>(row_off + r) * N + col_off;
            for (int c = 0; c < n_col; ++c) {
                double re = heom::c_re(row[c]);
                double im = heom::c_im(row[c]);
                double abs_sq = re * re + im * im;
                if (abs_sq > max_abs_sq) {
                    max_abs_sq = abs_sq;
                }
            }
        }
    }

    norms[alpha] = sqrt(max_abs_sq);
}

/**
 * @brief Warp-cooperative ADO norm kernel (primary path, N*N ≥ 64).
 *
 * Layout: blockDim = (32, WARPS_PER_BLOCK), one warp per ADO. Lane l strides
 * over the ADO's N*N elements with step 32; the per-lane max is reduced across
 * the warp via __shfl_xor_sync and lane 0 writes the sqrt. At large N this
 * turns one thread's uncoalesced N²-element scan into 32 coalesced lanes, and
 * it saturates the SMs at far smaller M (important once filtering shrinks M).
 * Indexes rho_in[alpha*N*N] with int, so the ceiling is M·N² < 2^31.
 */
__global__ void compute_ado_norms_warp_kernel(
    const complex_t* __restrict__ rho_in,
    double* __restrict__ norms,
    int n_scan,
    int N,
    int M,
    int batch,
    int row_off,
    int n_row,
    int col_off,
    int n_col,
    const int* __restrict__ scan_list)
{
    int slot = blockIdx.x * blockDim.y + threadIdx.y;
    if (slot >= n_scan) return;
    const int alpha = scan_list ? scan_list[slot] : slot;

    const int lane = threadIdx.x;
    // Union norm across batch slots (see compute_ado_norms_kernel). batch==1 ⇒
    // single slot, boff==0 ⇒ byte-identical to the pre-batch warp scan.
    const size_t N2 = static_cast<size_t>(N) * N;
    double local_max = 0.0;

    // Flat index over the sub-block, NOT nested row/col loops: the production
    // blocks are 7 columns wide (ESA 21x7, SE/GSB 7x1), so striding lanes along
    // a row would leave 25 of 32 idle. One integer division per element buys a
    // full warp; this kernel is bandwidth-bound (16 B read per element), so the
    // division hides. (0,N,0,N) ⇒ idx == e ⇒ byte-identical to the old scan.
    const int total = n_row * n_col;
    for (int b = 0; b < batch; ++b) {
        const complex_t* rho_alpha =
            &rho_in[static_cast<size_t>(b) * M * N2 + static_cast<size_t>(alpha) * N2];
        #pragma unroll 4
        for (int e = lane; e < total; e += 32) {
            const int idx = (row_off + e / n_col) * N + col_off + e % n_col;
            double re = heom::c_re(rho_alpha[idx]);
            double im = heom::c_im(rho_alpha[idx]);
            double abs_sq = re * re + im * im;
            if (abs_sq > local_max) local_max = abs_sq;
        }
    }

    // Warp shuffle max-reduction (32 lanes → 1)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_xor_sync(0xFFFFFFFFu, local_max, offset);
        if (other > local_max) local_max = other;
    }

    if (lane == 0) norms[alpha] = sqrt(local_max);
}

void compute_ado_norms(
    const complex_t* rho_in,
    double* norms,
    int M,
    int N,
    int row_off,
    int n_row,
    int col_off,
    int n_col,
    const int* scan_list,
    int scan_count,
    int batch
)
{
    // Same contract as HEOMContext (heom_derivatives.cuh): an empty block is
    // the default-constructed value, so rejecting it forces callers to write
    // "(0,N,0,N)" when they mean the whole matrix. Throw before launch — an
    // out-of-range block would read past the ADO inside the kernel.
    if (n_row <= 0 || n_col <= 0) {
        throw std::invalid_argument(
            "compute_ado_norms: manifold block must be non-empty; use "
            "(0,N,0,N) for the whole matrix");
    }
    if (row_off < 0 || col_off < 0 ||
        row_off + n_row > N || col_off + n_col > N) {
        throw std::invalid_argument(
            "compute_ado_norms: manifold block out of range");
    }

    // U-local: launch over the compact scan set instead of full M.
    const int n_scan = scan_list ? scan_count : M;
    if (n_scan <= 0) return;
    // norms[] is [M] (one union entry per ADO), NOT [batch*M]: the union max
    // over slots collapses the batch dimension here so downstream marking and
    // the shared active_flags stay batch-agnostic.

    // N*N ≥ 64 (empirical Ampere cross-over): warp kernel; smaller stays on the
    // per-thread path where its lower overhead wins.
    // Deliberately still keyed on N*N, not on the block's n_row*n_col: swapping
    // a 7-element block onto the per-thread path would serialise 7 loads in one
    // thread instead of issuing them from 7 lanes. Both kernels agree bitwise
    // (max is order-independent), so this stays a pure performance choice —
    // and it should not be re-tuned without a measurement.
    if (N * N >= 64) {
        constexpr int WARPS_PER_BLOCK = 4;          // 4 warps × 32 lanes = 128 threads/block
        dim3 block(32, WARPS_PER_BLOCK);
        int num_blocks = (n_scan + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        compute_ado_norms_warp_kernel<<<num_blocks, block>>>(
            rho_in, norms, n_scan, N, M, batch,
            row_off, n_row, col_off, n_col, scan_list
        );
    } else {
        int num_blocks = (n_scan + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;
        compute_ado_norms_kernel<<<num_blocks, HEOM_BLOCK_SIZE>>>(
            rho_in, norms, n_scan, N, M, batch,
            row_off, n_row, col_off, n_col, scan_list
        );
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
 */
__global__ void filter_ados_kernel(
    const double* __restrict__ norms,
    int8_t* __restrict__ active_flags,
    double threshold,
    int n_scan,
    const int* __restrict__ scan_list
)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_scan) return;
    // U-local domain: flags outside scan_list keep their value. Callers pass
    // the previous step's WRITE SET, so any α outside it is inactive (A ⊆ U)
    // with a frozen sub-threshold norm — its flag is already 0 and stays 0.
    const int alpha = scan_list ? scan_list[slot] : slot;
    active_flags[alpha] = (alpha == 0 || norms[alpha] > threshold) ? 1 : 0;
}

void filter_ados_mark_only(
    const double* norms,
    int8_t* active_flags,
    double threshold,
    int M,
    const int* scan_list,
    int scan_count
)
{
    const int n_scan = scan_list ? scan_count : M;
    if (n_scan <= 0) return;
    const int num_blocks = (n_scan + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;
    filter_ados_kernel<<<num_blocks, HEOM_BLOCK_SIZE>>>(
        norms, active_flags, threshold, n_scan, scan_list
    );
    CUDA_CHECK_KERNEL();
    // No count here: |A| is derived on demand from the level distribution
    // (see emit_ado_stats). This drops one D2H sync per filter step.
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
    int M, int n_bath, int K, int max_level
)
{
    extern __shared__ int s_level[];   // [max_level + 1]

    for (int p = threadIdx.x; p <= max_level; p += blockDim.x) {
        s_level[p] = 0;
    }
    __syncthreads();

    int alpha = blockIdx.x * blockDim.x + threadIdx.x;
    if (alpha < M && active_flags[alpha]) {
        const int level = filter_stats_level_for_ado(
            level_indices, alpha, n_bath, K);
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
    int M, int n_bath, int K,
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
        active_flags, level_indices, d_level_counts, M, n_bath, K, max_level
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_level_counts, d_level_counts,
                          (max_level + 1) * sizeof(int), cudaMemcpyDeviceToHost));
}

