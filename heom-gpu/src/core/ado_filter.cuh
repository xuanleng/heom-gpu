#ifndef ADO_FILTER_CUH
#define ADO_FILTER_CUH

#include <cuComplex.h>
#include "core/precision.hpp"
#include "core/mode_layout.hpp"       // ModeLayout, HEOM_HD
#include "core/manifold_layout.hpp"   // ManifoldBlock
#include <stdint.h>

/**
 * ADO filtering by ||rho||_max = max_ij |rho_ij|.
 */

/**
 * Max-norm over an optional compact ADO list.
 *
 * scan_list != nullptr writes norms[alpha] only for those ADOs (same map as
 * the derivative u_list). Stale entries outside the list are never read.
 * The list must cover the ADOs written on the previous step.
 *
 * batch > 1: norms[b*M + alpha] is that slot's norm. rho_in is
 * [batch][M][Nb]. scan_off (device, length batch+1) selects per-slot
 * segments; nullptr shares one list. batch == 1 is the single-state scan.
 *
 * The block is the storage shape (Nb contiguous elements). A wrong block is
 * a wrong stride; the launcher validates it. The max is order-independent,
 * so the warp and thread paths agree bitwise. Any NaN element makes
 * norms[alpha] NaN.
 */
void compute_ado_norms(
    const heom::complex_t* rho_in,
    double* norms,
    int M,
    int N,
    const heom::ManifoldBlock& blk,
    const int* scan_list = nullptr,
    int scan_count = 0,
    int batch = 1,
    const int* scan_off = nullptr
);

/**
 * Mark ADOs active when norm > threshold. Does not return a count.
 *
 * Alpha 0 stays active. |A| is summed later from the level histogram, so the
 * hot path never copies a count back to the host.
 *
 * scan_list != nullptr writes flags only for those ADOs. Flags outside the
 * list stay put: those ADOs were frozen below threshold.
 *
 * A NaN norm marks the ADO active and sets *nan_flag. nan_flag is required.
 * norms and flags are [batch*M], with the same scan_list / scan_off contract
 * as compute_ado_norms. norms_squared means the values are max|rho_ij|^2
 * and are square-rooted before the compare.
 */
void filter_ados_mark_only(
    const double* norms,
    int8_t* active_flags,
    int* nan_flag,
    double threshold,
    int M,
    const int* scan_list = nullptr,
    int scan_count = 0,
    int batch = 1,
    const int* scan_off = nullptr,
    bool norms_squared = false
);

/**
 * Per-slot count of ADOs whose norm is exactly 0 (full M, one copy-back).
 * norms must be a fresh full-M result. NaN is not zero. Tiny FP64 values
 * whose square underflows also count as zero.
 */
void count_zero_norms(const double* norms, int M, int batch, int* d_count,
                      int* h_count);

/**
 * @brief Compute the filtering-statistics level for one ADO.
 *
 * The level is the full occupation sum s = Σ_{n,k} α_{n,k} over all D slots
 * of the ADO's row, matching HierarchyIndex::sum(). The histogram buffer is
 * sized by the occupation cutoff L_cutoff.
 */
HEOM_HD inline int filter_stats_level_for_ado(
    const int8_t* level_indices,
    int alpha,
    int D
) {
    // Σ over every slot of this ADO is a flat sum over its D contiguous
    // bytes; the (n,k) structure does not matter to a sum.
    const int8_t* row = level_indices + alpha * D;
    int level = 0;
    for (int s = 0; s < D; ++s) level += row[s];
    return level;
}

/**
 * Histogram of active ADOs by hierarchy level. Caller owns the buffers.
 * level_indices is [M x D], D = sum_n P_n.
 */
void compute_active_level_distribution(
    const int8_t* active_flags,
    const int8_t* level_indices,
    int M, int D,
    int max_level,
    int* d_level_counts,
    int* h_level_counts
);

#endif // ADO_FILTER_CUH

