#ifndef ADO_FILTER_CUH
#define ADO_FILTER_CUH

#include <cuComplex.h>
#include "core/precision.hpp"
#include <stdint.h>

#if defined(__CUDACC__)
#define HEOM_ADO_FILTER_HD __host__ __device__
#else
#define HEOM_ADO_FILTER_HD
#endif

/**
 * ADO filtering by ||rho||_max = max_ij |rho_ij|.
 */

/**
 * Max-norm over an optional compact ADO list.
 *
 * scan_list != nullptr writes norms[alpha] only for those ADOs. Stale
 * entries outside the list are never read. The list must cover the ADOs
 * written on the previous step.
 *
 * batch > 1 stores the max across slots in norms[alpha]. The slots share
 * one active set, so an ADO needed by any slot stays. rho_in is
 * [batch][M][N*N]. batch == 1 matches the single-state scan.
 *
 * The scan covers rows [row_off, row_off+n_row) and cols
 * [col_off, col_off+n_col). Outside a 2DES block the elements are exact
 * zeros, so the block max equals the full-matrix max. (0, N, 0, N) is the
 * full matrix. The four block arguments have no defaults.
 */
void compute_ado_norms(
    const heom::complex_t* rho_in,
    double* norms,
    int M,
    int N,
    int row_off,
    int n_row,
    int col_off,
    int n_col,
    const int* scan_list = nullptr,
    int scan_count = 0,
    int batch = 1
);

/**
 * Mark ADOs active when norm > threshold. Does not return a count.
 *
 * Alpha 0 stays active. |A| is summed later from the level histogram.
 * scan_list != nullptr writes flags only for those ADOs; flags outside the
 * list stay put.
 */
void filter_ados_mark_only(
    const double* norms,
    int8_t* active_flags,
    double threshold,
    int M,
    const int* scan_list = nullptr,
    int scan_count = 0
);

/**
 * @brief Compute the filtering-statistics level for one ADO.
 *
 * The level is the full occupation sum s = Σ_{n,k} α_{n,k} over all K+1
 * per-bath slots (k = 0 Drude + k = 1..K poles), matching HierarchyIndex::sum().
 * The histogram buffer is sized by the occupation cutoff L_cutoff.
 */
HEOM_ADO_FILTER_HD inline int filter_stats_level_for_ado(
    const int8_t* level_indices,
    int alpha,
    int n_bath,
    int K
) {
    const int stride = K + 1;

    int level = 0;
    for (int n = 0; n < n_bath; ++n) {
        for (int k = 0; k < stride; ++k) {
            const int idx = alpha * n_bath * stride + n * stride + k;
            level += level_indices[idx];
        }
    }
    return level;
}

/**
 * Histogram of active ADOs by hierarchy level. Caller owns the buffers.
 * level_indices is [M x n_bath x (K+1)]; k = 0 is the Drude pole.
 */
void compute_active_level_distribution(
    const int8_t* active_flags,
    const int8_t* level_indices,
    int M, int n_bath, int K,
    int max_level,
    int* d_level_counts,
    int* h_level_counts
);

#endif // ADO_FILTER_CUH

