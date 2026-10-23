#include "dressed_start.cuh"
#include "cuda_common.cuh"
#include "core/mode_layout.hpp"
#include "core/precision.hpp"

#include <cuComplex.h>

using heom::complex_t;
using heom::real_t;
using heom::ManifoldBlock;

namespace {

__device__ __forceinline__ cuDoubleComplex to_d(complex_t z) {
    return make_cuDoubleComplex(heom::c_re(z), heom::c_im(z));
}

// One thread per (α, stored element) of slot blockIdx.y, like the derivative
// kernel. Threads whose α is not on `tier` return at once.
__global__ void dress_tier_kernel(
    complex_t* __restrict__ rho,
    const complex_t* __restrict__ c_bath,
    const complex_t* __restrict__ cbar_bath,
    const real_t* __restrict__ coeff_minus,
    const real_t* __restrict__ markov_residue,
    const int* __restrict__ minus_map,
    const int8_t* __restrict__ level_indices,
    const int* __restrict__ mode_off,
    const cuDoubleComplex* __restrict__ total_decay,
    const int8_t* __restrict__ d_coupling,
    int M, int N, int D, int n_bath, ManifoldBlock blk, int tier)
{
    const int Nb = blk.n_row * blk.n_col;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= M * Nb) return;
    const int alpha = gid / Nb;
    const int lij   = gid % Nb;

    int level = 0;
    for (int s = 0; s < D; ++s) level += level_indices[alpha * D + s];
    if (level != tier) return;

    const int i = blk.row_off + lij / blk.n_col;
    const int j = blk.col_off + lij % blk.n_col;
    const heom::ModeLayout ml{n_bath, D, mode_off};
    complex_t* slot = rho + static_cast<size_t>(blockIdx.y) * M * Nb;

    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    double markov = 0.0;
    for (int n = 0; n < n_bath; ++n) {
        const int di = d_coupling[n * N + i];
        const int dj = d_coupling[n * N + j];
        if (di != dj) markov += markov_residue[n];
        if (di == 0 && dj == 0) continue;
        for (int k = 0; k < ml.count(n); ++k) {
            const int idx = ml.idx(alpha, n, k);
            if (level_indices[idx] == 0) continue;
            cuDoubleComplex factor = make_cuDoubleComplex(0.0, 0.0);
            if (di) factor = cuCadd(factor, to_d(c_bath[ml.slot(n, k)]));
            if (dj) factor = cuCsub(factor, to_d(cbar_bath[ml.slot(n, k)]));
            const cuDoubleComplex lower =
                to_d(slot[static_cast<size_t>(minus_map[idx]) * Nb + lij]);
            const cuDoubleComplex t = cuCmul(factor, lower);
            const double w = coeff_minus[idx];
            acc = cuCadd(acc, make_cuDoubleComplex(w * cuCreal(t), w * cuCimag(t)));
        }
    }
    // −i·acc / (Σαν + Δ): Term 2 and Term 3 are the diagonal of L₀.
    const cuDoubleComplex num = make_cuDoubleComplex(cuCimag(acc), -cuCreal(acc));
    const cuDoubleComplex den = cuCadd(total_decay[alpha], make_cuDoubleComplex(markov, 0.0));
    const cuDoubleComplex v = cuCdiv(num, den);
    slot[static_cast<size_t>(alpha) * Nb + lij] =
        heom::c_make(static_cast<real_t>(cuCreal(v)), static_cast<real_t>(cuCimag(v)));
}

}  // namespace

void launch_dress_tier(const HEOMContext& ctx, complex_t* rho, int tier)
{
    const int total = heom::checked_rho_elements(ctx.M, ctx.block.elems());
    const int blocks = (total + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;
    const dim3 grid(blocks, ctx.batch);
    dress_tier_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
        rho, ctx.c_bath, ctx.cbar_bath, ctx.coeff_minus, ctx.markov_residue,
        ctx.neighbor_map_minus, ctx.level_indices, ctx.mode_off, ctx.total_decay,
        ctx.d_coupling, ctx.M, ctx.N, ctx.D, ctx.n_bath, ctx.block, tier);
    CUDA_CHECK_KERNEL();
}
