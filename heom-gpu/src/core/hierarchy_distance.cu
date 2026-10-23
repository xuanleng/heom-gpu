#include "hierarchy_distance.cuh"
#include "cuda_common.cuh"

#include <algorithm>

namespace heom {

namespace {

__device__ __forceinline__ void carried(const complex_t* v, const complex_t* c,
                                        size_t e, double& re, double& im) {
    re = c_re(v[e]);
    im = c_im(v[e]);
    if (c) { re -= c_re(c[e]); im -= c_im(c[e]); }
}

// Grid-stride accumulation, then a halving tree in shared memory (blockDim is
// HEOM_BLOCK_SIZE, a power of 2). Fixed lane→element assignment for a given n.
__global__ void hierarchy_distance_kernel(
    const complex_t* __restrict__ rho, const complex_t* __restrict__ rho_comp,
    const complex_t* __restrict__ ref, const complex_t* __restrict__ ref_comp,
    size_t n, double* __restrict__ partials)
{
    __shared__ double s_dist[HEOM_BLOCK_SIZE];
    __shared__ double s_norm[HEOM_BLOCK_SIZE];
    const int tid = threadIdx.x;
    double dist = 0.0, norm = 0.0;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + tid; e < n; e += stride) {
        double ar, ai, br, bi;
        carried(rho, rho_comp, e, ar, ai);
        carried(ref, ref_comp, e, br, bi);
        const double dr = ar - br, di = ai - bi;
        dist += dr * dr + di * di;
        norm += ar * ar + ai * ai;
    }
    s_dist[tid] = dist;
    s_norm[tid] = norm;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_dist[tid] += s_dist[tid + s];
            s_norm[tid] += s_norm[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[2 * blockIdx.x]     = s_dist[0];
        partials[2 * blockIdx.x + 1] = s_norm[0];
    }
}

}  // namespace

int launch_hierarchy_distance(const complex_t* rho, const complex_t* rho_comp,
                              const complex_t* ref, const complex_t* ref_comp,
                              size_t n, double* partials)
{
    const size_t want = (n + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;
    const int blocks = static_cast<int>(
        std::max<size_t>(1, std::min<size_t>(want, kDistMaxBlocks)));
    hierarchy_distance_kernel<<<blocks, HEOM_BLOCK_SIZE>>>(
        rho, rho_comp, ref, ref_comp, n, partials);
    CUDA_CHECK_KERNEL();
    return blocks;
}

}  // namespace heom
