#include "dipole_apply.cuh"
#include "cuda_common.cuh"
#include "core/precision.hpp"

using heom::complex_t;
using heom::c_add;
using heom::c_mul;
using heom::c_make;

// One thread per (α, i, j) output element; blockIdx.y = slot b (grid.y = B).
// Reuses one half of the Term-1 commutator GEMM (heom_derivatives.cu): μ·ρ is
// the left half, ρ·μ the right. Slot b reads/writes only its own [b·M·N²] range
// (boff) and uses its own μ[b·N²] (muoff), so slots are fully independent.
__global__ void apply_single_side_mu_kernel(
    const complex_t* __restrict__ rho_in,
    complex_t* __restrict__ rho_out,
    const complex_t* __restrict__ mu,
    int side, int M, int N)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N2 = N * N;
    if (gid >= M * N2) return;

    const size_t boff  = static_cast<size_t>(blockIdx.y) * M * N2;
    const size_t muoff = static_cast<size_t>(blockIdx.y) * N2;

    const int alpha = gid / N2;
    const int rem = gid % N2;
    const int i = rem / N;
    const int j = rem % N;

    const complex_t* r  = &rho_in[boff + static_cast<size_t>(alpha) * N2];
    const complex_t* mb = &mu[muoff];
    complex_t acc = c_make(0.0, 0.0);
    if (side == 0) {                       // left:  (μ·ρ)_{ij} = Σ_k μ_{ik} ρ_{kj}
        for (int k = 0; k < N; ++k)
            acc = c_add(acc, c_mul(mb[i * N + k], r[k * N + j]));
    } else {                               // right: (ρ·μ)_{ij} = Σ_k ρ_{ik} μ_{kj}
        for (int k = 0; k < N; ++k)
            acc = c_add(acc, c_mul(r[i * N + k], mb[k * N + j]));
    }
    rho_out[boff + static_cast<size_t>(alpha) * N2 + i * N + j] = acc;
}

void launch_apply_single_side_mu_batch(
    const complex_t* rho_in, complex_t* rho_out,
    const complex_t* mu, int side, int M, int N, int batch)
{
    const int total = M * N * N;
    const int blocks = (total + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;
    const dim3 grid(blocks, batch);
    apply_single_side_mu_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
        rho_in, rho_out, mu, side, M, N);
    CUDA_CHECK_KERNEL();
}

void launch_apply_single_side_mu(
    const complex_t* rho_in, complex_t* rho_out,
    const complex_t* mu, int side, int M, int N)
{
    // Single-slot path: B=1, single-copy μ. grid.y=1 ⇒ boff=muoff=0,
    // bit-identical to the pre-batch kernel.
    launch_apply_single_side_mu_batch(rho_in, rho_out, mu, side, M, N, 1);
}

// One thread per staging element; blockIdx.y = slot b (grid.y = B). e = gid/N²,
// t = gid%N²: gathers the scattered source block ids[e] into the dense slot e
// (or scatters back). ρ slot stride is M·N², staging slot stride n_active·N²;
// all slots share `ids` (union closure). B=1 ⇒ boff=0, pre-batch behaviour.
__global__ void gather_ado_blocks_kernel(
    const complex_t* __restrict__ src,
    complex_t* __restrict__ dst,
    const int* __restrict__ ids, int n_active, int N2, int M)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_active * N2) return;
    const size_t soff = static_cast<size_t>(blockIdx.y) * M * N2;          // ρ slot
    const size_t doff = static_cast<size_t>(blockIdx.y) * n_active * N2;   // staging slot
    const int e = gid / N2;
    const int t = gid % N2;
    dst[doff + gid] = src[soff + static_cast<size_t>(ids[e]) * N2 + t];
}

__global__ void scatter_ado_blocks_kernel(
    const complex_t* __restrict__ src,
    complex_t* __restrict__ dst,
    const int* __restrict__ ids, int n_active, int N2, int M)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_active * N2) return;
    const size_t soff = static_cast<size_t>(blockIdx.y) * n_active * N2;   // staging slot
    const size_t doff = static_cast<size_t>(blockIdx.y) * M * N2;          // ρ slot
    const int e = gid / N2;
    const int t = gid % N2;
    dst[doff + static_cast<size_t>(ids[e]) * N2 + t] = src[soff + gid];
}

void launch_gather_ado_blocks_batch(
    const complex_t* rho, complex_t* staging,
    const int* ids, int n_active, int N, int M, int batch)
{
    if (n_active <= 0) return;
    const int N2 = N * N;
    const int total = n_active * N2;
    const int blocks = (total + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;
    const dim3 grid(blocks, batch);
    gather_ado_blocks_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
        rho, staging, ids, n_active, N2, M);
    CUDA_CHECK_KERNEL();
}

void launch_scatter_ado_blocks_batch(
    const complex_t* staging, complex_t* rho,
    const int* ids, int n_active, int N, int M, int batch)
{
    if (n_active <= 0) return;
    const int N2 = N * N;
    const int total = n_active * N2;
    const int blocks = (total + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;
    const dim3 grid(blocks, batch);
    scatter_ado_blocks_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
        staging, rho, ids, n_active, N2, M);
    CUDA_CHECK_KERNEL();
}

void launch_gather_ado_blocks(
    const complex_t* rho, complex_t* staging,
    const int* ids, int n_active, int N)
{
    // Single-slot: B=1, slot 0 only (boff=0 ⇒ M unused, pass n_active).
    launch_gather_ado_blocks_batch(rho, staging, ids, n_active, N, n_active, 1);
}

void launch_scatter_ado_blocks(
    const complex_t* staging, complex_t* rho,
    const int* ids, int n_active, int N)
{
    launch_scatter_ado_blocks_batch(staging, rho, ids, n_active, N, n_active, 1);
}

// Tr(B·ρ) in one block. Lane strides over the N² (i,k) products, partial sums
// land in shared memory, then a halving tree reduction. blockDim is fixed at
// HEOM_BLOCK_SIZE (power of 2, as enforced in cuda_common.cuh) so the halving
// loop is exact.
__global__ void trace_BR_kernel(
    const complex_t* __restrict__ B,
    const complex_t* __restrict__ rho,
    complex_t* __restrict__ out,
    int idx, int N, int M, int n_samples)
{
    __shared__ complex_t sdata[HEOM_BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int N2 = N * N;
    const int b = blockIdx.y;                                    // slot
    const complex_t* Bb = &B[static_cast<size_t>(b) * N2];       // per-slot μ_det
    const complex_t* rb = &rho[static_cast<size_t>(b) * M * N2]; // slot b's α=0 block

    complex_t acc = c_make(0.0, 0.0);
    for (int e = tid; e < N2; e += blockDim.x) {
        const int i = e / N;
        const int k = e % N;
        // Tr(B·ρ) = Σ B[i,k]·ρ[k,i]; matches HEOMTrajectory::expectation().
        acc = c_add(acc, c_mul(Bb[i * N + k], rb[k * N + i]));
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = c_add(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[static_cast<size_t>(b) * n_samples + idx] = sdata[0];
}

void launch_trace_BR_batch(
    const complex_t* B, const complex_t* rho,
    complex_t* out, int idx, int N, int M, int batch, int n_samples)
{
    const dim3 grid(1, batch);
    trace_BR_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
        B, rho, out, idx, N, M, n_samples);
    CUDA_CHECK_KERNEL();
}

void launch_trace_BR(
    const complex_t* B, const complex_t* rho,
    complex_t* out, int idx, int N)
{
    // Single-slot: B=1, slot 0. b·M·N²=0 and b·n_samples=0 for b=0, so
    // M / n_samples are unused — pass 1 each. out[idx] as before.
    launch_trace_BR_batch(B, rho, out, idx, N, 1, 1, 1);
}
