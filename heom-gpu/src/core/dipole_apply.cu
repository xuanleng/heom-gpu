#include "dipole_apply.cuh"
#include "cuda_common.cuh"
#include "core/precision.hpp"
#include "heom_derivatives.cuh"   // checked_rho_elements

using heom::complex_t;
using heom::c_add;
using heom::c_mul;
using heom::c_make;
using heom::ManifoldBlock;

// One thread per (α, r, c) output element of the TARGET block T; blockIdx.y =
// slot b (grid.y = B). ρ_in is stored compacted to the SOURCE block S, ρ_out to
// T. Reuses one half of the Term-1 commutator GEMM (heom_derivatives.cu):
// μ·ρ is the left half, ρ·μ the right. Slot b reads its own [b·M·Nb_S] range,
// writes its own [b·M·Nb_T] range and uses its own μ[b·N²] (muoff), so slots
// are fully independent.
//
// The k-sum runs over S's rows (left) or S's cols (right) only: every other k
// has ρ_α(k,j) (resp. ρ_α(i,k)) structurally zero and NOT stored. Skipping
// those terms is bit-identical to the full N-loop over a zero-padded matrix:
// acc starts at +0 and the skipped products are μ·(±0) = (±0, ±0), which under
// round-to-nearest never change a running sum that started at +0 (x + ±0 = x
// for x ≠ 0; +0 + ±0 = +0; a cancelling x + (−x) yields +0). The host has
// already checked that the fixed axis is shared (T.cols == S.cols for left,
// T.rows == S.rows for right) and that μ maps S into T.
//
// `total` = M·Nb_T arrives from the host, already range-checked
// (checked_rho_elements); the kernel never re-derives it. ManifoldBlock is a
// 4-int POD, passed by value; only its fields are read on the device.
__global__ void apply_single_side_mu_kernel(
    const complex_t* __restrict__ rho_in,
    complex_t* __restrict__ rho_out,
    const complex_t* __restrict__ mu,
    int side, int M, int N, ManifoldBlock S, ManifoldBlock T, int total)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const int N2   = N * N;
    const int Nb_S = S.n_row * S.n_col;
    const int Nb_T = T.n_row * T.n_col;

    const size_t boff_S = static_cast<size_t>(blockIdx.y) * M * Nb_S;
    const size_t boff_T = static_cast<size_t>(blockIdx.y) * M * Nb_T;
    const size_t muoff  = static_cast<size_t>(blockIdx.y) * N2;

    const int alpha = gid / Nb_T;
    const int rem   = gid % Nb_T;
    const int r = rem / T.n_col;           // local (r, c) in T
    const int c = rem % T.n_col;
    const int i = T.row_off + r;           // global (i, j)
    const int j = T.col_off + c;

    const complex_t* rs = &rho_in[boff_S + static_cast<size_t>(alpha) * Nb_S];
    const complex_t* mb = &mu[muoff];
    complex_t acc = c_make(0.0, 0.0);
    if (side == 0) {                       // left:  (μ·ρ)_{ij} = Σ_k μ_{ik} ρ_{kj}, k ∈ rows(S)
        const int lj = j - S.col_off;      // j ∈ cols(S) == cols(T), host-checked
        const int k_end = S.row_off + S.n_row;
        for (int k = S.row_off; k < k_end; ++k)
            acc = c_add(acc, c_mul(mb[i * N + k], rs[(k - S.row_off) * S.n_col + lj]));
    } else {                               // right: (ρ·μ)_{ij} = Σ_k ρ_{ik} μ_{kj}, k ∈ cols(S)
        const int li = i - S.row_off;      // i ∈ rows(S) == rows(T), host-checked
        const int k_end = S.col_off + S.n_col;
        for (int k = S.col_off; k < k_end; ++k)
            acc = c_add(acc, c_mul(rs[li * S.n_col + (k - S.col_off)], mb[k * N + j]));
    }
    rho_out[boff_T + static_cast<size_t>(alpha) * Nb_T + rem] = acc;
}

void launch_apply_single_side_mu(
    const complex_t* rho_in, complex_t* rho_out,
    const complex_t* mu, int side, int M, int N,
    const ManifoldBlock& S, const ManifoldBlock& T, int batch)
{
    S.validate(N, "launch_apply_single_side_mu(S)");
    T.validate(N, "launch_apply_single_side_mu(T)");
    const int total = heom::checked_rho_elements(M, T.elems());
    const int blocks = (total + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;
    const dim3 grid(blocks, batch);
    apply_single_side_mu_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
        rho_in, rho_out, mu, side, M, N, S, T, total);
    CUDA_CHECK_KERNEL();
}

// Tr(B·ρ) in one block. Lane strides over the N² (i,k) products, partial sums
// land in shared memory, then a halving tree reduction. blockDim is fixed at
// HEOM_BLOCK_SIZE (power of 2, as enforced in cuda_common.cuh) so the halving
// loop is exact.
//
// ρ is stored compacted to `blk`; B (μ_det) is the full N×N. The loop still
// walks the VIRTUAL N² index space so that lane → (i,k) assignment — and hence
// the reduction grouping — is exactly what it was over a full matrix; (k,i)
// outside the block contribute nothing, which is bit-identical to adding
// B·(±0) (see apply_single_side_mu_kernel for the ±0 argument).
__global__ void trace_BR_kernel(
    const complex_t* __restrict__ B,
    const complex_t* __restrict__ rho,
    complex_t* __restrict__ out,
    int idx, int N, int M, int n_samples, ManifoldBlock blk)
{
    __shared__ complex_t sdata[HEOM_BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int N2  = N * N;
    const int Nb  = blk.n_row * blk.n_col;
    const int b = blockIdx.y;                                    // slot
    const complex_t* Bb = &B[static_cast<size_t>(b) * N2];       // per-slot μ_det
    const complex_t* rb = &rho[static_cast<size_t>(b) * M * Nb]; // slot b's α=0 block

    complex_t acc = c_make(0.0, 0.0);
    for (int e = tid; e < N2; e += blockDim.x) {
        const int i = e / N;
        const int k = e % N;
        // Tr(B·ρ) = Σ B[i,k]·ρ[k,i], i outer, k inner (the host oracle order).
        const int lk = k - blk.row_off;
        const int li = i - blk.col_off;
        if (lk < 0 || lk >= blk.n_row || li < 0 || li >= blk.n_col) continue;
        acc = c_add(acc, c_mul(Bb[i * N + k], rb[lk * blk.n_col + li]));
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = c_add(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[static_cast<size_t>(b) * n_samples + idx] = sdata[0];
}

void launch_trace_BR(
    const complex_t* B, const complex_t* rho,
    complex_t* out, int idx, int N, int M, int batch, int n_samples,
    const ManifoldBlock& blk)
{
    blk.validate(N, "launch_trace_BR");
    const dim3 grid(1, batch);
    trace_BR_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
        B, rho, out, idx, N, M, n_samples, blk);
    CUDA_CHECK_KERNEL();
}
