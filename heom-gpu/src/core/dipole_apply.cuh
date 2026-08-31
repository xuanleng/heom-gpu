#ifndef HEOM_CORE_DIPOLE_APPLY_CUH
#define HEOM_CORE_DIPOLE_APPLY_CUH

#include <cuComplex.h>
#include <cuda_runtime.h>
#include "core/precision.hpp"

// Apply a single-side dipole μ to ALL M auxiliary density operators (ADOs),
// carry-through style (2DES plan commit b.1):
//   side == 0 : left-multiply   out_α = μ · ρ_α
//   side == 1 : right-multiply  out_α = ρ_α · μ
// for every α = 0..M-1, where each ρ_α is an N×N (row-major) block.
//
// Kernel structure reuses one half of the Term-1 commutator GEMM in
// heom_derivatives.cu (μ·ρ is the left half, ρ·μ the right half).
//
// rho_in and rho_out MUST be distinct buffers (dense multiply is not in-place).
// mu is a device-side N×N row-major matrix.
void launch_apply_single_side_mu(
    const heom::complex_t* rho_in, heom::complex_t* rho_out,
    const heom::complex_t* mu, int side, int M, int N);

// Batched per-slot variant (2DES batched walk): apply a DISTINCT μ to each of B slots in
// one launch. rho_in/rho_out are [B·M·N²] (slot stride M·N²), mu is [B·N²]
// (slot stride N²): slot b gets mu[b·N²]. grid.y = B; slot b only reads/writes
// its own [b·M·N², (b+1)·M·N²) range (boff), so slots are independent. batch==1
// with a single-copy mu degenerates to launch_apply_single_side_mu bit-for-bit
// (the single-slot launcher above is a thin wrapper of this at batch=1).
void launch_apply_single_side_mu_batch(
    const heom::complex_t* rho_in, heom::complex_t* rho_out,
    const heom::complex_t* mu, int side, int M, int N, int batch);

// Gather n_active scattered ADO blocks into a contiguous staging buffer:
//   staging[e·N² + t] = rho[ids[e]·N² + t]   for e∈[0,n_active), t∈[0,N²)
// Scatter does the reverse:
//   rho[ids[e]·N² + t] = staging[e·N² + t]
// ids is a device int array (ascending ADO ids). Replaces the per-ADO
// cudaMemcpy storm in HEOMTrajectory::{snapshot,restore}_full_hierarchy with
// one kernel + one bulk transfer per direction. n_active·N² < 2³¹ holds
// because n_active ≤ M and the project ceiling is M·N² < 2³¹.
void launch_gather_ado_blocks(
    const heom::complex_t* rho, heom::complex_t* staging,
    const int* ids, int n_active, int N);

void launch_scatter_ado_blocks(
    const heom::complex_t* staging, heom::complex_t* rho,
    const int* ids, int n_active, int N);

// Batched per-slot variants (2DES batched walk): gather/scatter B slots in one launch.
// rho is [B·M·N²] (slot stride M·N²), staging is [B·n_active·N²] (slot stride
// n_active·N²); every slot shares the SAME `ids` (union active closure).
// grid.y = B. batch==1 degenerates to launch_{gather,scatter}_ado_blocks
// bit-for-bit (slot 0 only; M unused since boff=0).
void launch_gather_ado_blocks_batch(
    const heom::complex_t* rho, heom::complex_t* staging,
    const int* ids, int n_active, int N, int M, int batch);

void launch_scatter_ado_blocks_batch(
    const heom::complex_t* staging, heom::complex_t* rho,
    const int* ids, int n_active, int N, int M, int batch);

// Device-side trace Tr(B·ρ) = Σ_i Σ_k B[i,k]·ρ[k,i], written into out[idx].
// ρ is one N×N (row-major) block — for 2DES detection it is the α=0 ADO. B is
// a device N×N row-major matrix. One block, shared-memory reduction.
//
// 2DES carry-through detection: the t-loop samples this once
// per t-point into a per-tree device grid (out[idx]), so the whole t-scan is
// computed with NO host round-trip. The single bulk D2H of `out` happens once
// at the end of the tree, replacing the former per-point expectation() D2H +
// implicit device sync. Arithmetic matches HEOMTrajectory::expectation()
// term-for-term; only the summation ORDER differs (strided + tree reduce vs
// host row-major), so results agree to a few ULPs, not bit-exact.
void launch_trace_BR(
    const heom::complex_t* B, const heom::complex_t* rho,
    heom::complex_t* out, int idx, int N);

// Batched per-slot variant (2DES batched walk): one trace per slot in a single launch of
// B blocks (grid.y = B). rho is [B·M·N²] (slot stride M·N²; slot b's α=0 block
// is at b·M·N²), B_op(μ_det) is [B·N²] (slot stride N²), out is [B·n_samples]
// (slot b writes out[b·n_samples + idx]). batch==1 degenerates to launch_trace_BR
// bit-for-bit (slot 0; M / n_samples unused since b·… = 0).
void launch_trace_BR_batch(
    const heom::complex_t* B, const heom::complex_t* rho,
    heom::complex_t* out, int idx, int N, int M, int batch, int n_samples);

#endif  // HEOM_CORE_DIPOLE_APPLY_CUH
