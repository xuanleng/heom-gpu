#ifndef HEOM_CORE_DIPOLE_APPLY_CUH
#define HEOM_CORE_DIPOLE_APPLY_CUH

#include <cuComplex.h>
#include <cuda_runtime.h>
#include "core/precision.hpp"
#include "core/manifold_layout.hpp"

// Every launcher here is batched over B slots (grid.y = B): slot b reads and
// writes only its own [b·M·Nb, (b+1)·M·Nb) range of ρ and uses its own operator
// at [b·N²]. There is no separate single-slot launcher — B=1 is simply grid.y=1
// with slot offset 0. Pass real M even at B=1; it only
// enters via b·M·Nb, which is 0 for b=0.
//
// ρ is stored COMPACTED to its ManifoldBlock — each ADO is an
// n_row×n_col row-major block, Nb = n_row·n_col elements, not N². Operators
// (μ, μ_det) stay full N×N.

// Apply a single-side dipole μ to ALL M auxiliary density operators (ADOs) of
// every slot:
//   side == 0 : left-multiply   out_α = μ_b · ρ_α
//   side == 1 : right-multiply  out_α = ρ_α · μ_b
// for every α = 0..M-1.
//
// rho_in is stored on the SOURCE block S, rho_out on the TARGET block T; the
// kernel does the rectangular product and only touches stored elements. The
// caller guarantees the shared axis (left: T.cols == S.cols; right: T.rows ==
// S.rows) and that μ maps S into T — HEOMTrajectory::apply_dipole derives T
// from the ManifoldLayout for exactly that reason. Skipping the structurally
// zero terms is bit-identical to the old full-N loop (±0 argument in the .cu).
//
// Kernel structure reuses one half of the Term-1 commutator GEMM in
// heom_derivatives.cu (μ·ρ is the left half, ρ·μ the right half).
//
// rho_in [B·M·Nb_S] / rho_out [B·M·Nb_T] MUST be distinct buffers (dense
// multiply is not in-place); mu is [B·N²], device-side row-major.
void launch_apply_single_side_mu(
    const heom::complex_t* rho_in, heom::complex_t* rho_out,
    const heom::complex_t* mu, int side, int M, int N,
    const heom::ManifoldBlock& S, const heom::ManifoldBlock& T, int batch);

// Device-side trace Tr(B_b·ρ_b) = Σ_i Σ_k B[i,k]·ρ[k,i] for every slot b, written
// into out[b·n_samples + idx]. ρ_b is slot b's α=0 ADO (at b·M·Nb), stored on
// `blk`; B_op(μ_det) is the full [B·N²]. One block per slot, shared-memory
// reduction. (k,i) outside `blk` are structurally zero and skipped.
//
// 2DES carry-through detection: the t-loop samples this once
// per t-point into a per-tree device grid, so the whole t-scan is computed with
// NO host round-trip. The single bulk D2H of `out` happens once at the end of
// the tree, replacing the former per-point host expectation D2H + implicit
// device sync. Arithmetic is Tr(B·ρ) = Σ_i Σ_k B[i,k]·ρ[k,i] term-for-term; only
// the summation ORDER differs (strided + tree reduce vs host row-major), so
// results agree to a few ULPs, not bit-exact.
void launch_trace_BR(
    const heom::complex_t* B, const heom::complex_t* rho,
    heom::complex_t* out, int idx, int N, int M, int batch, int n_samples,
    const heom::ManifoldBlock& blk);

#endif  // HEOM_CORE_DIPOLE_APPLY_CUH
