#ifndef HEOM_CORE_DRESSED_START_CUH
#define HEOM_CORE_DRESSED_START_CUH

#include "core/heom_derivatives.cuh"   // HEOMContext

// One tier of the dressed (displaced) start, fluorescence theory eq:dressed:
// every ADO α with Σα = tier, of every slot in grid.y = ctx.batch, becomes
//   ρ̃_α[ij] = −i Σ_{n,k: α_nk>0} coeff_minus·(c d_i − c̄ d_j)·ρ̃_{α−e_nk}[ij]
//             / (total_decay[α] + Σ_n markov_residue[n]·(d_i ≠ d_j)),
// i.e. L₀ρ̃ = 0 with L₀ = Terms 2 + 3 + 5 of heom_derivatives.cu, solved
// with the same coefficient arrays. Reads only tier − 1, writes only tier, so
// the caller launches tier = 1..L in order. Arithmetic in double, rounded once
// on store. ADOs of other tiers are untouched.
void launch_dress_tier(const HEOMContext& ctx, heom::complex_t* rho, int tier);

#endif  // HEOM_CORE_DRESSED_START_CUH
