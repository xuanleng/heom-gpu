#include "heom_derivatives.cuh"
#include "cuda_common.cuh"
#include "core/precision.hpp"
#include <cuComplex.h>

// ============================================================================
// Complex-multiply helpers now live in core/precision.hpp as precision-agnostic
// wrappers (c_scale = z·s real, c_scale_i = z·(i·s) pure imaginary; both strip
// the dead ×0 ops the general c_mul would do). Kept as local aliases so the
// kernel body below reads unchanged.
// ============================================================================
using heom::complex_t;
using heom::real_t;
using heom::c_make;
using heom::c_re;
using heom::c_im;
using heom::c_add;
using heom::c_sub;
using heom::c_mul;
using heom::c_conj;
using heom::c_scale;
using heom::c_scale_i;
using heom::c_from_double;

/**
 * HEOM right-hand side: five bath terms, plus an optional sink.
 * One thread owns one element inside the ADO block. Inactive ADOs are
 * skipped when active_flags is set; alpha 0 is always computed.
 * n_bath is the pigment count, not the Hilbert dimension N.
 */
// The read-only pointers are distinct cudaMalloc allocations (SystemContext /
// HEOMTrajectory), never aliased, so __restrict__ holds and the compiler can
// keep loads in registers across the inner loops. hamiltonian[N*N] and
// d_coupling[n_bath*N] are block-invariant, so each block caches them once in
// dynamic shared memory (sH/sD); the launcher passes the exact byte count.
// RK4-fused epilogue modes. The derivative value for element (α,i,j) is born
// in this kernel's `result` register; writing it to a k-buffer and re-reading
// it in a separate epilogue kernel is a pure M·N² memory round-trip plus a
// launch. Fusing the element-wise epilogue here makes it nearly free: this
// kernel is compute-bound, so the added
// streaming loads/stores hide under the arithmetic.
//
//   kDerivOnly  : drho_out[idx] = k                      (standalone API, tests)
//   kStage      : temp[idx] = c·k + rho[idx]             (RK4 stages 1-3)
//                 acc[idx]  = init_acc ? k : w·k + acc[idx]
//   kFinal      : rho[idx] += dt6·(w·k + acc[idx])       (RK4 stage 4,
//                 optional Kahan carry in comp)           in-place)
//
// Race safety: the derivative reads rho_in CROSS-element (Term 1 row/column,
// Term 4/5 neighbors), so a fused mode must never write the buffer serving as
// rho_in. rk4_step ping-pongs the two temp buffers (rho→a→b→a) to guarantee
// it; rho/acc/comp are only ever touched at the thread's own element.
//
// The epilogue expressions are kept token-identical to the retired standalone
// kernels, so the FP64 build stays bit-identical to history.
enum RK4Fuse { kDerivOnly = 0, kStage = 1, kFinal = 2 };

// Forcing a higher occupancy spills registers and slows this kernel.
// Leave the launch bounds at max threads.
template <int MODE, bool IF>
__global__ void __launch_bounds__(HEOM_BLOCK_SIZE)
compute_heom_derivatives_kernel(
    const complex_t* __restrict__ rho_in,             // [M][N][N] all density matrices (hier precision)
    complex_t* __restrict__ drho_out,                 // [M][N][N] output derivatives (hier precision)
    const complex_t* __restrict__ hamiltonian,        // [N][N] system Hamiltonian (hier precision)
    const int* __restrict__ neighbor_map_plus,        // [M × n_bath × (K+1)] plus neighbor indices
    const int* __restrict__ neighbor_map_minus,       // [M × n_bath × (K+1)] minus neighbor indices
    const complex_t* __restrict__ c_bath,             // [n_bath*(K+1)] per-site coeffs c_{nk} (hier precision)
    const real_t* __restrict__ coeff_plus,            // [M*n_bath*(K+1)] √[(α+1)|c|] (Term 4)
    const real_t* __restrict__ coeff_minus,           // [M*n_bath*(K+1)] √[α/|c|]    (Term 5)
    const real_t* __restrict__ markov_residue,        // [n_bath] per-site Markov coefficient
    const int8_t* __restrict__ level_indices,         // [M × n_bath × (K+1)] hierarchy indices α_{nk}
    const double* __restrict__ total_decay,           // [M] pre-computed Σ α_{nk}ν_k per ADO
    const int8_t* __restrict__ active_flags,          // [M] ADO activation flags (nullptr=all active)
    const int8_t* __restrict__ d_coupling,            // [n_bath × N] bath coupling d^(n)_i
    const int* __restrict__ u_list,                   // [n_launch_ado] compact ADO indices, or nullptr
    int n_launch_ado,                                 // ADOs to launch: u_count (compact) or M (full)
    int M,          // Number of hierarchy indices
    int N,          // System size (Hilbert dimension)
    int K,          // Last bath-pole index (= n_bath_modes_total == n_extra_poles);
                    // arrays [K+1]: k=0 Drude, k=1..K extra Matsubara/Padé poles.
    int n_bath,     // Number of physical baths (= n_states)
    // Manifold block: threads cover rows [row_off, row_off+n_row) × cols
    // [col_off, col_off+n_col) only. (0,N,0,N) is the whole matrix. Addressing
    // is still i*N+j — only the (i,j) set shrinks. See HEOMContext::row_off.
    int row_off, int n_row, int col_off, int n_col,
    bool sink_enabled,  // Whether sink is enabled
    int sink_site,      // Sink site index
    double gamma_sink,  // Sink rate (a.u.)
    // ---- fused RK4 epilogue (kStage/kFinal; all unused in kDerivOnly) ----
    // NOTE on aliasing: in RK4 stage 1 rho_state aliases rho_in (both = ρ).
    // Both are only READ in kStage, so the __restrict__ promise still holds
    // for every written buffer (temp_out/acc are always distinct allocations).
    complex_t* __restrict__ rho_state,       // ρ (kStage: read; kFinal: in-place)
    complex_t* __restrict__ temp_out,        // kStage: next stage's f() input
    complex_t* __restrict__ acc,             // running k1+2k2+2k3 (+k4 fold)
    complex_t* __restrict__ comp,            // kFinal: Kahan carry or nullptr
    complex_t stage_c,                       // kStage: dt/2 or dt
    complex_t stage_w,                       // fold weight (1 or 2)
    complex_t dt6,                           // kFinal: dt/6
    bool init_acc,                           // kStage stage 1: acc = k
    // ---- integrating-factor RK4 (IF=true only; ignored when IF=false) ----
    const real_t* __restrict__ if_e_half,    // [M] E  = exp(-½·dt·g_α)
    const real_t* __restrict__ if_e_full,    // [M] E2 = exp(-dt·g_α)
    int if_stage                             // IF kStage: 1/2/3
)
{
    // Layout: sH (N*N complex_t, aligned at smem base) then sD
    // (n_bath*N int8). ALL threads must reach the load + __syncthreads before
    // any return, or an early-returning thread would deadlock its siblings at
    // the barrier.
    extern __shared__ char smem[];
    complex_t* sH = reinterpret_cast<complex_t*>(smem);
    int8_t* sD = reinterpret_cast<int8_t*>(sH + N * N);

    const int tid = threadIdx.x;
    const int N2  = N * N;
    const int dc_size = n_bath * N;
    // H arrives already at hier precision (rounded once at upload), so
    // staging it is a plain copy — no per-block FP64 converts.
    for (int p = tid; p < N2;      p += blockDim.x) sH[p] = hamiltonian[p];
    for (int p = tid; p < dc_size; p += blockDim.x) sD[p] = d_coupling[p];
    __syncthreads();

    // Batch slot = grid.y (batch==1 ⇒ single row, b==0, boff==0 ⇒ the whole
    // batch machinery folds away to the pre-batch address math). Each slot owns
    // a full [M][N²] ρ block; the launch domain (n_launch_ado) may be < M in the
    // whole-step path, but the per-slot STRIDE is always M·N². All slots share
    // the same read-only tables, u_list and active_flags (union closure).
    const int b = blockIdx.y;
    const size_t boff = static_cast<size_t>(b) * M * N2;

    int global_idx = blockIdx.x * blockDim.x + tid;
    // Launch domain is n_launch_ado ADOs × the manifold block. Full M
    // (u_list == nullptr) or the compact closure set; whole matrix
    // (n_row == n_col == N) or one manifold block. Threads beyond it return.
    const int Nb = n_row * n_col;      // launcher guarantees n_row, n_col >= 1
    int total_elements = n_launch_ado * Nb;

    // Inactive-ADO contract: filtering is a per-step optimisation, not a death
    // sentence. The hierarchy is bidirectional — an inactive ρ̃_α can be
    // re-pumped above threshold by an active neighbor through Term 4/5, so it
    // must be able to reappear on a later filter pass. Therefore:
    //   active   → Term 1+2+3 + Term 4+5 + Term 6
    //   inactive → Term 4+5 only (the reactivation channel)
    // Term 1-3 and the sink read this ADO's own state (O(ε) when inactive), so
    // they are gated on is_active. Term 4/5 run on EVERY ADO and gate on the
    // *neighbor's* flag instead — that gate is what keeps the reactivation
    // channel alive. Filtering off ⇒ is_active ≡ true and every term runs.
    if (global_idx >= total_elements) return;

    int slot = global_idx / Nb;
    // Compact launch: slot indexes u_list to the real ADO id; full launch:
    // slot IS the ADO id. All downstream addressing uses the real α.
    int alpha = u_list ? u_list[slot] : slot;
    int remainder = global_idx % Nb;
    int i = row_off + remainder / n_col;   // Row within the block
    int j = col_off + remainder % n_col;   // Column within the block
    const int ij = i * N + j;           // this thread's (i,j) offset into an N×N ADO

    const bool is_active = (active_flags == nullptr) || (active_flags[alpha] != 0);

    const complex_t* rho_alpha = &rho_in[boff + static_cast<size_t>(alpha) * N2];
    complex_t result = c_make(0.0, 0.0);

    if (is_active) {
        // Term 1: -i [H, rho].
        //
        // The k-sum has two halves that read ρ differently:
        //   A(k):  sH[i,k] · ρ[k,j]   — ρ's ROW index varies, so A is zero
        //                               unless k is in the block's row range
        //   B(k):  ρ[i,k] · sH[k,j]   — ρ's COLUMN index varies, so B is zero
        //                               unless k is in the block's col range
        // Restricting each half to its own range is what turns the block
        // structure into arithmetic saved, and it is also what keeps garbage
        // out: with the full loop a NaN parked outside the block would come
        // back as NaN·0 = NaN instead of being skipped.
        //
        // ORDER IS THE WHOLE BALLGAME. Floating-point addition is not
        // associative, so "bit-identical to the full launch" holds only if the
        // surviving terms are accumulated in the order the full loop would
        // have produced them: ascending k, and at equal k, A before B. Hence
        // three branches rather than one predicated union loop — the branch is
        // warp-uniform (all four bounds are kernel arguments) so it costs
        // nothing, whereas a union loop would reorder the sum.
        //
        // What this does NOT preserve: a result that would have ended at -0.0
        // can come out as +0.0, because the skipped iterations used to add an
        // exact +0.0. That is the one documented deviation.
        if (row_off == col_off && n_row == n_col) {
            // Diagonal block (includes the whole-matrix default): the two
            // ranges coincide, so keep the original interleaving verbatim.
            const int k_end = row_off + n_row;
            for (int k = row_off; k < k_end; ++k) {
                result = c_add(result,
                    c_mul(sH[i * N + k], rho_alpha[k * N + j]));
                result = c_sub(result,
                    c_mul(rho_alpha[i * N + k], sH[k * N + j]));
            }
        } else if (row_off < col_off) {
            // Disjoint, rows below columns: every surviving A precedes every
            // surviving B in ascending k.
            const int a_end = row_off + n_row;
            for (int k = row_off; k < a_end; ++k) {
                result = c_add(result,
                    c_mul(sH[i * N + k], rho_alpha[k * N + j]));
            }
            const int b_end = col_off + n_col;
            for (int k = col_off; k < b_end; ++k) {
                result = c_sub(result,
                    c_mul(rho_alpha[i * N + k], sH[k * N + j]));
            }
        } else {
            // Disjoint, columns below rows: B first, same argument.
            const int b_end = col_off + n_col;
            for (int k = col_off; k < b_end; ++k) {
                result = c_sub(result,
                    c_mul(rho_alpha[i * N + k], sH[k * N + j]));
            }
            const int a_end = row_off + n_row;
            for (int k = row_off; k < a_end; ++k) {
                result = c_add(result,
                    c_mul(sH[i * N + k], rho_alpha[k * N + j]));
            }
        }
        // Multiply by -i (hbar = 1): (re, im) -> (im, -re).
        result = c_scale_i(static_cast<real_t>(-1.0), result);

        // Term 2: hierarchy decay. total_decay is precomputed.
        // IF path integrates Term 2 EXACTLY in the epilogue (E/E2 factors), so
        // the RK4 derivative must NOT include it — else it is double-counted.
        //
        // Filtered-out ADOs are not decayed on the plain path; the IF
        // epilogue still scales them. Do not gate the IF epilogue on is_active.
        if constexpr (!IF) {
            double decay = total_decay[alpha];
            if (decay > 0.0) {
                result = c_sub(result, c_scale(static_cast<real_t>(decay), rho_alpha[ij]));
            }
        }

        // Term 3: Markov tail. Diagonal Q gives (d_i - d_j)^2 * rho_ij.
        real_t markov_sum = 0;
        for (int n = 0; n < n_bath; ++n) {
            if (sD[n * N + i] != sD[n * N + j]) markov_sum += markov_residue[n];
        }
        if (markov_sum != 0) {
            result = c_sub(result, c_scale(markov_sum, rho_alpha[ij]));
        }
    }

    // Term 4 (plus neighbor) and Term 5 (minus neighbor), one loop.
    // Both are zero when d_i = d_j = 0.
    for (int n = 0; n < n_bath; ++n) {
        const int di = sD[n * N + i];
        const int dj = sD[n * N + j];
        if (di == 0 && dj == 0) continue;   // Term 4 and Term 5 are both zero
        const int diff = di - dj;
        const int base = alpha * n_bath * (K + 1) + n * (K + 1);
        // Per-site coefficient base: c_{nk}/|c_{nk}| live at cbase + k (flat
        // row-major n*(K+1)+k), distinct from the per-ADO map base above.
        const int cbase = n * (K + 1);

        for (int k_idx = 0; k_idx <= K; ++k_idx) {
            const int map_idx = base + k_idx;
            const int a_nk = level_indices[map_idx];

            // ---- Term 4: plus neighbor (needs diff != 0) ----
            if (diff != 0) {
                int alpha_plus = neighbor_map_plus[map_idx];
                if (alpha_plus >= 0 &&
                    (active_flags == nullptr || active_flags[alpha_plus])) {
                    const complex_t* rho_plus = &rho_in[boff + static_cast<size_t>(alpha_plus) * N2];
                    // diff is -1 or +1. The prefactor is pure imaginary.
                    complex_t comm = c_scale(static_cast<real_t>(diff), rho_plus[ij]);
                    result = c_sub(result, c_scale_i(coeff_plus[map_idx], comm));
                }
            }

            // ---- Term 5: minus neighbor (needs alpha_nk > 0) ----
            if (a_nk > 0) {
                int alpha_minus = neighbor_map_minus[map_idx];
                if (alpha_minus >= 0 &&
                    (active_flags == nullptr || active_flags[alpha_minus])) {
                    const complex_t* rho_minus = &rho_in[boff + static_cast<size_t>(alpha_minus) * N2];
                    // c_k is already at hierarchy precision.
                    complex_t c_k = c_bath[cbase + k_idx];
                    // factor = c_k*d_i - conj(c_k)*d_j, and d is 0 or 1.
                    complex_t factor = c_make(0.0, 0.0);
                    if (di) factor = c_add(factor, c_k);
                    if (dj) factor = c_sub(factor, c_conj(c_k));
                    complex_t matrix_term = c_mul(factor, rho_minus[ij]);
                    // Prefactor i*sqrt(alpha/|c|) is pure imaginary.
                    result = c_sub(result, c_scale_i(coeff_minus[map_idx], matrix_term));
                }
            }
        }
    }
    
    // Term 6: Lindblad sink, self-evolution, gated like Term 1.
    // The sink modifies the system Liouvillian and acts on every tier.
    // Kreisbeck & Kramer, JPCL 3, 2828 (2012), Eq. (3);
    //        Ishizaki & Fleming, PNAS 106, 17255 (2009), Eq. (S3);
    //        Strümpfer & Schulten, JCTC 8, 2808 (2012), Eq. (1)
    //
    // ∂ρ̃_α/∂t |_sink = -γ_sink/2 · (δ_{i,s} + δ_{j,s}) · ρ̃_α[i,j]
    //
    // Reads this ADO's own state, so it is self-evolution and shares the
    // is_active gate with Term 1-3.
    // ========================================================================
    if (is_active && sink_enabled) {
        if (i == sink_site && j == sink_site) {
            result = c_sub(result, c_scale(static_cast<real_t>(gamma_sink), rho_alpha[ij]));
        }
        else if (i == sink_site || j == sink_site) {
            result = c_sub(result, c_scale(static_cast<real_t>(gamma_sink / 2.0), rho_alpha[ij]));
        }
    }
    
    // kDerivOnly writes the derivative. kStage / kFinal fold the RK4
    // epilogue in place. The FP64 build stays bit-identical.
    const size_t output_idx = boff + static_cast<size_t>(alpha) * N2 + ij;

    if constexpr (MODE == kDerivOnly) {
        drho_out[output_idx] = result;
    } else if constexpr (MODE == kStage) {
        if constexpr (IF) {
            // Lawson IF-RK4 stage (per-ADO E=exp(-½dt·g), E2=exp(-dt·g)):
            //   s1: temp = E·ρ + (E·c)·k,  acc = E2·k
            //   s2: temp = E·ρ + c·k,      acc += w·(E·k)
            //   s3: temp = E2·ρ + (E·c)·k, acc += w·(E·k)
            const real_t E  = if_e_half[alpha];
            const real_t E2 = if_e_full[alpha];
            const complex_t base = c_scale((if_stage == 3) ? E2 : E,
                                           rho_state[output_idx]);
            const complex_t kc = (if_stage == 2)
                ? c_mul(stage_c, result)
                : c_mul(c_scale(E, stage_c), result);
            temp_out[output_idx] = c_add(base, kc);
            acc[output_idx] = (if_stage == 1)
                ? c_scale(E2, result)
                : c_add(c_mul(stage_w, c_scale(E, result)), acc[output_idx]);
        } else {
            temp_out[output_idx] = c_add(c_mul(stage_c, result), rho_state[output_idx]);
            acc[output_idx] = init_acc ? result
                                       : c_add(c_mul(stage_w, result), acc[output_idx]);
        }
    } else {  // kFinal
        if constexpr (IF) {
            // ρ(t+dt) = E2·ρ(t) + dt/6·(k1+2k2+2k3+k4), acc already folds the
            // per-stage E factors. The homogeneous ρ term carries E2, so the
            // Kahan carry (which tracks lost bits OF ρ) must be scaled by E2 too
            // — else the compensation drifts against the rescaled ρ.
            const real_t E2 = if_e_full[alpha];
            const complex_t folded = c_add(c_mul(stage_w, result), acc[output_idx]);
            const complex_t y_raw  = c_mul(dt6, folded);
            const complex_t base   = c_scale(E2, rho_state[output_idx]);
            if (comp == nullptr) {
                rho_state[output_idx] = c_add(y_raw, base);
            } else {
                const complex_t cc = c_scale(E2, comp[output_idx]);
                const complex_t y  = c_sub(y_raw, cc);
                const complex_t t  = c_add(base, y);
                comp[output_idx] = c_sub(c_sub(t, base), y);
                rho_state[output_idx] = t;
            }
        } else {
            const complex_t folded = c_add(c_mul(stage_w, result), acc[output_idx]);
            const complex_t y_raw  = c_mul(dt6, folded);
            if (comp == nullptr) {
                rho_state[output_idx] = c_add(y_raw, rho_state[output_idx]);
            } else {
                // Kahan: y = increment − carry; t = ρ + y; carry = (t−ρ)−y; ρ = t.
                const complex_t r = rho_state[output_idx];
                const complex_t cc = comp[output_idx];
                const complex_t y = c_sub(y_raw, cc);
                const complex_t t = c_add(r, y);
                comp[output_idx] = c_sub(c_sub(t, r), y);
                rho_state[output_idx] = t;
            }
        }
    }
}

// ============================================================================
// Host launcher (see heom_derivatives.cuh for the full contract)
// ============================================================================

namespace {

// Queried once: this is a device property, and every RK4 stage of every step
// goes through here. Nothing in the codebase calls cudaSetDevice, so the
// current device cannot change under us.
size_t current_derivative_smem_limit_bytes() {
    static const size_t limit = [] {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        int bytes = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &bytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
        return static_cast<size_t>(bytes);
    }();
    return limit;
}

// One launch path for all three modes: identical domain/smem math, only the
// template parameter and the epilogue pointer pack differ.
template <int MODE, bool IF>
void launch_derivatives(const HEOMContext& ctx, const heom::RK4StageArgs& st)
{
    heom::validate_derivative_context(ctx);
    // Launch domain: compact closure (u_list set) or full M. checked_total_elements
    // reuses the M·N² overflow guard with the launch ADO count in place of M.
    const int launch_ado =
        ctx.is_wholestep() ? ctx.u_count : ctx.M;
    if (launch_ado < 0 || launch_ado > ctx.M) {
        throw std::invalid_argument("HEOMContext.u_count out of range [0, M]");
    }
    if (launch_ado == 0) {
        // Empty launch set (no active ADOs at all): nothing to compute. In the
        // whole-step path α∉U is frozen, so drho_out needs no zeroing here.
        return;
    }
    // The overflow guard still uses the FULL N²: the manifold block only
    // shrinks the launch, while addressing (and hence the 2^31 index ceiling)
    // still spans alpha*N² + i*N + j. Guarding the shrunken domain would let a
    // blocked run slip past a limit its own addressing does not respect.
    (void)heom::checked_total_elements(launch_ado, ctx.N);
    const int total_elements = launch_ado * (ctx.n_row * ctx.n_col);
    const int num_blocks = (total_elements + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;

    // Batch is an ORTHOGONAL grid dimension (grid.y). batch==1 ⇒ grid.y==1 and
    // the launch is byte-identical to the pre-batch path. Slot b (blockIdx.y)
    // strides its own [M][N²] ρ block via boff inside the kernel; grid.x still
    // covers exactly the per-slot launch domain (n_launch_ado·N²).
    if (ctx.batch < 1) {
        throw std::invalid_argument("HEOMContext.batch must be >= 1");
    }
    const dim3 grid(static_cast<unsigned>(num_blocks),
                    static_cast<unsigned>(ctx.batch));

    // Dynamic shared memory == sH (N^2 complex_t) + sD (n_bath*N int8).
    // sH placed first so its 16-B alignment is satisfied by smem base.
    const size_t smem_bytes = heom::checked_derivative_smem_bytes(
        ctx.N, ctx.n_bath, current_derivative_smem_limit_bytes());

    compute_heom_derivatives_kernel<MODE, IF><<<grid, HEOM_BLOCK_SIZE, smem_bytes>>>(
        ctx.rho_in, ctx.drho_out, ctx.hamiltonian,
        ctx.neighbor_map_plus, ctx.neighbor_map_minus,
        ctx.c_bath, ctx.coeff_plus, ctx.coeff_minus, ctx.markov_residue,
        ctx.level_indices, ctx.total_decay, ctx.active_flags,
        ctx.d_coupling,
        ctx.u_list, launch_ado,
        ctx.M, ctx.N, ctx.K, ctx.n_bath,
        ctx.row_off, ctx.n_row, ctx.col_off, ctx.n_col,
        ctx.sink_enabled, ctx.sink_site, ctx.gamma_sink,
        st.rho, st.temp_out, st.acc, st.comp,
        st.c, st.w, st.dt6, st.init_acc,
        st.e_half, st.e_full, st.if_stage
    );

    CUDA_CHECK_KERNEL();
}

}  // namespace

void compute_heom_derivatives(const HEOMContext& ctx)
{
    launch_derivatives<kDerivOnly, false>(ctx, heom::RK4StageArgs{});
}

void compute_heom_derivatives_stage(const HEOMContext& ctx,
                                    const heom::RK4StageArgs& st)
{
    launch_derivatives<kStage, false>(ctx, st);
}

void compute_heom_derivatives_final(const HEOMContext& ctx,
                                    const heom::RK4StageArgs& st)
{
    launch_derivatives<kFinal, false>(ctx, st);
}

// Integrating-factor RK4 variants: same launch geometry, IF=true instantiation
// skips Term 2 in the derivative and applies the per-ADO E/E2 factors in the
// epilogue. st.e_half/st.e_full/st.if_stage MUST be populated by the caller.
void compute_heom_derivatives_stage_if(const HEOMContext& ctx,
                                       const heom::RK4StageArgs& st)
{
    launch_derivatives<kStage, true>(ctx, st);
}

void compute_heom_derivatives_final_if(const HEOMContext& ctx,
                                       const heom::RK4StageArgs& st)
{
    launch_derivatives<kFinal, true>(ctx, st);
}

// ============================================================================
// Integrating-factor precompute: per-ADO E=exp(-½·dt·g_α), E2=exp(-dt·g_α).
// The exponentials are evaluated in FP64 (they are precision-critical and
// shared across the whole run) and stored at hier precision, so the O(M·N²)
// threads that read them every step pay no FP64-pipe cost. O(M) work per dt.
// ============================================================================
__global__ void compute_if_factors_kernel(
    const double* __restrict__ total_decay, double dt_au, int M,
    real_t* __restrict__ e_half, real_t* __restrict__ e_full)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= M) return;
    const double g = total_decay[a];
    e_half[a] = static_cast<real_t>(exp(-0.5 * dt_au * g));
    e_full[a] = static_cast<real_t>(exp(-dt_au * g));
}

void compute_if_factors(const double* d_total_decay, double dt_au, int M,
                        real_t* d_e_half, real_t* d_e_full)
{
    if (M <= 0) return;
    const int block = 256;
    const int grid  = (M + block - 1) / block;
    compute_if_factors_kernel<<<grid, block>>>(
        d_total_decay, dt_au, M, d_e_half, d_e_full);
    CUDA_CHECK_KERNEL();
}
