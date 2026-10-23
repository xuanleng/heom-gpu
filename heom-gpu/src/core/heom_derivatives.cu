#include "heom_derivatives.cuh"
#include "cuda_common.cuh"
#include "core/precision.hpp"
#include "core/mode_layout.hpp"
#include "core/ado_norm.cuh"
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
// keep loads in registers across the inner loops. hamiltonian[N*N] (real part
// only) and d_coupling[n_bath*N] are block-invariant, so each block caches
// the sub-blocks it reads in dynamic shared memory (sA/sB/sD); the launcher
// passes the exact byte count.
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
//
// A tighter launch bound spills and was dropped. PART=false compiles out
// partial staging when the whole Hamiltonian is staged. SLOTS compiles out
// the per-slot pointer path when batch == 1.
template <int MODE, bool IF, bool PART, bool SLOTS>
__global__ void __launch_bounds__(HEOM_BLOCK_SIZE)
compute_heom_derivatives_kernel(
    const complex_t* __restrict__ rho_in,             // [B][M][Nb] all density matrices, block-compact (hier precision)
    complex_t* __restrict__ drho_out,                 // [B][M][Nb] output derivatives (hier precision)
    const complex_t* __restrict__ hamiltonian,        // [N][N] system Hamiltonian (hier precision)
    const int* __restrict__ neighbor_map_plus,        // [M × D] plus neighbor indices
    const int* __restrict__ neighbor_map_minus,       // [M × D] minus neighbor indices
    const complex_t* __restrict__ c_bath,             // [D] per-site coeffs c_{nk} (hier precision)
    const complex_t* __restrict__ cbar_bath,          // [D] right-action coeffs c̄_{nk} (hier precision)
    const real_t* __restrict__ coeff_plus,            // [M*D] √[(α+1)s] (Term 4)
    const real_t* __restrict__ coeff_minus,           // [M*D] √[α/s]    (Term 5)
    const real_t* __restrict__ markov_residue,        // [n_bath] per-site Markov coefficient
    const int8_t* __restrict__ level_indices,         // [M × D] hierarchy indices α_{nk}
    const int* __restrict__ mode_off,                 // [n_bath+1] ladder prefix sums, mode_off[n_bath]==D
    const cuDoubleComplex* __restrict__ total_decay,  // [M] pre-computed Σ α_{nk}ν_k per ADO (complex)
    const int8_t* __restrict__ active_flags_all,      // [B][M] per-slot ADO activation flags (nullptr=all active)
    const int8_t* __restrict__ d_coupling,            // [n_bath × N] bath coupling d^(n)_i
    const int* __restrict__ u_list_all,               // compact ADO indices, or nullptr
    const int* __restrict__ u_off,                    // [B+1] per-slot segments of u_list_all, or nullptr = shared list
    const int* __restrict__ block_off,                // [B+1] flat block prefixes, or nullptr
    const uint8_t* __restrict__ u_flags_all,          // [B][M] α ∈ U_b for a full launch, or nullptr = all
    int n_launch_ado,                                 // ADOs to launch: max_b |U_b| (compact) or M (full)
    int n_slots,                                      // batch width; used by flat per-slot walk (SLOTS)
    int M,          // Number of hierarchy indices
    int N,          // System size (Hilbert dimension)
    int D,          // Total ladder count Σ_n P_n (row width of every per-ADO table)
    int n_bath,     // Number of physical baths (= n_states)
    // Manifold block: the STORAGE SHAPE of every ADO and the launch domain.
    // ρ_α(i,j) for (i,j) in rows [row_off, row_off+n_row) × cols
    // [col_off, col_off+n_col) lives at rho[α·Nb + (i−row_off)·n_col +
    // (j−col_off)], Nb = n_row·n_col; nothing outside the block is stored.
    // (0,N,0,N) is the whole matrix. See HEOMContext::block.
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
    const complex_t* __restrict__ if_e_half, // [M] E  = exp(-½·dt·g_α), complex
    const complex_t* __restrict__ if_e_full, // [M] E2 = exp(-dt·g_α)
    int if_stage,                            // IF kStage: 1/2/3
    unsigned long long* __restrict__ norm_sq_bits  // kFinal: fused max|ρ_new|² [M] or nullptr
)
{
    // H is real in every basis this code builds (SystemContext throws
    // otherwise), so only the real part is staged: half the smem and half the
    // Term 1 multiplies (measured 1.88x on LHC-II ESA when it took sH from
    // 1 to 2 blocks/SM on large Hamiltonians).
    //
    // Only the part of H this block's elements actually touch is staged
    // (lhc2-smem step 1). Term 1's A half reads H[i, row range] for the
    // rows i of this block's elements; the B half reads H[col range, j] for j
    // in the col range. One 256-thread block covers ~blockDim/n_col rows i,
    // so for a disjoint block (LHC-II ESA f×e = 91×14) staging the whole N×N
    // H loaded ~5.6x more than was read, and its 46/91 KB (mixed/FP64) capped
    // occupancy at 2/1 blocks per SM. Layout (real_t first, then int8):
    //   diagonal block (row range == col range, incl. the whole matrix):
    //       sB = H[row range][row range], n_row×n_row, serves A and B
    //   disjoint block:
    //       sA = R rows × n_row: sA[lr·n_row + (k−row_off)] = H[i][k], with
    //            lr = (li − li0) mod n_row — rows numbered cyclically from the
    //            block's first element, so a block straddling two ADOs still
    //            needs at most R distinct li
    //       sB = H[col range][col range], n_col×n_col
    //   sD = the d_coupling columns of those same states (step 2), int8:
    //       diagonal: sD[n·n_row + (s−row_off)], s in the row range
    //       disjoint: sDi[n·sA_rows + lr] (rows, cyclic like sA) then
    //                 sDj[n·n_col + (j−col_off)]
    //       (was all n_bath×N: 1484 B for LHC-II ESA, as many byte loads
    //       per block as the H sub-blocks after step 1)
    // The staged values and the k order of every sum are unchanged, so the
    // result is bit-identical to staging all of H.
    // ALL threads must reach the load + __syncthreads before any return, or
    // an early-returning thread would deadlock its siblings at the barrier.
    extern __shared__ char smem[];
    const bool diag_block = (row_off == col_off && n_row == n_col);
    const int Nb_blk = n_row * n_col;
    const int Nb = Nb_blk;      // launcher guarantees n_row, n_col >= 1

    // Batch slot addressing (per-slot active sets, 2026-10-22 / flat 2026-10-23).
    // Each slot owns a [M][Nb] ρ block; stride is always M·Nb. Compact+SLOTS
    // with u_off+block_off: flat launch (grid.y=1); binary-search block_off
    // for (b, local_bx). Else: b = blockIdx.y (dense / B=1 / shared list).
    int b = blockIdx.y;
    int local_bx = blockIdx.x;
    const int* u_list = u_list_all;
    int n_launch_b = n_launch_ado;
    if constexpr (SLOTS) {
        if (u_list && u_off && block_off) {
            // Flat launch: binary search blockIdx.x in block_off[0..n_slots].
            int lo = 0, hi = n_slots;
            while (lo + 1 < hi) {
                const int mid = (lo + hi) >> 1;
                if (block_off[mid] <= static_cast<int>(blockIdx.x)) lo = mid;
                else hi = mid;
            }
            b = lo;
            local_bx = static_cast<int>(blockIdx.x) - block_off[b];
            u_list += u_off[b];
            n_launch_b = u_off[b + 1] - u_off[b];
        }
    }
    const size_t boff = static_cast<size_t>(b) * M * Nb;
    const int8_t* active_flags = active_flags_all;
    const uint8_t* u_flags = u_flags_all;
    if constexpr (SLOTS) {
        if (active_flags) active_flags += static_cast<size_t>(b) * M;
        if (u_flags) u_flags += static_cast<size_t>(b) * M;
    }
    int total_elements = n_launch_b * Nb;

    // e0/e_last use local_bx / n_launch_b so PART staging sees the same
    // per-slot element range the flat launch (SLOTS) actually covers.
    const int e0 = local_bx * blockDim.x;                 // first element of this block
    const int e_last = min(e0 + static_cast<int>(blockDim.x),
                           n_launch_b * Nb_blk) - 1;      // last element (>= e0)
    const int lij0 = e0 % Nb_blk;
    const int li0  = lij0 / n_col;
    // Step 6: small systems stage the whole H + d_coupling (contiguous
    // copies, small-table loop); block-uniform, same rule as the launcher.
    const bool whole = !PART || (diag_block && n_row == N)
                    || heom::derivative_stage_whole(N, n_bath);
    const int R    = (diag_block || whole) ? 0
                   : min(n_row, (lij0 + (e_last - e0)) / n_col - li0 + 1);
    // sA capacity = heom::derivative_smem_rows(n_row, n_col, HEOM_BLOCK_SIZE)
    // rows, the same worst case the launcher sized smem with (host-only
    // helper, so the one-line formula is repeated here).
    const int sA_rows  = min(n_row, (n_col - 1 + HEOM_BLOCK_SIZE - 1) / n_col + 1);
    const int sA_elems = (diag_block || whole) ? 0 : sA_rows * n_row;
    // One real_t array and one int8 array, addressed by INTEGER offsets
    // (step 3). Selecting between two smem pointers per thread (step 1/2:
    // hA = diag ? sB+.. : sA+..) made nvcc fall back to generic loads and a
    // stack frame (cuobjdump STACK:8/16), which cost FMO 5-9%.
    real_t* sH = reinterpret_cast<real_t*>(smem);
    const int offB = sA_elems;                               // sB = sH + offB
    int8_t* sD = reinterpret_cast<int8_t*>(sH + (whole ? N * N : sA_elems + n_col * n_col));
    const int offDj = (diag_block || whole) ? 0 : n_bath * sA_rows;  // sDj = sD + offDj
    const int sDi_stride = whole ? N : (diag_block ? n_row : sA_rows);
    const int sB_stride  = whole ? N : n_col;   // row stride of the H[k][j] reads and of sDj
    // mode_off stays in global memory: the loads below are block-uniform
    // (address depends only on n), so they broadcast out of L1; staging 15
    // ints into smem would buy nothing and would have to be realigned after
    // the int8 sD block.
    const heom::ModeLayout ml{n_bath, D, mode_off};

    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    // H arrives already at hier precision (rounded once at upload), so
    // staging it is a plain copy — no per-block FP64 converts.
    if (whole) {
        // Whole matrix (dynamics, or any system under the step-6 byte limit):
        // sH is H itself, sD is d_coupling itself — contiguous copies,
        // exactly the small-table loop (no index math).
        #pragma unroll 1
        for (int p = tid; p < N * N; p += nthr) sH[p] = c_re(hamiltonian[p]);
        #pragma unroll 1
        for (int p = tid; p < n_bath * N; p += nthr) sD[p] = d_coupling[p];
    } else {
        // 2-D copies with one division per thread per array (not per
        // element): (r, c) of p = tid + t·nthr advance by (nthr / w, nthr % w).
        // `unroll 1` everywhere here: unrolled, these loops grew the kernel
        // from ~2.1k to ~3.1k SASS instructions (35 STS) and FP64 spilled.
        if (!diag_block) {          // sA: R cyclic rows × n_row
            int r = tid / n_row, c = tid - (tid / n_row) * n_row;
            const int dr = nthr / n_row, dc = nthr - dr * n_row;
            #pragma unroll 1
            for (int p = tid; p < R * n_row; p += nthr) {
                int li_s = li0 + r;
                if (li_s >= n_row) li_s -= n_row;
                sH[p] = c_re(hamiltonian[(row_off + li_s) * N + row_off + c]);
                c += dc; r += dr; if (c >= n_row) { c -= n_row; ++r; }
            }
        }
        {                           // sB: n_col × n_col
            int r = tid / n_col, c = tid - (tid / n_col) * n_col;
            const int dr = nthr / n_col, dc = nthr - dr * n_col;
            #pragma unroll 1
            for (int p = tid; p < n_col * n_col; p += nthr) {
                sH[offB + p] = c_re(hamiltonian[(col_off + r) * N + col_off + c]);
                c += dc; r += dr; if (c >= n_col) { c -= n_col; ++r; }
            }
        }
        if (diag_block) {           // sD: n_bath × n_row
            int n = tid / n_row, c = tid - (tid / n_row) * n_row;
            const int dn = nthr / n_row, dc = nthr - dn * n_row;
            #pragma unroll 1
            for (int p = tid; p < n_bath * n_row; p += nthr) {
                sD[p] = d_coupling[n * N + row_off + c];
                c += dc; n += dn; if (c >= n_row) { c -= n_row; ++n; }
            }
        } else {                    // sDi: n_bath × R (stride sA_rows), sDj: n_bath × n_col
            int n = tid / R, c = tid - (tid / R) * R;
            const int dn = nthr / R, dc = nthr - dn * R;
            #pragma unroll 1
            for (int p = tid; p < n_bath * R; p += nthr) {
                int li_s = li0 + c;
                if (li_s >= n_row) li_s -= n_row;
                sD[n * sA_rows + c] = d_coupling[n * N + row_off + li_s];
                c += dc; n += dn; if (c >= R) { c -= R; ++n; }
            }
            n = tid / n_col; c = tid - (tid / n_col) * n_col;
            const int dn2 = nthr / n_col, dc2 = nthr - dn2 * n_col;
            #pragma unroll 1
            for (int p = tid; p < n_bath * n_col; p += nthr) {
                sD[offDj + p] = d_coupling[n * N + col_off + c];
                c += dc2; n += dn2; if (c >= n_col) { c -= n_col; ++n; }
            }
        }
    }
    __syncthreads();


    // Slot / launch domain resolved above (local_bx, n_launch_b, total_elements).

    int global_idx = local_bx * blockDim.x + tid;

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

    // α∉U is frozen on every launch domain. Its derivative is exactly zero
    // (inactive, no active neighbor), but the IF epilogue would still scale ρ
    // and its carry by E2 (plain RK4 adds an exact zero, a no-op up to signed
    // zeros). The compact launch never reaches here for such α; the full
    // launch must agree. Stage buffers for α∉U go stale — harmless: a stage
    // value is read only where its ADO is active (self terms, or as an active
    // neighbor), and α∉U is inactive.
    if constexpr (MODE != kDerivOnly) {
        if (u_flags != nullptr && u_flags[alpha] == 0) return;
    }
    // Block-local element index lij = remainder; (i, j) are the GLOBAL matrix
    // indices it stands for — H, d_coupling and the sink site are addressed by those.
    // ρ is addressed by the local ones: ρ_α(k, j) is rho_alpha[(k−row_off)·n_col
    // + lj], ρ_α(i, k) is rho_alpha[li·n_col + (k−col_off)].
    const int lij = global_idx % Nb;
    const int li  = lij / n_col;
    const int lj  = lij % n_col;
    const int i   = row_off + li;          // Row within the matrix
    const int j   = col_off + lj;          // Column within the matrix

    const bool is_active = (active_flags == nullptr) || (active_flags[alpha] != 0);
    // Row H[i, row range] in the staged copy: sB row li (diagonal block) or
    // sA row lr (disjoint block, cyclic from li0 — see the prologue).
    int lr = li - li0;
    if (lr < 0) lr += n_row;
    // H[i, k] = sH[hA + k − row_off]; H[k, j] = sH[hB + (k − col_off)·sB_stride].
    const int hA = whole ? ((row_off + li) * N + row_off)
                 : (diag_block ? (offB + li * n_col) : (lr * n_row));
    const int hB = whole ? (col_off * N + col_off + lj) : (offB + lj);
    // d^(n)_i = sD[dI + n·sDi_stride], d^(n)_j = sD[dJ + n·sB_stride].
    const int dI = whole ? (row_off + li) : (diag_block ? li : lr);
    const int dJ = whole ? (col_off + lj) : (offDj + lj);

    const complex_t* rho_alpha = &rho_in[boff + static_cast<size_t>(alpha) * Nb];
    complex_t result = c_make(0.0, 0.0);

    if (is_active) {
        // Term 1: -i [H, rho].
        //
        // The k-sum has two halves that read ρ differently:
        //   A(k):  H[i,k] · ρ[k,j]    — ρ's ROW index varies, so A is zero
        //                               unless k is in the block's row range
        //   B(k):  ρ[i,k] · H[k,j]    — ρ's COLUMN index varies, so B is zero
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
                    c_scale(sH[hA + k - row_off], rho_alpha[(k - row_off) * n_col + lj]));
                result = c_sub(result,
                    c_scale(sH[hB + (k - col_off) * sB_stride], rho_alpha[li * n_col + (k - col_off)]));
            }
        } else if (row_off < col_off) {
            // Disjoint, rows below columns: every surviving A precedes every
            // surviving B in ascending k.
            const int a_end = row_off + n_row;
            for (int k = row_off; k < a_end; ++k) {
                result = c_add(result,
                    c_scale(sH[hA + k - row_off], rho_alpha[(k - row_off) * n_col + lj]));
            }
            const int b_end = col_off + n_col;
            for (int k = col_off; k < b_end; ++k) {
                result = c_sub(result,
                    c_scale(sH[hB + (k - col_off) * sB_stride], rho_alpha[li * n_col + (k - col_off)]));
            }
        } else {
            // Disjoint, columns below rows: B first, same argument.
            const int b_end = col_off + n_col;
            for (int k = col_off; k < b_end; ++k) {
                result = c_sub(result,
                    c_scale(sH[hB + (k - col_off) * sB_stride], rho_alpha[li * n_col + (k - col_off)]));
            }
            const int a_end = row_off + n_row;
            for (int k = row_off; k < a_end; ++k) {
                result = c_add(result,
                    c_scale(sH[hA + k - row_off], rho_alpha[(k - row_off) * n_col + lj]));
            }
        }
        // Multiply by -i (hbar = 1): (re, im) -> (im, -re).
        result = c_scale_i(static_cast<real_t>(-1.0), result);

        // Term 2: hierarchy decay. total_decay is precomputed.
        // IF path integrates Term 2 EXACTLY in the epilogue (E/E2 factors), so
        // the RK4 derivative must NOT include it — else it is double-counted.
        //
        // Inactive ADOs inside U are not decayed on the plain path; the IF
        // epilogue still scales them. Leave that difference: gating the IF
        // epilogue on is_active would change its result.
        // g_α is complex (conjugate ν± of a vibrational pair), so this is a
        // complex multiply, unconditionally: the root ADO's g = 0 costs one
        // c_mul(0, ρ) = 0 and a c_sub — no ordering test on a complex number.
        if constexpr (!IF) {
            const complex_t decay = c_from_double(total_decay[alpha]);
            result = c_sub(result, c_mul(decay, rho_alpha[lij]));
        }

        // Term 3: Markov tail. Diagonal Q gives (d_i - d_j)^2 * rho_ij.
        real_t markov_sum = 0;
        for (int n = 0; n < n_bath; ++n) {
            if (sD[dI + n * sDi_stride] != sD[dJ + n * sB_stride]) markov_sum += markov_residue[n];
        }
        if (markov_sum != 0) {
            result = c_sub(result, c_scale(markov_sum, rho_alpha[lij]));
        }
    }

    // Term 4 (plus neighbor) and Term 5 (minus neighbor), one loop.
    // Both are zero when d_i = d_j = 0. cbar is the right-action coefficient.
    // Visit only baths with d_i or d_j nonzero, lowest set bit first, so the
    // surviving terms stay in ascending-n order. n_bath > 64 keeps the dense
    // loop. Neighbour offsets are 32-bit; the host checks they fit. Keep the
    // level_indices test before the minus-map load: most a_nk are zero.
    const complex_t* __restrict__ rho_slot = rho_in + boff;
    const bool use_mask = n_bath <= 64;
    unsigned long long bmask = 0ull;
    if (use_mask) {
        for (int n = 0; n < n_bath; ++n)
            if ((sD[dI + n * sDi_stride] | sD[dJ + n * sB_stride]) != 0) bmask |= 1ull << n;
    }
    for (int n = 0; n < n_bath; ++n) {
        if (use_mask) {
            if (bmask == 0ull) break;
            n = __ffsll(static_cast<long long>(bmask)) - 1;
            bmask &= bmask - 1ull;
        }
        const int di = sD[dI + n * sDi_stride];
        const int dj = sD[dJ + n * sB_stride];
        if (di == 0 && dj == 0) continue;   // Term 4 and Term 5 are both zero (dense fallback only)
        const int diff = di - dj;
        // Per-ADO tables (maps, coeffs, levels) at ml.idx(alpha,n,k); per-site
        // coefficients (c_bath) at ml.slot(n,k). Same layout, one definition.
        // This bath owns kc slots (Drude + its poles); baths may differ.
        const int base  = ml.idx(alpha, n, 0);
        const int cbase = ml.slot(n, 0);
        const int kc    = ml.count(n);

        for (int k_idx = 0; k_idx < kc; ++k_idx) {
            const int map_idx = base + k_idx;
            const int a_nk = level_indices[map_idx];

            // ---- Term 4: plus neighbor (needs diff != 0) ----
            if (diff != 0) {
                int alpha_plus = neighbor_map_plus[map_idx];
                if (alpha_plus >= 0 &&
                    (active_flags == nullptr || active_flags[alpha_plus])) {
                    // diff is -1 or +1. The prefactor is pure imaginary.
                    complex_t comm = c_scale(static_cast<real_t>(diff), rho_slot[alpha_plus * Nb + lij]);
                    result = c_sub(result, c_scale_i(coeff_plus[map_idx], comm));
                }
            }

            // ---- Term 5: minus neighbor (needs alpha_nk > 0) ----
            if (a_nk > 0) {
                int alpha_minus = neighbor_map_minus[map_idx];
                if (alpha_minus >= 0 &&
                    (active_flags == nullptr || active_flags[alpha_minus])) {
                    // c_k and cbar are already at hierarchy precision.
                    // factor = c_k*d_i - cbar*d_j, and d is 0 or 1.
                    complex_t factor = c_make(0.0, 0.0);
                    if (di) factor = c_add(factor, c_bath[cbase + k_idx]);
                    if (dj) factor = c_sub(factor, cbar_bath[cbase + k_idx]);
                    complex_t matrix_term = c_mul(factor, rho_slot[alpha_minus * Nb + lij]);
                    // Prefactor i*sqrt(alpha/s) is pure imaginary.
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
            result = c_sub(result, c_scale(static_cast<real_t>(gamma_sink), rho_alpha[lij]));
        }
        else if (i == sink_site || j == sink_site) {
            result = c_sub(result, c_scale(static_cast<real_t>(gamma_sink / 2.0), rho_alpha[lij]));
        }
    }
    
    // kDerivOnly writes the derivative. kStage / kFinal fold the RK4
    // epilogue in place. The FP64 build stays bit-identical.
    const size_t output_idx = boff + static_cast<size_t>(alpha) * Nb + lij;

    if constexpr (MODE == kDerivOnly) {
        drho_out[output_idx] = result;
    } else if constexpr (MODE == kStage) {
        if constexpr (IF) {
            // Lawson IF-RK4 stage (per-ADO E=exp(-½dt·g), E2=exp(-dt·g)):
            //   s1: temp = E·ρ + (E·c)·k,  acc = E2·k
            //   s2: temp = E·ρ + c·k,      acc += w·(E·k)
            //   s3: temp = E2·ρ + (E·c)·k, acc += w·(E·k)
            // E/E2 are complex (g_α complex): every scale is a c_mul.
            const complex_t E  = if_e_half[alpha];
            const complex_t E2 = if_e_full[alpha];
            const complex_t base = c_mul((if_stage == 3) ? E2 : E,
                                         rho_state[output_idx]);
            const complex_t kc = (if_stage == 2)
                ? c_mul(stage_c, result)
                : c_mul(c_mul(E, stage_c), result);
            temp_out[output_idx] = c_add(base, kc);
            acc[output_idx] = (if_stage == 1)
                ? c_mul(E2, result)
                : c_add(c_mul(stage_w, c_mul(E, result)), acc[output_idx]);
        } else {
            temp_out[output_idx] = c_add(c_mul(stage_c, result), rho_state[output_idx]);
            acc[output_idx] = init_acc ? result
                                       : c_add(c_mul(stage_w, result), acc[output_idx]);
        }
    } else {  // kFinal
        complex_t rho_new;
        if constexpr (IF) {
            // ρ(t+dt) = E2·ρ(t) + dt/6·(k1+2k2+2k3+k4), acc already folds the
            // per-stage E factors. The homogeneous ρ term carries E2, so the
            // Kahan carry (which tracks lost bits OF ρ) must be scaled by E2 too
            // — else the compensation drifts against the rescaled ρ.
            const complex_t E2 = if_e_full[alpha];
            const complex_t folded = c_add(c_mul(stage_w, result), acc[output_idx]);
            const complex_t y_raw  = c_mul(dt6, folded);
            const complex_t base   = c_mul(E2, rho_state[output_idx]);
            if (comp == nullptr) {
                rho_new = c_add(y_raw, base);
            } else {
                const complex_t cc = c_mul(E2, comp[output_idx]);
                const complex_t y  = c_sub(y_raw, cc);
                const complex_t t  = c_add(base, y);
                comp[output_idx] = c_sub(c_sub(t, base), y);
                rho_new = t;
            }
        } else {
            const complex_t folded = c_add(c_mul(stage_w, result), acc[output_idx]);
            const complex_t y_raw  = c_mul(dt6, folded);
            if (comp == nullptr) {
                rho_new = c_add(y_raw, rho_state[output_idx]);
            } else {
                // Kahan: y = increment − carry; t = ρ + y; carry = (t−ρ)−y; ρ = t.
                const complex_t r = rho_state[output_idx];
                const complex_t cc = comp[output_idx];
                const complex_t y = c_sub(y_raw, cc);
                const complex_t t = c_add(r, y);
                comp[output_idx] = c_sub(c_sub(t, r), y);
                rho_new = t;
            }
        }
        rho_state[output_idx] = rho_new;
        // Filter on: the next step's filter needs max|ρ_α|² over exactly the
        // ADOs this launch wrote — fold it here from the stored value instead
        // of re-reading ρ in a separate norm kernel (ado_norm.cuh).
        if (norm_sq_bits != nullptr)
            // Per-slot row: [batch×M], same layout as compute_ado_norms / filter.
            heom::fused_norm_accumulate(norm_sq_bits, b * M + alpha, rho_new);
    }
}

// ============================================================================
// Host launcher (see heom_derivatives.cuh for the full contract)
// ============================================================================

namespace {

// Two CUDA numbers, both device properties, queried once.
//   def   = cudaDevAttrMaxSharedMemoryPerBlock
//           the *default* dynamic-smem cap (48 KiB on every recent GPU).
//           A launch above this fails unless the kernel opted in.
//   optin = cudaDevAttrMaxSharedMemoryPerBlockOptin
//           the actual hardware ceiling. Consumer Ampere/Ada/Blackwell
//           cards expose about 99 KiB; datacenter parts go higher. Whole-matrix N=106 staging would be
//           46428 B mixed / 91372 B FP64 (opt-in, 1 block/SM); since
//           only the block's H sub-blocks are staged, but a large diagonal
//           block can still exceed the default, so the opt-in path stays.
// Nothing in the codebase calls cudaSetDevice, so the current device
// cannot change under us.
struct DerivativeSmemCaps {
    size_t def;
    size_t optin;
};

DerivativeSmemCaps current_derivative_smem_caps() {
    static const DerivativeSmemCaps caps = [] {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        int def = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &def, cudaDevAttrMaxSharedMemoryPerBlock, device));
        int optin = 0;
        const cudaError_t err = cudaDeviceGetAttribute(
            &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (err != cudaSuccess || optin <= 0) optin = def;
        return DerivativeSmemCaps{
            static_cast<size_t>(def), static_cast<size_t>(optin)};
    }();
    return caps;
}

// Occupancy is set by the *launch* smem size, not this attribute. We opt
// the kernel in to the hardware ceiling once so any N that passes the
// overflow gate can actually launch.
template <int MODE, bool IF, bool PART, bool SLOTS>
void opt_in_derivative_dynamic_smem(size_t optin_bytes) {
    static const bool once = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(
            compute_heom_derivatives_kernel<MODE, IF, PART, SLOTS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(optin_bytes)));
        return true;
    }();
    (void)once;
}

// One launch path for all three modes: identical domain/smem math, only the
// template parameter and the epilogue pointer pack differ.
template <int MODE, bool IF>
void launch_derivatives(const HEOMContext& ctx, const heom::RK4StageArgs& st)
{
    heom::validate_derivative_context(ctx);
    // Launch domain: compact closure (u_list set) or full M. checked_rho_elements
    // reuses the M·Nb overflow guard with the launch ADO count in place of M.
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
    // Storage and launch domain are the same block, so the index ceiling is
    // checked on exactly what the kernel addresses: launch_ado · Nb.
    const heom::ManifoldBlock& blk = ctx.block;
    const int total_elements = heom::checked_rho_elements(launch_ado, blk.elems());
    const int num_blocks = (total_elements + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;

    // Compact per-slot (u_off): flat grid.x = Σ ceil(|U_b|·Nb/256), grid.y = 1
    // (no empty blocks). Else: grid.y = batch, grid.x covers one slot's domain
    // (B=1 / dense / shared list) — byte-identical to the pre-flat path.
    if (ctx.batch < 1) {
        throw std::invalid_argument("HEOMContext.batch must be >= 1");
    }
    const bool flat = ctx.is_wholestep() && ctx.u_off != nullptr && ctx.block_off != nullptr;
    if (flat && ctx.launch_blocks < 1) {
        throw std::invalid_argument("HEOMContext.launch_blocks must be >= 1 when u_off is set");
    }
    const dim3 grid(
        static_cast<unsigned>(flat ? ctx.launch_blocks : num_blocks),
        static_cast<unsigned>(flat ? 1 : ctx.batch));

    // Dynamic shared memory == staged H sub-blocks sA/sB (real_t) + the
    // matching d_coupling columns (int8), see the kernel prologue; real_t first so the smem
    // base alignment covers it.
    const auto smem_caps = current_derivative_smem_caps();
    const size_t smem_bytes = heom::checked_derivative_smem_bytes(
        blk, ctx.N, ctx.n_bath, smem_caps.optin);
    // Small systems get the PART=false instantiation (whole-matrix staging
    // only, step 8); same rule the kernel and derivative_smem_bytes use.
    // SLOTS=true only when batch > 1 (per-slot pointer math compiled out at B=1).
    const bool part = !heom::derivative_stage_whole(ctx.N, ctx.n_bath);
    const bool slots = ctx.batch > 1;
    if (smem_bytes > smem_caps.def) {
        if (part && slots)      opt_in_derivative_dynamic_smem<MODE, IF, true,  true >(smem_caps.optin);
        else if (part)          opt_in_derivative_dynamic_smem<MODE, IF, true,  false>(smem_caps.optin);
        else if (slots)         opt_in_derivative_dynamic_smem<MODE, IF, false, true >(smem_caps.optin);
        else                    opt_in_derivative_dynamic_smem<MODE, IF, false, false>(smem_caps.optin);
    }

    auto* kern = (part && slots) ? compute_heom_derivatives_kernel<MODE, IF, true,  true >
               : part            ? compute_heom_derivatives_kernel<MODE, IF, true,  false>
               : slots           ? compute_heom_derivatives_kernel<MODE, IF, false, true >
                                 : compute_heom_derivatives_kernel<MODE, IF, false, false>;
    kern<<<grid, HEOM_BLOCK_SIZE, smem_bytes>>>(
        ctx.rho_in, ctx.drho_out, ctx.hamiltonian,
        ctx.neighbor_map_plus, ctx.neighbor_map_minus,
        ctx.c_bath, ctx.cbar_bath, ctx.coeff_plus, ctx.coeff_minus, ctx.markov_residue,
        ctx.level_indices, ctx.mode_off, ctx.total_decay, ctx.active_flags,
        ctx.d_coupling,
        ctx.u_list, ctx.is_wholestep() ? ctx.u_off : nullptr,
        ctx.is_wholestep() ? ctx.block_off : nullptr,
        ctx.is_wholestep() ? nullptr : ctx.u_flags, launch_ado,
        ctx.batch,
        ctx.M, ctx.N, ctx.D, ctx.n_bath,
        blk.row_off, blk.n_row, blk.col_off, blk.n_col,
        ctx.sink_enabled, ctx.sink_site, ctx.gamma_sink,
        st.rho, st.temp_out, st.acc, st.comp,
        st.c, st.w, st.dt6, st.init_acc,
        st.e_half, st.e_full, st.if_stage,
        MODE == kFinal ? st.norm_sq_bits : nullptr
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
// Integrating-factor precompute: per-ADO E=exp(-½·dt·g_α), E2=exp(-dt·g_α),
// g_α complex. The device has no exp(cuDoubleComplex); write it out:
//   e^{-x g} = e^{-x Re g} (cos(x Im g), −sin(x Im g)).
// Evaluated in FP64 (precision-critical, shared across the whole run) and
// stored at hier precision, so the O(M·N²) threads that read them every step
// pay no FP64-pipe cost. O(M) work per dt.
// ============================================================================
__device__ __forceinline__ complex_t exp_neg_scaled(double x, cuDoubleComplex g)
{
    const double mag = exp(-x * cuCreal(g));
    const double ph  = x * cuCimag(g);
    return c_make(static_cast<real_t>(mag * cos(ph)),
                  static_cast<real_t>(-mag * sin(ph)));
}

__global__ void compute_if_factors_kernel(
    const cuDoubleComplex* __restrict__ total_decay, double dt_au, int M,
    complex_t* __restrict__ e_half, complex_t* __restrict__ e_full)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= M) return;
    const cuDoubleComplex g = total_decay[a];
    e_half[a] = exp_neg_scaled(0.5 * dt_au, g);
    e_full[a] = exp_neg_scaled(dt_au, g);
}

void compute_if_factors(const cuDoubleComplex* d_total_decay, double dt_au, int M,
                        complex_t* d_e_half, complex_t* d_e_full)
{
    if (M <= 0) return;
    const int block = 256;
    const int grid  = (M + block - 1) / block;
    compute_if_factors_kernel<<<grid, block>>>(
        d_total_decay, dt_au, M, d_e_half, d_e_full);
    CUDA_CHECK_KERNEL();
}
