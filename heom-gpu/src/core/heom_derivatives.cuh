#ifndef HEOM_DERIVATIVES_CUH
#define HEOM_DERIVATIVES_CUH

#include <cuComplex.h>
#include "core/precision.hpp"
#include "core/cuda_common.cuh"       // HEOM_BLOCK_SIZE (checked_rho_elements)
#include "core/manifold_layout.hpp"   // ManifoldBlock (HEOMContext::block)
#include <stdint.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

/**
 * @brief Bundle of all device pointers + scalars consumed by one HEOM
 *        derivative kernel launch.
 *
 * The struct is a value type — passed by const-reference into
 * compute_heom_derivatives() and copied wholesale by RK4 between stages.
 * It owns nothing; all pointers reference GPU buffers managed by
 * SystemContext (read-only) or HEOMTrajectory (rho_in / drho_out / filter
 * buffers). Lifetime of every pointer must outlive the kernel launch.
 *
 * Filtering contract:
 *   active_flags == nullptr  -> all M ADOs evaluated (no filtering pass)
 *   active_flags != nullptr  -> Terms 1-3 skip inactive ADOs; Term 4/5
 *                               also skip neighbours whose flag is 0,
 *                               and the alpha == 0 entry is held at 1
 *                               for life by HEOMTrajectory's constructor
 *                               (see filter_ados_kernel for why).
 *
 * total_decay is required and always points to SystemContext::d_total_decay
 * (precomputed once). The kernel dereferences it unconditionally.
 */
struct HEOMContext {
    // State vectors. drho_out is only consumed by the standalone
    // compute_heom_derivatives() (tests / diagnostics). rk4_step uses the
    // FUSED stage/final entry points, which never write a k buffer — leave
    // drho_out null there. hier precision (FP32 in mixed build).
    const heom::complex_t* rho_in;       // [B][M][Nb], Nb = block.elems()
    heom::complex_t* drho_out = nullptr; // [B][M][Nb]

    // System parameters (immutable). Stored complex_t at hier precision for
    // layout reasons, but the imaginary part is required to be zero
    // (SystemContext throws otherwise): the kernel stages only the real part
    // into shared memory. See sH in heom_derivatives.cu.
    const heom::complex_t* hamiltonian;  // [N*N], Im == 0
    // PER-SITE baths. c_bath is indexed by slot = mode_off[n]+k (bath n, pole
    // k), length D. Each physical bath n carries its own correlation function
    // C_n(t) = Σ_k c_{nk} e^{-ν_{nk} t}. Term 4/5 read c_bath[mode_off[n]+k];
    // Term 3 uses the per-site markov_residue[n] (homogeneous baths are just
    // the degenerate case "all rows equal").
    const heom::complex_t* c_bath;       // [D] c_{nk}
    // Right-action coefficient for Term 5's ρ·Q side: C*(t) = Σ c_k* e^{-ν_k* t},
    // so the ladder decaying with ν_k carries c_j* for the slot j with ν_j = ν_k*.
    // Real pole ⇒ c̄ = c* (what the kernel used to conj on the fly); conjugate
    // pair ⇒ the partner's c*. Precomputed host-side (physics_utils::set_cbar),
    // never derived in the kernel — the pairing is not a per-slot operation.
    const heom::complex_t* cbar_bath;    // [D] c̄_{nk}
    // ν_{nk} is NOT carried here: the kernel never reads bath frequencies. Term 2
    // (level decay) consumes the precomputed Σ α_{nk}ν_{nk} via total_decay below;
    // ν_{nk} itself is only needed host-side (the total_decay precompute in
    // SystemContext, and HDF5 metadata). It has no GPU mirror at all — don't
    // re-add one to feed the kernel.
    // Term 4/5 magnitudes, precomputed PER-ADO in SystemContext (same
    // [M*D] layout as level_indices, indexed by map_idx):
    //   coeff_plus[α,n,k]  = √[(α_{nk}+1)·s_{nk}]      (Term 4)
    //   coeff_minus[α,n,k] = √[α_{nk}/s_{nk}]  (0 if α_{nk}=0)  (Term 5)
    // with the Shi scale s_{nk} = √(|c_{nk}||c̄_{nk}|) (= |c_{nk}| for a real
    // pole; see SystemContext). These are (i,j)-independent, so precomputing
    // once turns the former N²-redundant sqrt() in the hot loop into a single
    // broadcast load. s_{nk} itself is not needed on the GPU.
    // Hier precision, like c_bath above: Term 3/4/5 read these per (thread,n,k),
    // and on the consumer cards the mixed build targets, FP64 runs at 1/64 rate,
    // so a per-read downcast would make these the kernel's limiting pipe.
    const heom::real_t* coeff_plus;      // [M*D] √[(α+1)s]  (Term 4)
    const heom::real_t* coeff_minus;     // [M*D] √[α/s]     (Term 5)
    const heom::real_t* markov_residue;  // [n_bath] per-site Markov term Σ_{k>K} c_{nk}/ν_{nk}

    // Hierarchy structure (indexed by n_bath, not N)
    const int* neighbor_map_plus;        // [M * D]
    const int* neighbor_map_minus;       // [M * D]
    const int8_t* level_indices;         // [M * D]

    // Ladder layout: bath n owns slots [mode_off[n], mode_off[n+1]) of every
    // per-ADO row; mode_off[n_bath] == D. Device pointer, [n_bath+1] ints,
    // owned by SystemContext. Read as a block-uniform broadcast in the Term
    // 4/5 bath loop. See core/mode_layout.hpp.
    const int* mode_off;                 // [n_bath+1]

    // Pre-computed per-ADO data. Complex: Σ α_{nk}ν_{nk} with ν± = γ ± iω₁
    // for a vibrational pair. FP64 (exact), downcast once per thread in Term 2.
    const cuDoubleComplex* total_decay;  // [M], required, never nullptr

    // Filtering. One row per slot: slot b reads active_flags[b·M ..] (per-slot
    // active sets; batch == 1 ⇒ the single [M] row).
    const int8_t* active_flags;          // [batch][M] (nullptr => no filtering)

    // U-list launch compaction. When u_list != nullptr the kernel launches
    // u_count·Nb threads and maps slot s → ADO α = u_list[s], instead of the
    // full M·Nb grid with α = slot. u_list MUST be the one-hop closure
    // E = A ∪ neighbors(A) so every α with a nonzero derivative is covered;
    // ADOs outside u_list are NOT written (the whole-step path freezes them).
    //   u_list == nullptr  -> full M·Nb launch, α = slot
    //   u_list != nullptr  -> u_count·Nb launch, α = u_list[slot]
    // u_list != nullptr selects the whole-step path: rk4_step shrinks every
    // ∝M operation (derivative, stage-epilogue and final-combine launches)
    // to the U-list and freezes α∉U by never indexing it. Frozen is a filter
    // choice, not an identity: their derivative is exactly zero, but IF-RK4
    // would still decay them by E2. The full launch freezes the same set via
    // u_flags. Both paths combine the final RK4 result in-place into rho_in's
    // storage — there is no ping-pong buffer or swap anywhere.
    const int* u_list = nullptr;         // [u_count] compact ADO indices, or nullptr
    int        u_count = 0;              // valid iff u_list != nullptr
    // Per-slot compact lists: u_off != nullptr ⇒ flat launch over concatenated
    // segments (grid.x = launch_blocks, grid.y = 1); u_count stays max_b |U_b|
    // for the filter scan domain. nullptr ⇒ every slot shares u_list[0..u_count)
    // and grid.y = batch (legacy / dense / B=1).
    const int* u_off = nullptr;          // [batch+1] device, or nullptr
    const int* block_off = nullptr;      // [batch+1] flat block prefixes, or nullptr
    int        launch_blocks = 0;        // Σ_b ceil(|U_b|·Nb/256); used iff u_off

    // U membership for a FULL launch: [M], nonzero ⇔ α ∈ U, or nullptr when
    // every α is in U (filter off, all active). The stage and final epilogues
    // return for α∉U, so the full launch freezes exactly the ADOs the compact
    // one never indexes — one rule ("α∉U is frozen") on both launch domains,
    // which is what makes them bit-identical under IF-RK4 (whose epilogue
    // would scale α∉U by E2). Unused by the compact launch (its set is U);
    // ignored by the standalone derivative (kDerivOnly writes every element).
    // Per slot: row b (u_flags[b·M ..]) gates slot b.
    const uint8_t* u_flags = nullptr;

    // SINGLE SOURCE OF TRUTH for the launch-domain selection (|U| vs M).
    // Every decision site reads this accessor instead of testing the raw
    // pointer, so the pointer↔path mapping lives in exactly one place.
    bool is_wholestep() const { return u_list != nullptr; }

    // Bath coupling: d^(n)_i = 1 iff state i has site n excited.
    // HARD ASSUMPTION — d is BINARY {0,1}. Two kernel terms encode this
    // differently: Term 3 uses (d_i != d_j) as a stand-in for the double
    // commutator weight (d_i - d_j)² (valid only because x²==x for x∈{0,1}),
    // while Term 5 uses d_i / d_j as multiplicative factors. If d ever takes a
    // value ≥2 (e.g. a non-indicator coupling), Term 5 stays correct but Term 3
    // silently computes the wrong weight. Keep d_coupling an indicator.
    const int8_t* d_coupling;            // [n_bath * N]

    // Dimensions
    int M;        // ADO count
    int N;        // Hilbert dimension
    // Total ladder count N_ell = Σ_n P_n: the width of every per-ADO row in
    // the neighbor/level/coeff maps and the length of c_bath. Slot k=0 of a
    // bath is its Drude pole, the rest its extra Matsubara/Padé poles (and,
    // plus vibrational-mode poles). How many a given bath has is in
    // mode_off, not here.
    int D;
    int n_bath;   // physical bath count (== n_states; != N for non-EXCITED_ONLY bases)

    // Lindblad sink (Term 6)
    bool   sink_enabled;
    int    sink_site;
    double gamma_sink;

    // Batch width. B independent ρ states share ONE launch via grid.y:
    // buffer layout is [B][M][Nb], slot b = blockIdx.y owns block boff = b·M·Nb.
    // batch == 1 ⇒ grid.y == 1, boff ≡ 0, kernel body is byte-identical to the
    // single-slot path (no extra math on the hot index). All B slots share the
    // same read-only system tables; flags / U are per slot (see above).
    int batch = 1;

    // Manifold block: the STORAGE SHAPE of every ADO in rho_in / drho_out, and
    // therefore the launch domain. ρ in a 2DES segment lives entirely inside
    // ONE (row manifold, column manifold) block — after the second pulse of an
    // ESA pathway it is (f,e), 4 of 29² elements at FMO scale — and the rest
    // is STRUCTURALLY zero, so it is not stored at all: element (i,j) of
    // ADO α sits at α·Nb + (i−row_off)·n_col + (j−col_off), Nb = n_row·n_col.
    // Dynamics declares full(N) and stores the whole matrix.
    // History: ESA manifold blocking (launch-domain
    // blocking), 2026-09-08-lhcii-s5-compaction (storage).
    //
    //   row_off/n_row : rows [row_off, row_off + n_row)
    //   col_off/n_col : cols [col_off, col_off + n_col)
    //
    // NO SENTINEL. "Whole matrix" is spelled ManifoldBlock::full(N), which
    // every construction site must fill explicitly; validate_derivative_context
    // rejects anything else, including the empty block a brace-initialised
    // context leaves behind. A sentinel would buy one line at each construction
    // site and cost a silent full-N² launch (over a buffer that may hold far
    // fewer elements) at every site that forgot.
    //
    // The shape rule (non-empty, in range, identical-or-disjoint) lives in
    // ManifoldBlock::validate — see core/manifold_layout.hpp.
    heom::ManifoldBlock block;
};

namespace heom {

// HARD SCALABILITY CEILING — read this before quoting any "supported" size.
//
// Every kernel in this project addresses ρ through a 32-bit linear index
// (alpha*Nb + local (r,c); norm: rho_in[alpha*Nb]; axpy: u_list[..]*Nb+..)
// and sizes its launch grid in int, where Nb = block.elems() is the number
// of stored elements per ADO (ρ is stored compacted to its manifold
// block, so Nb ≤ N² and for spectra usually far smaller). The largest problem
// that can run end to end is therefore
//        M * Nb ≤ INT_MAX − (HEOM_BLOCK_SIZE − 1) ≈ 2.1e9 elements.
// This is NOT merely a launch-time guard: exceeding it wraps a signed index to
// a negative offset and corrupts memory. We throw here instead. The block
// slack is what keeps the rounded-up grid (total + block − 1) / block and the
// last block's thread index blockIdx.x·blockDim.x + tid inside int; with only
// total ≤ INT_MAX, both wrap negative and slip past the `idx >= total` guard.
//
// Where the shipping configs sit: LHC-II DOUBLE_EXCITED (N=106) 2DES at L=3
// D=70 stores at most the (f,e) block, Nb = 91·14 = 1274, so M·Nb ≈ 7.9e7;
// L=4 D=56 is ≈ 6.2e8. Both are under the ceiling. A dynamics run stores the
// full N², and there M·N² can still blow the index range at LHC-II M — that
// is the S6 (int64 index) problem, not this one.
inline int checked_rho_elements(int M, int Nb) {
    if (M <= 0 || Nb <= 0) {
        throw std::invalid_argument("HEOM launch dimensions must be positive");
    }

    const size_t total = static_cast<size_t>(M) * static_cast<size_t>(Nb);
    const size_t ceiling =
        static_cast<size_t>(std::numeric_limits<int>::max()) - (HEOM_BLOCK_SIZE - 1);

    if (total > ceiling) {
        throw std::overflow_error("HEOM launch M*Nb exceeds int kernel index range");
    }
    return static_cast<int>(total);
}

// The SECOND int ceiling, independent of M·N². The per-ADO tables
// (neighbor_map_±, coeff_±, level_indices) are addressed as
// alpha*D + mode_off[n] + k in int (derivative kernel Term 4/5, filter
// stats), so M·D < 2^31 too. It is not implied by the first: N is the
// Hilbert dimension while D is the ladder count, and with per-pigment modes
// the second can be the one that bites first.
inline int checked_map_elements(int M, int D) {
    if (M <= 0 || D <= 0) {
        throw std::invalid_argument("HEOM mode-table dimensions must be positive");
    }
    const size_t total = static_cast<size_t>(M) * static_cast<size_t>(D);
    const size_t int_max =
        static_cast<size_t>(std::numeric_limits<int>::max());
    if (total > int_max) {
        throw std::overflow_error("HEOM M*D exceeds int kernel index range");
    }
    return static_cast<int>(total);
}

inline size_t derivative_smem_bytes(int N, int n_bath) {
    if (N <= 0 || n_bath <= 0) {
        throw std::invalid_argument("HEOM shared-memory dimensions must be positive");
    }
    // sH caches the REAL part of the Hamiltonian at hier precision, so the
    // footprint is N²·sizeof(real_t): 46428 B at N=106 mixed (under the
    // 48 KiB default), 91372 B in FP64 (needs the opt-in path).
    return static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(heom::real_t)
         + static_cast<size_t>(n_bath) * static_cast<size_t>(N) * sizeof(int8_t);
}

// lhc2-smem step 6: when the WHOLE H (real_t) + d_coupling (int8) is at
// most this many bytes, every block stages all of it with the original
// contiguous loops instead of the per-block sub-blocks. Then smem never
// limits occupancy (6 blocks × 12 KiB < 100 KiB/SM), and small systems (FMO,
// N=29 ESA: 3567 B mixed / 6931 B FP64) skip the index math of partial
// staging, which ncu showed doubling FMO K1L3's smem_setup samples (7.6% →
// 16% of the kernel). LHC-II (N=106: 46428 / 91372 B) stays partial.
// Kernel and launcher both use derivative_stage_whole(), so they agree.
constexpr int kDerivWholeStageMaxBytes = 12288;
__host__ __device__ inline bool derivative_stage_whole(int N, int n_bath) {
    return N * N * static_cast<int>(sizeof(heom::real_t)) + n_bath * N
           <= kDerivWholeStageMaxBytes;
}

// Rows i one block of `block_threads` consecutive elements can touch in an
// n_row×n_col manifold block: the worst start is lj0 = n_col−1. Capped at
// n_row (a block straddling ADOs revisits rows cyclically).
inline int derivative_smem_rows(int n_row, int n_col, int block_threads) {
    const int rows = (n_col - 1 + block_threads - 1) / n_col + 1;
    return rows < n_row ? rows : n_row;
}

// Exact dynamic smem of the derivative kernel for one manifold block (see the
// kernel prologue): diagonal block → n²·real_t + n_bath·n int8; disjoint
// block → (R·n_row + n_col²)·real_t + n_bath·(R + n_col) int8. Never larger
// than the whole-matrix N²·real_t + n_bath·N, which derivative_smem_bytes(N,
// n_bath) still returns (the whole matrix IS a diagonal block).
inline size_t derivative_smem_bytes(const ManifoldBlock& blk, int N, int n_bath,
                                    int block_threads = HEOM_BLOCK_SIZE) {
    if (N <= 0 || n_bath <= 0 || blk.n_row <= 0 || blk.n_col <= 0 || block_threads <= 0) {
        throw std::invalid_argument("HEOM shared-memory dimensions must be positive");
    }
    if (derivative_stage_whole(N, n_bath)) return derivative_smem_bytes(N, n_bath);
    const bool diag = (blk.row_off == blk.col_off && blk.n_row == blk.n_col);
    const size_t a = diag ? 0
        : static_cast<size_t>(derivative_smem_rows(blk.n_row, blk.n_col, block_threads))
          * static_cast<size_t>(blk.n_row);
    const size_t b = static_cast<size_t>(blk.n_col) * static_cast<size_t>(blk.n_col);
    const size_t d_states = diag ? static_cast<size_t>(blk.n_row)
        : static_cast<size_t>(derivative_smem_rows(blk.n_row, blk.n_col, block_threads))
          + static_cast<size_t>(blk.n_col);
    return (a + b) * sizeof(heom::real_t)
         + static_cast<size_t>(n_bath) * d_states * sizeof(int8_t);
}

inline size_t checked_derivative_smem_bytes(
    const ManifoldBlock& blk,
    int N,
    int n_bath,
    size_t max_smem_bytes
) {
    if (max_smem_bytes == 0) {
        throw std::invalid_argument("CUDA shared-memory limit must be nonzero");
    }
    const size_t bytes = derivative_smem_bytes(blk, N, n_bath);
    if (bytes > max_smem_bytes) {
        throw std::overflow_error(
            "HEOM derivative dynamic shared memory exceeds device limit ("
            + std::to_string(bytes) + " B needed, "
            + std::to_string(max_smem_bytes) + " B max)");
    }
    return bytes;
}

// Whole-matrix form (the historical N²-staging footprint, an upper bound).
inline size_t checked_derivative_smem_bytes(
    int N,
    int n_bath,
    size_t max_smem_bytes
) {
    if (max_smem_bytes == 0) {
        throw std::invalid_argument("CUDA shared-memory limit must be nonzero");
    }
    const size_t bytes = derivative_smem_bytes(N, n_bath);
    if (bytes > max_smem_bytes) {
        throw std::overflow_error(
            "HEOM derivative dynamic shared memory exceeds device limit ("
            + std::to_string(bytes) + " B needed, "
            + std::to_string(max_smem_bytes) + " B max)");
    }
    return bytes;
}

inline void validate_derivative_context(const HEOMContext& ctx) {
    // Block first: elems() of an invalid block is meaningless, and the shape
    // rule's message names the offending field.
    ctx.block.validate(ctx.N, "HEOMContext");
    (void)checked_rho_elements(ctx.M, ctx.block.elems());
    (void)checked_map_elements(ctx.M, ctx.D);
    (void)derivative_smem_bytes(ctx.block, ctx.N, ctx.n_bath);
    if (!ctx.total_decay) {
        throw std::invalid_argument("HEOMContext.total_decay must not be nullptr");
    }
    if (!ctx.mode_off) {
        throw std::invalid_argument("HEOMContext.mode_off must not be nullptr");
    }
    if (!ctx.cbar_bath) {
        throw std::invalid_argument("HEOMContext.cbar_bath must not be nullptr");
    }
    if (!ctx.coeff_plus) {
        throw std::invalid_argument("HEOMContext.coeff_plus must not be nullptr");
    }
    if (!ctx.coeff_minus) {
        throw std::invalid_argument("HEOMContext.coeff_minus must not be nullptr");
    }
    if (!ctx.markov_residue) {
        throw std::invalid_argument("HEOMContext.markov_residue must not be nullptr");
    }
}

}  // namespace heom

namespace heom {

/**
 * @brief Pointer/scalar pack for the RK4 epilogue fused into the derivative
 *        kernel (see RK4Fuse in heom_derivatives.cu).
 *
 * The derivative value is consumed straight from the kernel's register —
 * no k-buffer store/reload, no separate epilogue launch. All buffers are
 * indexed at the thread's own element only; rho is the lone cross-element
 * READ via ctx.rho_in, so the caller must never pass the current rho_in
 * as temp_out (rk4_step ping-pongs its two temp buffers to guarantee it).
 */
struct RK4StageArgs {
    complex_t* rho      = nullptr;  // stage: read ρ; final: in-place update
    complex_t* temp_out = nullptr;  // stage: next stage's f() input
    complex_t* acc      = nullptr;  // running RK4 sum
    complex_t* comp     = nullptr;  // final: Kahan carry or nullptr
    complex_t  c        = {};       // stage: dt/2 or dt
    complex_t  w        = {};       // fold weight (1 or 2)
    complex_t  dt6      = {};       // final: dt/6
    bool       init_acc = false;    // stage 1: acc = k

    // Integrating-factor RK4 (IF path only; ignored by the plain kStage/kFinal).
    // Per-ADO COMPLEX factors E=exp(-½·dt·g_α), E2=exp(-dt·g_α), g_α =
    // total_decay[α] (complex once a bath has a vibrational pair). The exp() is
    // evaluated in FP64 from the exact g_α and stored rounded, so the epilogue
    // reads them at hier precision (same rule as c_bath).
    const complex_t* e_half   = nullptr;  // [M] exp(-½·dt·g_α)
    const complex_t* e_full   = nullptr;  // [M] exp(-dt·g_α)
    int           if_stage = 0;        // IF kStage: 1/2/3 (see rk4_step_if); else unused

    // Final stage only: fused ADO max-norm for the next filter pass.
    // [batch×M] IEEE bits of max|ρ_new|² per slot (idx = b·M + α); the caller
    // zeroes the buffer before the step. nullptr = no norm work (filter off,
    // or a fill-window step).
    unsigned long long* norm_sq_bits = nullptr;
};

}  // namespace heom

/**
 * @brief Launch the HEOM derivative kernel (standalone).
 *
 * Computes drho/dt for every (alpha, i, j) into ctx.drho_out, including
 * Terms 1-6 (commutator, decay, Markov, plus-neighbour, minus-neighbour,
 * sink). Block/grid sizes and dynamic shared-memory bytes are derived
 * from ctx.M / ctx.N / ctx.n_bath; the caller only needs to populate ctx.
 */
void compute_heom_derivatives(const HEOMContext& ctx);

/**
 * @brief Derivative + fused RK4 stage epilogue (stages 1-3):
 *        temp_out = st.c·k + rho,  acc = st.init_acc ? k : st.w·k + acc.
 *        ctx.drho_out is unused (leave null); k never touches memory.
 */
void compute_heom_derivatives_stage(const HEOMContext& ctx,
                                    const heom::RK4StageArgs& st);

/**
 * @brief Derivative + fused RK4 final combine (stage 4), in place:
 *        rho += st.dt6·(st.w·k + acc), optional Kahan carry in st.comp.
 */
void compute_heom_derivatives_final(const HEOMContext& ctx,
                                    const heom::RK4StageArgs& st);

/**
 * @brief Integrating-factor RK4 stage/final (IF=true instantiation).
 *
 * Same contract as the plain stage/final above, but Term 2 (hierarchy decay)
 * is EXCLUDED from the derivative and instead applied exactly via the per-ADO
 * factors st.e_half (E=exp(-½·dt·g_α)) / st.e_full (E2=exp(-dt·g_α)). The
 * stage entry additionally reads st.if_stage ∈ {1,2,3} to select the Lawson
 * update. Callers MUST populate e_half/e_full/if_stage; see rk4_step_if.
 */
void compute_heom_derivatives_stage_if(const HEOMContext& ctx,
                                       const heom::RK4StageArgs& st);

void compute_heom_derivatives_final_if(const HEOMContext& ctx,
                                       const heom::RK4StageArgs& st);

/**
 * @brief Precompute the per-ADO integrating-factor tables ([M]):
 *        e_half[α] = exp(-½·dt_au·g_α), e_full[α] = exp(-dt_au·g_α),
 *        with g_α = total_decay[α] (complex). Cheap (O(M) exp); recomputed
 *        only when dt changes (see rk4_step_if / HEOMTrajectory::run_step).
 *        The exp is evaluated in FP64 as e^{-x Re g}(cos(x Im g), −sin(x Im g))
 *        and the result stored at hier precision, so the per-thread epilogue
 *        read costs no FP64-pipe traffic.
 * @param d_total_decay device [M] Σ α_{nk}ν_k per ADO (SystemContext, FP64 complex).
 * @param dt_au         internal timestep in a.u.
 * @param M             number of ADOs.
 * @param d_e_half      device out [M].
 * @param d_e_full      device out [M].
 */
void compute_if_factors(const cuDoubleComplex* d_total_decay, double dt_au, int M,
                        heom::complex_t* d_e_half, heom::complex_t* d_e_full);

#endif // HEOM_DERIVATIVES_CUH

