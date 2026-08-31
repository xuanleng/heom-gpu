#ifndef HEOM_DERIVATIVES_CUH
#define HEOM_DERIVATIVES_CUH

#include <cuComplex.h>
#include "core/precision.hpp"
#include <stdint.h>
#include <cstddef>
#include <limits>
#include <stdexcept>

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
    const heom::complex_t* rho_in;       // [M*N*N]
    heom::complex_t* drho_out = nullptr; // [M*N*N]

    // System parameters (immutable). Hamiltonian stays FP64 storage (tiny N²,
    // the kernel narrows it into the FP32 shared cache.
    const heom::complex_t* hamiltonian;  // [N*N]
    // PER-SITE baths. c_bath/abs_c_bath are indexed by (bath n, pole k) in the
    // flat row-major layout n*(K+1)+k, length n_bath*(K+1). Each physical bath n
    // carries its own correlation function C_n(t) = Σ_k c_{nk} e^{-ν_{nk} t}.
    // Term 4/5 read c_bath[n*(K+1)+k]; Term 3 uses the per-site markov_residue[n]
    // (homogeneous baths are just the degenerate case "all rows equal").
    const heom::complex_t* c_bath;       // [n_bath*(K+1)] c_{nk}
    // ν_{nk} is NOT carried here: the kernel never reads bath frequencies. Term 2
    // (level decay) consumes the precomputed Σ α_{nk}ν_{nk} via total_decay below;
    // ν_{nk} itself is only needed host-side (the total_decay precompute in
    // SystemContext, and HDF5 metadata). It has no GPU mirror at all — don't
    // re-add one to feed the kernel.
    // Term 4/5 magnitudes, precomputed PER-ADO in SystemContext (same
    // [M*n_bath*(K+1)] layout as level_indices, indexed by map_idx):
    //   coeff_plus[α,n,k]  = √[(α_{nk}+1)·|c_{nk}|]      (Term 4)
    //   coeff_minus[α,n,k] = √[α_{nk}/|c_{nk}|]  (0 if α_{nk}=0)  (Term 5)
    // These are (i,j)-independent, so precomputing once turns the former
    // N²-redundant sqrt() in the hot loop into a single broadcast load. |c_{nk}|
    // itself is no longer needed on the GPU (it only ever fed these two sqrt).
    // Hier precision, like c_bath above: Term 3/4/5 read these per (thread,n,k),
    // and on the consumer cards the mixed build targets, FP64 runs at 1/64 rate,
    // so a per-read downcast would make these the kernel's limiting pipe.
    const heom::real_t* coeff_plus;      // [M*n_bath*(K+1)] √[(α+1)|c|]  (Term 4)
    const heom::real_t* coeff_minus;     // [M*n_bath*(K+1)] √[α/|c|]     (Term 5)
    const heom::real_t* markov_residue;  // [n_bath] per-site Markov term Σ_{k>K} c_{nk}/ν_{nk}

    // Hierarchy structure (indexed by n_bath, not N)
    const int* neighbor_map_plus;        // [M * n_bath * (K+1)]
    const int* neighbor_map_minus;       // [M * n_bath * (K+1)]
    const int8_t* level_indices;         // [M * n_bath * (K+1)]

    // Pre-computed per-ADO data
    const double* total_decay;           // [M], required, never nullptr

    // Filtering
    const int8_t* active_flags;          // [M] (nullptr => no filtering)

    // U-list launch compaction. When u_list != nullptr the kernel launches
    // u_count·N² threads and maps slot s → ADO α = u_list[s], instead of the
    // full M·N² grid with α = slot. u_list MUST be the one-hop closure
    // E = A ∪ neighbors(A) so every α with a nonzero derivative is covered;
    // ADOs outside u_list are NOT written (the whole-step path freezes them).
    //   u_list == nullptr  -> full M·N² launch, α = slot
    //   u_list != nullptr  -> u_count·N² launch, α = u_list[slot]
    // u_list != nullptr selects the whole-step path: rk4_step shrinks every
    // ∝M operation (derivative, stage-epilogue and final-combine launches)
    // to the U-list and
    // freezes α∉U (their ρ is provably unchanged this interval). Both paths
    // combine the final RK4 result in-place into rho_in's storage — there is
    // no ping-pong buffer or swap anywhere; the launch domain is the ONLY
    // difference. The derivative launcher reads only u_list/u_count.
    const int* u_list = nullptr;         // [u_count] compact ADO indices, or nullptr
    int        u_count = 0;              // valid iff u_list != nullptr

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
    // Last bath-pole index: K == n_bath_modes_total == n_extra_poles. Every
    // per-pole array (c_bath/nu_bath) and the per-(n,k) stride in the
    // neighbor/level maps is sized K+1 with this same K: index k=0 is the
    // Drude pole, k=1..K the extra Matsubara/Padé poles.
    int K;
    int n_bath;   // physical bath count (== n_states; != N for non-EXCITED_ONLY bases)

    // Lindblad sink (Term 6)
    bool   sink_enabled;
    int    sink_site;
    double gamma_sink;

    // Batch width. B independent ρ states share ONE launch via grid.y:
    // buffer layout is [B][M][N²], slot b = blockIdx.y owns block boff = b·M·N².
    // batch == 1 ⇒ grid.y == 1, boff ≡ 0, kernel body is byte-identical to the
    // pre-batch path (no extra math on the hot index). All B slots share the
    // same read-only system tables, u_list and active_flags (union closure).
    int batch = 1;

    // Manifold block: the sub-rectangle of the N×N matrix the launch covers.
    // ρ in a 2DES segment lives entirely inside ONE (row manifold, column
    // manifold) block — after the second pulse of an ESA pathway it is (f,e),
    // 4 of 29² elements at FMO scale — and the rest is STRUCTURALLY zero, not
    // merely small. Shrinking the launch domain to the block is what makes
    // that structure pay.
    //
    //   row_off/n_row : rows [row_off, row_off + n_row)
    //   col_off/n_col : cols [col_off, col_off + n_col)
    //
    // NO SENTINEL. "Whole matrix" is spelled (0, N, 0, N), which every
    // construction site must fill explicitly; validate_derivative_context
    // rejects anything else, including the zeros a brace-initialised context
    // would leave behind. A sentinel would buy one line at each construction
    // site and cost a silent full-N² launch at every site that forgot.
    //
    // ADDRESSING IS UNCHANGED: elements are still at i*N + j. Only which (i,j)
    // get a thread changes. Nothing is compacted or restrided.
    //
    // The two ranges must be either IDENTICAL or DISJOINT — a partial overlap
    // has no physical meaning (manifolds partition the basis) and would break
    // the Term 1 dispatch in heom_derivatives.cu, which relies on the k-sum
    // splitting into at most two contiguous runs.
    int row_off = 0;
    int n_row   = 0;
    int col_off = 0;
    int n_col   = 0;
};

namespace heom {

// HARD SCALABILITY CEILING — read this before quoting any "supported" size.
//
// Every kernel in this project addresses ρ through a 32-bit linear index
// (derivative: alpha*N²+ij; norm: rho_in[alpha*N*N]; axpy: u_list[..]*N²+..)
// and sizes its launch grid in int. The largest problem that can run end to
// end is therefore
//        M * N² < 2^31 ≈ 2.1e9 elements.
// This is NOT merely a launch-time guard: exceeding it wraps a signed index to
// a negative offset and corrupts memory. We throw here instead.
//
// Concretely this rules out the big light-harvesting configs that some
// comments elsewhere use as motivating examples (e.g. LHC-II DOUBLE_EXCITED,
// N=106 with M in the millions ⇒ M·N² ~ 1e10). Those are NOT runnable on the
// current kernels; making them run requires migrating the linear index AND the
// grid math to int64/size_t. The host-side hierarchy build and memory-budget
// estimate use size_t and can represent larger tables, but the GPU propagation
// stops at this ceiling. Do not advertise a size past it as "shipping".
inline int checked_total_elements(int M, int N) {
    if (M <= 0 || N <= 0) {
        throw std::invalid_argument("HEOM launch dimensions must be positive");
    }

    const size_t n2 = static_cast<size_t>(N) * static_cast<size_t>(N);
    const size_t total = static_cast<size_t>(M) * n2;
    const size_t int_max =
        static_cast<size_t>(std::numeric_limits<int>::max());

    if (n2 > int_max || total > int_max) {
        throw std::overflow_error("HEOM launch M*N*N exceeds int kernel index range");
    }
    return static_cast<int>(total);
}

inline size_t derivative_smem_bytes(int N, int n_bath) {
    if (N <= 0 || n_bath <= 0) {
        throw std::invalid_argument("HEOM shared-memory dimensions must be positive");
    }
    // sH caches the Hamiltonian in HIER precision (downcast on load), so the
    // shared-memory footprint follows complex_t, not the FP64 global storage.
    return static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(heom::complex_t)
         + static_cast<size_t>(n_bath) * static_cast<size_t>(N) * sizeof(int8_t);
}

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
        throw std::overflow_error("HEOM derivative dynamic shared memory exceeds device limit");
    }
    return bytes;
}

inline void validate_derivative_context(const HEOMContext& ctx) {
    (void)checked_total_elements(ctx.M, ctx.N);
    (void)derivative_smem_bytes(ctx.N, ctx.n_bath);
    if (!ctx.total_decay) {
        throw std::invalid_argument("HEOMContext.total_decay must not be nullptr");
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

    // Manifold block. n_row/n_col == 0 is the default-constructed value, so
    // rejecting it is what forces every construction site to state its intent
    // — "(0,N,0,N)" for the whole matrix — rather than inheriting a launch
    // domain of nothing.
    if (ctx.n_row <= 0 || ctx.n_col <= 0) {
        throw std::invalid_argument(
            "HEOMContext manifold block must be non-empty; use (0,N,0,N) for "
            "the whole matrix");
    }
    if (ctx.row_off < 0 || ctx.col_off < 0 ||
        ctx.row_off + ctx.n_row > ctx.N || ctx.col_off + ctx.n_col > ctx.N) {
        throw std::invalid_argument("HEOMContext manifold block out of range");
    }
    const bool identical = (ctx.row_off == ctx.col_off && ctx.n_row == ctx.n_col);
    const bool disjoint  = (ctx.row_off + ctx.n_row <= ctx.col_off) ||
                           (ctx.col_off + ctx.n_col <= ctx.row_off);
    if (!identical && !disjoint) {
        throw std::invalid_argument(
            "HEOMContext manifold row/col ranges must be identical or disjoint");
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
    // Per-ADO real factors E=exp(-½·dt·g_α), E2=exp(-dt·g_α), g_α=total_decay[α].
    // The exp() is evaluated in FP64 from the exact g_α and stored rounded, so
    // the epilogue reads them at hier precision (same rule as c_bath).
    const real_t* e_half   = nullptr;  // [M] exp(-½·dt·g_α)
    const real_t* e_full   = nullptr;  // [M] exp(-dt·g_α)
    int           if_stage = 0;        // IF kStage: 1/2/3 (see rk4_step_if); else unused
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
 *        with g_α = total_decay[α]. Cheap (O(M) exp); recomputed only when
 *        dt changes (see rk4_step_if / HEOMTrajectory::run_step). The exp is
 *        evaluated in FP64 and the result stored at hier precision, so the
 *        per-thread epilogue read costs no FP64-pipe traffic.
 * @param d_total_decay device [M] Σ α_{nk}ν_k per ADO (SystemContext, FP64).
 * @param dt_au         internal timestep in a.u.
 * @param M             number of ADOs.
 * @param d_e_half      device out [M].
 * @param d_e_full      device out [M].
 */
void compute_if_factors(const double* d_total_decay, double dt_au, int M,
                        heom::real_t* d_e_half, heom::real_t* d_e_full);

#endif // HEOM_DERIVATIVES_CUH

