#include "heom_trajectory.hpp"
#include "util/stream_format.hpp"
#include "core/cuda_common.cuh"
#include "core/rk4_integrator.cuh"
#include "core/ado_filter.cuh"
#include "core/dipole_apply.cuh"
#include "core/precision.hpp"
#include "gpu_alloc_plan.hpp"
#include "memory_budget.hpp"
#include "ulist_builder.hpp"
#include "output/hdf5_writer.hpp"
#include "output/console_printer.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>      // std::swap (move-only GPUMemory ping-pong)

// Definition required because HEOMTrajectory holds std::unique_ptr<HDF5Writer>
// (forward-declared in heom_trajectory.hpp); ~unique_ptr needs the complete
// HDF5Writer destructor at the point the trajectory's destructor is emitted.
HEOMTrajectory::~HEOMTrajectory() = default;

HEOMTrajectory::HEOMTrajectory(const SystemContext& sys, bool enable_hdf5,
                               int batch)
    : sys_(sys)
    , enable_hdf5_(enable_hdf5)
    , batch_(batch)
    , active_batch_(batch)
    , use_filtering_(sys.config().filter.enabled)
    , filter_threshold_(sys.config().filter.tolerance)
    , integrator_(sys.config().time.integrator)
{
    if (batch_ < 1) {
        throw std::invalid_argument(
            "HEOMTrajectory: batch must be >= 1, got " + std::to_string(batch_));
    }
    // Enforce the M·N² < 2^31 kernel index ceiling HERE, the one chokepoint
    // every propagation path passes through. The derivative launcher re-checks
    // per launch, but the dipole/gather/trace kernels compute int totals
    // without a guard — and in 2DES the FIRST kernel to run is a dipole apply.
    // Failing loud at construction closes that window.
    (void)heom::checked_total_elements(sys_.M(), sys_.N());
    allocate_trajectory_buffers();
    if (enable_hdf5_) {
        writer_ = std::make_unique<HDF5Writer>(sys_.config().output.use_timestamp);
    }
}

void HEOMTrajectory::allocate_trajectory_buffers() {
    // Batched runs (B>1) multiply the two O(M·N²) mutable buffers plus the Kahan
    // carry by B. SystemContext's pre-alloc budget check only saw the B=1
    // footprint, so re-check here with the real batch BEFORE cudaMalloc'ing B×
    // the biggest arrays. B==1 skips it (SystemContext already covered it).
    if (batch_ > 1) {
        auto budget = heom::estimate_budget(
            sys_.config(), static_cast<size_t>(sys_.M()),
            static_cast<size_t>(batch_));
        size_t gpu_free = 0, gpu_total = 0;
        heom::query_gpu_memory(gpu_free, gpu_total);
        std::string err;
        switch (heom::check_budget(budget, sys_.config(), gpu_free, err)) {
            case heom::BudgetCheck::Fits:
                break;
            case heom::BudgetCheck::Exceeds:
                heom::print_budget(budget, gpu_free, gpu_total);
                throw std::runtime_error(err);
            case heom::BudgetCheck::Unknown:
                std::cerr << "[WARN] could not query GPU memory; "
                             "skipping batched budget check\n";
                break;
        }
    }

    // Only the trajectory-mutable subset of the canonical plan. batch_ scales
    // d_rho / d_workspace inside the plan (read-only structure stays ×1).
    const auto plan = heom::plan_gpu_buffers(
        static_cast<size_t>(sys_.M()), sys_.N(), sys_.K(),
        sys_.n_bath(), sys_.L(), use_filtering_, static_cast<size_t>(batch_));

    // typed_alloc throws std::logic_error if the plan's elem_bytes for the
    // named entry doesn't match sizeof(T), so a future change to either
    // side (this call site's GPUMemory<T> or gpu_alloc_plan.hpp's
    // sizeof(...) literal) cannot silently under/over-allocate.
    d_rho_       = heom::typed_alloc<heom::complex_t>(plan, "d_rho");
    d_workspace_ = heom::typed_alloc<heom::complex_t>(plan, "d_workspace");

    // Kahan carry for the mixed-precision RK4 combine: same shape
    // as ρ (batch·M·N²), allocated + zeroed ONCE here and persisted for the run.
    // FP64 build skips it entirely (empty ⇒ rk4_step gets nullptr ⇒ plain
    // combine ⇒ bit-identical to history). Not in the alloc plan because it is a
    // precision-mode-only cost; memory_budget.cpp accounts for it separately.
    if constexpr (heom::kMixedPrecision) {
        d_kahan_comp_ = GPUMemory<heom::complex_t>(
            static_cast<size_t>(batch_) * sys_.M() * sys_.N() * sys_.N());
        d_kahan_comp_.zero();
    }

    // IF-RK4 factor tables [M] (E/E2). Allocated ONLY on the IfRk4 path so the
    // default rk4 path pays zero extra GPU memory. Values are filled lazily on
    // the first run_step (dt is a run-time property), not here.
    if (integrator_ == Integrator::IfRk4) {
        d_if_e_half_ = GPUMemory<heom::real_t>(static_cast<size_t>(sys_.M()));
        d_if_e_full_ = GPUMemory<heom::real_t>(static_cast<size_t>(sys_.M()));
    }

    if (use_filtering_) {
        d_ado_norms_    = heom::typed_alloc<double>(plan, "d_ado_norms");
        d_active_flags_ = heom::typed_alloc<int8_t>(plan, "d_active_flags");
        d_level_counts_ = heom::typed_alloc<int>(plan, "d_level_counts");

        h_level_counts_.resize(d_level_counts_.size());

        // All ADOs active for the pre-first-filter warmup. active_flags[0] is
        // re-asserted every pass by the filter kernel (rho_0 active for life),
        // so this is only a default, not a correctness dependency.
        std::vector<int8_t> h_active_flags(sys_.M(), 1);
        d_active_flags_.copy_from_host(h_active_flags.data(), sys_.M());
        total_active_ados_ = sys_.M();
    }

    // Host mirror of the α=0 block, one per batch slot ([B][N²], slot-major).
    // B==1 ⇒ N² (unchanged). d2h_alpha0_batch() gathers the B non-contiguous
    // device α=0 blocks (spaced M·N² apart) into this contiguous host buffer.
    h_rho0_.resize(static_cast<size_t>(batch_) * sys_.N() * sys_.N());
}

// Gather the B α=0 blocks (device slot stride M·N²) into the contiguous host
// mirror h_rho0_ ([B][N²]). One cudaMemcpy2D. B==1 degenerates to a single N²
// D2H (dpitch==spitch==width, height 1), bit-identical to the old copy_to_host.
void HEOMTrajectory::probe_blocks(const char* where) const {
    const heom::StreamFormat fmt_guard(std::cout);
    static const bool on = (std::getenv("HEOM_BLOCK_PROBE") != nullptr);
    if (!on) return;

    const int N = sys_.N();
    const int M = sys_.M();
    const int B = active_batch_;

    // Manifold boundaries come from the declared basis, never from guessing at
    // N: an unrecognised basis must say so, not silently mislabel the blocks.
    const int n_states = sys_.config().physics.n_states;
    int bound[4] = {0, 0, 0, N};   // [g_lo, e_lo, f_lo, N) — empty ranges collapse
    const char* names[3] = {"g", "e", "f"};
    switch (sys_.config().physics.basis) {
        case BasisType::EXCITED_ONLY:     // no ground, no doubles: all of N is e
            bound[0] = 0; bound[1] = 0; bound[2] = N; break;
        case BasisType::WITH_GROUND:      // g=[0,1), e=[1,N), no f
            bound[0] = 0; bound[1] = 1; bound[2] = N; break;
        case BasisType::DOUBLE_EXCITED:   // g=[0,1), e=[1,1+n), f=[1+n,N)
            bound[0] = 0; bound[1] = 1; bound[2] = 1 + n_states; break;
        default:
            std::cout << "[blockprobe] unknown basis; refusing to label blocks\n";
            return;
    }

    std::vector<heom::complex_t> h(static_cast<size_t>(B) * M * N * N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_rho_.get(),
                          h.size() * sizeof(heom::complex_t),
                          cudaMemcpyDeviceToHost));

    // max |ρ| per (row manifold, column manifold), folded over every slot and
    // every ADO: one occupied block is the hypothesis, two is the refutation.
    double blk[3][3] = {};
    for (size_t e = 0; e < h.size(); ++e) {
        const double mag = std::abs(std::complex<double>(heom::c_re(h[e]),
                                                         heom::c_im(h[e])));
        if (mag == 0.0) continue;
        const int ij = static_cast<int>(e % (static_cast<size_t>(N) * N));
        const int i = ij / N, j = ij % N;
        int ri = 0, cj = 0;
        while (ri < 2 && i >= bound[ri + 1]) ++ri;
        while (cj < 2 && j >= bound[cj + 1]) ++cj;
        if (mag > blk[ri][cj]) blk[ri][cj] = mag;
    }

    std::cout << "[blockprobe] seg=" << (log_prefix_.empty() ? "-" : log_prefix_)
              << " at=" << where << " N=" << N << " B=" << B << " occ=";
    int n_occ = 0;
    int occ_r = -1, occ_c = -1;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            if (blk[r][c] > 0.0) {
                std::cout << (n_occ ? "," : "") << names[r] << names[c] << ":"
                          << std::scientific << std::setprecision(2) << blk[r][c];
                ++n_occ;
                occ_r = r; occ_c = c;
            }
    if (n_occ == 0) std::cout << "-";
    std::cout << " n_occ=" << n_occ;

    // What the kernel will ACTUALLY launch over — read from ctx_, not from the
    // blk_* members. The difference matters: with HEOM_NO_MANIFOLD_BLOCK the
    // driver still declares blocks, and printing the declaration would show a
    // block while the kernel sweeps the whole matrix. This column is the one
    // that can be compared against occ.
    //
    // Before the first propagation ctx_ does not exist yet (build_context is
    // lazy), and printing its zero-initialised fields would flag every init and
    // post-dipole observation as a mismatch against a launch that is not going
    // to happen. Say "unbuilt" and skip the comparison.
    if (!ctx_built_) {
        std::cout << " launch=unbuilt\n" << std::flush;
        return;
    }
    std::cout << " launch=" << ctx_.row_off << "+" << ctx_.n_row
              << "," << ctx_.col_off << "+" << ctx_.n_col;

    // Occupancy outside the launch domain is the failure this probe exists for:
    // those elements are neither read nor written, so they are frozen and the
    // answer is silently wrong. Say so in a word grep can find.
    if (n_occ == 1) {
        const bool inside = bound[occ_r]     >= ctx_.row_off
                         && bound[occ_r + 1] <= ctx_.row_off + ctx_.n_row
                         && bound[occ_c]     >= ctx_.col_off
                         && bound[occ_c + 1] <= ctx_.col_off + ctx_.n_col;
        if (!inside) std::cout << " MISMATCH";
    } else if (n_occ > 1) {
        std::cout << " MISMATCH";   // more than one block occupied: no single
                                    // declaration can be right
    }
    std::cout << "\n" << std::flush;
}

void HEOMTrajectory::on_rho_replaced() {
    // Empty in the FP64 build (the kernel gets nullptr and runs the plain
    // combine), so this whole hook costs nothing there.
    if (d_kahan_comp_.size() > 0) d_kahan_comp_.zero();
    clear_manifold_block();
    // The new ρ has not been propagated yet: whatever the hierarchy looked like
    // a moment ago says nothing about it. See fresh_state_.
    fresh_state_ = true;
}

namespace {
// The kill switch, read once. It lives at the point of CONSUMPTION (see
// apply_manifold_block_to_ctx): switching it off at the setter instead would
// skip validation, so the two halves of an A/B run would not even execute the
// same checks, and blk_set_ could no longer mean "the driver declared a block".
bool manifold_blocking_enabled() {
    static const bool on = (std::getenv("HEOM_NO_MANIFOLD_BLOCK") == nullptr);
    return on;
}
}  // namespace

void HEOMTrajectory::apply_manifold_block_to_ctx() {
    const bool use = blk_set_ && manifold_blocking_enabled();
    ctx_.row_off = use ? blk_row_off_ : 0;
    ctx_.n_row   = use ? blk_n_row_   : sys_.N();
    ctx_.col_off = use ? blk_col_off_ : 0;
    ctx_.n_col   = use ? blk_n_col_   : sys_.N();
}

void HEOMTrajectory::set_manifold_block(int row_off, int n_row,
                                        int col_off, int n_col) {
    const int N = sys_.N();
    if (n_row <= 0 || n_col <= 0 || row_off < 0 || col_off < 0 ||
        row_off + n_row > N || col_off + n_col > N) {
        throw std::invalid_argument(
            "HEOMTrajectory::set_manifold_block: block out of range");
    }
    const bool identical = (row_off == col_off && n_row == n_col);
    const bool disjoint  = (row_off + n_row <= col_off) ||
                           (col_off + n_col <= row_off);
    if (!identical && !disjoint) {
        throw std::invalid_argument(
            "HEOMTrajectory::set_manifold_block: row/col ranges must be "
            "identical or disjoint");
    }

    blk_row_off_ = row_off; blk_n_row_ = n_row;
    blk_col_off_ = col_off; blk_n_col_ = n_col;
    blk_set_     = true;
    // build_context() is idempotent (ctx_built_ guard), so a built context has
    // to be updated here — the same reason set_active_batch touches ctx_.batch.
    if (ctx_built_) apply_manifold_block_to_ctx();

    if (!manifold_blocking_enabled()) {
        // Say so once. A silent kill switch turns an A/B run into a coin flip
        // over whether the variable was actually exported.
        static bool announced = false;
        if (!announced) {
            std::cout << "[manifold] HEOM_NO_MANIFOLD_BLOCK set — declaring "
                         "blocks but launching the full N^2 domain\n" << std::flush;
            announced = true;
        }
    }
}

void HEOMTrajectory::clear_manifold_block() {
    // Silent: on_rho_replaced() calls this on every dipole/restore/expand.
    blk_set_ = false;
    if (ctx_built_) apply_manifold_block_to_ctx();
}

void HEOMTrajectory::d2h_alpha0_batch() {
    const size_t N2b   = static_cast<size_t>(sys_.N()) * sys_.N()
                       * sizeof(heom::complex_t);
    const size_t slot  = static_cast<size_t>(sys_.M()) * sys_.N() * sys_.N()
                       * sizeof(heom::complex_t);
    CUDA_CHECK(cudaMemcpy2D(
        /*dst=*/h_rho0_.data(), /*dpitch=*/N2b,
        /*src=*/d_rho_.get(),   /*spitch=*/slot,
        /*width=*/N2b, /*height=*/static_cast<size_t>(active_batch_),
        cudaMemcpyDeviceToHost));
}

void HEOMTrajectory::set_initial(const Eigen::MatrixXcd& rho0) {
    if (batch_ != 1) {
        throw std::runtime_error(
            "HEOMTrajectory::set_initial: single-state API on a batched "
            "trajectory (batch=" + std::to_string(batch_)
            + "); use set_initial_batch()");
    }
    // batch_==1 forces active_batch_==1, so the B=1 batch path is exactly this
    // API: one N² H2D into the α=0 slot of a zeroed tensor, initial_norm_ =
    // ‖ρ_0‖_F, and the per-tree counter reset.
    set_initial_batch({rho0});
}

void HEOMTrajectory::set_initial_batch(const std::vector<Eigen::MatrixXcd>& rho0s) {
    // Bound to active_batch_ (not batch_): the 2DES walk sets the τ layer at
    // width 3 on a batch=21 trajectory. Default active_batch_==batch_ keeps the
    // absorption path (B=3, active=3) unchanged.
    if (static_cast<int>(rho0s.size()) != active_batch_) {
        throw std::runtime_error(
            "HEOMTrajectory::set_initial_batch: expected " + std::to_string(active_batch_)
            + " initial matrices, got " + std::to_string(rho0s.size()));
    }
    const int N = sys_.N();
    const size_t N2 = static_cast<size_t>(N) * N;

    // Zero the whole [batch_][M][N²] tensor (all slots, so idle slots > active
    // are clean), then pack each active slot's ρ_0 into the contiguous host
    // mirror and scatter the active α=0 blocks in one cudaMemcpy2D.
    d_rho_.zero();
    initial_norm_ = 0.0;
    for (int b = 0; b < active_batch_; ++b) {
        const Eigen::MatrixXcd& r = rho0s[b];
        if (r.rows() != N || r.cols() != N) {
            throw std::runtime_error(
                "HEOMTrajectory::set_initial_batch: rho0[" + std::to_string(b)
                + "] size mismatch (expected " + std::to_string(N) + "x"
                + std::to_string(N) + ")");
        }
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                h_rho0_[static_cast<size_t>(b) * N2 + i * N + j] = heom::c_make(
                    static_cast<heom::real_t>(r(i, j).real()),
                    static_cast<heom::real_t>(r(i, j).imag()));
        // Divergence ceiling is a single scalar for the whole batch: take the
        // max ‖ρ‖_F so the guard holds for the most energetic slot too.
        initial_norm_ = std::max(initial_norm_, r.norm());
    }

    const size_t N2b  = N2 * sizeof(heom::complex_t);
    const size_t slot = static_cast<size_t>(sys_.M()) * N2 * sizeof(heom::complex_t);
    CUDA_CHECK(cudaMemcpy2D(
        /*dst=*/d_rho_.get(), /*dpitch=*/slot,
        /*src=*/h_rho0_.data(), /*spitch=*/N2b,
        /*width=*/N2b, /*height=*/static_cast<size_t>(active_batch_),
        cudaMemcpyHostToDevice));

    on_rho_replaced();
    initial_rho_       = rho0s[0];   // HDF5 metadata unused in batch, keep slot 0
    initial_set_       = true;
    internal_step_     = 0;
    filter_pass_index_ = 0;
    ulist_dense_       = -1;
    probe_blocks("init");
}

void HEOMTrajectory::set_initial_from_config() {
    const heom::StreamFormat fmt_guard(std::cout);
    // cfg.initial is optional: spectroscopy tasks build ρ(0) explicitly via
    // set_initial(rho); only the dynamics entry path comes through here. Keep a
    // defence-in-depth check against a future caller dereferencing an empty one.
    if (!sys_.config().initial.has_value()) {
        throw std::runtime_error(
            "HEOMTrajectory::set_initial_from_config(): cfg.initial is empty. "
            "Build ρ(0) yourself with set_initial(rho), or run the dynamics "
            "task entry which requires 'initial_state:' in yaml.");
    }
    const InitialState& init = *sys_.config().initial;

    // The human-readable summary lives here, not in build_initial_state(), so
    // 2DES drivers that build ρ(0) themselves stay quiet.
    switch (init.type) {
        case InitialState::Type::SITE_EXCITATION:
            std::cout << " pigment "
                      << hilbert_to_pigment(init.excited_site,
                                            sys_.config().physics.basis)
                      << " excited";
            break;
        case InitialState::Type::EXCITON_EXCITATION:
            heom::console::print_eigenspectrum(
                sys_.eigenvalues(), init.exciton_index, sys_.N(),
                sys_.config().physics.w0_cm);
            heom::console::print_exciton_composition(
                sys_.eigenvectors(), init.exciton_index, sys_.N());
            break;
        case InitialState::Type::COHERENT:
            std::cout << " coherent superposition";
            break;
        case InitialState::Type::CUSTOM_MATRIX:
            std::cout << " custom matrix";
            break;
    }

    initial_rho_ = build_initial_state(sys_, init);

    if (init.type == InitialState::Type::EXCITON_EXCITATION) {
        // print_density_matrix_summary needs the actual built matrix
        heom::console::print_density_matrix_summary(initial_rho_, sys_.N());
    } else if (init.type == InitialState::Type::COHERENT) {
        // Per-site populations need init.coherent_coeffs
        for (int i = 0; i < sys_.N(); ++i) {
            const double pop = std::norm(init.coherent_coeffs[i]);
            if (pop > 1e-6)
                std::cout << " |c" << i << "|^2="
                          << std::fixed << std::setprecision(3) << pop;
        }
    }

    set_initial(initial_rho_);
}

double HEOMTrajectory::final_trace() const {
    double t = 0.0;
    const int N = sys_.N();
    for (int i = 0; i < N; ++i) t += heom::c_re(h_rho0_[i * N + i]);
    return t;
}

double HEOMTrajectory::rho0_frobenius_norm() const {
    // Batched: return max_b ‖ρ_0^(b)‖_F so the divergence guard trips on the
    // most energetic slot. B==1 ⇒ the single historical norm (bit-identical).
    // Only over ACTIVE slots: d2h_alpha0_batch() filled the first active_batch_
    // entries; higher slots hold stale data that must not fold into the max.
    const size_t N2 = static_cast<size_t>(sys_.N()) * sys_.N();
    double max_norm = 0.0;
    for (int b = 0; b < active_batch_; ++b) {
        double sum_sq = 0.0;
        for (size_t k = 0; k < N2; ++k) {
            const auto v = h_rho0_[static_cast<size_t>(b) * N2 + k];
            const double re = heom::c_re(v);
            const double im = heom::c_im(v);
            sum_sq += re * re + im * im;
        }
        max_norm = std::max(max_norm, std::sqrt(sum_sq));
    }
    return max_norm;
}

Eigen::MatrixXcd HEOMTrajectory::final_rho0() const {
    // Read from h_rho0_: solve() guarantees this mirror holds the final ρ_0
    // (the conditional D2H at the bottom of solve()).
    const int N = sys_.N();
    Eigen::MatrixXcd out(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const auto v = h_rho0_[i * N + j];
            out(i, j) = std::complex<double>(heom::c_re(v), heom::c_im(v));
        }
    return out;
}

Eigen::MatrixXcd HEOMTrajectory::slot_rho0(int b) const {
    if (b < 0 || b >= batch_) {
        throw std::out_of_range(
            "HEOMTrajectory::slot_rho0: slot " + std::to_string(b)
            + " out of range [0, " + std::to_string(batch_) + ")");
    }
    const int N = sys_.N();
    const size_t N2 = static_cast<size_t>(N) * N;
    std::vector<heom::complex_t> buf(N2);
    // slot b's α=0 block sits at offset b·M·N² (α=0 is the first ADO).
    CUDA_CHECK(cudaMemcpy(buf.data(),
                          d_rho_.get() + static_cast<size_t>(b) * sys_.M() * N2,
                          N2 * sizeof(heom::complex_t), cudaMemcpyDeviceToHost));
    Eigen::MatrixXcd out(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            out(i, j) = std::complex<double>(heom::c_re(buf[i * N + j]),
                                             heom::c_im(buf[i * N + j]));
    return out;
}

const cuDoubleComplex* HEOMTrajectory::rho0_as_fp64() const {
    if constexpr (heom::kMixedPrecision) {
        // Mixed build: uplift the FP32 mirror to FP64 for the float64-only
        // HDF5 writer (observable uplift). Reused scratch, no per-call alloc.
        const size_t n = h_rho0_.size();
        h_rho0_fp64_.resize(n);
        for (size_t k = 0; k < n; ++k)
            h_rho0_fp64_[k] = make_cuDoubleComplex(
                heom::c_re(h_rho0_[k]), heom::c_im(h_rho0_[k]));
        return h_rho0_fp64_.data();
    } else {
        // FP64 build: complex_t IS cuDoubleComplex — hand back the mirror
        // directly, zero copy, bit-identical to history.
        return reinterpret_cast<const cuDoubleComplex*>(h_rho0_.data());
    }
}

void HEOMTrajectory::build_context() {
    if (ctx_built_) return;
    const auto& cfg = sys_.config();
    ctx_.hamiltonian        = sys_.d_hamiltonian();
    ctx_.c_bath             = sys_.d_c_bath();
    ctx_.coeff_plus         = sys_.d_coeff_plus();
    ctx_.coeff_minus        = sys_.d_coeff_minus();
    ctx_.markov_residue     = sys_.d_markov_residue();
    ctx_.neighbor_map_plus  = sys_.d_plus_map();
    ctx_.neighbor_map_minus = sys_.d_minus_map();
    ctx_.level_indices      = sys_.d_level_indices();
    ctx_.total_decay        = sys_.d_total_decay();
    ctx_.active_flags       = use_filtering_ ? d_active_flags_.get() : nullptr;
    ctx_.d_coupling         = sys_.d_d_coupling();
    ctx_.M = sys_.M();  ctx_.N = sys_.N();  ctx_.K = sys_.K();
    ctx_.n_bath       = sys_.n_bath();
    ctx_.batch        = active_batch_;   // grid.y width; set_active_batch mutates
                                         // this too (build_context is idempotent)
    // Spelled out even when no block is set — the kernel gets a real rectangle,
    // never a sentinel to decode (see HEOMContext::row_off).
    apply_manifold_block_to_ctx();
    ctx_.sink_enabled = cfg.physics.sink.enabled;
    ctx_.sink_site    = cfg.physics.sink.site;
    ctx_.gamma_sink   = cfg.physics.sink.gamma;

    // U-list launch compaction (opt-in via time_evolution.ulist). Build the
    // collector once (D2H of the neighbor maps) and point ctx_ at the identity
    // list; the first filter step swaps in the real closure set.
    init_ulist_if_enabled();
    if (ulist_builder_) {
        ctx_.u_list  = ulist_builder_->device_list();
        ctx_.u_count = ulist_builder_->count();
    }
    ctx_built_ = true;
}

void HEOMTrajectory::set_active_batch(int batch) {
    if (batch < 1 || batch > batch_) {
        throw std::invalid_argument(
            "HEOMTrajectory::set_active_batch: batch must be in [1, "
            + std::to_string(batch_) + "], got " + std::to_string(batch));
    }
    active_batch_ = batch;
    // build_context() is guarded by ctx_built_ and will not re-run, so update
    // the live launch width directly. If ctx_ is not built yet, build_context()
    // will pick up active_batch_ on its first (and only) run.
    if (ctx_built_) ctx_.batch = batch;
}

void HEOMTrajectory::propagate_steps(int n_internal_steps, bool check_div) {
    if (!initial_set_) {
        throw std::runtime_error(
            "HEOMTrajectory::propagate_steps() called before set_initial()");
    }
    // Once per SEGMENT, not per step. A missing block is correct but slow, and
    // a bitwise A/B gate cannot see it — both halves agree precisely because
    // nothing was blocked. This counter is the only thing that can.
    if (!blk_set_) ++propagations_without_block_;
    // The 2DES carry-through driver never calls solve(), so this is the only
    // place a blown-up segment can be caught before its NaN/Inf silently fills
    // the response grid. diverged_ latches across the whole tree (set_initial
    // does not clear it) so once tripped, the remaining propagation short-
    // circuits and compute_2des()'s diverged() check aborts the run.
    if (diverged_) return;
    build_context();
    for (int s = 0; s < n_internal_steps; ++s) {
        run_one_internal_step(ctx_, internal_step_);
        ++internal_step_;
    }
    // One check per call, as documented. Only the 2DES carry-through driver
    // calls this, and it has no output cadence to piggyback on: the previous
    // `internal_step_ % max(1, time.output_interval)` clause read a knob 2DES
    // pins to 0, so it clamped to 1 and forced a D2H + implicit sync on EVERY
    // internal step of the tau scan and the T gaps. A blow-up is not missed by
    // checking only here: NaN/Inf propagate forward, so the segment-end test
    // still catches it.
    // t_fs is this trajectory's total propagated time, not a physical delay:
    // internal_step_ accumulates across every branch of the 2DES tree walk,
    // including the ones later discarded.
    if (check_div && n_internal_steps > 0) {
        check_divergence(internal_step_ * sys_.config().time.dt, internal_step_, 0);
    }
    probe_blocks("after-propagate");
}

bool HEOMTrajectory::check_divergence_now(double t_fs, int step) {
    if (diverged_) return true;
    return check_divergence(t_fs, step, 0);
}

void HEOMTrajectory::detect_begin(const Eigen::MatrixXcd& mu_det, int n_samples) {
    // Single-state API: on a wider active batch it would trace slot 0 only and
    // silently drop the rest. Mirror set_initial's guard.
    if (active_batch_ != 1) {
        throw std::runtime_error(
            "HEOMTrajectory::detect_begin: single-state API at active_batch="
            + std::to_string(active_batch_) + "; use detect_begin_batch()");
    }
    // B_det=1 sizes d_mu_det_ to N² and d_detect_ to n_samples, exactly as the
    // single-slot path did.
    detect_begin_batch({mu_det}, n_samples);
}

void HEOMTrajectory::detect_sample(int idx) {
    // detect_batch_==1 ⇒ grid.y==1, slot offset 0, out[0·n_samples+idx]==out[idx]:
    // the same launch the single-slot trace used.
    detect_sample_batch(idx);
}

void HEOMTrajectory::detect_flush(std::complex<double>* out_host, int n_samples) {
    detect_flush_batch(out_host, n_samples);
}

void HEOMTrajectory::detect_begin_batch(
    const std::vector<Eigen::MatrixXcd>& mus_det, int n_samples) {
    const int N = sys_.N();
    const int B_det = static_cast<int>(mus_det.size());
    if (B_det < 1 || B_det > batch_) {
        throw std::invalid_argument(
            "detect_begin_batch: B_det=" + std::to_string(B_det)
            + " out of range [1, " + std::to_string(batch_) + "]");
    }
    if (n_samples <= 0) {
        throw std::runtime_error("detect_begin_batch: n_samples must be > 0");
    }
    for (int b = 0; b < B_det; ++b) {
        if (mus_det[b].rows() != N || mus_det[b].cols() != N) {
            throw std::runtime_error(
                "detect_begin_batch: mu_det[" + std::to_string(b)
                + "] size mismatch (expected " + std::to_string(N) + "x"
                + std::to_string(N) + ")");
        }
    }
    const size_t N2 = static_cast<size_t>(N) * N;

    // Detection width comes from mus_det.size() and is stored; detect_sample_
    // batch launches at this width regardless of the current active_batch_.
    detect_batch_     = B_det;
    detect_n_samples_ = n_samples;

    // Per-slot μ_det, packed slot-major [B_det·N²].
    const size_t mu_need = static_cast<size_t>(B_det) * N2;
    d_mu_det_.ensure(mu_need);
    h_mu_pack_.resize(mu_need);
    for (int b = 0; b < B_det; ++b) {
        const auto& mu = mus_det[b];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                h_mu_pack_[static_cast<size_t>(b) * N2 + i * N + j] = heom::c_make(
                    static_cast<heom::real_t>(mu(i, j).real()),
                    static_cast<heom::real_t>(mu(i, j).imag()));
    }
    d_mu_det_.copy_from_host(h_mu_pack_.data(), mu_need);

    // Result grid [B_det·n_samples], slot-major (slot b at [b·n_samples..)).
    const size_t det_need = static_cast<size_t>(B_det) * n_samples;
    d_detect_.ensure(det_need);
    h_detect_.resize(det_need);
}

void HEOMTrajectory::detect_sample_batch(int idx) {
    // grid.y = detect_batch_; slot b traces its own α=0 block (b·M·N²) against
    // its own μ_det (b·N²) into out[b·n_samples+idx]. No sync / no D2H.
    launch_trace_BR_batch(d_mu_det_.get(), d_rho_.get(), d_detect_.get(),
                          idx, sys_.N(), sys_.M(), detect_batch_,
                          detect_n_samples_);
}

void HEOMTrajectory::detect_flush_batch(std::complex<double>* out_host,
                                        int n_samples) {
    const size_t total = static_cast<size_t>(detect_batch_) * n_samples;
    CUDA_CHECK(cudaMemcpy(h_detect_.data(), d_detect_.get(),
                          total * sizeof(heom::complex_t),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < total; ++i)
        out_host[i] = std::complex<double>(heom::c_re(h_detect_[i]),
                                           heom::c_im(h_detect_[i]));
}

void HEOMTrajectory::apply_dipole_full_hierarchy(
    const Eigen::MatrixXcd& mu, heom::spectra::DipoleSide side) {
    if (!initial_set_) {
        throw std::runtime_error(
            "HEOMTrajectory::apply_dipole_full_hierarchy() before set_initial()");
    }
    // Single-state API: at a wider active batch the single-slot launch would
    // touch slot 0 only while the refilter/norm refresh below spans all active
    // slots — a silent half-update. Mirror set_initial's guard.
    if (active_batch_ != 1) {
        throw std::runtime_error(
            "HEOMTrajectory::apply_dipole_full_hierarchy: single-state API at "
            "active_batch=" + std::to_string(active_batch_)
            + "; use apply_dipole_per_slot()");
    }
    // active_batch_==1 ⇒ the batched launch is grid.y==1 over the same [M·N²]
    // range with the same single-copy μ, so this is the batch path verbatim.
    apply_dipole_per_slot({mu}, side);
}

void HEOMTrajectory::apply_dipole_per_slot(
    const std::vector<Eigen::MatrixXcd>& mus, heom::spectra::DipoleSide side) {
    if (!initial_set_) {
        throw std::runtime_error(
            "HEOMTrajectory::apply_dipole_per_slot() before set_initial()");
    }
    if (static_cast<int>(mus.size()) != active_batch_) {
        throw std::runtime_error(
            "apply_dipole_per_slot: expected " + std::to_string(active_batch_)
            + " dipoles (active_batch_), got " + std::to_string(mus.size()));
    }
    const int N = sys_.N();
    const int M = sys_.M();
    const size_t N2 = static_cast<size_t>(N) * N;
    for (int b = 0; b < active_batch_; ++b) {
        if (mus[b].rows() != N || mus[b].cols() != N) {
            throw std::runtime_error(
                "apply_dipole_per_slot: mu[" + std::to_string(b)
                + "] size mismatch (expected " + std::to_string(N) + "x"
                + std::to_string(N) + ")");
        }
    }

    // Pack per-slot μ into [active_batch_·N²] device scratch (grow-only).
    const size_t mu_need = static_cast<size_t>(active_batch_) * N2;
    d_mu_scratch_.ensure(mu_need);
    h_mu_pack_.resize(mu_need);
    for (int b = 0; b < active_batch_; ++b) {
        const auto& mu = mus[b];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                h_mu_pack_[static_cast<size_t>(b) * N2 + i * N + j] = heom::c_make(
                    static_cast<heom::real_t>(mu(i, j).real()),
                    static_cast<heom::real_t>(mu(i, j).imag()));
    }
    d_mu_scratch_.copy_from_host(h_mu_pack_.data(), mu_need);

    // Batched single-side multiply into the RK4 scratch (idle here), then D2D
    // copy the active prefix back. Slot b uses mu[b·N²], reads/writes b·M·N².
    const int s = (side == heom::spectra::DipoleSide::Left) ? 0 : 1;
    const size_t MN2 = static_cast<size_t>(M) * N2;
    launch_apply_single_side_mu_batch(d_rho_.get(), d_workspace_.get(),
                                      d_mu_scratch_.get(), s, M, N, active_batch_);
    CUDA_CHECK(cudaMemcpy(d_rho_.get(), d_workspace_.get(),
                          static_cast<size_t>(active_batch_) * MN2
                              * sizeof(heom::complex_t),
                          cudaMemcpyDeviceToDevice));
    on_rho_replaced();

    // μ replaced ρ, so the active set describes a hierarchy that no longer
    // exists. This used to refilter here ("μ shifted the norm distribution"),
    // which at a tree start read exact zeros and marked everything inactive.
    // All-active is the only claim that is true of an unpropagated hierarchy,
    // and it is what on_rho_replaced()/fresh_state_ would install on the next
    // step anyway — so state this directly instead of computing a verdict that
    // is always either wrong or discarded.
    if (use_filtering_) {
        build_context();
        activate_all_ados(ctx_);
    }
    d2h_alpha0_batch();
    initial_norm_ = rho0_frobenius_norm();
    probe_blocks("after-dipole");
}

void HEOMTrajectory::snapshot_device(DeviceHierarchySnapshot& snap) {
    const int N = sys_.N();
    const int M = sys_.M();
    const size_t N2 = static_cast<size_t>(N) * N;
    snap.N = N;
    snap.batch = active_batch_;   // # slots captured (variable-width walk)

    // Active set = recompute norms→flags NOW (never read stale flags); this also
    // refreshes the ulist state, part of the snapshot contract. perform_ado_
    // filtering folds the union over the active_batch_ slots.
    std::vector<int> ado_ids;
    if (use_filtering_) {
        refresh_active_set_and_ulist(filter_threshold_);
        std::vector<int8_t> hflags(M);
        d_active_flags_.copy_to_host(hflags.data(), M);
        ado_ids.reserve(M);
        for (int a = 0; a < M; ++a)
            if (hflags[a]) ado_ids.push_back(a);
    } else {
        ado_ids.reserve(M);
        for (int a = 0; a < M; ++a) ado_ids.push_back(a);
    }

    const int n_active = static_cast<int>(ado_ids.size());
    snap.n_active = n_active;

    // alpha = 0 divergence baseline: max over active slots of ||rho_0||_F.
    // d2h_alpha0_batch fills h_rho0_ with the
    // active_batch_ α=0 blocks; rho0_frobenius_norm folds the max.
    d2h_alpha0_batch();
    snap.alpha0_norm = rho0_frobenius_norm();

    if (n_active == 0) return;

    // ids buffer sized to M (grow-only) so it is allocated at most once. Shared
    // by all slots (union closure): one ids list drives every slot's gather.
    snap.ids.ensure(static_cast<size_t>(M));
    snap.ids.copy_from_host(ado_ids.data(), n_active);

    // Block buffer [batch·n_active·N²], grow-only. After filter warmup n_active
    // stabilises, so reallocs stop.
    const size_t need = static_cast<size_t>(active_batch_) * n_active * N2;
    snap.blocks.ensure(need);
    // Device gather over active slots: NO D2H, NO host conversion. Every slot
    // gathers the SAME ids into its own [b·n_active·N²] staging region.
    launch_gather_ado_blocks_batch(d_rho_.get(), snap.blocks.get(),
                                   snap.ids.get(), n_active, N, M, active_batch_);

    if (use_filtering_) {
        snap.flags.ensure(static_cast<size_t>(M));
        CUDA_CHECK(cudaMemcpy(snap.flags.get(), d_active_flags_.get(),
                              static_cast<size_t>(M),
                              cudaMemcpyDeviceToDevice));
    }
}

void HEOMTrajectory::restore_device(const DeviceHierarchySnapshot& snap) {
    const int N = sys_.N();
    const int M = sys_.M();
    if (snap.N != N) {
        throw std::runtime_error(
            "restore_device: snapshot N mismatch (expected "
            + std::to_string(N) + ", got " + std::to_string(snap.N) + ")");
    }

    // Restore the captured width first: sets active_batch_ AND ctx_.batch, so
    // every subsequent launch (and the zero/scatter below) spans snap.batch
    // slots. Without this the launch width would stay at the caller's prior
    // (e.g. t-layer 21) width and advance stale slots.
    set_active_batch(snap.batch);

    // Zero the active working prefix → device scatter the active blocks back
    // (D2D, no H2D). Only snap.batch·M·N² is cleared: slots ≥ snap.batch are
    // inactive (never launched) until a later expand overwrites them.
    const size_t N2 = static_cast<size_t>(N) * N;
    CUDA_CHECK(cudaMemset(
        d_rho_.get(), 0,
        static_cast<size_t>(snap.batch) * M * N2 * sizeof(heom::complex_t)));
    if (snap.n_active > 0) {
        launch_scatter_ado_blocks_batch(snap.blocks.get(), d_rho_.get(),
                                        snap.ids.get(), snap.n_active, N, M,
                                        snap.batch);
    }
    on_rho_replaced();

    // Restore active_flags from the device snapshot + rebuild U immediately
    // (flags come from the snapshot, so we rebuild U but do NOT refilter — do
    // not wait for the next filter step, which would freeze snapshot-active
    // ADOs for a full step). Flags/ulist are shared across all slots (union).
    if (use_filtering_) {
        CUDA_CHECK(cudaMemcpy(d_active_flags_.get(), snap.flags.get(),
                              static_cast<size_t>(M),
                              cudaMemcpyDeviceToDevice));
        total_active_ados_ = snap.n_active;
        build_context();
        rebuild_ulist_into_ctx();
    }

    initial_norm_ = snap.alpha0_norm;
    initial_set_  = true;
}

void HEOMTrajectory::expand_slots(const DeviceHierarchySnapshot& parent,
                                  const std::vector<int>& child_to_parent) {
    const int N = sys_.N();
    const int M = sys_.M();
    if (parent.N != N) {
        throw std::runtime_error(
            "expand_slots: snapshot N mismatch (expected "
            + std::to_string(N) + ", got " + std::to_string(parent.N) + ")");
    }
    const int B_child = static_cast<int>(child_to_parent.size());
    if (B_child < 1 || B_child > batch_) {
        throw std::invalid_argument(
            "expand_slots: child count " + std::to_string(B_child)
            + " out of range [1, " + std::to_string(batch_) + "]");
    }
    const size_t N2  = static_cast<size_t>(N) * N;
    const size_t MN2 = static_cast<size_t>(M) * N2;

    // Set the child width (active_batch_ + ctx_.batch) BEFORE any launch.
    set_active_batch(B_child);

    // Zero the child working prefix, then fill each child slot from its parent
    // slot's active blocks (shared ids). The map is general/uneven (3→9, 9→21).
    CUDA_CHECK(cudaMemset(d_rho_.get(), 0,
                          static_cast<size_t>(B_child) * MN2 * sizeof(heom::complex_t)));
    if (parent.n_active > 0) {
        for (int c = 0; c < B_child; ++c) {
            const int p = child_to_parent[c];
            if (p < 0 || p >= parent.batch) {
                throw std::out_of_range(
                    "expand_slots: child_to_parent[" + std::to_string(c) + "]="
                    + std::to_string(p) + " out of range [0, "
                    + std::to_string(parent.batch) + ")");
            }
            // Scatter parent slot p's gathered blocks into child slot c's ρ
            // region (single-slot scatter into a per-child base pointer; shared
            // ids place each block at its ADO id within slot c).
            launch_scatter_ado_blocks(
                parent.blocks.get() + static_cast<size_t>(p) * parent.n_active * N2,
                d_rho_.get() + static_cast<size_t>(c) * MN2,
                parent.ids.get(), parent.n_active, N);
        }
    }
    on_rho_replaced();

    // Inherit the shared flags/ulist + divergence baseline from the parent.
    if (use_filtering_) {
        CUDA_CHECK(cudaMemcpy(d_active_flags_.get(), parent.flags.get(),
                              static_cast<size_t>(M),
                              cudaMemcpyDeviceToDevice));
        total_active_ados_ = parent.n_active;
        build_context();
        rebuild_ulist_into_ctx();
    }

    initial_norm_ = parent.alpha0_norm;
    initial_set_  = true;
}

std::complex<double> HEOMTrajectory::expectation(const Eigen::MatrixXcd& B) const {
    const int N = sys_.N();
    if (B.rows() != N || B.cols() != N) {
        throw std::runtime_error(
            "expectation: B size mismatch (expected "
            + std::to_string(N) + "x" + std::to_string(N) + ")");
    }
    const size_t N2 = static_cast<size_t>(N) * N;

    // Own D2H of the α=0 block — propagate_steps() does not refresh h_rho0_.
    std::vector<heom::complex_t> rho0(N2);
    CUDA_CHECK(cudaMemcpy(rho0.data(), d_rho_.get(),
                          N2 * sizeof(heom::complex_t), cudaMemcpyDeviceToHost));

    // Tr(B·ρ0) = Σ_i Σ_k B(i,k) ρ0(k,i)
    std::complex<double> tr(0.0, 0.0);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            const heom::complex_t v = rho0[static_cast<size_t>(k) * N + i];
            tr += B(i, k) * std::complex<double>(heom::c_re(v), heom::c_im(v));
        }
    return tr;
}

void HEOMTrajectory::perform_ado_filtering(double threshold,
                                           int row_off, int n_row,
                                           int col_off, int n_col,
                                           const int* scan_list,
                                           int scan_count) {
    // U-local contract: the scan domain MUST cover the previous
    // write set of ρ. New threshold crossings can only happen where ρ was
    // written; everything outside kept a frozen sub-threshold norm and an
    // inactive flag (A ⊆ U), so skipping it is exact — not an approximation.
    //   scan_list == nullptr : full-M sweep (previous write set was the full
    //                          grid: dense fallback, ulist off, dipole apply,
    //                          restore-scatter).
    //   scan_list != nullptr : compact sweep over the previous whole-step's U.
    // Batched (active_batch_>1): the norm kernel folds the union max over the
    // ACTIVE slots, so the single active_flags is the union closure — an ADO
    // active in ANY active slot survives. Only active_batch_ slots
    // hold live state (higher slots are stale/idle), so union over active_batch_
    // (not batch_) is both correct and avoids folding stale slots.
    // active_batch_==1 ⇒ ordinary single-state norms.
    // Manifold block: scanning only the sub-block is exact, not an
    // approximation — out-of-block entries are exactly zero and the max norm
    // has no accumulation, so max(sub-block) ≡ max(whole matrix).
    compute_ado_norms(d_rho_.get(), d_ado_norms_.get(), sys_.M(), sys_.N(),
                      row_off, n_row, col_off, n_col,
                      scan_list, scan_count, active_batch_);

    // Mark-only: write active_flags on device, no count / no D2H. |A| is
    // derived on demand from the level distribution in emit_ado_stats, so the
    // hot per-step filter path carries zero host synchronization.
    filter_ados_mark_only(
        d_ado_norms_.get(), d_active_flags_.get(), threshold, sys_.M(),
        scan_list, scan_count);
}

void HEOMTrajectory::rebuild_ulist_into_ctx() {
    if (!ulist_builder_) return;
    const int uc = rebuild_ulist();
    closure_count_ = uc;
    ctx_.u_list  = ulist_builder_->device_list();
    ctx_.u_count = uc;
}

void HEOMTrajectory::refresh_active_set_and_ulist(double threshold) {
    build_context();
    // Whole matrix, deliberately — do NOT "optimise" this into ctx_'s block.
    // Callers are apply_dipole() and snapshot_device(), and the 2DES driver
    // declares the next segment's block only just before propagate_steps:
    //     two_dim.cpp:252  apply_dipole_full_hierarchy()   <- we are in here
    //     two_dim.cpp:274  apply_block(traj, blk[0])       <- block updates here
    // So at this point ctx_ still carries the PREVIOUS segment's block while ρ
    // has just been moved into a new one. Scanning that stale block would read
    // exact zeros, yield zero norms, and mark every ADO but α=0 inactive —
    // silently. build_context() above refreshes the context, NOT the block;
    // that is exactly what makes the trap look safe.
    // Cost of the full sweep: once per segment against dozens of steps.
    perform_ado_filtering(threshold, 0, sys_.N(), 0, sys_.N());
    rebuild_ulist_into_ctx();
}

void HEOMTrajectory::activate_all_ados(HEOMContext& ctx) {
    CUDA_CHECK(cudaMemset(d_active_flags_.get(), 1,
                          static_cast<size_t>(sys_.M())));
    total_active_ados_ = sys_.M();
    // |A| = M ⇒ U = M, so the compact launch would only add a full-length
    // indirection table. Take the dense path (byte-equal, no indirection) —
    // the same choice the adaptive gate below would make at |E|/M = 1.
    if (ulist_builder_) {
        closure_count_ = sys_.M();
        ctx.u_list     = nullptr;
        ctx.u_count    = 0;
        ulist_dense_   = 1;
    }
}

void HEOMTrajectory::init_ulist_if_enabled() {
    if (!use_filtering_ || sys_.config().filter.ulist_tau <= 0.0) return;

    // Fully device-side builder: borrows SystemContext's GPU neighbor maps
    // directly (no D2H, no permanent host mirror — the old host builder
    // kept 2·M·channels·4 B of RAM alive for the whole run).
    ulist_builder_ = std::make_unique<UListBuilder>(
        sys_.M(), sys_.n_bath(), sys_.K(),
        sys_.d_plus_map(), sys_.d_minus_map());

    // τ comes from the config (filter.ulist, the single knob); the env var is
    // an optional override for parameter sweeps without editing the YAML.
    ulist_tau_ = sys_.config().filter.ulist_tau;
    if (const char* tau = std::getenv("HEOM_ULIST_TAU")) {
        if (tau[0] != '\0') ulist_tau_ = std::atof(tau);
    }

    std::cout << "[ulist] enabled; whole-step over U=E "
                 "(axpy+combine in-place, no swap, freeze a/=U), "
                 "adaptive gate=" << ulist_tau_ << "\n";
}

int HEOMTrajectory::rebuild_ulist() {
    if (!ulist_builder_) return sys_.M();
    return ulist_builder_->build(d_active_flags_.get());
}

void HEOMTrajectory::emit_ado_stats(const std::string& prefix) {
    // Read-only: recompute the per-level distribution from the CURRENT
    // active_flags (does NOT refilter / does NOT touch ρ or the active set),
    // so it is safe to call off the filter cadence (e.g. forced τ/T progress
    // lines from the 2DES driver). |A| and |E| are the most recent filter /
    // ulist-rebuild values held in members.
    compute_active_level_distribution(
        d_active_flags_.get(), sys_.d_level_indices(),
        sys_.M(), sys_.n_bath(), sys_.K(), sys_.L(),
        d_level_counts_.get(), h_level_counts_.data());

    // |A| derived from the distribution we just pulled back: every active ADO
    // lands in exactly one level bin, so Σ_lev counts == active count. This is
    // the only place that needs the host-side |A| (display + history), so the
    // per-step filter path no longer computes or transfers it.
    int active_total = 0;
    for (int lev = 0; lev <= sys_.L(); ++lev) active_total += h_level_counts_[lev];
    total_active_ados_ = active_total;

    const heom::StreamFormat fmt_guard(std::cout);

    std::cout << std::fixed << prefix;

    if (closure_count_ >= 0) {
        // U-list on: |A|/|E|/M with active% and closure% (|E|=A∪neighbors(A)).
        std::cout << "ADOs |A|/|E|/M = "
                  << total_active_ados_ << "/" << closure_count_ << "/" << sys_.M()
                  << " (" << std::setprecision(1)
                  << (100.0 * total_active_ados_ / sys_.M()) << "%, "
                  << (100.0 * closure_count_ / sys_.M()) << "%)";
    } else {
        // U-list off (no |E| computed): keep the original |A|/M line.
        std::cout << "Active ADOs = "
                  << total_active_ados_ << "/" << sys_.M()
                  << " (" << std::setprecision(1)
                  << (100.0 * total_active_ados_ / sys_.M()) << "%)";
    }

    std::cout << " [L0:" << h_level_counts_[0];
    for (int lev = 1; lev <= sys_.L(); ++lev) {
        if (h_level_counts_[lev] > 0) {
            std::cout << " L" << lev << ":" << h_level_counts_[lev];
        }
    }
    std::cout << "]\n";

    // Batched per-slot diagnostic (filter.log_per_slot, default off). The |A|
    // line above reads the last filter cadence (can be stale); slot_active_stats
    // recomputes union + per-slot from ONE fresh ρ snapshot, so the printed
    // union |A| is consistent with the slots, and r = union/max_b >= 1.
    // Off the hot path — only when the flag is set AND we are batching.
    if (use_filtering_ && active_batch_ > 1 && sys_.config().filter.log_per_slot) {
        const SlotActiveStats st = slot_active_stats();
        std::cout << prefix << "    \u2514 union|A|=" << st.union_count << " slots[";
        for (size_t b = 0; b < st.per_slot.size(); ++b)
            std::cout << st.per_slot[b] << (b + 1 < st.per_slot.size() ? "," : "");
        std::cout << "]/" << sys_.M() << "\n";
    }

    // U-list gate flip, as an indented continuation of the line just printed.
    // Only when the compact/full-grid mode actually changed since the last
    // emitted line (ulist_dense_ is the live state set in run_one_internal_step;
    // -1 means ulist off / before first build ⇒ no line). No numbers: |E|/M is
    // already the closure% above and the gate value is in the "[ulist] enabled"
    // banner, so the flip line carries only the decision.
    if (ulist_dense_ >= 0 && ulist_dense_ != last_logged_dense_) {
        std::cout << "    \u2514 U-list \u2192 "
                  << (ulist_dense_ == 1 ? "full-grid fallback" : "compact whole-step")
                  << "\n";
        last_logged_dense_ = ulist_dense_;
    }

}

HEOMTrajectory::SlotActiveStats HEOMTrajectory::slot_active_stats() {
    // One norm pass per live slot's sub-buffer; count |A_b| and OR into a union
    // mask on the host, all from the SAME ρ snapshot. Uses active_batch_ (live
    // width 3/9/21 in 2DES), NOT batch_ (allocated 21) — otherwise stale/idle
    // slots would be counted (T0a fix). Union and per-slot are thus temporally
    // consistent, so union_count >= max_b(per_slot).
    const int M = sys_.M();
    const size_t slot_stride =
        static_cast<size_t>(M) * sys_.N() * sys_.N();
    // Grow-only member scratch (see header): no per-call heap alloc, and the
    // union mask lives off the stack (GCC 12/13 -Wfree-nonheap-object FP).
    h_slot_norms_.resize(M);
    h_slot_union_.assign(M, 0);
    h_slot_union_[0] = 1;  // α=0 is always live
    SlotActiveStats st;
    st.per_slot.assign(active_batch_, 1);
    for (int b = 0; b < active_batch_; ++b) {
        // Whole matrix: this is the per-slot diagnostic (filter.log_per_slot,
        // off by default), not the hot path. Keeping it unblocked also makes it
        // an independent oracle for the blocked main path — both must produce
        // the same |A| from the same ρ.
        compute_ado_norms(d_rho_.get() + b * slot_stride, d_ado_norms_.get(),
                          M, sys_.N(), 0, sys_.N(), 0, sys_.N());
        d_ado_norms_.copy_to_host(h_slot_norms_.data(), M);
        for (int a = 1; a < M; ++a)
            if (h_slot_norms_[a] > filter_threshold_) {
                ++st.per_slot[b];
                h_slot_union_[a] = 1;
            }
    }
    for (int a = 0; a < M; ++a) st.union_count += h_slot_union_[a];
    return st;
}

void HEOMTrajectory::log_stats() {
    // Forced one-off stats line for the 2DES driver (segment-boundary progress,
    // e.g. each τ/T advance). No-op without filtering — the cadence path is
    // also filtering-gated, so the two stay consistent.
    if (use_filtering_) emit_ado_stats(log_prefix_);
}

void HEOMTrajectory::print_filtering_statistics(int step,
                                                double current_time_fs) {
    std::string prefix;
    if (log_prefix_.empty()) {
        // Dynamics / absorption: historical "Step N (t=cumulative fs):" line.
        std::ostringstream os;
        os << "Step " << std::setw(4) << step
           << " (t=" << std::setw(7) << std::fixed << std::setprecision(1)
           << current_time_fs << " fs): ";
        prefix = os.str();
    } else {
        // 2DES: the driver's prefix already carries the physical τ/T/t axis
        // coordinates (run_pathway_tree). The cumulative internal step counter
        // and step*dt "time" are meaningless across the depth-first τ→tw→t tree
        // (they also count discarded branches), so they are dropped here.
        prefix = log_prefix_;
    }

    emit_ado_stats(prefix);  // also refreshes h_level_counts_ for the history

    filtering_history_.push_back({
        step,
        current_time_fs,
        total_active_ados_,
        sys_.M(),
        h_level_counts_
    });
}

void HEOMTrajectory::run_step(HEOMContext& ctx, double dt_internal_au) {
    // rk4_step combines IN-PLACE into d_rho_ on both paths (full grid and
    // whole-step), so d_rho_ is always the current state. ctx.drho_out stays
    // null — the RK4 epilogues are fused into the derivative kernel, so no
    // k buffer exists at all (workspace = temp_a | temp_b | acc).
    ctx.rho_in = d_rho_.get();
    // d_kahan_comp_ is empty in the FP64 build ⇒ get() is nullptr ⇒ rk4_step
    // does the plain (bit-identical) combine. Non-null only in mixed precision.
    if (integrator_ == Integrator::IfRk4) {
        // Factors depend on dt only; rebuild on the first step (last_if_dt_<0)
        // or whenever the internal dt changes (τ vs t vs dynamics grids).
        if (dt_internal_au != last_if_dt_) {
            compute_if_factors(sys_.d_total_decay(), dt_internal_au, sys_.M(),
                               d_if_e_half_.get(), d_if_e_full_.get());
            last_if_dt_ = dt_internal_au;
        }
        rk4_step_if(ctx, dt_internal_au, d_workspace_.get(),
                    d_kahan_comp_.get(),
                    d_if_e_half_.get(), d_if_e_full_.get());
    } else {
        rk4_step(ctx, dt_internal_au, d_workspace_.get(), d_kahan_comp_.get());
    }
}

void HEOMTrajectory::report_divergence(double t_fs, double rho_norm,
                                       double norm_ceiling,
                                       int step, int n_steps) {
    const heom::StreamFormat fmt_guard(std::cerr);
    // Plain ASCII so the message survives stripped-down logs (CI capture,
    // tee | less, journald with non-UTF locale, ...).
    //
    // We report ‖ρ_0‖_F instead of Tr(ρ_0) because the divergence guard
    // is norm-based: trace alone misses traceless off-diagonal explosion
    // (e.g. absorption ρ(0) is off-diagonal, trace ≡ 0 even when the
    // trajectory blows up). ‖·‖_F catches both that case and the
    // trace-violating exponential growth that the old guard targeted.
    std::cerr << "\n"
              << "----------------------------------------------------\n"
              << "  DIVERGENCE DETECTED -- COMPUTATION ABORTED\n"
              << "----------------------------------------------------\n"
              << "  Time:           " << std::setprecision(6)  << t_fs       << " fs\n"
              << "  ||rho_0(t)||_F: " << std::setprecision(6)  << rho_norm   << "\n"
              << "  Threshold:      " << std::setprecision(6)  << norm_ceiling << "\n"
              << "  ||rho_0(0)||_F: " << std::setprecision(6)  << initial_norm_ << "\n"
              << "  Step:           " << step  << " / " << n_steps           << "\n"
              << "----------------------------------------------------\n"
              << "  Check: truncation level L, time step dt,\n"
              << "         Matsubara terms K, temperature.\n"
              << "----------------------------------------------------\n";
}

bool HEOMTrajectory::check_divergence(double t_fs, int step, int n_steps) {
    // Refresh the host α=0 mirror(s) so rho0_frobenius_norm() sees the current
    // device state (solve() also D2H's via its emit lambda; the extra copy here
    // is a few KB per slot and keeps this guard self-contained for callers like
    // propagate_steps() that do not). Batched ⇒ all B α=0 blocks.
    d2h_alpha0_batch();
    const double rho_norm  = rho0_frobenius_norm();
    const double norm_ceil = std::max(initial_norm_ * kDivergenceNormFactor,
                                      kDivergenceNormFloor);
    if (rho_norm > norm_ceil || std::isnan(rho_norm) || std::isinf(rho_norm)) {
        report_divergence(t_fs, rho_norm, norm_ceil, step, n_steps);
        diverged_ = true;
    }
    return diverged_;
}

void HEOMTrajectory::run_one_internal_step(HEOMContext& ctx, int step) {
    const auto& cfg = sys_.config();
    const double dt_internal_au = cfg.time.dt_au;
    const int    print_interval = cfg.time.print_interval;
    const double step_time_fs   = step * cfg.time.dt;

    // Warmup gating: relax the threshold while still inside
    // [0, filter_start_step) OR on the first-ever pass, so the warmup knob
    // can never silently no-op.
    const bool in_warmup = (step < cfg.filter.start_step)
                        || (filter_pass_index_ == 0);
    const double effective_threshold = in_warmup
        ? filter_threshold_ * cfg.filter.warmup_factor
        : filter_threshold_;

    // Unified cadence: filtering + U rebuild run EVERY step on both
    // paths — one semantics, no interval knob. Boundary ADOs (E\A) get their
    // Term 1-3 self-evolution the step after they cross the threshold, so an
    // ADO that is growing lags the full grid by one step. (This used to claim
    // the lag was "exactly like the full grid". It is not: one RK4 step applies
    // A four times, so the unfiltered hierarchy grows FOUR hops per step. The
    // lag is real, which is why it must not compound — see fresh_state_.)
    //
    // fresh_state_: ρ was just replaced (trajectory start, dipole, restore or
    // slot expand), so the hierarchy is not populated and the cold path that
    // replaced it may have left the flags saying so. Activate everything and let
    // THIS step populate the hierarchy; the next one filters real data.
    //
    // Marking active is never wrong, only slower. Filtering here is wrong at any
    // threshold: every α>0 norm is exactly 0, so all of them go inactive and
    // Term 4/5's neighbor gate then releases one level per step.
    //
    // This used to read `step > 0`, which covers the trajectory start (hence
    // dynamics/absorption) but not the 2DES dipoles. Keying off ρ replacement
    // covers all four cold paths at once — they all funnel through
    // on_rho_replaced().
    if (use_filtering_ && fresh_state_) {
        activate_all_ados(ctx);
    } else if (use_filtering_) {
        // Scan domain = the PREVIOUS step's write set (U-local contract).
        // ctx.u_list is exactly the launch set the last run_step wrote through
        // (nullptr = full grid: ulist off, dense fallback, or pre-first-build
        // full launch), and rebuild_ulist below overwrites the device list
        // only AFTER this scan — so reading it here is the correct domain.
        // Block and scan domain both come from the SAME ctx, so they cannot
        // disagree: whatever launch domain the last run_step wrote through is
        // the domain we scan, inside the block it wrote into.
        const bool prev_compact = ulist_builder_ && (ctx.u_list != nullptr);
        perform_ado_filtering(effective_threshold,
                              ctx.row_off, ctx.n_row, ctx.col_off, ctx.n_col,
                              prev_compact ? ctx.u_list  : nullptr,
                              prev_compact ? ctx.u_count : 0);
        ++filter_pass_index_;

        // Recompute the compact launch set from the just-updated active_flags.
        if (ulist_builder_) {
            const int ucount = rebuild_ulist();
            closure_count_ = ucount;   // |E| for the stats line (raw, pre-gate)
            ctx.u_list  = ulist_builder_->device_list();
            ctx.u_count = ucount;

            // Hard threshold: dense (|E|/M > gate) falls back to the full grid
            // (byte-equal, no u_list indirection); else compact whole-step. The
            // flip is NOT logged here — emit_ado_stats prints it as an indented
            // continuation of the stats line so it rides a real Step / τ·T·t
            // line (and carries no @step, meaningless in the 2DES tree).
            const double frac = static_cast<double>(ucount) / sys_.M();
            const bool dense = frac > ulist_tau_;
            if (dense) { ctx.u_list = nullptr; ctx.u_count = 0; }
            ulist_dense_ = dense ? 1 : 0;
        }
    }

    if (use_filtering_ && print_interval > 0 && cadence_logging_
        && (step % print_interval == 0))
        print_filtering_statistics(step, step_time_fs);

    // This step is the one that populates the fresh hierarchy, so from the next
    // one on there is real data to filter.
    fresh_state_ = false;
    run_step(ctx, dt_internal_au);
}

// Shared solve loop for both the single-state solve() and the batched
// solve_batch(). `emit` is the per-record sink (record_idx, t_fs); it owns the
// D2H + fan-out (writer / callback), so the loop stays fan-out-agnostic. The
// emit's D2H(via d2h_alpha0_batch) and the divergence guard (max over slots)
// both degenerate for B==1, so the single path runs byte-identical to history.
void HEOMTrajectory::run_solve_loop(
    const std::function<void(int, double)>& emit)
{
    std::cout << "=== Time Evolution ===\n";

    const auto& cfg = sys_.config();
    const double dt_output_fs    = cfg.time.dt;
    const int    n_output_steps  = cfg.time.n_steps;
    const int    output_interval = cfg.time.output_interval;

    // Formatting stated here, not inherited. This line used to print "1000.0 fs"
    // or "1000.000 fs" for the SAME run length depending on which initial-state
    // branch had printed last and what precision it leaked.
    {
        const heom::StreamFormat fmt_guard(std::cout);
        std::cout << "Running " << n_output_steps << " steps ("
                  << std::fixed << std::setprecision(1)
                  << (n_output_steps * dt_output_fs) << " fs)...\n";
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // t=0 goes through emit() so the record_idx=0 / writer / callback contract
    // is identical to every later step.
    emit(0, 0.0);

    // Build once and reuse the cached HEOM context. run_one_internal_step
    // mutates ctx_.u_list/u_count at filter steps through this reference.
    build_context();
    HEOMContext& ctx = ctx_;

    // record_idx == 0 was the t=0 emit above; main loop starts at record 1.
    int next_record_idx = 1;

    for (int step = 0; step < n_output_steps; ++step) {
        run_one_internal_step(ctx, step);

        const double output_time_fs = (step + 1) * dt_output_fs;

        if ((step + 1) % output_interval == 0) {
            emit(next_record_idx, output_time_fs);
            ++next_record_idx;

            // Norm-based divergence guard: catches both trace-violating
            // exponential growth (dynamics: ρ(0) trace=1) and traceless
            // off-diagonal explosion (spectroscopy: ρ(0)=μ⁺ρ_eq, trace≡0). A
            // trace-only guard gives false negatives on the absorption path.
            if (check_divergence(output_time_fs, step + 1, n_output_steps))
                break;
        }
    }

    // Only D2H if h_rho0_ doesn't already mirror the final ρ_0:
    // - n_steps % output_interval == 0: the last iteration emitted — skip.
    // - divergence: the in-loop emit ran before diverged_ was set — skip.
    // - loop ended on a non-output step without diverging: D2H so 2DES drivers
    //   can call final_rho0()/final_trace(). No callback fires (no record_idx).
    const bool last_was_output_step =
        (n_output_steps % output_interval == 0);
    if (!last_was_output_step && !diverged_) {
        d2h_alpha0_batch();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    simulation_time_seconds_ = elapsed.count();

    std::cout << "done. (computation time: "
              << heom::console::format_computation_time(simulation_time_seconds_) << ")\n\n";
}

void HEOMTrajectory::solve(const StepCallback& on_output_step) {
    if (batch_ != 1) {
        throw std::runtime_error(
            "HEOMTrajectory::solve(): batched trajectory (batch="
            + std::to_string(batch_) + "); use solve_batch()");
    }
    if (!initial_set_) {
        throw std::runtime_error(
            "HEOMTrajectory::solve() called before set_initial()");
    }

    if (writer_) {
        writer_->open(sys_, initial_rho_, use_filtering_, filter_threshold_);
    }

    // Lazy allocation of the std::complex mirror handed to StepCallback. The
    // dynamics CLI / 2DES enable_hdf5=false drivers pass no callback and pay
    // nothing here.
    const bool has_callback = static_cast<bool>(on_output_step);
    if (has_callback) {
        h_rho_alpha0_mirror_.resize(static_cast<size_t>(sys_.N()) * sys_.N());
    }

    // Unified per-record sink: one D2H of ρ_0, then fan out to the HDF5 writer
    // (if enabled) and the StepCallback (if registered).
    auto emit = [&](int record_idx, double t_fs) {
        const int N = sys_.N();
        const size_t N2 = static_cast<size_t>(N) * N;

        d2h_alpha0_batch();   // B==1 ⇒ single N² D2H into h_rho0_

        if (writer_) {
            // HDF5 is float64-only (schema unchanged): hand it the FP64 view.
            // FP64 build ⇒ zero-copy alias of h_rho0_; mixed ⇒ uplift.
            writer_->write_timestep(t_fs, rho0_as_fp64(), N);
        }

        if (has_callback) {
            // Elementwise hier → std::complex<double>, not reinterpret_cast:
            // layout compatibility is implementation-defined, and
            // spectra/dipole.hpp must stay CUDA-free. c_re/c_im promote the
            // hier element (float in mixed) into the double observable.
            for (size_t i = 0; i < N2; ++i) {
                h_rho_alpha0_mirror_[i] = std::complex<double>(
                    heom::c_re(h_rho0_[i]), heom::c_im(h_rho0_[i]));
            }
            on_output_step(record_idx, t_fs,
                           h_rho_alpha0_mirror_.data(), N);
        }
    };

    run_solve_loop(emit);
}

void HEOMTrajectory::solve_batch(const BatchStepCallback& on_output_step) {
    if (!initial_set_) {
        throw std::runtime_error(
            "HEOMTrajectory::solve_batch() called before set_initial_batch()");
    }
    // Absorption batching serves only the enable_hdf5=false in-memory drivers: the HDF5
    // schema is single-ρ0, so a batched write has nowhere to land. Fail loud
    // rather than silently drop B-1 slots.
    if (enable_hdf5_) {
        throw std::runtime_error(
            "HEOMTrajectory::solve_batch(): batching requires enable_hdf5=false "
            "(HDF5 schema is single-ρ0)");
    }
    // The emit lambda converts batch_ slots and hands batch_ to the callback,
    // but d2h_alpha0_batch() only refreshes active_batch_ slots. A narrowed
    // width would silently feed the callback stale bytes for slots ≥ active —
    // solve_batch requires the full allocated width.
    if (active_batch_ != batch_) {
        throw std::runtime_error(
            "HEOMTrajectory::solve_batch(): active_batch ("
            + std::to_string(active_batch_) + ") != batch ("
            + std::to_string(batch_) + "); reset via set_active_batch(batch())");
    }

    const bool has_callback = static_cast<bool>(on_output_step);
    if (has_callback) {
        h_rho_alpha0_mirror_.resize(
            static_cast<size_t>(batch_) * sys_.N() * sys_.N());
    }

    auto emit = [&](int record_idx, double t_fs) {
        const int N = sys_.N();
        const size_t tot = static_cast<size_t>(batch_) * N * N;

        d2h_alpha0_batch();   // fills h_rho0_[b·N²] for all B slots

        if (has_callback) {
            for (size_t i = 0; i < tot; ++i) {
                h_rho_alpha0_mirror_[i] = std::complex<double>(
                    heom::c_re(h_rho0_[i]), heom::c_im(h_rho0_[i]));
            }
            on_output_step(record_idx, t_fs,
                           h_rho_alpha0_mirror_.data(), N, batch_);
        }
    };

    run_solve_loop(emit);
}

void HEOMTrajectory::save_results() {
    const heom::StreamFormat fmt_guard(std::cout);
    if (writer_) {
        writer_->close(simulation_time_seconds_,
                       use_filtering_ ? &filtering_history_ : nullptr);
    }

    // solve() ends with a final D2H into h_rho0_, so the host mirror holds the
    // final ρ_0 here. No extra cudaMemcpy.
    const double trace = final_trace();

    if (diverged_) {
        // `fixed` stated explicitly: setprecision alone means SIGNIFICANT
        // digits, so an exact trace of 1 printed as "1", not "1.0000000000".
        // It only ever looked right because `fixed` had leaked in from a
        // previous printer.
        std::cerr << "Final trace: " << std::fixed << std::setprecision(10) << trace
                  << " (DIVERGED)\n";
    } else {
        std::cout << "Final trace: " << std::fixed << std::setprecision(6) << trace << "\n";
    }
    if (writer_) {
        std::cout << "Results saved to: " << writer_->get_output_path() << "\n";
    }
}

// ---------------------------------------------------------------------------
// build_initial_state — free function
// ---------------------------------------------------------------------------

// Pure builder: no console output (that lives in the callers), so it can be
// reused by response-function pathways without polluting stdout.
Eigen::MatrixXcd build_initial_state(const SystemContext& sys,
                                     const InitialState& init) {
    const int N = sys.N();

    switch (init.type) {
        case InitialState::Type::SITE_EXCITATION: {
            Eigen::MatrixXcd rho = Eigen::MatrixXcd::Zero(N, N);
            rho(init.excited_site, init.excited_site) = 1.0;
            return rho;
        }

        case InitialState::Type::EXCITON_EXCITATION: {
            const int idx = init.exciton_index;
            if (idx < 0 || idx >= N) {
                throw std::runtime_error("exciton_index "
                    + std::to_string(idx + 1) + " out of range [1, "
                    + std::to_string(N) + "]");
            }
            Eigen::MatrixXcd rho_exc = Eigen::MatrixXcd::Zero(N, N);
            rho_exc(idx, idx) = 1.0;
            return sys.eigenvectors() * rho_exc * sys.eigenvectors().adjoint();
        }

        case InitialState::Type::COHERENT: {
            Eigen::MatrixXcd rho = Eigen::MatrixXcd::Zero(N, N);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    rho(i, j) = init.coherent_coeffs[i]
                              * std::conj(init.coherent_coeffs[j]);
                }
            }
            return rho;
        }

        case InitialState::Type::CUSTOM_MATRIX:
            return init.custom_matrix;
    }

    throw std::runtime_error("Unknown initial state type");
}
