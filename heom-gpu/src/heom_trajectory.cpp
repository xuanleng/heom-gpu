#include "heom_trajectory.hpp"
#include "util/stream_format.hpp"
#include "core/cuda_common.cuh"
#include "core/rk4_integrator.cuh"
#include "core/ado_filter.cuh"
#include "core/dipole_apply.cuh"
#include "core/dressed_start.cuh"
#include "core/hierarchy_distance.cuh"
#include "core/precision.hpp"
#include "gpu_alloc_plan.hpp"
#include "memory_budget.hpp"
#include "ulist_builder.hpp"

#include <algorithm>    // std::fill (block expansion), std::max
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>      // std::swap (move-only GPUMemory ping-pong)

// Definition required because HEOMTrajectory holds std::unique_ptr<UListBuilder>
// (forward-declared in heom_trajectory.hpp); ~unique_ptr needs the complete
// type at the point the trajectory's destructor is emitted.
HEOMTrajectory::~HEOMTrajectory() = default;

HEOMTrajectory::HEOMTrajectory(const SystemContext& sys, int batch)
    : sys_(sys)
    , batch_(batch)
    , capacity_(sys.rho_capacity())
    , active_batch_(batch)
    , support_{}          // no ρ yet ⇒ no support; set_initial installs the first
    , use_filtering_(sys.config().filter.enabled)
    , filter_threshold_(sys.config().filter.tolerance)
    , integrator_(sys.config().time.integrator)
{
    if (batch_ < 1) {
        throw std::invalid_argument(
            "HEOMTrajectory: batch must be >= 1, got " + std::to_string(batch_));
    }
    // Enforce the M·Nb < 2^31 kernel index ceiling HERE, the one chokepoint
    // every propagation path passes through, at the largest Nb this run can
    // hold. The derivative launcher re-checks per launch, but the
    // dipole/trace kernels compute int totals without a guard — and in
    // 2DES the FIRST kernel to run is a dipole apply. Failing loud at
    // construction closes that window.
    (void)heom::checked_rho_elements(sys_.M(), capacity_);
    allocate_trajectory_buffers();
}

void HEOMTrajectory::allocate_trajectory_buffers() {
    // Batched runs (B>1) multiply the two O(M·N²) mutable buffers plus the Kahan
    // carry by B. SystemContext's pre-alloc budget check only saw the B=1
    // footprint, so re-check here with the real batch BEFORE cudaMalloc'ing B×
    // the biggest arrays. B==1 skips it (SystemContext already covered it).
    // SystemContext's tables are on the device by now: free memory already
    // excludes them.
    if (batch_ > 1) {
        auto budget = heom::estimate_budget(
            sys_.config(), static_cast<size_t>(sys_.M()),
            static_cast<size_t>(batch_), heom::Resident::SystemTables);
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
    // d_rho / d_workspace inside the plan (read-only structure stays ×1);
    // capacity_ is the per-ADO element count they are sized for.
    const auto plan = heom::plan_gpu_buffers(
        static_cast<size_t>(sys_.M()), sys_.N(), sys_.D(),
        sys_.n_bath(), sys_.L(), use_filtering_, static_cast<size_t>(batch_),
        static_cast<size_t>(capacity_));

    // typed_alloc throws std::logic_error if the plan's elem_bytes for the
    // named entry doesn't match sizeof(T), so a future change to either
    // side (this call site's GPUMemory<T> or gpu_alloc_plan.hpp's
    // sizeof(...) literal) cannot silently under/over-allocate.
    d_rho_       = heom::typed_alloc<heom::complex_t>(plan, "d_rho");
    d_workspace_ = heom::typed_alloc<heom::complex_t>(plan, "d_workspace");
    d_dist_partials_ = heom::typed_alloc<double>(plan, "d_dist_partials");
    h_dist_partials_.resize(d_dist_partials_.size());

    // Kahan carry for the mixed-precision RK4 combine: same shape
    // as ρ (batch·M·Nb_cap), allocated + zeroed ONCE here and persisted for the
    // run. FP64 build skips it entirely (empty ⇒ rk4_step gets nullptr ⇒ plain
    // combine ⇒ bit-identical to history). Not in the alloc plan because it is a
    // precision-mode-only cost; memory_budget.cpp accounts for it separately.
    if constexpr (heom::kMixedPrecision) {
        d_kahan_comp_ = GPUMemory<heom::complex_t>(
            static_cast<size_t>(batch_) * sys_.M() * capacity_);
        d_kahan_comp_.zero();
    }

    // IF-RK4 factor tables [M] (E/E2). Allocated ONLY on the IfRk4 path so the
    // default rk4 path pays zero extra GPU memory. Values are filled lazily on
    // the first run_step (dt is a run-time property), not here.
    if (integrator_ == Integrator::IfRk4) {
        d_if_e_half_ = GPUMemory<heom::complex_t>(static_cast<size_t>(sys_.M()));
        d_if_e_full_ = GPUMemory<heom::complex_t>(static_cast<size_t>(sys_.M()));
    }

    if (use_filtering_) {
        d_ado_norms_    = heom::typed_alloc<double>(plan, "d_ado_norms");
        d_active_flags_ = heom::typed_alloc<int8_t>(plan, "d_active_flags");
        d_level_counts_ = heom::typed_alloc<int>(plan, "d_level_counts");
        d_nan_flag_     = GPUMemory<int>(1);   // 4 bytes; not worth a plan entry
        d_nan_flag_.zero();
        d_zero_count_   = GPUMemory<int>(static_cast<size_t>(batch_));
        h_zero_counts_.resize(batch_);

        h_level_counts_.resize(d_level_counts_.size());

        // All ADOs of every slot active until the first filter pass.
        // active_flags[b·M + 0] is re-asserted every pass by the filter kernel
        // (rho_0 active for life), so this is only a default, not a
        // correctness dependency.
        CUDA_CHECK(cudaMemset(d_active_flags_.get(), 1, d_active_flags_.size()));
        total_active_ados_ = sys_.M();
    }

    // Host mirror of the α=0 block, one per batch slot ([B][N²], slot-major,
    // FULL matrix). d2h_alpha0_batch() pulls the B compact α=0 blocks (spaced
    // M·Nb apart) into h_pack_ and expands them here with +0 outside the block.
    h_rho0_.resize(static_cast<size_t>(batch_) * sys_.N() * sys_.N());
    h_pack_.resize(static_cast<size_t>(batch_) * capacity_);
    norm0_.assign(batch_, 0.0);
    fresh_state_.assign(batch_, 1);
    fresh_zeros_.assign(batch_, -1);
}

void HEOMTrajectory::on_rho_replaced() {
    // Empty in the FP64 build (the kernel gets nullptr and runs the plain
    // combine), so the clear costs nothing there.
    if (d_kahan_comp_.size() > 0) d_kahan_comp_.zero();
    fresh_state_.assign(batch_, 1);
    fresh_zeros_.assign(batch_, -1);
    norms_fused_ = false;
}

namespace {
void copy_d2d(heom::complex_t* dst, const heom::complex_t* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(heom::complex_t),
                          cudaMemcpyDeviceToDevice));
}

// Host-side check that `r` is exactly 0.0 outside `blk`. Returns the first
// offending (i, j) or nullopt. -0.0 == 0.0, so signed zeros pass; anything
// merely small does not — the block is the storage shape, and whatever is out
// there would simply be dropped.
std::optional<std::pair<int, int>> first_nonzero_outside(
    const Eigen::MatrixXcd& r, const heom::ManifoldBlock& blk) {
    const int N = static_cast<int>(r.rows());
    for (int i = 0; i < N; ++i) {
        const bool row_in = i >= blk.row_off && i < blk.row_off + blk.n_row;
        for (int j = 0; j < N; ++j) {
            const bool col_in = j >= blk.col_off && j < blk.col_off + blk.n_col;
            if (row_in && col_in) continue;
            if (r(i, j) != std::complex<double>(0.0, 0.0)) return std::make_pair(i, j);
        }
    }
    return std::nullopt;
}

std::string block_str(const heom::ManifoldBlock& b) {
    return "rows [" + std::to_string(b.row_off) + "," + std::to_string(b.row_off + b.n_row)
         + ") x cols [" + std::to_string(b.col_off) + ","
         + std::to_string(b.col_off + b.n_col) + ")";
}
}  // namespace

void HEOMTrajectory::set_support_(const heom::ManifoldBlock& blk, const char* who) {
    blk.validate(sys_.N(), who);
    if (blk.elems() > capacity_) {
        throw std::invalid_argument(
            std::string(who) + ": block " + block_str(blk) + " has "
            + std::to_string(blk.elems()) + " elements, but this trajectory was "
            "allocated for at most " + std::to_string(capacity_)
            + " per ADO (rho_capacity of the config's task)");
    }
    support_ = blk;
    if (ctx_built_) ctx_.block = support_;
}

void HEOMTrajectory::expand_block_(const heom::complex_t* packed,
                                   heom::complex_t* full) const {
    // Compact [n_row][n_col] → full [N][N] with exact +0 outside the block.
    // Same value the old zero-initialised full storage held there, so every
    // host consumer of h_rho0_ (norm, HDF5, callbacks) is bit-identical.
    const int N = sys_.N();
    std::fill(full, full + static_cast<size_t>(N) * N, heom::c_make(0.0, 0.0));
    for (int r = 0; r < support_.n_row; ++r)
        for (int c = 0; c < support_.n_col; ++c)
            full[(support_.row_off + r) * N + (support_.col_off + c)] =
                packed[r * support_.n_col + c];
}

void HEOMTrajectory::d2h_alpha0_batch() {
    const int    N   = sys_.N();
    const size_t N2  = static_cast<size_t>(N) * N;
    const size_t Nb  = static_cast<size_t>(support_.elems());
    const size_t Nbb = Nb * sizeof(heom::complex_t);
    const size_t slot = static_cast<size_t>(sys_.M()) * Nbb;
    CUDA_CHECK(cudaMemcpy2D(
        /*dst=*/h_pack_.data(), /*dpitch=*/Nbb,
        /*src=*/d_rho_.get(),   /*spitch=*/slot,
        /*width=*/Nbb, /*height=*/static_cast<size_t>(active_batch_),
        cudaMemcpyDeviceToHost));
    for (int b = 0; b < active_batch_; ++b)
        expand_block_(h_pack_.data() + static_cast<size_t>(b) * Nb,
                      h_rho0_.data() + static_cast<size_t>(b) * N2);
}

void HEOMTrajectory::set_initial(const std::vector<Eigen::MatrixXcd>& rho0s,
                                 const heom::ManifoldBlock& support) {
    // The width IS the number of matrices handed in: the 2DES walk sets 3 on a
    // batch=21 trajectory and the τ layer runs at 3.
    const int B = static_cast<int>(rho0s.size());
    if (B < 1 || B > batch_) {
        throw std::invalid_argument(
            "HEOMTrajectory::set_initial: expected 1.." + std::to_string(batch_)
            + " initial matrices, got " + std::to_string(B));
    }
    const int N = sys_.N();

    // All host-side checks before any state change, so a rejected call leaves
    // the trajectory exactly as it was.
    support.validate(N, "HEOMTrajectory::set_initial");
    for (int b = 0; b < B; ++b) {
        const Eigen::MatrixXcd& r = rho0s[b];
        if (r.rows() != N || r.cols() != N) {
            throw std::runtime_error(
                "HEOMTrajectory::set_initial: rho0[" + std::to_string(b)
                + "] size mismatch (expected " + std::to_string(N) + "x"
                + std::to_string(N) + ")");
        }
        if (const auto ij = first_nonzero_outside(r, support)) {
            std::ostringstream os;
            os << "HEOMTrajectory::set_initial: rho0[" << b << "](" << ij->first
               << "," << ij->second << ") = " << r(ij->first, ij->second)
               << " lies outside the declared support " << block_str(support)
               << "; it would be dropped (the block is the storage shape)";
            throw std::invalid_argument(os.str());
        }
    }

    // Shape first: the block is the storage shape, so it is installed —
    // and checked against capacity_ — before a single element is written.
    set_support_(support, "HEOMTrajectory::set_initial");
    // Zero the whole tensor (all slots, so idle slots > active are clean), then
    // pack each active slot's ρ_0 block into h_pack_ [B][Nb] and scatter the
    // active α=0 blocks in one cudaMemcpy2D (slot pitch M·Nb).
    set_width_(B);
    d_rho_.zero();
    const size_t Nb = static_cast<size_t>(support_.elems());
    for (int b = 0; b < active_batch_; ++b) {
        const Eigen::MatrixXcd& r = rho0s[b];
        heom::complex_t* pk = h_pack_.data() + static_cast<size_t>(b) * Nb;
        for (int lr = 0; lr < support_.n_row; ++lr)
            for (int lc = 0; lc < support_.n_col; ++lc) {
                const auto v = r(support_.row_off + lr, support_.col_off + lc);
                pk[lr * support_.n_col + lc] = heom::c_make(
                    static_cast<heom::real_t>(v.real()),
                    static_cast<heom::real_t>(v.imag()));
            }
        norm0_[b] = r.norm();
    }

    const size_t Nbb  = Nb * sizeof(heom::complex_t);
    const size_t slot = static_cast<size_t>(sys_.M()) * Nbb;
    CUDA_CHECK(cudaMemcpy2D(
        /*dst=*/d_rho_.get(), /*dpitch=*/slot,
        /*src=*/h_pack_.data(), /*spitch=*/Nbb,
        /*width=*/Nbb, /*height=*/static_cast<size_t>(active_batch_),
        cudaMemcpyHostToDevice));

    on_rho_replaced();
    initial_set_       = true;
    internal_step_     = 0;
    ulist_mode_        = UlistMode::Unset;
}

double HEOMTrajectory::rho0_norm_(int b) const {
    const size_t N2 = static_cast<size_t>(sys_.N()) * sys_.N();
    const heom::complex_t* r = h_rho0_.data() + static_cast<size_t>(b) * N2;
    double sum_sq = 0.0;
    for (size_t k = 0; k < N2; ++k) {
        const double re = heom::c_re(r[k]);
        const double im = heom::c_im(r[k]);
        sum_sq += re * re + im * im;
    }
    return std::sqrt(sum_sq);
}

Eigen::MatrixXcd HEOMTrajectory::slot_rho0(int b) const {
    if (b < 0 || b >= batch_) {
        throw std::out_of_range(
            "HEOMTrajectory::slot_rho0: slot " + std::to_string(b)
            + " out of range [0, " + std::to_string(batch_) + ")");
    }
    const int N = sys_.N();
    const size_t Nb = static_cast<size_t>(support_.elems());
    std::vector<heom::complex_t> packed(Nb);
    std::vector<heom::complex_t> full(static_cast<size_t>(N) * N);
    // slot b's α=0 block sits at offset b·M·Nb (α=0 is the first ADO).
    CUDA_CHECK(cudaMemcpy(packed.data(),
                          d_rho_.get() + static_cast<size_t>(b) * sys_.M() * Nb,
                          Nb * sizeof(heom::complex_t), cudaMemcpyDeviceToHost));
    expand_block_(packed.data(), full.data());
    Eigen::MatrixXcd out(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            out(i, j) = std::complex<double>(heom::c_re(full[i * N + j]),
                                             heom::c_im(full[i * N + j]));
    return out;
}

void HEOMTrajectory::build_context() {
    if (ctx_built_) return;
    const auto& cfg = sys_.config();
    ctx_.hamiltonian        = sys_.d_hamiltonian();
    ctx_.c_bath             = sys_.d_c_bath();
    ctx_.cbar_bath          = sys_.d_cbar_bath();
    ctx_.coeff_plus         = sys_.d_coeff_plus();
    ctx_.coeff_minus        = sys_.d_coeff_minus();
    ctx_.markov_residue     = sys_.d_markov_residue();
    ctx_.neighbor_map_plus  = sys_.d_plus_map();
    ctx_.neighbor_map_minus = sys_.d_minus_map();
    ctx_.level_indices      = sys_.d_level_indices();
    ctx_.total_decay        = sys_.d_total_decay();
    ctx_.active_flags       = use_filtering_ ? d_active_flags_.get() : nullptr;
    ctx_.d_coupling         = sys_.d_d_coupling();
    ctx_.mode_off           = sys_.d_mode_off();
    ctx_.M = sys_.M();  ctx_.N = sys_.N();  ctx_.D = sys_.D();
    ctx_.n_bath       = sys_.n_bath();
    ctx_.batch        = active_batch_;   // grid.y width; set_width_ mutates this
                                         // too (build_context is idempotent)
    // The kernel gets a real rectangle, never a sentinel to decode (see
    // HEOMContext::block). Both writers of ctx_.block — here and set_support_
    // — copy support_ verbatim; there is no other block to launch over.
    ctx_.block = support_;
    ctx_.sink_enabled = cfg.physics.sink.enabled;
    ctx_.sink_site    = cfg.physics.sink.site;
    ctx_.gamma_sink   = cfg.physics.sink.gamma;

    // Closure builder (filter on), and with filter.ulist on the compact launch:
    // point ctx_ at the identity list; the first filter step swaps in the
    // real closure set.
    init_closure_builder();
    if (ulist_on()) {
        ctx_.u_list  = ulist_builder_->device_list();
        ctx_.u_count = ulist_builder_->count();
    }
    ctx_built_ = true;
}

void HEOMTrajectory::set_width_(int w) {
    if (w < 1 || w > batch_) {
        throw std::invalid_argument(
            "HEOMTrajectory: active width must be in [1, "
            + std::to_string(batch_) + "], got " + std::to_string(w));
    }
    active_batch_ = w;
    // build_context() is guarded by ctx_built_ and will not re-run, so update
    // the live launch width directly. If ctx_ is not built yet, build_context()
    // will pick up active_batch_ on its first (and only) run.
    if (ctx_built_) ctx_.batch = w;
}

void HEOMTrajectory::propagate_steps(int n_internal_steps, bool check_div) {
    if (!initial_set_) {
        throw std::runtime_error(
            "HEOMTrajectory::propagate_steps() called before set_initial()");
    }
    // diverged_ latches across a whole 2DES tree (set_initial does not clear
    // it), so once tripped the remaining propagation short-circuits and the
    // driver's diverged() check aborts the run instead of letting NaN/Inf
    // silently fill the response grid.
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
}

bool HEOMTrajectory::check_divergence_now(double t_fs, int step, int n_total) {
    if (diverged_) return true;
    return check_divergence(t_fs, step, n_total);
}

void HEOMTrajectory::read_alpha0(std::vector<std::complex<double>>& out) {
    if (!initial_set_) {
        throw std::runtime_error(
            "HEOMTrajectory::read_alpha0() called before set_initial()");
    }
    const size_t tot = static_cast<size_t>(active_batch_) * sys_.N() * sys_.N();
    d2h_alpha0_batch();   // fills h_rho0_[b·N²] for the active slots
    out.resize(tot);
    // Elementwise hier → std::complex<double>, not reinterpret_cast: layout
    // compatibility is implementation-defined, and consumers must stay
    // CUDA-free. c_re/c_im promote the hier element (float in mixed) into the
    // double observable — the uplift boundary.
    for (size_t i = 0; i < tot; ++i)
        out[i] = std::complex<double>(heom::c_re(h_rho0_[i]), heom::c_im(h_rho0_[i]));
}

void HEOMTrajectory::read_system_bath_energy(std::vector<std::complex<double>>& out) {
    if (!initial_set_) {
        throw std::runtime_error(
            "HEOMTrajectory::read_system_bath_energy() called before set_initial()");
    }

    const int n_bath = sys_.n_bath();
    const int N      = sys_.N();
    const int M      = sys_.M();
    const int B      = active_batch_;
    const size_t Nb  = static_cast<size_t>(support_.elems());
    const size_t slot = static_cast<size_t>(M) * Nb;
    const auto& ft = sys_.first_tier();
    // Site occupation n̂_n = L_n from build_d_coupling (cached on SystemContext):
    // 1-ex projector on EXCITED_ONLY / WITH_GROUND; on DOUBLE_EXCITED also the
    // f-states that contain pigment n (the bath kernel's number operator).
    const std::vector<int8_t>& dcoup = sys_.d_coupling_host();

    out.assign(static_cast<size_t>(B) * n_bath, {0.0, 0.0});
    if (ft.empty() || Nb == 0) return;

    std::vector<heom::complex_t> packed(Nb);
    std::vector<heom::complex_t> full(static_cast<size_t>(N) * N);

    for (int b = 0; b < B; ++b) {
        const heom::complex_t* slot_ptr =
            d_rho_.get() + static_cast<size_t>(b) * slot;
        for (const auto& e : ft) {
            if (e.alpha < 0 || e.alpha >= M)
                throw std::logic_error("first_tier alpha out of range");
            if (e.site < 0 || e.site >= n_bath)
                throw std::logic_error("first_tier site out of range");
            CUDA_CHECK(cudaMemcpy(
                packed.data(),
                slot_ptr + static_cast<size_t>(e.alpha) * Nb,
                Nb * sizeof(heom::complex_t),
                cudaMemcpyDeviceToHost));
            expand_block_(packed.data(), full.data());
            // Formula uses √s_k; FirstTierAdo::scale stores s_k = √(|c||c̄|).
            const double sqrt_s = std::sqrt(e.scale);
            std::complex<double> tr(0.0, 0.0);
            const int8_t* drow = dcoup.data() + static_cast<size_t>(e.site) * N;
            for (int i = 0; i < N; ++i) {
                if (drow[i] == 0) continue;
                const heom::complex_t z = full[static_cast<size_t>(i) * N + i];
                tr += static_cast<double>(drow[i])
                    * std::complex<double>(heom::c_re(z), heom::c_im(z));
            }
            out[static_cast<size_t>(b) * n_bath + e.site] += sqrt_s * tr;
        }
    }
}

void HEOMTrajectory::apply_dipole(
    const std::vector<Eigen::MatrixXcd>& mus, heom::spectra::DipoleSide side,
    const heom::ManifoldBlock& support_after) {
    if (!initial_set_) {
        throw std::runtime_error(
            "HEOMTrajectory::apply_dipole() before set_initial()");
    }
    if (static_cast<int>(mus.size()) != active_batch_) {
        throw std::runtime_error(
            "apply_dipole: expected " + std::to_string(active_batch_)
            + " dipoles (active_batch_), got " + std::to_string(mus.size()));
    }
    const int N = sys_.N();
    const int M = sys_.M();
    const size_t N2 = static_cast<size_t>(N) * N;
    support_after.validate(N, "HEOMTrajectory::apply_dipole");
    const bool left = (side == heom::spectra::DipoleSide::Left);
    for (int b = 0; b < active_batch_; ++b) {
        if (mus[b].rows() != N || mus[b].cols() != N) {
            throw std::runtime_error(
                "apply_dipole: mu[" + std::to_string(b)
                + "] size mismatch (expected " + std::to_string(N) + "x"
                + std::to_string(N) + ")");
        }
        // Left: (μρ)(i,j) = Σ_k μ(i,k) ρ(k,j) — columns stay, rows move to
        // {i : μ(i,k) ≠ 0 for some k in the current rows R}. Right: (ρμ)(i,j)
        // = Σ_k ρ(i,k) μ(k,j) — rows stay, columns move likewise from C. So:
        // the fixed axis must be unchanged, and along the moving axis every
        // nonzero of μ reachable from the current range must land in the
        // declared one. Only the reachable k-slice of μ is scanned.
        const heom::ManifoldBlock& S = support_;
        const heom::ManifoldBlock& T = support_after;
        const bool fixed_ok = left ? (T.col_off == S.col_off && T.n_col == S.n_col)
                                   : (T.row_off == S.row_off && T.n_row == S.n_row);
        if (!fixed_ok) {
            throw std::invalid_argument(
                std::string("HEOMTrajectory::apply_dipole: a ") + (left ? "Left" : "Right")
                + " dipole keeps the " + (left ? "columns" : "rows")
                + " of the support, but " + block_str(S) + " -> " + block_str(T)
                + " changes them");
        }
        const auto& mu = mus[b];
        for (int k = left ? S.row_off : S.col_off,
                 k_end = left ? S.row_off + S.n_row : S.col_off + S.n_col;
             k < k_end; ++k) {
            for (int i = 0; i < N; ++i) {
                const std::complex<double> v = left ? mu(i, k) : mu(k, i);
                if (v == std::complex<double>(0.0, 0.0)) continue;
                const bool inside = left ? (i >= T.row_off && i < T.row_off + T.n_row)
                                         : (i >= T.col_off && i < T.col_off + T.n_col);
                if (inside) continue;
                std::ostringstream os;
                os << "HEOMTrajectory::apply_dipole: mu[" << b << "]("
                   << (left ? i : k) << "," << (left ? k : i) << ") = " << v
                   << " maps the current support " << block_str(S)
                   << " outside the declared support " << block_str(T)
                   << " (" << (left ? "Left" : "Right") << " side)";
                throw std::invalid_argument(os.str());
            }
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

    // Shape first: install and capacity-check the new block before the
    // kernel writes it. The kernel still needs the OLD block to read ρ, so it
    // is copied out first; nothing reads d_rho_ between here and the D2D copy.
    const heom::ManifoldBlock S_old = support_;
    set_support_(support_after, "HEOMTrajectory::apply_dipole");

    // Batched rectangular multiply S→T into the RK4 scratch (idle here), then
    // D2D copy the active prefix back. Slot b uses mu[b·N²], reads b·M·Nb_S,
    // writes b·M·Nb_T; both fit: Nb ≤ capacity_ and the workspace is 3× ρ.
    const int s = left ? 0 : 1;
    const size_t MNb_T = static_cast<size_t>(M) * support_.elems();
    launch_apply_single_side_mu(d_rho_.get(), d_workspace_.get(),
                                d_mu_scratch_.get(), s, M, N,
                                S_old, support_, active_batch_);
    CUDA_CHECK(cudaMemcpy(d_rho_.get(), d_workspace_.get(),
                          static_cast<size_t>(active_batch_) * MNb_T
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
    for (int b = 0; b < active_batch_; ++b) norm0_[b] = rho0_norm_(b);
}

void HEOMTrajectory::snapshot_device(DeviceHierarchySnapshot& snap) {
    if (!initial_set_) {
        throw std::runtime_error(
            "HEOMTrajectory::snapshot_device() before set_initial()");
    }
    const size_t M = static_cast<size_t>(sys_.M());
    snap.sys     = &sys_;
    snap.batch   = active_batch_;
    snap.support = support_;
    snap.fresh.assign(fresh_state_.begin(), fresh_state_.begin() + active_batch_);
    snap.fresh_zeros.assign(fresh_zeros_.begin(), fresh_zeros_.begin() + active_batch_);
    snap.norm0.assign(norm0_.begin(), norm0_.begin() + active_batch_);

    const size_t cap = static_cast<size_t>(active_batch_) * M * capacity_;
    const size_t n   = static_cast<size_t>(active_batch_) * M * support_.elems();
    snap.rho.ensure(cap);
    copy_d2d(snap.rho.get(), d_rho_.get(), n);
    if (d_kahan_comp_.size() > 0) {
        snap.kahan.ensure(cap);
        copy_d2d(snap.kahan.get(), d_kahan_comp_.get(), n);
    }
    if (use_filtering_) {
        // One flag row per active slot (per-slot active sets).
        const size_t nf = static_cast<size_t>(active_batch_) * M;
        snap.flags.ensure(nf);
        CUDA_CHECK(cudaMemcpy(snap.flags.get(), d_active_flags_.get(), nf,
                              cudaMemcpyDeviceToDevice));
    }
}

void HEOMTrajectory::check_snapshot_(const DeviceHierarchySnapshot& snap,
                                     int width, const char* who) const {
    if (snap.sys != &sys_) {
        throw std::invalid_argument(
            std::string(who) + (snap.sys ? ": snapshot was taken on another "
                                           "SystemContext"
                                         : ": snapshot was never filled"));
    }
    if (width < 1 || width > batch_) {
        throw std::invalid_argument(
            std::string(who) + ": " + std::to_string(width) + " slots requested, "
            "this trajectory has " + std::to_string(batch_));
    }
}

// Shape and width are installed (set_support_ / set_width_) before ρ is
// written, so ctx_.block and ctx_.batch describe the copied state on the next
// launch; neither can throw once check_snapshot_ has passed.
//
// After the copy the flags are the snapshot's. The U-list is not: the builder
// still holds the last branch's list. Pointing ctx_ at the dense launch makes
// the next filter pass scan all M, which by the U-local contract (see
// perform_ado_filtering) yields exactly the flags the compact scan would have,
// and that pass rebuilds U from them.
void HEOMTrajectory::restore_device(const DeviceHierarchySnapshot& snap) {
    check_snapshot_(snap, snap.batch, "HEOMTrajectory::restore_device");
    const size_t M = static_cast<size_t>(sys_.M());
    set_support_(snap.support, "HEOMTrajectory::restore_device");
    set_width_(snap.batch);
    const size_t n = static_cast<size_t>(snap.batch) * M * support_.elems();
    copy_d2d(d_rho_.get(), snap.rho.get(), n);
    if (d_kahan_comp_.size() > 0) copy_d2d(d_kahan_comp_.get(), snap.kahan.get(), n);
    if (use_filtering_)
        CUDA_CHECK(cudaMemcpy(d_active_flags_.get(), snap.flags.get(),
                              static_cast<size_t>(snap.batch) * M,
                              cudaMemcpyDeviceToDevice));
    if (ctx_built_) { ctx_.u_list = nullptr; ctx_.u_count = 0; ctx_.u_flags = nullptr; ctx_.u_off = nullptr; ctx_.block_off = nullptr; ctx_.launch_blocks = 0; }
    std::copy(snap.fresh.begin(), snap.fresh.end(), fresh_state_.begin());
    std::copy(snap.fresh_zeros.begin(), snap.fresh_zeros.end(), fresh_zeros_.begin());
    norms_fused_ = false;
    std::copy(snap.norm0.begin(), snap.norm0.end(), norm0_.begin());
    initial_set_ = true;
}

void HEOMTrajectory::expand_slots(const DeviceHierarchySnapshot& parent,
                                  const std::vector<int>& child_to_parent) {
    const int B_child = static_cast<int>(child_to_parent.size());
    check_snapshot_(parent, B_child, "HEOMTrajectory::expand_slots");
    for (int c = 0; c < B_child; ++c) {
        const int p = child_to_parent[c];
        if (p < 0 || p >= parent.batch) {
            throw std::invalid_argument(
                "HEOMTrajectory::expand_slots: child_to_parent[" + std::to_string(c)
                + "]=" + std::to_string(p) + " out of range [0, "
                + std::to_string(parent.batch) + ")");
        }
    }
    const size_t M = static_cast<size_t>(sys_.M());
    set_support_(parent.support, "HEOMTrajectory::expand_slots");
    set_width_(B_child);
    const size_t MNb = M * support_.elems();
    for (int c = 0; c < B_child; ++c) {
        const int p = child_to_parent[c];
        copy_d2d(d_rho_.get() + c * MNb, parent.rho.get() + p * MNb, MNb);
        if (d_kahan_comp_.size() > 0)
            copy_d2d(d_kahan_comp_.get() + c * MNb, parent.kahan.get() + p * MNb, MNb);
        norm0_[c] = parent.norm0[p];
        // Child c inherits its parent's own active set and fill state.
        if (use_filtering_)
            CUDA_CHECK(cudaMemcpy(d_active_flags_.get() + c * M,
                                  parent.flags.get() + p * M, M,
                                  cudaMemcpyDeviceToDevice));
        fresh_state_[c] = parent.fresh[p];
        fresh_zeros_[c] = parent.fresh_zeros[p];
    }
    if (ctx_built_) { ctx_.u_list = nullptr; ctx_.u_count = 0; ctx_.u_flags = nullptr; ctx_.u_off = nullptr; ctx_.block_off = nullptr; ctx_.launch_blocks = 0; }
    norms_fused_ = false;
    initial_set_ = true;
}

double HEOMTrajectory::relative_distance(const DeviceHierarchySnapshot& snap) {
    check_snapshot_(snap, snap.batch, "HEOMTrajectory::relative_distance");
    if (snap.batch != active_batch_ || snap.support != support_) {
        throw std::invalid_argument(
            "HEOMTrajectory::relative_distance: snapshot holds "
            + std::to_string(snap.batch) + " slot(s) on " + block_str(snap.support)
            + ", the trajectory " + std::to_string(active_batch_) + " on "
            + block_str(support_));
    }
    const size_t n = static_cast<size_t>(active_batch_) * sys_.M() * support_.elems();
    const bool carry = d_kahan_comp_.size() > 0;
    const int blocks = heom::launch_hierarchy_distance(
        d_rho_.get(), carry ? d_kahan_comp_.get() : nullptr,
        snap.rho.get(), carry ? snap.kahan.get() : nullptr,
        n, d_dist_partials_.get());
    d_dist_partials_.copy_to_host(h_dist_partials_.data(), 2 * static_cast<size_t>(blocks));
    double dist = 0.0, norm = 0.0;
    for (int b = 0; b < blocks; ++b) {
        dist += h_dist_partials_[2 * b];
        norm += h_dist_partials_[2 * b + 1];
    }
    if (norm == 0.0) {
        throw std::runtime_error(
            "HEOMTrajectory::relative_distance: the hierarchy is identically zero");
    }
    return std::sqrt(dist / norm);
}

void HEOMTrajectory::dress_ados() {
    if (!initial_set_) {
        throw std::logic_error("HEOMTrajectory::dress_ados() before set_initial()");
    }
    build_context();
    for (int tier = 1; tier <= sys_.L(); ++tier)
        launch_dress_tier(ctx_, d_rho_.get(), tier);
    on_rho_replaced();
    if (use_filtering_) activate_all_ados(ctx_);
}

void HEOMTrajectory::perform_ado_filtering(double threshold,
                                           const heom::ManifoldBlock& blk,
                                           const int* scan_list,
                                           int scan_count,
                                           const int* scan_off) {
    // U-local contract: the scan domain MUST cover the previous
    // write set of ρ. New threshold crossings can only happen where ρ was
    // written; everything outside kept a frozen sub-threshold norm and an
    // inactive flag (A ⊆ U), so skipping it is exact — not an approximation.
    //   scan_list == nullptr : full-M sweep (previous write set was the full
    //                          grid: dense fallback, ulist off; or unknown:
    //                          the first step after restore / expand). A
    //                          superset of the write set gives the same flags:
    //                          everything outside it was inactive at its last
    //                          scan, is unchanged since, and the threshold
    //                          is fixed for the run.
    //   scan_list != nullptr : compact sweep over the previous whole-step's U.
    // Batched (active_batch_>1): every active slot is filtered on its OWN
    // state into its own flag row (per-slot active sets, 2026-10-22-batch-
    // per-slot; this replaced the union of all slots, which made every slot
    // compute every other slot's ADOs). scan_off gives each slot its own
    // previous write set. Only active_batch_ slots hold live state.
    // active_batch_==1 ⇒ ordinary single-state norms.
    // Manifold block: it is the storage shape, so the Nb elements it names are
    // the whole ADO — the norm scan reads exactly what exists.
    // Fused (norms_fused_): the last kFinal already folded max|ρ_α|² over
    // its launch set (per slot), which is exactly this scan domain (both are
    // the previous step's ctx.u_list / full grid), so the scan is skipped and
    // the filter takes the sqrt. Same max of the same values ⇒ same flags, bitwise.
    const bool fused = norms_fused_;
    norms_fused_ = false;
    if (!fused)
        compute_ado_norms(d_rho_.get(), d_ado_norms_.get(), sys_.M(), sys_.N(),
                          blk, scan_list, scan_count, active_batch_, scan_off);

    // Mark-only: write active_flags on device, no count / no D2H. |A| is
    // derived on demand from the level distribution in emit_ado_stats, so the
    // hot per-step filter path carries zero host synchronization.
    filter_ados_mark_only(
        d_ado_norms_.get(), d_active_flags_.get(), d_nan_flag_.get(),
        threshold, sys_.M(), scan_list, scan_count, active_batch_, scan_off, fused);
}

void HEOMTrajectory::activate_all_ados(HEOMContext& ctx) {
    CUDA_CHECK(cudaMemset(d_active_flags_.get(), 1, d_active_flags_.size()));
    total_active_ados_ = sys_.M();
    // |A| = M ⇒ U = M, so the compact launch would only add a full-length
    // indirection table. Take the dense path (byte-equal, no indirection) —
    // the same choice the adaptive gate below would make at |E|/M = 1 — with
    // no U gate: the builder's flags describe the previous active set.
    ctx.u_flags = nullptr;
    if (ulist_on()) {
        closure_count_ = active_batch_ * sys_.M();
        ctx.u_list     = nullptr;
        ctx.u_off      = nullptr;
        ctx.u_count    = 0;
        ctx.block_off  = nullptr;
        ctx.launch_blocks = 0;
        ulist_mode_    = UlistMode::Dense;
    }
}

void HEOMTrajectory::hierarchy_filling_(const heom::ManifoldBlock& blk) {
    // Per slot, like the filter: each slot's fill window is judged on its own
    // exact-zero count, so a batched slot opens and closes its window on the
    // same step as that slot alone would (batch: false).
    norms_fused_ = false;   // overwrites d_ado_norms_
    compute_ado_norms(d_rho_.get(), d_ado_norms_.get(), sys_.M(), sys_.N(),
                      blk, nullptr, 0, active_batch_);
    count_zero_norms(d_ado_norms_.get(), sys_.M(), active_batch_,
                     d_zero_count_.get(), h_zero_counts_.data());
    for (int b = 0; b < active_batch_; ++b) {
        if (!fresh_state_[b]) continue;
        const int zeros = h_zero_counts_[b];
        const bool filling = fresh_zeros_[b] < 0
                          || (zeros > 0 && zeros < fresh_zeros_[b]);
        fresh_zeros_[b]  = filling ? zeros : -1;
        fresh_state_[b]  = filling ? 1 : 0;
    }
}

void HEOMTrajectory::init_closure_builder() {
    if (!use_filtering_) return;

    // Fully device-side builder: borrows SystemContext's GPU neighbor maps
    // directly (no D2H, no permanent host mirror — the old host builder
    // kept 2·M·channels·4 B of RAM alive for the whole run).
    ulist_builder_ = std::make_unique<UListBuilder>(
        sys_.M(), sys_.D(), batch_,
        sys_.d_plus_map(), sys_.d_minus_map());

    // τ comes from the config (filter.ulist) and nowhere else, so it is part
    // of config_hash like every other number that changes the result.
    ulist_tau_ = sys_.config().filter.ulist_tau;
    if (!ulist_on()) return;

    std::cout << "[ulist] enabled; whole-step over U=E "
                 "(axpy+combine in-place, no swap, freeze a/=U), "
                 "adaptive gate=" << ulist_tau_ << "\n";
}

int HEOMTrajectory::rebuild_ulist() {
    return ulist_builder_->build(d_active_flags_.get(), active_batch_);
}

void HEOMTrajectory::emit_ado_stats(const std::string& prefix) {
    // Read-only: recompute the per-level distribution from the CURRENT
    // active_flags (does NOT refilter / does NOT touch ρ or the active set),
    // so it is safe to call off the filter cadence (e.g. forced τ/T progress
    // lines from the 2DES driver). |A| and |E| are the most recent filter /
    // ulist-rebuild values held in members.
    // Batched: one flag row per slot, so |A| and the level bins are summed
    // over the active slots and M is printed as active_batch_·M (the size of
    // the flag space). active_batch_ == 1 ⇒ the historical line exactly.
    std::vector<int> level_sum(h_level_counts_.size(), 0);
    for (int b = 0; b < active_batch_; ++b) {
        compute_active_level_distribution(
            d_active_flags_.get() + static_cast<size_t>(b) * sys_.M(),
            sys_.d_level_indices(),
            sys_.M(), sys_.D(), sys_.L(),
            d_level_counts_.get(), h_level_counts_.data());
        for (size_t lev = 0; lev < level_sum.size(); ++lev) level_sum[lev] += h_level_counts_[lev];
    }
    h_level_counts_ = level_sum;
    const int M_flags = active_batch_ * sys_.M();

    // |A| derived from the distribution we just pulled back: every active ADO
    // lands in exactly one level bin, so Σ_lev counts == active count. This is
    // the only place that needs the host-side |A| (display + history), so the
    // per-step filter path no longer computes or transfers it.
    int active_total = 0;
    for (int lev = 0; lev <= sys_.L(); ++lev) active_total += h_level_counts_[lev];
    total_active_ados_ = active_total;

    const heom::StreamFormat fmt_guard(std::cout);

    std::cout << std::fixed << prefix;

    if (closure_count_) {
        // U-list on: |A|/|E|/M with active% and closure% (|E|=A∪neighbors(A)).
        std::cout << "ADOs |A|/|E|/M = "
                  << total_active_ados_ << "/" << *closure_count_ << "/" << M_flags
                  << " (" << std::setprecision(1)
                  << (100.0 * total_active_ados_ / M_flags) << "%, "
                  << (100.0 * *closure_count_ / M_flags) << "%)";
    } else {
        // U-list off (no |E| computed): keep the original |A|/M line.
        std::cout << "Active ADOs = "
                  << total_active_ados_ << "/" << M_flags
                  << " (" << std::setprecision(1)
                  << (100.0 * total_active_ados_ / M_flags) << "%)";
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
    // emitted line (ulist_mode_ is the live state set in run_one_internal_step;
    // Unset means ulist off / before first build ⇒ no line). No numbers: |E|/M
    // is already the closure% above and the gate value is in the "[ulist]
    // enabled" banner, so the flip line carries only the decision.
    if (ulist_mode_ != UlistMode::Unset && ulist_mode_ != last_logged_mode_) {
        std::cout << "    \u2514 U-list \u2192 "
                  << (ulist_mode_ == UlistMode::Dense ? "full-grid fallback"
                                                      : "compact whole-step")
                  << "\n";
        last_logged_mode_ = ulist_mode_;
    }

}

HEOMTrajectory::SlotActiveStats HEOMTrajectory::slot_active_stats() {
    // One norm pass per live slot's sub-buffer; count |A_b| and OR into a union
    // mask on the host, all from the SAME ρ snapshot. Uses active_batch_ (live
    // width 3/9/21 in 2DES), NOT batch_ (allocated 21) — otherwise stale/idle
    // slots would be counted (T0a fix). Union and per-slot are thus temporally
    // consistent, so union_count >= max_b(per_slot).
    const int M = sys_.M();
    const size_t slot_stride = static_cast<size_t>(M) * support_.elems();
    // Grow-only member scratch (see header): no per-call heap alloc, and the
    // union mask lives off the stack (GCC 12/13 -Wfree-nonheap-object FP).
    h_slot_norms_.resize(M);
    h_slot_union_.assign(M, 0);
    h_slot_union_[0] = 1;  // α=0 is always live
    norms_fused_ = false;  // the per-slot scans below overwrite d_ado_norms_
    SlotActiveStats st;
    st.per_slot.assign(active_batch_, 1);
    for (int b = 0; b < active_batch_; ++b) {
        // Per-slot diagnostic (filter.log_per_slot, off by default). Same
        // support as the main path: it is the storage shape, there is nothing
        // else to scan.
        compute_ado_norms(d_rho_.get() + b * slot_stride, d_ado_norms_.get(),
                          M, sys_.N(), support_);
        d_ado_norms_.copy_to_host(h_slot_norms_.data(), M);
        // Same predicate as filter_ados_kernel (NaN counts as active) so this
        // oracle and the device flags agree on |A| even mid-blow-up.
        for (int a = 1; a < M; ++a)
            if (h_slot_norms_[a] > filter_threshold_ ||
                h_slot_norms_[a] != h_slot_norms_[a]) {
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
        // coordinates (seg_prefix in two_dim.cpp). The cumulative internal step counter
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
    // Filter on and past the fill window: the next step filters, so let the
    // final stage fold the norms it needs (norms_fused_). In the fill window
    // the next step runs hierarchy_filling_'s own full scan instead.
    unsigned long long* norm_sq_bits = nullptr;
    // Fuse only when no live slot is still in the fill window — otherwise the
    // next step's hierarchy_filling_ overwrites d_ado_norms_. Per-slot rows.
    bool any_fresh = false;
    if (use_filtering_) {
        for (int b = 0; b < active_batch_; ++b)
            if (fresh_state_[b]) { any_fresh = true; break; }
    }
    if (use_filtering_ && !any_fresh) {
        CUDA_CHECK(cudaMemsetAsync(d_ado_norms_.get(), 0, d_ado_norms_.bytes()));
        norm_sq_bits = reinterpret_cast<unsigned long long*>(d_ado_norms_.get());
    }
    // d_kahan_comp_ is empty in the FP64 build ⇒ get() is nullptr ⇒ rk4_step
    // does the plain (bit-identical) combine. Non-null only in mixed precision.
    if (integrator_ == Integrator::IfRk4) {
        // Factors depend on dt only; rebuild on the first step (no dt yet) or
        // whenever the internal dt changes (τ vs t vs dynamics grids).
        if (if_factors_dt_ != dt_internal_au) {
            compute_if_factors(sys_.d_total_decay(), dt_internal_au, sys_.M(),
                               d_if_e_half_.get(), d_if_e_full_.get());
            if_factors_dt_ = dt_internal_au;
        }
        rk4_step_if(ctx, dt_internal_au, d_workspace_.get(),
                    d_kahan_comp_.get(),
                    d_if_e_half_.get(), d_if_e_full_.get(), norm_sq_bits);
    } else {
        rk4_step(ctx, dt_internal_au, d_workspace_.get(), d_kahan_comp_.get(),
                 norm_sq_bits);
    }
    norms_fused_ = norm_sq_bits != nullptr;
}

void HEOMTrajectory::report_divergence(double t_fs, int slot, double rho_norm,
                                       double norm_ceiling,
                                       int step, int n_steps,
                                       bool nan_in_hierarchy) {
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
              << "  Slot:           " << slot << " of " << active_batch_      << "\n"
              << "  ||rho_0(t)||_F: " << std::setprecision(6)  << rho_norm   << "\n"
              << "  Threshold:      " << std::setprecision(6)  << norm_ceiling << "\n"
              << "  Baseline:       " << std::setprecision(6)  << norm0_[slot]
              << "  (||rho_0||_F at the last set_initial / dipole)\n"
              << "  Step:           " << step  << " / " << n_steps           << "\n"
              << "  NaN in hierarchy: " << (nan_in_hierarchy ? "yes" : "no")
              << (nan_in_hierarchy
                  ? " (an ADO norm is NaN; rho_0 above may still look finite)\n"
                  : "\n")
              << "----------------------------------------------------\n"
              << "  Check: truncation level L, time step dt,\n"
              << "         Matsubara terms K, temperature.\n"
              << "----------------------------------------------------\n";
}

bool HEOMTrajectory::check_divergence(double t_fs, int step, int n_steps) {
    // Refresh the host α=0 mirror(s) so rho0_norm_() sees the current device
    // state. A few KB per slot; keeps this guard self-contained for every
    // caller.
    d2h_alpha0_batch();
    // The filter kernel raises this when any scanned ADO norm is NaN. It is
    // the only way a NaN that has not yet reached ρ_0 becomes visible: with
    // the old norm kernel such an ADO reported a finite norm, was marked
    // inactive, and the run finished with a finite, wrong result. It is one
    // flag for the whole hierarchy, so it fails on the first slot checked.
    int nan_in_hierarchy = 0;
    if (use_filtering_) d_nan_flag_.copy_to_host(&nan_in_hierarchy, 1);
    for (int b = 0; b < active_batch_; ++b) {
        const double rho_norm  = rho0_norm_(b);
        const double norm_ceil = std::max(norm0_[b] * kDivergenceNormFactor,
                                          kDivergenceNormFloor);
        if (nan_in_hierarchy || !(rho_norm <= norm_ceil)) {   // NaN fails <=
            report_divergence(t_fs, b, rho_norm, norm_ceil, step, n_steps,
                              nan_in_hierarchy != 0);
            diverged_ = true;
            break;
        }
    }
    return diverged_;
}

void HEOMTrajectory::run_one_internal_step(HEOMContext& ctx, int step) {
    const auto& cfg = sys_.config();
    const double dt_internal_au = cfg.time.dt_au;
    const int    print_interval = cfg.time.print_interval;
    const double step_time_fs   = step * cfg.time.dt;

    // Unified cadence: filtering + U rebuild run EVERY step on both
    // paths — one semantics, no interval knob. Boundary ADOs (E\A) get their
    // Term 1-3 self-evolution the step after they cross the threshold, so an
    // ADO that is growing lags the full grid by one step. (This used to claim
    // the lag was "exactly like the full grid". It is not: one RK4 step applies
    // A four times, so the unfiltered hierarchy grows FOUR hops per step. The
    // lag is real, which is why it must not compound — see fresh_state_.)
    //
    // fresh_state_: ρ was replaced (trajectory start, dipole, or a restore /
    // expand of a snapshot taken in that state) and the hierarchy is still
    // filling, so some ADOs are exactly 0 only because no step has reached
    // them yet. Activate everything and let full steps populate it; filtering
    // starts on the first step that finds the fill finished.
    //
    // Marking active is never wrong, only slower. Filtering a not-yet-reached
    // ADO is wrong at any threshold: its norm is exactly 0, so it goes inactive
    // and Term 4/5's neighbor gate then releases one level per step.
    //
    // Keying off ρ replacement covers all four cold paths at once:
    // set_initial / apply_dipole set it in on_rho_replaced(), restore / expand
    // reinstall the snapshot's. The window ends with ctx on the dense launch
    // (activate_all_ados, or restore / expand), so the filter pass that
    // follows scans all M.
    //
    // Per slot (batched): each slot has its own window. While every live
    // slot is filling the step is dense, as before; once some have finished,
    // those filter and the still-filling ones keep an all-active flag row
    // (their closure is then all M) — each slot does what it would do alone.
    auto count_fresh = [&] {
        int n = 0;
        for (int b = 0; b < active_batch_; ++b) n += fresh_state_[b] ? 1 : 0;
        return n;
    };
    if (use_filtering_ && count_fresh() > 0) hierarchy_filling_(ctx.block);
    const int n_fresh = use_filtering_ ? count_fresh() : 0;
    if (use_filtering_ && n_fresh == active_batch_) {
        activate_all_ados(ctx);
    } else if (use_filtering_) {
        // Scan domain = the PREVIOUS step's write set (U-local contract).
        // ctx.u_list is exactly the launch set the last run_step wrote through
        // (nullptr = full grid: ulist off, dense fallback, pre-first-build
        // full launch, or a restore that does not know the last write set),
        // and rebuild_ulist below overwrites the device list
        // only AFTER this scan — so reading it here is the correct domain.
        // Block and scan domain both come from the SAME ctx, so they cannot
        // disagree: whatever launch domain the last run_step wrote through is
        // the domain we scan, inside the block it wrote into.
        const bool prev_compact = ctx.u_list != nullptr;
        perform_ado_filtering(filter_threshold_, ctx.block,
                              prev_compact ? ctx.u_list  : nullptr,
                              prev_compact ? ctx.u_count : 0,
                              prev_compact ? ctx.u_off   : nullptr);
        // Slots still in their fill window: all active (see above).
        for (int b = 0; b < active_batch_ && n_fresh > 0; ++b)
            if (fresh_state_[b])
                CUDA_CHECK(cudaMemset(d_active_flags_.get() + static_cast<size_t>(b) * sys_.M(),
                                      1, static_cast<size_t>(sys_.M())));

        // Recompute U_b from the just-updated active_flags (returns Σ_b |U_b|).
        const int ucount = rebuild_ulist();
        closure_count_ = ucount;   // |E| for the stats line (raw, pre-gate)

        // Hard threshold: dense (|E|/M > gate, or ulist off) takes the full
        // grid (no u_list indirection, U gated by u_flags); else compact
        // whole-step. The flip is NOT logged here — emit_ado_stats prints it
        // as an indented continuation of the stats line so it rides a real
        // Step / τ·T·t line (and carries no @step, meaningless in the 2DES
        // tree).
        const double frac = static_cast<double>(ucount)
                          / (static_cast<double>(active_batch_) * sys_.M());
        const bool dense = !ulist_on() || frac > ulist_tau_;
        ctx.u_list  = dense ? nullptr : ulist_builder_->device_list();
        ctx.u_off   = dense ? nullptr : ulist_builder_->device_offsets();
        ctx.u_count = dense ? 0 : ulist_builder_->max_count();
        if (dense) {
            ctx.launch_blocks = 0;
            ctx.block_off = nullptr;
        } else {
            ctx.launch_blocks = ulist_builder_->prepare_launch(
                ctx.block.elems(), HEOM_BLOCK_SIZE);
            ctx.block_off = ulist_builder_->device_block_off();
        }
        ctx.u_flags = dense ? ulist_builder_->device_flags() : nullptr;
        if (ulist_on()) ulist_mode_ = dense ? UlistMode::Dense : UlistMode::Compact;
    }

    if (use_filtering_ && print_interval > 0 && cadence_logging_
        && (step % print_interval == 0))
        print_filtering_statistics(step, step_time_fs);

    run_step(ctx, dt_internal_au);
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
