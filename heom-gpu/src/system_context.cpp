#include "system_context.hpp"
#include "core/hierarchy_truncation.hpp"
#include "core/manifold_layout.hpp"
#include "core/mode_layout.hpp"
#include "gpu_alloc_plan.hpp"
#include "memory_budget.hpp"
#include "util/stream_format.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

SystemContext::SystemContext(const HEOMConfig::Config& cfg)
    : cfg_(cfg)
    , N_(cfg.physics.n_sites)
    , D_(cfg.physics.n_ladders())
    , n_bath_(cfg.physics.n_states)
    , L_(cfg.L_cutoff)
    , rho_capacity_(heom::rho_capacity(cfg))
{
    std::cout << "=== Initialization ===\n";

    std::cout << "Querying system information..." << std::flush;
    sysinfo_ = SystemInfo::query();
    std::cout << " done.\n";

    // Compile-time tunable (see cuda_common.cuh). Logged so a perf-regression
    // post-mortem can tell which value the binary was built with.
    std::cout << "CUDA threads/block: " << HEOM_BLOCK_SIZE
              << "  (override with cmake -DHEOM_BLOCK_SIZE=...)\n";

    std::cout << "Generating hierarchy..." << std::flush;
    HostHierarchy hh = generate_hierarchy_and_maps();
    M_ = static_cast<int>(hh.indices.size());
    std::cout << " M=" << M_ << "\n";

    check_memory_budget();
    compute_eigenspectrum();

    // Only the read-only tables are uploaded here; ρ, RK4 workspace and the
    // filter buffers are HEOMTrajectory's. Print the two owners' shares
    // separately so the line says who allocates what (the trajectory share is
    // the B=1 plan; a batched trajectory re-checks with its real B).
    std::cout << "Uploading hierarchy tables..." << std::flush;
    upload_gpu_readonly(hh);
    // hh goes out of scope here: indices + plus_map + minus_map (the dominant
    // host-side init footprint) are released before the constructor returns.

    {
        const auto plan = heom::plan_gpu_buffers(
            static_cast<size_t>(M_), N_, D_, n_bath_, L_,
            cfg_.filter.enabled, /*batch=*/1,
            static_cast<size_t>(rho_capacity_));
        using Bk = heom::GPUBufferSpec::Bucket;
        size_t mine = 0, traj = 0;
        for (const auto& s : plan) {
            const bool trajectory_owned =
                s.bucket == Bk::RhoTensor || s.bucket == Bk::RK4Workspace ||
                s.bucket == Bk::FilterBuf;
            (trajectory_owned ? traj : mine) += s.bytes();
        }
        // Formatting stated here, not inherited: this line used to render as
        // "56.9 MB" only because an earlier printer had left fixed+precision(1)
        // on std::cout.
        const heom::StreamFormat fmt_guard(std::cout);
        std::cout << std::fixed << std::setprecision(1)
                  << " read-only tables " << (mine / (1024.0 * 1024.0))
                  << " MB (SystemContext); trajectory buffers "
                  << (traj / (1024.0 * 1024.0)) << " MB (HEOMTrajectory, B=1)\n";
    }
}

SystemContext::HostHierarchy SystemContext::generate_hierarchy_and_maps() {
    HierarchyTruncation hierarchy(cfg_.physics, L_);
    HostHierarchy hh;
    hh.indices = hierarchy.generate_hierarchy_indices();
    auto [plus_map, minus_map] = hierarchy.build_neighbor_maps(hh.indices);
    hh.plus_map  = std::move(plus_map);
    hh.minus_map = std::move(minus_map);
    return hh;
}

// Each excitation manifold (g / e / f) is diagonalised on its own, energy-
// ascending within the manifold, and the blocks are laid out in manifold
// order. Column lo(e) + k is then the k-th single exciton in every basis. A
// single global sort put |g⟩ (E = 0) behind every exciton below w0, so
// "exciton 1" in with_ground silently selected the ground state.
void SystemContext::compute_eigenspectrum() {
    const auto& H = cfg_.physics.hamiltonian;
    const auto lay = heom::ManifoldLayout::from(cfg_.physics.basis,
                                                cfg_.physics.n_states, N_);
    eigenvalues_  = Eigen::VectorXd::Zero(N_);
    eigenvectors_ = Eigen::MatrixXcd::Zero(N_, N_);
    for (int lv = 0; lv < 3; ++lv) {
        const int o = lay.lo(lv), n = lay.size(lv);
        if (n == 0) continue;
        Eigen::MatrixXcd off = H.middleRows(o, n);
        off.middleCols(o, n).setZero();
        if (off.cwiseAbs().maxCoeff() != 0.0)
            throw std::runtime_error(
                "SystemContext: Hamiltonian couples excitation manifolds; "
                "the per-manifold eigenbasis would be wrong");
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H.block(o, o, n, n));
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("SystemContext: failed to diagonalize Hamiltonian");
        eigenvalues_.segment(o, n)      = solver.eigenvalues();
        eigenvectors_.block(o, o, n, n) = solver.eigenvectors();
    }
}

void SystemContext::check_memory_budget() {
    auto budget = heom::estimate_budget(cfg_, static_cast<size_t>(M_));
    size_t gpu_free = 0, gpu_total = 0;
    heom::query_gpu_memory(gpu_free, gpu_total);

    // Full per-bucket detail only when the user asked for it or the job is
    // non-trivial (>50% of free GPU); default-quiet keeps CI logs clean.
    const bool verbose = cfg_.output.verbose_memory
                      || (gpu_free > 0 && budget.gpu_total > gpu_free / 2);
    if (verbose) heom::print_budget(budget, gpu_free, gpu_total);
    else         heom::print_budget_compact(budget, gpu_free);

    std::string err;
    switch (heom::check_budget(budget, cfg_, gpu_free, err)) {
        case heom::BudgetCheck::Fits:
            break;
        case heom::BudgetCheck::Exceeds:
            // Force full detail on failure — user needs the breakdown to
            // see which bucket dominated (compact mode hid this).
            if (!verbose) heom::print_budget(budget, gpu_free, gpu_total);
            throw std::runtime_error(err);
        case heom::BudgetCheck::Unknown:
            std::cerr << "[WARN] could not query GPU memory; "
                         "skipping pre-allocation budget check\n";
            break;
    }
}

void SystemContext::upload_gpu_readonly(const HostHierarchy& hh) {
    // Allocate via the canonical plan; HEOMTrajectory owns the filter buffers.
    const auto plan = heom::plan_gpu_buffers(
        static_cast<size_t>(M_), N_, D_, n_bath_, L_,
        /*use_filtering=*/false, /*batch=*/1,
        static_cast<size_t>(rho_capacity_));

    // typed_alloc throws std::logic_error on `sizeof(T) != plan.elem_bytes`,
    // so a unilateral change on either side fails loud at startup instead
    // of silently under/over-allocating GPU memory.
    d_hamiltonian_   = heom::typed_alloc<heom::complex_t>(plan, "d_hamiltonian");
    d_c_bath_        = heom::typed_alloc<heom::complex_t>(plan, "d_c_bath");
    d_cbar_bath_     = heom::typed_alloc<heom::complex_t>(plan, "d_cbar_bath");
    d_markov_residue_ = heom::typed_alloc<heom::real_t>(plan, "d_markov_residue");
    d_plus_map_      = heom::typed_alloc<int>(plan, "d_plus_map");
    d_minus_map_     = heom::typed_alloc<int>(plan, "d_minus_map");
    d_level_indices_ = heom::typed_alloc<int8_t>(plan, "d_level_indices");
    d_coeff_plus_    = heom::typed_alloc<heom::real_t>(plan, "d_coeff_plus");
    d_coeff_minus_   = heom::typed_alloc<heom::real_t>(plan, "d_coeff_minus");
    d_total_decay_   = heom::typed_alloc<cuDoubleComplex>(plan, "d_total_decay");
    d_d_coupling_    = heom::typed_alloc<int8_t>(plan, "d_d_coupling");
    d_mode_off_      = heom::typed_alloc<int>(plan, "d_mode_off");

    // Bath-correlation coefficients c_k, c̄_k, ν_k and the Shi scale s_k ([D]
    // each). s_k stays host-only: it feeds the s_k > 0 guard below and the
    // per-ADO coeff_plus/coeff_minus precompute, but the kernel never reads
    // it directly. ν_k must be ready before the decay accumulation below.
    const heom::ModeLayout ml = cfg_.physics.mode_layout();
    const size_t coeff_len = static_cast<size_t>(ml.channels());
    // h_c_bath / h_cbar_bath are upload buffers, so they carry hier precision;
    // every derived quantity below (s_k, the Term 4/5 magnitudes) is computed
    // from the exact FP64 cfg_ values and rounded only on the way out.
    std::vector<heom::complex_t>      h_c_bath(coeff_len);
    std::vector<heom::complex_t>      h_cbar_bath(coeff_len);
    std::vector<std::complex<double>> h_nu_bath(coeff_len);
    std::vector<double>               h_scale(coeff_len);
    for (size_t idx = 0; idx < coeff_len; ++idx) {
        const std::complex<double> c    = cfg_.physics.c_bath[idx];
        const std::complex<double> cbar = cfg_.physics.cbar_bath[idx];
        h_c_bath[idx] = heom::c_make(
            static_cast<heom::real_t>(c.real()), static_cast<heom::real_t>(c.imag()));
        h_cbar_bath[idx] = heom::c_make(
            static_cast<heom::real_t>(cbar.real()), static_cast<heom::real_t>(cbar.imag()));
        h_nu_bath[idx] = cfg_.physics.nu_bath[idx];
        // Shi scaling: ρ̃_α = ρ_α / Π_k √(α_k! s_k^{α_k}). Any s_k > 0 is a
        // legal similarity transform (Term 4 × Term 5 product is invariant);
        // s_k = √(|c_k||c̄_k|) keeps the two ladders of a conjugate pair at the
        // same magnitude so the ADO filter threshold means the same thing on
        // both. Real pole: c̄ = c* ⇒ s = |c| to within 1 ulp.
        h_scale[idx] = std::sqrt(std::abs(c) * std::abs(cbar));

        // HARD CONTRACT — s_k must be strictly positive. Term 5's precomputed
        // magnitude is sqrt(a_nk / s_k) once α_{nk} > 0; a zero would push an
        // Inf through the whole hierarchy with no diagnostic. Physically every
        // built pole is nonzero, so a zero here means an upstream construction
        // bug or a degenerate input (η_n = 0, huang_rhys = 0). Fail loud at
        // build time, mirroring the d_coupling {0,1} guard in
        // heom_derivatives.cuh, instead of producing NaNs 100 steps later.
        if (!(h_scale[idx] > 0.0)) {   // also catches NaN
            const int n = ml.site_of(static_cast<int>(idx));
            const int k = ml.pole_of(static_cast<int>(idx));
            static const char* kind_name[] = {"Drude", "Padé/Matsubara", "BO+", "BO−"};
            const auto& kinds = cfg_.physics.mode_kind;
            const std::string kind = idx < kinds.size()
                ? kind_name[static_cast<int>(kinds[idx]) & 3] : "?";
            throw std::runtime_error(
                "SystemContext: √(|c||c̄|) must be > 0 for every (site, pole), "
                "but at (n=" + std::to_string(n) + ", k=" + std::to_string(k)
                + ", mode_kind=" + kind + ") |c| = " + std::to_string(std::abs(c))
                + ", |c̄| = " + std::to_string(std::abs(cbar))
                + ". Term 5 divides by this. Check the bath coefficients "
                  "(η_n > 0? huang_rhys > 0? non-degenerate poles?).");
        }
    }
    std::vector<heom::real_t> h_markov_residue(n_bath_);
    for (int n = 0; n < n_bath_; ++n)
        h_markov_residue[n] = static_cast<heom::real_t>(cfg_.physics.markov_residue[n]);

    // Flatten each variable-length HierarchyIndex into the dense (M, D) int8
    // layout the GPU kernels expect, and accumulate the per-ADO
    // total decay Σ_{n,k} α_{nk}·ν_k in the SAME pass — both walk the identical
    // (alpha, n, k) space, so there is no second read-back of the flattened
    // array. total_decay is complex (ν± of a vibrational pair are conjugate);
    // it feeds the derivative kernel (Term 2) and the IF-RK4 factor tables.
    // α_{nk} ≥ 0 by construction, so the ν_k accumulation needs no guard.
    // Precompute, in the SAME (alpha, n, k) pass, the per-ADO Term 4/5
    // magnitudes the derivative kernel used to recompute via sqrt() on every
    // one of the N² threads of an ADO (an (i,j)-independent loop invariant):
    //   coeff_plus[α,n,k]  = √[(α_{nk}+1)·s_{nk}]
    //   coeff_minus[α,n,k] = √[α_{nk}/s_{nk}]   (0 when α_{nk}=0; that branch
    //                        is guarded by α_{nk}>0 in the kernel, so storing 0
    //                        keeps the array free of the 0/s edge value).
    // s_{nk} > 0 is already enforced above, so the division is always safe.
    const size_t map_len = static_cast<size_t>(M_) * ml.channels();
    std::vector<int8_t> h_level_indices(map_len);
    std::vector<heom::real_t> h_coeff_plus(map_len);
    std::vector<heom::real_t> h_coeff_minus(map_len);
    std::vector<cuDoubleComplex> h_total_decay(M_);
    for (int alpha = 0; alpha < M_; ++alpha) {
        const HierarchyIndex idx = hh.indices[alpha];
        const size_t base = static_cast<size_t>(alpha) * ml.channels();
        std::complex<double> decay(0.0, 0.0);
        for (int n = 0; n < n_bath_; ++n) {
            const int kc = ml.count(n);
            for (int k = 0; k < kc; ++k) {
                const int    local = ml.slot(n, k);
                const int8_t a     = idx[local];
                h_level_indices[base + local] = a;
                decay += static_cast<double>(a) * h_nu_bath[local];
                const double sc = h_scale[local];
                h_coeff_plus[base + local]  =
                    static_cast<heom::real_t>(std::sqrt((a + 1) * sc));
                h_coeff_minus[base + local] = (a > 0)
                    ? static_cast<heom::real_t>(std::sqrt(a / sc)) : heom::real_t{0};
            }
        }
        h_total_decay[alpha] = make_cuDoubleComplex(decay.real(), decay.imag());

        // First-tier list: exactly one occupation on one ladder. Built here
        // so the rest of the host hierarchy can still be discarded.
        int tier = 0, last = -1;
        for (int s = 0; s < D_; ++s) {
            const int8_t a = idx[s];
            if (a > 0) {
                tier += a;
                last = s;
            }
        }
        if (tier == 1) {
            FirstTierAdo e;
            e.alpha = alpha;
            e.site  = ml.site_of(last);  // bath index (pigment or shared)
            e.k     = ml.pole_of(last);
            e.scale = h_scale[static_cast<size_t>(last)];
            first_tier_.push_back(e);
        }
    }

    // The derivative kernel stages H as REAL (sH is real_t: half the smem, so
    // N=106 stays under the 48 KiB default and gets 2 blocks/SM instead of 1
    // — measured 1.88x on LHC-II ESA). Every H this code builds is real
    // (YAML doubles, expand_hamiltonian, mc_disorder), so an imaginary part
    // here is a bug upstream, not a feature to support silently.
    std::vector<heom::complex_t> h_hamiltonian(static_cast<size_t>(N_) * N_);
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            std::complex<double> val = cfg_.physics.hamiltonian(i, j);
            if (val.imag() != 0.0) {
                throw std::runtime_error(
                    "SystemContext: Hamiltonian has a nonzero imaginary part at ("
                    + std::to_string(i) + "," + std::to_string(j)
                    + "); the derivative kernel assumes a real H");
            }
            h_hamiltonian[i * N_ + j] = heom::c_make(
                static_cast<heom::real_t>(val.real()),
                static_cast<heom::real_t>(val.imag()));
        }
    }

    h_d_coupling_ = build_d_coupling(cfg_.physics);

    d_hamiltonian_.copy_from_host(h_hamiltonian.data(), h_hamiltonian.size());
    d_c_bath_.copy_from_host(h_c_bath.data(), h_c_bath.size());
    d_cbar_bath_.copy_from_host(h_cbar_bath.data(), h_cbar_bath.size());
    d_markov_residue_.copy_from_host(h_markov_residue.data(), h_markov_residue.size());
    d_plus_map_.copy_from_host(hh.plus_map.data(), hh.plus_map.size());
    d_minus_map_.copy_from_host(hh.minus_map.data(), hh.minus_map.size());
    d_level_indices_.copy_from_host(h_level_indices.data(), h_level_indices.size());
    d_coeff_plus_.copy_from_host(h_coeff_plus.data(), h_coeff_plus.size());
    d_coeff_minus_.copy_from_host(h_coeff_minus.data(), h_coeff_minus.size());
    d_total_decay_.copy_from_host(h_total_decay.data(), h_total_decay.size());
    d_d_coupling_.copy_from_host(h_d_coupling_.data(), h_d_coupling_.size());
    d_mode_off_.copy_from_host(cfg_.physics.mode_off.data(), cfg_.physics.mode_off.size());
}
