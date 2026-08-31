#include "system_context.hpp"
#include "core/hierarchy_truncation.hpp"
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
    , K_(cfg.physics.n_bath_modes_total)
    , n_bath_(cfg.physics.n_states)
    , L_(cfg.L_cutoff)
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

    std::cout << "Allocating GPU memory..." << std::flush;
    upload_gpu_readonly(hh);
    // hh goes out of scope here: indices + plus_map + minus_map (the dominant
    // host-side init footprint) are released before the constructor returns.

    // Print total via the SAME plan the budget walked — single source of
    // truth, cannot drift.
    {
        const auto plan = heom::plan_gpu_buffers(
            static_cast<size_t>(M_), N_, K_, n_bath_, L_,
            cfg_.filter.enabled);
        size_t total_bytes = 0;
        for (const auto& s : plan) total_bytes += s.bytes();
        // Formatting stated here, not inherited: this line used to render as
        // "56.9 MB" only because an earlier printer had left fixed+precision(1)
        // on std::cout.
        const heom::StreamFormat fmt_guard(std::cout);
        std::cout << " " << std::fixed << std::setprecision(1)
                  << (total_bytes / (1024.0 * 1024.0)) << " MB\n";
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

void SystemContext::compute_eigenspectrum() {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(cfg_.physics.hamiltonian);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("SystemContext: failed to diagonalize Hamiltonian");
    }
    eigenvalues_  = solver.eigenvalues();
    eigenvectors_ = solver.eigenvectors();
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
        static_cast<size_t>(M_), N_, K_, n_bath_, L_,
        /*use_filtering=*/false);

    // typed_alloc throws std::logic_error on `sizeof(T) != plan.elem_bytes`,
    // so a unilateral change on either side fails loud at startup instead
    // of silently under/over-allocating GPU memory.
    d_hamiltonian_   = heom::typed_alloc<heom::complex_t>(plan, "d_hamiltonian");
    d_c_bath_        = heom::typed_alloc<heom::complex_t>(plan, "d_c_bath");
    d_markov_residue_ = heom::typed_alloc<heom::real_t>(plan, "d_markov_residue");
    d_plus_map_      = heom::typed_alloc<int>(plan, "d_plus_map");
    d_minus_map_     = heom::typed_alloc<int>(plan, "d_minus_map");
    d_level_indices_ = heom::typed_alloc<int8_t>(plan, "d_level_indices");
    d_coeff_plus_    = heom::typed_alloc<heom::real_t>(plan, "d_coeff_plus");
    d_coeff_minus_   = heom::typed_alloc<heom::real_t>(plan, "d_coeff_minus");
    d_total_decay_   = heom::typed_alloc<double>(plan, "d_total_decay");
    d_d_coupling_    = heom::typed_alloc<int8_t>(plan, "d_d_coupling");

    // Bath-correlation coefficients c_k, ν_k, |c_k| ([K+1] each). |c_k| stays
    // host-only: it feeds the divide-by-|c| guard below and the per-ADO
    // coeff_plus/coeff_minus precompute, but the kernel never reads it directly.
    // ν_k must be ready before the decay accumulation below.
    const size_t coeff_len = static_cast<size_t>(n_bath_) * (K_ + 1);
    // h_c_bath is the upload buffer, so it carries hier precision; every
    // derived quantity below (|c|, the Term 4/5 magnitudes) is computed from
    // the exact FP64 cfg_ values and rounded only on the way out.
    std::vector<heom::complex_t> h_c_bath(coeff_len);
    std::vector<double>          h_nu_bath(coeff_len);
    std::vector<double>          h_abs_c_bath(coeff_len);
    for (size_t idx = 0; idx < coeff_len; ++idx) {
        h_c_bath[idx] = heom::c_make(
            static_cast<heom::real_t>(cfg_.physics.c_bath[idx].real()),
            static_cast<heom::real_t>(cfg_.physics.c_bath[idx].imag()));
        h_nu_bath[idx] = cfg_.physics.nu_bath[idx];
        h_abs_c_bath[idx] = std::abs(cfg_.physics.c_bath[idx]);

        // HARD CONTRACT — |c_{nk}| must be strictly positive. Term 5 of the
        // derivative kernel divides by abs_c_bath: sqrt(a_nk / |c_{nk}|) once
        // α_{nk} > 0. A zero coefficient would push an Inf through the whole
        // hierarchy with no diagnostic. Physically every built pole is nonzero
        // (Drude and Matsubara/Padé poles), so a
        // zero here means an upstream construction bug or a degenerate input
        // (e.g. η_n = 0). Fail loud at build time, mirroring the d_coupling
        // {0,1} guard in heom_derivatives.cuh, instead of producing NaNs on
        // the GPU 100 steps later.
        if (!(h_abs_c_bath[idx] > 0.0)) {   // also catches NaN
            const int n = static_cast<int>(idx) / (K_ + 1);
            const int k = static_cast<int>(idx) % (K_ + 1);
            throw std::runtime_error(
                "SystemContext: |c_bath| must be > 0 for every (site, pole), "
                "but |c[n=" + std::to_string(n) + ", k=" + std::to_string(k)
                + "]| = " + std::to_string(h_abs_c_bath[idx])
                + ". Term 5 divides by this (sqrt(a_nk/|c_nk|)). Check the bath "
                  "coefficients (η_n > 0? non-degenerate poles?).");
        }
    }
    std::vector<heom::real_t> h_markov_residue(n_bath_);
    for (int n = 0; n < n_bath_; ++n)
        h_markov_residue[n] = static_cast<heom::real_t>(cfg_.physics.markov_residue[n]);

    // Flatten each variable-length HierarchyIndex into the dense (M, n_bath,
    // K+1) int8 layout the GPU kernels expect, and accumulate the per-ADO
    // total decay Σ_{n,k} α_{nk}·ν_k in the SAME pass — both walk the identical
    // (alpha, n, k) space, so there is no second read-back of the flattened
    // array. total_decay feeds the derivative kernel (Term 2) and the ADO
    // filter cutoff. α_{nk} ≥ 0 by construction, so the ν_k accumulation needs
    // no positivity guard (adding 0·ν_k is harmless).
    // Precompute, in the SAME (alpha, n, k) pass, the per-ADO Term 4/5
    // magnitudes the derivative kernel used to recompute via sqrt() on every
    // one of the N² threads of an ADO (an (i,j)-independent loop invariant):
    //   coeff_plus[α,n,k]  = √[(α_{nk}+1)·|c_{nk}|]
    //   coeff_minus[α,n,k] = √[α_{nk}/|c_{nk}|]   (0 when α_{nk}=0; that branch
    //                        is guarded by α_{nk}>0 in the kernel, so storing 0
    //                        keeps the array free of the 0/|c| edge value).
    // |c_{nk}| > 0 is already enforced above, so the division is always safe.
    const int Kp1    = K_ + 1;
    const int stride = n_bath_ * Kp1;
    std::vector<int8_t> h_level_indices(static_cast<size_t>(M_) * stride);
    std::vector<heom::real_t> h_coeff_plus(static_cast<size_t>(M_) * stride);
    std::vector<heom::real_t> h_coeff_minus(static_cast<size_t>(M_) * stride);
    std::vector<double>       h_total_decay(M_);
    for (int alpha = 0; alpha < M_; ++alpha) {
        const HierarchyIndex idx = hh.indices[alpha];
        const size_t base = static_cast<size_t>(alpha) * stride;
        double decay = 0.0;
        for (int n = 0; n < n_bath_; ++n) {
            for (int k = 0; k <= K_; ++k) {
                const int    local = n * Kp1 + k;
                const int8_t a     = idx[local];
                h_level_indices[base + local] = a;
                decay += a * h_nu_bath[local];
                const double absc = h_abs_c_bath[local];
                h_coeff_plus[base + local]  =
                    static_cast<heom::real_t>(std::sqrt((a + 1) * absc));
                h_coeff_minus[base + local] = (a > 0)
                    ? static_cast<heom::real_t>(std::sqrt(a / absc)) : heom::real_t{0};
            }
        }
        h_total_decay[alpha] = decay;
    }

    std::vector<heom::complex_t> h_hamiltonian(static_cast<size_t>(N_) * N_);
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            std::complex<double> val = cfg_.physics.hamiltonian(i, j);
            h_hamiltonian[i * N_ + j] = heom::c_make(
                static_cast<heom::real_t>(val.real()),
                static_cast<heom::real_t>(val.imag()));
        }
    }

    std::vector<int8_t> h_d_coupling = build_d_coupling(cfg_.physics);

    d_hamiltonian_.copy_from_host(h_hamiltonian.data(), h_hamiltonian.size());
    d_c_bath_.copy_from_host(h_c_bath.data(), h_c_bath.size());
    d_markov_residue_.copy_from_host(h_markov_residue.data(), h_markov_residue.size());
    d_plus_map_.copy_from_host(hh.plus_map.data(), hh.plus_map.size());
    d_minus_map_.copy_from_host(hh.minus_map.data(), hh.minus_map.size());
    d_level_indices_.copy_from_host(h_level_indices.data(), h_level_indices.size());
    d_coeff_plus_.copy_from_host(h_coeff_plus.data(), h_coeff_plus.size());
    d_coeff_minus_.copy_from_host(h_coeff_minus.data(), h_coeff_minus.size());
    d_total_decay_.copy_from_host(h_total_decay.data(), h_total_decay.size());
    d_d_coupling_.copy_from_host(h_d_coupling.data(), h_d_coupling.size());
}
