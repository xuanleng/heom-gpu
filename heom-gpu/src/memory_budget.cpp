#include "memory_budget.hpp"
#include "util/stream_format.hpp"
#include "gpu_alloc_plan.hpp"
#include "types.hpp"
#include "core/cuda_common.cuh"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace heom {

namespace {

// IEC binary unit formatter. Picks the largest unit that keeps the value ≥ 1
// (or stays in MiB if the value would jump straight from KiB to GiB).
std::string format_iec(size_t bytes) {
    constexpr double KiB = 1024.0;
    constexpr double MiB = 1024.0 * 1024.0;
    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    std::ostringstream s;
    s << std::fixed;
    if (bytes >= static_cast<size_t>(GiB)) {
        s << std::setprecision(2) << (bytes / GiB) << " GiB";
    } else if (bytes >= static_cast<size_t>(MiB)) {
        s << std::setprecision(2) << (bytes / MiB) << " MiB";
    } else if (bytes >= static_cast<size_t>(KiB)) {
        s << std::setprecision(1) << (bytes / KiB) << " KiB";
    } else {
        s << bytes << " B";
    }
    return s.str();
}

// Per-row pretty-printer: "  label    value   (annotation)"
void print_row(const char* label, size_t bytes, const std::string& annotation = {}) {
    std::cout << "    " << std::left << std::setw(28) << label
              << std::right << std::setw(12) << format_iec(bytes);
    if (!annotation.empty()) {
        std::cout << "   " << annotation;
    }
    std::cout << "\n";
}

// Note: vector header / allocator overhead are intentionally NOT counted in
// host buckets. For LHC-II (M ≈ 10⁶, n_bath·(K+1) ≈ 70) the raw int8_t data
// dwarfs any reasonable allocator overhead by 3+ orders of magnitude. Trying
// to model libstdc++ + glibc ptmalloc constants here was a portability bug
// (libc++/MSVC STL/jemalloc/tcmalloc all differ).

}  // namespace

// ---------------------------------------------------------------------------
// estimate_budget
// ---------------------------------------------------------------------------

// CAVEAT: a config that fits in GPU memory here is NOT necessarily runnable.
// The kernels index ρ with a 32-bit linear offset, so propagation also requires
// M·N² < 2^31 (see heom::checked_total_elements). At LHC-II scale (M ≈ 10⁶,
// N = 106) the byte budget may sit comfortably under an 80 GB card while M·N²
// ~ 1e10 still blows the int index ceiling. This estimator reports bytes only;
// the index ceiling is enforced (loudly) at the first RK4 launch.
MemoryBudget estimate_budget(const HEOMConfig::Config& cfg, size_t M,
                             size_t batch) {
    MemoryBudget b{};
    b.M             = M;
    b.N             = cfg.physics.n_sites;
    b.n_bath        = cfg.physics.n_states;
    b.K             = cfg.physics.n_bath_modes_total;
    b.L             = cfg.L_cutoff;
    b.batch         = batch;
    b.use_filtering = cfg.filter.enabled;

    // GPU buckets: aggregate from the canonical plan in gpu_alloc_plan.hpp.
    // This is the single source of truth — allocate_gpu_memory() walks the
    // same plan, so the prediction here cannot drift from reality. batch scales
    // d_rho / d_workspace inside the plan.
    const auto plan = plan_gpu_buffers(M, b.N, b.K, b.n_bath, b.L, b.use_filtering, batch);
    using Bk = GPUBufferSpec::Bucket;
    for (const auto& s : plan) {
        switch (s.bucket) {
            case Bk::RhoTensor:    b.rho_tensor     += s.bytes(); break;
            case Bk::RK4Workspace: b.rk4_workspace  += s.bytes(); break;
            case Bk::HierarchyMap: b.hierarchy_maps += s.bytes(); break;
            case Bk::FilterBuf:    b.filter_buffers += s.bytes(); break;
            case Bk::Auxiliary:    b.auxiliary      += s.bytes(); break;
        }
    }
    // Mixed-precision Kahan compensation carry: same shape as ρ
    // (M·N²·sizeof(complex_t)), persistent, allocated only in the mixed build.
    // Not in the alloc plan (precision-mode-only cost), so add it explicitly and
    // fold into the RK4 workspace bucket — it lives with the integrator state.
    if constexpr (heom::kMixedPrecision) {
        const size_t kahan = batch * M * static_cast<size_t>(b.N) * b.N
                           * sizeof(heom::complex_t);
        b.rk4_workspace += kahan;
    }

    b.gpu_total = b.rho_tensor + b.rk4_workspace
                + b.hierarchy_maps + b.filter_buffers + b.auxiliary;

    // Host buckets: raw data only (allocator/header overhead negligible).
    const size_t n_bath  = static_cast<size_t>(b.n_bath);
    const size_t Kp1     = static_cast<size_t>(b.K) + 1;
    const size_t map_sz  = M * n_bath * Kp1;
    // h_rho0_ mirror element size follows hier precision; the mixed build also
    // keeps an FP64 staging copy (h_rho0_fp64_) for the float64-only HDF5 write.
    const size_t cd      = sizeof(heom::complex_t)
                         + (heom::kMixedPrecision ? sizeof(cuDoubleComplex) : 0);

    // HierarchyTable is one contiguous int8_t[M][n_bath·(K+1)] buffer, plus the
    // two int[M] row permutations the lexicographic sort holds simultaneously
    // (freed before build_neighbor_maps runs, but part of the init peak).
    b.host_hierarchy      = M * (n_bath * Kp1 * sizeof(int8_t) + 2 * sizeof(int));
    b.host_neighbor_maps  = 2 * map_sz * sizeof(int);
    b.host_density_buffer = static_cast<size_t>(b.N) * b.N * cd;
    b.host_total          = b.host_hierarchy + b.host_neighbor_maps + b.host_density_buffer;

    // h_level_indices in upload_gpu_readonly: a flat int8 copy of the same
    // hierarchy data, allocated while host_hierarchy is still alive.
    // h_total_decay (M·8B) and h_active_flags staging (M·1B) are negligible.
    b.host_init_peak = map_sz * sizeof(int8_t);

    return b;
}

// ---------------------------------------------------------------------------
// print_budget
// ---------------------------------------------------------------------------

void print_budget(const MemoryBudget& b, size_t gpu_free, size_t gpu_total) {
    const heom::StreamFormat fmt_guard(std::cout);
    std::cout << "\n[MEMORY BUDGET]  (estimated, before any cudaMalloc)\n";

    // Parameter echo line (helps diagnose "M is huge because L is too high")
    std::cout << "  Params:  M=" << b.M
              << "  N=" << b.N
              << "  n_bath=" << b.n_bath
              << "  K=" << b.K
              << "  L=" << b.L
              << "  filter=" << (b.use_filtering ? "on" : "off")
              << "\n";

    if (gpu_total > 0) {
        std::cout << "  Device:  " << format_iec(gpu_free) << " free / "
                  << format_iec(gpu_total) << " total\n";
    }

    std::cout << "  ---------------------------------------------------------\n";
    std::cout << "  GPU buffers (will be cudaMalloc'd):\n";
    // Element size reflects the build's hier precision (FP64 16B / mixed 8B).
    const int csz = static_cast<int>(sizeof(heom::complex_t));
    const std::string bpref =
        (b.batch > 1) ? (std::to_string(b.batch) + " * ") : std::string();
    print_row("rho tensor (in-place RK4)", b.rho_tensor,
              bpref + "1 * M * N^2 * " + std::to_string(csz) + "B");
    print_row("RK4 workspace (3 buffers)", b.rk4_workspace,
              bpref + "3 * M * N^2 * " + std::to_string(csz)
              + "B  (k_curr + temp + acc"
              + (heom::kMixedPrecision ? " + Kahan)" : ")"));
    print_row("hierarchy maps (x3)", b.hierarchy_maps,
              "M * n_bath * (K+1) * 9B");
    if (b.filter_buffers > 0) {
        print_row("filter buffers", b.filter_buffers,
                  "ado_norms + active_flags + partials");
    } else {
        print_row("filter buffers", 0, "(filtering disabled)");
    }
    print_row("auxiliary", b.auxiliary,
              "H + Matsubara + decay + d_coupling");
    std::cout << "    " << std::left << std::setw(28) << "GPU TOTAL"
              << std::right << std::setw(12) << format_iec(b.gpu_total);
    if (gpu_total > 0) {
        const double pct = 100.0 * static_cast<double>(b.gpu_total)
                                  / static_cast<double>(gpu_free);
        std::cout << "   (" << std::fixed << std::setprecision(1) << pct
                  << "% of free)";
    }
    std::cout << "\n";

    std::cout << "  Host memory (transient peak during SystemContext init,\n"
                 "              freed before solve(); only h_rho0_ persists):\n";
    print_row("hierarchy index objects", b.host_hierarchy,
              "M * (n_bath * (K+1) + 2 * 4B)");
    print_row("neighbor maps (host)", b.host_neighbor_maps,
              "2 * M * n_bath * (K+1) * 4B");
    print_row("output buffer h_rho0_", b.host_density_buffer,
              "N^2 * 16B  (rho_0 only, persistent)");
    std::cout << "    " << std::left << std::setw(28) << "HOST TOTAL"
              << std::right << std::setw(12) << format_iec(b.host_total) << "\n";
    print_row("  + upload staging peak", b.host_init_peak,
              "h_level_indices, overlaps with hierarchy");
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// print_budget_compact
// ---------------------------------------------------------------------------

void print_budget_compact(const MemoryBudget& b, size_t gpu_free) {
    const heom::StreamFormat fmt_guard(std::cout);
    std::cout << "[MEMORY BUDGET]  GPU: " << format_iec(b.gpu_total);
    if (gpu_free > 0) {
        const double pct = 100.0 * static_cast<double>(b.gpu_total)
                                  / static_cast<double>(gpu_free);
        std::cout << "  (" << std::fixed << std::setprecision(1) << pct
                  << "% of free)";
    }
    std::cout << "  M=" << b.M << " N=" << b.N
              << "  filter=" << (b.use_filtering ? "on" : "off") << "\n";
}

// ---------------------------------------------------------------------------
// check_budget  (tri-state)
// ---------------------------------------------------------------------------

BudgetCheck check_budget(const MemoryBudget& b,
                         const HEOMConfig::Config& cfg,
                         size_t gpu_free,
                         std::string& err,
                         double safety_margin)
{
    err.clear();

    // No silent clamp — bad input is a programming error.
    if (!(safety_margin > 0.0 && safety_margin <= 1.0)) {
        throw std::invalid_argument(
            "check_budget: safety_margin must be in (0, 1], got "
            + std::to_string(safety_margin));
    }

    if (gpu_free == 0) {
        // Caller couldn't query the device. Don't lie that everything fits;
        // tell them we couldn't check, let them decide.
        return BudgetCheck::Unknown;
    }

    const size_t budget_with_margin =
        static_cast<size_t>(gpu_free * safety_margin);

    if (b.gpu_total <= budget_with_margin) {
        return BudgetCheck::Fits;
    }

    std::ostringstream s;
    s << "GPU memory budget exceeded.\n"
      << "  Required:        " << format_iec(b.gpu_total) << "\n"
      << "  Available:       " << format_iec(gpu_free)
      << " free  (with " << std::fixed << std::setprecision(0)
      << (safety_margin * 100.0) << "% safety margin = "
      << format_iec(budget_with_margin) << " usable)\n"
      << "  Shortfall:       " << format_iec(b.gpu_total - budget_with_margin) << "\n"
      << "\n"
      << "  Suggestions (largest impact first):\n";

    // Mitigation hints. RK4 workspace + rho tensor together scale as
    // 5·M·N²·16 (3 RK4 + 2 rho), dominate every realistic config — always
    // show this hint.
    s << "    - Reduce L_cutoff (current=" << b.L
      << "): smaller L → fewer ADOs (M scales steeply with L)\n";

    // K-reduction hint is unconditional. The old `if hierarchy_maps
    // > gpu_total/4` check was dead code (ratio is ~10⁻²-10⁻⁴ in practice).
    if (b.K > 0) {
        s << "    - Reduce n_extra_poles (current=" << b.K
          << "): each step removes one mode per bath in the hierarchy maps\n";
    }

    if (!b.use_filtering) {
        s << "    - Enable ADO filtering (filter.enabled: true, "
             "tolerance: e.g. 1e-7); runtime active-set is typically 1-10% of M\n";
    } else {
        s << "    - Tighten filter.tolerance (current="
          << std::scientific << std::setprecision(1)
          << cfg.filter.tolerance
          << "): a larger threshold prunes more ADOs\n";
    }

    if (cfg.physics.basis == BasisType::DOUBLE_EXCITED) {
        const int n_st = cfg.physics.n_states;
        const int n_with_ground = 1 + n_st;
        s << "    - Switch basis from double_excited (n_sites=" << b.N
          << ") to with_ground (n_sites=" << n_with_ground
          << "): N^2 shrinks by " << std::fixed << std::setprecision(0)
          << (static_cast<double>(b.N * b.N)
              / static_cast<double>(n_with_ground * n_with_ground))
          << "x\n";
    }

    s << "    - Move to a GPU with at least "
      << format_iec(static_cast<size_t>(b.gpu_total / safety_margin))
      << " free memory\n";

    err = s.str();
    return BudgetCheck::Exceeds;
}

// ---------------------------------------------------------------------------
// query_gpu_memory
// ---------------------------------------------------------------------------

bool query_gpu_memory(size_t& free, size_t& total) {
    free  = 0;
    total = 0;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) {
        free  = 0;
        total = 0;
        return false;
    }
    return true;
}

}  // namespace heom
