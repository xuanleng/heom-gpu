#include "hierarchy_truncation.hpp"
#include "util/stream_format.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>

HierarchyTruncation::HierarchyTruncation(const PhysicalParameters& params,
                                         int L_cutoff,
                                         std::ostream* log)
    : phys_params_(params), L_cutoff_(L_cutoff), log_(log) {
    if (log_) *log_ << "Hierarchy truncation: simple Σα ≤ " << L_cutoff_ << "\n";
}

HierarchyTable HierarchyTruncation::generate_hierarchy_indices() {
    const int D = index_width();
    HierarchyTable result(D);

    // Start with zero index α⃗ = 0⃗, then walk. `current` is the single scratch
    // row for the entire recursion — the generator never allocates per index.
    std::vector<int8_t> current(D, 0);
    result.append(current.data());
    generate_recursive(current, 0, 0, result);

    // Sort lexicographically so build_neighbor_maps() can locate any index
    // by binary search. Rows are fixed-width and live in one buffer, so we
    // sort a permutation of row ids and apply it in place rather than moving
    // D-byte rows through std::sort.
    const size_t M = result.size();
    std::vector<int> dest(M);
    {
        std::vector<int> order(M);
        for (size_t i = 0; i < M; ++i) order[i] = static_cast<int>(i);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return std::memcmp(result.row(a), result.row(b), D) < 0;
        });
#ifndef NDEBUG
        // The recursive generator visits each (position, alpha) tuple exactly
        // once by construction; this catches a regression breaking that.
        for (size_t r = 1; r < M; ++r) {
            assert(std::memcmp(result.row(order[r - 1]), result.row(order[r]), D) != 0
                   && "generate_recursive produced duplicates");
        }
#endif
        for (size_t r = 0; r < M; ++r) dest[order[r]] = static_cast<int>(r);
    }
    result.permute_rows(dest);

    print_statistics(result);

    return result;
}

void HierarchyTruncation::generate_recursive(
    std::vector<int8_t>& current,
    int position,
    int bath_pole_sum,
    HierarchyTable& result) {

    if (position >= index_width()) {
        return;
    }

    // Upper bound on α at this position: residual budget against L_cutoff_.
    // Defensive clamp to 126 because storage is int8_t; validate() already
    // rejects L_cutoff ≥ 127, the clamp is belt-and-suspenders.
    const int alpha_max = std::min(L_cutoff_ - bath_pole_sum, 126);

    for (int8_t alpha = 1; alpha <= alpha_max; ++alpha) {
        current[position] = alpha;
        result.append(current.data());
        generate_recursive(current, position + 1, bath_pole_sum + alpha, result);
    }

    // Backtrack: leave a 0 at this position.
    current[position] = 0;
    generate_recursive(current, position + 1, bath_pole_sum, result);
}

// indices is sorted lexicographically by generate_hierarchy_indices(), so we
// look neighbours up by binary search — O(log M * D) per query, no
// auxiliary container. This is preferred over a std::unordered_map<hash,int>:
// the hash map wasted ~M*48 B of host RAM (≥500 MB at LHC-II scale), needed a
// lazy-FNV cache on every HierarchyIndex, and required runtime collision
// detection that did not actually catch anything in practice.
std::pair<std::vector<int>, std::vector<int>>
HierarchyTruncation::build_neighbor_maps(const HierarchyTable& indices) const {
    // log_ may be &std::cout: never leak fixed/precision to it.
    const heom::StreamFormat fmt_guard(log_);
    const int M = static_cast<int>(indices.size());
    const heom::ModeLayout ml = mode_layout();
    const int N = ml.n_bath;
    const int D = ml.channels();
    // size_t on purpose: this is the host table, and the comment below promises
    // it can be larger than the GPU's int ceiling. Keep the promise.
    const size_t total_size = static_cast<size_t>(M) * D;

    if (log_) *log_ << "Building neighbor maps (M=" << M << ")..." << std::flush;

    // Host cost is O(M · D · log M): the triple loop below visits every
    // (ADO, site, pole) slot and resolves each ±neighbor by std::lower_bound
    // over the sorted M-vector. Single-threaded; for the largest host-buildable
    // config (LHC-II scale, M~9M) this is the dominant CPU init cost, so it is
    // timed here. (That M is a host-side data-structure stress point only — the
    // GPU propagation is separately capped at M·N² < 2^31 AND M·D
    // < 2^31; see checked_rho_elements / checked_map_elements. The host
    // table here is size_t and has neither limit.) Revisit
    // (parallelize / cache neighbors) only if this line ever shows up as a
    // startup bottleneck.
    const auto build_t0 = std::chrono::steady_clock::now();

    std::vector<int> plus_map(total_size, -1);
    std::vector<int> minus_map(total_size, -1);

    // Cheap precondition: confirm indices is sorted. The binary search is
    // meaningless if it is not, and the only producer
    // (generate_hierarchy_indices) guarantees it — but the cost is M-1 row
    // compares (~ms even for LHC-II) and the alternative is silent corruption.
#ifndef NDEBUG
    for (int i = 1; i < M; ++i) {
        assert(std::memcmp(indices.row(i - 1), indices.row(i), D) < 0);
    }
#endif

    auto find_pos = [&](const int8_t* target) -> int {
        int lo = 0, hi = M;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (std::memcmp(indices.row(mid), target, D) < 0) lo = mid + 1;
            else                                              hi = mid;
        }
        return (lo < M && std::memcmp(indices.row(lo), target, D) == 0) ? lo : -1;
    };

    // One scratch row, mutated in place and restored after each probe. The
    // old code materialised a fresh HierarchyIndex per ± neighbour (2·M·D
    // heap allocations) and recomputed Σα inside the validity test, making the
    // walk O(M·D²); the row sum is loop-invariant in (n,k), so hoist it.
    std::vector<int8_t> probe(D);
    for (int i = 0; i < M; ++i) {
        std::memcpy(probe.data(), indices.row(i), D);
        const int row_sum = indices[i].sum();
        const bool can_raise = (row_sum < L_cutoff_);

        for (int n = 0; n < N; ++n) {
            const int kc = ml.count(n);
            for (int k = 0; k < kc; ++k) {
                const int    slot       = ml.slot(n, k);
                const size_t linear_idx = static_cast<size_t>(i) * D + slot;

                if (can_raise) {
                    ++probe[slot];
                    plus_map[linear_idx] = find_pos(probe.data());
                    --probe[slot];
                }
                if (probe[slot] > 0) {
                    --probe[slot];
                    minus_map[linear_idx] = find_pos(probe.data());
                    ++probe[slot];
                }
            }
        }
    }

    const double build_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - build_t0).count();
    if (log_) *log_ << " done (" << std::fixed << std::setprecision(2)
                    << build_s << " s).\n";

    return {plus_map, minus_map};
}

void HierarchyTruncation::print_statistics(const HierarchyTable& indices) const {
    const heom::StreamFormat fmt_guard(log_);
    // By-level distribution. Tells the user at a glance whether
    // the truncation is tight (last level dominates → bumping the cap
    // explodes M) or loose (last level negligible → cap is over-conservative).
    // Also doubles as a cheap consistency check: Σ count == M.
    if (log_) {
        *log_ << "Generated " << indices.size() << " hierarchy indices"
              << " (L=" << L_cutoff_ << ")\n";
    }

    std::vector<int> count_by_level(L_cutoff_ + 1, 0);
    for (const auto& idx : indices) {
        const int lev = idx.sum();
        if (lev >= 0 && lev <= L_cutoff_) ++count_by_level[lev];
    }
    // total + [BUG] check run unconditionally; only the table printout is
    // gated on log_ (silent batch sweeps still surface a leaking hierarchy).
    int total = 0;
    for (int lev = 0; lev <= L_cutoff_; ++lev) total += count_by_level[lev];
    if (log_) {
        for (int lev = 0; lev <= L_cutoff_; ++lev) {
            const double pct = indices.empty()
                ? 0.0
                : 100.0 * count_by_level[lev] / static_cast<double>(indices.size());
            *log_ << "  L=" << lev << ": "
                  << std::setw(10) << count_by_level[lev]
                  << "  ("
                  << std::fixed << std::setprecision(2) << std::setw(6) << pct
                  << "%)\n";
        }
    }
    if (total != static_cast<int>(indices.size())) {
        std::cerr << "[BUG] hierarchy by-level sum " << total
                  << " != M " << indices.size()
                  << " (this means generate_hierarchy_indices is leaking ADOs"
                     " outside [0, L])\n";
    }
    if (log_) *log_ << "\n";
}
