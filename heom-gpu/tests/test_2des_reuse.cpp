// Pure-host grouping check for the 3→9→21 orientation-reuse tree.
//
// group_isotropic() must reorganise the 21 isotropic components into the prefix
// tree WITHOUT changing the set: 3 m1 branches, 9 (m1,m2) sub-branches, 21
// leaves total, each isotropic_components() index appearing exactly once, and
// every leaf consistent with its branch (leaf.m[0]==m1, leaf.m[1]==m2). This is
// the invariant the reuse walk relies on for canonical-order
// accumulation. No GPU, no propagation — just the reshuffle.

#include "spectra/two_dim.hpp"
#include "spectra/dipole.hpp"
#include "heom_config.hpp"
#include "memory_budget.hpp"   // rho_capacity (pin test)
#include "system_context.hpp"
#include "harness.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool test_grouping() {
    const auto m1s = heom::spectra::group_isotropic();

    if (m1s.size() != 3) {
        std::cout << "grouping: FAIL — expected 3 m1 branches, got "
                  << m1s.size() << "\n";
        return false;
    }

    int total_m2 = 0, total_leaves = 0;
    std::array<int, 21> seen{};  // count each comp_idx

    for (const auto& b1 : m1s) {
        total_m2 += static_cast<int>(b1.m2s.size());
        for (const auto& b2 : b1.m2s) {
            for (const auto& leaf : b2.leaves) {
                ++total_leaves;
                if (leaf.m[0] != b1.m1 || leaf.m[1] != b2.m2) {
                    std::cout << "grouping: FAIL — leaf (" << leaf.m[0] << ","
                              << leaf.m[1] << ") under branch (" << b1.m1 << ","
                              << b2.m2 << ")\n";
                    return false;
                }
                if (leaf.comp_idx < 0 || leaf.comp_idx >= 21) {
                    std::cout << "grouping: FAIL — comp_idx out of range: "
                              << leaf.comp_idx << "\n";
                    return false;
                }
                ++seen[leaf.comp_idx];
            }
        }
    }

    if (total_m2 != 9) {
        std::cout << "grouping: FAIL — expected 9 m2 branches, got "
                  << total_m2 << "\n";
        return false;
    }
    if (total_leaves != 21) {
        std::cout << "grouping: FAIL — expected 21 leaves, got "
                  << total_leaves << "\n";
        return false;
    }
    for (int i = 0; i < 21; ++i) {
        if (seen[i] != 1) {
            std::cout << "grouping: FAIL — comp_idx " << i << " seen "
                      << seen[i] << " times (expected 1)\n";
            return false;
        }
    }

    std::cout << "grouping (3->9->21, bijection over 21 components): PASS\n";
    return true;
}

// B1: build_branch_plan() flattens group_isotropic() into slot-indexed fan-out
// maps. Checks sizes/widths, the leaves[b].comp_idx==b canonical-order invariant
// (bit-exact accumulation depends on it), even fan-out counts (each τ-slot → 3
// T-slots; each T-slot → 2 or 3 t-slots), and consistency with group_isotropic().
bool test_branch_plan() {
    using heom::spectra::build_branch_plan;
    const auto plan = build_branch_plan();

    bool ok = true;
    auto fail = [&](const std::string& msg) { std::cout << "  " << msg << "\n"; ok = false; };

    if (plan.B_tau != 3 || plan.B_T != 9 || plan.B_t != 21)
        fail("widths not 3/9/21");
    if (plan.m2_axis.size() != 9 || plan.tau_to_T.size() != 9)
        fail("T-layer maps not length 9");
    if (plan.T_to_t.size() != 21)
        fail("t-layer map not length 21");

    // Canonical order: leaf slot == comp_idx (the accumulation-order invariant).
    for (int b = 0; b < 21; ++b)
        if (plan.leaves[b].comp_idx != b)
            fail("leaves[" + std::to_string(b) + "].comp_idx="
                 + std::to_string(plan.leaves[b].comp_idx) + " != " + std::to_string(b));

    // Each τ-slot fans to exactly 3 T-slots.
    std::array<int, 3> tau_fanout{};
    for (int T = 0; T < 9; ++T) {
        if (plan.tau_to_T[T] < 0 || plan.tau_to_T[T] >= 3) fail("tau_to_T out of range");
        else ++tau_fanout[plan.tau_to_T[T]];
    }
    for (int i = 0; i < 3; ++i)
        if (tau_fanout[i] != 3)
            fail("tau-slot " + std::to_string(i) + " fanout=" + std::to_string(tau_fanout[i]) + " != 3");

    // Each T-slot fans to 2 or 3 t-slots, summing to 21.
    std::array<int, 9> T_fanout{};
    for (int t = 0; t < 21; ++t) {
        if (plan.T_to_t[t] < 0 || plan.T_to_t[t] >= 9) fail("T_to_t out of range");
        else ++T_fanout[plan.T_to_t[t]];
    }
    int sum = 0;
    for (int T = 0; T < 9; ++T) {
        if (T_fanout[T] != 2 && T_fanout[T] != 3)
            fail("T-slot " + std::to_string(T) + " fanout=" + std::to_string(T_fanout[T]) + " not 2/3");
        sum += T_fanout[T];
    }
    if (sum != 21) fail("T fanout sum=" + std::to_string(sum) + " != 21");

    // Consistency: every leaf's m1 == its T-slot's parent τ-slot m1; m2 == m2_axis.
    for (int t = 0; t < 21; ++t) {
        const int T = plan.T_to_t[t];
        const int tau = plan.tau_to_T[T];
        if (plan.leaves[t].m[0] != plan.m1_axis[tau])
            fail("leaf " + std::to_string(t) + " m1 mismatch vs m1_axis");
        if (plan.leaves[t].m[1] != plan.m2_axis[T])
            fail("leaf " + std::to_string(t) + " m2 mismatch vs m2_axis");
    }

    // Cross-check the flattened plan against group_isotropic() directly.
    const auto m1s = heom::spectra::group_isotropic();
    int t = 0;
    for (int i1 = 0; i1 < static_cast<int>(m1s.size()); ++i1) {
        if (plan.m1_axis[i1] != m1s[i1].m1) fail("m1_axis mismatch");
        for (const auto& b2 : m1s[i1].m2s)
            for (const auto& leaf : b2.leaves) {
                if (plan.leaves[t].m != leaf.m || plan.leaves[t].bucket != leaf.bucket)
                    fail("leaf " + std::to_string(t) + " content mismatch vs group_isotropic");
                ++t;
            }
    }

    // Width 7: 3 m1 subtrees of 1/3/7 slots; together
    // they are the 21-leaf plan, in order, with the same parent maps.
    const auto subs = heom::spectra::walk_plans(plan, 7);
    if (subs.size() != 3) fail("walk_plans(7) not 3 subtrees");
    int leaf = 0, Tbase = 0;
    for (int i1 = 0; i1 < static_cast<int>(subs.size()); ++i1) {
        const auto& s = subs[i1];
        if (s.B_tau != 1 || s.B_T != 3 || s.B_t != 7) fail("subtree widths not 1/3/7");
        if (s.m1_axis[0] != plan.m1_axis[i1]) fail("subtree m1 mismatch");
        for (int T = 0; T < s.B_T; ++T)
            if (s.tau_to_T[T] != 0 || s.m2_axis[T] != plan.m2_axis[Tbase + T])
                fail("subtree T-layer mismatch");
        for (int b = 0; b < s.B_t; ++b, ++leaf) {
            if (s.leaves[b].comp_idx != leaf || s.leaves[b].m != plan.leaves[leaf].m)
                fail("subtree leaf " + std::to_string(b) + " != plan leaf " + std::to_string(leaf));
            if (Tbase + s.T_to_t[b] != plan.T_to_t[leaf]) fail("subtree T_to_t mismatch");
        }
        Tbase += s.B_T;
    }
    if (leaf != 21) fail("subtrees cover " + std::to_string(leaf) + " leaves, not 21");
    if (heom::spectra::walk_plans(plan, 21).size() != 1) fail("walk_plans(21) not the plan");

    if (ok) std::cout << "build_branch_plan (3->9->21 maps, canonical order, 3x7 subtrees): PASS\n";
    else    std::cout << "build_branch_plan: FAIL\n";
    return ok;
}

// ---------------------------------------------------------------------------
// Pure host: the derived τ/T/t manifold accounting against what the GPU was
// MEASURED to do. pathway_segment_blocks() computes the blocks from the pulse
// table with a four-line ladder rule; the table below was read off a probe
// build that dumped max|ρ| per (row manifold, column manifold) after every
// propagation (re-confirmed on the blocking test config: 1140
// observations, every one single-block).
//
// The measurement lives HERE and not in the implementation on purpose. A
// measurement transcribed into the code it was meant to check can no longer
// falsify anything; as an oracle it can, and the mutation check below (flip the
// column-ladder sign) shows it does.
// ---------------------------------------------------------------------------

bool test_manifold_accounting_matches_probe_table() {
    using namespace heom::spectra;

    // Two-letter labels exactly as the probe printed them: row manifold then
    // column manifold, so a reviewer can diff this against the log by eye.
    struct Expect { bool rp; const char* name; const char* seg[3]; };
    static const Expect kOracle[6] = {
        //                        after μ1   after μ2   after μ3
        //                        (τ seg)    (T seg)    (t seg)
        {true,  "SE",  {"ge",     "ee",      "eg"}},
        {true,  "GSB", {"ge",     "gg",      "eg"}},
        {true,  "ESA", {"ge",     "ee",      "fe"}},
        {false, "SE",  {"eg",     "ee",      "eg"}},
        {false, "GSB", {"eg",     "gg",      "eg"}},
        {false, "ESA", {"eg",     "ee",      "fe"}},
    };

    const int n_states = 3;
    // N per basis, spelled out rather than derived, so a bug in the derivation
    // cannot agree with itself: WITH_GROUND 1+3, DOUBLE_EXCITED 1+3+3.
    const auto lay_single = ManifoldLayout::from(BasisType::WITH_GROUND,  n_states, 4);
    const auto lay_double = ManifoldLayout::from(BasisType::DOUBLE_EXCITED, n_states, 7);

    // Index ranges written out as literals for n_states=3, NOT fetched from
    // ManifoldLayout. Expressing the expectation through lay.block() would make
    // the check self-referential — a transposition inside block() would appear
    // on both sides and cancel. (It did, the first time this was written.)
    auto range = [](char m) -> std::pair<int, int> {   // {offset, count}
        switch (m) {
            case 'g': return {0, 1};                   // both bases
            case 'e': return {1, 3};                   // both bases
            default:  return {4, 3};                   // f, DOUBLE_EXCITED only
        }
    };
    int checked = 0;

    for (const auto& want : kOracle) {
        const PathSpec* spec = nullptr;
        for (const auto& p : two_des_path_table())
            if (p.rephasing == want.rp && std::string(p.name) == want.name) spec = &p;
        if (!spec) {
            std::cout << "manifold accounting: FAIL — no PathSpec for "
                      << (want.rp ? "rp " : "nr ") << want.name << "\n";
            return false;
        }
        const auto& lay = (spec->basis == BasisType::DOUBLE_EXCITED)
                        ? lay_double : lay_single;
        const auto got = pathway_segment_blocks(*spec, lay);

        for (int s = 0; s < 3; ++s) {
            const auto r = range(want.seg[s][0]);      // row manifold
            const auto c = range(want.seg[s][1]);      // column manifold
            const ManifoldBlock ref{r.first, r.second, c.first, c.second};
            if (!(got[s] == ref)) {
                std::cout << "manifold accounting: FAIL — "
                          << (want.rp ? "rp " : "nr ") << want.name
                          << " segment " << s << " want " << want.seg[s]
                          << " (" << ref.row_off << "," << ref.n_row << ","
                          << ref.col_off << "," << ref.n_col << ") got ("
                          << got[s].row_off << "," << got[s].n_row << ","
                          << got[s].col_off << "," << got[s].n_col << ")\n";
                return false;
            }
            ++checked;
        }
    }

    // The blocks the kernel will be told to launch over must be non-empty and
    // in range, or apply_dipole would throw at runtime instead of here.
    // ESA's t segment (f,e) is the whole point of the exercise: 3×3 out of 7×7.
    const auto esa = pathway_segment_blocks(
        *[&] { for (const auto& p : two_des_path_table())
                   if (!p.rephasing && std::string(p.name) == "ESA") return &p;
               return static_cast<const PathSpec*>(nullptr); }(), lay_double);
    if (!(esa[2] == ManifoldBlock{4, 3, 1, 3})) {
        std::cout << "manifold accounting: FAIL — nr ESA t segment is not the "
                     "(f,e) 3x3 block at (4,1)\n";
        return false;
    }

    std::cout << "manifold accounting (" << checked
              << " probe-measured cells + ESA (f,e) shape): PASS\n";
    return true;
}

// heom::rho_capacity(cfg) sizes every ρ buffer of a 2DES trajectory
// from the LAYOUT alone (max(n_e², n_f·n_e)), because heom_core cannot see the
// pathway table in spectra. This test walks the real table + segment blocks,
// takes the largest block a SystemContext of each basis will ever be asked to
// hold, and pins rho_capacity to it — plus the (g,g) start block every tree
// begins in. Change the pathway table without this going red and a trajectory
// can be allocated too small for a block it is later asked to install.
bool test_rho_capacity_matches_path_table() {
    using namespace heom::spectra;
    // Only task / basis / n_states / n_sites feed rho_capacity, so a bare
    // Config is enough — no yaml, no propagation.
    HEOMConfig::Config cfg{};
    cfg.task = HEOMConfig::Config::Task::TwoDES;
    cfg.physics.n_states = 3;
    cfg.physics.basis    = BasisType::DOUBLE_EXCITED;
    cfg.physics.n_sites  = 7;                        // 1 + 3 + 3
    HEOMConfig::Config cfg_wg = cfg;
    cfg_wg.physics.basis   = BasisType::WITH_GROUND;
    cfg_wg.physics.n_sites = 4;                      // 1 + 3

    bool ok = true;
    for (const HEOMConfig::Config* c : {&cfg, &cfg_wg}) {
        const auto lay = ManifoldLayout::from(c->physics.basis, c->physics.n_states,
                                              c->physics.n_sites);
        int walked = lay.block(0, 0).elems();          // every tree starts in |g><g|
        for (const auto& p : two_des_path_table()) {
            if (p.basis != c->physics.basis) continue;
            for (const auto& b : pathway_segment_blocks(p, lay))
                walked = std::max(walked, b.elems());
        }
        const int cap = heom::rho_capacity(*c);
        if (cap != walked) {
            std::cout << "rho_capacity vs path table: FAIL — basis "
                      << (c->physics.basis == BasisType::DOUBLE_EXCITED ? "double" : "with_ground")
                      << " table max Nb=" << walked << " rho_capacity=" << cap << "\n";
            ok = false;
        }
    }
    // The numbers themselves, spelled out for n_states=3 so a self-consistent
    // wrong layout cannot pass: double = max(3·3, 3·3) = 9, with_ground = 9.
    if (heom::rho_capacity(cfg) != 9 || heom::rho_capacity(cfg_wg) != 9) {
        std::cout << "rho_capacity vs path table: FAIL — expected 9/9, got "
                  << heom::rho_capacity(cfg) << "/" << heom::rho_capacity(cfg_wg) << "\n";
        ok = false;
    }
    if (ok) std::cout << "rho_capacity == max block of the pathway table (both bases): PASS\n";
    return ok;
}

// The three ways a caller can get this API wrong, each of which would otherwise
// produce a silently mis-sized block: wrong basis, wrong N, off-ladder pulse.
bool test_manifold_layout_rejects_bad_input() {
    using namespace heom::spectra;
    struct Case { const char* what; std::function<void()> run; };
    const Case cases[] = {
        {"EXCITED_ONLY (no ground manifold) asked for |g><g|",
         [] { ManifoldLayout::from(BasisType::EXCITED_ONLY, 3, 3).block(0, 0); }},
        {"EXCITED_ONLY walked as a 2DES pathway",
         [] { pathway_segment_blocks(two_des_path_table()[0],
                  ManifoldLayout::from(BasisType::EXCITED_ONLY, 3, 3)); }},
        {"WITH_GROUND with the double-excited N",
         [] { ManifoldLayout::from(BasisType::WITH_GROUND, 3, 7); }},
        {"DOUBLE_EXCITED with the single-excitation N",
         [] { ManifoldLayout::from(BasisType::DOUBLE_EXCITED, 3, 4); }},
        {"raising out of f",
         [] { const auto l = ManifoldLayout::from(BasisType::DOUBLE_EXCITED, 3, 7);
              apply_pulse(l, {2, 1}, {DipoleSide::Left, true}); }},
        {"lowering out of g",
         [] { const auto l = ManifoldLayout::from(BasisType::WITH_GROUND, 3, 4);
              apply_pulse(l, {0, 0}, {DipoleSide::Left, false}); }},
        {"raising out of e in a basis with no f",
         [] { const auto l = ManifoldLayout::from(BasisType::WITH_GROUND, 3, 4);
              apply_pulse(l, {1, 0}, {DipoleSide::Left, true}); }},
        {"block() with a level the basis does not have",
         [] { ManifoldLayout::from(BasisType::WITH_GROUND, 3, 4).block(2, 0); }},
    };
    for (const auto& c : cases) {
        bool threw = false;
        try { c.run(); } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) {
            std::cout << "manifold layout rejection: FAIL — accepted " << c.what
                      << "\n";
            return false;
        }
    }
    std::cout << "manifold layout rejection (" << std::size(cases)
              << " bad inputs): PASS\n";
    return true;
}

// ---------------------------------------------------------------------------
// End-to-end: batched walk vs reuse walk on the SAME small 2DES problem, with a
// multi-T grid (exercises the T-segment branch). Bit-exact with filter OFF and,
// since every slot keeps its own active set (2026-10-22-batch-per-slot), with
// filter ON too (it used to be the 21-slot union, within tolerance). These two
// are each other's only oracle since the flat 21-tree was removed.
// benchmark_3sites_2des locks the absolute numbers.
// Requires GPU.
// ---------------------------------------------------------------------------

// Tiny 3-site 2DES: 3 τ × 3 T × 3 t. Multi-T so the (m1,m2) T-fanout is covered.
// ulist_tau > 0 exercises the union u_list compact-launch gate (v2); 0 = full-M.
// L_cutoff controls hierarchy depth: 2 (fast, nothing prunes) for the bit-exact
// gates; 4 (M≈210, ADOs actually collapse during T-gaps → compact whole-step
// fires) for the v2 stress test.
std::string write_2des_yaml(bool filter_on, double ulist_tau = 0.0,
                            int l_cutoff = 2, double tolerance = 1.0e-6) {
    // Tolerance goes into the filename: the vanishing-tolerance gate below
    // writes a second filter-on config for the SAME (ulist, L), and two configs
    // sharing a path would silently test the same file twice.
    const bool custom_thr = tolerance != 1.0e-6;
    const std::string path = heom_test::tmp("test_2des_reuse_")
                           + (filter_on ? "filt" : "nofilt")
                           + (ulist_tau > 0 ? "_ulist" : "")
                           + "_L" + std::to_string(l_cutoff)
                           + (custom_thr ? "_thr0" : "") + ".yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 3\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12400.0, -106.0,    8.0]\n"
      << "    - [ -106.0, 12540.0,  28.0]\n"
      << "    - [    8.0,   28.0, 12120.0]\n"
      << "  w0: 12300.0\n"
      << "  dipoles:\n"
      << "    - [-0.7410, -0.5606, -0.3696]\n"
      << "    - [-0.8571,  0.5038, -0.1073]\n"
      << "    - [-0.1971,  0.9574, -0.2110]\n"
      << "  strengths: [1.0, 1.0, 1.0]\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 70.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: pade\n"
      << "  n_extra_poles: 1\n"
      << "  L_cutoff: " << l_cutoff << "\n"
      << "filter:\n"
      << "  enabled: " << (filter_on ? "true" : "false") << "\n"
      << "  tolerance: " << tolerance << "\n"
      << "  ulist: " << ulist_tau << "\n"
      << "2des:\n"
      << "  integrator: rk4\n"
      << "  dt: 16.0\n"
      << "  print_interval: 0\n"
      << "  tau_max: 32.0\n"
      << "  dtau: 16.0\n"
      << "  tw: [\"0\", \"16\", \"32\"]\n"
      << "  t_max: 32.0\n"
      << "  dt_detect: 16.0\n"
      << "  pathways: [\"SE\", \"GSB\", \"ESA\"]\n"
      << "  signals: [\"rephasing\", \"nonrephasing\"]\n"
      << "  orientation: \"isotropic\"\n"
      << "  output:\n"
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
      << "    filename: \"twodes_reuse_test\"\n"
      << "    use_timestamp: false\n";
    return path;
}

// Walk selector: reuse (B=1) or batched (B=21).
enum class Mode { Reuse, Batch, Batch7, Auto };

// v2 stress config: single pathway/signal but L_cutoff=4 and a LONG t-scan
// (t_max=160, dt_detect=8 → 21 t-points) so ADOs collapse during the T-gaps and
// the compact whole-step u_list gate fires (|E|/M drops well below 0.5). This is
// what actually drives the batched derivative kernel through the u_list path;
// the short bit-exact grid never prunes. Kept to one pathway to stay fast.
std::string write_2des_yaml_v2() {
    const std::string path = heom_test::tmp("test_2des_reuse_v2stress.yaml");
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 3\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12400.0, -106.0,    8.0]\n"
      << "    - [ -106.0, 12540.0,  28.0]\n"
      << "    - [    8.0,   28.0, 12120.0]\n"
      << "  w0: 12300.0\n"
      << "  dipoles:\n"
      << "    - [-0.7410, -0.5606, -0.3696]\n"
      << "    - [-0.8571,  0.5038, -0.1073]\n"
      << "    - [-0.1971,  0.9574, -0.2110]\n"
      << "  strengths: [1.0, 1.0, 1.0]\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 70.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: pade\n"
      << "  n_extra_poles: 1\n"
      << "  L_cutoff: 4\n"
      << "filter:\n"
      << "  enabled: true\n"
      << "  tolerance: 1.0e-6\n"
      << "  ulist: 0.5\n"
      << "2des:\n"
      << "  integrator: rk4\n"
      << "  dt: 8.0\n"
      << "  print_interval: 0\n"
      << "  tau_max: 80.0\n"
      << "  dtau: 16.0\n"
      << "  tw: [\"0\", \"16\", \"32\"]\n"
      << "  t_max: 160.0\n"
      << "  dt_detect: 8.0\n"
      << "  pathways: [\"SE\"]\n"
      << "  signals: [\"rephasing\"]\n"
      << "  orientation: \"isotropic\"\n"
      << "  output:\n"
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
      << "    filename: \"twodes_reuse_v2\"\n"
      << "    use_timestamp: false\n";
    return path;
}

// Run one full compute_2des in the given mode; return out.resp. `edit` may
// rewrite the request after it is built from the yaml.
std::vector<std::complex<double>> run_resp(
        const std::string& yaml, Mode mode,
        const std::function<void(heom::spectra::TwoDESRequest&)>& edit = {}) {
    HEOMConfig::Config cfg = HEOMConfig::Config::load_yaml(yaml);
    HEOMConfig::resolve_2des_config(cfg);  // forces basis = double_excited

    SystemContext sys_double(cfg);
    HEOMConfig::Config cfg_wg = cfg;
    HEOMConfig::apply_basis(cfg_wg, BasisType::WITH_GROUND);
    SystemContext sys_single(cfg_wg);

    heom::spectra::TwoDESRequest req =
        heom::spectra::build_2des_request(*cfg.twodes, cfg.time.dt);
    req.batch = (mode == Mode::Batch) ? 21
              : (mode == Mode::Batch7) ? 7
              : (mode == Mode::Auto)   ? heom::spectra::kBatchAuto
                                       : 1;

    auto mu_double = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths,
        BasisType::DOUBLE_EXCITED, sys_double.N());
    auto mu_single = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths,
        BasisType::WITH_GROUND, sys_single.N());
    req.dipole_double = &mu_double;
    req.dipole_single = &mu_single;
    if (edit) edit(req);

    return heom::spectra::compute_2des(sys_single, sys_double, req).resp;
}

// Every name / ladder / budget check runs before the first pathway. A
// typo in the LAST signal used to surface only after all rephasing pathways
// had been computed, and the finished results were thrown away with it.
bool test_bad_request_fails_before_any_pathway() {
    using Req = heom::spectra::TwoDESRequest;
    const std::string yaml = write_2des_yaml(/*filter_on=*/false);
    const std::vector<std::pair<const char*, std::function<void(Req&)>>> cases = {
        {"signal typo",   [](Req& r) { r.signals  = {"rephasing", "nonrephasng"}; }},
        {"pathway typo",  [](Req& r) { r.pathways = {"SE", "GSB", "ESX"}; }},
        {"pathway twice", [](Req& r) { r.pathways = {"SE", "SE"}; }},
    };
    for (const auto& [what, edit] : cases) {
        std::ostringstream log;
        std::streambuf* const real = std::cout.rdbuf(log.rdbuf());
        bool threw = false;
        try { run_resp(yaml, Mode::Reuse, edit); }
        catch (const std::invalid_argument&) { threw = true; }
        catch (...) {}
        std::cout.rdbuf(real);
        const bool ran = log.str().find("[reuse m1=") != std::string::npos;
        if (!threw || ran) {
            std::cout << "bad 2DES request (" << what << "): FAIL — invalid_argument="
                      << threw << ", a pathway ran first=" << ran << "\n";
            return false;
        }
    }
    std::cout << "bad 2DES request rejected before any pathway: PASS\n";
    return true;
}

// Relative max deviation: max|Δ| / max|ref|. Filter-scale differences (union
// closure) scale with the response magnitude, so an ABSOLUTE tolerance is too
// strict for large systems (verified: FMO ESA max|Δ|=2.1e-6 but max|resp|=56 ⇒
// only 3.7e-8 relative). Bit-exact paths give max|Δ|=0 ⇒ relative 0 exactly, so
// the bit-exact gates are unaffected.
double rel_diff(const std::vector<std::complex<double>>& ra,
                const std::vector<std::complex<double>>& rb) {
    if (ra.size() != rb.size()) {
        std::cout << "  resp size mismatch: " << ra.size()
                  << " vs " << rb.size() << "\n";
        return 1e30;
    }
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ra.size(); ++i) {
        num = std::max(num, std::abs(ra[i] - rb[i]));
        den = std::max(den, std::abs(ra[i]));
    }
    return den > 0.0 ? num / den : num;
}

double max_diff_yaml(const std::string& yaml, Mode a, Mode b) {
    return rel_diff(run_resp(yaml, a), run_resp(yaml, b));
}

double max_diff_modes(bool filter_on, Mode a, Mode b, double ulist_tau = 0.0,
                      int l_cutoff = 2) {
    const std::string yaml = write_2des_yaml(filter_on, ulist_tau, l_cutoff);
    return rel_diff(run_resp(yaml, a), run_resp(yaml, b));
}

// ---------------------------------------------------------------------------
// The invariant that should have existed from day one: filtering is an
// APPROXIMATION WITH A KNOB, so driving the knob to zero must recover the
// unfiltered answer exactly. If it does not, the filter is not pruning the
// hierarchy — it is changing the equations of motion, and no choice of
// tolerance can make the answer right.
//
// Two things about this test are load-bearing; changing either turns it into
// a test that passes while the bug is present (see
// ADO filter growth-throttle probe):
//
//  1. It MUST run a 2DES path: apply_dipole replaces ρ mid-run, and it once
//     refiltered while the hierarchy was still empty. This 3-site L=2 grid
//     fills in one step, so it cannot see a fill window that is too short;
//     test_vanishing_tolerance_fmo_2des below covers that.
//  2. ulist MUST be 0. The compact launch domain U = A ∪ N(A) reaches only one
//     hop past the active set, which is a SECOND, independent growth limiter.
//     Keep this gate pointed at the derivative kernel alone.
// ---------------------------------------------------------------------------
bool test_vanishing_tolerance_equals_filter_off() {
    const auto off = run_resp(write_2des_yaml(/*filter_on=*/false), Mode::Reuse);
    // 1e-30: only an ADO that is EXACTLY zero can be marked inactive, so a
    // filter that merely prunes has nothing left to prune.
    const auto on  = run_resp(
        write_2des_yaml(/*filter_on=*/true, /*ulist_tau=*/0.0, /*l_cutoff=*/2,
                        /*tolerance=*/1.0e-30),
        Mode::Reuse);
    const double d = rel_diff(off, on);
    if (d != 0.0) {
        std::cout << "vanishing tolerance vs filter off: FAIL — rel|Δ|=" << d
                  << " (expected 0)\n"
                  << "  tolerance=1e-30 must degenerate to filter-off. A nonzero\n"
                  << "  delta means the filter alters the dynamics rather than\n"
                  << "  pruning it — the threshold axis cannot fix this.\n";
        return false;
    }
    std::cout << "vanishing tolerance vs filter off (bit-exact): PASS\n";
    return true;
}

// The same invariant on a hierarchy that does not fill in one step. The
// 3-site L=2 gate above is filled after one full step, so it passed while every
// FMO 2DES at IF dt=10 sat on a 3.5e-8 (L=4) / 1.5e-7 (L=5) floor: after the
// first step exact zeros were left, went inactive at any threshold, and the
// neighbor gate released them one level per step. L=5 needs three full steps
// from |g><g|, so a fixed two-step window fails here too. The τ chain
// snapshots and restores mid-window, so the window state rides the snapshot.
std::string write_fmo_2des_yaml(bool filter_on) {
    const std::string path = heom_test::tmp(std::string("test_2des_fmo_L5_")
                                            + (filter_on ? "thr0" : "off") + ".yaml");
    std::ofstream f(path);
    f << "system:\n  n_states: 7\n  temperature: 77.0\n  hamiltonian:\n"
         "    - [12415.0,  -87.7,    5.5,   -5.9,    6.7,  -13.7,   -9.9]\n"
         "    - [ -87.7, 12479.7,   30.8,    8.2,    0.7,   11.8,    4.3]\n"
         "    - [   5.5,   30.8, 12134.1,  -53.5,   -2.2,   -9.6,    6.0]\n"
         "    - [  -5.9,    8.2,  -53.5, 12317.4,  -70.7,  -17.0,  -63.3]\n"
         "    - [   6.7,    0.7,   -2.2,  -70.7, 12445.0,   81.1,   -1.3]\n"
         "    - [ -13.7,   11.8,   -9.6,  -17.0,   81.1, 12610.0,   39.7]\n"
         "    - [  -9.9,    4.3,    6.0,  -63.3,   -1.3,   39.7, 12417.0]\n"
         "  w0: 12300.0\n  dipoles:\n"
         "    - [-0.7410, -0.5606, -0.3696]\n    - [-0.8571,  0.5038, -0.1073]\n"
         "    - [-0.1971,  0.9574, -0.2110]\n    - [-0.7992, -0.5336, -0.2766]\n"
         "    - [-0.7369,  0.6558,  0.1641]\n    - [-0.1350, -0.8792,  0.4569]\n"
         "    - [-0.4951, -0.7083, -0.5031]\n"
         "  strengths: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n"
         "bath:\n  spectral_density: Drude-Lorentz\n  eta: 70.0\n  gamma_inverse: 100.0\n"
         "  expansion_method: pade\n  n_extra_poles: 1\n  L_cutoff: 5\n"
         "filter:\n  enabled: " << (filter_on ? "true" : "false") << "\n"
         "  tolerance: 1e-300\n  ulist: 0.9\n"
         "2des:\n  integrator: if-rk4\n  dt: 10\n  print_interval: 0\n  tau_max: 30.0\n"
         "  dtau: 10.0\n  tw: [\"0\"]\n  t_max: 80.0\n  dt_detect: 10.0\n"
         "  pathways: [\"SE\", \"GSB\", \"ESA\"]\n  signals: [\"rephasing\"]\n"
         "  orientation: isotropic\n  output:\n"
         "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
         "    filename: \"twodes_fmo_L5\"\n    use_timestamp: false\n";
    return path;
}

bool test_vanishing_tolerance_fmo_2des() {
    const auto off = run_resp(write_fmo_2des_yaml(/*filter_on=*/false), Mode::Reuse);
    const auto on  = run_resp(write_fmo_2des_yaml(/*filter_on=*/true),  Mode::Reuse);
    const double d = rel_diff(off, on);
    if (d != 0.0) {
        std::cout << "vanishing tolerance vs filter off (FMO-7 L=5 2DES): FAIL — rel|Δ|="
                  << d << " (expected 0)\n";
        return false;
    }
    std::cout << "vanishing tolerance vs filter off (FMO-7 L=5 2DES, bit-exact): PASS\n";
    return true;
}

// B3: batched vs reuse. Bit-exact with filter OFF (main gate) and
// filter ON (per-slot active sets).
bool test_batch_vs_reuse_off() {
    const double d = max_diff_modes(/*filter_on=*/false, Mode::Batch, Mode::Reuse);
    if (d != 0.0) {
        std::cout << "batched vs reuse (filter off): FAIL — max|Δ|=" << d
                  << " (expected 0)\n";
        return false;
    }
    std::cout << "batched vs reuse (filter off, bit-exact): PASS\n";
    return true;
}

// Width 7 (3 m1 subtrees of 1/3/7 slots) vs 21 and vs reuse: filter OFF ⇒ bit-exact,
// same gate as above.
bool test_batch7_off() {
    const double d21 = max_diff_modes(/*filter_on=*/false, Mode::Batch7, Mode::Batch);
    const double d1  = max_diff_modes(/*filter_on=*/false, Mode::Batch7, Mode::Reuse);
    if (d21 != 0.0 || d1 != 0.0) {
        std::cout << "batch 7 (filter off): FAIL — vs 21 max|Δ|=" << d21
                  << ", vs reuse max|Δ|=" << d1 << " (expected 0)\n";
        return false;
    }
    std::cout << "batch 7 vs 21 vs reuse (filter off, bit-exact): PASS\n";
    return true;
}

// Filter ON: the active set is the union over the live slots, so width 7 (union of
// 7) sits between 21 and reuse — within the same filter-scale tolerance.
bool test_batch7_on() {
    const double tol = 1e-6;  // relative
    const double d = max_diff_yaml(write_2des_yaml_v2(), Mode::Batch7, Mode::Batch);
    if (d > tol) {
        std::cout << "batch 7 vs 21 (filter on, v2 ulist gate): FAIL — rel|Δ|=" << d
                  << " > " << tol << "\n";
        return false;
    }
    std::cout << "batch 7 vs 21 (filter on, v2 ulist gate, rel within " << tol
              << "): PASS  (rel|Δ|=" << d << ")\n";
    return true;
}

bool test_batch_vs_reuse_on() {
    const double d = max_diff_modes(/*filter_on=*/true, Mode::Batch, Mode::Reuse);
    if (d != 0.0) {
        std::cout << "batched vs reuse (filter on): FAIL — rel|Δ|=" << d
                  << " (expected 0: per-slot active sets)\n";
        return false;
    }
    std::cout << "batched vs reuse (filter on, per-slot, bit-exact): PASS\n";
    return true;
}

// B4 v2: u_list compact-launch gate ON (ulist_tau>0). Each slot builds its own
// closure U_b and launches over it (per-slot segments of one compact list), so
// batched must equal reuse exactly. This is the test that exercises the
// batched derivative kernel THROUGH the compact per-slot u_list path.
bool test_batch_vs_reuse_v2() {
    // Long-t L_cutoff=4 config so ADOs collapse during T-gaps and the compact
    // whole-step u_list gate fires (verified: reuse hits |E|/M≈13% → compact),
    // driving the batched derivative kernel THROUGH the u_list path.
    const double d = max_diff_yaml(write_2des_yaml_v2(), Mode::Batch, Mode::Reuse);
    if (d != 0.0) {
        std::cout << "batched vs reuse (v2 ulist gate): FAIL — rel|Δ|=" << d
                  << " (expected 0: per-slot closures)\n";
        return false;
    }
    std::cout << "batched vs reuse (v2 ulist compact path, per-slot, bit-exact): PASS\n";
    return true;
}


// Auto on a card that fits B=21 must pick 21 and match an explicit
// B=21 run bit-for-bit with filtering off. Also checks the reason string and
// TwoDESResult::batch_width.
bool test_batch_auto_off() {
    const auto yaml = write_2des_yaml(/*filter_on=*/false);
    HEOMConfig::Config cfg = HEOMConfig::Config::load_yaml(yaml);
    HEOMConfig::resolve_2des_config(cfg);
    SystemContext sys_double(cfg);
    HEOMConfig::Config cfg_wg = cfg;
    HEOMConfig::apply_basis(cfg_wg, BasisType::WITH_GROUND);
    SystemContext sys_single(cfg_wg);

    auto run = [&](int batch) {
        heom::spectra::TwoDESRequest req =
            heom::spectra::build_2des_request(*cfg.twodes, cfg.time.dt);
        req.batch = batch;
        auto mu_double = heom::spectra::DipoleOperator::from_pigment_dipoles(
            cfg.physics.dipoles, cfg.physics.strengths,
            BasisType::DOUBLE_EXCITED, sys_double.N());
        auto mu_single = heom::spectra::DipoleOperator::from_pigment_dipoles(
            cfg.physics.dipoles, cfg.physics.strengths,
            BasisType::WITH_GROUND, sys_single.N());
        req.dipole_double = &mu_double;
        req.dipole_single = &mu_single;
        return heom::spectra::compute_2des(sys_single, sys_double, req);
    };
    const auto auto_out = run(heom::spectra::kBatchAuto);
    const auto b21_out  = run(21);
    if (auto_out.batch_width != 21) {
        std::cout << "batch auto (filter off): FAIL — selected B="
                  << auto_out.batch_width << " (expected 21 on ample GPU)\n";
        return false;
    }
    if (auto_out.batch_auto_reason.find("B=21") == std::string::npos) {
        std::cout << "batch auto (filter off): FAIL — reason missing B=21: "
                  << auto_out.batch_auto_reason << "\n";
        return false;
    }
    if (auto_out.resp.size() != b21_out.resp.size()) {
        std::cout << "batch auto (filter off): FAIL — resp size mismatch\n";
        return false;
    }
    double max_abs = 0.0;
    for (size_t i = 0; i < auto_out.resp.size(); ++i)
        max_abs = std::max(max_abs, std::abs(auto_out.resp[i] - b21_out.resp[i]));
    if (max_abs != 0.0) {
        std::cout << "batch auto vs 21 (filter off): FAIL — max|Δ|=" << max_abs
                  << " (expected 0)\n";
        return false;
    }
    std::cout << "batch auto → 21 vs explicit 21 (filter off, bit-exact): PASS\n";
    return true;
}

}  // namespace

int main() {
    RUN_BOOL(test_grouping);
    RUN_BOOL(test_branch_plan);
    RUN_BOOL(test_manifold_accounting_matches_probe_table);
    RUN_BOOL(test_rho_capacity_matches_path_table);
    RUN_BOOL(test_manifold_layout_rejects_bad_input);
    RUN_BOOL(test_bad_request_fails_before_any_pathway);
    RUN_BOOL(test_vanishing_tolerance_equals_filter_off);
    RUN_BOOL(test_vanishing_tolerance_fmo_2des);
    RUN_BOOL(test_batch_vs_reuse_off);
    RUN_BOOL(test_batch_vs_reuse_on);
    RUN_BOOL(test_batch_vs_reuse_v2);
    RUN_BOOL(test_batch7_off);
    RUN_BOOL(test_batch7_on);
    RUN_BOOL(test_batch_auto_off);
    return heom_test_summary("test_2des_reuse");
}
