#include "spectra/two_dim.hpp"

#include "system_context.hpp"
#include "heom_trajectory.hpp"
#include "detector.hpp"                 // heom::Detector (device-side Tr(μ_det·ρ_0))
#include "memory_budget.hpp"            // estimate/check_budget (batch=21 guard)
#include "output/console_printer.hpp"   // format_computation_time (per-tree elapsed)

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <sstream>
#include <string>

namespace heom::spectra {

TwoDESRequest build_2des_request(const HEOMConfig::TwoDESConfig& spec,
                                 double dt_fs) {
    const double dt = dt_fs;  // fs
    auto nsamp = [&](double max, double step) {
        return step > 0.0 ? static_cast<int>(std::llround(max / step)) + 1 : 1;
    };
    auto nstep = [&](double step) { return static_cast<int>(std::llround(step / dt)); };

    TwoDESRequest req;
    req.n_tau = nsamp(spec.tau_max, spec.dtau);
    req.n_t   = nsamp(spec.t_max, spec.dt_detect);
    req.steps_per_tau = nstep(spec.dtau);
    req.steps_per_t   = nstep(spec.dt_detect);
    req.dtau_fs       = spec.dtau;
    req.dt_detect_fs  = spec.dt_detect;
    req.tw_fs = spec.tw_fs;
    req.n_tw  = static_cast<int>(spec.tw_fs.size());
    req.tw_gap_steps.resize(req.n_tw);
    {
        double prev = 0.0;
        for (int k = 0; k < req.n_tw; ++k) {
            req.tw_gap_steps[k] = nstep(spec.tw_fs[k] - prev);
            prev = spec.tw_fs[k];
        }
    }
    req.signals  = spec.signals;
    req.pathways = spec.pathways;
    req.batch    = spec.batch;
    return req;
}

// The 6 single-side Liouville pathways. (side, raise) is fixed.
// Order = [rp SE,GSB,ESA] then [nr SE,GSB,ESA]
// so a writer/post-processor can rely on a stable layout. NEVER fill these from
// memory — they are the physics contract.
//
//   rp: first interaction on the RIGHT (ρ·μ⁻ → |g><e|)
//   nr: first interaction on the LEFT  (μ⁺·ρ → |e><g|)
//   detection is always μ⁻ (trace), handled by the driver, not stored here.
const std::array<PathSpec, 6>& two_des_path_table() {
    using S = DipoleSide;
    static const std::array<PathSpec, 6> kTable = {{
        // ---- rephasing (φ1,φ2,φ3 = Φ1,Φ2,Φ3) ----  (last field = combine sign)
        {"SE",  true,  BasisType::WITH_GROUND,
            {{ {S::Right, false}, {S::Left,  true }, {S::Right, true } }}, +1.0},
        {"GSB", true,  BasisType::WITH_GROUND,
            {{ {S::Right, false}, {S::Right, true }, {S::Left,  true } }}, +1.0},
        {"ESA", true,  BasisType::DOUBLE_EXCITED,
            {{ {S::Right, false}, {S::Left,  true }, {S::Left,  true } }}, -1.0},
        // ---- nonrephasing (φ1,φ2,φ3 = Φ4,Φ5,Φ6) ----
        {"SE",  false, BasisType::WITH_GROUND,
            {{ {S::Left,  true }, {S::Right, false}, {S::Right, true } }}, +1.0},
        {"GSB", false, BasisType::WITH_GROUND,
            {{ {S::Left,  true }, {S::Left,  false}, {S::Left,  true } }}, +1.0},
        {"ESA", false, BasisType::DOUBLE_EXCITED,
            {{ {S::Left,  true }, {S::Right, false}, {S::Left,  true } }}, -1.0},
    }};
    return kTable;
}

std::pair<int, int> apply_pulse(const ManifoldLayout& lay,
                                std::pair<int, int> level, PathStep step) {
    const int d = step.raise ? +1 : -1;
    auto [r, c] = level;
    if (step.side == DipoleSide::Left) r += d;
    else                               c -= d;   // (ρμ⁺)_ij sums over μ⁺_kj, and
                                                 // μ⁺ is nonzero one level UP
                                                 // from j — so j sits one level
                                                 // BELOW ρ's old column.
    // Off the ladder: below g, above f, or onto a level this basis does not
    // populate (f in WITH_GROUND). Either way the pulse table and the basis
    // disagree, which is a config bug, not a zero result to propagate.
    if (r < 0 || r >= 3 || c < 0 || c >= 3 || lay.empty(r) || lay.empty(c))
        throw std::invalid_argument(
            "apply_pulse: dipole walks off the manifold ladder — the pulse "
            "table and the basis disagree");
    return {r, c};
}

std::array<ManifoldBlock, 3> pathway_segment_blocks(const PathSpec& spec,
                                                    const ManifoldLayout& lay) {
    if (lay.empty(0))
        throw std::invalid_argument(
            "pathway_segment_blocks: 2DES needs a ground state in the basis");
    std::pair<int, int> lv{0, 0};        // |g><g|
    std::array<ManifoldBlock, 3> out{};
    for (int p = 0; p < 3; ++p) {
        lv = apply_pulse(lay, lv, spec.pulses[p]);
        out[p] = lay.block(lv.first, lv.second);
    }
    return out;                          // [after μ1, after μ2, after μ3]
}                                        //  = [τ segment, T segment, t segment]

namespace {

// |0><0| — the ground projector. In both WITH_GROUND and DOUBLE_EXCITED the
// ground state sits at Hilbert index 0 (compute_hilbert_dim layout).
Eigen::MatrixXcd ground_projector(int N) {
    Eigen::MatrixXcd P = Eigen::MatrixXcd::Zero(N, N);
    P(0, 0) = std::complex<double>(1.0, 0.0);
    return P;
}

// Its support: the (g,g) block, which lay.block(0,0) spells as {0,1,0,1}.
ManifoldBlock ground_block(const ManifoldLayout& lay) { return lay.block(0, 0); }

const PathSpec& find_path(bool rephasing, const std::string& pathway) {
    for (const auto& p : two_des_path_table())
        if (p.rephasing == rephasing && pathway == p.name) return p;
    throw std::invalid_argument(
        "compute_2des: unknown pathway '" + pathway + "' (want SE/GSB/ESA)");
}

// The 21 molecular-frame components with non-zero isotropic weight: the union
// of the three delta-pairing patterns aabb/abab/abba over a,b in {x,y,z}.
// I1 if m1==m2 and m3==m4, I2 if m1==m3 and m2==m4, I3 if m1==m4 and m2==m3.
// xxxx/yyyy/zzzz feed all three.
struct Component {
    std::array<int, 4> m;          // cartesian axes (0=x,1=y,2=z) per interaction
    std::array<bool, 3> bucket;    // belongs to {I1, I2, I3}
};

const std::vector<Component>& isotropic_components() {
    static const std::vector<Component> kComps = [] {
        std::vector<Component> v;
        for (int m1 = 0; m1 < 3; ++m1)
        for (int m2 = 0; m2 < 3; ++m2)
        for (int m3 = 0; m3 < 3; ++m3)
        for (int m4 = 0; m4 < 3; ++m4) {
            const bool b1 = (m1 == m2 && m3 == m4);  // R_aabb -> I1
            const bool b2 = (m1 == m3 && m2 == m4);  // R_abab -> I2
            const bool b3 = (m1 == m4 && m2 == m3);  // R_abba -> I3
            if (b1 || b2 || b3)
                v.push_back({{m1, m2, m3, m4}, {b1, b2, b3}});
        }
        return v;  // size == 21
    }();
    return kComps;
}

}  // namespace

// Reorganise the 21 isotropic components into the 3→9→21 prefix tree in
// first-seen order. Pure reshuffle — every component lands in exactly one leaf,
// tagged with its isotropic_components() index for canonical-order accumulation.
std::vector<ReuseM1Branch> group_isotropic() {
    const auto& comps = isotropic_components();
    std::vector<ReuseM1Branch> m1s;
    for (int ci = 0; ci < static_cast<int>(comps.size()); ++ci) {
        const auto& c = comps[ci];
        // Find (or append) the m1 branch, then the m2 sub-branch, in first-seen
        // order — no sorting, so the walk order is deterministic/reproducible.
        auto m1it = std::find_if(m1s.begin(), m1s.end(),
                                 [&](const ReuseM1Branch& b){ return b.m1 == c.m[0]; });
        if (m1it == m1s.end()) { m1s.push_back({c.m[0], {}}); m1it = std::prev(m1s.end()); }

        auto& m2s = m1it->m2s;
        auto m2it = std::find_if(m2s.begin(), m2s.end(),
                                 [&](const ReuseM2Branch& b){ return b.m2 == c.m[1]; });
        if (m2it == m2s.end()) { m2s.push_back({c.m[1], {}}); m2it = std::prev(m2s.end()); }

        m2it->leaves.push_back({ci, c.m, c.bucket});
    }
    return m1s;
}

BranchPlan build_branch_plan() {
    const auto m1s = group_isotropic();
    BranchPlan plan;
    int t_slot = 0;   // 0..20 leaf slot
    int T_slot = 0;   // 0..8  (m1,m2) slot
    for (int i1 = 0; i1 < static_cast<int>(m1s.size()); ++i1) {
        plan.m1_axis[i1] = m1s[i1].m1;
        for (const auto& b2 : m1s[i1].m2s) {
            plan.m2_axis.push_back(b2.m2);
            plan.tau_to_T.push_back(i1);            // this T-slot fans from τ-slot i1
            for (const auto& leaf : b2.leaves) {
                plan.T_to_t.push_back(T_slot);       // this t-slot fans from T-slot
                plan.leaves[t_slot] = LeafSpec{leaf.comp_idx, leaf.m, leaf.bucket};
                ++t_slot;
            }
            ++T_slot;
        }
    }
    return plan;
}

std::vector<BranchPlan> walk_plans(const BranchPlan& plan, int width) {
    if (width == plan.B_t) return {plan};
    if (width != 7)
        throw std::invalid_argument("walk_plans: width " + std::to_string(width) +
                                    " (want 21 or 7)");
    // One subtree per m1 (τ-slot): its T-slots and leaves, renumbered locally.
    // Depth-first flatten ⇒ each subtree is a contiguous run of T- and t-slots.
    std::vector<BranchPlan> subs;
    for (int i1 = 0; i1 < plan.B_tau; ++i1) {
        BranchPlan s;
        s.B_tau = 1;
        s.m1_axis = {plan.m1_axis[i1], -1, -1};
        s.m2_axis.clear(); s.tau_to_T.clear(); s.T_to_t.clear();
        std::vector<int> local_T(plan.B_T, -1);
        for (int T = 0; T < plan.B_T; ++T) {
            if (plan.tau_to_T[T] != i1) continue;
            local_T[T] = static_cast<int>(s.m2_axis.size());
            s.m2_axis.push_back(plan.m2_axis[T]);
            s.tau_to_T.push_back(0);
        }
        int nt = 0;
        for (int t = 0; t < plan.B_t; ++t) {
            if (local_T[plan.T_to_t[t]] < 0) continue;
            s.T_to_t.push_back(local_T[plan.T_to_t[t]]);
            s.leaves[nt++] = plan.leaves[t];
        }
        s.B_T = static_cast<int>(s.m2_axis.size());
        s.B_t = nt;
        if (s.B_t != width)   // 3 m1 × 7 leaves; anything else is a new tree
            throw std::logic_error("walk_plans: m1 subtree " + std::to_string(i1) + " has " +
                                   std::to_string(s.B_t) + " leaves, expected 7");
        subs.push_back(std::move(s));
    }
    return subs;
}

// ===========================================================================
// Orientation-reuse walk (3→9→21 prefix tree, B=1): the B=1 production walk
// (LHC-II ESA lives here). Depth-first τ→T→t carry-through with the
// isotropic fanout shared by prefix: set_initial+μ0 once per m1 (3), μ1+T once
// per (m1,m2) (9), the t-scan + μ⁻(m4) detection per leaf (21). Only 2 device
// snapshots live at once (tau_snap, tw_snap). Its independent oracle is the
// batched walk below: filter OFF ⇒ bit-identical, filter ON ⇒ within filter
// tolerance (test_2des_reuse). The historical flat 21-tree that both were
// once checked against is gone; its numbers live on as the
// benchmark_3sites_2des baseline.
// ===========================================================================
namespace {

using Cd = std::complex<double>;

// FINE-log label helpers: "tau=<fs> (<it>) T=<fs> (<k>) t=<fs> (<j>) fs".
// Axes not yet started (their pulse not yet applied) show "-".
std::string fmt_num(double v) {
    char b[32];
    long long iv = static_cast<long long>(v);
    if (static_cast<double>(iv) == v) std::snprintf(b, sizeof b, "%lld", iv);
    else                              std::snprintf(b, sizeof b, "%.1f", v);
    return std::string(b);
}
std::string ax_lbl(const char* nm, int idx, double d) {
    if (idx < 0) return std::string(nm) + "=-";
    return std::string(nm) + "=" + fmt_num(idx * d) + " (" + std::to_string(idx) + ")";
}
std::string axv_lbl(const char* nm, int idx, double val) {
    if (idx < 0) return std::string(nm) + "=-";
    return std::string(nm) + "=" + fmt_num(val) + " (" + std::to_string(idx) + ")";
}
std::string seg_prefix(const TwoDESRequest& req, const std::string& tag,
                       int it, int k, int j) {
    return tag + " " + ax_lbl("tau", it, req.dtau_fs) + " "
         + axv_lbl("T", k, k >= 0 ? req.tw_fs[k] : 0.0) + " "
         + ax_lbl("t", j, req.dt_detect_fs) + " fs : ";
}
std::string leaf_label(const ReuseLeaf& leaf) {
    static const char ax[] = "xyz";
    std::string s = "(";
    for (int i = 0; i < 4; ++i) s += ax[leaf.m[i]];
    s += ")->";
    bool first = true;
    for (int ip = 0; ip < 3; ++ip) {
        if (!leaf.bucket[ip]) continue;
        if (!first) s += ",";
        s += "I" + std::to_string(ip + 1);
        first = false;
    }
    return s;
}

// Invariants shared across the whole (signal,pathway) walk. Bundled so the walk
// helpers stay shallow.
struct ReuseCtx {
    HEOMTrajectory& traj;
    heom::Detector& det;
    const PathSpec& spec;
    const std::array<const Eigen::MatrixXcd*, 3>& Rax;
    const TwoDESRequest& req;
    int N;
    ManifoldBlock gg;                   // (g,g): where |g><g| lives before μ0
    std::array<ManifoldBlock, 3> blk;   // [τ, T, t] — pathway_segment_blocks()
    std::string base;       // "rp SE" etc — FINE-log tag stem
    bool log_on;
    int print_interval;
};

Eigen::MatrixXcd rsel(const ReuseCtx& c, int axis, bool raise) {
    return raise ? Eigen::MatrixXcd(*c.Rax[axis])
                 : Eigen::MatrixXcd(c.Rax[axis]->adjoint());
}

// Innermost primitive: one leaf's t-scan + μ⁻(m4) detection. Restores the clean
// T-snapshot, applies μ2, scans t with async device detection, checks
// divergence once, flushes n_t samples into grid_c at the (it,k) plane. The
// batched tscan_batched is this loop over 21 slots at once; keep them in step.
// Returns true iff diverged.
bool detect_leaf(ReuseCtx& c, const DeviceHierarchySnapshot& tw_snap,
                 const ReuseLeaf& leaf, int it, int k, std::vector<Cd>& grid_c) {
    const TwoDESRequest& req = c.req;
    c.traj.restore_device(tw_snap);
    c.traj.apply_dipole({rsel(c, leaf.m[2], c.spec.pulses[2].raise)},
                        c.spec.pulses[2].side, c.blk[2]);
    const Eigen::MatrixXcd mu_det = c.Rax[leaf.m[3]]->adjoint();  // μ⁻(m4)
    c.det.begin({mu_det}, static_cast<size_t>(req.n_t));
    int t_since_print = 0;
    for (int j = 0; j < req.n_t; ++j) {
        if (j > 0) {
            c.traj.propagate_steps(req.steps_per_t, /*check_div=*/false);
            t_since_print += req.steps_per_t;
        }
        c.det.sample(static_cast<size_t>(j));
        if (c.log_on && c.print_interval > 0 && t_since_print >= c.print_interval) {
            c.traj.set_log_prefix(
                seg_prefix(req, c.base + " " + leaf_label(leaf), it, k, j));
            c.traj.log_stats();
            t_since_print = 0;
        }
    }
    const bool div = c.traj.check_divergence_now(
        (req.n_t - 1) * req.dt_detect_fs, it * req.n_tw + k);
    c.det.flush(grid_c.data() + static_cast<size_t>(it * req.n_tw + k) * req.n_t);
    return div;
}

// One τ point's (m2,k,leaf) fanout: for each m2 branch restore the clean τ
// snapshot, apply μ1, walk the T chain (snapshot per k), and run every leaf.
// Returns true iff any leaf diverged (short-circuits the walk).
bool run_tau_fanout(ReuseCtx& c, const DeviceHierarchySnapshot& tau_snap,
                    DeviceHierarchySnapshot& tw_snap, const ReuseM1Branch& b1,
                    int it, std::vector<std::vector<Cd>>& leaf_grid) {
    const TwoDESRequest& req = c.req;
    for (const auto& b2 : b1.m2s) {
        c.traj.restore_device(tau_snap);
        c.traj.apply_dipole({rsel(c, b2.m2, c.spec.pulses[1].raise)},
                            c.spec.pulses[1].side, c.blk[1]);
        for (int k = 0; k < req.n_tw; ++k) {
            if (req.tw_gap_steps[k] > 0) {
                if (c.log_on)
                    c.traj.set_log_prefix(seg_prefix(req, c.base + " m1m2", it, k, -1));
                c.traj.propagate_steps(req.tw_gap_steps[k]);
                if (c.log_on) c.traj.log_stats();
            }
            c.traj.snapshot_device(tw_snap);
            for (const auto& leaf : b2.leaves)
                if (detect_leaf(c, tw_snap, leaf, it, k, leaf_grid[leaf.comp_idx]))
                    return true;
            // Back to the clean μ1-state at T[k]: the next k advances the T chain
            // from here (each detect_leaf left the device on a t-propagated
            // state).
            c.traj.restore_device(tw_snap);
        }
    }
    return false;
}

// The full 3→9→21 walk for one (signal,pathway). Two snapshots live at once
// (tau_snap across the m2 fanout, tw_snap across leaves): 2·M·N² on top of the
// B=1 trajectory. Emits one coarse heartbeat per m1 group (3 lines / (s,p)).
void run_pathway_reuse(ReuseCtx& c, const std::vector<ReuseM1Branch>& m1s,
                       std::vector<std::vector<Cd>>& leaf_grid) {
    const TwoDESRequest& req = c.req;
    c.traj.set_cadence_logging(false);
    DeviceHierarchySnapshot tau_snap, tw_snap;
    static const char ax[] = "xyz";
    for (const auto& b1 : m1s) {
        const auto t0 = std::chrono::steady_clock::now();
        c.traj.set_initial({ground_projector(c.N)}, c.gg);
        c.traj.apply_dipole({rsel(c, b1.m1, c.spec.pulses[0].raise)},
                            c.spec.pulses[0].side, c.blk[0]);
        bool diverged = false;
        for (int it = 0; it < req.n_tau && !diverged; ++it) {
            if (it > 0) {
                if (c.log_on)
                    c.traj.set_log_prefix(seg_prefix(req, c.base + " m1", it, -1, -1));
                c.traj.propagate_steps(req.steps_per_tau);
                if (c.log_on) c.traj.log_stats();
            }
            c.traj.snapshot_device(tau_snap);
            diverged = run_tau_fanout(c, tau_snap, tw_snap, b1, it, leaf_grid);
            c.traj.restore_device(tau_snap);
        }
        const double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "  [reuse m1=" << ax[b1.m1] << "] " << c.base << " N="
                  << c.N << "  (" << heom::console::format_computation_time(s)
                  << ")\n" << std::flush;
        if (diverged) break;
    }
}

// ===========================================================================
// Batched walk (3→9→21 lockstep). Same τ→T→t carry-through as the reuse walk,
// but each layer's fanout runs as ONE batched propagation over active_batch_
// slots (3 / 9 / 21). Same L (field-free between pulses), per-slot μ picks the
// axis — physics identical, only the execution is batched. Fills the SAME
// leaf_grid[comp_idx] layout as run_pathway_reuse ⇒ identical accumulation ⇒
// bit-exact vs reuse when filtering is off.
// ===========================================================================

// One T point's t-scan for all 21 leaf slots: expand tw_snap 9→21, apply μ2(m3)
// per leaf slot, scan t with batched device detection. Returns true iff diverged.
bool tscan_batched(ReuseCtx& c, const DeviceHierarchySnapshot& tw_snap,
                   const BranchPlan& plan, int it, int k) {
    const TwoDESRequest& req = c.req;
    c.traj.expand_slots(tw_snap, plan.T_to_t);           // 9→21 (sets active=21)
    std::vector<Eigen::MatrixXcd> mu2(plan.B_t);
    for (int b = 0; b < plan.B_t; ++b)
        mu2[b] = rsel(c, plan.leaves[b].m[2], c.spec.pulses[2].raise);
    c.traj.apply_dipole(mu2, c.spec.pulses[2].side, c.blk[2]);
    const size_t base = static_cast<size_t>(it * req.n_tw + k) * req.n_t;
    int t_since_print = 0;                                // mirror detect_leaf
    for (int j = 0; j < req.n_t; ++j) {
        if (j > 0) {
            c.traj.propagate_steps(req.steps_per_t, /*check_div=*/false);
            t_since_print += req.steps_per_t;
        }
        c.det.sample(base + j);                          // writes 21 traces at once
        // t-segment stats (active=21): union |A|/|E| and per-slot |A_b|.
        if (c.log_on && c.print_interval > 0 && t_since_print >= c.print_interval) {
            c.traj.set_log_prefix(seg_prefix(req, c.base + " t", it, k, j));
            c.traj.log_stats();
            t_since_print = 0;
        }
    }
    return c.traj.check_divergence_now((req.n_t - 1) * req.dt_detect_fs,
                                       it * req.n_tw + k);
}

// The full batched 3→9→21 walk for one (signal,pathway). Only τ-snap(B=3) +
// tw-snap(B=9) live at once (t layer has no snapshot) — same footprint rule as
// the reuse walk.
void run_pathway_batched(ReuseCtx& c, const BranchPlan& plan,
                         std::vector<std::vector<Cd>>& leaf_grid) {
    const TwoDESRequest& req = c.req;
    const int N = c.N;
    c.traj.set_cadence_logging(false);
    const auto t0 = std::chrono::steady_clock::now();

    // τ: 3 slots = ground + μ0(m1); set_initial sets the width from the count.
    std::vector<Eigen::MatrixXcd> grounds(plan.B_tau, ground_projector(N));
    c.traj.set_initial(grounds, c.gg);
    std::vector<Eigen::MatrixXcd> mu0(plan.B_tau);
    for (int b = 0; b < plan.B_tau; ++b)
        mu0[b] = rsel(c, plan.m1_axis[b], c.spec.pulses[0].raise);
    c.traj.apply_dipole(mu0, c.spec.pulses[0].side, c.blk[0]);

    // Detection: 21 leaf μ_det(m4)=(μ⁺)†, opened once for the whole tree; one
    // flat [21·(n_tau·n_tw·n_t)] device grid, one D2H at the end.
    std::vector<Eigen::MatrixXcd> mu_det(plan.B_t);
    for (int b = 0; b < plan.B_t; ++b)
        mu_det[b] = c.Rax[plan.leaves[b].m[3]]->adjoint();
    const size_t grid_sz = static_cast<size_t>(req.n_tau) * req.n_tw * req.n_t;
    c.det.begin(mu_det, grid_sz);

    DeviceHierarchySnapshot tau_snap, tw_snap;
    bool diverged = false;
    for (int it = 0; it < req.n_tau && !diverged; ++it) {
        if (it > 0) {                                            // τ advance, active=3
            if (c.log_on)
                c.traj.set_log_prefix(seg_prefix(req, c.base + " m1", it, -1, -1));
            c.traj.propagate_steps(req.steps_per_tau);
            if (c.log_on) c.traj.log_stats();
        }
        c.traj.snapshot_device(tau_snap);            // B=3

        // 3→9 + μ1(m2). expand_slots sets active=9.
        c.traj.expand_slots(tau_snap, plan.tau_to_T);
        std::vector<Eigen::MatrixXcd> mu1(plan.B_T);
        for (int b = 0; b < plan.B_T; ++b)
            mu1[b] = rsel(c, plan.m2_axis[b], c.spec.pulses[1].raise);
        c.traj.apply_dipole(mu1, c.spec.pulses[1].side, c.blk[1]);

        for (int k = 0; k < req.n_tw; ++k) {
            if (req.tw_gap_steps[k] > 0) {                        // T advance, active=9
                if (c.log_on)
                    c.traj.set_log_prefix(seg_prefix(req, c.base + " m1m2", it, k, -1));
                c.traj.propagate_steps(req.tw_gap_steps[k]);
                if (c.log_on) c.traj.log_stats();
            }
            c.traj.snapshot_device(tw_snap);         // B=9
            if (tscan_batched(c, tw_snap, plan, it, k)) { diverged = true; break; }
            c.traj.restore_device(tw_snap);          // back to B=9 at T[k] for next k
        }
        c.traj.restore_device(tau_snap);             // back to B=3 for next τ advance
    }

    // One D2H of the whole [21][tree]; scatter into per-component leaf grids so
    // the driver's accumulation matches the reuse layout exactly.
    std::vector<Cd> flat(static_cast<size_t>(plan.B_t) * grid_sz);
    c.det.flush(flat.data());
    for (int b = 0; b < plan.B_t; ++b) {
        const int ci = plan.leaves[b].comp_idx;      // == b (canonical), explicit
        std::copy(flat.begin() + static_cast<size_t>(b) * grid_sz,
                  flat.begin() + static_cast<size_t>(b + 1) * grid_sz,
                  leaf_grid[ci].begin());
    }

    const double s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "  [batch B=" << plan.B_t << " m1=" << plan.m1_axis[0]
              << (plan.B_tau > 1 ? "..": "") << "] " << c.base << " N=" << N
              << (diverged ? "  DIVERGED" : "")
              << "  (" << heom::console::format_computation_time(s) << ")\n"
              << std::flush;
}

// One (signal, pathway) cell of the output, resolved before any propagation.
struct PathwayJob {
    int s, p;                            // index into req.signals / req.pathways
    const PathSpec& spec;
    const SystemContext& ctx;
    ManifoldLayout lay;
    std::array<ManifoldBlock, 3> blk;    // [τ, T, t] — pathway_segment_blocks()
};

// Budget guard for the walk the config asked for. The alloc plan only covers
// trajectory-owned buffers; the driver-owned extras it cannot see are folded
// INTO the budget (not out of gfree) so the printed shortfall report stays
// honest:
//   batched (B=21): tau_snap (B=3) + tw_snap (B=9), each a full ρ copy
//     ⇒ 12·M·Nb, plus the [21·tree] device detection grid;
//   reuse (B=1): tau_snap + tw_snap at B=1 ⇒ 2·M·Nb. SystemContext's check
//     never saw them: S0 D=42 printed "87.8% of free" and then the snapshot
//     cudaMalloc threw.
//   The mixed build snapshots the Kahan carry alongside ρ, doubling both.
// Nb = ctx.rho_capacity(): the largest block this run stores per ADO, the same
// number the trajectory buffers are sized by.
// Exceeding the budget is an ERROR, not a reason to pick a different walk: a
// run that silently drops from B=21 to B=1 is a run whose wall-clock the user
// did not sign up for. Say what does not fit and what to
// change.
// Slot widths come from the plan, not literals: the batched walk holds a
// τ-snapshot of B_tau slots and a T-snapshot of B_T slots and detects B_t
// leaves at once; the reuse walk holds the same two snapshots at B=1.
// Width 7 runs the same walk per m1 subtree (B_tau=1, B_T=3, B_t=7): 4 snapshot
// slots and a [7·tree] detection grid, one subtree at a time.
heom::MemoryBudget walk_budget(const SystemContext& ctx, const BranchPlan& bplan,
                               int width, size_t grid_sz) {
    const size_t MNb = static_cast<size_t>(ctx.M()) * ctx.rho_capacity();
    auto budget = heom::estimate_budget(ctx.config(), static_cast<size_t>(ctx.M()), width,
                                        heom::Resident::SystemTables);
    size_t snap_slots = 2;                                     // reuse: tau_snap + tw_snap
    if (width > 1) {
        const BranchPlan p = walk_plans(bplan, width).front();
        snap_slots = static_cast<size_t>(p.B_tau + p.B_T);
    }
    const size_t per_slot = heom::kMixedPrecision ? 2 : 1;
    size_t extra = snap_slots * per_slot * MNb * sizeof(heom::complex_t);
    if (width > 1)
        extra += static_cast<size_t>(width) * grid_sz * sizeof(heom::complex_t);
    budget.driver    += extra;
    budget.gpu_total += extra;
    return budget;
}

void check_walk_budget(const SystemContext& ctx, const TwoDESRequest& req,
                       const BranchPlan& bplan, size_t grid_sz) {
    auto budget = walk_budget(ctx, bplan, req.batch, grid_sz);
    size_t gfree = 0, gtot = 0;
    heom::query_gpu_memory(gfree, gtot);
    std::string budget_err;
    if (heom::check_budget(budget, ctx.config(), gfree, budget_err)
            != heom::BudgetCheck::Exceeds)
        return;
    heom::print_budget(budget, gfree, gtot);
    // Say what every width would need, so the user can pick one that fits.
    std::string widths;
    for (int w : {21, 7, 1}) {
        const double mib = walk_budget(ctx, bplan, w, grid_sz).gpu_total / 1048576.0;
        widths += (widths.empty() ? "" : ", ") + std::to_string(w) + ": " +
                  std::to_string(static_cast<long long>(mib + 0.5)) + " MiB";
    }
    throw std::runtime_error(
        "compute_2des: 2des.batch: " + std::to_string(req.batch) +
        " walk exceeds GPU budget (estimated need per width — " + widths + "; " +
        std::to_string(gfree / 1048576) + " MiB free): " + budget_err);
}

// Everything that can reject a (signal, pathway) without computing it: the
// names, the manifold ladder, the basis/context pairing and the GPU budget.
// All cells are checked before the first one runs — a check inside the loop
// threw away every pathway already finished (SE and GSB done, ESA over
// budget ⇒ hours of GPU time, nothing written).
std::vector<PathwayJob> plan_pathways(const SystemContext& sys_single,
                                      const SystemContext& sys_double,
                                      const TwoDESRequest& req,
                                      const BranchPlan& bplan, size_t grid_sz) {
    auto unique = [](const std::vector<std::string>& v, const char* what) {
        for (size_t i = 0; i < v.size(); ++i)
            if (std::count(v.begin(), v.end(), v[i]) > 1)
                throw std::invalid_argument(std::string("compute_2des: ") + what +
                                            " '" + v[i] + "' listed twice");
    };
    unique(req.signals, "signal");
    unique(req.pathways, "pathway");

    std::vector<PathwayJob> jobs;
    for (int s = 0; s < static_cast<int>(req.signals.size()); ++s) {
        const std::string& sig = req.signals[s];
        if (sig != "rephasing" && sig != "nonrephasing")
            throw std::invalid_argument(
                "compute_2des: unknown signal '" + sig +
                "' (want rephasing/nonrephasing)");
        for (int p = 0; p < static_cast<int>(req.pathways.size()); ++p) {
            const PathSpec& spec = find_path(sig == "rephasing", req.pathways[p]);
            const bool dbl = (spec.basis == BasisType::DOUBLE_EXCITED);
            const SystemContext& ctx = dbl ? sys_double : sys_single;
            // The blocks are declarations the kernel takes literally, so they
            // must be derived from the SAME basis the trajectory runs in. NOT
            // assert(): shipped builds are Release, where assert() compiles to
            // nothing.
            if (spec.basis != ctx.config().physics.basis)
                throw std::logic_error(
                    "compute_2des: pathway basis / SystemContext mismatch");
            const auto lay = ManifoldLayout::from(
                spec.basis, ctx.config().physics.n_states, ctx.N());
            jobs.push_back({s, p, spec, ctx, lay, pathway_segment_blocks(spec, lay)});
        }
    }
    for (const auto& j : jobs) check_walk_budget(j.ctx, req, bplan, grid_sz);
    return jobs;
}

}  // namespace

// Try 21 → 7 → 1, first that fits under the same usable-free
// rule as check_budget. Pure (no GPU query) so unit tests can mock free bytes.
int pick_batch_width(const std::array<size_t, 3>& need_bytes,
                     size_t gpu_free,
                     std::string& reason,
                     double safety_margin) {
    if (!(safety_margin > 0.0 && safety_margin <= 1.0))
        throw std::invalid_argument(
            "pick_batch_width: safety_margin must be in (0, 1], got " +
            std::to_string(safety_margin));
    auto mib = [](size_t bytes) {
        return static_cast<long long>(bytes / 1048576.0 + 0.5);
    };
    std::string needs_txt;
    for (size_t i = 0; i < kBatchWidthCandidates.size(); ++i) {
        if (i) needs_txt += ", ";
        needs_txt += std::to_string(kBatchWidthCandidates[i]) + ": " +
                     std::to_string(mib(need_bytes[i])) + " MiB";
    }

    if (gpu_free == 0) {
        // Match check_walk_budget: Unknown capacity is not Exceeds.
        reason = "auto selected B=21 (GPU free unknown; needs — " + needs_txt + ")";
        return 21;
    }
    const size_t usable = static_cast<size_t>(gpu_free * safety_margin);
    for (size_t i = 0; i < kBatchWidthCandidates.size(); ++i) {
        if (need_bytes[i] <= usable) {
            const int w = kBatchWidthCandidates[i];
            reason = "auto selected B=" + std::to_string(w) + " (need " +
                     std::to_string(mib(need_bytes[i])) + " MiB <= usable " +
                     std::to_string(mib(usable)) + " MiB of " +
                     std::to_string(mib(gpu_free)) + " MiB free; widths — " +
                     needs_txt + ")";
            return w;
        }
    }
    throw std::runtime_error(
        "compute_2des: 2des.batch: auto — no walk width fits GPU budget "
        "(estimated need per width — " + needs_txt + "; " +
        std::to_string(mib(gpu_free)) + " MiB free, usable " +
        std::to_string(mib(usable)) + " MiB with " +
        std::to_string(static_cast<int>(safety_margin * 100 + 0.5)) +
        "% safety margin)");
}


TwoDESResult compute_2des(const SystemContext& sys_single,
                          const SystemContext& sys_double,
                          const TwoDESRequest& req) {
    using Cd = std::complex<double>;

    if (!req.dipole_single || !req.dipole_double)
        throw std::invalid_argument("compute_2des: null dipole operator(s)");

    TwoDESResult out;
    out.n_tau = req.n_tau;
    out.n_tw  = req.n_tw;
    out.n_t   = req.n_t;
    out.n_signal  = static_cast<int>(req.signals.size());
    out.n_pathway = static_cast<int>(req.pathways.size());
    // Isotropic orientational average: 3 rotational invariants (I1,I2,I3).
    out.n_component = 3;
    out.tau_fs.resize(req.n_tau);
    out.tw_fs.resize(req.n_tw);
    out.t_fs.resize(req.n_t);
    for (int i = 0; i < req.n_tau; ++i) out.tau_fs[i] = i * req.dtau_fs;
    out.tw_fs = req.tw_fs;
    for (int j = 0; j < req.n_t;   ++j) out.t_fs[j]   = j * req.dt_detect_fs;
    out.resp.assign(static_cast<size_t>(out.n_component) * out.n_signal *
                        out.n_pathway * req.n_tau * req.n_tw * req.n_t,
                    Cd(0.0, 0.0));

    // Per-basis raising operators μ⁺ along x/y/z (component projections pick one
    // per interaction); μ⁻ = (μ⁺)†.
    const std::array<const Eigen::MatrixXcd*, 3> Rsingle = {
        &req.dipole_single->mu_x, &req.dipole_single->mu_y, &req.dipole_single->mu_z};
    const std::array<const Eigen::MatrixXcd*, 3> Rdouble = {
        &req.dipole_double->mu_x, &req.dipole_double->mu_y, &req.dipole_double->mu_z};

    const size_t grid_sz = static_cast<size_t>(req.n_tau) * req.n_tw * req.n_t;

    // Two-level 2DES log (mirrors absorption's "[abs x] propagating" always-on
    // header + print_interval-gated per-step lines):
    //   - COARSE: the header + the walk's own heartbeat lines (one per m1 group
    //     for reuse, one per (signal,pathway) for batched), printed AFTER each
    //     finishes with its elapsed time so nothing the trajectory emits mid-run
    //     interleaves into the label.
    //   - FINE: the per-step "... fs : ADOs |A|/|E|/M ..." lines inside each
    //     walk, gated by print_interval (==0 ⇒ off) via `log_on`.
    const int  print_interval = sys_single.config().time.print_interval;
    const bool log_on         = (print_interval > 0);
    const int  n_comp_trees   = static_cast<int>(isotropic_components().size());

    auto join = [](const std::vector<std::string>& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) { if (i) s += ", "; s += v[i]; }
        return s;
    };

    // Resolve batch: auto → 21/7/1; explicit widths unchanged.
    // Local copy so plan_pathways / the walk see a concrete width; out records
    // what was chosen for HDF5 (spec.batch may still be kBatchAuto).
    TwoDESRequest resolved = req;
    const BranchPlan bplan = build_branch_plan();
    if (resolved.batch == kBatchAuto) {
        size_t gfree = 0, gtot = 0;
        heom::query_gpu_memory(gfree, gtot);
        std::array<size_t, 3> needs{};
        for (size_t i = 0; i < kBatchWidthCandidates.size(); ++i) {
            const int w = kBatchWidthCandidates[i];
            needs[i] = std::max(
                walk_budget(sys_single, bplan, w, grid_sz).gpu_total,
                walk_budget(sys_double, bplan, w, grid_sz).gpu_total);
        }
        std::string reason;
        resolved.batch = pick_batch_width(needs, gfree, reason);
        out.batch_auto_reason = reason;
    } else if (resolved.batch != 1 && resolved.batch != 7 && resolved.batch != 21) {
        throw std::invalid_argument(
            "compute_2des: batch width " + std::to_string(resolved.batch) +
            " (want 1, 7, 21 or auto)");
    }
    out.batch_width = resolved.batch;

    std::cout << "\n======== 2DES (carry-through) ========\n"
              << "  signals    : " << join(resolved.signals)
              << "  (" << out.n_signal << ")\n"
              << "  pathways   : " << join(resolved.pathways)
              << "  (" << out.n_pathway << ")\n";
    std::cout << "  orientation: isotropic -> I1,I2,I3 ("
              << n_comp_trees << " components)\n";
    std::cout << "  grid       : tau " << resolved.n_tau << " x " << resolved.dtau_fs
              << "fs | ";
    if (resolved.n_tw == 1) std::cout << "T 1 (T=" << resolved.tw_fs[0] << "fs)";
    else std::cout << "T " << resolved.n_tw << " pts ["
                   << resolved.tw_fs.front() << ".." << resolved.tw_fs.back() << "]fs";
    std::cout << " | t " << resolved.n_t << " x " << resolved.dt_detect_fs << "fs\n"
              << "  walk       : " << (resolved.batch == 21 ? "batched 3->9->21 (B=21)"
                                     : resolved.batch == 7   ? "batched per m1 subtree 1->3->7 (B=7, x3)"
                                                            : "reuse 3->9->21 (B=1)")
              << "  (print_interval=" << print_interval << ")\n";
    if (!out.batch_auto_reason.empty())
        std::cout << "  batch auto : " << out.batch_auto_reason << "\n";
    std::cout << "======================================\n" << std::flush;

    const int B = resolved.batch;
    const std::vector<BranchPlan> wplans =
        (B > 1) ? walk_plans(bplan, B) : std::vector<BranchPlan>{};
    for (const PathwayJob& job :
         plan_pathways(sys_single, sys_double, resolved, bplan, grid_sz)) {
        const int s = job.s, p = job.p;
        const std::string& sig = req.signals[s];
        const bool rp = (sig == "rephasing");
        const PathSpec& spec = job.spec;
        const SystemContext& ctx = job.ctx;
        const auto& Rax = (spec.basis == BasisType::DOUBLE_EXCITED) ? Rdouble : Rsingle;
        const int N = ctx.N();

        // One trajectory per (signal,pathway), reused across the whole walk
        // via set_initial (never `new` per tree → cudaMalloc thrash). Only
        // one HEOMTrajectory is alive at any time.
        HEOMTrajectory traj(ctx, B);
        // Its detector lives just as long: grow-only buffers, so the
        // per-tree begin() re-uploads μ_det but never re-allocates.
        heom::Detector det(traj);

        // Both walks fill the same per-component leaf grids
        // (leaf_grid[ci] = isotropic_components()[ci]'s tree) and then run
        // the one accumulation below, so their float-add order into
        // out.resp is byte-identical — the bit-exact parity gate depends on
        // this.
        //
        // The five-deep nest (ci→ip→it→k→j) is THE SHAPE THE GATE LOCKS, not
        // a loop nobody got around to tidying. Floating-point addition is not
        // associative, so collapsing, fusing, reordering, or parallelising
        // these loops changes the sum and breaks reuse/batched parity — which
        // is the only evidence we have that the two walks compute the same
        // physics — and moves the committed golden baseline. Want it faster?
        // Prove the new order is bit-identical first. Otherwise leave it
        // exactly as it is.
        const auto& comps = isotropic_components();
        std::vector<std::vector<Cd>> leaf_grid(comps.size(),
                                               std::vector<Cd>(grid_sz));
        auto accumulate = [&]() {
            for (int ci = 0; ci < static_cast<int>(comps.size()); ++ci) {
                const auto& c = comps[ci];
                for (int ip = 0; ip < 3; ++ip) {
                    if (!c.bucket[ip]) continue;
                    for (int it = 0; it < req.n_tau; ++it)
                    for (int k = 0; k < req.n_tw; ++k)
                    for (int j = 0; j < req.n_t; ++j)
                        out.resp[out.idx(ip, s, p, it, k, j)] +=
                            spec.sign *
                            leaf_grid[ci][static_cast<size_t>(it * req.n_tw + k)
                                          * req.n_t + j];
                }
            }
        };

        ReuseCtx rc{traj, det, spec, Rax, req, N, ground_block(job.lay), job.blk,
                    std::string(rp ? "rp " : "nr ") + spec.name,
                    log_on, print_interval};
        if (B > 1)
            for (const BranchPlan& wp : wplans)                       // 1 plan (21) or 3 subtrees (7)
                run_pathway_batched(rc, wp, leaf_grid);               // batched lockstep
        else
            run_pathway_reuse(rc, group_isotropic(), leaf_grid);      // reuse, prefix-shared
        accumulate();

        if (traj.diverged())
            throw std::runtime_error(
                std::string("compute_2des: trajectory diverged (") +
                sig + " " + spec.name + ")");
    }
    return out;
}

}  // namespace heom::spectra
