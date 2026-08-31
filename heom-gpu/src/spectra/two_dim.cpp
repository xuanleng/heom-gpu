#include "spectra/two_dim.hpp"

#include "system_context.hpp"
#include "heom_trajectory.hpp"
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
    req.signals      = spec.signals;
    req.pathways     = spec.pathways;
    req.orient_reuse = spec.orient_reuse;
    req.batch        = spec.batch;
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

ManifoldLayout ManifoldLayout::from(BasisType basis, int n_states, int N) {
    ManifoldLayout lay;
    int want = 0;
    switch (basis) {
        case BasisType::WITH_GROUND:      // g=[0,1), e=[1,1+n)
            lay.n_level = 2;
            lay.bound = {0, 1, 1 + n_states, 1 + n_states};
            want = 1 + n_states;
            break;
        case BasisType::DOUBLE_EXCITED:   // g=[0,1), e=[1,1+n), f=[1+n,N)
            lay.n_level = 3;
            lay.bound = {0, 1, 1 + n_states, N};
            want = 1 + n_states + n_states * (n_states - 1) / 2;
            break;
        default:
            // EXCITED_ONLY has no ground manifold, so |g><g| — where every 2DES
            // pathway starts — is not representable. 2DES never builds it.
            throw std::invalid_argument(
                "ManifoldLayout::from: 2DES needs a ground state in the basis");
    }
    // The one mistake this argument exists to catch: compute_2des holds a
    // single-excitation and a double-excitation context at the same time, so
    // the wrong N is always within reach, and a layout built from it would
    // declare blocks that are quietly the wrong size.
    if (N != want)
        throw std::invalid_argument(
            "ManifoldLayout::from: N=" + std::to_string(N) + " does not match "
            "n_states=" + std::to_string(n_states) + " for this basis (want "
            + std::to_string(want) + ") — wrong SystemContext?");
    return lay;
}

ManifoldBlock ManifoldLayout::block(int row_level, int col_level) const {
    if (row_level < 0 || row_level >= n_level ||
        col_level < 0 || col_level >= n_level)
        throw std::invalid_argument("ManifoldLayout::block: level out of range");
    return {bound[row_level], bound[row_level + 1] - bound[row_level],
            bound[col_level], bound[col_level + 1] - bound[col_level]};
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
    if (r < 0 || r >= lay.n_level || c < 0 || c >= lay.n_level)
        throw std::invalid_argument(
            "apply_pulse: dipole walks off the manifold ladder — the pulse "
            "table and the basis disagree");
    return {r, c};
}

std::array<ManifoldBlock, 3> pathway_segment_blocks(const PathSpec& spec,
                                                    const ManifoldLayout& lay) {
    std::pair<int, int> lv{0, 0};        // |g><g|
    std::array<ManifoldBlock, 3> out{};
    for (int p = 0; p < 3; ++p) {
        lv = apply_pulse(lay, lv, spec.pulses[p]);
        out[p] = lay.block(lv.first, lv.second);
    }
    return out;                          // [after μ1, after μ2, after μ3]
}                                        //  = [τ segment, T segment, t segment]

namespace {

// The ONLY place the four positional ints of set_manifold_block are spelled
// out. Nine call sites pass a ManifoldBlock; none of them can transpose it
// into a block — (e,f) instead of (f,e) — that no pathway ever occupies.
void apply_block(HEOMTrajectory& traj, const ManifoldBlock& b) {
    traj.set_manifold_block(b.row_off, b.n_row, b.col_off, b.n_col);
}

// |0><0| — the ground projector. In both WITH_GROUND and DOUBLE_EXCITED the
// ground state sits at Hilbert index 0 (compute_hilbert_dim layout).
Eigen::MatrixXcd ground_projector(int N) {
    Eigen::MatrixXcd P = Eigen::MatrixXcd::Zero(N, N);
    P(0, 0) = std::complex<double>(1.0, 0.0);
    return P;
}

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

// Run one depth-first τ→tw→t carry-through tree for a single dipole projection,
// writing the full grid[(it*n_tw+k)*n_t+j]. The trajectory is reused across
// calls (set_initial resets internal_step_), so the caller keeps exactly one
// one hierarchy resident per basis.
void run_pathway_tree(HEOMTrajectory& traj, const PathSpec& spec,
                      const Eigen::MatrixXcd& mu0, const Eigen::MatrixXcd& mu1,
                      const Eigen::MatrixXcd& mu2, const Eigen::MatrixXcd& mu_det,
                      const TwoDESRequest& req, int N,
                      const std::array<ManifoldBlock, 3>& blk,
                      std::vector<std::complex<double>>& grid,
                      const std::string& tree_tag, bool log_on,
                      int print_interval) {
    // Tag every per-step filtering line with the PHYSICAL 2D coordinates being
    // propagated toward: "tau=<fs> (<it>) T=<fs> (<k>) t=<fs> (<j>) fs". Axes
    // not yet started (their dipole pulse not yet applied) show "-". Only built
    // when logging is on (print_interval>0); else the trajectory never prints.
    auto fmt = [](double v) {
        char b[32];
        long long iv = static_cast<long long>(v);
        if (static_cast<double>(iv) == v) std::snprintf(b, sizeof b, "%lld", iv);
        else                              std::snprintf(b, sizeof b, "%.1f", v);
        return std::string(b);
    };
    auto ax = [&](const char* nm, int idx, double d) {
        if (idx < 0) return std::string(nm) + "=-";
        return std::string(nm) + "=" + fmt(idx * d)
             + " (" + std::to_string(idx) + ")";
    };
    // T axis is non-uniform: label with the explicit sample time, not idx*step.
    auto axv = [&](const char* nm, int idx, double val) {
        if (idx < 0) return std::string(nm) + "=-";
        return std::string(nm) + "=" + fmt(val)
             + " (" + std::to_string(idx) + ")";
    };
    auto seg = [&](int it, int k, int j) {
        if (!log_on) return;
        traj.set_log_prefix(tree_tag + " "
            + ax("tau", it, req.dtau_fs) + " "
            + axv("T",  k,  k >= 0 ? req.tw_fs[k] : 0.0) + " "
            + ax("t",   j,  req.dt_detect_fs) + " fs : ");
    };

    // 2DES owns ALL logging from the driver: one forced line per τ/T advance,
    // plus one t-line every print_interval internal steps SNAPPED to a t-point
    // (printed AFTER the expectation, so the label and the shown active set are
    // the same point — no off-by-one — and the t-prints land on round j across
    // every τ/T because the counter resets per t-scan). The trajectory's own
    // per-step cadence printout is therefore disabled here.
    traj.set_cadence_logging(false);

    traj.set_initial(ground_projector(N));
    traj.apply_dipole_full_hierarchy(mu0, spec.pulses[0].side);

    // Batch detection on the GPU: μ_det is fixed for the whole tree, so upload
    // it once and size the device result grid to the full tree. Each t-point
    // launches a trace kernel into its slot (no D2H/sync); one bulk D2H lands in
    // `grid` at the end (detect_flush). Divergence is checked once per t-scan.
    const size_t grid_sz =
        static_cast<size_t>(req.n_tau) * req.n_tw * req.n_t;
    traj.detect_begin(mu_det, static_cast<int>(grid_sz));
    bool diverged = false;

    // Device-resident snapshots, reused (grow-only) across the whole tree so the
    // τ/T carry-through never round-trips the hierarchy to host. Two live at
    // once: τ (held across the T loop) and T (per T point).
    DeviceHierarchySnapshot tau_snap, tw_snap;

    for (int it = 0; it < req.n_tau && !diverged; ++it) {
        if (it > 0) {
            seg(it, -1, -1);
            // Re-declared on every advance, not hoisted: the μ and restore calls
            // below clear the descriptor (it described the previous ρ), so the
            // only placement that survives refactoring is next to the launch.
            apply_block(traj, blk[0]);
            traj.propagate_steps(req.steps_per_tau);
            if (log_on) traj.log_stats();  // force one τ-progress line per point
        }
        traj.snapshot_device(tau_snap);
        traj.apply_dipole_full_hierarchy(mu1, spec.pulses[1].side);
        for (int k = 0; k < req.n_tw && !diverged; ++k) {
            if (req.tw_gap_steps[k] > 0) {
                seg(it, k, -1);
                apply_block(traj, blk[1]);
                traj.propagate_steps(req.tw_gap_steps[k]);
                if (log_on) traj.log_stats();  // force one T-progress line per advance
            }
            traj.snapshot_device(tw_snap);
            traj.apply_dipole_full_hierarchy(mu2, spec.pulses[2].side);
            // Reset per t-scan so every scan prints at the same round j
            // (print_interval, 2·print_interval, … in internal steps), instead
            // of wherever the carried-over global step counter happens to land.
            int t_since_print = 0;
            for (int j = 0; j < req.n_t; ++j) {
                if (j > 0) {
                    // check_div=false: no per-t-point D2H; divergence is checked
                    // once per t-scan below.
                    apply_block(traj, blk[2]);
                    traj.propagate_steps(req.steps_per_t, /*check_div=*/false);
                    t_since_print += req.steps_per_t;
                }
                // Async device-side Tr(μ_det·ρ_0) into the tree grid slot.
                traj.detect_sample(
                    static_cast<int>(
                        static_cast<size_t>(it * req.n_tw + k) * req.n_t + j));
                if (log_on && print_interval > 0 && t_since_print >= print_interval) {
                    seg(it, k, j);          // label = the t-point we are AT now
                    traj.log_stats();
                    t_since_print = 0;
                }
            }
            // One divergence guard per t-scan (instead of per t-point). The
            // label is the last t-coordinate of this scan.
            if (traj.check_divergence_now(
                    (req.n_t - 1) * req.dt_detect_fs, it * req.n_tw + k))
                diverged = true;
            traj.restore_device(tw_snap);
        }
        traj.restore_device(tau_snap);
    }

    // Single bulk D2H of the whole tree's detection grid.
    traj.detect_flush(grid.data(), static_cast<int>(grid_sz));
}

// Compose the "(xxxx)->I1,I2,I3" component label for the per-tree header line.
std::string component_label(const Component& c) {
    static const char ax[] = "xyz";
    std::string s = "(";
    for (int i = 0; i < 4; ++i) s += ax[c.m[i]];
    s += ")->";
    bool first = true;
    for (int ip = 0; ip < 3; ++ip) {
        if (!c.bucket[ip]) continue;
        if (!first) s += ",";
        s += "I" + std::to_string(ip + 1);
        first = false;
    }
    return s;
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

// ===========================================================================
// Orientation-reuse walk (3→9→21 prefix tree, B=1). Pure host reshuffle of
// run_pathway_tree's τ→T→t carry-through: set_initial+μ0 once per m1 (3), μ1+T
// once per (m1,m2) (9), and the t-scan + μ⁻(m4) detection per leaf (21). Only 2
// device snapshots live at once (tau_snap, tw_snap) — same as the flat tree.
// Filter OFF is bit-identical to the flat walk; filter ON stays within tolerance.
// ===========================================================================
namespace {

using Cd = std::complex<double>;

// FINE-log label helpers (own copies; the flat path keeps its lambdas untouched
// so it stays the byte-for-byte golden reference).
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
    const PathSpec& spec;
    const std::array<const Eigen::MatrixXcd*, 3>& Rax;
    const TwoDESRequest& req;
    int N;
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
// divergence once, flushes n_t samples into grid_c at the (it,k) plane. Mirrors
// run_pathway_tree's j-loop exactly. Returns true iff diverged.
bool detect_leaf(ReuseCtx& c, const DeviceHierarchySnapshot& tw_snap,
                 const ReuseLeaf& leaf, int it, int k, std::vector<Cd>& grid_c) {
    const TwoDESRequest& req = c.req;
    c.traj.restore_device(tw_snap);
    c.traj.apply_dipole_full_hierarchy(rsel(c, leaf.m[2], c.spec.pulses[2].raise),
                                       c.spec.pulses[2].side);
    const Eigen::MatrixXcd mu_det = c.Rax[leaf.m[3]]->adjoint();  // μ⁻(m4)
    c.traj.detect_begin(mu_det, req.n_t);
    int t_since_print = 0;
    for (int j = 0; j < req.n_t; ++j) {
        if (j > 0) {
            apply_block(c.traj, c.blk[2]);
            c.traj.propagate_steps(req.steps_per_t, /*check_div=*/false);
            t_since_print += req.steps_per_t;
        }
        c.traj.detect_sample(j);
        if (c.log_on && c.print_interval > 0 && t_since_print >= c.print_interval) {
            c.traj.set_log_prefix(
                seg_prefix(req, c.base + " " + leaf_label(leaf), it, k, j));
            c.traj.log_stats();
            t_since_print = 0;
        }
    }
    const bool div = c.traj.check_divergence_now(
        (req.n_t - 1) * req.dt_detect_fs, it * req.n_tw + k);
    c.traj.detect_flush(
        grid_c.data() + static_cast<size_t>(it * req.n_tw + k) * req.n_t, req.n_t);
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
        c.traj.apply_dipole_full_hierarchy(
            rsel(c, b2.m2, c.spec.pulses[1].raise), c.spec.pulses[1].side);
        for (int k = 0; k < req.n_tw; ++k) {
            if (req.tw_gap_steps[k] > 0) {
                if (c.log_on)
                    c.traj.set_log_prefix(seg_prefix(req, c.base + " m1m2", it, k, -1));
                apply_block(c.traj, c.blk[1]);
                c.traj.propagate_steps(req.tw_gap_steps[k]);
                if (c.log_on) c.traj.log_stats();
            }
            c.traj.snapshot_device(tw_snap);
            for (const auto& leaf : b2.leaves)
                if (detect_leaf(c, tw_snap, leaf, it, k, leaf_grid[leaf.comp_idx]))
                    return true;
            // Back to the clean μ1-state at T[k]: the next k advances the T chain
            // from here (each detect_leaf left the device on a t-propagated
            // state). Restore before the next waiting-time step.
            c.traj.restore_device(tw_snap);
        }
    }
    return false;
}

// The full 3→9→21 walk for one (signal,pathway). Two snapshots live at once
// (tau_snap across the m2 fanout, tw_snap across leaves) — same footprint as the
// flat tree. Emits one coarse heartbeat per m1 group (3 lines / (s,p)).
void run_pathway_reuse(ReuseCtx& c, const std::vector<ReuseM1Branch>& m1s,
                       std::vector<std::vector<Cd>>& leaf_grid) {
    const TwoDESRequest& req = c.req;
    c.traj.set_cadence_logging(false);
    DeviceHierarchySnapshot tau_snap, tw_snap;
    static const char ax[] = "xyz";
    for (const auto& b1 : m1s) {
        const auto t0 = std::chrono::steady_clock::now();
        c.traj.set_initial(ground_projector(c.N));
        c.traj.apply_dipole_full_hierarchy(
            rsel(c, b1.m1, c.spec.pulses[0].raise), c.spec.pulses[0].side);
        bool diverged = false;
        for (int it = 0; it < req.n_tau && !diverged; ++it) {
            if (it > 0) {
                if (c.log_on)
                    c.traj.set_log_prefix(seg_prefix(req, c.base + " m1", it, -1, -1));
                apply_block(c.traj, c.blk[0]);
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
    c.traj.apply_dipole_per_slot(mu2, c.spec.pulses[2].side);
    const int base = (it * req.n_tw + k) * req.n_t;
    int t_since_print = 0;                                // mirror detect_leaf
    for (int j = 0; j < req.n_t; ++j) {
        if (j > 0) {
            apply_block(c.traj, c.blk[2]);
            c.traj.propagate_steps(req.steps_per_t, /*check_div=*/false);
            t_since_print += req.steps_per_t;
        }
        c.traj.detect_sample_batch(base + j);            // writes 21 traces at once
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
// the reuse and flat walks.
void run_pathway_batched(ReuseCtx& c, const BranchPlan& plan,
                         std::vector<std::vector<Cd>>& leaf_grid) {
    const TwoDESRequest& req = c.req;
    const int N = c.N;
    c.traj.set_cadence_logging(false);
    const auto t0 = std::chrono::steady_clock::now();

    // τ: 3 slots = ground + μ0(m1). set_active_batch(3) first so set_initial_batch
    // expects exactly 3 initial matrices.
    c.traj.set_active_batch(plan.B_tau);
    std::vector<Eigen::MatrixXcd> grounds(plan.B_tau, ground_projector(N));
    c.traj.set_initial_batch(grounds);
    std::vector<Eigen::MatrixXcd> mu0(plan.B_tau);
    for (int b = 0; b < plan.B_tau; ++b)
        mu0[b] = rsel(c, plan.m1_axis[b], c.spec.pulses[0].raise);
    c.traj.apply_dipole_per_slot(mu0, c.spec.pulses[0].side);

    // Detection: 21 leaf μ_det(m4)=(μ⁺)†, opened once for the whole tree; one
    // flat [21·(n_tau·n_tw·n_t)] device grid, one D2H at the end.
    std::vector<Eigen::MatrixXcd> mu_det(plan.B_t);
    for (int b = 0; b < plan.B_t; ++b)
        mu_det[b] = c.Rax[plan.leaves[b].m[3]]->adjoint();
    const int n_samples = req.n_tau * req.n_tw * req.n_t;
    c.traj.detect_begin_batch(mu_det, n_samples);

    DeviceHierarchySnapshot tau_snap, tw_snap;
    bool diverged = false;
    for (int it = 0; it < req.n_tau && !diverged; ++it) {
        if (it > 0) {                                            // τ advance, active=3
            if (c.log_on)
                c.traj.set_log_prefix(seg_prefix(req, c.base + " m1", it, -1, -1));
            apply_block(c.traj, c.blk[0]);
            c.traj.propagate_steps(req.steps_per_tau);
            if (c.log_on) c.traj.log_stats();
        }
        c.traj.snapshot_device(tau_snap);            // B=3

        // 3→9 + μ1(m2). expand_slots sets active=9.
        c.traj.expand_slots(tau_snap, plan.tau_to_T);
        std::vector<Eigen::MatrixXcd> mu1(plan.B_T);
        for (int b = 0; b < plan.B_T; ++b)
            mu1[b] = rsel(c, plan.m2_axis[b], c.spec.pulses[1].raise);
        c.traj.apply_dipole_per_slot(mu1, c.spec.pulses[1].side);

        for (int k = 0; k < req.n_tw; ++k) {
            if (req.tw_gap_steps[k] > 0) {                        // T advance, active=9
                if (c.log_on)
                    c.traj.set_log_prefix(seg_prefix(req, c.base + " m1m2", it, k, -1));
                apply_block(c.traj, c.blk[1]);
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
    const size_t grid_sz = static_cast<size_t>(req.n_tau) * req.n_tw * req.n_t;
    std::vector<Cd> flat(static_cast<size_t>(plan.B_t) * grid_sz);
    c.traj.detect_flush_batch(flat.data(), static_cast<int>(grid_sz));
    for (int b = 0; b < plan.B_t; ++b) {
        const int ci = plan.leaves[b].comp_idx;      // == b (canonical), explicit
        std::copy(flat.begin() + static_cast<size_t>(b) * grid_sz,
                  flat.begin() + static_cast<size_t>(b + 1) * grid_sz,
                  leaf_grid[ci].begin());
    }

    const double s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "  [batch B=21] " << c.base << " N=" << N
              << (diverged ? "  DIVERGED" : "")
              << "  (" << heom::console::format_computation_time(s) << ")\n"
              << std::flush;
}

}  // namespace

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
    std::vector<Cd> grid(grid_sz);

    // Two-level 2DES log (mirrors absorption's "[abs x] propagating" always-on
    // header + print_interval-gated per-step lines):
    //   - COARSE: the header + one "[k/total]" line per component tree. Always
    //     printed — it is the progress heartbeat (which pathway/component are we
    //     on, how long did it take), cheap at `tree_total` lines.
    //   - FINE: the per-step "... fs : ADOs |A|/|E|/M ..." lines inside each
    //     tree, gated by print_interval (==0 ⇒ off) via `log_on`.
    const int  print_interval = sys_single.config().time.print_interval;
    const bool log_on         = (print_interval > 0);
    const int  n_comp_trees   = static_cast<int>(isotropic_components().size());
    const int  tree_total     = out.n_signal * out.n_pathway * n_comp_trees;
    int        tree_idx       = 0;

    auto join = [](const std::vector<std::string>& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) { if (i) s += ", "; s += v[i]; }
        return s;
    };

    std::cout << "\n======== 2DES (carry-through) ========\n"
              << "  signals    : " << join(req.signals)
              << "  (" << out.n_signal << ")\n"
              << "  pathways   : " << join(req.pathways)
              << "  (" << out.n_pathway << ")\n";
    std::cout << "  orientation: isotropic -> I1,I2,I3 ("
              << n_comp_trees << " components)\n";
    std::cout << "  grid       : tau " << req.n_tau << " x " << req.dtau_fs
              << "fs | ";
    if (req.n_tw == 1) std::cout << "T 1 (T=" << req.tw_fs[0] << "fs)";
    else std::cout << "T " << req.n_tw << " pts ["
                   << req.tw_fs.front() << ".." << req.tw_fs.back() << "]fs";
    std::cout << " | t " << req.n_t << " x " << req.dt_detect_fs << "fs\n"
              << "  trees      : " << tree_total
              << "  (print_interval=" << print_interval << ")\n"
              << "======================================\n" << std::flush;

    // Coarse per-tree heartbeat: one flushed line per component tree, printed
    // AFTER it finishes, carrying its elapsed time. Printed after (not before)
    // so nothing the trajectory emits mid-run — the one-time "[ulist] enabled"
    // banner, a divergence warning, or (FINE mode) the per-step ADO lines that
    // already carry the same "[k/total]" tag — interleaves into the label. The
    // running k/total on every line is the "where are we" signal; the trailing
    // (elapsed) tells which pathway/component is slow.
    auto timed_tree = [&](const std::string& label, auto&& run) {
        const auto t0 = std::chrono::steady_clock::now();
        run();
        const double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << label << "  (" << heom::console::format_computation_time(s)
                  << ")\n" << std::flush;
    };

    for (int s = 0; s < out.n_signal; ++s) {
        const std::string& sig = req.signals[s];
        const bool rp = (sig == "rephasing");
        if (!rp && sig != "nonrephasing")
            throw std::invalid_argument(
                "compute_2des: unknown signal '" + sig +
                "' (want rephasing/nonrephasing)");

        for (int p = 0; p < out.n_pathway; ++p) {
            const PathSpec& spec = find_path(rp, req.pathways[p]);
            const bool dbl = (spec.basis == BasisType::DOUBLE_EXCITED);
            const SystemContext& ctx = dbl ? sys_double : sys_single;
            const auto& Rax = dbl ? Rdouble : Rsingle;
            const int N = ctx.N();

            // The blocks below are declarations the kernel takes literally, so
            // they must be derived from the SAME basis the trajectory runs in.
            // NOT assert(): shipped builds are Release, where assert() compiles
            // to nothing — a guard that vanishes in exactly the builds that run
            // is worse than none, because the plan says it is there.
            if (spec.basis != ctx.config().physics.basis)
                throw std::logic_error(
                    "compute_2des: pathway basis / SystemContext mismatch");
            const auto lay = ManifoldLayout::from(
                spec.basis, ctx.config().physics.n_states, N);
            const auto blk = pathway_segment_blocks(spec, lay);  // [τ, T, t]

            // Batched path decision + budget guard. B=21 needs 21× the two
            // mutable O(M·N²) buffers, PLUS the driver-owned extras the alloc
            // plan cannot see (it only covers trajectory-owned buffers):
            //   - tau_snap (B=3) + tw_snap (B=9) gathered blocks, worst case
            //     n_active = M ⇒ 12·M·N² elements;
            //   - the [21·tree] device detection grid.
            // Fold them INTO the budget (not out of gfree) so the printed
            // shortfall report stays honest. If it doesn't fit, fall back to
            // reuse/flat with a clear message instead of a raw ctor OOM throw
            bool use_batch = req.batch;
            if (use_batch) {
                auto budget = heom::estimate_budget(
                    ctx.config(), static_cast<size_t>(ctx.M()), /*batch=*/21);
                const size_t MN2 = static_cast<size_t>(ctx.M()) * N * N;
                const size_t snap_bytes =
                    12 * MN2 * sizeof(heom::complex_t);
                const size_t det_bytes = static_cast<size_t>(21) * req.n_tau
                    * req.n_tw * req.n_t * sizeof(heom::complex_t);
                budget.auxiliary  += snap_bytes + det_bytes;
                budget.gpu_total  += snap_bytes + det_bytes;
                size_t gfree = 0, gtot = 0;
                heom::query_gpu_memory(gfree, gtot);
                std::string budget_err;
                if (heom::check_budget(budget, ctx.config(), gfree, budget_err)
                        == heom::BudgetCheck::Exceeds) {
                    std::cout << "  [2des.batch] B=21 exceeds GPU budget; falling "
                                 "back to " << (req.orient_reuse ? "reuse" : "flat")
                              << "\n" << std::flush;
                    use_batch = false;
                }
            }

            // One trajectory per (signal,pathway), reused across all components
            // via set_initial (never `new` per tree → cudaMalloc thrash). Only
            // one HEOMTrajectory is alive at any time. Batched path opens it at
            // B=21 (max fanout); every other path at B=1.
            HEOMTrajectory traj(ctx, /*enable_hdf5=*/false, use_batch ? 21 : 1);

            auto sel = [&](int axis, bool raise) {
                return raise ? Eigen::MatrixXcd(*Rax[axis])
                             : Eigen::MatrixXcd(Rax[axis]->adjoint());
            };

            // Canonical-order accumulation into out.resp, shared by the reuse and
            // batched paths so their float-add order is byte-identical (the
            // bit-exact parity gate depends on this).
            //
            // The five-deep nest (ci→ip→it→k→j) is THE SHAPE THE GATE LOCKS, not
            // a loop nobody got around to tidying. Floating-point addition is not
            // associative, so collapsing, fusing, reordering, or parallelising
            // these loops changes the sum and breaks flat/reuse/batched parity —
            // which is the only evidence we have that the three walks compute the
            // same physics. Want it faster? Prove the new order is bit-identical
            // first. Otherwise leave it exactly as it is.
            auto accumulate = [&](const std::vector<std::vector<Cd>>& leaf_grid) {
                const auto& comps = isotropic_components();
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

            if (use_batch) {
                // Lockstep 3→9→21 batched walk. Same leaf_grid layout as
                // reuse ⇒ identical accumulation.
                const auto& comps = isotropic_components();
                std::vector<std::vector<Cd>> leaf_grid(comps.size(),
                                                       std::vector<Cd>(grid_sz));
                ReuseCtx rc{traj, spec, Rax, req, N, blk,
                            std::string(rp ? "rp " : "nr ") + spec.name,
                            log_on, print_interval};
                run_pathway_batched(rc, build_branch_plan(), leaf_grid);
                tree_idx += static_cast<int>(comps.size());
                accumulate(leaf_grid);
            } else if (req.orient_reuse) {
                // 3→9→21 prefix-share walk. Runs the whole (s,p) fanout into
                // per-component leaf grids, then accumulates in canonical
                // isotropic_components() order (float-add order == flat ⇒
                // bit-exact when filtering is off).
                const auto& comps = isotropic_components();
                std::vector<std::vector<Cd>> leaf_grid(comps.size(),
                                                       std::vector<Cd>(grid_sz));
                ReuseCtx rc{traj, spec, Rax, req, N, blk,
                            std::string(rp ? "rp " : "nr ") + spec.name,
                            log_on, print_interval};
                run_pathway_reuse(rc, group_isotropic(), leaf_grid);
                tree_idx += static_cast<int>(comps.size());  // heartbeat accounting
                accumulate(leaf_grid);
            } else {
                for (const auto& c : isotropic_components()) {
                    const std::string tag = "[" + std::to_string(++tree_idx)
                                          + "/" + std::to_string(tree_total) + "]";
                    const std::string label = tag + " " + (rp ? "rp" : "nr") + " "
                                            + spec.name + " " + component_label(c)
                                            + "  N=" + std::to_string(N);
                    const Eigen::MatrixXcd mu0 = sel(c.m[0], spec.pulses[0].raise);
                    const Eigen::MatrixXcd mu1 = sel(c.m[1], spec.pulses[1].raise);
                    const Eigen::MatrixXcd mu2 = sel(c.m[2], spec.pulses[2].raise);
                    const Eigen::MatrixXcd mu_det = Rax[c.m[3]]->adjoint();  // μ⁻
                    timed_tree(label, [&] {
                        run_pathway_tree(traj, spec, mu0, mu1, mu2, mu_det, req, N,
                                         blk, grid, tag, log_on, print_interval);
                    });
                    for (int ip = 0; ip < 3; ++ip) {
                        if (!c.bucket[ip]) continue;
                        for (int it = 0; it < req.n_tau; ++it)
                        for (int k = 0; k < req.n_tw; ++k)
                        for (int j = 0; j < req.n_t; ++j)
                            out.resp[out.idx(ip, s, p, it, k, j)] +=
                                spec.sign *
                                grid[static_cast<size_t>(it * req.n_tw + k) * req.n_t + j];
                    }
                }
            }

            if (traj.diverged())
                throw std::runtime_error(
                    std::string("compute_2des: trajectory diverged (") +
                    sig + " " + spec.name + ")");

            // Every propagation in a 2DES walk sits in a τ, T or t segment, and
            // all three have a declared block. A nonzero count means a launch
            // site was added without one — correct, slower, and invisible to
            // the bitwise A/B gate, which is exactly why it is checked here and
            // not left as a silent slowdown. The
            // trajectory is fresh per (signal,pathway), so this reads as a
            // per-walk total.
            if (traj.propagations_without_block() != 0)
                throw std::logic_error(
                    "compute_2des: " + std::to_string(
                        traj.propagations_without_block()) +
                    " propagation segment(s) ran without a declared manifold "
                    "block (" + sig + " " + spec.name + ")");
        }
    }
    return out;
}

}  // namespace heom::spectra
