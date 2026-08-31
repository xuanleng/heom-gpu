#ifndef HEOM_SPECTRA_TWO_DIM_HPP
#define HEOM_SPECTRA_TWO_DIM_HPP

#include "types.hpp"            // BasisType (CUDA-free)
#include "heom_config.hpp"     // TwoDESConfig (host-only)
#include "spectra/dipole.hpp"  // DipoleOperator (host-only)

#include <array>
#include <complex>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

class SystemContext;  // fwd: compute_2des takes two contexts (single + double)

// 2DES shared data structures. Pure host-side,
// CUDA-free: HEOMTrajectory's carry-through primitives return/consume these,
// and the compute_2des driver fills TwoDESResult.
namespace heom::spectra {

// Single-side dipole action edge: Left = μ·ρ, Right = ρ·μ.
enum class DipoleSide { Left, Right };

// Raw product of compute_2des.
struct TwoDESResult {
    int n_tau = 0, n_tw = 0, n_t = 0;
    int n_signal = 0;     // 1 or 2, order [rephasing, nonrephasing]
    int n_pathway = 0;    // 1..3, order [SE, GSB, ESA]
    int n_component = 0;  // always 3: isotropic rotational invariants (I1,I2,I3)
    std::vector<double> tau_fs, tw_fs, t_fs;            // per-axis sample times
    std::vector<std::complex<double>> resp;             // layout via idx()
    // Layout [n_component][n_signal][n_pathway][n_tau][n_tw][n_t], row-major.
    size_t idx(int c, int s, int p, int it, int k, int j) const {
        return ((((static_cast<size_t>(c) * n_signal + s) * n_pathway + p)
                 * n_tau + it) * n_tw + k) * n_t + j;
    }
};

// ---------------------------------------------------------------------------
// A 2DES Liouville pathway is a fixed sequence of single-side dipole
// actions, propagations, and a final mu- trace. No commutators. The driver just
// walks a constant table of 6 PathSpec (signal × pathway), eliminating if/else.
// ---------------------------------------------------------------------------

// One pulse interaction: which side to multiply on, and raising (μ⁺) vs
// lowering (mu-). (side, raise) is fixed for each pathway.
struct PathStep {
    DipoleSide side;   // Left = μ·ρ, Right = ρ·μ
    bool       raise;  // true = μ⁺, false = μ⁻
};

struct PathSpec {
    const char*           name;    // "SE" / "GSB" / "ESA"
    bool                  rephasing;
    BasisType             basis;   // WITH_GROUND (SE/GSB) or DOUBLE_EXCITED (ESA)
    std::array<PathStep,3> pulses; // pulse0/1/2 (first/second/third interaction)
    // Combination sign baked into resp so the total is a plain pathway sum:
    // S = phi_SE + phi_GSB - phi_ESA. SE/GSB = +1,
    // ESA = −1. The driver multiplies grid by this before accumulating, so every
    // resp consumer (post-processing, plots) just adds components — no name-based
    // sign special case downstream. NOTE: the stored ESA component is therefore
    // its SIGNED contribution (−φ3), not the raw positive bracket.
    double                sign;
};

// The 6 pathways, order [rephasing SE,GSB,ESA] then [nonrephasing SE,GSB,ESA].
// Returns a stable span (size 6). Defined in two_dim.cpp.
const std::array<PathSpec, 6>& two_des_path_table();

// ---------------------------------------------------------------------------
// Where ρ actually lives during each propagation segment. A pathway's pulse
// sequence already determines this, so it is DERIVED here rather than tabulated
// beside PathSpec: two encodings of one physics contract can disagree, and
// nothing in the build would say so. The test oracle is an independent
// measurement — a measurement written into the implementation
// can no longer falsify it.
// ---------------------------------------------------------------------------

// A (row manifold, col manifold) block of ρ, in the (offset,count) form
// HEOMTrajectory::set_manifold_block takes. The three manifolds are CONTIGUOUS
// index ranges by construction of the Hilbert basis (compute_hilbert_dim), so
// no mask / index table is ever needed.
struct ManifoldBlock {
    int row_off = 0, n_row = 0, col_off = 0, n_col = 0;
    bool operator==(const ManifoldBlock& o) const {
        return row_off == o.row_off && n_row == o.n_row
            && col_off == o.col_off && n_col == o.n_col;
    }
};

// g/e/f index ranges for one basis. f is empty unless DOUBLE_EXCITED.
// Throws on EXCITED_ONLY (no ground manifold — 2DES never uses it) AND on an
// N that does not match the basis: WITH_GROUND wants 1+n, DOUBLE_EXCITED wants
// 1+n+n(n-1)/2. That check is the whole point of taking N as an argument — the
// one mistake this API exists to prevent is building the layout for one basis
// out of the OTHER context's N (compute_2des keeps sys_single and sys_double
// alive simultaneously, so the wrong one is always within reach).
struct ManifoldLayout {
    static ManifoldLayout from(BasisType basis, int n_states, int N);
    ManifoldBlock block(int row_level, int col_level) const;  // 0=g,1=e,2=f
    int n_level = 0;                   // 2 (WITH_GROUND) or 3 (DOUBLE_EXCITED)
    std::array<int, 4> bound{};        // bound[L]..bound[L+1) is manifold L
};

// Support of ρ after one dipole interaction. Ladder rule: Left ⇒ the ROW level
// moves, Right ⇒ the COLUMN level moves, and a RAISING operator applied from
// the RIGHT LOWERS the column level (ρμ⁺ pairs ρ's column with μ⁺'s row).
// Throws if the step walks off the ladder (raise(f) / lower(g)) — that means
// the pulse table and the basis disagree, which is a config bug, not a zero
// result to be propagated silently.
std::pair<int, int> apply_pulse(const ManifoldLayout& lay,
                                std::pair<int, int> level, PathStep step);

// The three propagation-segment blocks (τ, T, t — i.e. after μ1, μ2, μ3) for
// one pathway, derived from spec.pulses starting at |g><g|. This is the ONLY
// place the τ/T/t accounting exists in the code.
std::array<ManifoldBlock, 3> pathway_segment_blocks(const PathSpec& spec,
                                                    const ManifoldLayout& lay);

// Everything compute_2des needs that is NOT a SystemContext. The driver
// runs the isotropic orientational average (21 components → I1/I2/I3) using
// per-basis μ⁺ blocks along x/y/z; μ⁻ = (μ⁺)†. Sample counts and the internal-
// step spacing between samples are pre-resolved (resolve_2des_config validated
// dt divisibility).
struct TwoDESRequest {
    int n_tau = 0, n_tw = 0, n_t = 0;
    int steps_per_tau = 0, steps_per_t = 0;
    double dtau_fs = 0, dt_detect_fs = 0;  // uniform tau/t axis labels
    // Non-uniform waiting (T) axis: explicit sample times (fs) and the per-gap
    // internal step counts. tw_gap_steps[k] = steps from tw_fs[k-1] to tw_fs[k]
    // with tw_fs[-1] := 0 (so tw_gap_steps[0] is the advance from T=0 to the
    // first sample, 0 when tw_fs[0]==0). n_tw == tw_fs.size().
    std::vector<double> tw_fs;
    std::vector<int>    tw_gap_steps;
    std::vector<std::string> signals;       // subset/order of {"rephasing","nonrephasing"}
    std::vector<std::string> pathways;      // subset/order of {"SE","GSB","ESA"}
    bool orient_reuse = false;              // isotropic 3->9->21 prefix-share walk;
                                            // off = flat 21-tree (bit-exact).
    bool batch = false;                     // isotropic 3->9->21 lockstep batched walk;
                                            // supersedes orient_reuse.
    const DipoleOperator* dipole_single = nullptr;  // WITH_GROUND μ⁺ blocks
    const DipoleOperator* dipole_double = nullptr;  // DOUBLE_EXCITED μ⁺ blocks
};

// Build the axis/sampling fields of a TwoDESRequest from a resolved
// TwoDESConfig and the HEOM step dt (fs). Does NOT set the dipole pointers
// (caller wires dipole_single/dipole_double, which outlive the req). Single
// source of truth shared by the single-shot dispatch (main.cpp) and the MC
// shell (mc_driver) so the two cannot drift on n_tau/tw-gap arithmetic.
TwoDESRequest build_2des_request(const HEOMConfig::TwoDESConfig& spec,
                                 double dt_fs);

// ---------------------------------------------------------------------------
// Orientation reuse (3 to 9 to 21 prefix tree). The 21 isotropic components share
// τ-segment propagation by m1 (3 groups) and T-segment by (m1,m2) (9 branches);
// m4 is the trace-only projection (free). group_isotropic() reorganises the 21
// components (from the driver's isotropic_components()) into that tree WITHOUT
// changing the set — each component appears exactly once, tagged with its
// original index (comp_idx) so the driver accumulates in canonical order.
struct ReuseLeaf {
    int comp_idx;                 // index into isotropic_components() (accumulation)
    std::array<int, 4> m;         // cartesian axes (m1,m2,m3,m4)
    std::array<bool, 3> bucket;   // membership in {I1,I2,I3}
};
struct ReuseM2Branch { int m2; std::vector<ReuseLeaf> leaves; };
struct ReuseM1Branch { int m1; std::vector<ReuseM2Branch> m2s; };

// Reorganise the 21 isotropic components into the 3→9→21 prefix tree, in
// first-seen order (deterministic, reproducible). size() == 3.
std::vector<ReuseM1Branch> group_isotropic();

// ---------------------------------------------------------------------------
// Batched fan-out. Flattens group_isotropic()'s 3-9-21 tree
// into SLOT-indexed maps for the lockstep batched walk (run_pathway_batched):
// τ layer = 3 slots (one per m1), T layer = 9 slots (one per (m1,m2)), t layer
// = 21 slots (one per leaf). expand_slots() consumes tau_to_T / T_to_t as its
// child→parent maps. Derived from group_isotropic() — nothing hard-coded.
// ---------------------------------------------------------------------------
struct LeafSpec {
    int comp_idx;                 // index into isotropic_components() (== t-slot)
    std::array<int, 4> m;         // cartesian axes (m1,m2,m3,m4)
    std::array<bool, 3> bucket;   // membership in {I1,I2,I3}
};
struct BranchPlan {
    int B_tau = 3, B_T = 9, B_t = 21;
    std::array<int, 3>       m1_axis;   // τ-slot i → m1 value (μ0 axis per τ-slot)
    std::vector<int>         m2_axis;   // T-slot → m2 value (μ1 axis per T-slot), len 9
    std::vector<int>         tau_to_T;  // T-slot → parent τ-slot, len 9, ∈[0,3)
    std::vector<int>         T_to_t;    // t-slot → parent T-slot, len 21, ∈[0,9)
    std::array<LeafSpec, 21> leaves;    // t-slot → leaf (walk order == comp_idx order)
};

// Derive the batched fan-out plan from group_isotropic(). Depth-first flatten
// (m1 → m2 → leaves), so leaves[b].comp_idx == b (the reuse walk's canonical
// accumulation order). Pure host, no GPU.
BranchPlan build_branch_plan();

// Isotropic carry-through 2DES driver. `sys_single` is the WITH_GROUND context
// (SE/GSB); `sys_double` the DOUBLE_EXCITED context (ESA). Fills TwoDESResult
// with n_component=3 (rotational invariants I1,I2,I3).
TwoDESResult compute_2des(const SystemContext& sys_single,
                          const SystemContext& sys_double,
                          const TwoDESRequest& req);

}  // namespace heom::spectra

#endif  // HEOM_SPECTRA_TWO_DIM_HPP
