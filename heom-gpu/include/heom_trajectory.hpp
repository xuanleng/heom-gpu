#ifndef HEOM_TRAJECTORY_HPP
#define HEOM_TRAJECTORY_HPP

#include "system_context.hpp"
#include "types.hpp"
#include "gpu_memory.hpp"
#include "spectra/two_dim.hpp"         // DipoleSide
#include "core/heom_derivatives.cuh"   // HEOMContext (used by run_step)
#include "core/precision.hpp"          // heom::complex_t (hier precision buffers)

#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class UListBuilder;  // compact launch set U = E (opt-in via time_evolution.ulist)

// Device-resident copy of a trajectory's whole state, for the 2DES τ/T tree.
// restore(snapshot()) is the identity: propagating on from a restored state is
// propagating on from the original, bit for bit, in both builds. Everything
// here is what the next step reads; the U-list is not, because the first step
// after a restore scans all M, which yields the same flags.
//
// The mixed build's Kahan carry is state too. Restarting it at zero on every
// restore looked like an ulp-level shortcut and measured as 5x the build's own
// error on a branch-every-step tree (FMO L=5: 1.9e-7 vs 3.7e-8 against FP64;
// historical snapshot-identity probe notes).
//
// A snapshot is replayed only on the SystemContext that took it, at a width
// the target can hold: restore/expand check both before writing anything. The
// ρ buffer is sized once, to batch·M·rho_capacity, on the first capture — the
// figure the 2DES budget reserves for it — so it never grows mid-tree.
struct DeviceHierarchySnapshot {
    const SystemContext* sys = nullptr;  // producer; null ⇒ never filled
    int  batch = 0;                      // slots captured
    heom::ManifoldBlock support;         // where ρ lives (its storage shape)
    std::vector<char> fresh;             // [batch] fill window open (see fresh_state_)
    std::vector<int>  fresh_zeros;       // [batch] zero-ADO counts (see fresh_zeros_)
    std::vector<double> norm0;           // [batch] per-slot divergence baselines
    GPUMemory<heom::complex_t> rho;      // [batch·M·Nb] verbatim, Nb = support.elems()
    GPUMemory<int8_t>          flags;    // [batch·M] per-slot active_flags (filtering on)
    GPUMemory<heom::complex_t> kahan;    // [batch·M·Nb] RK4 carry (mixed build only)
};

/**
 * @brief The propagation engine: B density-matrix hierarchies on top of an
 * immutable SystemContext, and the primitives that move them.
 *
 * Owns the trajectory-specific GPU buffers (d_rho_, d_workspace_, filter
 * state), the hier-precision host mirror of ρ_0, the divergence flag, and the
 * running step counter. RK4 combines in-place into d_rho_ (no ping-pong).
 * Cheap to construct against the same SystemContext, by design.
 *
 * The state is ρ plus two facts about it that every launch needs:
 *   - its WIDTH  (active_batch(): how many slots hold a matrix), and
 *   - its SUPPORT (support(): the manifold block it lives in).
 * Both change only where ρ does. Four calls replace ρ and each carries the
 * new width and support with it — set_initial(rho0s, block), apply_dipole(
 * mus, side, block_after), restore_device(snap), expand_slots(parent, map) —
 * and propagate_steps(n) advances whatever is there. That is the whole
 * protocol; dynamics, absorption and 2DES are all callers of it.
 *
 * Every state-touching call is batched: it takes B matrices / dipoles and acts
 * on B slots. B==1 is the batched API with a one-element vector.
 *
 * Not here, by design:
 *   - the output loop / HDF5 file / "Final trace" — heom::run_output_loop and
 *     heom::run_dynamics (dynamics_run.hpp), built on propagate_steps;
 *   - device-side 2DES detection — heom::Detector (detector.hpp), built on
 *     device_rho().
 */
class HEOMTrajectory {
public:
    /**
     * @param sys    Immutable system context (must outlive this trajectory).
     * @param batch  Batch width B: propagate B density matrices that share one
     *               SystemContext (H / maps / bath) in a single launch. The
     *               ctor sizes every mutable buffer by B and runs the GPU
     *               budget check with B up front — one-shot, no re-alloc /
     *               re-check window (hence a ctor arg, not a setter).
     */
    explicit HEOMTrajectory(const SystemContext& sys, int batch = 1);
    ~HEOMTrajectory();

    HEOMTrajectory(const HEOMTrajectory&)            = delete;
    HEOMTrajectory& operator=(const HEOMTrajectory&) = delete;

    /// Initial state: rho0s[b] is written to slot b's α=0 block, all α>0 slots
    /// zeroed. The active width becomes rho0s.size() (1 ≤ size ≤ batch()): the
    /// 2DES τ layer sets 3 matrices on a batch=21 trajectory and runs at width
    /// 3 from there. Slot b's divergence baseline becomes ‖rho0s[b]‖_F.
    /// Resets the internal step counter (opens a fresh tree for every slot).
    ///
    /// `support` is where ρ lives — rows [row_off, +n_row) × cols [col_off,
    /// +n_col) — and every launch from here on is confined to it (see
    /// support_). Dynamics passes ManifoldBlock::full(N); absorption (e,g); a
    /// 2DES tree starts at the ground block. Checked, not trusted: an element of
    /// any rho0s[b] outside the block that is not exactly 0.0 throws
    /// invalid_argument before anything reaches the device.
    void set_initial(const std::vector<Eigen::MatrixXcd>& rho0s,
                     const heom::ManifoldBlock& support);

    /// Batch width B this trajectory was constructed with.
    int batch() const { return batch_; }

    /// Number of ACTIVE slots B_active ≤ batch(): how many density matrices the
    /// current ρ holds. Every launch (derivatives / RK4 / filter / detection /
    /// divergence) spans only these via grid.y = ctx_.batch. It is a property
    /// of ρ, so it changes only where ρ does: set_initial (rho0s.size()),
    /// restore_device (snap.batch), expand_slots (child_to_parent.size()). The
    /// τ/T/t layers of the 3→9→21 walk run at width 3/9/21 on ONE batch=21
    /// trajectory. Default == batch() ⇒ B==1 and absorption (B=3) unchanged.
    int active_batch() const { return active_batch_; }

    /// Where the current ρ has support (see set_initial / apply_dipole). It is
    /// the STORAGE SHAPE: every ADO holds exactly the elems() elements inside
    /// this block, row-major, and nothing outside exists in device memory.
    /// Changed only by the four calls that replace ρ, each of which checks the
    /// new ρ actually fits (and that the block fits the allocated capacity)
    /// before installing it. Empty until set_initial.
    const heom::ManifoldBlock& support() const { return support_; }

    bool diverged() const { return diverged_; }

    /// The α=0 block of every ACTIVE slot, read out in double: out is resized
    /// to active_batch()·N² (slot-major, row-major N×N) and each element is
    /// the hier-precision device value promoted via c_re/c_im — where the
    /// dynamics observable crosses FP32→FP64 in the mixed build. The
    /// spectra observables do not come through here: Detector takes Tr(μ·ρ)
    /// on the device (trace_BR), so in the mixed build that sum is FP32 and
    /// only its result is promoted. One D2H.
    /// This is what the output loop hands to its sink and what the HDF5 file
    /// receives; throws before set_initial.
    void read_alpha0(std::vector<std::complex<double>>& out);

    /// System-bath interaction energy from the first-tier ADOs, one complex
    /// per (slot, bath):
    ///   E_n = Σ_k √s_k Tr[n̂_n ρ̃^{(1)}_{nk}],  s_k = √(|c_k||c̄_k|)
    /// n̂_n is the site occupation from build_d_coupling (1-ex projector on
    /// EXCITED_ONLY / WITH_GROUND; on DOUBLE_EXCITED also f-states that
    /// contain pigment n). FirstTierAdo::scale stores s_k; the √ is taken here.
    /// Sums every first-tier ADO still resident in device memory.
    /// With filtering off this is the physical ⟨H_SB⟩. With filtering on,
    /// inactive ADOs are frozen (not cleared), so the sum is the *stored*
    /// tier-1 correlation including inactive leftovers — not ⟨H_SB⟩ on the
    /// active set alone. Prefer filter-off when comparing to analytic
    /// interaction energies. out is resized to active_batch()·n_bath(),
    /// slot-major. Throws before set_initial.
    void read_system_bath_energy(std::vector<std::complex<double>>& out);

    /// Per-filter-step records accumulated so far (empty with filtering off).
    /// The dynamics writer stores them; nothing in the core reads them back.
    const std::vector<FilteringRecord>& filtering_history() const {
        return filtering_history_;
    }

    /// Copy slot b's α=0 block (device) to a fresh N×N matrix (one D2H). b must
    /// be in [0, batch()). Per-slot readback for the 2DES batched parity tests.
    /// Reads the raw device state — no filter / no host-mirror dependency.
    Eigen::MatrixXcd slot_rho0(int b) const;

    // 2DES carry-through primitives.
    /// Apply a single-side dipole to the FULL ADO hierarchy of every active
    /// slot in one launch: slot b gets mus[b] (N×N host matrix) on the given
    /// `side`; mus.size() must equal active_batch(). Result is written back
    /// into d_rho_. Post: the hierarchy is marked all-active (μ replaced ρ, so
    /// the old active set describes nothing) and slot b's divergence baseline
    /// becomes ‖μ_b ρ_0‖ (or ‖ρ_0 μ_b‖) — the t-scan's check_divergence_now
    /// then uses the post-dipole baseline, not the pre-excitation one.
    ///
    /// `support_after` is where μρ (Left) or ρμ (Right) lives. Checked against
    /// μ on the host before the launch: Left keeps the columns and may only
    /// move rows to those μ(i,k) ≠ 0 reaches from the current rows k; Right is
    /// the transpose statement. A μ that would carry a nonzero outside the
    /// declared block throws invalid_argument — the block is a claim about the
    /// physics and this is the one place the claim can be falsified cheaply.
    void apply_dipole(const std::vector<Eigen::MatrixXcd>& mus,
                      heom::spectra::DipoleSide side,
                      const heom::ManifoldBlock& support_after);

    /// Device-resident snapshot/restore (the 2DES carry-through primitive).
    /// snapshot copies the state out (D2D: ρ, the flags, the mixed build's
    /// Kahan carry) and changes nothing; restore copies it back. `snap` is reused across the
    /// tree by the caller. Restore throws invalid_argument, before any write,
    /// if `snap` came from another SystemContext (or was never filled) or holds
    /// more slots than this trajectory has.
    void snapshot_device(DeviceHierarchySnapshot& snap);
    void restore_device(const DeviceHierarchySnapshot& snap);

    /// 2DES batched-walk fan-out primitive. Child slot c becomes a
    /// copy of parent slot child_to_parent[c] — its ρ, Kahan carry and
    /// divergence baseline — for a general, possibly uneven map (3→9 by m1, 9→21 by
    /// (m1,m2)); the shared flags come from the parent and active_batch_ =
    /// child_to_parent.size(). Same checks as restore_device, plus every
    /// child_to_parent[c] in [0, parent.batch), all before any write.
    void expand_slots(const DeviceHierarchySnapshot& parent,
                      const std::vector<int>& child_to_parent);

    /// ‖ρ − snap.rho‖_F / ‖ρ‖_F over every stored element of every ADO of the
    /// active slots, accumulated in double. In the mixed build both sides are
    /// the Kahan-corrected values ρ − comp. Deterministic: fixed grid, per-block
    /// partials summed on the host in index order. Throws invalid_argument if
    /// snap is from another SystemContext or its width / support differ from
    /// the current state, runtime_error if ‖ρ‖ = 0.
    double relative_distance(const DeviceHierarchySnapshot& snap);

    /// Fill every α>0 ADO of every active slot from α=0, tier by tier, with the
    /// solution of L₀σ = 0 (Terms 2 + 3 + 5 only; fluorescence theory
    /// eq:dressed):
    ///   ρ̃_α[ij] = −i Σ_{n,k} √(α_nk/s_nk)(c_nk d_i − c̄_nk d_j) ρ̃_{α−e_nk}[ij]
    ///             / (Σ α_nk ν_nk + Σ_n Δ_n (d_i − d_j)²).
    /// The exact steady state for a monomer and for uncoupled sites; otherwise
    /// what is left to relax is only what the inter-site coupling causes. The
    /// result is a new unpropagated hierarchy: Kahan carry cleared, all ADOs
    /// active on the next step. Throws logic_error before set_initial.
    void dress_ados();

    /// Propagate n internal RK4 steps with no reset / no HDF5 / no record
    /// callback (carry-through). Uses the cross-segment internal step counter.
    /// check_div=true (default) keeps the per-segment ‖ρ_0‖_F divergence guard
    /// (one N² D2H at the end of the call). Callers that already pay a D2H at
    /// their own cadence pass false and call check_divergence_now() there:
    /// dynamics once per record (after read_alpha0), the 2DES t-scan once per
    /// scan instead of per t-point.
    void propagate_steps(int n_internal_steps, bool check_div = true);

    /// Public one-shot divergence guard: one N² D2H of ρ_0 + Frobenius-norm
    /// ceiling test. Latches diverged() and returns it. The output loop calls
    /// it once per record (n_total = the run's step count, for the report);
    /// the 2DES t-scan once per scan instead of once per t-point, with no
    /// meaningful total (0).
    bool check_divergence_now(double t_fs, int step, int n_total = 0);

    /// The immutable system this trajectory propagates.
    const SystemContext& sys() const { return sys_; }

    /// Borrowed device pointer to the hierarchy, [batch][M][Nb] in hier
    /// precision with Nb = support().elems() (compact block storage, so the
    /// layout changes whenever support() does — read both together). Allocated
    /// once in the ctor and never moved, so the pointer is stable for the
    /// trajectory's lifetime. For read-only side consumers (Detector); nothing
    /// outside this class writes through it.
    const heom::complex_t* device_rho() const { return d_rho_.get(); }

    /// 2DES structured logging: prefix prepended to every per-step filtering
    /// statistics line (the ones gated by print_interval). Lets the 2DES driver
    /// tag each line with its segment + indices (tau/T/t · it/k/j) without the
    /// trajectory knowing anything about pathways. Empty (default) reproduces
    /// the historical dynamics/absorption line byte-for-byte. print_interval==0
    /// still disables printing entirely (the line never fires), so this is a
    /// no-op in that case.
    void set_log_prefix(const std::string& prefix) { log_prefix_ = prefix; }

    /// 2DES forced progress line: print one ADO-statistics line using the
    /// current log prefix, independent of print_interval. Used by the driver at
    /// each τ/T segment boundary so the (single-step) τ/T advances are visible
    /// even when print_interval would otherwise skip them. Read-only w.r.t. the
    /// hierarchy (no refilter); no-op when filtering is off.
    void log_stats();

    /// Per-slot |A_b| plus the TRUE union |A_union| over the CURRENT active_batch_
    /// slots, computed from ONE fresh norm snapshot (OR of the per-slot active
    /// masks). This keeps union and per-slot temporally consistent so the
    /// inflation ratio r = union_count / max_b(per_slot) is always ≥ 1 — unlike
    /// the printed |A| line, which reads the last (possibly stale) filter cadence.
    /// Read-only w.r.t. the hierarchy (d_ado_norms_ is scratch). per_slot.size()
    /// == active_batch_ (live width, not allocated batch_).
    struct SlotActiveStats {
        std::vector<int> per_slot;  // |A_b| for each live slot
        int union_count = 0;        // |A_union| = |⋃_b A_b|
    };
    SlotActiveStats slot_active_stats();

    /// Gate the per-step (cadence) filtering printout. Dynamics/absorption keep
    /// it ON (default). The 2DES driver turns it OFF and emits all lines itself
    /// via log_stats() (one per τ/T advance + point-snapped t-lines), so the
    /// label always matches the shown active set and t-prints land on round j.
    void set_cadence_logging(bool on) { cadence_logging_ = on; }

private:
    const SystemContext& sys_;

    // Batch width B (ctor arg). Every mutable O(M·Nb) buffer (d_rho_,
    // d_workspace_, d_kahan_comp_) and the α=0 host mirror are sized by B;
    // read-only system tables stay single-copy. B==1 ⇒ pre-batch layout.
    const int batch_;

    // Elements per ADO the mutable buffers are allocated for (Nb_cap =
    // sys_.rho_capacity()). Every block installed through set_support_ must
    // satisfy elems() ≤ capacity_; that check is what makes the compacted
    // layout safe to write into.
    const int capacity_;

    // 2DES batched-walk active width B_active ≤ batch_ (see active_batch()). Drives
    // grid.y for every launch and the slot count for filter/divergence/init.
    // Initialised to batch_ so the non-2DES paths (dynamics/absorption) are
    // unchanged. Buffers stay sized by batch_ (max); only the launch width moves.
    int active_batch_;

    // The one write point for active_batch_ (1 ≤ w ≤ batch_, else throws).
    // Also updates ctx_.batch when the context already exists: build_context()
    // is idempotent, so mutating only the member would leave the launch width
    // frozen at whatever it was first built with (the idempotency trap).
    void set_width_(int w);

    // GPU mutable buffers (per-trajectory). d_rho_ is always the current
    // state: rk4_step's final combine is in-place (rho += dt/6 · acc), so
    // no second rho buffer / swap exists. hier precision (FP32 in mixed): the
    // O(M·N²) big arrays that own the bandwidth cost.
    GPUMemory<heom::complex_t> d_rho_;
    GPUMemory<heom::complex_t> d_workspace_;

    // Kahan compensation carry for the RK4 final combine. Only
    // allocated + zero-inited in the mixed-precision build (heom::kMixedPrecision);
    // empty (nullptr) in the FP64 build so rk4_step's combine stays bit-identical
    // to history. Persists across steps — the carry lives here. Same layout as
    // d_rho_ (the combine indexes both with one offset), so a snapshot copies
    // it with ρ's extents.
    GPUMemory<heom::complex_t> d_kahan_comp_;

    // Support of the current ρ (see support()). State, not a hint: written by
    // exactly the four calls that replace ρ (set_initial / apply_dipole /
    // restore_device / expand_slots), always through set_support_(), always
    // BEFORE the ρ it describes is written (shape first: the block is the
    // storage shape and the capacity check must precede the first write).
    // Empty (validate() rejects it) until set_initial; initial_set_ guards
    // every reader.
    heom::ManifoldBlock support_;

    // The one write point for support_: validate, store, and if ctx_ already
    // exists copy it into ctx_.block (build_context is idempotent, so a built
    // ctx has to be updated here). ctx_.block is always support_ — the kernel
    // launches over exactly the block ρ lives in, nothing else.
    void set_support_(const heom::ManifoldBlock& blk, const char* who);

    // Invariant hook: call this immediately after ρ is replaced by a NEW,
    // unpropagated hierarchy (set_initial / apply_dipole). The Kahan carry
    // records bits lost while accumulating the OLD ρ, so a stale carry would be
    // injected straight back into the new one: it is cleared. Clearing is not
    // an optimisation, it is the lifecycle rule. Sets fresh_state_.
    //
    // restore_device / expand_slots do not come here: they reinstall a state,
    // carry and fresh flag included, rather than start one.
    //
    // Zeroes the WHOLE carry buffer, not the active prefix: the prefix version
    // needs the reader to carry "slots past the prefix are always re-zeroed by
    // the next widening" in their head, and these calls happen once per τ/T
    // segment against a propagation that touches the same footprint on every
    // RK4 stage.
    void on_rho_replaced();

    // Restore/expand precondition: `snap` was taken on sys_ and its `width`
    // slots fit. Throws invalid_argument naming `who`; touches nothing.
    void check_snapshot_(const DeviceHierarchySnapshot& snap, int width,
                         const char* who) const;

    // 2DES: lazy-allocated device scratch for the [B·N²] per-slot dipoles μ
    // (apply_dipole). Single-side μ needs a distinct output buffer; we reuse
    // d_workspace_ as the output then copy back into d_rho_.
    GPUMemory<heom::complex_t> d_mu_scratch_;

    // relative_distance's [2·kDistMaxBlocks] partial sums and their host twin.
    GPUMemory<double>   d_dist_partials_;
    std::vector<double> h_dist_partials_;

    // 2DES: reusable host buffer for packing the [B·N²] μ matrices before their
    // H2D (apply_dipole). A member so we don't heap-alloc a fresh std::vector
    // on every pulse interaction.
    std::vector<heom::complex_t> h_mu_pack_;

    // Cached HEOM context: immutable device pointers + scalars borrowed from
    // SystemContext, built once (build_context()) and reused by every
    // propagate_steps(). Per-filter fields (u_list/u_count) still mutate each
    // filter step.
    HEOMContext ctx_;
    bool ctx_built_ = false;

    // Running internal-step counter for propagate_steps(): accumulates across
    // carry-through segments so the stats cadence stays continuous.
    // set_initial() resets it (opening a new tree).
    int internal_step_ = 0;

    // 2DES structured-log prefix (set via set_log_prefix). Prepended to each
    // per-step filtering statistics line so the 2DES driver can tag segment +
    // it/k/j indices. Empty for dynamics/absorption (historical line verbatim).
    std::string log_prefix_;

    // 2DES: when false, the per-step cadence printout is suppressed (the driver
    // forces a single τ/T line via log_stats() instead). Always true for
    // dynamics/absorption. See set_cadence_logging.
    bool cadence_logging_ = true;

    // Filter buffers (only allocated when filtering is on). One row of M per
    // slot ([batch][M]): every slot has its own active set (2026-10-22-batch-
    // per-slot), so a batched slot computes what it would compute alone.
    GPUMemory<double>  d_ado_norms_;
    GPUMemory<int8_t>  d_active_flags_;
    GPUMemory<int>     d_level_counts_;
    // One device int, |= 1 by the filter kernel when any scanned ADO norm is
    // NaN. Read only in check_divergence (piggybacks on its existing D2H).
    // Never cleared: it latches like diverged_ — one NaN anywhere in a 2DES
    // tree voids the whole run.
    GPUMemory<int>     d_nan_flag_;
    GPUMemory<int>     d_zero_count_;   // [batch] count_zero_norms scratch
    std::vector<int>   h_zero_counts_;  // its host twin

    // Host mirror of ρ_0 only (N², not M·N²), [B][N²] FULL
    // matrix. hier precision (matches d_rho_'s element type). Consumers uplift
    // each element to std::complex<double> via c_re/c_im (observable
    // uplift): the FP32→FP64 promotion is exactly the "read out in double"
    // boundary. Deliberately NOT compacted: 1.9 MB of host memory
    // keeps read_alpha0 / HDF5 / rho0_norm_ untouched.
    std::vector<heom::complex_t> h_rho0_;
    // Compact [B][Nb_cap] staging for the α=0 D2H/H2D: the device holds the
    // block, the host mirror holds the full matrix, this is the wire format
    // between them (expand_block_ does the conversion).
    std::vector<heom::complex_t> h_pack_;
    std::vector<int> h_level_counts_;

    // slot_active_stats() scratch (batched per-slot diagnostic, log_per_slot).
    // Members (grow-only) not locals: keeps the diagnostic path off the heap on
    // every log line, and keeps the vector<char> off the stack — a local one
    // trips a GCC 12/13 -Wfree-nonheap-object false positive (fixed in GCC 14).
    std::vector<double> h_slot_norms_;
    std::vector<char>   h_slot_union_;

    // Closure builder (whenever the filter is on). Builds U = E = A ∪ neighbors(A)
    // each filter step: its membership flags gate the full launch
    // (ctx.u_flags), and with filter.ulist on its compact list is what the
    // kernels launch over. Until the first build the list is the identity.
    std::unique_ptr<UListBuilder> ulist_builder_;

    // Adaptive gate (hard threshold, no hysteresis): at a filter step a dense
    // interval (|E|/M > gate) falls back to the full grid (ctx.u_list=nullptr,
    // full M·N² launch gated by ctx.u_flags, bit-identical to compact, no
    // u_list indirection); else it runs the
    // compact whole-step. Guards the non-sparse regime (dynamics, E/M≈0.82) from
    // indirection overhead. `gate` value from filter.ulist. ulist_mode_ is the live
    // gate state, updated every step and read by emit_ado_stats to log a mode
    // flip; last_logged_mode_ is the last mode actually printed (so a flip is
    // logged once, on the stats line that follows it, not by a separate line).
    // Unset = ulist off / before the first build ⇒ no flip line.
    // ulist_tau_ = filter.ulist; 0 = compact launch off.
    enum class UlistMode { Unset, Compact, Dense };
    double    ulist_tau_        = 0.0;
    bool ulist_on() const { return ulist_tau_ > 0.0; }
    UlistMode ulist_mode_       = UlistMode::Unset;
    UlistMode last_logged_mode_ = UlistMode::Unset;
    // Last closure size |E| = |A ∪ neighbors(A)| from the most recent filter
    // step's rebuild (raw, pre-gate; the fallback below zeroes ctx.u_count but
    // not |E|). Empty means no |E| available (ulist off or before first build),
    // in which case print_filtering_statistics falls back to the |A|/M-only line.
    std::optional<int> closure_count_;

    // State
    std::vector<FilteringRecord> filtering_history_;
    bool diverged_ = false;
    bool initial_set_ = false;

    // Divergence guard baselines, one per slot: norm0_[b] = ‖ρ_0^(b)‖_F at the
    // last set_initial / apply_dipole, the only two calls that set them
    // (restore and expand inherit the snapshot's). Slot b fails loud once
    // ‖ρ_0^(b)(t)‖_F exceeds K · norm0_[b] (K = kDivergenceNormFactor below),
    // so a batched walk judges every slot exactly as the reuse walk judges
    // that leaf alone. The norm catches both trace-violating exponential
    // growth and traceless off-diagonal explosion — the latter mattering most
    // for spectroscopy paths where ρ(0) = μ⁺·ρ_eq is off-diagonal-only and
    // trace stays ≈ 0 even when the trajectory numerically blows up.
    std::vector<double> norm0_;

    // Filtering knobs (cached from cfg.filter.enabled / cfg.filter.tolerance)
    bool   use_filtering_;
    double filter_threshold_;
    int    total_active_ados_ = 0;

    // Time integrator (cached from cfg.time.integrator). IfRk4 folds Term 2 in
    // exactly (see rk4_step_if); Rk4 (default) is the historical bit-identical
    // path. The two GPUMemory tables below are the per-ADO factors E/E2; they
    // are allocated ONLY on the IfRk4 path (rk4 keeps zero extra footprint) and
    // recomputed (compute_if_factors) only when the internal dt changes.
    Integrator integrator_;
    GPUMemory<heom::complex_t> d_if_e_half_; // [M] exp(-½·dt·g_α), complex; empty unless IfRk4
    GPUMemory<heom::complex_t> d_if_e_full_; // [M] exp(-dt·g_α)
    std::optional<double> if_factors_dt_;  // dt (a.u.) the factors were built for; empty = not yet
    // "ρ was replaced and the hierarchy is still filling from zero." Set by
    // on_rho_replaced() (set_initial / apply_dipole), reinstalled from the
    // snapshot by restore_device / expand_slots, and cleared by
    // hierarchy_filling_() once a full step stops removing exact zeros.
    //
    // It guards ONE invariant: a filter pass must never decide the fate of an
    // ADO that has not had a chance to be populated. Such an ADO's norm is
    // exactly 0, and `norms[α] > threshold` is false at ANY positive threshold
    // (ado_filter.cu), so filtering marks it inactive and Term 4/5's neighbor
    // gate then releases the hierarchy one level per step — an error with no
    // tolerance dependence at all.
    //
    // One full step is not enough to populate it. FMO needs ⌈L/2⌉ (one RK4
    // step applies 𝓛 four times, each new site costs two applications), a
    // sparser H needs more. Hence a data-driven window, not a step count.
    //
    // This used to be spelled `step > 0`, which covered only the trajectory
    // start and therefore only dynamics/absorption; every 2DES dipole walked
    // straight into it. Keying off ρ replacement instead of the step counter
    // makes the guard structural rather than something each caller must know.
    // Evidence: filter growth-throttle / warmup-removal studies (filter→0
    // must match filter-off bit-for-bit on the same hierarchy).
    //
    // Per slot ([batch]): each slot opens and closes its own window.
    std::vector<char> fresh_state_;
    // Per-slot exact-zero ADO count at the start of the last fill-window
    // step; -1 = no step of this window has run yet.
    std::vector<int>  fresh_zeros_;
    // d_ado_norms_ holds max|ρ_α|² (IEEE bits) folded by the last step's
    // fused kFinal over exactly the ADOs it wrote (per slot: [batch×M]), and
    // nothing has touched ρ or d_ado_norms_ since. The next filter pass then
    // skips the standalone norm kernel. Set only by run_step; cleared by the
    // filter that consumes it and by every other writer of ρ or d_ado_norms_.
    bool   norms_fused_ = false;

    void allocate_trajectory_buffers();
    // Gather the B compact α=0 device blocks (slot stride M·Nb) into h_pack_
    // with one cudaMemcpy2D, then expand each into the full-matrix host mirror
    // h_rho0_ ([B][N²]) with exact +0 outside the block.
    void d2h_alpha0_batch();
    // Compact [n_row][n_col] (support_) → full [N][N], +0 outside the block.
    void expand_block_(const heom::complex_t* packed, heom::complex_t* full) const;
    // Build the cached ctx_ once (immutable fields + initial ulist pointers).
    // Idempotent: guarded by ctx_built_.
    void build_context();
    // Recompute norms → active_flags. U-local contract: scan_list
    // must cover the previous write set of ρ (nullptr = full M). The per-step
    // cadence in run_one_internal_step passes the previous step's launch set,
    // or nullptr after a dense step or a restore.
    // The manifold block bounds the element scan the same way it bounds the
    // derivative launch; it has NO default, so every caller has to say what it
    // scans: the cadence path passes ctx.block (== support_ by construction).
    void perform_ado_filtering(double threshold,
                               const heom::ManifoldBlock& blk,
                               const int* scan_list = nullptr,
                               int scan_count = 0,
                               const int* scan_off = nullptr);
    // "ρ was replaced, so nothing is known about the new hierarchy yet" — mark
    // every ADO active and put `ctx` on the dense launch. Marking active is
    // never wrong, only slower: it means "compute all six terms for this ADO".
    // The next propagation step's own filter pass prunes on propagated data.
    // Called instead of a filter pass when fresh_state_ is set.
    void activate_all_ados(HEOMContext& ctx);
    // Fill-window rule: the first step after a ρ replacement is always full;
    // after that, stay full while the previous step cut the number of
    // exact-zero ADOs and some are left. A count that stalls above 0 (ADOs
    // that are zero in the exact solution too) also ends the window, so it
    // always terminates. One full-M norm scan + one [width]-int D2H per call.
    // Updates fresh_state_[b] / fresh_zeros_[b] of every slot still filling.
    void hierarchy_filling_(const heom::ManifoldBlock& blk);
    void print_filtering_statistics(int step, double t_fs);
    // Compose+print one ADO-stats line ("|A|/|E|/M ... [L..]") with the given
    // prefix. Read-only: recomputes the level distribution from current
    // active_flags but never refilters. Shared by the cadence path
    // (print_filtering_statistics) and the forced path (log_stats).
    void emit_ado_stats(const std::string& prefix);

    // Build the UListBuilder (device-side, borrows GPU maps) and read
    // filter.ulist. No-op unless filtering is on. rebuild_ulist() recomputes U
    // from the current device active_flags at a filter step and returns |U|.
    void init_closure_builder();
    int  rebuild_ulist();

    // One internal RK4 step (filter + ulist rebuild/gate + RK4); the only
    // step logic there is, propagate_steps() loops over it. `step` is the
    // running index for the stats cadence and the logged time. Filtering runs
    // EVERY step after the fill window on both
    // paths — one cadence, no interval knob (cadence unification).
    void run_one_internal_step(HEOMContext& ctx, int step);
    void run_step(HEOMContext& ctx, double dt_internal_au);
    void report_divergence(double t_fs, int slot, double rho_norm,
                           double norm_ceiling, int step, int n_steps,
                           bool nan_in_hierarchy);

    // Single source of truth for the norm-based divergence guard, behind both
    // propagate_steps(check_div) and check_divergence_now(). Refreshes the
    // host α=0 mirror via one D2H, compares every active slot's ‖ρ_0‖_F
    // against that slot's ceiling, also reads the filter's NaN flag
    // (hierarchy NaN that ρ_0 has not felt yet), and on a blow-up prints the
    // report and latches diverged_. Returns diverged_.
    // With filtering off there is no norm pass and hence no flag; a NaN then
    // reaches ρ_0 unfiltered within ≤ L steps and the ‖ρ_0‖_F test catches it.
    bool check_divergence(double t_fs, int step, int n_steps);

    /// Frobenius norm of slot b's ρ_α=0, read from the host mirror h_rho0_.
    /// Caller must have refreshed it (d2h_alpha0_batch) first.
    double rho0_norm_(int b) const;
};

/// Multiplicative safety factor for the ‖ρ‖_F divergence ceiling of slot b:
///     ceil_b = max(norm0_[b] * kDivergenceNormFactor, kDivergenceNormFloor)
/// Picked at 10 to leave one order of magnitude of slack over the baseline
/// norm, well above transient hierarchy build-up but tight enough to catch
/// any actual exponential blow-up before NaN/Inf hits. The floor at 10
/// keeps the bound meaningful when the baseline is sub-unity (e.g. weak
/// dipole in absorption). Tunable in one place if a future spectroscopy
/// pathway proves it routinely needs more headroom.
inline constexpr double kDivergenceNormFactor = 10.0;
inline constexpr double kDivergenceNormFloor  = 10.0;

/**
 * @brief Build the initial ρ_0 matrix from a YAML-driven InitialState spec.
 *
 * Free function (not a member of either class) because it consumes both
 * SystemContext (eigenvectors for EXCITON_EXCITATION) and InitialState
 * (user input). Lives outside HEOMTrajectory so 2DES drivers can reuse
 * it for non-trivial initial states without instantiating a trajectory.
 */
Eigen::MatrixXcd build_initial_state(const SystemContext& sys,
                                     const InitialState& init);

#endif // HEOM_TRAJECTORY_HPP
