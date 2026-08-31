#ifndef HEOM_TRAJECTORY_HPP
#define HEOM_TRAJECTORY_HPP

#include "system_context.hpp"
#include "types.hpp"
#include "gpu_memory.hpp"
#include "spectra/two_dim.hpp"         // DipoleSide
#include "core/heom_derivatives.cuh"   // HEOMContext (used by run_step)
#include "core/precision.hpp"          // heom::complex_t (hier precision buffers)

#include <Eigen/Dense>
#include <cuComplex.h>
#include <complex>       // StepCallback uses std::complex<double>*
#include <functional>    // StepCallback is std::function<...>
#include <memory>
#include <string>
#include <vector>

class HDF5Writer;
class UListBuilder;  // compact launch set U = E (opt-in via time_evolution.ulist)

// Device-resident carry-through snapshot. Holds the active ADO blocks gathered
// ON the GPU — no D2H/H2D and no per-element cuDoubleComplex↔std::complex
// conversion in the τ/T tree (a host-side snapshot would pay both per point).
// Buffers are grow-only so they are reused across the whole tree with no
// per-point cudaMalloc churn; two are live at once (τ and T).
struct DeviceHierarchySnapshot {
    int    N           = 0;
    int    n_active    = 0;
    int    batch       = 1;              // # slots captured (2DES batched-walk variable width)
    double alpha0_norm = 0.0;            // max_b ‖ρ_0^(b)‖_F at snapshot (divergence baseline)
    GPUMemory<heom::complex_t> blocks;   // [batch·n_active·N²] gathered active blocks (hier precision)
    GPUMemory<int>             ids;      // [n_active] ascending ADO ids (incl. α=0), SHARED by all slots
    GPUMemory<int8_t>          flags;    // [M] active_flags snapshot (filtering on), SHARED by all slots
};

/**
 * @brief One propagation run on top of an immutable SystemContext.
 *
 * Owns trajectory-specific GPU buffers (d_rho_, d_workspace_, filter
 * state), the host mirror of ρ_0, the HDF5 writer, divergence flag,
 * and timing. RK4 combines in-place into d_rho_ (no ping-pong buffer).
 *
 * Cheap to construct against the same SystemContext, by design — for 2DES
 * we need ~10⁴ HEOMTrajectory instances per Hamiltonian.
 *
 * Usage (1D site/exciton excitation, current default):
 *   SystemContext sys(cfg);
 *   HEOMTrajectory traj(sys);
 *   traj.set_initial_from_config();
 *   traj.solve();
 *   traj.save_results();
 *
 * Usage (2DES, future):
 *   SystemContext sys(cfg);
 *   HEOMTrajectory traj(sys, false);   // second arg = enable_hdf5
 *   traj.set_initial(rho_pathway);
 *   traj.solve();
 *   double trace = traj.final_trace();
 *
 * Replaces the "per-trajectory" half of the old HEOMSolver
 * (solve, save_results, initialize_density_matrices, allocate_gpu_memory
 * for mutable buffers).
 */
class HEOMTrajectory {
public:
    /**
     * @param sys          Immutable system context (must outlive this trajectory).
     * @param enable_hdf5  If true (default), opens an HDF5 file on solve() and
     *                     writes timesteps. For 2DES in-memory loops, pass false.
     * @param batch        Batch width B: propagate B density matrices that share
     *                     one SystemContext (H / maps / bath) in a single launch.
     *                     B==1 (default) is bit-exact to the historical single
     *                     path. The ctor sizes every mutable buffer by B and runs
     *                     the GPU budget check with B up front — one-shot, no
     *                     re-alloc/re-check window (hence a ctor arg, not a setter).
     */
    explicit HEOMTrajectory(const SystemContext& sys, bool enable_hdf5 = true,
                            int batch = 1);
    ~HEOMTrajectory();

    HEOMTrajectory(const HEOMTrajectory&)            = delete;
    HEOMTrajectory& operator=(const HEOMTrajectory&) = delete;

    /// Set initial state from an arbitrary N×N density matrix.
    /// The α=0 ADO slot is loaded with rho0; all α>0 slots are zeroed.
    /// Single-state API: throws on a batched trajectory (use set_initial_batch).
    void set_initial(const Eigen::MatrixXcd& rho0);

    /// Batched initial state: rho0s[b] is written to slot b's α=0 block, all
    /// α>0 slots zeroed. Requires rho0s.size() == batch(). The divergence
    /// ceiling uses max_b ‖ρ0_b‖_F (holds for the most energetic slot too).
    /// Resets the internal step counter (opens a fresh tree for every slot).
    void set_initial_batch(const std::vector<Eigen::MatrixXcd>& rho0s);

    /// Convenience wrapper: build initial ρ_0 from cfg.initial (YAML path).
    /// Internally calls set_initial(). Also captures pigment / exciton
    /// metadata for HDF5 attributes when enable_hdf5 is true.
    void set_initial_from_config();

    /// Per-output-record callback the solver
    /// invokes after each D2H copy of ρ_0 (the α=0 ADO).
    ///   record_idx     : 0 == t=0 initial state; thereafter +1 per
    ///                    output_interval HEOM step.
    ///   t_fs           : simulation time at this record.
    ///   h_rho_alpha0   : host pointer, length N²; row-major i*N+j layout
    ///                    matching GPU kernels and HDF5 datasets.
    ///   N              : Hilbert dimension (== sys.N()).
    /// Pure host-side type — spectra/dipole.hpp etc. need no CUDA include.
    using StepCallback = std::function<
        void(int record_idx,
             double t_fs,
             const std::complex<double>* h_rho_alpha0,
             int N)>;

    /// Run the time evolution. Reads time params from sys_.config().time.
    /// If enable_hdf5 is true, opens the HDF5 file and writes timesteps.
    /// If on_output_step is non-empty, it is invoked once per output record
    /// (record 0 == t=0 initial state, then every output_interval step).
    /// The default empty callback preserves the historical zero-overhead
    /// path for the dynamics CLI and 2DES in-memory drivers.
    void solve(const StepCallback& on_output_step = {});

    /// Batched per-output-record callback: h_rho_batch points to [batch][N²]
    /// (slot-major, contiguous), the B physical density matrices at this record.
    ///   record_idx : 0 == t=0 initial state; then +1 per output_interval step.
    ///   t_fs       : simulation time at this record.
    ///   N          : Hilbert dimension (== sys.N()).
    ///   batch      : number of slots B (== batch()).
    using BatchStepCallback = std::function<
        void(int record_idx,
             double t_fs,
             const std::complex<double>* h_rho_batch,
             int N,
             int batch)>;

    /// Batched time evolution: propagate all B slots together and fan each
    /// output record out through the BatchStepCallback. Requires the trajectory
    /// was constructed with enable_hdf5=false (the HDF5 schema is single-ρ0).
    /// B==1 is equivalent to solve() with a single-state callback.
    void solve_batch(const BatchStepCallback& on_output_step);

    /// Batch width B this trajectory was constructed with.
    int batch() const { return batch_; }

    /// 2DES batched-walk variable width: set the number of ACTIVE slots B_active
    /// (1 ≤ B_active ≤ batch()). Every subsequent launch (derivatives / RK4 /
    /// filter / detection / divergence) spans only the first B_active slots via
    /// grid.y = ctx_.batch. The τ/T/t layers of the 3→9→21 walk run at width
    /// 3/9/21 on ONE batch=21 trajectory. Directly updates ctx_.batch too:
    /// build_context() is idempotent (ctx_built_ guard), so mutating only the
    /// member would leave ctx_.batch frozen and silently pin the launch width.
    /// Default active_batch_ == batch_ ⇒ B==1 and absorption (B=3) unchanged.
    void set_active_batch(int batch);

    /// Current active batch width (see set_active_batch).
    int active_batch() const { return active_batch_; }

    /// Declare that ρ's support is confined to rows [row_off, row_off+n_row) ×
    /// cols [col_off, col_off+n_col), so every launch can skip the rest.
    ///
    /// This is a CLAIM ABOUT THE PHYSICS, and the kernel takes it literally:
    /// elements outside the block are neither read nor written, so if any is
    /// actually nonzero it is silently frozen and the answer is wrong. Only
    /// declare a block a pulse sequence guarantees — after μ_ge the support is
    /// (e,g), after a second μ on an ESA pathway it is (f,e).
    ///
    /// The two ranges must be identical or disjoint (manifolds partition the
    /// basis); anything else throws.
    ///
    /// The descriptor is NOT sticky across a ρ replacement: dipole / restore /
    /// expand each call on_rho_replaced(), which clears it back to the whole
    /// matrix. Forgetting to re-declare a block therefore costs speed, never
    /// correctness — the failure mode that leaves fingerprints instead of the
    /// one that doesn't.
    ///
    /// Setting HEOM_NO_MANIFOLD_BLOCK in the environment makes the kernel run
    /// the whole matrix anyway, which is how you A/B the optimisation without
    /// touching a config (and hence without perturbing config_hash). The switch
    /// acts where the descriptor is CONSUMED, not here: this call still records
    /// and still validates, so both halves of an A/B run take the same path and
    /// propagations_without_block() keeps measuring the driver rather than the
    /// environment.
    void set_manifold_block(int row_off, int n_row, int col_off, int n_col);

    /// Drop back to the whole N×N matrix. Idempotent, silent.
    void clear_manifold_block();

    /// Propagation segments that ran without any declared block, i.e. over the
    /// whole N×N matrix. Correct but unaccelerated, and invisible to a bitwise
    /// A/B gate — the two runs agree precisely because nothing was blocked. The
    /// 2DES driver asserts this is zero; absorption declares the e×g block, so
    /// it too should stay at zero. Dynamics never declares one (ρ fills the ee
    /// block), so for it a nonzero count is simply the truth.
    long long propagations_without_block() const {
        return propagations_without_block_;
    }

    /// Close HDF5 file (if open) and print final trace.
    /// Safe to call even if enable_hdf5 was false — just prints trace.
    void save_results();

    bool diverged()                 const { return diverged_; }
    double simulation_time_seconds() const { return simulation_time_seconds_; }

    /// Tr(ρ_0) reading from the host mirror. Cheap (N additions).
    double final_trace() const;

    /// Copy ρ_0 from device to a fresh Eigen matrix. Used by 2DES drivers
    /// for response-function expectation values. Performs one D2H copy.
    Eigen::MatrixXcd final_rho0() const;

    /// Copy slot b's α=0 block (device) to a fresh N×N matrix (one D2H). b must
    /// be in [0, batch()). Per-slot readback for the 2DES batched parity tests
    /// (final_rho0() only sees slot 0). Reads the raw device state — no filter /
    /// no host-mirror dependency.
    Eigen::MatrixXcd slot_rho0(int b) const;

    // 2DES carry-through primitives.
    /// Apply a single-side dipole μ (N×N host matrix) to the FULL ADO
    /// hierarchy; result is written back into d_rho_. Post-condition: if
    /// filtering is on, immediately refilter (recompute norms→flags) + rebuild
    /// ulist — μ shifts the norm distribution, so stale flags are invalid.
    void apply_dipole_full_hierarchy(const Eigen::MatrixXcd& mu,
                                     heom::spectra::DipoleSide side);

    /// 2DES per-slot dipole. Apply a distinct operator to each of the
    /// active_batch_ slots' full hierarchy in one batched launch: slot b gets
    /// mus[b] on the given `side`. mus.size() must equal active_batch(). Post:
    /// same as apply_dipole_full_hierarchy but batched — refilter (union over
    /// active slots) and refresh the divergence baseline to the max ||rho0||,
    /// so the t-scan's check_divergence_now uses the post-dipole
    /// baseline, not the pre-excitation one). B=1 degenerates bit-for-bit to
    /// apply_dipole_full_hierarchy.
    void apply_dipole_per_slot(const std::vector<Eigen::MatrixXcd>& mus,
                               heom::spectra::DipoleSide side);

    /// Device-resident snapshot/restore (the 2DES carry-through primitive).
    /// snapshot: recompute norms→flags, gather the active ADO blocks; restore:
    /// zero+scatter+flags+ulist. The gathered blocks stay on the GPU (D2D),
    /// eliminating the per-τ/T-point n_active·N² D2H/H2D and the host element-
    /// conversion loops. `snap` buffers grow-only and are reused across the tree
    /// by the caller. Restore is safe to propagate immediately afterwards (no
    /// "wait for next filter step" window).
    void snapshot_device(DeviceHierarchySnapshot& snap);
    void restore_device(const DeviceHierarchySnapshot& snap);

    /// 2DES batched-walk fan-out primitive. Expand a parent snapshot
    /// (B_parent slots, shared ids/flags) into B_child = child_to_parent.size()
    /// slots of d_rho_: child slot c is filled from parent slot child_to_parent[c]
    /// (a general, possibly uneven, map — 3→9 by m1, 9→21 by (m1,m2)). Zeroes the
    /// B_child·M·N² working prefix, scatters each child from its parent's active
    /// blocks (shared ids), restores the shared flags/ulist + divergence baseline
    /// from the parent, and sets active_batch_ = B_child. child_to_parent[c] must
    /// be in [0, parent.batch).
    void expand_slots(const DeviceHierarchySnapshot& parent,
                      const std::vector<int>& child_to_parent);

    /// Propagate n internal RK4 steps with no reset / no HDF5 / no record
    /// callback (carry-through). Uses the cross-segment internal step counter.
    /// check_div=true (default) keeps the per-segment ‖ρ_0‖_F divergence guard
    /// (one N² D2H at the end of the call). The 2DES t-scan passes false so the
    /// innermost advance does NOT pay a per-t-point D2H+sync; the driver runs
    /// one check_divergence_now() per t-scan instead.
    void propagate_steps(int n_internal_steps, bool check_div = true);

    /// Public one-shot divergence guard for the 2DES carry-through driver:
    /// one N² D2H of ρ_0 + Frobenius-norm ceiling test. Latches diverged() and
    /// returns it. Lets the t-scan coarsen divergence detection to once per scan
    /// instead of once per t-point.
    bool check_divergence_now(double t_fs, int step);

    // ── 2DES batched device-side detection ─────────────
    /// Begin a detection batch for one pathway tree: upload the (fixed) μ_det
    /// operator once and size the device result grid to n_samples. Replaces
    /// the per-t-point expectation() host round-trip.
    void detect_begin(const Eigen::MatrixXcd& mu_det, int n_samples);
    /// Sample Tr(μ_det·ρ_0) into device slot `idx` (async, no sync / no D2H).
    /// Must be called after detect_begin and a propagate; reads the α=0 block.
    void detect_sample(int idx);
    /// Flush the whole detection grid to host with ONE D2H (n_samples scalars),
    /// converting cuDoubleComplex → std::complex into out_host[0..n_samples).
    void detect_flush(std::complex<double>* out_host, int n_samples);

    /// 2DES batched detection. Set up B_det = mus_det.size()
    /// distinct detection dipoles (one per slot / pathway) and a
    /// [B_det·n_samples] result grid. The detection width is taken from
    /// mus_det.size() and stored (detect_batch_) — it is DECOUPLED from
    /// active_batch_, which may differ at the call site; detect_sample_batch
    /// always launches at detect_batch_. B_det must be in [1, batch()].
    void detect_begin_batch(const std::vector<Eigen::MatrixXcd>& mus_det,
                            int n_samples);
    /// Sample Tr(μ_det^(b)·ρ_0^(b)) for every detection slot b into column idx
    /// (async, no sync / no D2H). grid.y = detect_batch_; slot b reads its own
    /// α=0 block (b·M·N²) and its own μ_det (b·N²), writes out[b·n_samples+idx].
    void detect_sample_batch(int idx);
    /// Flush the [B_det·n_samples] grid to host (ONE D2H). out_host must hold
    /// detect_batch()·n_samples scalars, slot-major: slot b at [b·n_samples..).
    void detect_flush_batch(std::complex<double>* out_host, int n_samples);
    /// Detection width set by the last detect_begin/detect_begin_batch.
    int detect_batch() const { return detect_batch_; }

    /// Manifold-occupancy probe. Reports, for
    /// every (row manifold, column manifold) block, the max |ρ| found anywhere
    /// in the hierarchy; prints ONLY the occupied blocks plus their count, so a
    /// line carrying two block names IS the violation. Gated by the
    /// HEOM_BLOCK_PROBE environment variable; a plain early return otherwise.
    ///
    /// ACCEPTANCE TOOLING, NOT A ROUTINE SWITCH: each call D2Hs the entire
    /// [B][M][N][N] hierarchy (2 GB per call at FMO B=21). Use it on
    /// dimer-scale recipes to localise which block a bookkeeping bug landed in
    /// — never on a production grid.
    void probe_blocks(const char* where) const;

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

    /// Physical-layer expectation Tr(B·ρ_0). Performs its OWN D2H of the α=0
    /// block — do NOT use final_rho0() here, its host mirror is stale after
    /// propagate_steps() (which does not emit).
    std::complex<double> expectation(const Eigen::MatrixXcd& B) const;

    /// Number of output records this trajectory will emit, given the time
    /// configuration in sys_.config().time. Forwards to the free function
    /// `output_record_count(TimeParameters)` defined in include/types.hpp
    /// (single source of truth for the formula).
    int n_output_records() const {
        return output_record_count(sys_.config().time);
    }

private:
    const SystemContext& sys_;
    const bool enable_hdf5_;

    // Batch width B (ctor arg). Every mutable O(M·N²) buffer (d_rho_,
    // d_workspace_, d_kahan_comp_) and the α=0 host mirror are sized by B;
    // read-only system tables stay single-copy. B==1 ⇒ pre-batch layout.
    const int batch_;

    // 2DES batched-walk active width B_active ≤ batch_ (set_active_batch). Drives
    // grid.y for every launch and the slot count for filter/divergence/init.
    // Initialised to batch_ so the non-2DES paths (dynamics/absorption) are
    // unchanged. Buffers stay sized by batch_ (max); only the launch width moves.
    int active_batch_;

    // GPU mutable buffers (per-trajectory). d_rho_ is always the current
    // state: rk4_step's final combine is in-place (rho += dt/6 · acc), so
    // no second rho buffer / swap exists. hier precision (FP32 in mixed): the
    // O(M·N²) big arrays that own the bandwidth cost.
    GPUMemory<heom::complex_t> d_rho_;
    GPUMemory<heom::complex_t> d_workspace_;

    // Kahan compensation carry for the RK4 final combine. Only
    // allocated + zero-inited in the mixed-precision build (heom::kMixedPrecision);
    // empty (nullptr) in the FP64 build so rk4_step's combine stays bit-identical
    // to history. Persists across steps — the carry lives here.
    GPUMemory<heom::complex_t> d_kahan_comp_;

    // Manifold block descriptor (see set_manifold_block). blk_set_ == false
    // means "whole matrix"; ctx_ then gets (0,N,0,N) rather than a sentinel for
    // the kernel to interpret. blk_set_ means exactly "the driver declared a
    // block for the current ρ" — HEOM_NO_MANIFOLD_BLOCK does not touch it.
    int  blk_row_off_ = 0;
    int  blk_n_row_   = 0;
    int  blk_col_off_ = 0;
    int  blk_n_col_   = 0;
    bool blk_set_     = false;

    long long propagations_without_block_ = 0;

    // The single place ctx_'s four launch-domain ints are written, so the
    // descriptor and the context cannot drift apart. Also the single place the
    // kill switch is honoured: we know the block, we choose not to use it.
    void apply_manifold_block_to_ctx();

    // Invariant hook: call this immediately after ANY operation that replaces ρ
    // wholesale (dipole / restore / expand / set_initial). Everything derived
    // from the OLD ρ is invalid at that instant — the Kahan carry records bits
    // lost while accumulating it, so a stale carry is injected straight back
    // into elements the replacement just cleared. Clearing is not an
    // optimisation, it is the lifecycle rule.
    //
    // Zeroes the WHOLE buffer, not the active prefix: the prefix version needs
    // the reader to carry "slots past the prefix are always re-zeroed by the
    // next widening" in their head, and these calls happen once per τ/T segment
    // against a propagation that touches the same footprint on every RK4 stage.
    //
    // Also clears the manifold block, for the same reason: it described where
    // the OLD ρ had support.
    void on_rho_replaced();

    // 2DES: lazy-allocated device scratch for the N×N dipole μ
    // (apply_dipole_full_hierarchy). Single-side μ needs a distinct output
    // buffer; we reuse d_workspace_ as the output then copy back into d_rho_.
    GPUMemory<heom::complex_t> d_mu_scratch_;

    // 2DES: reusable host buffer for packing the N×N μ matrix before its H2D
    // (apply_dipole_full_hierarchy). A member so we don't heap-alloc a fresh
    // std::vector on every pulse interaction.
    std::vector<heom::complex_t> h_mu_pack_;

    // 2DES batched detection. d_mu_det_ holds the per-tree
    // μ_det (N²); d_detect_ is the per-tree result grid (n_samples scalars)
    // the trace kernel writes into; h_detect_ stages the single bulk D2H.
    // Lazy-allocated on the first detect_begin; dynamics/absorption never pay.
    GPUMemory<heom::complex_t> d_mu_det_;
    GPUMemory<heom::complex_t> d_detect_;
    std::vector<heom::complex_t> h_detect_;

    // 2DES batched detection width + result-grid column count, set by
    // detect_begin[_batch]. detect_batch_ drives the trace launch grid.y and the
    // out[b·n_samples+idx] slot stride; single-slot detect keeps detect_batch_=1
    // with the historical out[idx] layout. detect_n_samples_ is the n_samples
    // stored at begin (the per-slot column count for the batched trace).
    int detect_batch_     = 1;
    int detect_n_samples_ = 0;

    // Cached HEOM context: immutable device pointers + scalars borrowed from
    // SystemContext, built once (build_context()) and reused by solve() and
    // propagate_steps(). Per-filter fields (u_list/u_count) still mutate each
    // filter step.
    HEOMContext ctx_;
    bool ctx_built_ = false;

    // Running internal-step counter for propagate_steps(): accumulates across
    // carry-through segments so filter warmup/interval stays continuous.
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

    // Filter buffers (only allocated when filtering is on)
    GPUMemory<double>  d_ado_norms_;
    GPUMemory<int8_t>  d_active_flags_;
    GPUMemory<int>     d_level_counts_;

    // Host mirror of ρ_0 only (N², not M·N²). hier precision
    // (matches d_rho_'s element type for the raw D2H). Consumers uplift each
    // element to std::complex<double> via c_re/c_im (observable uplift):
    // the FP32→FP64 promotion is exactly the "read out in double" boundary.
    std::vector<heom::complex_t> h_rho0_;
    std::vector<int> h_level_counts_;

    // slot_active_stats() scratch (batched per-slot diagnostic, log_per_slot).
    // Members (grow-only) not locals: keeps the diagnostic path off the heap on
    // every log line, and keeps the vector<char> off the stack — a local one
    // trips a GCC 12/13 -Wfree-nonheap-object false positive (fixed in GCC 14).
    std::vector<double> h_slot_norms_;
    std::vector<char>   h_slot_union_;

    // FP64 staging for the HDF5 writer (which is float64-only, schema unchanged).
    // In the FP64 build complex_t IS cuDoubleComplex, so rho0_as_fp64() returns
    // h_rho0_ directly (zero copy); in the mixed build it fills this on demand.
    mutable std::vector<cuDoubleComplex> h_rho0_fp64_;

    // Lazy std::complex mirror of rho_0 used to
    // hand out a stable host pointer to StepCallback. Only allocated on the
    // first solve() call that actually has a callback (zero overhead for
    // the dynamics CLI / 2DES enable_hdf5=false drivers that don't pass
    // one). Layout matches h_rho0_ (row-major, length N²) but with the
    // CUDA-free std::complex<double> element type required by the public
    // callback signature.
    std::vector<std::complex<double>> h_rho_alpha0_mirror_;

    // U-list launch compaction (only when time_evolution.ulist is on). Builds the
    // compact closure set U = E = A ∪ neighbors(A) each filter step and exposes
    // a device int list the kernels launch over. Until the first build it is
    // the identity (full launch), so warmup stays byte-equal.
    std::unique_ptr<UListBuilder> ulist_builder_;

    // Adaptive gate (hard threshold, no hysteresis): at a filter step a dense
    // interval (|E|/M > gate) falls back to the full grid (ctx.u_list=nullptr,
    // full M·N² launch, byte-equal, no u_list indirection); else it runs the
    // compact whole-step. Guards the non-sparse regime (dynamics, E/M≈0.82) from
    // indirection overhead. `gate` value from filter.ulist (env HEOM_ULIST_TAU
    // overrides for sweeps; kept for backward compat). ulist_dense_ is the live
    // gate state, updated every step and read by emit_ado_stats to log a mode
    // flip; last_logged_dense_ is the last mode actually printed (so a flip is
    // logged once, on the stats line that follows it, not by a separate line).
    double ulist_tau_        = 0.5;
    int    ulist_dense_      = -1;   // -1 unset, 0 compact, 1 full fallback
    int    last_logged_dense_ = -1;  // last mode emitted by emit_ado_stats
    // Last closure size |E| = |A ∪ neighbors(A)| from the most recent filter
    // step's rebuild (raw, pre-gate; the fallback below zeroes ctx.u_count but
    // not |E|). -1 means no |E| available (ulist off or before first build), in
    // which case print_filtering_statistics falls back to the |A|/M-only line.
    int    closure_count_ = -1;

    // Output / state
    std::unique_ptr<HDF5Writer> writer_;
    std::vector<FilteringRecord> filtering_history_;
    bool diverged_ = false;
    bool initial_set_ = false;
    double simulation_time_seconds_ = 0.0;

    // Divergence guard: ‖ρ_0(0)‖_F captured at set_initial() time. The
    // solve() loop fails loud once ‖ρ_0(t)‖_F exceeds K · initial_norm_
    // (K = kDivergenceNormFactor below). This catches both trace-violating
    // exponential growth and traceless off-diagonal explosion — the latter
    // mattering most for spectroscopy paths where ρ(0) = μ⁺·ρ_eq is
    // off-diagonal-only and trace stays ≈ 0 even when the trajectory
    // numerically blows up. See the bug 3 fix in the issue tracker for
    // the prior trace-only guard's failure mode.
    double initial_norm_ = 0.0;

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
    GPUMemory<heom::real_t> d_if_e_half_; // [M] exp(-½·dt·g_α); empty unless IfRk4
    GPUMemory<heom::real_t> d_if_e_full_; // [M] exp(-dt·g_α)
    double            last_if_dt_ = -1.0;  // dt (a.u.) the factors were built for
    // Count of filter passes that have actually run this trajectory. The
    // first-ever pass (index 0) is pinned to the warmup threshold so the
    // warmup knob can never silently no-op. See run_one_internal_step.
    int    filter_pass_index_ = 0;

    // "ρ was just replaced, the hierarchy is not populated yet." Set by
    // on_rho_replaced() — the single choke point for all four replacements
    // (set_initial_batch / apply_dipole / restore_device / expand_slots) — and
    // consumed by the first run_one_internal_step after it.
    //
    // It guards ONE invariant: a filter pass must never decide the fate of an
    // ADO that has not had a chance to be populated. Norms of a fresh hierarchy
    // are exactly 0 for every α>0, and `norms[α] > threshold` is false at ANY
    // positive threshold (ado_filter.cu), so filtering here marks the whole
    // hierarchy inactive and Term 4/5's neighbor gate then holds it there —
    // an error with no tolerance dependence at all.
    //
    // This used to be spelled `step > 0`, which covered only the trajectory
    // start and therefore only dynamics/absorption; every 2DES dipole walked
    // straight into it. Keying off ρ replacement instead of the step counter
    // makes the guard structural rather than something each caller must know.
    bool   fresh_state_ = true;

    // Initial state metadata captured at set_initial_from_config time
    // (HDF5 writes pigment-1-based excited_site / exciton_index attrs).
    Eigen::MatrixXcd initial_rho_;   // exact ρ_0 used; passed to HDF5

    void allocate_trajectory_buffers();
    // Gather the B α=0 device blocks (slot stride M·N²) into the contiguous
    // host mirror h_rho0_ ([B][N²]) with one cudaMemcpy2D. B==1 degenerates to
    // a single N² D2H, bit-identical to the historical copy_to_host.
    void d2h_alpha0_batch();
    // Shared main loop for solve()/solve_batch(): `emit(record_idx, t_fs)` owns
    // the per-record D2H + fan-out; the loop is fan-out-agnostic.
    void run_solve_loop(const std::function<void(int, double)>& emit);
    // Build the cached ctx_ once (immutable fields + initial ulist pointers).
    // Idempotent: guarded by ctx_built_.
    void build_context();
    // Recompute norms → active_flags. U-local contract: scan_list
    // must cover the previous write set of ρ (nullptr = full M). Cold paths
    // (dipole apply, snapshot, restore) pass nullptr; the per-step cadence in
    // run_one_internal_step passes the previous step's launch set.
    // The manifold block bounds the element scan the same way it bounds the
    // derivative launch; it has NO default, so every caller has to say whether
    // it means a real block or the whole matrix (0, N, 0, N). Cold callers must
    // pass the whole matrix — see the trap documented at the call site in
    // refresh_active_set_and_ulist().
    void perform_ado_filtering(double threshold,
                               int row_off, int n_row, int col_off, int n_col,
                               const int* scan_list = nullptr,
                               int scan_count = 0);
    // Rebuild the compact launch set U = A∪neighbors(A) from the current device
    // active_flags and point ctx_ at it (also updates closure_count_ for stats).
    // No-op when ulist is off. This is the identical tail every cold path used
    // to inline; the per-step cadence in run_one_internal_step keeps its own
    // variant because it layers the adaptive dense/compact gate on top.
    void rebuild_ulist_into_ctx();
    // "State just changed, resync the active set" — the shared cold-path
    // primitive: build ctx_, refilter ρ's norms → active_flags (full-M scan),
    // then rebuild_ulist_into_ctx(). Used after a dipole pulse and at snapshot
    // time, where the norm distribution shifted and stale flags are invalid.
    // Restore paths deliberately do NOT call this: their flags come from the
    // snapshot, so they only rebuild_ulist_into_ctx() (no refilter).
    void refresh_active_set_and_ulist(double threshold);
    // "ρ was replaced, so nothing is known about the new hierarchy yet" — mark
    // every ADO active and put `ctx` on the dense launch. Marking active is
    // never wrong, only slower: it means "compute all six terms for this ADO".
    // The next propagation step's own filter pass prunes on propagated data.
    // Called instead of a filter pass when fresh_state_ is set.
    void activate_all_ados(HEOMContext& ctx);
    void print_filtering_statistics(int step, double t_fs);
    // Compose+print one ADO-stats line ("|A|/|E|/M ... [L..]") with the given
    // prefix. Read-only: recomputes the level distribution from current
    // active_flags but never refilters. Shared by the cadence path
    // (print_filtering_statistics) and the forced path (log_stats).
    void emit_ado_stats(const std::string& prefix);

    // Lazily build the UListBuilder (device-side, borrows GPU maps) and, when
    // set, the initial identity list. No-op unless filtering is on and
    // time_evolution.ulist is on. rebuild_ulist() recomputes U from the current device
    // active_flags at a filter step and returns |U| (ctx.u_count).
    void init_ulist_if_enabled();
    int  rebuild_ulist();

    // One internal RK4 step (filter + ulist rebuild/gate
    // + RK4), shared by solve() and propagate_steps() so the step logic has
    // exactly one copy (DRY). `step` is the running index driving filter
    // warmup and ulist gate logging. Filtering runs EVERY step (>0) on both
    // paths — one cadence, no interval knob (Phase 3 / cadence unification).
    void run_one_internal_step(HEOMContext& ctx, int step);
    void run_step(HEOMContext& ctx, double dt_internal_au);
    void report_divergence(double t_fs, double rho_norm, double norm_ceiling,
                           int step, int n_steps);

    // Single source of truth for the norm-based divergence guard, shared by
    // solve() (output-step cadence) and propagate_steps() (the 2DES carry-
    // through path, which never calls solve()). Refreshes the host α=0 mirror
    // via one N² D2H, compares ‖ρ_0‖_F against the ceiling, and on a blow-up
    // (or NaN/Inf) prints the report and latches diverged_. Returns diverged_.
    bool check_divergence(double t_fs, int step, int n_steps);

    /// Frobenius norm of the physical density matrix ρ_α=0(t) read from
    /// the host mirror h_rho0_. Caller must have refreshed h_rho0_ via the
    /// solve() loop's emit lambda (D2H copy) before calling this. Pure
    /// host-side N² float-ops; cheap.
    double rho0_frobenius_norm() const;

    // FP64 view of the ρ_0 host mirror for the float64-only HDF5 writer. FP64
    // build: returns h_rho0_.data() reinterpreted (same type, zero copy). Mixed
    // build: uplifts h_rho0_ (float) into h_rho0_fp64_ and returns that. The one
    // place the observable crosses FP32→FP64 for on-disk output.
    const cuDoubleComplex* rho0_as_fp64() const;
};

/// Multiplicative safety factor for the ‖ρ‖_F divergence ceiling:
///     ceil = max(initial_norm_ * kDivergenceNormFactor, kDivergenceNormFloor)
/// Picked at 10 to leave one order of magnitude of slack over the initial
/// norm, well above transient hierarchy build-up but tight enough to catch
/// any actual exponential blow-up before NaN/Inf hits. The floor at 10
/// keeps the bound meaningful when initial_norm_ is sub-unity (e.g. weak
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
