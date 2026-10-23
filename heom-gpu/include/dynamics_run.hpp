#ifndef HEOM_DYNAMICS_RUN_HPP
#define HEOM_DYNAMICS_RUN_HPP

// The output loop and the dynamics CLI entry, OUTSIDE the propagation core.
//
// HEOMTrajectory knows how to move ρ (set_initial / apply_dipole / snapshot /
// restore / expand / propagate_steps) and nothing about when to look at it.
// "Every output_interval steps, hand ρ_0 to someone" is a policy of the
// dynamics and absorption drivers, and the HDF5 file is a policy of the CLI
// alone — so both live here, built on propagate_steps, and the core carries
// no writer, no timing, no record count. 2DES never calls either function.

#include <complex>
#include <functional>

class HEOMTrajectory;
class SystemContext;

namespace heom {

/// Per-record sink for run_output_loop. rho_b points to [B][N²] (slot-major,
/// row-major N×N per slot) in std::complex<double> — the hier→double uplift
/// has already happened. rec 0 is t=0; then +1 per output_interval steps.
using StepCallback = std::function<void(int rec, double t_fs,
                                        const std::complex<double>* rho_b,
                                        int N, int B)>;

/// Propagate traj over sys.config().time and call `sink` at every output
/// record, exactly as the old HEOMTrajectory::solve did: record 0 at t=0,
/// one record per output_interval steps, a divergence check after each
/// record (the record is delivered first, then the check runs — a diverged
/// record still reaches the sink), and a silent tail of n_steps %
/// output_interval steps that emits nothing. Prints the "=== Time Evolution
/// ===" / "Running N steps" / "done." lines. Requires set_initial to have
/// been called with batch() matrices (the sink receives every slot).
/// Returns the wall-clock seconds spent in the loop.
double run_output_loop(HEOMTrajectory& traj, const StepCallback& sink);

/// The `dynamics` task: ρ0 from cfg.initial (with the human-readable summary
/// on stdout), B=1, HDF5 output, "Final trace:" line. Returns true iff the
/// trajectory diverged.
bool run_dynamics(const SystemContext& sys);

}  // namespace heom

#endif  // HEOM_DYNAMICS_RUN_HPP
