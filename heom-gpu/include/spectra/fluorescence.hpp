#ifndef HEOM_SPECTRA_FLUORESCENCE_HPP
#define HEOM_SPECTRA_FLUORESCENCE_HPP

#include "heom_config.hpp"
#include "spectra/dipole.hpp"
#include "spectra/linear_response.hpp"

#include <Eigen/Dense>
#include <vector>

class SystemContext;
class HEOMTrajectory;
struct DeviceHierarchySnapshot;

namespace heom::spectra {

/// One row per equilibration check: delta = ||rho(t) - rho(t-dt)|| / ||rho(t)||,
/// r = δ/δ_prev, ε = δ·r/(1−r) (∞ while r ≥ 1 or undefined).
struct EquilibrationHistory {
    std::vector<double> time_fs, delta, ratio, estimate;
    double t_eq_fs = 0.0;
};

struct FluorescenceResult {
    LinearResponseResult linear;   // rho_xyz / time_fs: same shape and meaning as absorption
    Eigen::MatrixXcd rho_eq;       // σ^eq_0, N×N (with_ground); empty in an MC ensemble file
    EquilibrationHistory equilibration;
};

/// Bare-H_S Boltzmann state of the (e,e) block, embedded N×N, exact zeros
/// elsewhere. Bare H_S = hamiltonian_site − diag(reorg_cm)·CM_TO_AU.
Eigen::MatrixXcd bare_boltzmann_ee(const SystemContext& sys);

/// Propagate traj (already set_initial'd) until max(δ, ε) ≤ tol.
/// On return traj holds σ^eq; snap is scratch (holds the previous check).
/// Throws runtime_error at t_max (with the last δ, r, ε and a hint) or on
/// divergence; invalid_argument if the intervals do not fit the time grid.
EquilibrationHistory equilibrate(HEOMTrajectory& traj, DeviceHierarchySnapshot& snap,
                                 const HEOMConfig::EquilibrationConfig& eq);

/// From a snapshot of σ^eq (width 1, (e,e) block): right-multiply μ⁺_α on every
/// ADO and record ρ_α(t) exactly as compute_linear_response does. traj.batch()
/// must be |active directions| (batched) or 1 (serial). Public so that a test
/// can feed it a factorized state.
LinearResponseResult emit_from_equilibrium(HEOMTrajectory& traj,
                                           const DeviceHierarchySnapshot& eq,
                                           const DipoleOperator& mu);

/// Bare Boltzmann start → dressed ADOs → equilibrate → emit.
FluorescenceResult compute_fluorescence(const SystemContext& sys, const DipoleOperator& mu);

}  // namespace heom::spectra

#endif  // HEOM_SPECTRA_FLUORESCENCE_HPP
