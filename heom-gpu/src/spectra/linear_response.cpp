#include "spectra/linear_response.hpp"
#include "spectra/linear_record.hpp"

#include "system_context.hpp"
#include "heom_trajectory.hpp"

#include <Eigen/Dense>
#include <complex>

namespace heom::spectra {

namespace {

/// |g⟩⟨g| in the WITH_GROUND basis (ground sits at Hilbert index 0). This is
/// the ρ_eq approximation, valid for visible-light spectra in the
/// k_B T ≪ ℏω_eg, weak-coupling regime.
Eigen::MatrixXcd ground_projector(int N) {
    Eigen::MatrixXcd P = Eigen::MatrixXcd::Zero(N, N);
    P(0, 0) = std::complex<double>(1.0, 0.0);
    return P;
}

}  // namespace

LinearResponseResult compute_linear_response(
    const SystemContext& sys,
    const LinearResponseRequest& req)
{
    validate_linear_response_inputs(sys.config().physics.basis, req.dipole, sys.N());

    LinearResponseResult out = empty_linear_result(sys);
    // ρ_eq ≈ |g⟩⟨g| in the visible-light regime (ℏω_eg ≫ η, k_B T).
    const Eigen::MatrixXcd rho_eq = ground_projector(sys.N());
    const auto dirs = active_directions(req.dipole, "abs");

    for (const auto& group : direction_groups(dirs, sys.config().linear_batch)) {
        const std::string label = announce_group("abs", group);
        // A fresh trajectory per group is by design — the ctor is cheap
        // (sub-ms vs seconds of solve), and reusing one across groups would
        // need a reset() surface (active_flags, diverged_, filtering_history_,
        // …) that risks direction-to-direction state leakage. Don't.
        HEOMTrajectory traj(sys, static_cast<int>(group.size()));
        // ρ(0) = μ⁺|g⟩⟨g| lives entirely in the e×g block (column 0).
        std::vector<Eigen::MatrixXcd> rho0s;
        rho0s.reserve(group.size());
        for (int dir : group) rho0s.push_back(mu_plus(req.dipole, dir) * rho_eq);
        traj.set_initial(rho0s, eg_block(sys));
        record_group(traj, group, out, label);
    }

    // Intentionally no Ct_xyz / Ct_iso: they are derived quantities that
    // post-processing rebuilds from the saved ρ and dipole attrs on demand.
    return out;
}

}  // namespace heom::spectra
