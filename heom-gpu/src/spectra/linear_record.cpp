#include "spectra/linear_record.hpp"

#include "system_context.hpp"
#include "heom_trajectory.hpp"
#include "dynamics_run.hpp"   // run_output_loop
#include "types.hpp"          // output_record_count(TimeParameters)

#include <complex>
#include <cstring>           // std::memcpy
#include <iostream>
#include <stdexcept>

namespace heom::spectra {

namespace {
constexpr char kAxis[3] = {'x', 'y', 'z'};
}

const Eigen::MatrixXcd& mu_plus(const DipoleOperator& mu, int dir) {
    const Eigen::MatrixXcd* dirs[3] = {&mu.mu_x, &mu.mu_y, &mu.mu_z};
    return *dirs[dir];
}

// Skip zero-direction trajectories. out.rho_xyz is already zero, so an
// untouched slice correctly means "this direction contributes nothing"
// — saves ~1/3 of the GPU work when all dipoles align with one axis.
std::vector<int> active_directions(const DipoleOperator& mu, const char* tag) {
    std::vector<int> dirs;
    for (int dir = 0; dir < 3; ++dir) {
        if (mu_plus(mu, dir).norm() >= 1e-12)
            dirs.push_back(dir);
        else if (tag)
            std::cout << "[" << tag << " " << kAxis[dir]
                      << "] skipped (zero dipole projection)\n";
    }
    return dirs;
}

// Each direction is an independent trajectory sharing H/maps/bath. The batch
// switch decides only how they are GROUPED into propagations: one B-wide
// launch for all of them, or one B=1 launch each. Filter-off is bit-identical
// between the two groupings; filter-on differs because a batched group filters
// on the union active set.
std::vector<std::vector<int>> direction_groups(const std::vector<int>& dirs, bool batch) {
    std::vector<std::vector<int>> groups;
    if (batch) {
        if (!dirs.empty()) groups.push_back(dirs);
    } else {
        for (int dir : dirs) groups.push_back({dir});
    }
    return groups;
}

// Group the propagation's Step/done lines under a labelled header, so a
// 3-directions-per-disorder-node MC log stays readable (the trajectory keeps
// its own "Step N (t=..)" + "done. (computation time)" lines).
std::string announce_group(const char* tag, const std::vector<int>& group) {
    const int B = static_cast<int>(group.size());
    const std::string label = (B == 1)
        ? std::string("[") + tag + " " + kAxis[group[0]] + "]"
        : std::string("[") + tag + " batch B=" + std::to_string(B) + "]";
    if (B == 1) std::cout << "\n" << label << " propagating\n";
    else        std::cout << "\n[" << tag << " batch] propagating B=" << B
                          << " direction(s)\n";
    return label;
}

// Pre-allocate the entire flat tensor up front so the per-record callback
// only has to memcpy into a known offset (no resize during the hot loop).
// n_outputs comes from the time grid, NOT from a throwaway HEOMTrajectory —
// the ctor allocates several MB of GPU buffers.
LinearResponseResult empty_linear_result(const SystemContext& sys) {
    const int N = sys.N();
    const int n_outputs = output_record_count(sys.config().time);
    LinearResponseResult out;
    out.N         = N;
    out.n_outputs = n_outputs;
    out.time_fs.assign(n_outputs, 0.0);
    out.rho_xyz.assign(static_cast<size_t>(3) * n_outputs * N * N,
                       std::complex<double>(0.0, 0.0));
    return out;
}

// The HEOM generator commutes with the excitation-number operator, so the
// (e,g) block is preserved for all time — the remaining elements are exactly
// 0.0, not merely small. Declaring the block lets the kernels skip them,
// which is bit-identical rather than an approximation. Rows [1,N) and column
// [0,1) are disjoint, as required; set_initial / apply_dipole check the claim.
heom::ManifoldBlock eg_block(const SystemContext& sys) {
    const auto lay = heom::ManifoldLayout::from(
        sys.config().physics.basis, sys.config().physics.n_states, sys.N());
    return lay.block(/*row=*/1, /*col=*/0);   // (e, g)
}

void record_group(HEOMTrajectory& traj, const std::vector<int>& group,
                  LinearResponseResult& out, const std::string& label) {
    using Cd = std::complex<double>;
    const int N = out.N;
    const int n_outputs = out.n_outputs;
    const int B = static_cast<int>(group.size());

    int records_seen = 0;
    heom::run_output_loop(traj,
        [&](int record_idx, double t_fs, const Cd* h_rho_b,
            int N_local, int B_local) {
            // Callback contract pins record_idx to [0, n_outputs). Hard error
            // on out-of-range so a miscount can't silently corrupt a different
            // direction's slab.
            if (record_idx < 0 || record_idx >= n_outputs) {
                throw std::runtime_error(
                    "record_group " + label + ": callback record_idx="
                    + std::to_string(record_idx)
                    + " out of [0, " + std::to_string(n_outputs) + ")");
            }
            if (N_local != N || B_local != B) {
                throw std::runtime_error(
                    "record_group " + label + ": callback N/B mismatch "
                    "(got N=" + std::to_string(N_local) + ", B="
                    + std::to_string(B_local) + "; expected N="
                    + std::to_string(N) + ", B=" + std::to_string(B) + ")");
            }
            ++records_seen;
            out.time_fs[record_idx] = t_fs;
            for (int s = 0; s < B; ++s) {
                const size_t rho_off =
                    (static_cast<size_t>(group[s]) * n_outputs + record_idx)
                    * static_cast<size_t>(N) * N;
                std::memcpy(
                    out.rho_xyz.data() + rho_off,
                    h_rho_b + static_cast<size_t>(s) * N * N,
                    sizeof(Cd) * static_cast<size_t>(N) * N);
            }
            // Deliberately do NOT compute Tr(B·ρ) here: C_α(t) is derived by
            // callers from out.rho_xyz on demand.
        });

    if (traj.diverged()) {
        throw std::runtime_error(
            "record_group " + label + ": HEOM trajectory diverged. "
            "Check L_cutoff, dt, K, or T.");
    }
    if (records_seen != n_outputs) {
        throw std::runtime_error(
            "record_group " + label + ": incomplete output records (saw "
            + std::to_string(records_seen) + " / "
            + std::to_string(n_outputs) + ")");
    }
}

}  // namespace heom::spectra
