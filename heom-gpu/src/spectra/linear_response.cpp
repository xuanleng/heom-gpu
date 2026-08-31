#include "spectra/linear_response.hpp"

#include "system_context.hpp"
#include "heom_trajectory.hpp"
#include "types.hpp"          // output_record_count(TimeParameters)

#include <Eigen/Dense>
#include <array>
#include <complex>
#include <cstring>           // std::memcpy
#include <iostream>
#include <stdexcept>

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

/// ρ(0) = μ⁺|g⟩⟨g| lives entirely in the e×g block: μ⁺ raises the left index
/// while the right one stays in g, so only column 0 is populated. The HEOM
/// generator commutes with the excitation-number operator, so that block is
/// preserved for all time — the remaining N²-(N-1) elements are exactly 0.0,
/// not merely small. Declaring the block lets the kernels skip them, which is
/// bit-identical rather than an approximation.
///
/// This is the narrowest block in the codebase (n_col = 1); the 2DES paths
/// declare wider ones. Rows [1,N) and column [0,1) are disjoint, as required.
///
/// MUST be called AFTER set_initial/set_initial_batch: installing a new ρ
/// clears the descriptor (HEOMTrajectory::on_rho_replaced).
void declare_eg_block(HEOMTrajectory& traj, int N) {
    traj.set_manifold_block(/*row_off=*/1, /*n_row=*/N - 1,
                            /*col_off=*/0, /*n_col=*/1);
}

}  // namespace

LinearResponseResult compute_linear_response(
    const SystemContext& sys,
    const LinearResponseRequest& req)
{
    using Cd = std::complex<double>;

    const int N = sys.N();
    validate_linear_response_inputs(sys.config().physics.basis, req.dipole, N);

    // Single source of truth for the output record count. Do NOT construct a
    // throwaway HEOMTrajectory to read it — the ctor allocates several MB of GPU
    // buffers just to compute n_steps/output_interval + 1.
    const int n_outputs = output_record_count(sys.config().time);

    const auto& mu = req.dipole;
    const std::array<const Eigen::MatrixXcd*, 3> dirs = {
        &mu.mu_x, &mu.mu_y, &mu.mu_z
    };

    // Pre-allocate the entire flat tensor up front so the per-record
    // callback only has to memcpy into a known offset (no resize during
    // hot loop, no realloc during long runs).
    LinearResponseResult out;
    out.N         = N;
    out.n_outputs = n_outputs;
    out.time_fs.assign(n_outputs, 0.0);
    out.rho_xyz.assign(static_cast<size_t>(3) * n_outputs * N * N,
                       Cd(0.0, 0.0));

    // ρ_eq ≈ |g⟩⟨g| in the visible-light regime (ℏω_eg ≫ η, k_B T).
    const Eigen::MatrixXcd rho_eq = ground_projector(N);

    static constexpr char kAxis[3] = {'x', 'y', 'z'};

    // Batched path (absorption.batch: true): pack the ≤3 non-zero polarization
    // directions into ONE B-wide propagation. Same physics as the serial loop
    // below (each direction is an independent ρ(0)=μ⁺ρ_eq trajectory sharing
    // H/maps/bath); filter-off is bit-identical, filter-on uses the union
    // active set, so directions no longer filter fully independently.
    if (sys.config().absorption_batch) {
        std::vector<int> dirs_active;
        for (int dir = 0; dir < 3; ++dir) {
            if (dirs[dir]->norm() < 1e-12) {
                std::cout << "[abs " << kAxis[dir]
                          << "] skipped (zero dipole projection)\n";
            } else {
                dirs_active.push_back(dir);
            }
        }
        if (dirs_active.empty()) return out;   // all directions zero

        const int B = static_cast<int>(dirs_active.size());
        std::cout << "\n[abs batch] propagating B=" << B << " direction(s)\n";

        HEOMTrajectory traj(sys, /*enable_hdf5=*/false, /*batch=*/B);
        std::vector<Eigen::MatrixXcd> rho0s;
        rho0s.reserve(B);
        for (int dir : dirs_active) rho0s.push_back((*dirs[dir]) * rho_eq);
        traj.set_initial_batch(rho0s);
        declare_eg_block(traj, N);
        traj.set_log_prefix("[abs batch B=" + std::to_string(B) + "] ");

        int records_seen = 0;
        traj.solve_batch(
            [&](int record_idx, double t_fs, const Cd* h_rho_b,
                int N_local, int B_local) {
                if (record_idx < 0 || record_idx >= n_outputs) {
                    throw std::runtime_error(
                        "compute_linear_response: batch callback record_idx="
                        + std::to_string(record_idx)
                        + " out of [0, " + std::to_string(n_outputs) + ")");
                }
                if (N_local != N) {
                    throw std::runtime_error(
                        "compute_linear_response: batch callback N mismatch");
                }
                ++records_seen;
                out.time_fs[record_idx] = t_fs;
                for (int s = 0; s < B_local; ++s) {
                    const int dir = dirs_active[s];
                    const size_t rho_off =
                        (static_cast<size_t>(dir) * n_outputs + record_idx)
                        * static_cast<size_t>(N) * N;
                    std::memcpy(
                        out.rho_xyz.data() + rho_off,
                        h_rho_b + static_cast<size_t>(s) * N * N,
                        sizeof(Cd) * static_cast<size_t>(N) * N);
                }
            });

        if (traj.diverged()) {
            throw std::runtime_error(
                "compute_linear_response: HEOM trajectory diverged during "
                "batched absorption propagation. Check L_cutoff, dt, K, or T.");
        }
        if (records_seen != n_outputs) {
            throw std::runtime_error(
                "compute_linear_response: incomplete batched output records "
                "(saw " + std::to_string(records_seen) + " / "
                + std::to_string(n_outputs) + ")");
        }
        return out;
    }

    for (int dir = 0; dir < 3; ++dir) {
        // Skip zero-direction trajectories. out.rho_xyz is already zero, so an
        // untouched slice correctly means "this direction contributes nothing"
        // — saves ~1/3 of the GPU work when all dipoles align with one axis.
        if (dirs[dir]->norm() < 1e-12) {
            std::cout << "[abs " << kAxis[dir]
                      << "] skipped (zero dipole projection)\n";
            continue;
        }
        // Group the direction's Step/done lines under a labelled header, so a
        // 3-directions-per-disorder-node MC log stays readable (the trajectory
        // keeps its own "Step N (t=..)" + "done. (computation time)" lines).
        std::cout << "\n[abs " << kAxis[dir] << "] propagating\n";

        // ρ(0) = μ⁺_α · ρ_eq.
        const Eigen::MatrixXcd rho0 = (*dirs[dir]) * rho_eq;

        // enable_hdf5=false: the spectroscopy compute layer never opens files.
        // A fresh trajectory per direction is by design — the ctor is cheap
        // (sub-ms vs seconds of solve), and reusing one across directions would
        // need a reset() surface (active_flags, diverged_, filtering_history_,
        // …) that risks direction-to-direction state leakage. Don't.
        HEOMTrajectory traj(sys, /*enable_hdf5=*/false);
        traj.set_initial(rho0);
        declare_eg_block(traj, N);

        int records_seen = 0;
        traj.solve(
            [&, dir](int record_idx,
                     double t_fs,
                     const Cd* h_rho,
                     int N_local) {
                // Callback contract pins record_idx to [0, n_outputs). Hard
                // error on out-of-range so a miscount can't silently corrupt a
                // different direction's slab.
                if (record_idx < 0 || record_idx >= n_outputs) {
                    throw std::runtime_error(
                        "compute_linear_response: callback record_idx="
                        + std::to_string(record_idx)
                        + " out of [0, " + std::to_string(n_outputs) + ")");
                }
                if (N_local != N) {
                    throw std::runtime_error(
                        "compute_linear_response: callback N mismatch "
                        "(got " + std::to_string(N_local) + ", expected "
                        + std::to_string(N) + ")");
                }
                ++records_seen;
                out.time_fs[record_idx] = t_fs;

                // Direct flat-tensor memcpy at the right offset. No
                // Eigen::Map round-trip; this is a one-shot bulk copy of
                // an already-host-side N² block per record.
                const size_t rho_off =
                    (static_cast<size_t>(dir) * n_outputs + record_idx)
                    * static_cast<size_t>(N) * N;
                std::memcpy(out.rho_xyz.data() + rho_off,
                            h_rho,
                            sizeof(Cd) * static_cast<size_t>(N) * N);

                // Deliberately do NOT compute Tr(B·ρ) here. C_α(t) is derived:
                // callers compute it on demand from out.rho_xyz. Keeping it out
                // of the struct means post-processing changes (orientation
                // averaging, alternative B operators) have one place to live.
            });

        // Do not let half a trajectory slip into the spectrum: a diverged or
        // short direction leaves zero-padding past the failure point that a
        // downstream FFT would treat as physical decay. Fail loudly here.
        if (traj.diverged()) {
            throw std::runtime_error(
                "compute_linear_response: HEOM trajectory diverged during "
                "absorption propagation (direction="
                + std::to_string(dir) + "). Check L_cutoff, dt, K, or T.");
        }
        if (records_seen != n_outputs) {
            throw std::runtime_error(
                "compute_linear_response: incomplete output records from "
                "HEOMTrajectory (direction=" + std::to_string(dir)
                + ", saw " + std::to_string(records_seen)
                + " / " + std::to_string(n_outputs) + ")");
        }
    }

    // Intentionally no Ct_xyz / Ct_iso: they are derived quantities that
    // post-processing rebuilds from the saved ρ and dipole attrs on demand.
    return out;
}

}  // namespace heom::spectra
