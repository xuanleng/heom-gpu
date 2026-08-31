#ifndef HEOM_SPECTRA_LINEAR_RESPONSE_HPP
#define HEOM_SPECTRA_LINEAR_RESPONSE_HPP

#include "spectra/dipole.hpp"
#include "types.hpp"

#include <Eigen/Dense>
#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

class SystemContext;   // forward decl: real header pulled in by .cpp only

namespace heom::spectra {

/**
 * @brief Inputs to `compute_linear_response()`.
 *
 * Stage 1: a single field — the dipole operator. Stage 2 (emission / CD /
 * 2DES) will add fields for the preparation-side selector and the sampling
 * operator; new fields are non-breaking under the API stability rule. Removing
 * or renaming a field is breaking.
 *
 * **Don't add Stage-2 hooks now**: the public-API stability
 * promise locks this struct. The semantics of the preparation-side selector
 * (left mul vs. right mul vs. commutator?) and the sampling operator (μ⁻ vs.
 * something else?) cannot be pinned until emission / CD / 2DES algorithms
 * are actually written. Reserving the fields now would commit us to a
 * guess. Stage 2 adds them with concrete semantics; Stage 1 does not pay
 * an API stability tax for fields it does not consume.
 */
struct LinearResponseRequest {
    DipoleOperator dipole;
};

inline void validate_linear_response_inputs(
    BasisType basis,
    const DipoleOperator& dipole,
    int N
) {
    if (basis != BasisType::WITH_GROUND) {
        throw std::invalid_argument(
            "compute_linear_response requires basis=with_ground");
    }
    if (N <= 0) {
        throw std::invalid_argument("compute_linear_response requires N > 0");
    }
    if (dipole.mu_x.rows() != N || dipole.mu_x.cols() != N ||
        dipole.mu_y.rows() != N || dipole.mu_y.cols() != N ||
        dipole.mu_z.rows() != N || dipole.mu_z.cols() != N) {
        throw std::invalid_argument(
            "compute_linear_response dipole matrices must be N x N");
    }
    if (dipole.mu_x.norm() < 1e-12 &&
        dipole.mu_y.norm() < 1e-12 &&
        dipole.mu_z.norm() < 1e-12) {
        throw std::invalid_argument(
            "compute_linear_response requires at least one nonzero dipole direction");
    }
}

/**
 * @brief Output of `compute_linear_response()` — three-direction ρ_α(t) only.
 *
 *   - `rho_xyz` is the **only raw fact** stored: `[3 × n_outputs × N × N]`
 *     row-major, std::complex<double>. Direction outermost, then output
 *     record (== HEOM step / output_interval, plus record 0 = t=0), then
 *     row, then col.
 *   - `time_fs` is `[n_outputs]` simulation time per record.
 *   - We deliberately do NOT store `Ct_α(t) = Tr(B · ρ_α(t))` or the
 *     isotropic average; both are trivially derived from rho_xyz on demand
 *     (C++ unit tests use `(B * out.rho(α, step)).trace()`; Python
 *     post-processing computes the isotropic mean from rho_xyz). Storing
 *     derived quantities in the same struct as the primitive that defines
 *     them invites the "components are new but the average is stale" class
 *     of bugs.
 *   - Layout is fixed row-major to match the GPU kernel's `i*N + j` indexing
 *     and the existing HDF5 writer convention. The Eigen::Map view helpers
 *     below pin the row-major interpretation explicitly so consumers cannot
 *     accidentally use the default column-major Eigen::Matrix and silently
 *     transpose the data.
 */
struct LinearResponseResult {
    int N         = 0;     // Hilbert dimension
    int n_outputs = 0;     // == output_record_count(sys.config().time)

    // Time per output record, in fs. Length n_outputs.
    std::vector<double> time_fs;

    // Flat density-matrix tensor [3 × n_outputs × N × N], row-major.
    // Direction order: 0 = x, 1 = y, 2 = z (matches DipoleOperator member
    // order). std::complex<double>, length 3 * n_outputs * N * N.
    std::vector<std::complex<double>> rho_xyz;

    using RowMajorMatrixXcd = Eigen::Matrix<std::complex<double>,
                                            Eigen::Dynamic, Eigen::Dynamic,
                                            Eigen::RowMajor>;

    /// Read-only view onto ρ_α(t_record) without copying.
    Eigen::Map<const RowMajorMatrixXcd> rho(int dir, int record) const {
        const size_t off = (static_cast<size_t>(dir) * n_outputs + record)
                           * static_cast<size_t>(N) * N;
        return Eigen::Map<const RowMajorMatrixXcd>(
            rho_xyz.data() + off, N, N);
    }

    /// Mutable view (used by `compute_linear_response` to write each record).
    Eigen::Map<RowMajorMatrixXcd> rho_mut(int dir, int record) {
        const size_t off = (static_cast<size_t>(dir) * n_outputs + record)
                           * static_cast<size_t>(N) * N;
        return Eigen::Map<RowMajorMatrixXcd>(
            rho_xyz.data() + off, N, N);
    }
};

/**
 * @brief Drive HEOM dynamics from ρ(0) = μ⁺_α · ρ_eq for α ∈ {x,y,z}.
 *
 * Pure host-side: this function takes no device
 * pointers, opens no files, and is the public API surface that
 * Python bindings / ML fitting drivers will consume directly.
 *
 * Algorithm:
 *   - For each direction α (skip if `req.dipole.mu_α.norm() < 1e-12`):
 *       1. ρ(0) = μ⁺_α · ρ_eq   (ρ_eq ≈ |g⟩⟨g|).
 *       2. Build a HEOMTrajectory with HDF5 disabled (the spectroscopy
 *          compute layer never opens files) and seed it
 *          with ρ(0) via `traj.set_initial(ρ(0))`.
 *       3. `traj.solve(callback)` — the callback memcpy's ρ_0 from the host
 *          mirror into `out.rho_xyz` at the right offset (see the
 *          callback contract), and records `out.time_fs[record] = t_fs`.
 *   - After each direction, throw if
 *     `traj.diverged()` or if `records_seen != n_outputs`. This avoids
 *     leaking zero-initialised tail records into a "spectrum" the caller
 *     would then FFT and treat as truth.
 *
 * `n_outputs` comes from the free function
 * `output_record_count(sys.config().time)` — NOT from constructing a
 * throwaway `HEOMTrajectory` just to read its shape. A trajectory ctor
 * allocates several GPU buffers; we will not pay that cost just to read
 * one int.
 *
 * @throws std::runtime_error on divergence or incomplete record stream.
 */
LinearResponseResult compute_linear_response(
    const SystemContext& sys,
    const LinearResponseRequest& req);

}  // namespace heom::spectra

#endif  // HEOM_SPECTRA_LINEAR_RESPONSE_HPP
