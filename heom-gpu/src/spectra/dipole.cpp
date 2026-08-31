#include "spectra/dipole.hpp"

#include "core/hamiltonian_expansion.hpp"   // DoubleExcitedIndex (e↔f block layout)

#include <stdexcept>
#include <string>

namespace heom::spectra {

DipoleOperator DipoleOperator::from_pigment_dipoles(
    const std::vector<std::array<double, 3>>& unit_dipoles,
    const std::vector<double>& strengths,
    BasisType basis,
    int N)
{
    // Stage 1 absorption only handles WITH_GROUND. Reject EXCITED_ONLY at the
    // factory boundary so the caller sees it at their basis selection, not 200
    // lines deeper inside an all-zero spectrum. The "requires basis=with_ground"
    // wording is asserted by a test — keep it.
    if (basis == BasisType::EXCITED_ONLY) {
        throw std::invalid_argument(
            "DipoleOperator::from_pigment_dipoles: Stage 1 absorption "
            "requires basis=with_ground; got basis=excited_only "
            "(no ground→excited transition block exists in this basis).");
    }
    if (unit_dipoles.size() != strengths.size()) {
        throw std::invalid_argument(
            "DipoleOperator::from_pigment_dipoles: unit_dipoles.size() ("
            + std::to_string(unit_dipoles.size()) + ") != strengths.size() ("
            + std::to_string(strengths.size()) + ").");
    }

    // DOUBLE_EXCITED: build the raising operator μ⁺ on the {g, e, f} basis
    // (2DES needs the excited-state-absorption e→f manifold). μ⁺ holds only the
    // raising blocks (g→e column + e→f block); consumers take .adjoint() for μ⁻.
    // Placed after the size-match check (uses strengths[k-1]) but before the
    // WITH_GROUND-specific N==n+1 check (the f-manifold makes N larger).
    if (basis == BasisType::DOUBLE_EXCITED) {
        const int n = static_cast<int>(unit_dipoles.size());
        DoubleExcitedIndex didx(n);
        if (didx.total_dim() != N) {
            throw std::invalid_argument(
                "DipoleOperator::from_pigment_dipoles: DOUBLE_EXCITED requires "
                "1 + n + n(n-1)/2 == N (got n=" + std::to_string(n)
                + ", expected N=" + std::to_string(didx.total_dim())
                + ", actual N=" + std::to_string(N) + ").");
        }
        DipoleOperator op;
        op.mu_x = Eigen::MatrixXcd::Zero(N, N);
        op.mu_y = Eigen::MatrixXcd::Zero(N, N);
        op.mu_z = Eigen::MatrixXcd::Zero(N, N);
        // += guards the 3+-pigment case where |ij> reachable from both |i> and
        // |j> could touch the same element; harmless for 2 pigments.
        auto set = [&](Eigen::MatrixXcd& M, int r, int c, double v) {
            M(r, c) += std::complex<double>(v, 0.0);
        };
        // g→e raising block (column 0): |k> <- |g>, transition dipole of pigment k.
        for (int k = 1; k <= n; ++k) {
            const double s = strengths[k - 1];
            const auto& d = unit_dipoles[k - 1];
            set(op.mu_x, didx.single_index(k), 0, s * d[0]);
            set(op.mu_y, didx.single_index(k), 0, s * d[1]);
            set(op.mu_z, didx.single_index(k), 0, s * d[2]);
        }
        // e→f raising block: |ij> <- |i> excites pigment j; |ij> <- |j> excites pigment i.
        for (int i = 1; i <= n; ++i) {
            for (int j = i + 1; j <= n; ++j) {
                const int row = didx.double_index(i, j);
                const double si = strengths[i - 1], sj = strengths[j - 1];
                const auto& di = unit_dipoles[i - 1];
                const auto& dj = unit_dipoles[j - 1];
                set(op.mu_x, row, didx.single_index(i), sj * dj[0]);
                set(op.mu_y, row, didx.single_index(i), sj * dj[1]);
                set(op.mu_z, row, didx.single_index(i), sj * dj[2]);
                set(op.mu_x, row, didx.single_index(j), si * di[0]);
                set(op.mu_y, row, didx.single_index(j), si * di[1]);
                set(op.mu_z, row, didx.single_index(j), si * di[2]);
            }
        }
        return op;
    }
    // WITH_GROUND: N = n_states + 1. n_states comes from the dipoles vector;
    // a mismatch here means the caller built `DipoleOperator` against a
    // different system than the SystemContext it will eventually drive.
    if (static_cast<int>(unit_dipoles.size()) + 1 != N) {
        throw std::invalid_argument(
            "DipoleOperator::from_pigment_dipoles: WITH_GROUND requires "
            "unit_dipoles.size() + 1 == N (got "
            + std::to_string(unit_dipoles.size()) + " + 1 vs N="
            + std::to_string(N) + ").");
    }

    DipoleOperator op;
    op.mu_x = Eigen::MatrixXcd::Zero(N, N);
    op.mu_y = Eigen::MatrixXcd::Zero(N, N);
    op.mu_z = Eigen::MatrixXcd::Zero(N, N);

    // WITH_GROUND layout (matches expand_with_ground in
    // src/core/hamiltonian_expansion.{hpp,cpp}):
    //   - Hilbert index 0 == ground state
    //   - Hilbert index k == pigment k for k = 1..n_states
    //
    // μ⁺_α = Σ_k strengths[k-1] · unit_dipoles[k-1][α] · |k⟩⟨g|
    // i.e. only the first column (the "from-ground" column) is non-zero,
    // and only in the excited rows k = 1..n_states.
    const int n_states = static_cast<int>(unit_dipoles.size());
    for (int k = 1; k <= n_states; ++k) {
        const double s = strengths[k - 1];
        const auto& dvec = unit_dipoles[k - 1];
        op.mu_x(k, 0) = std::complex<double>(s * dvec[0], 0.0);
        op.mu_y(k, 0) = std::complex<double>(s * dvec[1], 0.0);
        op.mu_z(k, 0) = std::complex<double>(s * dvec[2], 0.0);
    }

    return op;
}

}  // namespace heom::spectra
