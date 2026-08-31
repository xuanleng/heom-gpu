#ifndef HEOM_SPECTRA_DIPOLE_HPP
#define HEOM_SPECTRA_DIPOLE_HPP

#include "types.hpp"   // BasisType (CUDA-free header)

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace heom::spectra {

/**
 * @brief Three-direction electric-dipole raising operators μ⁺_α (α ∈ {x,y,z}).
 *
 *   - Each `mu_{x,y,z}` is the raising operator mu+_alpha on the full
 *     Hilbert basis (N x N, N = SystemContext::N()).
 *   - Stored as host-side Eigen::MatrixXcd. No device pointers.
 *   - The lowering operator μ⁻_α = (μ⁺_α)† is **not** stored — derived
 *     quantities never live in the same struct as the primitives that
 *     define them.
 *     Consumers that need μ⁻ compute `mu_*.adjoint()` on the fly.
 *
 * Used by:
 *   - `compute_linear_response()` — initial state ρ(0) = μ⁺_α · ρ_eq.
 *   - Future emission / CD / 2DES drivers — same μ⁺_α applied to different
 *     equilibrium / pump states.
 */
struct DipoleOperator {
    Eigen::MatrixXcd mu_x;   // μ⁺_x, Hilbert dimension N×N
    Eigen::MatrixXcd mu_y;   // μ⁺_y
    Eigen::MatrixXcd mu_z;   // μ⁺_z

    /**
     * @brief Build μ⁺ from per-pigment unit dipole vectors and strengths.
     *
     * Supports `BasisType::WITH_GROUND` (linear absorption: g→e block) and
     * `BasisType::DOUBLE_EXCITED` (2DES: g→e column + e→f raising block).
     * `EXCITED_ONLY` throws `std::invalid_argument`:
     *   - `EXCITED_ONLY`: no ground state in the Hilbert basis, so there is
     *     no g→e transition block to fill — early failure beats silently
     *     returning zero spectra and burning two days of debugging.
     *   - `DOUBLE_EXCITED`: fills the g→e column (|k⟩⟨g|) plus the e→f raising
     *     block (|ij⟩ from |i⟩/|j⟩, exciting the other pigment), indexed via
     *     `DoubleExcitedIndex`. Only μ⁺ (raising) blocks are stored.
     *
     * Layout for WITH_GROUND (ground at Hilbert index 0, pigment k (1-based)
     * at Hilbert index k):
     *   `mu_α[k][0] = strengths[k-1] * unit_dipoles[k-1][α]`
     * for k = 1..n_states. All other entries are zero.
     *
     * @param unit_dipoles  Per-pigment unit dipole vectors. Size must equal
     *                      `n_states = N - 1`. Each row is a 3-vector
     *                      (x, y, z components, no normalisation enforced —
     *                      pass already-normalised vectors and let
     *                      `strengths` carry the magnitude).
     * @param strengths     Per-pigment dipole magnitudes (Debye). Same length
     *                      as `unit_dipoles`.
     * @param basis         `BasisType::WITH_GROUND` or `DOUBLE_EXCITED`.
     * @param N             Hilbert dimension (== `SystemContext::N()` ==
     *                      `compute_hilbert_dim(n_states, basis)`).
     *
     * @throws std::invalid_argument
     *   - basis == EXCITED_ONLY (no g→e transition block exists)
     *   - `unit_dipoles.size() != strengths.size()`
     *   - WITH_GROUND: `unit_dipoles.size() + 1 != N`
     *   - DOUBLE_EXCITED: `1 + n + n(n-1)/2 != N`
     */
    static DipoleOperator from_pigment_dipoles(
        const std::vector<std::array<double, 3>>& unit_dipoles,
        const std::vector<double>& strengths,
        BasisType basis,
        int N);
};

}  // namespace heom::spectra

#endif  // HEOM_SPECTRA_DIPOLE_HPP
