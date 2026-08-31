#ifndef HEOM_MC_MC_DISORDER_HPP
#define HEOM_MC_MC_DISORDER_HPP

#include "heom_config.hpp"

#include <vector>

namespace heom::mc {

/**
 * @brief Injection point: freeze one static-disorder realization.
 *
 * Adds the per-site energy offsets `delta_eps_cm` (cm⁻¹) onto the site-basis
 * Hamiltonian diagonal, then re-expands into the config's current basis via
 * HEOMConfig::apply_basis(). This is the ONLY place static disorder touches the
 * physics — the HEOM kernel is oblivious.
 *
 * Semantics:
 *   - `delta_eps_cm.size()` MUST equal `cfg.physics.n_states` (single-excitation
 *     site offsets); throws std::invalid_argument otherwise.
 *   - The offset is applied to `hamiltonian_site` (n_states × n_states, a.u.)
 *     BEFORE re-expansion, so the double-excited diagonal |ij⟩ automatically
 *     picks up δ_i + δ_j (apply_basis/expand_with_double_excited sums the two
 *     single-excitation energies). No special-casing of the basis is needed.
 *   - All-zero offsets ⇒ bit-identical to the pre-call Hamiltonian (the += 0.0
 *     path plus a deterministic re-expansion).
 *
 * Idempotency: NOT idempotent — each call adds to the current diagonal. The MC
 * driver applies it exactly once on a fresh `Config` clone per realization.
 */
void apply_disorder(HEOMConfig::Config& cfg, const std::vector<double>& delta_eps_cm);

} // namespace heom::mc

#endif // HEOM_MC_MC_DISORDER_HPP
