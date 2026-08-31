#ifndef HEOM_SPECTRA_ABSORPTION_HPP
#define HEOM_SPECTRA_ABSORPTION_HPP

#include "spectra/dipole.hpp"
#include "spectra/linear_response.hpp"

class SystemContext;   // forward decl: real header pulled in by .cpp only

namespace heom::spectra {

/**
 * @brief Linear absorption spectrum forward driver.
 *
 * Wrapper around `compute_linear_response()` with one algorithm:
 * (ρ(0) = μ⁺_α · ρ_eq, sample with μ⁻ in post-processing) by simply
 * passing the dipole through. No prep-side / sample-op branching here:
 * Stage 1 has exactly one sensible algorithm, and dispatch logic for
 * future spectra (emission, CD, 2DES) lives in their own
 * `compute_*` functions, not in compute_absorption().
 *
 * Does not take paths or YAML. `run_absorption` owns that.
 */
LinearResponseResult compute_absorption(
    const SystemContext& sys,
    const DipoleOperator& mu);

}  // namespace heom::spectra

#endif  // HEOM_SPECTRA_ABSORPTION_HPP
