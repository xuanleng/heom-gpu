#ifndef HEOM_SPECTRA_LINEAR_RECORD_HPP
#define HEOM_SPECTRA_LINEAR_RECORD_HPP

// Internal (not under include/): the part of a linear-response run that does
// not depend on how the initial state is put into the trajectory. Absorption
// seeds μ⁺|g⟩⟨g|, fluorescence right-multiplies an equilibrated (e,e)
// hierarchy; both then group directions and record ρ_α(t) through here.

#include "spectra/linear_response.hpp"
#include "core/manifold_layout.hpp"

#include <string>
#include <vector>

class HEOMTrajectory;

namespace heom::spectra {

/// μ⁺_α for α = 0,1,2 (x,y,z).
const Eigen::MatrixXcd& mu_plus(const DipoleOperator& mu, int dir);

/// Directions with a nonzero μ⁺_α; prints "[<tag> x] skipped ..." for the rest
/// (silent when tag is null).
std::vector<int> active_directions(const DipoleOperator& mu, const char* tag);

/// batch ⇒ one group holding every direction; otherwise one group each.
std::vector<std::vector<int>> direction_groups(const std::vector<int>& dirs, bool batch);

/// Prints the group header; returns its label ("[abs x]" / "[abs batch B=3]").
std::string announce_group(const char* tag, const std::vector<int>& group);

/// Zero-filled rho_xyz / time_fs sized for sys's time grid.
LinearResponseResult empty_linear_result(const SystemContext& sys);

/// The (e,g) block of the WITH_GROUND basis: where μ⁺ρ_g and ρ_ee μ⁺ live.
heom::ManifoldBlock eg_block(const SystemContext& sys);

/// run_output_loop on a seeded traj of width group.size(); slot s lands in
/// direction group[s]'s slab of out. Throws on divergence or a short stream,
/// so zero padding never reaches an FFT as fake decay.
void record_group(HEOMTrajectory& traj, const std::vector<int>& group,
                  LinearResponseResult& out, const std::string& label);

}  // namespace heom::spectra

#endif  // HEOM_SPECTRA_LINEAR_RECORD_HPP
