#include "mc/mc_disorder.hpp"

#include "types.hpp"

#include <stdexcept>
#include <string>

namespace heom::mc {

void apply_disorder(HEOMConfig::Config& cfg, const std::vector<double>& delta_eps_cm) {
    const int n = cfg.physics.n_states;
    if (static_cast<int>(delta_eps_cm.size()) != n) {
        throw std::invalid_argument(
            "apply_disorder: delta_eps_cm.size()=" +
            std::to_string(delta_eps_cm.size()) + " != n_states=" +
            std::to_string(n) + " (offsets are per single-excitation site)");
    }
    // Inject on the site-basis diagonal (a.u.), BEFORE re-expansion.
    for (int i = 0; i < n; ++i) {
        cfg.physics.hamiltonian_site(i, i) += delta_eps_cm[i] * CM_TO_AU;
    }
    // Re-expand into whatever basis the task already chose. The double-excited
    // path sums single-site energies for |ij⟩, so δ_i + δ_j falls out for free.
    HEOMConfig::apply_basis(cfg, cfg.physics.basis);
}

} // namespace heom::mc
