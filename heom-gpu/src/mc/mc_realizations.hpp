#ifndef HEOM_MC_MC_REALIZATIONS_HPP
#define HEOM_MC_MC_REALIZATIONS_HPP

#include <string>
#include <vector>

namespace heom::mc {

// Mean = bare-sum / count (iid MC);
// Quadrature = weighted sum Σ_r w_r·O_r (sparse_gh), weights absolute (Σ=1),
// no count normalisation. The loader reads it from the `estimator` attribute
// and the driver picks the accumulate + write branch accordingly.
enum class Estimator { Mean, Quadrature };

// Significant digits of each σ inside disorder_identity. ON-DISK CONTRACT:
// every shard ever written groups by the exact identity string, so changing
// this orphans them all. See load_realizations for why 6 is also sufficient.
constexpr int kSigmaIdentityDigits = 6;

/**
 * @brief In-memory image of a disorder_realizations.h5 node file.
 *
 * The solver is oblivious to how the nodes were sampled — it only sees the
 * weighted node stream (ξ_r, w_r, group). `delta_eps` is the per-site energy
 * offset ξ_r in cm⁻¹, row-major [n_node × n_site].
 */
struct RealizationSet {
    int n_node = 0;
    int n_site = 0;
    std::vector<double> delta_eps;   // [n_node*n_site] row-major, cm⁻¹
    std::vector<double> weight;      // [n_node] (mean mode: all 1/n_node)
    std::vector<int>    group;       // [n_node]
    Estimator estimator = Estimator::Mean;
    std::string generator;           // e.g. "iid_gauss@v1"
    std::string disorder_identity;   // "gen|seed=..|sigma=.." — binds partials

    // View of row r: the n_site offsets for realization r.
    std::vector<double> row(int r) const;
};

/**
 * @brief Load and validate a disorder node file (HDF5 C API).
 *
 * Throws std::runtime_error if the file is missing datasets/attrs, dimensions
 * are inconsistent, or the estimator is neither "mean" nor "quadrature".
 */
RealizationSet load_realizations(const std::string& path);

} // namespace heom::mc

#endif // HEOM_MC_MC_REALIZATIONS_HPP
