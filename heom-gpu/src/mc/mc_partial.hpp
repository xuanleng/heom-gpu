#ifndef HEOM_MC_MC_PARTIAL_HPP
#define HEOM_MC_MC_PARTIAL_HPP

#include "mc/mc_realizations.hpp"   // Estimator

#include <complex>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace heom::mc {

/**
 * @brief Scatter-add accumulator, shared by mean and quadrature.
 *
 * A single scaled scatter-add serves both estimators; the scale
 * and what gets persisted are the only differences:
 *   Mean       add(O, 1.0, r):     sum_obs += O,      sum_obs_sq += |O|²
 *              ⟨O⟩ = (Σ_i sum_obs)/(Σ_i count), computed by the merge script.
 *              (scale=1.0 is an exact IEEE no-op, so mean stays bit-identical.)
 *   Quadrature add(O, w_r, r):     sum_obs += w_r·O   (sum_obs_sq UNUSED)
 *              ⟨O⟩ = Σ_i sum_obs (weights absolute, Σ=1, NO count divide).
 * `count` is book-kept in both modes but only normalises the mean; quadrature
 * uses it for progress/provenance only. The solver never needs the global N:
 * mean defers the ÷N to merge; quadrature weights are already absolute.
 */
struct Accumulator {
    Estimator estimator = Estimator::Mean;
    std::vector<size_t> obs_shape;                 // e.g. {3, n_out, N, N}
    long long count = 0;
    std::vector<std::complex<double>> sum_obs;     // Σ_r scale·O_r
    std::vector<double> sum_obs_sq;                // Σ_r |O_r|²  (mean only)
    std::vector<long long> r_done;                 // folded realization indices

    size_t obs_size() const {   // product of obs_shape
        return std::accumulate(obs_shape.begin(), obs_shape.end(),
                               static_cast<size_t>(1), std::multiplies<size_t>());
    }

    // Allocate zeroed buffers for an observable of the given shape. sum_obs_sq
    // is allocated only for Mean (quadrature has no √N SE semantics — skipped).
    void init(const std::vector<size_t>& shape, Estimator est) {
        estimator = est;
        obs_shape = shape;
        const size_t n = obs_size();
        count = 0;
        sum_obs.assign(n, std::complex<double>(0.0, 0.0));
        sum_obs_sq.assign(est == Estimator::Mean ? n : 0, 0.0);
        r_done.clear();
    }

    // Fold one realization: count++, sum_obs += scale·O, |O|² for mean, record r.
    void add(const std::vector<std::complex<double>>& O, double scale, int r) {
        if (O.size() != sum_obs.size()) {
            throw std::runtime_error("Accumulator::add: observable size " +
                std::to_string(O.size()) + " != accumulator size " +
                std::to_string(sum_obs.size()) + " (call init() first)");
        }
        ++count;
        for (size_t k = 0; k < O.size(); ++k) sum_obs[k] += scale * O[k];
        if (estimator == Estimator::Mean) {
            for (size_t k = 0; k < O.size(); ++k) sum_obs_sq[k] += std::norm(O[k]);
        }
        r_done.push_back(r);
    }

    // Mean only: ⟨O⟩ = sum_obs / count (element-wise). Written into /results by
    // the driver so a rank file is a valid (under-sampled) single-shot spectrum.
    // Quadrature writes sum_obs directly (a partial weighted sum placeholder).
    // Requires count > 0.
    std::vector<std::complex<double>> mean() const {
        if (count <= 0) {
            throw std::runtime_error("Accumulator::mean: count == 0 (no "
                                     "realizations folded — nothing to average)");
        }
        std::vector<std::complex<double>> m(sum_obs.size());
        const double inv = 1.0 / static_cast<double>(count);
        for (size_t k = 0; k < sum_obs.size(); ++k) m[k] = sum_obs[k] * inv;
        return m;
    }
};

/**
 * @brief Append the MC provenance block to a single-shot-format spectrum file.
 *
 * Adds a `/mc` group carrying what the merge step needs, branched by estimator:
 *   both       count (int64), estimator (attr), r_done / config_fingerprint /
 *              disorder_identity attrs
 *   Mean       sum_obs_sq (float64, obs_shape) — merge builds the SEM from it
 *   Quadrature sum_obs (complex128, obs_shape) — merge sums these directly;
 *              sum_obs_sq is NOT written (no √N SE semantics)
 * Presence of `/mc` marks the file as an MC shard; single-shot files have none.
 */
void append_mc_provenance(const std::string& path, const Accumulator& acc,
                          const std::string& config_fingerprint,
                          const std::string& disorder_identity);

} // namespace heom::mc

#endif // HEOM_MC_MC_PARTIAL_HPP
