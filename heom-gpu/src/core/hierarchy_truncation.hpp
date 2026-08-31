#ifndef HIERARCHY_TRUNCATION_HPP
#define HIERARCHY_TRUNCATION_HPP

#include "types.hpp"
#include <vector>
#include <iostream>

/**
 * @brief Hierarchy Truncation Manager (simple occupation cutoff).
 *
 * A single knob:
 *
 *   L_cutoff — occupation cap:
 *       Σα := Σ_{n, k} α_{n,k}  ≤  L_cutoff
 *
 * The generator enumerates every hierarchy index whose total occupation is
 * within L_cutoff.
 */
class HierarchyTruncation {
public:
    /**
     * @brief Constructor.
     * @param params    Physical parameters (provides n_states, n_bath_modes_total).
     * @param L_cutoff  Occupation cap (see class doc).
     * @param log       Progress / statistics sink. Defaults to &std::cout so
     *                  existing callers are byte-identical. Pass nullptr to
     *                  silence ALL informational output (batch parameter
     *                  sweeps); [BUG] consistency errors still go to std::cerr
     *                  unconditionally — a corrupted hierarchy is never hidden.
     */
    HierarchyTruncation(const PhysicalParameters& params,
                        int L_cutoff,
                        std::ostream* log = &std::cout);

    /**
     * @brief Generate all hierarchy indices satisfying Σα ≤ L_cutoff,
     *        sorted lexicographically.
     */
    HierarchyTable generate_hierarchy_indices();

    /**
     * @brief Print by-level statistics about generated indices.
     */
    void print_statistics(const HierarchyTable& indices) const;

    /**
     * @brief Build (plus_map, minus_map) for the GPU kernel.
     */
    std::pair<std::vector<int>, std::vector<int>>
    build_neighbor_maps(const HierarchyTable& indices) const;

private:
    const PhysicalParameters& phys_params_;
    int L_cutoff_;   // Occupation cap (Σα ≤ L_cutoff)
    std::ostream* log_;  // Injected info sink; nullptr => silent. See ctor doc.

    int index_width() const {
        return phys_params_.n_states * (phys_params_.n_bath_modes_total + 1);
    }

    /**
     * @brief Recursive generator. `current` is a scratch row of index_width()
     *        bytes reused across the whole walk; running state:
     *   bath_pole_sum — Σ over α visited so far (≤ L_cutoff_).
     */
    void generate_recursive(
        std::vector<int8_t>& current,
        int position,
        int bath_pole_sum,
        HierarchyTable& result);
};

#endif // HIERARCHY_TRUNCATION_HPP
