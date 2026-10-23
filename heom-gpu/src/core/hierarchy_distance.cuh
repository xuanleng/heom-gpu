#ifndef HEOM_CORE_HIERARCHY_DISTANCE_CUH
#define HEOM_CORE_HIERARCHY_DISTANCE_CUH

#include <cstddef>
#include "core/precision.hpp"

namespace heom {

// Upper bound on the reduction grid; partials holds 2 doubles per block.
inline constexpr int kDistMaxBlocks = 1024;

// Σ|a − b|² and Σ|a|² over the n stored elements of two hierarchies, in double,
// where a = rho − rho_comp and b = ref − ref_comp (comp = Kahan carry, may be
// nullptr: the true value of a carried element is ρ − c). Block k writes
// partials[2k] (distance²) and partials[2k+1] (norm²); the grid depends on n
// only, so the host's fixed-order sum of the partials is run-to-run exact.
// Returns the number of blocks launched (≤ kDistMaxBlocks).
int launch_hierarchy_distance(const complex_t* rho, const complex_t* rho_comp,
                              const complex_t* ref, const complex_t* ref_comp,
                              size_t n, double* partials);

}  // namespace heom

#endif  // HEOM_CORE_HIERARCHY_DISTANCE_CUH
