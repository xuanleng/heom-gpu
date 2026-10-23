#include "core/manifold_layout.hpp"
#include "core/hamiltonian_expansion.hpp"   // compute_hilbert_dim, BasisType

#include <stdexcept>
#include <string>

namespace heom {

void ManifoldBlock::validate(int N, const char* who) const {
    const std::string w = who;
    if (n_row <= 0 || n_col <= 0) {
        throw std::invalid_argument(
            w + ": manifold block must be non-empty; use (0,N,0,N) for the "
            "whole matrix");
    }
    if (row_off < 0 || col_off < 0 ||
        row_off + n_row > N || col_off + n_col > N) {
        throw std::invalid_argument(w + ": manifold block out of range");
    }
    const bool identical = (row_off == col_off && n_row == n_col);
    const bool disjoint  = (row_off + n_row <= col_off) ||
                           (col_off + n_col <= row_off);
    if (!identical && !disjoint) {
        throw std::invalid_argument(
            w + ": manifold row/col ranges must be identical or disjoint");
    }
}

ManifoldLayout ManifoldLayout::from(BasisType basis, int n_states, int N) {
    const int want = compute_hilbert_dim(n_states, basis);   // throws on unknown basis
    if (N != want)
        throw std::invalid_argument(
            "ManifoldLayout::from: N=" + std::to_string(N) + " does not match "
            "n_states=" + std::to_string(n_states) + " for this basis (want "
            + std::to_string(want) + ") — wrong SystemContext?");

    ManifoldLayout lay;
    switch (basis) {
        case BasisType::EXCITED_ONLY:    // e = [0, N); no g, no f
            lay.bound = {0, 0, N, N};
            break;
        case BasisType::WITH_GROUND:     // g = [0,1), e = [1, N); no f
            lay.bound = {0, 1, N, N};
            break;
        case BasisType::DOUBLE_EXCITED:  // g = [0,1), e = [1, 1+n), f = [1+n, N)
            lay.bound = {0, 1, 1 + n_states, N};
            break;
    }
    return lay;
}

ManifoldBlock ManifoldLayout::block(int row_level, int col_level) const {
    for (int lv : {row_level, col_level}) {
        if (lv < 0 || lv >= 3)
            throw std::invalid_argument("ManifoldLayout::block: level out of range");
        if (empty(lv))
            throw std::invalid_argument(
                "ManifoldLayout::block: level " + std::to_string(lv)
                + " is empty in this basis");
    }
    return {lo(row_level), size(row_level), lo(col_level), size(col_level)};
}

}  // namespace heom
