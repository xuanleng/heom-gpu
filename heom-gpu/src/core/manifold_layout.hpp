#ifndef HEOM_MANIFOLD_LAYOUT_HPP
#define HEOM_MANIFOLD_LAYOUT_HPP

// Where the g / e / f manifolds sit inside one Hilbert basis, and the
// (row manifold, col manifold) blocks of rho the kernels launch over.
//
// ONE definition. Consumers: the 2DES pathway walk (spectra/two_dim),
// absorption's e x g block (spectra/linear_response), HEOMTrajectory's
// support, HEOMContext, the ADO norm scan,
// DoubleExcitedIndex (hamiltonian_expansion) and the d_coupling builder
// (physics_utils). None of them spells "1 + n_states" any more.
//
// Header-only declarations, CUDA- and Eigen-free: .cu kernels include it for
// ManifoldBlock, host code for ManifoldLayout. Bodies that throw live in
// manifold_layout.cpp.

#include <array>
#include <cstddef>

enum class BasisType;   // types.hpp; opaque here so this header stays light

namespace heom {

// A (row manifold, col manifold) block of rho in (offset, count) form. The
// manifolds are CONTIGUOUS index ranges by construction of the Hilbert basis
// (compute_hilbert_dim), so no mask / index table is ever needed.
// THE BLOCK IS THE STORAGE SHAPE: every ADO stores exactly the elems() =
// n_row*n_col elements inside it, row-major, at (i-row_off)*n_col +
// (j-col_off). Nothing outside the block exists in device memory.
struct ManifoldBlock {
    int row_off = 0, n_row = 0, col_off = 0, n_col = 0;

    bool operator==(const ManifoldBlock& o) const {
        return row_off == o.row_off && n_row == o.n_row
            && col_off == o.col_off && n_col == o.n_col;
    }
    bool operator!=(const ManifoldBlock& o) const { return !(*this == o); }

    // "Whole matrix", spelled out. There is NO default meaning that; a
    // brace-initialised block is empty and validate() rejects it, so every
    // construction site has to state its intent.
    static ManifoldBlock full(int N) { return {0, N, 0, N}; }

    // Number of elements in the block (Nb = n_row * n_col). The one formula
    // every "how big is rho" question reduces to.
    int elems() const { return n_row * n_col; }

    // The ONLY copy of the block rule. Throws std::invalid_argument unless:
    //   non-empty (n_row, n_col > 0), inside [0, N) on both axes, and the row
    //   and col ranges are either IDENTICAL or DISJOINT — a partial overlap has
    //   no physical meaning (manifolds partition the basis) and would break the
    //   Term 1 dispatch in heom_derivatives.cu, which relies on the k-sum
    //   splitting into at most two contiguous runs.
    // `who` prefixes the message (e.g. "HEOMContext").
    void validate(int N, const char* who) const;
};

// g = [bound[0], bound[1]), e = [bound[1], bound[2]), f = [bound[2], bound[3]).
// EXCITED_ONLY has neither g nor f: bound = {0, 0, N, N}. WITH_GROUND has no
// f: bound = {0, 1, N, N}. Whether an empty manifold is an error is the
// CALLER's question (2DES cannot start from |g><g| without g) — block() on an
// empty level throws, from() does not.
struct ManifoldLayout {
    // Throws if N != compute_hilbert_dim(n_states, basis). That check is the
    // whole point of taking N: the one mistake this API exists to prevent is
    // building the layout for one basis out of the OTHER context's N
    // (compute_2des keeps sys_single and sys_double alive simultaneously).
    static ManifoldLayout from(BasisType basis, int n_states, int N);

    std::array<int, 4> bound{};

    int  lo(int level) const { return bound[static_cast<size_t>(level)]; }
    int  hi(int level) const { return bound[static_cast<size_t>(level) + 1]; }
    int  size(int level) const { return hi(level) - lo(level); }
    bool empty(int level) const { return size(level) == 0; }
    int  N() const { return bound[3]; }

    // Levels: 0 = g, 1 = e, 2 = f. Throws on a level outside [0, 3) or on an
    // empty one — the resulting block would fail validate() anyway, but
    // failing here names the level instead of the shape.
    ManifoldBlock block(int row_level, int col_level) const;
};

}  // namespace heom

#endif  // HEOM_MANIFOLD_LAYOUT_HPP
