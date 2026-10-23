#ifndef HAMILTONIAN_EXPANSION_HPP
#define HAMILTONIAN_EXPANSION_HPP

#include "types.hpp"
#include "core/manifold_layout.hpp"
#include <Eigen/Dense>

/**
 * @brief Compute Hilbert space dimension from basis type.
 *
 * This is the ground truth ManifoldLayout::from() checks N against — it
 * must stay independent of the layout, or the check would be circular.
 */
inline int compute_hilbert_dim(int n_states, BasisType basis) {
    switch (basis) {
        case BasisType::EXCITED_ONLY:
            return n_states;
        case BasisType::WITH_GROUND:
            return n_states + 1;
        case BasisType::DOUBLE_EXCITED:
            return 1 + n_states + n_states * (n_states - 1) / 2;
        default:
            throw std::runtime_error("Unknown basis type");
    }
}

/**
 * @brief Index manager for double-excited states
 * 
 * Maps double-excited states |ij⟩ (1 ≤ i < j ≤ n_states) to linear indices.
 * Uses lexicographic ordering:
 *   |1,2⟩ → 0, |1,3⟩ → 1, ..., |1,n⟩ → n-2, |2,3⟩ → n-1, ...
 * 
 * Where the g / e / f ranges start is NOT decided here: it comes from
 * ManifoldLayout (the single source), this class only adds the within-f
 * lexicographic offset.
 */
class DoubleExcitedIndex {
public:
    explicit DoubleExcitedIndex(int n_states)
        : n_(n_states)
        , lay_(heom::ManifoldLayout::from(
              BasisType::DOUBLE_EXCITED, n_states,
              compute_hilbert_dim(n_states, BasisType::DOUBLE_EXCITED)))
    {}
    
    // Ground state index
    int ground_index() const { return lay_.lo(0); }
    
    // Single-excited state |i⟩ → linear index (i: 1-based)
    int single_index(int i) const { return lay_.lo(1) + (i - 1); }
    
    // Double-excited state |ij⟩ → linear index (i,j: 1-based, i < j)
    int double_index(int i, int j) const {
        // Lexicographic ordering: offset = (i-1)*(2n-i)/2 + (j-i-1)
        const int offset_in_double = (i - 1) * (2 * n_ - i) / 2 + (j - i - 1);
        return lay_.lo(2) + offset_in_double;
    }
    
    // Dimension queries
    int total_dim() const { return lay_.N(); }
    int n_states() const { return n_; }
    int n_ground() const { return lay_.size(0); }
    int n_single() const { return lay_.size(1); }
    int n_double() const { return lay_.size(2); }
    
private:
    int n_;                        // Number of excited states
    heom::ManifoldLayout lay_;     // g/e/f bounds for DOUBLE_EXCITED
};

/**
 * @brief Expand Hamiltonian: excited_only (identity operation)
 */
Eigen::MatrixXcd expand_excited_only(const Eigen::MatrixXcd& H_excited);

/**
 * @brief Expand Hamiltonian: with_ground (add ground state at index 0)
 */
Eigen::MatrixXcd expand_with_ground(const Eigen::MatrixXcd& H_excited);

/**
 * @brief Expand Hamiltonian: double_excited (add ground + double-excited states)
 */
Eigen::MatrixXcd expand_with_double_excited(const Eigen::MatrixXcd& H_excited);

/**
 * @brief Unified Hamiltonian expansion interface
 * 
 * @param H_excited Input Hamiltonian (n_states × n_states, excited states only)
 * @param basis Basis type (excited_only/with_ground/double_excited)
 * @return Expanded Hamiltonian (n_sites × n_sites)
 */
Eigen::MatrixXcd expand_hamiltonian(const Eigen::MatrixXcd& H_excited, BasisType basis);

#endif // HAMILTONIAN_EXPANSION_HPP

