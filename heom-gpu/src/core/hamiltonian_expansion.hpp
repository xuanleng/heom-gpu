#ifndef HAMILTONIAN_EXPANSION_HPP
#define HAMILTONIAN_EXPANSION_HPP

#include "types.hpp"
#include <Eigen/Dense>

/**
 * @brief Index manager for double-excited states
 * 
 * Maps double-excited states |ij⟩ (1 ≤ i < j ≤ n_states) to linear indices.
 * Uses lexicographic ordering:
 *   |1,2⟩ → 0, |1,3⟩ → 1, ..., |1,n⟩ → n-2, |2,3⟩ → n-1, ...
 * 
 * Hilbert space structure:
 *   Index 0: ground state |0⟩
 *   Index 1 to n_states: single-excited |i⟩ (i=1 to n_states)
 *   Index n_states+1 onwards: double-excited |ij⟩
 */
class DoubleExcitedIndex {
public:
    explicit DoubleExcitedIndex(int n_states) 
        : n_(n_states)
        , n_ground_(1)
        , n_single_(n_states)
        , n_double_(n_states * (n_states - 1) / 2)
        , n_total_(1 + n_states + n_states * (n_states - 1) / 2)
        , offset_ground_(0)
        , offset_single_(1)
        , offset_double_(1 + n_states)
    {}
    
    // Ground state index
    int ground_index() const { return 0; }
    
    // Single-excited state |i⟩ → linear index (i: 1-based)
    int single_index(int i) const {
        return offset_single_ + (i - 1);
    }
    
    // Double-excited state |ij⟩ → linear index (i,j: 1-based, i < j)
    int double_index(int i, int j) const {
        // Lexicographic ordering: offset = (i-1)*(2n-i)/2 + (j-i-1)
        int offset_in_double = (i - 1) * (2 * n_ - i) / 2 + (j - i - 1);
        return offset_double_ + offset_in_double;
    }
    
    // Dimension queries
    int total_dim() const { return n_total_; }
    int n_states() const { return n_; }
    int n_ground() const { return n_ground_; }
    int n_single() const { return n_single_; }
    int n_double() const { return n_double_; }
    
private:
    int n_;              // Number of excited states
    int n_ground_;       // = 1
    int n_single_;       // = n
    int n_double_;       // = n(n-1)/2
    int n_total_;        // = 1 + n + n(n-1)/2
    
    int offset_ground_;  // = 0
    int offset_single_;  // = 1
    int offset_double_;  // = 1 + n
};

/**
 * @brief Compute Hilbert space dimension from basis type
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

