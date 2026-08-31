#include "hamiltonian_expansion.hpp"
#include <stdexcept>
#include <complex>

using Complex = std::complex<double>;

// ============================================================================
// Excited-only expansion (identity operation)
// ============================================================================
Eigen::MatrixXcd expand_excited_only(const Eigen::MatrixXcd& H_excited) {
    return H_excited;
}

// ============================================================================
// With-ground expansion (add ground state at index 0)
// ============================================================================
Eigen::MatrixXcd expand_with_ground(const Eigen::MatrixXcd& H_excited) {
    const int n = H_excited.rows();
    const int n_full = n + 1;
    
    // Initialize zero matrix
    Eigen::MatrixXcd H_full = Eigen::MatrixXcd::Zero(n_full, n_full);
    
    // Copy excited-state block (starting from index 1)
    H_full.block(1, 1, n, n) = H_excited;
    
    return H_full;
}

// ============================================================================
// Helper: Copy single-excited block
// ============================================================================
static void copy_single_excited_block(
    Eigen::MatrixXcd& H_full,
    const Eigen::MatrixXcd& H_excited,
    const DoubleExcitedIndex& idx
) {
    const int n = idx.n_states();
    
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            H_full(idx.single_index(i), idx.single_index(j)) = H_excited(i-1, j-1);
        }
    }
}

// ============================================================================
// Helper: Compute double-excited matrix element ⟨ij|H|i'j'⟩
// ============================================================================
/**
 * Physical formula:
 *   ⟨ij|H|i'j'⟩ = δ_ii'δ_jj'(ε_i + ε_j)     [diagonal]
 *                + δ_ii'(1-δ_jj')J_jj'      [i same, j different]
 *                + δ_ij'(1-δ_ji')J_ji'      [cross: i=j']
 *                + δ_ji'(1-δ_ij')J_ij'      [cross: j=i']
 *                + δ_jj'(1-δ_ii')J_ii'      [j same, i different]
 */
static Complex compute_double_coupling(
    const Eigen::MatrixXcd& H,
    int i, int j,
    int ip, int jp
) {
    auto V = [&H](int a, int b) -> Complex { return H(a - 1, b - 1); };
    auto delta = [](int a, int b) -> double { return (a == b) ? 1.0 : 0.0; };

    return delta(i, ip) * delta(j, jp) * (V(i, i) + V(j, j))
         + delta(i, ip) * (1.0 - delta(j, jp)) * V(j, jp)
         + delta(i, jp) * (1.0 - delta(j, ip)) * V(j, ip)
         + delta(j, ip) * (1.0 - delta(i, jp)) * V(i, jp)
         + delta(j, jp) * (1.0 - delta(i, ip)) * V(i, ip);
}

// ============================================================================
// Helper: Fill double-excited block (one row)
// ============================================================================
static void fill_double_row(
    Eigen::MatrixXcd& H_full,
    int row,
    const Eigen::MatrixXcd& H_excited,
    int i, int j,
    const DoubleExcitedIndex& idx
) {
    const int n = idx.n_states();
    
    for (int ip = 1; ip <= n; ++ip) {
        for (int jp = ip + 1; jp <= n; ++jp) {
            int col = idx.double_index(ip, jp);
            H_full(row, col) = compute_double_coupling(H_excited, i, j, ip, jp);
        }
    }
}

// ============================================================================
// Helper: Fill entire double-excited block
// ============================================================================
static void fill_double_excited_block(
    Eigen::MatrixXcd& H_full,
    const Eigen::MatrixXcd& H_excited,
    const DoubleExcitedIndex& idx
) {
    const int n = idx.n_states();
    
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            int row = idx.double_index(i, j);
            fill_double_row(H_full, row, H_excited, i, j, idx);
        }
    }
}

// ============================================================================
// Double-excited expansion (add ground + double-excited states)
// ============================================================================
Eigen::MatrixXcd expand_with_double_excited(const Eigen::MatrixXcd& H_excited) {
    const int n = H_excited.rows();
    DoubleExcitedIndex idx(n);
    const int n_full = idx.total_dim();
    
    // Initialize zero matrix (ground state block = 0 automatically)
    Eigen::MatrixXcd H_full = Eigen::MatrixXcd::Zero(n_full, n_full);
    
    // Block 1: Single-excited subspace
    copy_single_excited_block(H_full, H_excited, idx);
    
    // Block 2: Double-excited subspace
    fill_double_excited_block(H_full, H_excited, idx);
    
    return H_full;
}

// ============================================================================
// Unified expansion interface
// ============================================================================
Eigen::MatrixXcd expand_hamiltonian(const Eigen::MatrixXcd& H_excited, BasisType basis) {
    switch (basis) {
        case BasisType::EXCITED_ONLY:
            return expand_excited_only(H_excited);
        
        case BasisType::WITH_GROUND:
            return expand_with_ground(H_excited);
        
        case BasisType::DOUBLE_EXCITED:
            return expand_with_double_excited(H_excited);
        
        default:
            throw std::runtime_error("Unknown basis type in expand_hamiltonian");
    }
}

