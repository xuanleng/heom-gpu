#ifndef HEOM_SYSTEM_CONTEXT_HPP
#define HEOM_SYSTEM_CONTEXT_HPP

#include "heom_config.hpp"
#include "system_info.hpp"
#include "types.hpp"
#include "gpu_memory.hpp"
#include "core/precision.hpp"     // heom::real_t / heom::complex_t

#include <Eigen/Dense>
#include <cuComplex.h>
#include <vector>

/**
 * @brief Immutable system state shared across one or more HEOMTrajectory runs.
 *
 * Built ONCE from a Config. Owns:
 *   - Hamiltonian eigenspectrum (used by HDF5 metadata and exciton-basis ρ_0)
 *   - All GPU read-only buffers (Hamiltonian, neighbor maps, bath-correlation
 *     coefficients, bath coupling, decay coefficients)
 *
 * Designed for 2DES: one SystemContext drives N HEOMTrajectory instances
 * with different dipole-applied initial conditions, so the hierarchy +
 * eigenspectrum + GPU read-only buffers are computed/uploaded once.
 *
 * Host-side hierarchy data (indices, plus/minus maps) is only needed during
 * construction to populate the GPU buffers and is freed before the
 * constructor returns, except the O(D) first-tier table used for the
 * system-bath energy. The device copies are the single source of truth
 * during evolution.
 */
class SystemContext {
public:
    explicit SystemContext(const HEOMConfig::Config& cfg);

    SystemContext(const SystemContext&)            = delete;
    SystemContext& operator=(const SystemContext&) = delete;

    const HEOMConfig::Config& config() const { return cfg_; }
    const SystemInfo& system_info() const   { return sysinfo_; }

    int M() const       { return M_; }
    int N() const       { return N_; }
    int D() const       { return D_; }        // total ladder count Σ_n P_n
    int n_bath() const  { return n_bath_; }
    int L() const       { return L_; }        // occupation cutoff (= L_cutoff)
    // Nb the trajectory buffers are sized for: the largest manifold block this
    // config's task will ever store per ADO (heom::rho_capacity). Computed once
    // here so the budget check, the alloc plan and HEOMTrajectory::capacity_
    // are the same number.
    int rho_capacity() const { return rho_capacity_; }

    // One first-tier ADO: Σ_{n,k} α_{nk} = 1. Kept on the host after the
    // rest of the hierarchy tables are freed — O(D) ints, not O(M). The
    // system-bath energy sums √s_k Tr[n̂_n ρ̃_α] over α that belong to
    // bath n (see HEOMTrajectory::read_system_bath_energy).
    struct FirstTierAdo {
        int    alpha = 0;   // hierarchy index
        int    site  = 0;   // bath index 0…n_bath-1 (pigment or shared)
        int    k     = 0;   // pole within that bath
        double scale = 0.0; // s_k = √(|c_k||c̄_k|)
    };
    const std::vector<FirstTierAdo>& first_tier() const { return first_tier_; }

    // Host copy of build_d_coupling (n_bath × N); same layout as d_d_coupling_.
    // Cached so read_system_bath_energy does not rebuild every output step.
    const std::vector<int8_t>& d_coupling_host() const { return h_d_coupling_; }

    // Eigen decomposition of H, computed unconditionally at construction
    // (cheap even for LHC-II): consumed by HDF5 metadata and by the
    // HEOMTrajectory exciton-basis ρ_0 builder. Per manifold, not global:
    // columns [lo(m), hi(m)) are manifold m's eigenvectors, energy-ascending
    // within it, so column ground_slots(basis) + k is the k-th exciton.
    const Eigen::VectorXd&  eigenvalues()  const { return eigenvalues_; }
    const Eigen::MatrixXcd& eigenvectors() const { return eigenvectors_; }

    const heom::complex_t* d_hamiltonian()   const { return d_hamiltonian_.get(); }
    const int*             d_plus_map()      const { return d_plus_map_.get(); }
    const int*             d_minus_map()     const { return d_minus_map_.get(); }
    const int8_t*          d_level_indices() const { return d_level_indices_.get(); }
    // The Term 3/4/5 coefficients are stored in hier precision: the derivative
    // kernel reads them per (thread, n, k), so FP64 storage would mean a
    // per-read cvt on the 1/64-rate FP64 pipe. Rounded once at upload.
    const heom::complex_t* d_c_bath()        const { return d_c_bath_.get(); }
    // Right-action coefficients c̄_k (Term 5's ρ·Q side). Real poles: c̄ = c*;
    // conjugate pairs: the partner's c*. Filled by physics_utils::set_cbar.
    const heom::complex_t* d_cbar_bath()     const { return d_cbar_bath_.get(); }
    // No d_nu_bath(): ν_k is host-only (total_decay precompute); the GPU never
    // needs the bath frequencies, so no device mirror is allocated.
    // No d_abs_c_bath(): the Shi scale s_{nk} = √(|c||c̄|) is host-only (it
    // feeds only the Term 4/5 sqrt, precomputed per-ADO into coeff_plus/minus).
    const heom::real_t*    d_coeff_plus()    const { return d_coeff_plus_.get(); }
    const heom::real_t*    d_coeff_minus()   const { return d_coeff_minus_.get(); }
    const heom::real_t*    d_markov_residue() const { return d_markov_residue_.get(); }
    // FP64 complex: read once per thread (Term 2) and consumed by the
    // compute_if_factors exp() precompute, which wants the exact Σ α_{nk}ν_k.
    // Complex because ν of a vibrational pair is ν± = γ ± iω₁.
    const cuDoubleComplex* d_total_decay()   const { return d_total_decay_.get(); }
    const int8_t*          d_d_coupling()    const { return d_d_coupling_.get(); }
    // Ladder prefix sums [n_bath+1]; the kernel's Term 4/5 bath loop reads it
    // to find each bath's slot range. Host twin: cfg.physics.mode_off.
    const int*             d_mode_off()      const { return d_mode_off_.get(); }

private:
    const HEOMConfig::Config& cfg_;
    SystemInfo sysinfo_;

    int M_       = 0;
    int N_       = 0;
    int D_       = 0;
    int n_bath_  = 0;
    int L_       = 0;
    int rho_capacity_ = 0;

    Eigen::VectorXd  eigenvalues_;
    Eigen::MatrixXcd eigenvectors_;

    GPUMemory<heom::complex_t> d_hamiltonian_;
    GPUMemory<int>             d_plus_map_;
    GPUMemory<int>             d_minus_map_;
    GPUMemory<int8_t>          d_level_indices_;
    GPUMemory<heom::complex_t> d_c_bath_;       // [D] c_{nk}
    GPUMemory<heom::complex_t> d_cbar_bath_;    // [D] c̄_{nk} (right action)
    GPUMemory<heom::real_t>    d_coeff_plus_;   // [M*D] √[(α+1)s] (Term 4)
    GPUMemory<heom::real_t>    d_coeff_minus_;  // [M*D] √[α/s]    (Term 5)
    GPUMemory<heom::real_t>    d_markov_residue_; // [n_bath] per-site Markov term (Term 3)
    GPUMemory<cuDoubleComplex> d_total_decay_;  // [M] Σ α_{nk}ν_{nk}, complex
    GPUMemory<int8_t>          d_d_coupling_;
    GPUMemory<int>             d_mode_off_;     // [n_bath+1] ladder prefix sums

    // Host table of first-tier ADOs (see first_tier()). Built in
    // upload_gpu_readonly from the hierarchy that is then discarded.
    std::vector<FirstTierAdo> first_tier_;
    std::vector<int8_t>       h_d_coupling_;  // host twin of d_d_coupling_

    // Aggregate of host-side hierarchy data; lives only inside the constructor
    // body so it is destroyed once GPU upload completes.
    struct HostHierarchy {
        HierarchyTable   indices;
        std::vector<int>            plus_map;
        std::vector<int>            minus_map;
    };
    HostHierarchy generate_hierarchy_and_maps();
    void compute_eigenspectrum();
    void check_memory_budget();
    void upload_gpu_readonly(const HostHierarchy& hh);
};

#endif // HEOM_SYSTEM_CONTEXT_HPP
