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
 * constructor returns; the device copies are the single source of truth
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
    int K() const       { return K_; }
    int n_bath() const  { return n_bath_; }
    int L() const       { return L_; }        // occupation cutoff (= L_cutoff)

    // Eigen decomposition of H_site, computed unconditionally at construction
    // (cheap even for LHC-II): consumed by HDF5 metadata and by the
    // HEOMTrajectory exciton-basis ρ_0 builder.
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
    // No d_nu_bath(): ν_k is host-only (total_decay precompute); the GPU never
    // needs the bath frequencies, so no device mirror is allocated.
    // No d_abs_c_bath(): |c_{nk}| is host-only now (it fed only the Term 4/5
    // sqrt, which is precomputed per-ADO into coeff_plus/coeff_minus below).
    const heom::real_t*    d_coeff_plus()    const { return d_coeff_plus_.get(); }
    const heom::real_t*    d_coeff_minus()   const { return d_coeff_minus_.get(); }
    const heom::real_t*    d_markov_residue() const { return d_markov_residue_.get(); }
    // FP64: read once per thread (Term 2) and consumed by the compute_if_factors
    // exp() precompute, which wants the exact Σ α_{nk}ν_k.
    const double*          d_total_decay()   const { return d_total_decay_.get(); }
    const int8_t*          d_d_coupling()    const { return d_d_coupling_.get(); }

private:
    const HEOMConfig::Config& cfg_;
    SystemInfo sysinfo_;

    int M_       = 0;
    int N_       = 0;
    int K_       = 0;
    int n_bath_  = 0;
    int L_       = 0;

    Eigen::VectorXd  eigenvalues_;
    Eigen::MatrixXcd eigenvectors_;

    GPUMemory<heom::complex_t> d_hamiltonian_;
    GPUMemory<int>             d_plus_map_;
    GPUMemory<int>             d_minus_map_;
    GPUMemory<int8_t>          d_level_indices_;
    GPUMemory<heom::complex_t> d_c_bath_;       // [n_bath*(K+1)] c_{nk}
    GPUMemory<heom::real_t>    d_coeff_plus_;   // [M*n_bath*(K+1)] √[(α+1)|c|] (Term 4)
    GPUMemory<heom::real_t>    d_coeff_minus_;  // [M*n_bath*(K+1)] √[α/|c|]    (Term 5)
    GPUMemory<heom::real_t>    d_markov_residue_; // [n_bath] per-site Markov term (Term 3)
    GPUMemory<double>          d_total_decay_;
    GPUMemory<int8_t>          d_d_coupling_;

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
