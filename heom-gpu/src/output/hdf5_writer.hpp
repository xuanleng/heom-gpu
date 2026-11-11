#ifndef HDF5_WRITER_HPP
#define HDF5_WRITER_HPP

#include "types.hpp"
#include "system_info.hpp"
#include "output/hdf5_metadata.hpp"
#include <Eigen/Dense>
#include <hdf5.h>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

class SystemContext;   // forward decl: include from .cpp

namespace heom::hdf5 {

inline void validate_timestep_bounds(
    int N,
    int expected_N,
    int current_step,
    int n_records
) {
    if (N != expected_N) {
        throw std::invalid_argument("HDF5Writer::write_timestep: N mismatch");
    }
    if (current_step < 0 || current_step >= n_records) {
        throw std::out_of_range("HDF5Writer::write_timestep: output record index out of range");
    }
}

}  // namespace heom::hdf5

#define HDF5_CHECK_ERR(call) \
    do { \
        if ((call) < 0) { \
            throw std::runtime_error( \
                std::string("HDF5 error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__)); \
        } \
    } while(0)

#define HDF5_CHECK_HID(var, call) \
    do { \
        var = (call); \
        if (var < 0) { \
            throw std::runtime_error( \
                std::string("HDF5 error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__)); \
        } \
    } while(0)

/**
 * @brief HDF5 dynamics-task writer (group layout: see hdf5_metadata.hpp).
 *
 * Writes:
 *   /                  attrs: task="dynamics", config_hash, simulation_time_seconds,
 *                      n_records_written, status (last; see RunStatus)
 *   /parameters/       knobs + Hamiltonian + initial_state + initial_density_matrix
 *                      + config_yaml
 *   /derived/          eigenspectrum + bath-correlation coefficients + M_total
 *                      + final_active_ados (after close)
 *   /results/          time[T], density_matrix[T,N,N] (complex compound),
 *                      system_bath_energy[T, n_bath] (complex compound)
 *   /provenance/       hardware / OS / config_path strings
 *
 * write_timestep() incrementally fills /results.
 */
class HDF5Writer {
public:
    HDF5Writer(bool use_timestamp);
    ~HDF5Writer();

    HDF5Writer(const HDF5Writer&) = delete;
    HDF5Writer& operator=(const HDF5Writer&) = delete;

    /**
     * @brief Open HDF5 file and pre-allocate /results datasets.
     *
     * @param sys                 immutable system context (pulls config,
     *                            sysinfo, eigendata, hierarchy from here)
     * @param initial_rho_used    actual ρ_0 used for this trajectory
     *                            (stored at /parameters/initial_density_matrix);
     *                            for 2DES this is the dipole-applied state,
     *                            not what cfg.initial would have produced.
     * @param use_filtering       whether ADO filtering is on
     * @param filter_threshold    filter tolerance value
     */
    void open(const SystemContext& sys,
              const Eigen::MatrixXcd& initial_rho_used,
              bool use_filtering,
              double filter_threshold);

    /// rho: row-major N×N. esb: [n_bath] first-tier system-bath energy for
    /// the same record (required; nullptr throws). Breaking vs 2.2.0, which
    /// had no esb argument. Host type only — the writer knows nothing about
    /// how the trajectory stores its state.
    void write_timestep(double time_fs, const std::complex<double>* rho, int N,
                        const std::complex<double>* esb, int n_bath);

    /// status: Complete, or Diverged when the run stopped at a divergence
    /// check (fewer than the allocated records written). Destroying an open
    /// writer instead marks the file Aborted.
    void close(double simulation_time, heom::hdf5::RunStatus status,
               const std::vector<FilteringRecord>* filtering_history = nullptr);

    const std::string& get_output_path() const { return output_path_; }

private:
    bool use_timestamp_;
    std::string output_path_;

    hid_t file_;
    hid_t density_matrix_dataset_;  // /results/density_matrix (compound complex)
    hid_t esb_dataset_;              // /results/system_bath_energy
    hid_t time_dataset_;             // /results/time
    hid_t complex_type_;             // owned compound complex type
    int current_step_;
    int N_;
    int n_bath_;
    int n_records_;

    // Per-timestep staging buffer (N×N complex doubles, layout-compatible
    // with the compound complex HDF5 dtype).
    std::vector<std::complex<double>> tmp_rho_;
    std::vector<std::complex<double>> tmp_esb_;
};

#endif // HDF5_WRITER_HPP
