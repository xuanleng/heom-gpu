#ifndef HEOM_OUTPUT_SPECTRUM_WRITER_HPP
#define HEOM_OUTPUT_SPECTRUM_WRITER_HPP

#include "heom_config.hpp"
#include "spectra/linear_response.hpp"
#include "spectra/two_dim.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

class SystemContext;   // forward decl: full header pulled in by .cpp

/**
 * @file spectrum_writer.hpp
 *
 * Spectroscopy HDF5 writer. Shares the same group layout as the dynamics
 * writer (see src/output/hdf5_metadata.hpp):
 *
 *   /                  attrs: task="absorption", config_hash
 *   /parameters/       knobs + Hamiltonian + basis + dipoles[n_states,3]
 *                      + strengths[n_states] + spectrum_type + config_yaml
 *   /derived/          eigenspectrum + bath-correlation coefficients + M_total
 *   /results/          time[n_outputs] (fs)
 *                      rho_xyz[3, n_outputs, N, N]  complex compound
 *   /provenance/       hardware / OS / config_path
 *
 * Single-fact rule: we deliberately DO NOT store `Ct_xyz_*` or the
 * isotropic average. Both are trivially derivable from `rho_xyz` plus the
 * `dipoles`/`strengths` datasets (`las-post` in heom-tools does
 * this). Storing them here would risk "components updated, derived product
 * stale" bugs.
 */
namespace heom::output {

struct SpectrumFilteringMetadata {
    bool use_filtering = false;
    double filter_threshold = 0.0;
};

inline SpectrumFilteringMetadata spectrum_filtering_metadata(
    const HEOMConfig::Config& cfg
) {
    return {cfg.filter.enabled, cfg.filter.tolerance};
}

inline void validate_spectrum_result_shape(
    int expected_N,
    int expected_n_outputs,
    const heom::spectra::LinearResponseResult& out
) {
    if (out.N != expected_N) {
        throw std::invalid_argument("write_spectrum: LinearResponseResult.N mismatch");
    }
    if (out.n_outputs != expected_n_outputs) {
        throw std::invalid_argument("write_spectrum: LinearResponseResult.n_outputs mismatch");
    }
    if (static_cast<int>(out.time_fs.size()) != out.n_outputs) {
        throw std::invalid_argument("write_spectrum: time_fs length mismatch");
    }
    const size_t expected_rho =
        static_cast<size_t>(3) * static_cast<size_t>(out.n_outputs)
        * static_cast<size_t>(out.N) * static_cast<size_t>(out.N);
    if (out.rho_xyz.size() != expected_rho) {
        throw std::invalid_argument("write_spectrum: rho_xyz length mismatch");
    }
}

/// Write `out` (and the surrounding system/bath/hierarchy/eigenspectrum
/// metadata) to an HDF5 file. The output path is built from `cfg.output`
/// using the same resolve-and-mkdir-p logic as HDF5Writer (timestamp
/// suffix + intermediate dirs honoured).
///
/// `cfg.physics.dipoles` is REQUIRED (non-empty) — the dipoles/strengths
/// attributes are sourced from there. Throws std::invalid_argument
/// otherwise (defence-in-depth; resolve_absorption_config already gates
/// on this earlier).
///
/// `total_seconds` is the wall-clock time of the whole x/y/z computation,
/// written to the file root as the `simulation_time_seconds` attribute —
/// the same key the dynamics writer uses — so plotting tools can read one
/// consistent attribute across tasks to compare runtime vs parameters.
///
/// Returns the resolved output path (with .h5 extension).
std::string write_spectrum(const HEOMConfig::Config& cfg,
                           const SystemContext& sys,
                           const heom::spectra::LinearResponseResult& out,
                           double total_seconds);

/// Write a raw 2DES result (no FFT — that's Python's job).
/// Layout under /results:
///   resp[n_component, n_signal, n_pathway, n_tau, n_tw, n_t]  complex compound
///   tau_fs[n_tau], tw_fs[n_tw], t_fs[n_t]                     (fs)
/// /parameters carries spectrum_type="2des", dipoles/strengths, and the
/// signal/pathway ordering as attributes so the post-processor needs no guess.
/// `sys` is the double_excited context (metadata superset). Returns the path.
std::string write_2des(const HEOMConfig::Config& cfg,
                       const SystemContext& sys,
                       const heom::spectra::TwoDESResult& out,
                       double total_seconds);

}  // namespace heom::output

#endif  // HEOM_OUTPUT_SPECTRUM_WRITER_HPP
