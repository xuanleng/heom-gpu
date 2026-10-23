#ifndef HEOM_OUTPUT_SPECTRUM_WRITER_HPP
#define HEOM_OUTPUT_SPECTRUM_WRITER_HPP

#include "heom_config.hpp"
#include "spectra/fluorescence.hpp"
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
 *   /                  attrs: task="absorption" | "fluorescence", config_hash
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

/// Same contract for 2DES: every axis vector matches its n_*, the signal and
/// pathway axes match what the config asked for, and resp holds exactly the
/// full [3][n_signal][n_pathway][n_tau][n_tw][n_t] block. H5Dwrite would
/// otherwise read past a short buffer and write garbage without complaint.
inline void validate_2des_result_shape(
    const HEOMConfig::TwoDESConfig& spec,
    const heom::spectra::TwoDESResult& out
) {
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::invalid_argument(std::string("write_2des: ") + what);
    };
    need(out.n_component == 3, "n_component != 3");
    need(out.n_signal  == static_cast<int>(spec.signals.size()),  "n_signal != signals.size()");
    need(out.n_pathway == static_cast<int>(spec.pathways.size()), "n_pathway != pathways.size()");
    need(out.n_tau > 0 && out.n_tw > 0 && out.n_t > 0, "empty tau/tw/t axis");
    need(static_cast<int>(out.tau_fs.size()) == out.n_tau, "tau_fs length != n_tau");
    need(static_cast<int>(out.tw_fs.size())  == out.n_tw,  "tw_fs length != n_tw");
    need(static_cast<int>(out.t_fs.size())   == out.n_t,   "t_fs length != n_t");
    const size_t expected = static_cast<size_t>(out.n_component) * out.n_signal
                          * out.n_pathway * out.n_tau * out.n_tw * out.n_t;
    need(out.resp.size() == expected, "resp length != n_component*n_signal*n_pathway*n_tau*n_tw*n_t");
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
/// It and status="complete" are written after /results; an exception leaves
/// status="aborted" (both writers).
///
/// Returns the resolved output path (with .h5 extension).
std::string write_spectrum(const HEOMConfig::Config& cfg,
                           const SystemContext& sys,
                           const heom::spectra::LinearResponseResult& out,
                           double total_seconds);

/// Fluorescence: the write_spectrum layout (task / spectrum_type
/// "fluorescence") plus, when r.rho_eq is non-empty (not an MC ensemble file):
///   /results/rho_eq[N, N]                                  complex compound
///   /results/equilibration/{time_fs, delta, ratio, estimate}
///       attrs t_eq_fs, tolerance, check_interval_fs, t_max_fs
std::string write_fluorescence(const HEOMConfig::Config& cfg,
                               const SystemContext& sys,
                               const heom::spectra::FluorescenceResult& r,
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
