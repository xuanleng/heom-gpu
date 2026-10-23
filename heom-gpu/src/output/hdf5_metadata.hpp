#ifndef HEOM_OUTPUT_HDF5_METADATA_HPP
#define HEOM_OUTPUT_HDF5_METADATA_HPP

#include "heom_config.hpp"
#include "system_info.hpp"

#include <Eigen/Dense>
#include <complex>
#include <hdf5.h>
#include <cstdint>
#include <string>

class SystemContext;   // forward decl: include from .cpp

/**
 * @file hdf5_metadata.hpp
 *
 * Group-based HDF5 layout used by both writers (dynamics + absorption).
 * The file is partitioned into four sibling groups, each with a single
 * responsibility:
 *
 *   /                identity-only attrs (task, config_hash, wallclock)
 *   /parameters/     every knob the user could turn (physics, time grid,
 *                    truncation, sink, basis, initial state, Hamiltonian,
 *                    dipoles/strengths, config_yaml). Parameter-sweep
 *                    consumers only need to read this group.
 *   /derived/        cheap-to-recompute artefacts shipped for convenience
 *                    (eigenspectrum, bath-correlation coefficients, M_total).
 *   /results/        the actual bulk simulation output (time + density
 *                    matrix or ρ_xyz).
 *   /provenance/     hardware / OS / config-path strings used only for
 *                    debugging, never for science.
 *
 * Complex datasets use HDF5 native compound { double r; double i; }, which
 * h5py reads back as `np.complex128` without any client-side splitting.
 */

namespace heom::hdf5 {

// ---------------------------------------------------------------------------
// Compound complex type
//
// h5py recognises a 2-field compound of (double, double) with names
// "r"/"i" as np.complex128 on read, so consumers do not need to know the
// dataset is internally compound.
// ---------------------------------------------------------------------------

/// Build the canonical complex128 compound dtype. Caller owns the
/// returned hid_t and must H5Tclose() it.
hid_t make_complex_compound_type();

// ---------------------------------------------------------------------------
// Generic attribute / dataset helpers. Throw std::runtime_error via
// HDF5_CHECK_* on any HDF5 API failure. All helpers take the parent hid_t
// (file or group) so the same helper writes into / or /parameters / etc.
// ---------------------------------------------------------------------------

void write_scalar_attr (hid_t parent, const char* name, int    value);
void write_scalar_attr (hid_t parent, const char* name, double value);
void write_string_attr (hid_t parent, const char* name, const std::string& value);
void write_string_dataset(hid_t parent, const char* name, const std::string& value);
void write_dataset_1d  (hid_t parent, const char* name, const double* data, int n);
void write_dataset_int_1d (hid_t parent, const char* name, const int* data, int n);
void write_dataset_int8_1d(hid_t parent, const char* name, const int8_t* data, int n);
void write_dataset_2d  (hid_t parent, const char* name, const double* data,
                        int rows, int cols);

void write_dataset_complex_1d(hid_t parent, const char* name,
                              const std::complex<double>* data, int n);
void write_dataset_complex_2d(hid_t parent, const char* name,
                              const std::complex<double>* data,
                              int rows, int cols);

/// Group factory. Throws std::runtime_error if the group already exists or
/// HDF5 refuses to create it.
hid_t create_group(hid_t parent, const char* name);

// ---------------------------------------------------------------------------
// Group writers (one per top-level group). Each owns its slice of metadata.
// They are called by both the dynamics writer (HDF5Writer) and the
// absorption writer (spectrum_writer) so the two file shapes agree
// byte-for-byte on shared content.
// ---------------------------------------------------------------------------

/// Write root-level identity attrs:
///   task                    string  ("dynamics" | "absorption")
///   config_hash             string  (cfg.config_hash: task-scoped physics
///                                    fingerprint, see heom_config.hpp — NOT
///                                    a hash of the raw config_yaml text)
/// simulation_time_seconds is written separately by the writer's close()
/// path because it is only known after the run finishes.
void write_root_identity(hid_t file,
                         const HEOMConfig::Config& cfg,
                         const char* task_name);

/// Root attribute `status`. A writer sets "running" right after creating the
/// file and its final value LAST (after an H5Fflush of everything else):
/// "complete" | "diverged" | "aborted" (an exception unwound the writer).
/// "running" left behind = the process was killed. Absent = the file predates
/// the attribute. Reader side: heom_tools.common.heom_h5.require_complete.
/// Replaces an existing value.
enum class RunStatus { Running, Complete, Diverged, Aborted };
void write_run_status(hid_t file, RunStatus status);

/// Create and populate the /parameters group.
///
/// Writes every knob the user could turn (physics, bath, time grid,
/// HEOM truncation, sink, basis), the Hamiltonian (N×N float64), and a
/// full YAML snapshot in `config_yaml`. The returned group hid_t is
/// open so callers can stack task-specific extras on top:
///   - dynamics writer adds initial_state attrs + initial_density_matrix
///   - absorption writer adds spectrum_type + dipoles + strengths
/// CALLER MUST H5Gclose() the returned handle.
hid_t write_parameters_group(hid_t file,
                             const SystemContext& sys,
                             bool use_filtering,
                             double filter_threshold);

/// Create and populate /derived. Eigenspectrum is unconditional (cheap
/// even for LHC-II); bath-correlation coefficients are too small to skip.
/// Stored:
///   eigenvalues_relative_cm  [N]      float64   (absolute = relative+w0_cm)
///   eigenvectors             [N,N]    complex128  (per manifold g/e/f in basis
///                                      order, ascending within each manifold)
///   bath_c_cm2               [D]      complex128  (slot order, site n owns
///   bath_cbar_cm2            [D]      complex128   [mode_off[n], mode_off[n+1]))
///   bath_nu_cm               [D]      complex128  (BO pairs: γ ± iω₁)
///   mode_off                 [nb+1]   int32
///   mode_kind                [D]      int8   0 Drude / 1 Padé-Matsubara / 2 BO+ / 3 BO−
/// Attrs: M_total
void write_derived_group(hid_t file, const SystemContext& sys);

/// Create and populate /provenance. Pure debugging info — parameter
/// sweep consumers should not touch this group.
///   attrs: software_name, software_version, cpu_model, gpu_model, os,
///          cuda_version, compiler, config_path
void write_provenance_group(hid_t file, const SystemContext& sys);

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Recursive `mkdir -p` for the directory part of `path`. No-op if the
/// directory already exists. Throws std::runtime_error on permission errors.
void ensure_directory_exists(const std::string& path);

/// Resolve `dir + filename` into a full HDF5 file path. Appends a
/// `_YYYYMMDD_HHMMSS` timestamp before `.h5` when `use_timestamp` is true,
/// then runs `ensure_directory_exists()` on the result.
std::string resolve_output_path(const std::string& dir,
                                const std::string& filename,
                                bool use_timestamp);

/// Create the result file at `path` (from resolve_output_path with the same
/// `use_timestamp`). A timestamped name is the solver's own choice and must
/// never replace another run's file: it is created exclusively and an
/// existing file is an error. A fixed name (`use_timestamp: false`) is the
/// user's explicit target: an existing file is replaced, with a note on
/// stdout. Throws std::runtime_error; never returns a negative id.
hid_t create_output_file(const std::string& path, bool use_timestamp);

}  // namespace heom::hdf5

#endif // HEOM_OUTPUT_HDF5_METADATA_HPP
