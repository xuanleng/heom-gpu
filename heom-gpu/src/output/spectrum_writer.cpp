#include "output/spectrum_writer.hpp"

#include "system_context.hpp"
#include "output/hdf5_writer.hpp"      // HDF5_CHECK_* macros
#include "output/hdf5_metadata.hpp"
#include "types.hpp"

#include <hdf5.h>

#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace heom::output {

namespace {

void write_dataset_complex_4d(hid_t parent, const char* name,
                              const std::complex<double>* data,
                              hsize_t d0, hsize_t d1, hsize_t d2, hsize_t d3)
{
    hsize_t dims[4] = {d0, d1, d2, d3};
    hid_t space, dataset;
    hid_t ctype = heom::hdf5::make_complex_compound_type();
    HDF5_CHECK_HID(space,   H5Screate_simple(4, dims, NULL));
    HDF5_CHECK_HID(dataset, H5Dcreate2(parent, name, ctype, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHECK_ERR(H5Dwrite(dataset, ctype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    HDF5_CHECK_ERR(H5Dclose(dataset));
    HDF5_CHECK_ERR(H5Sclose(space));
    HDF5_CHECK_ERR(H5Tclose(ctype));
}

void write_dataset_complex_nd(hid_t parent, const char* name,
                              const std::complex<double>* data,
                              const std::vector<hsize_t>& dims)
{
    hid_t space, dataset;
    hid_t ctype = heom::hdf5::make_complex_compound_type();
    HDF5_CHECK_HID(space,   H5Screate_simple(static_cast<int>(dims.size()),
                                             dims.data(), NULL));
    HDF5_CHECK_HID(dataset, H5Dcreate2(parent, name, ctype, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHECK_ERR(H5Dwrite(dataset, ctype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    HDF5_CHECK_ERR(H5Dclose(dataset));
    HDF5_CHECK_ERR(H5Sclose(space));
    HDF5_CHECK_ERR(H5Tclose(ctype));
}

std::string join_csv(const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += ','; s += v[i]; }
    return s;
}

void write_dipoles_to_parameters(hid_t params, const PhysicalParameters& phys)
{
    const int n_states = static_cast<int>(phys.dipoles.size());
    std::vector<double> flat(static_cast<size_t>(n_states) * 3);
    for (int i = 0; i < n_states; ++i) {
        flat[i * 3 + 0] = phys.dipoles[i][0];
        flat[i * 3 + 1] = phys.dipoles[i][1];
        flat[i * 3 + 2] = phys.dipoles[i][2];
    }
    heom::hdf5::write_dataset_2d(params, "dipoles",   flat.data(), n_states, 3);
    heom::hdf5::write_dataset_1d(params, "strengths", phys.strengths.data(), n_states);
}

// Exception path: best-effort status="aborted", then close. Never throws —
// the caller is already propagating the real error.
void abort_file(hid_t file) {
    try {
        heom::hdf5::write_run_status(file, heom::hdf5::RunStatus::Aborted);
    } catch (...) {}
    H5Fclose(file);
}

// rho_eq and the equilibration history, under /results.
void write_equilibrium(hid_t results, const HEOMConfig::Config& cfg,
                       const heom::spectra::FluorescenceResult& r)
{
    const int N = static_cast<int>(r.rho_eq.rows());
    std::vector<std::complex<double>> flat(static_cast<size_t>(N) * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) flat[static_cast<size_t>(i) * N + j] = r.rho_eq(i, j);
    write_dataset_complex_nd(results, "rho_eq", flat.data(),
                             {static_cast<hsize_t>(N), static_cast<hsize_t>(N)});

    const auto& h = r.equilibration;
    const int n = static_cast<int>(h.time_fs.size());
    hid_t g = heom::hdf5::create_group(results, "equilibration");
    heom::hdf5::write_dataset_1d(g, "time_fs",  h.time_fs.data(),  n);
    heom::hdf5::write_dataset_1d(g, "delta",    h.delta.data(),    n);
    heom::hdf5::write_dataset_1d(g, "ratio",    h.ratio.data(),    n);
    heom::hdf5::write_dataset_1d(g, "estimate", h.estimate.data(), n);
    heom::hdf5::write_scalar_attr(g, "t_eq_fs",           h.t_eq_fs);
    heom::hdf5::write_scalar_attr(g, "tolerance",         cfg.equilibration.tolerance);
    heom::hdf5::write_scalar_attr(g, "check_interval_fs", cfg.equilibration.check_interval_fs);
    heom::hdf5::write_scalar_attr(g, "t_max_fs",          cfg.equilibration.t_max_fs);
    HDF5_CHECK_ERR(H5Gclose(g));
}

// Absorption and fluorescence share everything but the task name and what
// the equilibrium adds to /results.
std::string write_linear(const HEOMConfig::Config& cfg,
                         const SystemContext& sys,
                         const heom::spectra::LinearResponseResult& out,
                         double total_seconds,
                         const heom::spectra::FluorescenceResult* fl)
{
    const char* task = fl ? "fluorescence" : "absorption";
    if (cfg.physics.dipoles.empty()) {
        throw std::invalid_argument(
            "write_spectrum: cfg.physics.dipoles is empty (was the task entry "
            "resolve_<task>_config() called?)");
    }
    validate_spectrum_result_shape(sys.N(), output_record_count(cfg.time), out);

    const std::string path = heom::hdf5::resolve_output_path(
        cfg.output.save_path, cfg.output.filename, cfg.output.use_timestamp);

    const hid_t file = heom::hdf5::create_output_file(path, cfg.output.use_timestamp);

    try {
        // ---- shared group layout (see hdf5_metadata.hpp) ----
        heom::hdf5::write_root_identity(file, cfg, task);
        heom::hdf5::write_run_status(file, heom::hdf5::RunStatus::Running);

        const auto filtering = spectrum_filtering_metadata(cfg);
        hid_t params = heom::hdf5::write_parameters_group(file, sys,
                                                          filtering.use_filtering,
                                                          filtering.filter_threshold);
        // /parameters extras specific to the linear-spectrum tasks
        heom::hdf5::write_string_attr(params, "spectrum_type", task);
        write_dipoles_to_parameters(params, cfg.physics);
        HDF5_CHECK_ERR(H5Gclose(params));

        heom::hdf5::write_derived_group   (file, sys);
        heom::hdf5::write_provenance_group(file, sys);

        // ---- /results: time[T] + rho_xyz[3,T,N,N] complex ----
        hid_t results = heom::hdf5::create_group(file, "/results");

        heom::hdf5::write_dataset_1d(results, "time",
                                     out.time_fs.data(),
                                     out.n_outputs);

        write_dataset_complex_4d(results, "rho_xyz", out.rho_xyz.data(),
                                 3, out.n_outputs, out.N, out.N);
        if (fl && fl->rho_eq.size() > 0) write_equilibrium(results, cfg, *fl);

        HDF5_CHECK_ERR(H5Gclose(results));

        // Total x/y/z wall-clock under the SAME root attribute the dynamics
        // writer uses, so plotting can compare runtime across tasks/params.
        heom::hdf5::write_scalar_attr(file, "simulation_time_seconds", total_seconds);
        heom::hdf5::write_run_status(file, heom::hdf5::RunStatus::Complete);
    } catch (...) {
        abort_file(file);
        throw;
    }

    HDF5_CHECK_ERR(H5Fclose(file));

    std::cout << "Spectrum file created: " << path << "\n";
    return path;
}

}  // anonymous namespace

std::string write_spectrum(const HEOMConfig::Config& cfg,
                           const SystemContext& sys,
                           const heom::spectra::LinearResponseResult& out,
                           double total_seconds)
{
    return write_linear(cfg, sys, out, total_seconds, nullptr);
}

std::string write_fluorescence(const HEOMConfig::Config& cfg,
                               const SystemContext& sys,
                               const heom::spectra::FluorescenceResult& r,
                               double total_seconds)
{
    return write_linear(cfg, sys, r.linear, total_seconds, &r);
}

std::string write_2des(const HEOMConfig::Config& cfg,
                       const SystemContext& sys,
                       const heom::spectra::TwoDESResult& out,
                       double total_seconds)
{
    if (!cfg.twodes.has_value()) {
        throw std::invalid_argument(
            "write_2des: cfg.twodes is empty (was resolve_2des_config "
            "called?)");
    }
    const auto& spec = *cfg.twodes;
    validate_2des_result_shape(spec, out);

    const std::string path = heom::hdf5::resolve_output_path(
        cfg.output.save_path, cfg.output.filename, cfg.output.use_timestamp);

    const hid_t file = heom::hdf5::create_output_file(path, cfg.output.use_timestamp);

    try {
        heom::hdf5::write_root_identity(file, cfg, "2des");
        heom::hdf5::write_run_status(file, heom::hdf5::RunStatus::Running);

        const auto filtering = spectrum_filtering_metadata(cfg);
        hid_t params = heom::hdf5::write_parameters_group(file, sys,
                                                          filtering.use_filtering,
                                                          filtering.filter_threshold);
        heom::hdf5::write_string_attr(params, "spectrum_type", "2des");
        // Signal/pathway ordering — the post-processor maps resp axes via these
        // (matches TwoDESResult.idx component/signal/pathway order).
        heom::hdf5::write_string_attr(params, "signals",  join_csv(spec.signals));
        heom::hdf5::write_string_attr(params, "pathways", join_csv(spec.pathways));
        heom::hdf5::write_string_attr(params, "orientation", spec.orientation);
        // Walk width actually used (1/7/21). Not in config_hash; recorded
        // because with filtering on the width moves results at filter-error
        // level. Prefer out.batch_width so auto resolves to a concrete value
        // (spec.batch may still be kTwoDESBatchAuto).
        heom::hdf5::write_scalar_attr(params, "batch_width", out.batch_width);
        heom::hdf5::write_string_attr(
            params, "batch_requested",
            spec.batch == HEOMConfig::kTwoDESBatchAuto
                ? "auto"
                : std::to_string(spec.batch));
        if (!out.batch_auto_reason.empty())
            heom::hdf5::write_string_attr(params, "batch_auto_reason",
                                          out.batch_auto_reason);
        write_dipoles_to_parameters(params, cfg.physics);
        HDF5_CHECK_ERR(H5Gclose(params));

        heom::hdf5::write_derived_group   (file, sys);
        heom::hdf5::write_provenance_group(file, sys);

        hid_t results = heom::hdf5::create_group(file, "/results");
        heom::hdf5::write_dataset_1d(results, "tau_fs", out.tau_fs.data(), out.n_tau);
        heom::hdf5::write_dataset_1d(results, "tw_fs",  out.tw_fs.data(),  out.n_tw);
        heom::hdf5::write_dataset_1d(results, "t_fs",   out.t_fs.data(),   out.n_t);

        // resp[n_component, n_signal, n_pathway, n_tau, n_tw, n_t].
        // Each pathway slice already carries its combination sign (SE/GSB = +,
        // ESA = − per S=φ1+φ2−φ3, baked in by PathSpec.sign in compute_2des), so
        // the total signal is a plain sum over the pathway axis — the ESA slice
        // is its signed contribution (−φ3), NOT the raw positive bracket.
        const std::vector<hsize_t> dims = {
            static_cast<hsize_t>(out.n_component),
            static_cast<hsize_t>(out.n_signal),
            static_cast<hsize_t>(out.n_pathway),
            static_cast<hsize_t>(out.n_tau),
            static_cast<hsize_t>(out.n_tw),
            static_cast<hsize_t>(out.n_t)};
        write_dataset_complex_nd(results, "resp", out.resp.data(), dims);

        HDF5_CHECK_ERR(H5Gclose(results));
        heom::hdf5::write_scalar_attr(file, "simulation_time_seconds", total_seconds);
        heom::hdf5::write_run_status(file, heom::hdf5::RunStatus::Complete);
    } catch (...) {
        abort_file(file);
        throw;
    }

    HDF5_CHECK_ERR(H5Fclose(file));
    std::cout << "2DES file created: " << path << "\n";
    return path;
}

}  // namespace heom::output
