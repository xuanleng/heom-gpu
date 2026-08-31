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

}  // anonymous namespace

std::string write_spectrum(const HEOMConfig::Config& cfg,
                           const SystemContext& sys,
                           const heom::spectra::LinearResponseResult& out,
                           double total_seconds)
{
    if (cfg.physics.dipoles.empty()) {
        throw std::invalid_argument(
            "write_spectrum: cfg.physics.dipoles is empty (was the absorption "
            "task entry resolve_absorption_config() called?)");
    }
    validate_spectrum_result_shape(sys.N(), output_record_count(cfg.time), out);

    const std::string path = heom::hdf5::resolve_output_path(
        cfg.output.save_path, cfg.output.filename, cfg.output.use_timestamp);

    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("Cannot create HDF5 file: " + path);
    }

    try {
        // ---- shared group layout (see hdf5_metadata.hpp) ----
        heom::hdf5::write_root_identity(file, cfg, "absorption");
        // Total x/y/z wall-clock under the SAME root attribute the dynamics
        // writer uses, so plotting can compare runtime across tasks/params.
        heom::hdf5::write_scalar_attr(file, "simulation_time_seconds", total_seconds);

        const auto filtering = spectrum_filtering_metadata(cfg);
        hid_t params = heom::hdf5::write_parameters_group(file, sys,
                                                          filtering.use_filtering,
                                                          filtering.filter_threshold);
        // /parameters extras specific to the absorption task
        heom::hdf5::write_string_attr(params, "spectrum_type", "absorption");
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

        HDF5_CHECK_ERR(H5Gclose(results));

        H5Fflush(file, H5F_SCOPE_GLOBAL);
    } catch (...) {
        H5Fclose(file);
        throw;
    }

    HDF5_CHECK_ERR(H5Fclose(file));

    std::cout << "Spectrum file created: " << path << "\n";
    return path;
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

    const std::string path = heom::hdf5::resolve_output_path(
        cfg.output.save_path, cfg.output.filename, cfg.output.use_timestamp);

    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("Cannot create HDF5 file: " + path);
    }

    try {
        heom::hdf5::write_root_identity(file, cfg, "2des");
        heom::hdf5::write_scalar_attr(file, "simulation_time_seconds", total_seconds);

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
        H5Fflush(file, H5F_SCOPE_GLOBAL);
    } catch (...) {
        H5Fclose(file);
        throw;
    }

    HDF5_CHECK_ERR(H5Fclose(file));
    std::cout << "2DES file created: " << path << "\n";
    return path;
}

}  // namespace heom::output
