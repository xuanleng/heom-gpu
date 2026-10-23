#include "hdf5_writer.hpp"
#include "system_context.hpp"
#include "output/hdf5_metadata.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <ctime>

namespace {

struct H5Handle {
    hid_t id = -1;
    herr_t (*close_fn)(hid_t) = nullptr;

    H5Handle(hid_t handle, herr_t (*close)(hid_t))
        : id(handle)
        , close_fn(close)
    {
        if (id < 0) {
            throw std::runtime_error("HDF5 error while opening handle");
        }
    }

    ~H5Handle() {
        if (id >= 0 && close_fn) {
            close_fn(id);
        }
    }

    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;

    hid_t get() const { return id; }
};

const char* init_state_type_str(InitialState::Type t) {
    switch (t) {
        case InitialState::Type::SITE_EXCITATION:    return "site_excitation";
        case InitialState::Type::EXCITON_EXCITATION: return "exciton_excitation";
        case InitialState::Type::COHERENT:           return "coherent";
        case InitialState::Type::CUSTOM_MATRIX:      return "custom_matrix";
    }
    return "unknown";
}

}  // namespace

HDF5Writer::HDF5Writer(bool use_timestamp)
    : use_timestamp_(use_timestamp)
    , file_(-1)
    , density_matrix_dataset_(-1)
    , time_dataset_(-1)
    , complex_type_(-1)
    , current_step_(0)
    , N_(0)
    , n_records_(0)
{
}

HDF5Writer::~HDF5Writer() {
    if (file_ < 0) return;
    // Reaching the destructor with the file still open means the run never got
    // to close(): an exception is unwinding through run_dynamics. The file on
    // disk has a valid layout but only `current_step_` of `n_records_` records,
    // so it says status="aborted". A destructor cannot throw; say what happened
    // on stderr instead.
    std::cerr << "HDF5Writer: closing " << output_path_ << " ABORTED ("
              << current_step_ << "/" << n_records_ << " records written)\n";
    try {
        heom::hdf5::write_scalar_attr(file_, "n_records_written", current_step_);
        heom::hdf5::write_run_status(file_, heom::hdf5::RunStatus::Aborted);
    } catch (const std::exception& e) {
        std::cerr << "HDF5Writer: could not mark file aborted: " << e.what() << "\n";
    }
    auto close_quietly = [](hid_t& id, herr_t (*close_fn)(hid_t)) {
        if (id >= 0) { close_fn(id); id = -1; }
    };
    close_quietly(density_matrix_dataset_, H5Dclose);
    close_quietly(time_dataset_,           H5Dclose);
    close_quietly(complex_type_,           H5Tclose);
    close_quietly(file_,                   H5Fclose);
}

void HDF5Writer::open(const SystemContext& sys,
                      const Eigen::MatrixXcd& initial_rho_used,
                      bool use_filtering,
                      double filter_threshold)
{
    const auto& cfg     = sys.config();
    const auto& phys    = cfg.physics;
    const auto& time    = cfg.time;

    // cfg.initial is std::optional<InitialState>. The
    // dynamics writer path (heom::run_dynamics) always reaches here after
    // building ρ0 from cfg.initial (which requires has_value). The check below catches a future caller
    // that wires the dynamics writer onto an absorption-style cfg by mistake.
    if (!cfg.initial.has_value()) {
        throw std::runtime_error(
            "HDF5Writer::open(): cfg.initial is empty (use spectrum_writer "
            "for spectroscopy tasks; this writer is dynamics-only).");
    }
    const InitialState& init_state = *cfg.initial;

    N_ = phys.n_sites;
    output_path_ = heom::hdf5::resolve_output_path(cfg.output.save_path,
                                                   cfg.output.filename,
                                                   use_timestamp_);

    file_ = heom::hdf5::create_output_file(output_path_, use_timestamp_);

    // ------------------------------------------------------------------
    // Group layout: /  /parameters  /derived  /results  /provenance
    // See src/output/hdf5_metadata.hpp for the contract each group obeys.
    // ------------------------------------------------------------------
    heom::hdf5::write_root_identity(file_, cfg, "dynamics");
    heom::hdf5::write_run_status(file_, heom::hdf5::RunStatus::Running);

    hid_t params = heom::hdf5::write_parameters_group(file_, sys,
                                                     use_filtering,
                                                     filter_threshold);

    // /parameters extras specific to the dynamics task
    heom::hdf5::write_string_attr(params, "initial_state_type",
                                  init_state_type_str(init_state.type));
    heom::hdf5::write_string_attr(params, "output_basis", "site");
    switch (init_state.type) {
        case InitialState::Type::SITE_EXCITATION:
            // 1-based PHYSICAL pigment index (matches YAML input semantics).
            heom::hdf5::write_scalar_attr(params, "excited_site",
                hilbert_to_pigment(init_state.excited_site, phys.basis));
            break;
        case InitialState::Type::EXCITON_EXCITATION:
            heom::hdf5::write_scalar_attr(params, "selected_exciton_index",
                hilbert_to_pigment(init_state.exciton_index, phys.basis));
            break;
        case InitialState::Type::COHERENT:
        case InitialState::Type::CUSTOM_MATRIX:
            break;
    }
    {
        std::vector<std::complex<double>> rho0(static_cast<size_t>(N_) * N_);
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < N_; ++j)
                rho0[i * N_ + j] = initial_rho_used(i, j);
        heom::hdf5::write_dataset_complex_2d(params, "initial_density_matrix",
                                             rho0.data(), N_, N_);
    }
    HDF5_CHECK_ERR(H5Gclose(params));

    heom::hdf5::write_derived_group   (file_, sys);
    heom::hdf5::write_provenance_group(file_, sys);

    // ------------------------------------------------------------------
    // /results: time + density_matrix(complex compound). Both are written
    // incrementally by write_timestep().
    // ------------------------------------------------------------------
    hid_t results = heom::hdf5::create_group(file_, "/results");

    // Dataset shape sized by the single-source-of-truth output_record_count.
    n_records_ = output_record_count(time);
    complex_type_ = heom::hdf5::make_complex_compound_type();

    hsize_t dims_time[1] = {(hsize_t)n_records_};
    hsize_t dims_rho[3]  = {(hsize_t)n_records_, (hsize_t)N_, (hsize_t)N_};

    H5Handle space_time(H5Screate_simple(1, dims_time, NULL), H5Sclose);
    H5Handle space_rho (H5Screate_simple(3, dims_rho,  NULL), H5Sclose);

    HDF5_CHECK_HID(time_dataset_,
                   H5Dcreate2(results, "time", H5T_NATIVE_DOUBLE, space_time.get(),
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHECK_HID(density_matrix_dataset_,
                   H5Dcreate2(results, "density_matrix", complex_type_, space_rho.get(),
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    HDF5_CHECK_ERR(H5Gclose(results));

    tmp_rho_.resize(static_cast<size_t>(N_) * N_);
    current_step_ = 0;

    HDF5_CHECK_ERR(H5Fflush(file_, H5F_SCOPE_GLOBAL));

    std::cout << "HDF5 file created: " << output_path_ << "\n";
}

void HDF5Writer::write_timestep(double time_fs, const std::complex<double>* rho, int N) {
    heom::hdf5::validate_timestep_bounds(N, N_, current_step_, n_records_);

    // ---- write time[current_step_] ----
    hsize_t start_time[1] = {(hsize_t)current_step_};
    hsize_t count_time[1] = {1};
    H5Handle space(H5Dget_space(time_dataset_), H5Sclose);
    HDF5_CHECK_ERR(H5Sselect_hyperslab(space.get(), H5S_SELECT_SET,
                                       start_time, NULL, count_time, NULL));
    H5Handle memspace_time(H5Screate_simple(1, count_time, NULL), H5Sclose);
    HDF5_CHECK_ERR(H5Dwrite(time_dataset_, H5T_NATIVE_DOUBLE,
                            memspace_time.get(), space.get(),
                            H5P_DEFAULT, &time_fs));

    // ---- write density_matrix[current_step_] (complex compound) ----
    std::copy(rho, rho + static_cast<size_t>(N) * N, tmp_rho_.begin());

    hsize_t start_rho[3] = {(hsize_t)current_step_, 0, 0};
    hsize_t count_rho[3] = {1, (hsize_t)N, (hsize_t)N};
    H5Handle rho_space(H5Dget_space(density_matrix_dataset_), H5Sclose);
    HDF5_CHECK_ERR(H5Sselect_hyperslab(rho_space.get(), H5S_SELECT_SET,
                                       start_rho, NULL, count_rho, NULL));
    hsize_t mem_dims_rho[2] = {(hsize_t)N, (hsize_t)N};
    H5Handle rho_memspace(H5Screate_simple(2, mem_dims_rho, NULL), H5Sclose);
    HDF5_CHECK_ERR(H5Dwrite(density_matrix_dataset_, complex_type_,
                            rho_memspace.get(), rho_space.get(),
                            H5P_DEFAULT, tmp_rho_.data()));

    current_step_++;
}

void HDF5Writer::close(double simulation_time, heom::hdf5::RunStatus status,
                       const std::vector<FilteringRecord>* filtering_history) {
    if (file_ < 0) return;

    heom::hdf5::write_scalar_attr(file_, "simulation_time_seconds", simulation_time);
    heom::hdf5::write_scalar_attr(file_, "n_records_written", current_step_);

    if (filtering_history && !filtering_history->empty()) {
        // M11: written into /derived because final_active_ados is a runtime
        // outcome of the ADO filter, not a user-facing knob. We re-open the
        // group instead of carrying its hid_t through the writer's lifetime.
        // open() created /derived; failing to reopen it is a broken file, not
        // an optional extra — a missing final_active_ados must not be silent.
        hid_t derived;
        HDF5_CHECK_HID(derived, H5Gopen2(file_, "/derived", H5P_DEFAULT));
        heom::hdf5::write_scalar_attr(derived, "final_active_ados",
                                      filtering_history->back().active_ados);
        HDF5_CHECK_ERR(H5Gclose(derived));
    }
    heom::hdf5::write_run_status(file_, status);

    auto close_if_open = [](hid_t& id, herr_t (*close_fn)(hid_t), const char* name) {
        if (id < 0) return;
        if (close_fn(id) < 0) {
            throw std::runtime_error(std::string("HDF5 error closing ") + name);
        }
        id = -1;
    };

    close_if_open(density_matrix_dataset_, H5Dclose, "density_matrix");
    close_if_open(time_dataset_,           H5Dclose, "time");
    close_if_open(complex_type_,           H5Tclose, "complex_type");
    close_if_open(file_,                   H5Fclose, "file");

    N_ = 0;
    n_records_ = 0;
}
