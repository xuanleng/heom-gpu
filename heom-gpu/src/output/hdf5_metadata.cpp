#include "output/hdf5_metadata.hpp"

#include "system_context.hpp"
#include "heom_version.hpp"
#include "output/hdf5_writer.hpp"   // HDF5_CHECK_ERR / HDF5_CHECK_HID macros
#include "core/precision.hpp"       // heom::kMixedPrecision (precision_mode attr)

#include <chrono>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace heom::hdf5 {

// ---------------------------------------------------------------------------
// Compound complex type
// ---------------------------------------------------------------------------

hid_t make_complex_compound_type() {
    static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
                  "std::complex<double> must be layout-compatible with [r,i]");

    hid_t type;
    HDF5_CHECK_HID(type, H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>)));
    HDF5_CHECK_ERR(H5Tinsert(type, "r", 0,              H5T_NATIVE_DOUBLE));
    HDF5_CHECK_ERR(H5Tinsert(type, "i", sizeof(double), H5T_NATIVE_DOUBLE));
    return type;
}

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

void write_scalar_attr(hid_t parent, const char* name, int value) {
    hid_t space, attr;
    HDF5_CHECK_HID(space, H5Screate(H5S_SCALAR));
    HDF5_CHECK_HID(attr,  H5Acreate2(parent, name, H5T_NATIVE_INT, space,
                                     H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHECK_ERR(H5Awrite(attr, H5T_NATIVE_INT, &value));
    HDF5_CHECK_ERR(H5Aclose(attr));
    HDF5_CHECK_ERR(H5Sclose(space));
}

void write_scalar_attr(hid_t parent, const char* name, double value) {
    hid_t space, attr;
    HDF5_CHECK_HID(space, H5Screate(H5S_SCALAR));
    HDF5_CHECK_HID(attr,  H5Acreate2(parent, name, H5T_NATIVE_DOUBLE, space,
                                     H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHECK_ERR(H5Awrite(attr, H5T_NATIVE_DOUBLE, &value));
    HDF5_CHECK_ERR(H5Aclose(attr));
    HDF5_CHECK_ERR(H5Sclose(space));
}

void write_string_attr(hid_t parent, const char* name, const std::string& value) {
    hid_t str_type, space, attr;
    HDF5_CHECK_HID(str_type, H5Tcopy(H5T_C_S1));
    HDF5_CHECK_ERR(H5Tset_size(str_type, H5T_VARIABLE));
    HDF5_CHECK_HID(space, H5Screate(H5S_SCALAR));
    HDF5_CHECK_HID(attr,  H5Acreate2(parent, name, str_type, space,
                                     H5P_DEFAULT, H5P_DEFAULT));
    const char* cstr = value.c_str();
    HDF5_CHECK_ERR(H5Awrite(attr, str_type, &cstr));
    HDF5_CHECK_ERR(H5Aclose(attr));
    HDF5_CHECK_ERR(H5Sclose(space));
    HDF5_CHECK_ERR(H5Tclose(str_type));
}

void write_string_dataset(hid_t parent, const char* name, const std::string& value) {
    hid_t str_type, space, dataset;
    HDF5_CHECK_HID(str_type, H5Tcopy(H5T_C_S1));
    HDF5_CHECK_ERR(H5Tset_size(str_type, H5T_VARIABLE));
    HDF5_CHECK_HID(space, H5Screate(H5S_SCALAR));
    HDF5_CHECK_HID(dataset, H5Dcreate2(parent, name, str_type, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    const char* cstr = value.c_str();
    HDF5_CHECK_ERR(H5Dwrite(dataset, str_type, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, &cstr));
    HDF5_CHECK_ERR(H5Dclose(dataset));
    HDF5_CHECK_ERR(H5Sclose(space));
    HDF5_CHECK_ERR(H5Tclose(str_type));
}

void write_dataset_1d(hid_t parent, const char* name, const double* data, int n) {
    hsize_t dims[1] = { static_cast<hsize_t>(n) };
    hid_t space, dataset;
    HDF5_CHECK_HID(space,   H5Screate_simple(1, dims, NULL));
    HDF5_CHECK_HID(dataset, H5Dcreate2(parent, name, H5T_NATIVE_DOUBLE, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHECK_ERR(H5Dwrite(dataset, H5T_NATIVE_DOUBLE,
                            H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    HDF5_CHECK_ERR(H5Dclose(dataset));
    HDF5_CHECK_ERR(H5Sclose(space));
}

void write_dataset_2d(hid_t parent, const char* name, const double* data,
                      int rows, int cols)
{
    hsize_t dims[2] = { static_cast<hsize_t>(rows),
                        static_cast<hsize_t>(cols) };
    hid_t space, dataset;
    HDF5_CHECK_HID(space,   H5Screate_simple(2, dims, NULL));
    HDF5_CHECK_HID(dataset, H5Dcreate2(parent, name, H5T_NATIVE_DOUBLE, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHECK_ERR(H5Dwrite(dataset, H5T_NATIVE_DOUBLE,
                            H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    HDF5_CHECK_ERR(H5Dclose(dataset));
    HDF5_CHECK_ERR(H5Sclose(space));
}

void write_dataset_complex_1d(hid_t parent, const char* name,
                              const std::complex<double>* data, int n)
{
    hsize_t dims[1] = { static_cast<hsize_t>(n) };
    hid_t space, dataset;
    hid_t ctype = make_complex_compound_type();
    HDF5_CHECK_HID(space,   H5Screate_simple(1, dims, NULL));
    HDF5_CHECK_HID(dataset, H5Dcreate2(parent, name, ctype, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHECK_ERR(H5Dwrite(dataset, ctype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    HDF5_CHECK_ERR(H5Dclose(dataset));
    HDF5_CHECK_ERR(H5Sclose(space));
    HDF5_CHECK_ERR(H5Tclose(ctype));
}

void write_dataset_complex_2d(hid_t parent, const char* name,
                              const std::complex<double>* data,
                              int rows, int cols)
{
    hsize_t dims[2] = { static_cast<hsize_t>(rows),
                        static_cast<hsize_t>(cols) };
    hid_t space, dataset;
    hid_t ctype = make_complex_compound_type();
    HDF5_CHECK_HID(space,   H5Screate_simple(2, dims, NULL));
    HDF5_CHECK_HID(dataset, H5Dcreate2(parent, name, ctype, space,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHECK_ERR(H5Dwrite(dataset, ctype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    HDF5_CHECK_ERR(H5Dclose(dataset));
    HDF5_CHECK_ERR(H5Sclose(space));
    HDF5_CHECK_ERR(H5Tclose(ctype));
}

hid_t create_group(hid_t parent, const char* name) {
    hid_t group;
    HDF5_CHECK_HID(group, H5Gcreate2(parent, name,
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    return group;
}

namespace {

const char* basis_to_str(BasisType b) {
    switch (b) {
        case BasisType::EXCITED_ONLY:    return "excited_only";
        case BasisType::WITH_GROUND:     return "with_ground";
        case BasisType::DOUBLE_EXCITED:  return "double_excited";
    }
    return "unknown";
}

}  // namespace

// ---------------------------------------------------------------------------
// Group writers
// ---------------------------------------------------------------------------

void write_root_identity(hid_t file,
                         const HEOMConfig::Config& cfg,
                         const char* task_name)
{
    write_string_attr(file, "task",        task_name);
    write_string_attr(file, "config_hash", cfg.config_hash);
    // Which binary produced this file. NOT part of config_hash / merge
    // key — precision is a build choice, not a physical parameter. Written by
    // both builds so the MC merge guard (mc_merge.py) can refuse to average
    // FP64 and mixed-FP32 shards together (silent-mix defence).
    write_string_attr(file, "precision_mode",
                      heom::kMixedPrecision ? "mixed_fp32" : "fp64");
}

hid_t write_parameters_group(hid_t file,
                             const SystemContext& sys,
                             bool use_filtering,
                             double filter_threshold)
{
    const auto& cfg  = sys.config();
    const auto& phys = cfg.physics;
    const auto& time = cfg.time;

    hid_t params = create_group(file, "/parameters");

    // ---- physics ----
    write_scalar_attr(params, "n_sites",                  phys.n_sites);
    write_string_attr(params, "basis",                    basis_to_str(phys.basis));
    write_scalar_attr(params, "temperature",              phys.temperature_K);
    write_scalar_attr(params, "w0_cm",                    phys.w0_cm);
    // Per-site bath inputs as 1D datasets [n_bath] (1-based pigment order).
    {
        const int nb = phys.n_bath();
        std::vector<double> eta_cm(nb), gamma_cm(nb), gamma_inv_fs(nb);
        for (int n = 0; n < nb; ++n) {
            const BathSite& s = phys.bath_sites[n];
            eta_cm[n]       = s.eta_cm;
            gamma_cm[n]     = s.gamma_au / CM_TO_AU;
            gamma_inv_fs[n] = s.gamma_inverse_fs;
        }
        write_dataset_1d(params, "eta_cm",                  eta_cm.data(),       nb);
        write_dataset_1d(params, "gamma_cm",                gamma_cm.data(),     nb);
        write_dataset_1d(params, "gamma_inverse_fs",        gamma_inv_fs.data(), nb);
    }
    write_string_attr(params, "expansion_method",
                      phys.expansion == BathExpansion::PADE ? "pade" : "matsubara");
    write_scalar_attr(params, "n_extra_poles",            phys.n_extra_poles);
    write_scalar_attr(params, "n_bath_modes_total",       phys.n_bath_modes_total);

    // ---- time grid ----
    write_scalar_attr(params, "dt",                       time.dt);
    write_scalar_attr(params, "n_steps",                  time.n_steps);
    write_scalar_attr(params, "output_interval",          time.output_interval);

    // ---- HEOM truncation ----
    write_scalar_attr(params, "L_cutoff",                 sys.L());
    write_scalar_attr(params, "use_filtering",            use_filtering ? 1 : 0);
    write_scalar_attr(params, "filter_threshold",         filter_threshold);

    // ---- sink ----
    write_scalar_attr(params, "sink_enabled",             phys.sink.enabled ? 1 : 0);
    if (phys.sink.enabled) {
        // Store the 1-based PHYSICAL pigment index (matching YAML
        // input semantics). Downstream Python tools read this attribute
        // and treat it as a direct pigment label.
        write_scalar_attr(params, "sink_site",            phys.sink.site);
        write_scalar_attr(params, "gamma_sink_fs",        phys.sink.gamma_fs);
        write_scalar_attr(params, "gamma_sink_au",        phys.sink.gamma);
    }

    // ---- Hamiltonian (always; small; central to every downstream tool) ----
    {
        const int N = phys.n_sites;
        std::vector<double> H_real(static_cast<size_t>(N) * N);
        auto H_cm = phys.hamiltonian_in_cm();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                H_real[i * N + j] = H_cm(i, j).real();
        write_dataset_2d(params, "hamiltonian_cm", H_real.data(), N, N);
    }

    // ---- config_yaml (single source of truth for user input) ----
    write_string_dataset(params, "config_yaml", cfg.config_yaml);

    return params;
}

void write_derived_group(hid_t file, const SystemContext& sys) {
    const auto& cfg  = sys.config();
    const auto& phys = cfg.physics;

    hid_t derived = create_group(file, "/derived");

    write_scalar_attr(derived, "M_total", sys.M());

    // Eigenspectrum: relative energies only (absolute = relative + w0_cm
    // is one subtraction in any consumer; keeping both invites drift).
    const Eigen::VectorXd&  eigenvalues  = sys.eigenvalues();
    const Eigen::MatrixXcd& eigenvectors = sys.eigenvectors();
    if (eigenvalues.size() > 0 && eigenvectors.size() > 0) {
        const int N = phys.n_sites;
        std::vector<double> eig_rel(N);
        for (int i = 0; i < N; ++i) {
            eig_rel[i] = eigenvalues(i) / CM_TO_AU;
        }
        write_dataset_1d(derived, "eigenvalues_relative_cm", eig_rel.data(), N);

        std::vector<std::complex<double>> evec(static_cast<size_t>(N) * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                evec[i * N + j] = eigenvectors(i, j);
            }
        }
        write_dataset_complex_2d(derived, "eigenvectors", evec.data(), N, N);
    }

    // Per-site bath-correlation coefficients (c_{nk} complex, ν_{nk} real) in
    // cm^-1 units, shape [n_bath][K+1] row-major.
    {
        const int nb = phys.n_bath();
        const int K  = phys.n_bath_modes_total;
        const size_t len = static_cast<size_t>(nb) * (K + 1);
        std::vector<std::complex<double>> c_cm2(len);
        std::vector<double>               nu_cm(len);
        for (int n = 0; n < nb; ++n) {
            for (int k = 0; k <= K; ++k) {
                const int idx = phys.mode_idx(n, k);
                c_cm2[idx] = phys.c_bath_in_cm(n, k);
                nu_cm[idx] = phys.nu_bath_in_cm(n, k);
            }
        }
        write_dataset_complex_2d(derived, "bath_c_cm2", c_cm2.data(), nb, K + 1);
        write_dataset_2d        (derived, "bath_nu_cm", nu_cm.data(),  nb, K + 1);
    }

    // Padé path: dump the dimensionless poles ζ_l and rational weights χ_l
    // (cached in compute_pade_drude, length n_extra_poles + 1 with [0]=0
    // placeholder) for post-processing cross-checks. Never written on the
    // Matsubara path. Read straight from the cache — do NOT build empty
    // vectors here or the datasets come out all-zero.
    if (phys.expansion == BathExpansion::PADE) {
        write_dataset_1d(derived, "pade_zeta_dimensionless",
                         phys.pade_zeta.data(),
                         static_cast<int>(phys.pade_zeta.size()));
        write_dataset_1d(derived, "pade_chi",
                         phys.pade_chi.data(),
                         static_cast<int>(phys.pade_chi.size()));
    }

    HDF5_CHECK_ERR(H5Gclose(derived));
}

void write_provenance_group(hid_t file, const SystemContext& sys) {
    const auto& sysinfo = sys.system_info();
    const auto& cfg     = sys.config();

    hid_t prov = create_group(file, "/provenance");

    write_string_attr(prov, "software_name",         heom_software_name());
    write_string_attr(prov, "software_version",      heom_software_version());
    write_string_attr(prov, "cpu_model",             sysinfo.cpu_model);
    write_string_attr(prov, "gpu_model",             sysinfo.gpu_model);
    write_string_attr(prov, "os",                    sysinfo.os_name + " " + sysinfo.os_version);
    write_string_attr(prov, "cuda_version",          sysinfo.cuda_version);
    write_string_attr(prov, "compiler",              sysinfo.compiler_version);
    write_string_attr(prov, "config_path",           cfg.yaml_path);

    HDF5_CHECK_ERR(H5Gclose(prov));
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

void ensure_directory_exists(const std::string& path) {
    size_t last_slash = path.find_last_of("/");
    if (last_slash == std::string::npos) {
        return;  // No directory component
    }

    std::string dir = path.substr(0, last_slash);

    // Recursively create all parent directories (equivalent to mkdir -p).
    size_t pos = 0;
    while ((pos = dir.find('/', pos)) != std::string::npos) {
        if (pos > 0) {  // Skip leading '/'
            std::string subdir = dir.substr(0, pos);
            if (mkdir(subdir.c_str(), 0755) != 0 && errno != EEXIST) {
                throw std::runtime_error("Cannot create directory '" + subdir +
                                         "': " + std::strerror(errno));
            }
        }
        pos++;
    }

    if (!dir.empty() && mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory '" + dir +
                                 "': " + std::strerror(errno));
    }
}

std::string resolve_output_path(const std::string& dir,
                                const std::string& filename,
                                bool use_timestamp)
{
    if (!use_timestamp) {
        std::string path = dir + filename + ".h5";
        ensure_directory_exists(path);
        return path;
    }

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&time_t_now, &tm_buf);

    std::stringstream ts;
    ts << std::setfill('0')
       << std::setw(4) << (tm_buf.tm_year + 1900)
       << std::setw(2) << (tm_buf.tm_mon + 1)
       << std::setw(2) << tm_buf.tm_mday
       << "_"
       << std::setw(2) << tm_buf.tm_hour
       << std::setw(2) << tm_buf.tm_min;

    std::string path = dir + filename + "_" + ts.str() + ".h5";
    ensure_directory_exists(path);
    return path;
}

}  // namespace heom::hdf5
