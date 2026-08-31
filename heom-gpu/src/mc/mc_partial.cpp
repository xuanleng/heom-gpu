#include "mc/mc_partial.hpp"

#include "output/hdf5_metadata.hpp"   // make_complex_compound_type, write_string_attr
#include "output/hdf5_writer.hpp"     // HDF5_CHECK_ERR / HDF5_CHECK_HID

#include <hdf5.h>

#include <stdexcept>
#include <vector>

namespace heom::mc {

namespace {

void write_int64_array_attr(hid_t parent, const char* name,
                            const std::vector<long long>& data) {
    hsize_t n = data.size();
    hid_t space, attr;
    HDF5_CHECK_HID(space, H5Screate_simple(1, &n, nullptr));
    HDF5_CHECK_HID(attr, H5Acreate2(parent, name, H5T_NATIVE_LLONG, space,
                                    H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHECK_ERR(H5Awrite(attr, H5T_NATIVE_LLONG, data.data()));
    HDF5_CHECK_ERR(H5Aclose(attr));
    HDF5_CHECK_ERR(H5Sclose(space));
}

} // namespace

void append_mc_provenance(const std::string& path, const Accumulator& acc,
                          const std::string& config_fingerprint,
                          const std::string& disorder_identity) {
    hid_t file;
    HDF5_CHECK_HID(file, H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));

    hid_t mc;
    HDF5_CHECK_HID(mc, H5Gcreate2(file, "/mc", H5P_DEFAULT, H5P_DEFAULT,
                                  H5P_DEFAULT));

    std::vector<hsize_t> dims(acc.obs_shape.begin(), acc.obs_shape.end());
    const int rank = static_cast<int>(dims.size());

    if (acc.estimator == Estimator::Mean) {
        // sum_obs_sq = Σ_r |O_r|²: float64, same shape as /results observable.
        // Kept (not sum_obs) because the merge step needs it for the SEM; the
        // observable itself already lives in /results as the mean.
        hid_t space, dset;
        HDF5_CHECK_HID(space, H5Screate_simple(rank, dims.data(), nullptr));
        HDF5_CHECK_HID(dset, H5Dcreate2(mc, "sum_obs_sq", H5T_NATIVE_DOUBLE,
                                        space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        HDF5_CHECK_ERR(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                H5P_DEFAULT, acc.sum_obs_sq.data()));
        HDF5_CHECK_ERR(H5Dclose(dset));
        HDF5_CHECK_ERR(H5Sclose(space));
    } else {
        // Quadrature: persist the partial weighted sum Σ_r w_r·O_r as complex128
        // (h5py reads the {r,i} compound as np.complex128). The merge sums these
        // directly across shards — /results holds only a non-normalised
        // placeholder, so sum_obs is the authoritative quadrature payload.
        hid_t ctype = heom::hdf5::make_complex_compound_type();
        hid_t space, dset;
        HDF5_CHECK_HID(space, H5Screate_simple(rank, dims.data(), nullptr));
        HDF5_CHECK_HID(dset, H5Dcreate2(mc, "sum_obs", ctype, space,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        HDF5_CHECK_ERR(H5Dwrite(dset, ctype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                acc.sum_obs.data()));
        HDF5_CHECK_ERR(H5Dclose(dset));
        HDF5_CHECK_ERR(H5Sclose(space));
        HDF5_CHECK_ERR(H5Tclose(ctype));
    }

    // count: int64 scalar (this worker's realization count).
    {
        hid_t space, dset;
        HDF5_CHECK_HID(space, H5Screate(H5S_SCALAR));
        HDF5_CHECK_HID(dset, H5Dcreate2(mc, "count", H5T_NATIVE_LLONG, space,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        HDF5_CHECK_ERR(H5Dwrite(dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                                H5P_DEFAULT, &acc.count));
        HDF5_CHECK_ERR(H5Dclose(dset));
        HDF5_CHECK_ERR(H5Sclose(space));
    }

    const char* est_str =
        (acc.estimator == Estimator::Quadrature) ? "quadrature" : "mean";
    heom::hdf5::write_string_attr(mc, "estimator", est_str);
    write_int64_array_attr(mc, "r_done", acc.r_done);
    heom::hdf5::write_string_attr(mc, "config_fingerprint", config_fingerprint);
    heom::hdf5::write_string_attr(mc, "disorder_identity", disorder_identity);

    HDF5_CHECK_ERR(H5Gclose(mc));
    HDF5_CHECK_ERR(H5Fclose(file));
}

} // namespace heom::mc
