#include "mc/mc_realizations.hpp"

#include "output/hdf5_writer.hpp"   // HDF5_CHECK_ERR / HDF5_CHECK_HID

#include <hdf5.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace heom::mc {

std::vector<double> RealizationSet::row(int r) const {
    if (r < 0 || r >= n_node) {
        throw std::out_of_range("RealizationSet::row: r=" + std::to_string(r) +
                                " out of [0," + std::to_string(n_node) + ")");
    }
    const auto begin = delta_eps.begin() + static_cast<long>(r) * n_site;
    return std::vector<double>(begin, begin + n_site);
}

namespace {

// Read a scalar/array double dataset or attribute into a flat vector, returning
// its total element count. Datasets only (attrs handled separately below).
std::vector<double> read_double_dataset(hid_t file, const char* name,
                                        std::vector<hsize_t>& dims_out) {
    hid_t dset;
    HDF5_CHECK_HID(dset, H5Dopen2(file, name, H5P_DEFAULT));
    hid_t space = H5Dget_space(dset);
    int ndims = H5Sget_simple_extent_ndims(space);
    dims_out.assign(ndims, 0);
    H5Sget_simple_extent_dims(space, dims_out.data(), nullptr);
    hsize_t n = 1;
    for (auto d : dims_out) n *= d;
    std::vector<double> buf(n);
    HDF5_CHECK_ERR(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, buf.data()));
    H5Sclose(space);
    H5Dclose(dset);
    return buf;
}

std::vector<int> read_int_dataset(hid_t file, const char* name, hsize_t& n_out) {
    hid_t dset;
    HDF5_CHECK_HID(dset, H5Dopen2(file, name, H5P_DEFAULT));
    hid_t space = H5Dget_space(dset);
    hsize_t n = 0;
    H5Sget_simple_extent_dims(space, &n, nullptr);
    std::vector<int> buf(n);
    HDF5_CHECK_ERR(H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, buf.data()));
    H5Sclose(space);
    H5Dclose(dset);
    n_out = n;
    return buf;
}

std::string read_string_attr(hid_t file, const char* name) {
    hid_t attr;
    HDF5_CHECK_HID(attr, H5Aopen(file, name, H5P_DEFAULT));
    // Read with the attribute's own datatype: guarantees an identity conversion
    // path regardless of charset (h5py writes UTF-8) or fixed/variable length.
    hid_t atype = H5Aget_type(attr);
    std::string out;
    if (H5Tis_variable_str(atype)) {
        char* rdata = nullptr;
        HDF5_CHECK_ERR(H5Aread(attr, atype, &rdata));
        if (rdata) { out = rdata; H5free_memory(rdata); }
    } else {
        size_t sz = H5Tget_size(atype);
        std::vector<char> buf(sz + 1, '\0');
        HDF5_CHECK_ERR(H5Aread(attr, atype, buf.data()));
        out = buf.data();
    }
    H5Tclose(atype);
    H5Aclose(attr);
    return out;
}

std::uint64_t read_u64_attr(hid_t file, const char* name) {
    hid_t attr;
    HDF5_CHECK_HID(attr, H5Aopen(file, name, H5P_DEFAULT));
    std::uint64_t v = 0;
    HDF5_CHECK_ERR(H5Aread(attr, H5T_NATIVE_UINT64, &v));
    H5Aclose(attr);
    return v;
}

int read_int_attr(hid_t file, const char* name) {
    hid_t attr;
    HDF5_CHECK_HID(attr, H5Aopen(file, name, H5P_DEFAULT));
    int v = 0;
    HDF5_CHECK_ERR(H5Aread(attr, H5T_NATIVE_INT, &v));
    H5Aclose(attr);
    return v;
}

std::vector<double> read_double_attr(hid_t file, const char* name) {
    hid_t attr;
    HDF5_CHECK_HID(attr, H5Aopen(file, name, H5P_DEFAULT));
    hid_t space = H5Aget_space(attr);
    hssize_t n = H5Sget_simple_extent_npoints(space);
    std::vector<double> buf(n > 0 ? static_cast<size_t>(n) : 0);
    HDF5_CHECK_ERR(H5Aread(attr, H5T_NATIVE_DOUBLE, buf.data()));
    H5Sclose(space);
    H5Aclose(attr);
    return buf;
}

} // namespace

RealizationSet load_realizations(const std::string& path) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("load_realizations: cannot open '" + path + "'");
    }

    RealizationSet R;
    try {
        std::vector<hsize_t> de_dims;
        R.delta_eps = read_double_dataset(file, "delta_eps", de_dims);
        if (de_dims.size() != 2) {
            throw std::runtime_error("delta_eps must be 2D [n_node, n_site]");
        }
        R.n_node = static_cast<int>(de_dims[0]);
        R.n_site = static_cast<int>(de_dims[1]);

        std::vector<hsize_t> w_dims;
        R.weight = read_double_dataset(file, "weight", w_dims);
        hsize_t g_n = 0;
        R.group = read_int_dataset(file, "group", g_n);
        if (static_cast<int>(R.weight.size()) != R.n_node ||
            static_cast<int>(g_n) != R.n_node) {
            throw std::runtime_error("weight/group length != n_node");
        }

        const std::string est = read_string_attr(file, "estimator");
        if (est == "mean") {
            R.estimator = Estimator::Mean;
        } else if (est == "quadrature") {
            R.estimator = Estimator::Quadrature;
        } else {
            throw std::runtime_error(
                "load_realizations: estimator='" + est +
                "' unsupported (expected 'mean' or 'quadrature')");
        }

        R.generator = read_string_attr(file, "generator");
        const std::vector<double> sigma = read_double_attr(file, "sigma_cm");

        // disorder_identity binds shards for merge. Quadrature keys
        // on the Smolyak level (refinement = new identity ⇒ from-scratch rerun);
        // mean keys on the RNG seed. Both carry σ so a re-parameterised
        // run never silently merges with the old one.
        std::ostringstream id;
        if (R.estimator == Estimator::Quadrature) {
            const int level = read_int_attr(file, "sg_level");
            id << R.generator << "|level=" << level << "|sigma=";
        } else {
            const std::uint64_t seed = read_u64_attr(file, "master_seed");
            id << R.generator << "|seed=" << seed << "|sigma=";
        }
        for (size_t i = 0; i < sigma.size(); ++i) {
            id << (i ? "," : "") << sigma[i];
        }
        R.disorder_identity = id.str();
    } catch (...) {
        H5Fclose(file);
        throw;
    }
    H5Fclose(file);
    return R;
}

} // namespace heom::mc
