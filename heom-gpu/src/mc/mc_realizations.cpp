#include "mc/mc_realizations.hpp"

#include "output/hdf5_writer.hpp"   // HDF5_CHECK_ERR / HDF5_CHECK_HID

#include <hdf5.h>

#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
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

// Owns one HDF5 id and closes it on every path — a throwing read included.
class Hid {
public:
    Hid(hid_t id, herr_t (*close)(hid_t), const char* name) : id_(id), close_(close) {
        if (id_ < 0)
            throw std::runtime_error(std::string("load_realizations: cannot open '") +
                                     name + "'");
    }
    ~Hid() { close_(id_); }
    Hid(const Hid&) = delete;
    Hid& operator=(const Hid&) = delete;
    hid_t get() const { return id_; }

private:
    hid_t id_;
    herr_t (*close_)(hid_t);
};

std::vector<hsize_t> extent(hid_t space) {
    int ndims;
    HDF5_CHECK_ERR(ndims = H5Sget_simple_extent_ndims(space));
    std::vector<hsize_t> dims(ndims, 0);
    HDF5_CHECK_ERR(H5Sget_simple_extent_dims(space, dims.data(), nullptr));
    return dims;
}

// A dataset as a flat vector of T (mem_type), with its shape in dims_out.
template <class T>
std::vector<T> read_dataset(hid_t file, const char* name, hid_t mem_type,
                            std::vector<hsize_t>& dims_out) {
    Hid dset(H5Dopen2(file, name, H5P_DEFAULT), H5Dclose, name);
    Hid space(H5Dget_space(dset.get()), H5Sclose, name);
    dims_out = extent(space.get());
    hsize_t n = 1;
    for (auto d : dims_out) n *= d;
    std::vector<T> buf(n);
    if (n > 0)
        HDF5_CHECK_ERR(H5Dread(dset.get(), mem_type, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, buf.data()));
    return buf;
}

// The buffer is sized from the attribute's own extent, so an array where a
// scalar was expected can never write past it.
template <class T>
std::vector<T> read_attr(hid_t file, const char* name, hid_t mem_type) {
    Hid attr(H5Aopen(file, name, H5P_DEFAULT), H5Aclose, name);
    Hid space(H5Aget_space(attr.get()), H5Sclose, name);
    hssize_t n;
    HDF5_CHECK_ERR(n = H5Sget_simple_extent_npoints(space.get()));
    std::vector<T> buf(static_cast<size_t>(n));
    if (n > 0) HDF5_CHECK_ERR(H5Aread(attr.get(), mem_type, buf.data()));
    return buf;
}

template <class T>
T read_scalar_attr(hid_t file, const char* name, hid_t mem_type) {
    const std::vector<T> v = read_attr<T>(file, name, mem_type);
    if (v.size() != 1)
        throw std::runtime_error(std::string("load_realizations: attribute '") + name +
                                 "' must hold one value, got " + std::to_string(v.size()));
    return v[0];
}

// hsize_t → int for n_node / n_site. A file with > INT_MAX nodes is not a
// realization set we can index; refuse rather than wrap negative.
int checked_int(hsize_t v, const char* what) {
    if (v > static_cast<hsize_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("load_realizations: ") + what + "=" +
                                 std::to_string(v) + " exceeds int range");
    }
    return static_cast<int>(v);
}

std::string read_string_attr(hid_t file, const char* name) {
    Hid attr(H5Aopen(file, name, H5P_DEFAULT), H5Aclose, name);
    Hid space(H5Aget_space(attr.get()), H5Sclose, name);
    hssize_t n;
    HDF5_CHECK_ERR(n = H5Sget_simple_extent_npoints(space.get()));
    if (n != 1)
        throw std::runtime_error(std::string("load_realizations: attribute '") + name +
                                 "' must be one string, got " + std::to_string(n));
    // Read with the attribute's own datatype: guarantees an identity conversion
    // path regardless of charset (h5py writes UTF-8) or fixed/variable length.
    Hid atype(H5Aget_type(attr.get()), H5Tclose, name);
    if (H5Tget_class(atype.get()) != H5T_STRING)
        throw std::runtime_error(std::string("load_realizations: attribute '") + name +
                                 "' is not a string");
    std::string out;
    if (H5Tis_variable_str(atype.get())) {
        char* rdata = nullptr;
        HDF5_CHECK_ERR(H5Aread(attr.get(), atype.get(), &rdata));
        if (rdata) { out = rdata; H5free_memory(rdata); }
    } else {
        std::vector<char> buf(H5Tget_size(atype.get()) + 1, '\0');
        HDF5_CHECK_ERR(H5Aread(attr.get(), atype.get(), buf.data()));
        out = buf.data();
    }
    return out;
}

} // namespace

RealizationSet load_realizations(const std::string& path) {
    const Hid fh(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose,
                 path.c_str());
    const hid_t file = fh.get();

    RealizationSet R;
    std::vector<hsize_t> de_dims;
    R.delta_eps = read_dataset<double>(file, "delta_eps", H5T_NATIVE_DOUBLE, de_dims);
    if (de_dims.size() != 2) {
        throw std::runtime_error("delta_eps must be 2D [n_node, n_site]");
    }
    R.n_node = checked_int(de_dims[0], "n_node");
    R.n_site = checked_int(de_dims[1], "n_site");

    std::vector<hsize_t> w_dims, g_dims;
    R.weight = read_dataset<double>(file, "weight", H5T_NATIVE_DOUBLE, w_dims);
    R.group  = read_dataset<int>(file, "group", H5T_NATIVE_INT, g_dims);
    if (w_dims.size() != 1 || g_dims.size() != 1 ||
        w_dims[0] != de_dims[0] || g_dims[0] != de_dims[0]) {
        throw std::runtime_error("load_realizations: weight/group must be 1D of "
                                 "length n_node=" + std::to_string(R.n_node));
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
    const std::vector<double> sigma =
        read_attr<double>(file, "sigma_cm", H5T_NATIVE_DOUBLE);
    if (sigma.size() != static_cast<size_t>(R.n_site)) {
        throw std::runtime_error(
            "load_realizations: sigma_cm has " + std::to_string(sigma.size()) +
            " entries, delta_eps has n_site=" + std::to_string(R.n_site));
    }

    // disorder_identity binds shards for merge. Quadrature keys
    // on the Smolyak level (refinement = new identity ⇒ from-scratch rerun);
    // mean keys on the RNG seed. Both carry σ so a re-parameterised
    // run never silently merges with the old one.
    //
    // σ is written with SIX significant digits, and that is an on-disk
    // contract, not a default we forgot to override: every production shard
    // carries e.g. "sigma=42.4661,..." (FWHM 100 cm⁻¹), and the merge groups
    // shards by exact string. Two σ that agree to 1e-6 relative are the
    // same disorder for any MC purpose, so the 6-digit collision is harmless;
    // re-emitting at 17 digits would orphan every existing shard. Classic
    // locale so a de_DE LANG cannot turn the decimal point into the ','
    // that separates sites.
    std::ostringstream id;
    id.imbue(std::locale::classic());
    id << std::setprecision(kSigmaIdentityDigits);
    if (R.estimator == Estimator::Quadrature) {
        const int level = read_scalar_attr<int>(file, "sg_level", H5T_NATIVE_INT);
        id << R.generator << "|level=" << level << "|sigma=";
    } else {
        const auto seed =
            read_scalar_attr<std::uint64_t>(file, "master_seed", H5T_NATIVE_UINT64);
        id << R.generator << "|seed=" << seed << "|sigma=";
    }
    for (size_t i = 0; i < sigma.size(); ++i) {
        id << (i ? "," : "") << sigma[i];
    }
    R.disorder_identity = id.str();
    return R;
}

} // namespace heom::mc
