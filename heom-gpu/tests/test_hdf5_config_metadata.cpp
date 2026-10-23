#include "heom_config.hpp"
#include "output/hdf5_writer.hpp"
#include "output/hdf5_metadata.hpp"
#include "harness.hpp"

#include <hdf5.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

std::string read_string_attr(hid_t loc, const char* name) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    require(attr >= 0, std::string("missing attr: ") + name);
    hid_t type = H5Aget_type(attr);
    require(type >= 0, std::string("cannot get attr type: ") + name);

    char* value = nullptr;
    require(H5Aread(attr, type, &value) >= 0, std::string("cannot read attr: ") + name);
    std::string out = value ? value : "";
    if (value) {
        H5free_memory(value);
    }

    H5Tclose(type);
    H5Aclose(attr);
    return out;
}

std::string read_string_dataset(hid_t loc, const char* name) {
    hid_t dataset = H5Dopen2(loc, name, H5P_DEFAULT);
    require(dataset >= 0, std::string("missing dataset: ") + name);
    hid_t type = H5Dget_type(dataset);
    require(type >= 0, std::string("cannot get dataset type: ") + name);

    char* value = nullptr;
    require(H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) >= 0,
            std::string("cannot read dataset: ") + name);
    std::string out = value ? value : "";
    if (value) {
        H5free_memory(value);
    }

    H5Tclose(type);
    H5Dclose(dataset);
    return out;
}

}  // namespace

// Body throws on the first broken expectation (the `require` helpers above);
// the harness turns that into a FAIL with the message.
static void test_group_layout_metadata_body() {
    const std::string path = heom_test::tmp("test_heom_hdf5_config_metadata.h5");
    std::remove(path.c_str());

    HEOMConfig::Config cfg{};
    cfg.yaml_path = "/tmp/run_config.yaml";
    cfg.config_yaml =
        "system:\n"
        "  n_states: 2\n"
        "dynamics:\n"
        "  n_steps: 1\n";
    cfg.config_hash = "0123456789abcdef";

    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    require(file >= 0, "cannot create HDF5 test file");

    // Root identity is the single place that emits config_hash + task.
    heom::hdf5::write_root_identity(file, cfg, "dynamics");

    // /parameters owns the YAML snapshot — config_yaml is the user-input
    // source of truth, so it must travel with the parameters group, not
    // be scattered across the root or /provenance.
    hid_t params = heom::hdf5::create_group(file, "/parameters");
    heom::hdf5::write_string_dataset(params, "config_yaml", cfg.config_yaml);
    require(H5Gclose(params) >= 0, "cannot close /parameters");

    // /provenance owns the original on-disk YAML location (debug-only info).
    hid_t prov = heom::hdf5::create_group(file, "/provenance");
    heom::hdf5::write_string_attr(prov, "config_path", cfg.yaml_path);
    require(H5Gclose(prov) >= 0, "cannot close /provenance");

    require(H5Fclose(file) >= 0, "cannot close HDF5 test file after write");

    // ---- Verify ----
    file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    require(file >= 0, "cannot reopen HDF5 test file");

    require(read_string_attr(file, "task") == "dynamics", "task mismatch");
    require(read_string_attr(file, "config_hash") == cfg.config_hash,
            "config_hash mismatch");

    params = H5Gopen2(file, "/parameters", H5P_DEFAULT);
    require(params >= 0, "missing /parameters group");
    require(read_string_dataset(params, "config_yaml") == cfg.config_yaml,
            "config_yaml dataset mismatch");
    require(H5Gclose(params) >= 0, "cannot close /parameters after read");

    prov = H5Gopen2(file, "/provenance", H5P_DEFAULT);
    require(prov >= 0, "missing /provenance group");
    require(read_string_attr(prov, "config_path") == cfg.yaml_path,
            "config_path mismatch");
    require(H5Gclose(prov) >= 0, "cannot close /provenance after read");

    // ---- validate_timestep_bounds smoke-test (unchanged contract) ----
    bool caught = false;
    try {
        heom::hdf5::validate_timestep_bounds(
            /*N=*/3, /*expected_N=*/2,
            /*current_step=*/0, /*n_records=*/1);
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, "expected HDF5 bounds check to reject N mismatch");

    caught = false;
    try {
        heom::hdf5::validate_timestep_bounds(
            /*N=*/2, /*expected_N=*/2,
            /*current_step=*/1, /*n_records=*/1);
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, "expected HDF5 bounds check to reject step overflow");

    heom::hdf5::validate_timestep_bounds(
        /*N=*/2, /*expected_N=*/2,
        /*current_step=*/0, /*n_records=*/1);

    require(H5Fclose(file) >= 0, "cannot close HDF5 test file after read");
    std::remove(path.c_str());
}

static void test_group_layout_metadata() {
    TEST("HDF5 group layout metadata (root identity / parameters / provenance)");
    try {
        test_group_layout_metadata_body();
    } catch (const std::exception& e) {
        std::cout << "FAIL\n    " << e.what() << "\n";
        return;
    }
    PASS();
}

int main() {
    test_group_layout_metadata();
    return heom_test_summary("test_hdf5_config_metadata");
}
