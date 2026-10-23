// Host-only unit tests for the MC static-disorder shell.
// Scope here: the apply_disorder hook (no GPU, no SystemContext). Reuses the
// shared tests/harness.hpp (in-house over Catch2).
#include "types.hpp"
#include "heom_config.hpp"
#include "hamiltonian_expansion.hpp"
#include "mc/mc_disorder.hpp"
#include "mc/mc_realizations.hpp"
#include "mc/mc_partial.hpp"   // Accumulator
#include "mc/mc_queue.hpp"
#include "mc/mc_driver.hpp"   // mc_merge_key (inline, header-only)
#include "harness.hpp"

#include <Eigen/Dense>
#include <hdf5.h>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using HEOMConfig::Config;

// Minimal dynamics yaml with an n×n site Hamiltonian: distinct diagonal energies
// (12000 + 100*i cm⁻¹) and tiny symmetric off-diagonals so eigen/expansion have
// something non-trivial to chew on. Basis is applied by the caller via
// apply_basis(); load_yaml itself parses in whatever `basis` we write here.
static std::string write_mc_yaml(int n) {
    const std::string path = heom_test::tmp_dir() + "/test_mc_config.yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: " << n << "\n"
      << "  basis: \"excited_only\"\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n";
    for (int i = 0; i < n; ++i) {
        f << "    - [";
        for (int j = 0; j < n; ++j) {
            double v = (i == j) ? (12000.0 + 100.0 * i) : ((i < j) ? -5.0 : -5.0);
            f << v << (j + 1 < n ? ", " : "");
        }
        f << "]\n";
    }
    f << "  w0: 12000.0\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 35.0\n  gamma_inverse: 100.0\n"
      << "  expansion_method: matsubara\n  n_extra_poles: 1\n  L_cutoff: 1\n"
      << "filter:\n  enabled: true\n  tolerance: 1.0e-7\n"
      << "dynamics:\n  integrator: rk4\n  dt: 2.0\n  n_steps: 100\n  output_interval: 10\n"
      << "  print_interval: 100\n"
      << "  sink:\n    enabled: false\n    target_site: 1\n    rate: 0.001\n"
      << "  initial_state:\n    type: \"site_excitation\"\n    excited_site: 1\n"
      << "  output:\n    save_path: \"" << heom_test::tmp_dir() << "/\"\n    use_timestamp: false\n";
    f.close();
    return path;
}

static Config load_fmo_like(int n, BasisType basis) {
    Config cfg = Config::load_yaml(write_mc_yaml(n));
    HEOMConfig::apply_basis(cfg, basis);
    return cfg;
}

// σ=0 ⇒ the +0.0 injection plus a deterministic re-expansion must leave the
// full double-excited Hamiltonian bit-identical.
static void test_zero_offset_bit_identical() {
    TEST("apply_disorder: zero offset leaves DOUBLE_EXCITED H bit-identical");
    Config cfg = load_fmo_like(7, BasisType::DOUBLE_EXCITED);
    Eigen::MatrixXcd before = cfg.physics.hamiltonian;
    heom::mc::apply_disorder(cfg, std::vector<double>(7, 0.0));
    ASSERT_TRUE(cfg.physics.hamiltonian == before);
    PASS();
}

// Double-excited diagonal |ij⟩ must pick up δ_i + δ_j.
static void test_double_excited_picks_up_sum() {
    TEST("apply_disorder: double-excited diagonal picks up delta_i + delta_j");
    Config c2 = load_fmo_like(3, BasisType::DOUBLE_EXCITED);
    std::vector<double> d = {10.0, 20.0, 30.0};  // cm⁻¹
    Eigen::MatrixXcd h0 = c2.physics.hamiltonian;
    heom::mc::apply_disorder(c2, d);
    DoubleExcitedIndex idx(3);
    const int idx12 = idx.double_index(1, 2);
    double got = (c2.physics.hamiltonian(idx12, idx12) - h0(idx12, idx12)).real();
    ASSERT_NEAR(got, (10.0 + 20.0) * CM_TO_AU, 1e-18);
    // Single-excitation diagonal |1⟩ picks up δ_1 only.
    const int idx1 = idx.single_index(1);
    double got1 = (c2.physics.hamiltonian(idx1, idx1) - h0(idx1, idx1)).real();
    ASSERT_NEAR(got1, 10.0 * CM_TO_AU, 1e-18);
    PASS();
}

// Guard: wrong offset vector length must throw (n_states mismatch).
static void test_size_mismatch_throws() {
    TEST("apply_disorder: size != n_states throws");
    Config cfg = load_fmo_like(3, BasisType::WITH_GROUND);
    ASSERT_THROWS(heom::mc::apply_disorder(cfg, std::vector<double>(2, 0.0)));
    PASS();
}

// --- Disorder-node reader (round-trips the Python generator) ----------

// Empty when CMake found no heom-tools checkout (solver-only tree): the tests
// below that need the generator then SKIP instead of failing.
#ifndef MC_GEN_SCRIPT
#define MC_GEN_SCRIPT ""
#endif

// POSIX single-quotes a path for std::system: the heom-tools checkout or
// TMPDIR may contain spaces (e.g. "~/My Files/..."), which the shell
// would otherwise split into separate arguments.
static std::string shq(const std::string& s) {
    std::string r = "'";
    for (char c : s) r += (c == '\'') ? std::string("'\\''") : std::string(1, c);
    return r + "'";
}

// Runs the generator into tmp_dir()/name; "" (with the command printed) on failure.
static std::string run_generator(const std::string& name, const std::string& args) {
    const std::string out = heom_test::tmp(name);
    const std::string cmd = "python3 " + shq(MC_GEN_SCRIPT) + " --out " + shq(out) + args;
    if (std::system(cmd.c_str()) != 0) {
        std::cout << "\n    generator failed: " << cmd << "\n    ";
        return "";
    }
    return out;
}

// Declares `var` = generator output, or leaves the test (SKIP / FAIL).
#define GENERATE(var, name, args) \
    if (!*MC_GEN_SCRIPT) SKIP("heom-tools not configured"); \
    const std::string var = run_generator(name, args); \
    ASSERT_TRUE(!var.empty())

static std::string iid_args(int n, int s, double fwhm, int seed) {
    return " --n-node " + std::to_string(n) + " --n-site " + std::to_string(s) +
           " --fwhm " + std::to_string(fwhm) + " --seed " + std::to_string(seed) +
           " --method iid";
}

static void test_load_realizations_roundtrip() {
    TEST("load_realizations: reads generator 8x7 file");
    GENERATE(f, "test_mc_realizations.h5", iid_args(8, 7, 100.0, 42));
    const heom::mc::RealizationSet R = heom::mc::load_realizations(f);
    ASSERT_TRUE(R.n_node == 8);
    ASSERT_TRUE(R.n_site == 7);
    ASSERT_TRUE(static_cast<int>(R.row(0).size()) == 7);
    ASSERT_TRUE(R.estimator == heom::mc::Estimator::Mean);
    double wsum = 0.0;
    for (double w : R.weight) wsum += w;
    ASSERT_NEAR(wsum, 1.0, 1e-12);
    ASSERT_TRUE(!R.generator.empty());
    ASSERT_TRUE(!R.disorder_identity.empty());
    PASS();
}

// The σ text inside disorder_identity is an on-disk merge key. Lock the
// exact production string (FWHM 100 cm⁻¹ ⇒ σ = 42.46609… ⇒ "42.4661", six
// significant digits, '.' decimal point) so nobody "improves" the formatting
// and orphans every existing shard.
static void test_identity_sigma_format_is_frozen() {
    TEST("disorder_identity: sigma keeps the 6-digit on-disk format");
    GENERATE(f, "test_mc_realizations.h5", iid_args(4, 3, 100.0, 7));
    const heom::mc::RealizationSet R = heom::mc::load_realizations(f);
    const std::string want = "|sigma=42.4661,42.4661,42.4661";
    const bool ends_with =
        R.disorder_identity.size() >= want.size() &&
        R.disorder_identity.compare(R.disorder_identity.size() - want.size(),
                                    want.size(), want) == 0;
    if (!ends_with) {
        std::cout << "FAIL\n    identity = '" << R.disorder_identity << "'\n";
        return;
    }
    ASSERT_TRUE(R.disorder_identity.find("seed=7") != std::string::npos);
    PASS();
}

static void test_load_realizations_rejects_unknown_estimator() {
    TEST("load_realizations: unknown estimator throws");
    GENERATE(f, "test_mc_realizations.h5", iid_args(4, 7, 100.0, 1));
    // Flip the estimator attr to a value neither mean nor quadrature.
    const std::string flip = "python3 -c \"import h5py,sys; "
        "f=h5py.File(sys.argv[1],'a'); f.attrs['estimator']='bogus'; f.close()\" " + shq(f);
    if (std::system(flip.c_str()) != 0)
        SKIP("could not rewrite attr");
    ASSERT_THROWS(heom::mc::load_realizations(f));
    PASS();
}

// Attribute shapes are checked against the data, not trusted. A sigma of the
// wrong length used to be accepted (into disorder_identity); an array where
// the scalar master_seed belongs was read straight into one uint64.
static void test_load_realizations_rejects_bad_attr_shapes() {
    TEST("load_realizations: sigma_cm length / array master_seed throw");
    for (const char* edit : {"f.attrs['sigma_cm']=np.ones(3)",
                             "f.attrs['master_seed']=np.arange(4,dtype=np.uint64)"}) {
        GENERATE(f, "test_mc_bad_attr.h5", iid_args(4, 7, 100.0, 1));
        ASSERT_TRUE(heom::mc::load_realizations(f).n_site == 7);
        const std::string cmd = std::string("python3 -c \"import h5py,sys,numpy as np; "
            "f=h5py.File(sys.argv[1],'a'); ") + edit + "; f.close()\" " + shq(f);
        if (std::system(cmd.c_str()) != 0) SKIP("could not rewrite attr");
        ASSERT_THROWS(heom::mc::load_realizations(f));
    }
    PASS();
}

// --- Quadrature loader and scaled accumulator --------------------

static void test_load_realizations_quadrature() {
    TEST("load_realizations: accepts quadrature, identity keys on level");
    GENERATE(f, "test_mc_quad.h5", " --n-site 3 --fwhm 100 --method sparse_gh --level 2");
    const heom::mc::RealizationSet R = heom::mc::load_realizations(f);
    ASSERT_TRUE(R.estimator == heom::mc::Estimator::Quadrature);
    ASSERT_TRUE(R.n_site == 3);
    double wsum = 0.0;
    for (double w : R.weight) wsum += w;
    ASSERT_NEAR(wsum, 1.0, 1e-12);
    ASSERT_TRUE(R.disorder_identity.find("sparse_gh@v1") != std::string::npos);
    ASSERT_TRUE(R.disorder_identity.find("level=2") != std::string::npos);
    ASSERT_TRUE(R.disorder_identity.find("seed=") == std::string::npos);
    PASS();
}

using cd = std::complex<double>;

static void test_accumulator_mean_bare_sum() {
    TEST("Accumulator(mean): bare sum, mean()=Σ/count, keeps |O|²");
    heom::mc::Accumulator acc;
    acc.init({2}, heom::mc::Estimator::Mean);
    acc.add({cd(1.0, 0.0), cd(2.0, 0.0)}, 1.0, 0);
    acc.add({cd(3.0, 0.0), cd(4.0, 0.0)}, 1.0, 1);
    ASSERT_TRUE(acc.count == 2);
    const auto m = acc.mean();
    ASSERT_NEAR(m[0].real(), 2.0, 1e-15);            // (1+3)/2
    ASSERT_NEAR(m[1].real(), 3.0, 1e-15);            // (2+4)/2
    ASSERT_TRUE(acc.sum_obs_sq.size() == 2);
    ASSERT_NEAR(acc.sum_obs_sq[0], 1.0 + 9.0, 1e-12);  // |1|²+|3|²
    PASS();
}

static void test_accumulator_quadrature_weighted_sum() {
    TEST("Accumulator(quadrature): Σ w·O, no sum_obs_sq buffer");
    heom::mc::Accumulator acc;
    acc.init({2}, heom::mc::Estimator::Quadrature);
    acc.add({cd(1.0, 0.0), cd(2.0, 0.0)}, 0.25, 0);
    acc.add({cd(3.0, 0.0), cd(4.0, 0.0)}, 0.75, 1);
    ASSERT_NEAR(acc.sum_obs[0].real(), 0.25 * 1.0 + 0.75 * 3.0, 1e-15);  // 2.5
    ASSERT_NEAR(acc.sum_obs[1].real(), 0.25 * 2.0 + 0.75 * 4.0, 1e-15);  // 3.5
    ASSERT_TRUE(acc.sum_obs_sq.empty());             // skipped for quadrature
    PASS();
}

// --- Task5: flock queue claim-loop -------------------------------------------

static void test_queue_partitions_without_gaps_or_dups() {
    TEST("Queue: two claimers cover [0,100) exactly once");
    const std::string qpath = heom_test::tmp_dir() + "/test_mc_queue.bin";
    std::remove(qpath.c_str());
    const int N = 100;
    heom::mc::Queue qa(qpath, 0, N);   // 0-based half-open [0, N)
    heom::mc::Queue qb(qpath, 0, N);
    std::vector<int> seen(N, 0);
    int total = 0;
    // Alternate the two instances to mimic interleaved workers.
    bool useA = true;
    while (true) {
        const auto r = (useA ? qa : qb).claim();
        useA = !useA;
        if (!r) {
            // Both drained? claim once more on the other to be sure.
            if (!(useA ? qa : qb).claim()) break;
            continue;
        }
        ASSERT_TRUE(*r >= 0 && *r < N);
        ++seen[*r];
        ++total;
    }
    ASSERT_TRUE(total == N);
    for (int i = 0; i < N; ++i) ASSERT_TRUE(seen[i] == 1);
    PASS();
}

// The incremental batch: a non-zero start interval hands out exactly its ids.
// CLI --r-begin 201 --r-end 300 (1-based) → driver Queue(begin0=200, end0=300).
static void test_queue_interval_start() {
    TEST("Queue: [200,300) hands out 200..299 exactly");
    const std::string qpath = heom_test::tmp_dir() + "/test_mc_queue_interval.bin";
    std::remove(qpath.c_str());
    heom::mc::Queue q(qpath, 200, 300);
    int count = 0;
    std::optional<int> first, last;
    while (const auto r = q.claim()) {
        if (!first) first = *r;
        last = *r;
        ASSERT_TRUE(*r >= 200 && *r < 300);
        ++count;
    }
    ASSERT_TRUE(count == 100);   // 300 - 200
    ASSERT_TRUE(first == 200);   // seeds at begin0, in order for a lone worker
    ASSERT_TRUE(last == 299);    // last id < end0
    PASS();
}

// A queue file that is neither empty nor one whole int64 is corruption,
// not a counter. The old code checked only got < 0, so a 3-byte file produced
// `next` with five bytes of stack garbage and claim() "succeeded".
static void test_queue_rejects_corrupt_file() {
    TEST("Queue: short read / negative counter throw, never claim");
    const std::string qpath = heom_test::tmp_dir() + "/test_mc_queue_corrupt.bin";

    std::remove(qpath.c_str());
    { std::ofstream f(qpath, std::ios::binary); f.write("\x01\x02\x03", 3); }
    {
        heom::mc::Queue q(qpath, 0, 10);
        ASSERT_THROWS(q.claim());
    }

    std::remove(qpath.c_str());
    { std::int64_t neg = -1; std::ofstream f(qpath, std::ios::binary);
      f.write(reinterpret_cast<const char*>(&neg), sizeof(neg)); }
    {
        heom::mc::Queue q(qpath, 0, 10);
        ASSERT_THROWS(q.claim());
    }

    // A whole int64 past end0 is not corruption: it is a drained queue.
    std::remove(qpath.c_str());
    { std::int64_t done = 10; std::ofstream f(qpath, std::ios::binary);
      f.write(reinterpret_cast<const char*>(&done), sizeof(done)); }
    {
        heom::mc::Queue q(qpath, 0, 10);
        ASSERT_TRUE(!q.claim().has_value());
    }
    std::remove(qpath.c_str());
    PASS();
}

// The merge key must be stable for identical (config_hash, identity) and differ
// when either input changes — that stability is what lets shards from different
// timestamps/intervals group and merge across incremental batches.
static void test_merge_key_stable_and_sensitive() {
    TEST("mc_merge_key: stable on same params, differs on change");
    const std::string k1 = heom::mc::mc_merge_key("abc123", "iid_gauss@v1|seed=7|sigma=100");
    const std::string k2 = heom::mc::mc_merge_key("abc123", "iid_gauss@v1|seed=7|sigma=100");
    ASSERT_TRUE(k1 == k2);              // reproducible
    ASSERT_TRUE(k1.size() == 16);      // 16 hex chars
    const std::string k_cfg  = heom::mc::mc_merge_key("abc124", "iid_gauss@v1|seed=7|sigma=100");
    const std::string k_iden = heom::mc::mc_merge_key("abc123", "iid_gauss@v1|seed=8|sigma=100");
    ASSERT_TRUE(k_cfg != k1);          // config change ⇒ new key
    ASSERT_TRUE(k_iden != k1);         // disorder change ⇒ new key
    PASS();
}


// --- /mc close-out: r_done larger than an object-header message ---------------
//
// r_done is one int64 per folded realization. In the earliest HDF5 file format
// (what H5Fcreate(…, H5P_DEFAULT) gives with libhdf5 1.10) an attribute must fit
// in a 64 KiB object-header message, so > ~8190 realizations in one shard failed (8000 ok, 8191 fails)
// in append_mc_provenance. Shard files here are created exactly as the solver
// creates them (default property lists).
static bool mc_rdone_roundtrip(long long n, unsigned& superblock_before,
                               unsigned& superblock_after) {
    const std::string path = heom_test::tmp("mc_rdone_" + std::to_string(n) + ".h5");
    hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0) return false;
    H5F_info2_t info;
    H5Fget_info2(f, &info); superblock_before = info.super.version;
    H5Fclose(f);

    heom::mc::Accumulator acc;
    acc.init({2}, heom::mc::Estimator::Mean);
    for (long long r = 0; r < n; ++r)
        acc.add({cd(1.0, 0.0), cd(2.0, 0.0)}, 1.0, static_cast<int>(r));
    heom::mc::append_mc_provenance(path, acc, "0123456789abcdef",
                                   "iid_gauss@v1|seed=1|sigma=50.0000");

    f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) return false;
    H5Fget_info2(f, &info); superblock_after = info.super.version;
    hid_t a = H5Aopen_by_name(f, "/mc", "r_done", H5P_DEFAULT, H5P_DEFAULT);
    hid_t sp = H5Aget_space(a);
    const hssize_t npts = H5Sget_simple_extent_npoints(sp);
    std::vector<long long> back(static_cast<size_t>(npts > 0 ? npts : 0));
    const herr_t rd = npts > 0 ? H5Aread(a, H5T_NATIVE_LLONG, back.data()) : -1;
    long long count = -1;
    hid_t d = H5Dopen2(f, "/mc/count", H5P_DEFAULT);
    H5Dread(d, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &count);
    H5Dclose(d); H5Sclose(sp); H5Aclose(a); H5Fclose(f);
    if (rd < 0 || npts != n || count != n) return false;
    for (long long r = 0; r < n; ++r) if (back[static_cast<size_t>(r)] != r) return false;
    return true;
}

static void test_r_done_small() {
    TEST("append_mc_provenance: 4 realizations, r_done round-trips");
    unsigned sb0 = 99, sb1 = 99;
    ASSERT_TRUE(mc_rdone_roundtrip(4, sb0, sb1));
    ASSERT_EQ(sb0, sb1);   // the superblock is not rewritten
    PASS();
}

static void test_r_done_past_object_header_limit() {
    TEST("append_mc_provenance: 8192 and 100000 realizations (> 64 KiB r_done)");
    for (long long n : {8192LL, 100000LL}) {
        unsigned sb0 = 99, sb1 = 99;
        ASSERT_TRUE(mc_rdone_roundtrip(n, sb0, sb1));
        ASSERT_EQ(sb0, sb1);
    }
    PASS();
}

int main() {
    std::cout << "=== MC static-disorder unit tests ===\n";
    test_zero_offset_bit_identical();
    test_double_excited_picks_up_sum();
    test_size_mismatch_throws();
    test_load_realizations_roundtrip();
    test_identity_sigma_format_is_frozen();
    test_load_realizations_rejects_unknown_estimator();
    test_load_realizations_rejects_bad_attr_shapes();
    test_load_realizations_quadrature();
    test_accumulator_mean_bare_sum();
    test_accumulator_quadrature_weighted_sum();
    test_queue_partitions_without_gaps_or_dups();
    test_queue_interval_start();
    test_queue_rejects_corrupt_file();
    test_merge_key_stable_and_sensitive();
    test_r_done_small();
    test_r_done_past_object_header_limit();
    return heom_test_summary("test_mc");
}
