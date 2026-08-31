// Host-only unit tests for the MC static-disorder shell.
// Scope here: the apply_disorder hook (no GPU, no SystemContext). Reuses the
// lightweight TEST/ASSERT style from test_core.cpp (in-house over Catch2).
#include "types.hpp"
#include "heom_config.hpp"
#include "hamiltonian_expansion.hpp"
#include "mc/mc_disorder.hpp"
#include "mc/mc_realizations.hpp"
#include "mc/mc_partial.hpp"   // Accumulator
#include "mc/mc_queue.hpp"
#include "mc/mc_driver.hpp"   // mc_merge_key (inline, header-only)

#include <Eigen/Dense>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { ++tests_run; std::cout << "  " << name << "... " << std::flush; } while(0)
#define PASS() do { ++tests_passed; std::cout << "PASS\n"; } while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { std::cout << "FAIL\n    " #cond " is false\n"; return; } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    double _a = (a), _b = (b); \
    if (std::abs(_a - _b) > (tol)) { \
        std::cout << "FAIL\n    expected " << _b << " got " << _a \
                  << " (diff=" << std::abs(_a-_b) << ")\n"; return; } \
} while(0)

#define ASSERT_THROWS(expr) do { \
    bool caught = false; \
    try { expr; } catch (const std::exception&) { caught = true; } \
    if (!caught) { std::cout << "FAIL\n    expected exception, none thrown\n"; return; } \
} while(0)

using HEOMConfig::Config;

// Minimal dynamics yaml with an n×n site Hamiltonian: distinct diagonal energies
// (12000 + 100*i cm⁻¹) and tiny symmetric off-diagonals so eigen/expansion have
// something non-trivial to chew on. Basis is applied by the caller via
// apply_basis(); load_yaml itself parses in whatever `basis` we write here.
static std::string write_mc_yaml(int n) {
    const std::string path = "/tmp/test_mc_config.yaml";
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
      << "filter:\n  enabled: true\n  tolerance: 1.0e-7\n  start_step: 10\n"
      << "  warmup_factor: 0.001\n"
      << "dynamics:\n  integrator: rk4\n  dt: 2.0\n  n_steps: 100\n  output_interval: 10\n"
      << "  print_interval: 100\n"
      << "  sink:\n    enabled: false\n    target_site: 1\n    rate: 0.001\n"
      << "  initial_state:\n    type: \"site_excitation\"\n    excited_site: 1\n"
      << "  output:\n    save_path: \"/tmp/\"\n    use_timestamp: false\n";
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

#ifndef MC_GEN_SCRIPT
#define MC_GEN_SCRIPT ""
#endif

static std::string gen_realizations(int n, int s, double fwhm, int seed) {
    const std::string out = "/tmp/test_mc_realizations.h5";
    const std::string cmd = std::string("python3 ") + MC_GEN_SCRIPT +
        " --out " + out + " --n-node " + std::to_string(n) +
        " --n-site " + std::to_string(s) + " --fwhm " + std::to_string(fwhm) +
        " --seed " + std::to_string(seed) + " --method iid";
    if (std::system(cmd.c_str()) != 0) {
        throw std::runtime_error("generator invocation failed: " + cmd);
    }
    return out;
}

static void test_load_realizations_roundtrip() {
    TEST("load_realizations: reads generator 8x7 file");
    const std::string f = gen_realizations(8, 7, 100.0, 42);
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

static void test_load_realizations_rejects_unknown_estimator() {
    TEST("load_realizations: unknown estimator throws");
    const std::string f = gen_realizations(4, 7, 100.0, 1);
    // Flip the estimator attr to a value neither mean nor quadrature.
    const std::string flip = "python3 -c \"import h5py,sys; "
        "f=h5py.File(sys.argv[1],'a'); f.attrs['estimator']='bogus'; f.close()\" " + f;
    if (std::system(flip.c_str()) != 0) {
        std::cout << "SKIP (could not rewrite attr)\n"; --tests_run; return;
    }
    ASSERT_THROWS(heom::mc::load_realizations(f));
    PASS();
}

// --- Quadrature loader and scaled accumulator --------------------

static std::string gen_quadrature(int s, double fwhm, int level) {
    const std::string out = "/tmp/test_mc_quad.h5";
    const std::string cmd = std::string("python3 ") + MC_GEN_SCRIPT +
        " --out " + out + " --n-site " + std::to_string(s) +
        " --fwhm " + std::to_string(fwhm) +
        " --method sparse_gh --level " + std::to_string(level);
    if (std::system(cmd.c_str()) != 0) {
        throw std::runtime_error("quad generator invocation failed: " + cmd);
    }
    return out;
}

static void test_load_realizations_quadrature() {
    TEST("load_realizations: accepts quadrature, identity keys on level");
    const std::string f = gen_quadrature(3, 100.0, 2);
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
    const std::string qpath = "/tmp/test_mc_queue.bin";
    std::remove(qpath.c_str());
    const int N = 100;
    heom::mc::Queue qa(qpath, 0, N);   // 0-based half-open [0, N)
    heom::mc::Queue qb(qpath, 0, N);
    std::vector<int> seen(N, 0);
    int total = 0, r;
    // Alternate the two instances to mimic interleaved workers.
    bool useA = true;
    while (true) {
        r = (useA ? qa : qb).claim();
        useA = !useA;
        if (r < 0) {
            // Both drained? claim once more on the other to be sure.
            if ((useA ? qa : qb).claim() < 0) break;
            continue;
        }
        ASSERT_TRUE(r >= 0 && r < N);
        ++seen[r];
        ++total;
    }
    ASSERT_TRUE(total == N);
    for (int i = 0; i < N; ++i) ASSERT_TRUE(seen[i] == 1);
    PASS();
}

// The incremental batch: a non-zero start interval hands out exactly its ids.
// CLI --range 201 300 (1-based) → driver ClaimSource(begin0=200, end0=300).
static void test_queue_interval_start() {
    TEST("Queue: [200,300) hands out 200..299 exactly");
    const std::string qpath = "/tmp/test_mc_queue_interval.bin";
    std::remove(qpath.c_str());
    heom::mc::Queue q(qpath, 200, 300);
    int count = 0, r, first = -1, last = -1;
    while ((r = q.claim()) >= 0) {
        if (first < 0) first = r;
        last = r;
        ASSERT_TRUE(r >= 200 && r < 300);
        ++count;
    }
    ASSERT_TRUE(count == 100);   // 300 - 200
    ASSERT_TRUE(first == 200);   // seeds at begin0, in order for a lone worker
    ASSERT_TRUE(last == 299);    // last id < end0
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

int main() {
    std::cout << "=== MC static-disorder unit tests ===\n";
    test_zero_offset_bit_identical();
    test_double_excited_picks_up_sum();
    test_size_mismatch_throws();
    test_load_realizations_roundtrip();
    test_load_realizations_rejects_unknown_estimator();
    test_load_realizations_quadrature();
    test_accumulator_mean_bare_sum();
    test_accumulator_quadrature_weighted_sum();
    test_queue_partitions_without_gaps_or_dups();
    test_queue_interval_start();
    test_merge_key_stable_and_sensitive();
    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
