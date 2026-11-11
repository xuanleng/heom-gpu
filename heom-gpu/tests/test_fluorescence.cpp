// Fluorescence primitives and gates.
//
//   relative_distance : exact zeros, an exact hand value, a host recompute
//                       that includes the mixed-build Kahan carry, mismatch.
//   dress_ados (G3)   : monomer closed form Π(a_k/√s_k)^{n_k}/√n_k! per ADO,
//                       first tier summing to −2λσ₀, RK4 fixed point; J=0
//                       dimer one-step invariance; equilibrate stops at once
//                       from the dressed monomer.
//   bare_boltzmann_ee : embedding, bare (reorganisation-free) energies.
//   equilibrate (G7)  : not-converged text carries δ / r / ε; bad intervals.

#include "harness.hpp"
#include "spectra/absorption.hpp"
#include "spectra/dipole.hpp"
#include "spectra/fluorescence.hpp"
#include "heom_config.hpp"
#include "heom_trajectory.hpp"
#include "system_context.hpp"
#include "types.hpp"

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <fstream>
#include <string>
#include <vector>

namespace {

using cplx = std::complex<double>;

constexpr bool   kMP       = heom::kMixedPrecision;
constexpr double kAdoTol   = kMP ? 1e-5 : 1e-12;   // G3: FP64 1e-12, MP 1e-5
constexpr double kFixedTol = kMP ? 1e-5 : 1e-12;   // one τ_c from the dressed start
constexpr double kStepTol  = kMP ? 1e-5 : 1e-14;   // one RK4 step, J=0 dimer

// Monomer, Drude + Padé K=2 + 340 / 1180 cm⁻¹ modes, 77 K, L=4 (G3). Carries
// both task blocks on the same time grid (G1 compares the two).
std::string write_monomer_vib_yaml() {
    const std::string path = heom_test::tmp("fl_monomer_vib.yaml");
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 1\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12000.0]\n"
      << "  w0: 12000.0\n"
      << "  dipoles:\n"
      << "    - [1.0, 0.0, 0.0]\n"
      << "  strengths: [1.0]\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 70.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: pade\n"
      << "  n_extra_poles: 2\n"
      << "  L_cutoff: 4\n"
      << "  vibrational_modes:\n"
      << "    - {omega_cm: 340.0, huang_rhys: 0.05, gamma_inverse: 354.0}\n"
      << "    - {omega_cm: 1180.0, huang_rhys: 0.02, gamma_inverse: 354.0}\n"
      << "filter:\n"
      << "  enabled: false\n"
      << "absorption:\n"
      << "  integrator: rk4\n"
      << "  dt: 1\n"
      << "  n_steps: 10\n"
      << "  output_interval: 1\n"
      << "  print_interval: 0\n"
      << "  output:\n"
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
      << "    filename: \"fl_monomer_vib_abs\"\n"
      << "    use_timestamp: false\n"
      << "fluorescence:\n"
      << "  integrator: rk4\n"
      << "  dt: 1\n"
      << "  n_steps: 10\n"
      << "  output_interval: 1\n"
      << "  print_interval: 0\n"
      << "  equilibration:\n"
      << "    t_max: 500\n"
      << "    check_interval: 50\n"
      << "    tolerance: 1.0e-6\n"
      << "  output:\n"
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
      << "    filename: \"fl_monomer_vib\"\n"
      << "    use_timestamp: false\n";
    return path;
}

// Coupled dimer, 3-D dipoles (all three directions active), Drude only.
std::string write_coupled_dimer_yaml() {
    const std::string path = heom_test::tmp("fl_dimer_j100.yaml");
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100.0, 100.0]\n"
      << "    - [100.0, 12300.0]\n"
      << "  w0: 12200.0\n"
      << "  dipoles:\n"
      << "    - [1.0, 0.2, 0.0]\n"
      << "    - [0.3, -0.8, 0.5]\n"
      << "  strengths: [1.0, 1.0]\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 70.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: pade\n"
      << "  n_extra_poles: 1\n"
      << "  L_cutoff: 3\n"
      << "filter:\n"
      << "  enabled: false\n"
      << "fluorescence:\n"
      << "  integrator: rk4\n"
      << "  dt: 2\n"
      << "  n_steps: 30\n"
      << "  output_interval: 1\n"
      << "  print_interval: 0\n"
      << "  equilibration:\n"
      << "    t_max: 4000\n"
      << "    check_interval: 50\n"
      << "    tolerance: 1.0e-3\n"
      << "  output:\n"
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
      << "    filename: \"fl_dimer_j100\"\n"
      << "    use_timestamp: false\n";
    return path;
}

// Two uncoupled pigments (J = 0), each with Drude + Matsubara K=1 + one mode.
std::string write_uncoupled_dimer_yaml() {
    const std::string path = heom_test::tmp("fl_dimer_j0.yaml");
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100.0, 0.0]\n"
      << "    - [0.0, 12300.0]\n"
      << "  w0: 12200.0\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 70.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: matsubara\n"
      << "  n_extra_poles: 1\n"
      << "  L_cutoff: 3\n"
      << "  vibrational_modes:\n"
      << "    - {omega_cm: 340.0, huang_rhys: 0.05, gamma_inverse: 354.0}\n"
      << "filter:\n"
      << "  enabled: false\n"
      << "dynamics:\n"
      << "  integrator: rk4\n"
      << "  dt: 1.0\n"
      << "  n_steps: 1\n"
      << "  output_interval: 1\n"
      << "  print_interval: 0\n"
      << "  initial_state:\n"
      << "    type: \"site_excitation\"\n"
      << "    excited_site: 1\n"
      << "  output:\n"
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
      << "    filename: \"fl_dimer_j0\"\n"
      << "    use_timestamp: false\n";
    return path;
}

HEOMConfig::Config load_monomer() {
    auto cfg = HEOMConfig::Config::load_yaml(write_monomer_vib_yaml());
    HEOMConfig::resolve_fluorescence_config(cfg);
    return cfg;
}

heom::spectra::DipoleOperator dipole_of(const HEOMConfig::Config& cfg) {
    return heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths, cfg.physics.basis, cfg.physics.n_sites);
}

// max |a − b| over rho_xyz; 1e30 on a shape mismatch.
double max_diff(const heom::spectra::LinearResponseResult& a,
                const heom::spectra::LinearResponseResult& b) {
    if (a.rho_xyz.size() != b.rho_xyz.size() || a.rho_xyz.empty()) return 1e30;
    double d = 0.0;
    for (size_t k = 0; k < a.rho_xyz.size(); ++k)
        d = std::max(d, std::abs(a.rho_xyz[k] - b.rho_xyz[k]));
    return d;
}

const heom::ManifoldBlock kMonoEE{/*row_off=*/1, /*n_row=*/1, /*col_off=*/1, /*n_col=*/1};

Eigen::MatrixXcd monomer_excited(double p) {
    Eigen::MatrixXcd r = Eigen::MatrixXcd::Zero(2, 2);
    r(1, 1) = p;
    return r;
}

// ρ − carry of a snapshot, as doubles.
std::vector<cplx> corrected(const DeviceHierarchySnapshot& s) {
    const size_t n = s.rho.size();
    std::vector<heom::complex_t> r(n), c;
    s.rho.copy_to_host(r.data(), n);
    if (s.kahan.size() > 0) {
        c.resize(n);
        s.kahan.copy_to_host(c.data(), n);
    }
    std::vector<cplx> out(n);
    for (size_t k = 0; k < n; ++k) {
        out[k] = {heom::c_re(r[k]), heom::c_im(r[k])};
        if (!c.empty()) out[k] -= cplx(heom::c_re(c[k]), heom::c_im(c[k]));
    }
    return out;
}

void test_relative_distance_zero_and_restore() {
    TEST("relative_distance: own snapshot = 0, after restore = 0");
    auto cfg = load_monomer();
    SystemContext sys(cfg);
    HEOMTrajectory traj(sys);
    traj.set_initial({monomer_excited(1.0)}, kMonoEE);
    traj.dress_ados();
    traj.propagate_steps(7);
    DeviceHierarchySnapshot snap;
    traj.snapshot_device(snap);
    ASSERT_EQ(traj.relative_distance(snap), 0.0);
    traj.propagate_steps(5);
    ASSERT_TRUE(traj.relative_distance(snap) > 0.0);
    traj.restore_device(snap);
    ASSERT_EQ(traj.relative_distance(snap), 0.0);
    PASS();
}

// ρ = 2|e⟩⟨e| against a snapshot of |e⟩⟨e|, ADOs all zero: ‖2−1‖/‖2‖ = 1/2.
void test_relative_distance_hand_value() {
    TEST("relative_distance: 2|e><e| vs |e><e| = 1/2 exactly");
    auto cfg = load_monomer();
    SystemContext sys(cfg);
    HEOMTrajectory traj(sys);
    DeviceHierarchySnapshot snap;
    traj.set_initial({monomer_excited(1.0)}, kMonoEE);
    traj.snapshot_device(snap);
    traj.set_initial({monomer_excited(2.0)}, kMonoEE);
    ASSERT_EQ(traj.relative_distance(snap), 0.5);
    PASS();
}

// Propagated state (non-zero carry in the mixed build) against an earlier one,
// recomputed on the host from two snapshots, both sides taken as ρ − carry.
void test_relative_distance_host_recompute() {
    TEST("relative_distance: matches host recompute (incl. Kahan carry)");
    auto cfg = load_monomer();
    SystemContext sys(cfg);
    HEOMTrajectory traj(sys);
    traj.set_initial({monomer_excited(1.0)}, kMonoEE);
    traj.dress_ados();
    DeviceHierarchySnapshot then, now;
    traj.propagate_steps(3);
    traj.snapshot_device(then);
    traj.propagate_steps(4);
    traj.snapshot_device(now);
    if (kMP) ASSERT_TRUE(now.kahan.size() > 0);
    const auto a = corrected(now), b = corrected(then);
    double dist = 0.0, norm = 0.0;
    for (size_t k = 0; k < a.size(); ++k) {
        dist += std::norm(a[k] - b[k]);
        norm += std::norm(a[k]);
    }
    const double host = std::sqrt(dist / norm);
    ASSERT_TRUE(host > 0.0);
    ASSERT_NEAR(traj.relative_distance(then) / host, 1.0, 1e-12);
    PASS();
}

void test_relative_distance_mismatch_throws() {
    TEST("relative_distance: empty / foreign snapshot throws");
    auto cfg = load_monomer();
    SystemContext sys(cfg), other(cfg);
    HEOMTrajectory traj(sys), stranger(other);
    traj.set_initial({monomer_excited(1.0)}, kMonoEE);
    stranger.set_initial({monomer_excited(1.0)}, kMonoEE);
    DeviceHierarchySnapshot empty, foreign;
    ASSERT_THROWS(traj.relative_distance(empty));
    stranger.snapshot_device(foreign);
    ASSERT_THROWS(traj.relative_distance(foreign));
    PASS();
}

void test_dress_before_initial_throws() {
    TEST("dress_ados before set_initial throws");
    auto cfg = load_monomer();
    SystemContext sys(cfg);
    HEOMTrajectory traj(sys);
    ASSERT_THROWS(traj.dress_ados());
    PASS();
}

// Theory eq (monoeq): σ_α = Π_k a_k^{n_k} σ₀, a_k = −i(c_k − c̄_k)/ν_k; in Shi
// scaling ρ̃_α = Π_k (a_k/√s_k)^{n_k}/√(n_k!) σ₀ with s_k = √(|c_k||c̄_k|).
void test_dress_monomer_closed_form() {
    TEST("dress_ados (G3): monomer closed form, Σ tier 1 = -2λ, fixed point");
    auto cfg = load_monomer();
    SystemContext sys(cfg);
    const int M = sys.M(), D = sys.D();
    const auto& ph = cfg.physics;
    ASSERT_EQ(sys.n_bath(), 1);
    ASSERT_EQ(D, 7);                      // 1 Drude + 2 Padé + 2×2 mode ladders

    HEOMTrajectory traj(sys);
    traj.set_initial({monomer_excited(1.0)}, kMonoEE);
    traj.dress_ados();
    DeviceHierarchySnapshot snap;
    traj.snapshot_device(snap);
    const auto rho = corrected(snap);
    ASSERT_EQ(rho.size(), static_cast<size_t>(M));

    std::vector<int8_t> lvl(static_cast<size_t>(M) * D);
    ASSERT_EQ(cudaMemcpy(lvl.data(), sys.d_level_indices(), lvl.size(),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    std::vector<cplx> g(D);               // a_k / √s_k
    std::vector<double> s(D);
    for (int k = 0; k < D; ++k) {
        const cplx c = ph.c_bath[k], cb = ph.cbar_bath[k], nu = ph.nu_bath[k];
        s[k] = std::sqrt(std::abs(c) * std::abs(cb));
        g[k] = cplx(0.0, -1.0) * (c - cb) / nu / std::sqrt(s[k]);
    }

    double worst = 0.0;
    int nonzero = 0;
    cplx tier1(0.0, 0.0);
    for (int a = 0; a < M; ++a) {
        const int8_t* n = lvl.data() + static_cast<size_t>(a) * D;
        cplx want(1.0, 0.0);
        int tier = 0, last = -1;
        for (int k = 0; k < D; ++k) {
            for (int p = 1; p <= n[k]; ++p) want *= g[k] / std::sqrt(double(p));
            tier += n[k];
            if (n[k] > 0) last = k;
        }
        if (tier == 1) tier1 += rho[a] * std::sqrt(s[last]);
        const double err = std::abs(rho[a] - want) / std::max(1.0, std::abs(want));
        ASSERT_TRUE(std::isfinite(err));
        worst = std::max(worst, err);
        if (tier > 0 && std::abs(want) > 1e-3) ++nonzero;
    }
    ASSERT_TRUE(nonzero > 10);           // the comparison has teeth
    ASSERT_NEAR(worst, 0.0, kAdoTol);

    const double lam = ph.reorg_cm.at(0) * CM_TO_AU;
    ASSERT_NEAR(tier1.real() / (-2.0 * lam), 1.0, kAdoTol);
    ASSERT_NEAR(tier1.imag() / lam, 0.0, kAdoTol);

    ASSERT_EQ(static_cast<int>(sys.first_tier().size()), D);
    std::vector<cplx> esb;
    traj.read_system_bath_energy(esb);
    ASSERT_EQ(esb.size(), 1u);
    ASSERT_NEAR(esb[0].real() / (-2.0 * lam), 1.0, kAdoTol);
    ASSERT_NEAR(esb[0].imag() / lam, 0.0, kAdoTol);

    traj.propagate_steps(100);           // ~ τ_c of the Drude ladder (1/γ = 100 fs)
    ASSERT_TRUE(!traj.diverged());
    ASSERT_NEAR(traj.relative_distance(snap), 0.0, kFixedTol);
    const Eigen::MatrixXcd s0 = traj.slot_rho0(0);
    ASSERT_NEAR(s0.trace().real(), 1.0, kAdoTol);
    ASSERT_NEAR(s0.trace().imag(), 0.0, kAdoTol);

    // Already at the floor: ε becomes finite at the first r < 1 (E0: ≤ 4th check).
    // Floor of δ per 50 fs here: ~2e-14 FP64, ~1e-5 MP; tolerance ≈ 100× floor.
    const HEOMConfig::EquilibrationConfig eq{/*t_max_fs=*/500.0, /*check_interval_fs=*/50.0,
                                             /*tolerance=*/kMP ? 1e-3 : 1e-10};
    heom::spectra::EquilibrationHistory h;
    ASSERT_NOTHROW(h = heom::spectra::equilibrate(traj, snap, eq));
    ASSERT_TRUE(h.delta.size() <= 10);
    ASSERT_TRUE(h.t_eq_fs > 0.0);
    PASS();
}

void test_bare_boltzmann_monomer() {
    TEST("bare_boltzmann_ee: monomer = |e><e|, zeros elsewhere");
    auto cfg = load_monomer();
    SystemContext sys(cfg);
    const Eigen::MatrixXcd r = heom::spectra::bare_boltzmann_ee(sys);
    ASSERT_EQ(r.rows(), 2);
    ASSERT_EQ(r(1, 1), cplx(1.0, 0.0));
    ASSERT_EQ(r(0, 0), cplx(0.0, 0.0));
    ASSERT_EQ(r(0, 1), cplx(0.0, 0.0));
    ASSERT_EQ(r(1, 0), cplx(0.0, 0.0));
    PASS();
}

// J=0, YAML energies 12100 / 12300 are the bare ones (the solver adds λ): the
// population ratio is exp(−200 cm⁻¹/kT), independent of the reorganisation.
void test_bare_boltzmann_uses_bare_energies() {
    TEST("bare_boltzmann_ee: J=0 dimer populations from bare energies");
    auto cfg = HEOMConfig::Config::load_yaml(write_uncoupled_dimer_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const Eigen::MatrixXcd r = heom::spectra::bare_boltzmann_ee(sys);
    const double kT_cm = 77.0 * BOLTZMANN / CM_TO_AU;
    const double ratio = std::exp(-200.0 / kT_cm);
    ASSERT_NEAR(r(0, 0).real(), 1.0 / (1.0 + ratio), 1e-12);
    ASSERT_NEAR(r(1, 1).real(), ratio / (1.0 + ratio), 1e-12);
    ASSERT_NEAR(std::abs(r(0, 1)), 0.0, 1e-15);
    PASS();
}

void test_equilibrate_not_converged_message() {
    TEST("equilibrate (G7): t_max reached, message carries δ / r / ε");
    auto cfg = load_monomer();
    SystemContext sys(cfg);
    HEOMTrajectory traj(sys);
    traj.set_initial({heom::spectra::bare_boltzmann_ee(sys)}, kMonoEE);
    DeviceHierarchySnapshot snap;
    const HEOMConfig::EquilibrationConfig eq{20.0, 10.0, 1e-9};
    std::string what;
    try {
        heom::spectra::equilibrate(traj, snap, eq);
    } catch (const std::runtime_error& e) {
        what = e.what();
    }
    ASSERT_TRUE(what.find("did not converge") != std::string::npos);
    ASSERT_TRUE(what.find("δ=") != std::string::npos);
    ASSERT_TRUE(what.find("r=") != std::string::npos);
    ASSERT_TRUE(what.find("ε=") != std::string::npos);
    ASSERT_TRUE(what.find("1/Re ν") != std::string::npos);
    PASS();
}

void test_equilibrate_bad_intervals_throw() {
    TEST("equilibrate (G7): interval not a multiple of dt / too short throws");
    auto cfg = load_monomer();
    SystemContext sys(cfg);
    HEOMTrajectory traj(sys);
    traj.set_initial({monomer_excited(1.0)}, kMonoEE);
    DeviceHierarchySnapshot snap;
    ASSERT_THROWS(heom::spectra::equilibrate(traj, snap, {100.0, 2.5, 1e-6}));
    ASSERT_THROWS(heom::spectra::equilibrate(traj, snap, {15.0, 10.0, 1e-6}));
    PASS();
}

// Uncoupled sites, diagonal ρ_B: the dressed hierarchy is the exact steady
// state of the full generator (Term 1 and Term 4 vanish on the diagonal).
void test_dress_uncoupled_dimer_invariant() {
    TEST("dress_ados (G3): J=0 dimer, one RK4 step leaves it unchanged");
    auto cfg = HEOMConfig::Config::load_yaml(write_uncoupled_dimer_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    ASSERT_EQ(sys.N(), 2);
    Eigen::MatrixXcd r = Eigen::MatrixXcd::Zero(2, 2);
    r(0, 0) = 0.6;
    r(1, 1) = 0.4;
    HEOMTrajectory traj(sys);
    traj.set_initial({r}, heom::ManifoldBlock::full(2));
    traj.dress_ados();
    DeviceHierarchySnapshot snap;
    traj.snapshot_device(snap);
    const auto rho = corrected(snap);
    double ado_norm = 0.0;
    for (size_t k = 4; k < rho.size(); ++k) ado_norm += std::norm(rho[k]);
    ASSERT_TRUE(ado_norm > 1e-2);       // the ADOs were actually filled
    traj.propagate_steps(1);
    ASSERT_NEAR(traj.relative_distance(snap), 0.0, kStepTol);
    PASS();
}

// From the factorized |e><e| (all ADOs zero) ρ μ⁺ = μ⁺|g><g|,
// so emission and absorption walk the identical (e,g) hierarchy.
void test_g1_factorized_equals_absorption() {
    TEST("G1: factorized |e><e| emission == absorption, bit for bit");
    auto cfg_fl = load_monomer();
    auto cfg_abs = HEOMConfig::Config::load_yaml(write_monomer_vib_yaml());
    HEOMConfig::resolve_absorption_config(cfg_abs);
    const auto mu = dipole_of(cfg_fl);

    SystemContext sys_fl(cfg_fl);
    HEOMTrajectory traj(sys_fl);
    traj.set_initial({monomer_excited(1.0)}, kMonoEE);
    DeviceHierarchySnapshot eq;
    traj.snapshot_device(eq);
    const auto fl = heom::spectra::emit_from_equilibrium(traj, eq, mu);

    SystemContext sys_abs(cfg_abs);
    const auto abs = heom::spectra::compute_absorption(sys_abs, mu);
    ASSERT_EQ(max_diff(fl, abs), 0.0);
    PASS();
}

// G2 + G3 (ρ_eq): batch true vs false bit-identical; Tr ρ_eq = 1, Hermitian,
// support only in (e,e).
void test_g2_batch_bitwise_and_rho_eq() {
    TEST("G2: fluorescence batch == serial bit for bit; rho_eq trace 1, Hermitian");
    auto cfg = HEOMConfig::Config::load_yaml(write_coupled_dimer_yaml());
    HEOMConfig::resolve_fluorescence_config(cfg);
    SystemContext sys(cfg);
    const auto mu = dipole_of(cfg);

    cfg.linear_batch = false;
    const auto serial = heom::spectra::compute_fluorescence(sys, mu);
    cfg.linear_batch = true;
    const auto batch = heom::spectra::compute_fluorescence(sys, mu);
    ASSERT_EQ(max_diff(serial.linear, batch.linear), 0.0);
    ASSERT_TRUE(serial.rho_eq == batch.rho_eq);
    ASSERT_TRUE(serial.equilibration.delta == batch.equilibration.delta);

    const Eigen::MatrixXcd& r = serial.rho_eq;
    ASSERT_EQ(r.rows(), 3);
    ASSERT_NEAR(r.trace().real(), 1.0, kAdoTol);
    ASSERT_NEAR(r.trace().imag(), 0.0, kAdoTol);
    ASSERT_NEAR((r - r.adjoint()).norm(), 0.0, kAdoTol);
    ASSERT_EQ(r.row(0).norm(), 0.0);
    ASSERT_EQ(r.col(0).norm(), 0.0);
    ASSERT_TRUE(std::abs(r(1, 2)) > 1e-4);    // coupled: coherence in the site basis
    PASS();
}

}  // namespace

int main() {
    test_relative_distance_zero_and_restore();
    test_relative_distance_hand_value();
    test_relative_distance_host_recompute();
    test_relative_distance_mismatch_throws();
    test_dress_before_initial_throws();
    test_dress_monomer_closed_form();
    test_dress_uncoupled_dimer_invariant();
    test_bare_boltzmann_monomer();
    test_bare_boltzmann_uses_bare_energies();
    test_equilibrate_not_converged_message();
    test_equilibrate_bad_intervals_throw();
    test_g1_factorized_equals_absorption();
    test_g2_batch_bitwise_and_rho_eq();
    return heom_test_summary(kMP ? "test_fluorescence_mp" : "test_fluorescence");
}
