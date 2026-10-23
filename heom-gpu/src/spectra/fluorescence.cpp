#include "spectra/fluorescence.hpp"
#include "spectra/linear_record.hpp"

#include "core/manifold_layout.hpp"
#include "heom_trajectory.hpp"
#include "memory_budget.hpp"
#include "system_context.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace heom::spectra {

Eigen::MatrixXcd bare_boltzmann_ee(const SystemContext& sys) {
    const auto& ph = sys.config().physics;
    const int n = ph.n_states;
    if (static_cast<int>(ph.reorg_cm.size()) != n || ph.hamiltonian_site.rows() != n) {
        throw std::logic_error("bare_boltzmann_ee: reorg_cm / hamiltonian_site not built for "
                               + std::to_string(n) + " states");
    }
    if (!(ph.temperature > 0.0)) {
        throw std::invalid_argument("bare_boltzmann_ee: temperature must be > 0 K");
    }
    Eigen::MatrixXcd h = ph.hamiltonian_site;
    for (int i = 0; i < n; ++i) h(i, i) -= ph.reorg_cm[i] * CM_TO_AU;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(h);
    const Eigen::VectorXd& e = es.eigenvalues();
    Eigen::VectorXd w = (-(e.array() - e.minCoeff()) / ph.temperature).exp();
    w /= w.sum();
    const Eigen::MatrixXcd& v = es.eigenvectors();
    const Eigen::MatrixXcd rho_e = v * w.cast<std::complex<double>>().asDiagonal() * v.adjoint();

    const int N = sys.N();
    const heom::ManifoldBlock ee = heom::ManifoldLayout::from(ph.basis, n, N).block(1, 1);
    Eigen::MatrixXcd rho = Eigen::MatrixXcd::Zero(N, N);
    rho.block(ee.row_off, ee.col_off, n, n) = rho_e;
    return rho;
}

namespace {

int steps_of(double fs, double dt, const char* key) {
    const double x = fs / dt;
    const double n = std::round(x);
    if (!(n >= 1.0) || std::abs(x - n) > 1e-9 * x) {
        std::ostringstream os;
        os << "equilibration." << key << " = " << fs
           << " fs must be a positive integer multiple of dt = " << dt << " fs";
        throw std::invalid_argument(os.str());
    }
    return static_cast<int>(n);
}

// Slowest bath ladder: min Re ν over every slot (a.u.).
double slowest_rate(const SystemContext& sys) {
    double g = std::numeric_limits<double>::infinity();
    for (const auto& nu : sys.config().physics.nu_bath) g = std::min(g, nu.real());
    return g;
}

std::string not_converged_message(const SystemContext& sys, const EquilibrationHistory& h,
                                  const HEOMConfig::EquilibrationConfig& eq) {
    std::ostringstream os;
    os << "fluorescence equilibration did not converge within t_max = " << eq.t_max_fs
       << " fs (tolerance " << eq.tolerance << "). Last checks:";
    const size_t n = h.delta.size();
    for (size_t k = n - std::min<size_t>(n, 3); k < n; ++k) {
        os << "\n  t=" << h.time_fs[k] << " fs  δ=" << h.delta[k]
           << "  r=" << h.ratio[k] << "  ε=" << h.estimate[k];
    }
    // A steady geometric decay keeps every r < 1 (r = 0.98 is still a decay);
    // only δ that stops falling shows r ≥ 1 among the last checks.
    const double d = h.delta.back(), r = h.ratio.back();
    bool stalled = n < 3;
    for (size_t k = n - std::min<size_t>(n, 3); k < n; ++k)
        stalled = stalled || !(h.ratio[k] < 1.0);
    if (stalled) {
        os << "\n  δ is no longer decreasing (r ≥ 1 in the last checks). If δ ≈ " << d
           << " is the precision floor, raise tolerance to at least " << 100.0 * d
           << "; otherwise raise t_max, or check dt / L_cutoff.";
    } else {
        const double more = std::log(eq.tolerance * (1.0 - r) / (d * r)) / std::log(r);
        os << "\n  δ still decays (observed decay time −Δ/ln r = "
           << -eq.check_interval_fs / std::log(r) << " fs): raise t_max by about "
           << std::ceil(more) * eq.check_interval_fs << " fs.";
    }
    const double g = slowest_rate(sys);
    if (g > 0.0 && std::isfinite(g)) {
        os << "\n  Slowest bath ladder 1/Re ν = " << 1.0 / g / FS_TO_AU
           << " fs; bath-only estimate t_max ≳ 1.5·ln(1/tol)/γ_min = "
           << 1.5 * std::log(1.0 / eq.tolerance) / g / FS_TO_AU
           << " fs (exciton population relaxation can be slower).";
    }
    return os.str();
}

}  // namespace

EquilibrationHistory equilibrate(HEOMTrajectory& traj, DeviceHierarchySnapshot& snap,
                                 const HEOMConfig::EquilibrationConfig& eq) {
    const double dt = traj.sys().config().time.dt;
    const int n_chk = steps_of(eq.check_interval_fs, dt, "check_interval");
    const int n_checks = static_cast<int>(std::floor(eq.t_max_fs / eq.check_interval_fs + 1e-9));
    if (n_checks < 2) {
        throw std::invalid_argument("equilibration.t_max must cover at least two check_interval");
    }
    constexpr double inf = std::numeric_limits<double>::infinity();
    EquilibrationHistory h;
    double prev = std::numeric_limits<double>::quiet_NaN();
    for (int c = 1; c <= n_checks; ++c) {
        traj.snapshot_device(snap);
        traj.propagate_steps(n_chk);
        const double t = static_cast<double>(c) * n_chk * dt;
        const double d = traj.diverged() ? inf : traj.relative_distance(snap);
        if (!std::isfinite(d)) {
            throw std::runtime_error(
                "fluorescence equilibration diverged at t = " + std::to_string(t)
                + " fs; reduce dt or raise L_cutoff");
        }
        const double r = d / prev;
        const double e = (d == 0.0) ? 0.0 : (r < 1.0) ? d * r / (1.0 - r) : inf;
        h.time_fs.push_back(t);
        h.delta.push_back(d);
        h.ratio.push_back(r);
        h.estimate.push_back(e);
        std::printf("[fl eq] t=%.1f fs  δ=%.3e  r=%.4f  ε=%.3e\n", t, d, r, e);
        std::fflush(stdout);
        if (std::max(d, e) <= eq.tolerance) {
            h.t_eq_fs = t;
            return h;
        }
        prev = d;
    }
    throw std::runtime_error(not_converged_message(traj.sys(), h, eq));
}

LinearResponseResult emit_from_equilibrium(HEOMTrajectory& traj,
                                           const DeviceHierarchySnapshot& eq,
                                           const DipoleOperator& mu) {
    const SystemContext& sys = traj.sys();
    validate_linear_response_inputs(sys.config().physics.basis, mu, sys.N());
    LinearResponseResult out = empty_linear_result(sys);
    const auto dirs = active_directions(mu, "fl");
    for (const auto& group : direction_groups(dirs, traj.batch() > 1)) {
        const std::string label = announce_group("fl", group);
        if (group.size() > 1) traj.expand_slots(eq, std::vector<int>(group.size(), 0));
        else                  traj.restore_device(eq);
        std::vector<Eigen::MatrixXcd> mus;
        for (int dir : group) mus.push_back(mu_plus(mu, dir));
        traj.apply_dipole(mus, DipoleSide::Right, eg_block(sys));
        record_group(traj, group, out, label);
    }
    return out;
}

namespace {

heom::ManifoldBlock ee_block(const SystemContext& sys) {
    const auto& ph = sys.config().physics;
    return heom::ManifoldLayout::from(ph.basis, ph.n_states, sys.N()).block(1, 1);
}

// Trajectory (B slots) + one rolling snapshot (with the mixed build's carry).
void check_fluorescence_budget(const SystemContext& sys, int B) {
    const size_t MNb = static_cast<size_t>(sys.M()) * sys.rho_capacity();
    auto budget = heom::estimate_budget(sys.config(), static_cast<size_t>(sys.M()), B,
                                        heom::Resident::SystemTables);
    const size_t extra = (heom::kMixedPrecision ? 2 : 1) * MNb * sizeof(heom::complex_t);
    budget.driver    += extra;
    budget.gpu_total += extra;
    size_t gfree = 0, gtot = 0;
    heom::query_gpu_memory(gfree, gtot);
    std::string err;
    if (heom::check_budget(budget, sys.config(), gfree, err) != heom::BudgetCheck::Exceeds)
        return;
    heom::print_budget(budget, gfree, gtot);
    throw std::runtime_error(
        "compute_fluorescence: trajectory (B=" + std::to_string(B)
        + ") + 1 hierarchy snapshot exceeds GPU budget; try fluorescence.batch: false, "
          "a smaller L_cutoff or filter.enabled: " + err);
}

// Populations of clusters with no electronic path between them
// are fixed by the start, not equilibrated by the bath.
void warn_if_disconnected(const SystemContext& sys) {
    const Eigen::MatrixXcd& h = sys.config().physics.hamiltonian_site;
    const int n = static_cast<int>(h.rows());
    std::vector<int> seen(n, 0), stack{0};
    seen[0] = 1;
    int reached = 1;
    while (!stack.empty()) {
        const int i = stack.back();
        stack.pop_back();
        for (int j = 0; j < n; ++j) {
            if (seen[j] || h(i, j) == 0.0) continue;
            seen[j] = 1;
            ++reached;
            stack.push_back(j);
        }
    }
    if (reached < n) {
        std::printf("[warn] fluorescence: the electronic coupling graph is disconnected "
                    "(%d of %d pigments reachable from pigment 1); populations between "
                    "clusters keep their bare-Boltzmann start values\n", reached, n);
    }
}

// α=0 block of a width-1 snapshot as N×N, ρ − carry in the mixed build.
Eigen::MatrixXcd alpha0_of(const DeviceHierarchySnapshot& s, int N) {
    const size_t Nb = static_cast<size_t>(s.support.elems());
    std::vector<heom::complex_t> r(Nb), c;
    s.rho.copy_to_host(r.data(), Nb);
    if (s.kahan.size() > 0) {
        c.resize(Nb);
        s.kahan.copy_to_host(c.data(), Nb);
    }
    Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(N, N);
    for (int lr = 0; lr < s.support.n_row; ++lr)
        for (int lc = 0; lc < s.support.n_col; ++lc) {
            const size_t k = static_cast<size_t>(lr) * s.support.n_col + lc;
            std::complex<double> v(heom::c_re(r[k]), heom::c_im(r[k]));
            if (!c.empty()) v -= std::complex<double>(heom::c_re(c[k]), heom::c_im(c[k]));
            out(s.support.row_off + lr, s.support.col_off + lc) = v;
        }
    return out;
}

}  // namespace

FluorescenceResult compute_fluorescence(const SystemContext& sys, const DipoleOperator& mu) {
    validate_linear_response_inputs(sys.config().physics.basis, mu, sys.N());
    const auto dirs = active_directions(mu, nullptr);   // emit_from_equilibrium reports skips
    const int B = sys.config().linear_batch ? static_cast<int>(dirs.size()) : 1;
    check_fluorescence_budget(sys, B);
    warn_if_disconnected(sys);

    HEOMTrajectory traj(sys, B);
    DeviceHierarchySnapshot snap;
    FluorescenceResult out;
    traj.set_initial({bare_boltzmann_ee(sys)}, ee_block(sys));
    traj.dress_ados();
    out.equilibration = equilibrate(traj, snap, sys.config().equilibration);
    traj.snapshot_device(snap);
    out.rho_eq = alpha0_of(snap, sys.N());
    out.linear = emit_from_equilibrium(traj, snap, mu);
    return out;
}

}  // namespace heom::spectra
