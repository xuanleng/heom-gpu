#include "types.hpp"
#include "core/hamiltonian_expansion.hpp"   // ManifoldLayout, DoubleExcitedIndex
#include "core/mode_layout.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace {

// Relative-gap threshold for declaring a pole degenerate with the Drude pole
// (nu_k near gamma). A smaller relative gap is a degenerate pole.
constexpr double kResonanceRelTol = 1e-3;

// k=0 Drude pole for site n, identical form for both expansions:
//   ν_{n,0} = γ_n,  c_{n,0} = (η_n γ_n/2)[cot(γ_n/(2T)) − i]   (ℏ = 1 in a.u.)
// T is global; eta and gamma are per site.
void set_drude_pole(PhysicalParameters& phys, int n) {
    const BathSite& s = phys.bath_sites[n];
    const double gamma_au = s.gamma_au;
    const double cot_val = 1.0 / std::tan(gamma_au / (2.0 * phys.temperature));
    const int base = phys.mode_idx(n, 0);
    phys.nu_bath[base] = gamma_au;
    phys.c_bath[base] = std::complex<double>(
        0.5 * s.eta_au * gamma_au * cot_val,
        -0.5 * s.eta_au * gamma_au
    );
}

// Guard a non-Drude pole against collapsing onto ν_0 = γ. When ν_k ≈ γ the
// denominator ν_k²−γ² vanishes (c_k → NaN/Inf) AND two HEOM modes share one
// frequency, breaking the ±1-jump assumption. Caught before the division so
// corrupted coefficients never reach the kernel. Mirrors core.py::_check_resonance.
void check_bath_resonance(double nu_k, double gamma_au, int k, const char* method) {
    const double rel_gap = std::abs(nu_k - gamma_au) / std::max(nu_k, gamma_au);
    if (rel_gap < kResonanceRelTol) {
        throw std::runtime_error(
            std::string(method) + " resonance at k=" + std::to_string(k)
            + ": ν_k=" + std::to_string(nu_k) + " ≈ γ=" + std::to_string(gamma_au)
            + " (relative gap " + std::to_string(rel_gap) + " < "
            + std::to_string(kResonanceRelTol) + "). Two bath modes ν_0=γ and ν_"
            + std::to_string(k) + " collapse to the same frequency and "
            + "c[" + std::to_string(k) + "] = 2ηγ·ν_k·T/(ν_k²−γ²) blows up. "
            + "Pick a different n_extra_poles, or shift gamma_inverse_fs / "
            + "temperature_K slightly to break the degeneracy.");
    }
}

// Pin the ladder layout and pre-size the flat per-site coefficient arrays.
//
// This is the ONE place that knows how many slots a bath gets and in which
// order: [Drude, pole 1..K_n, BO₁+, BO₁−, BO₂+, BO₂−, …], P_n = 1 + K_n + 2ν_n.
// Everything downstream reads mode_off / D / mode_kind. Arrays are zero-filled.
void setup_bath_arrays(PhysicalParameters& phys) {
    const int n_bath = phys.n_bath();
    if (static_cast<int>(phys.bath_sites.size()) != n_bath)
        throw std::runtime_error("bath_sites must have one entry per pigment");
    std::vector<int> counts(n_bath);
    for (int n = 0; n < n_bath; ++n) counts[n] = phys.bath_sites[n].n_slots();
    phys.set_mode_counts(counts);
    const size_t D = static_cast<size_t>(phys.n_ladders());
    phys.c_bath.assign(D, std::complex<double>(0.0, 0.0));
    phys.cbar_bath.assign(D, std::complex<double>(0.0, 0.0));
    phys.nu_bath.assign(D, std::complex<double>(0.0, 0.0));
    phys.markov_residue.assign(n_bath, 0.0);
    phys.mode_kind.assign(D, ModeKind::Drude);
    for (int n = 0; n < n_bath; ++n) {
        const BathSite& s = phys.bath_sites[n];
        for (int l = 1; l <= s.n_extra_poles; ++l)
            phys.mode_kind[phys.mode_idx(n, l)] = ModeKind::Pole;
        for (size_t m = 0; m < s.modes.size(); ++m) {
            const int k = 1 + s.n_extra_poles + 2 * static_cast<int>(m);
            phys.mode_kind[phys.mode_idx(n, k)]     = ModeKind::BOPlus;
            phys.mode_kind[phys.mode_idx(n, k + 1)] = ModeKind::BOMinus;
        }
    }
}

// Underdamped Brownian oscillator m of site n → the conjugate ladder pair.
//   J_m(ω) = 4λγω₀²ω / ((ω₀²−ω²)² + 4γ²ω²) = Σ± b±ω/(ω²+a±²),
//   a± = γ ± iω₁,  b± = ±iλω₀²/ω₁,  ω₁ = sqrt(ω₀² − γ²).
// γ is the C(t) envelope decay rate (same meaning as the Drude γ); the mode's
// Lorentzian FWHM is 2γ. Each Lorentzian is a Drude term with complex (a, b),
// so the Drude residue formula c = (b/2)[cot(a/(2T)) − i] applies verbatim with
// complex arguments: ν± = a±, c± = (b±/2)[cot(a±/(2T)) − i]. This equals the
// textbook c+ = P(κ+1), c− = P(κ*−1), P = λω₀²/(2ω₁), κ = coth(β(ω₁−iγ)/2)
// cbar comes from the pairing rule in set_cbar.
void set_bo_pair(PhysicalParameters& phys, int n, int m, int k_plus) {
    const VibMode& v = phys.bath_sites[n].modes[m];
    const double w0 = v.omega_au, g = v.gamma_au, lam = v.lambda_au;
    auto in_cm = [](double au) { return std::to_string(au / CM_TO_AU); };
    const std::string who = "bath site " + std::to_string(n + 1) + " vibrational mode "
                          + std::to_string(m + 1) + " (ω₀=" + in_cm(w0) + " cm⁻¹, γ="
                          + in_cm(g) + " cm⁻¹ = FWHM/2, 1/γ=" + std::to_string(v.gamma_inverse_fs)
                          + " fs)";
    if (!(w0 > 0.0))  throw std::runtime_error(who + ": omega_cm must be > 0");
    if (!(lam > 0.0)) throw std::runtime_error(who + ": huang_rhys must be > 0");
    if (!(g > 0.0))   throw std::runtime_error(who + ": gamma_inverse must be > 0");
    if (g >= w0)
        throw std::runtime_error(
            who + ": γ ≥ ω₀ is not underdamped. Overdamped/critical modes are "
            "not supported; need gamma_inverse > " + std::to_string(1.0 / (w0 * FS_TO_AU))
            + " fs (5308.8/omega_cm).");
    const double w1 = std::sqrt(w0 * w0 - g * g);
    if (w1 / w0 < 1e-3)
        throw std::runtime_error(
            who + ": near-critical damping (ω₁/ω₀ = " + std::to_string(w1 / w0)
            + " < 1e-3); c± ∝ 1/ω₁ cancel catastrophically. Move γ away from ω₀.");

    const std::complex<double> I(0.0, 1.0);
    const double T = phys.temperature;
    const std::complex<double> a[2] = { g + I * w1, g - I * w1 };
    const std::complex<double> b[2] = { I * (lam * w0 * w0 / w1), -I * (lam * w0 * w0 / w1) };
    for (int s = 0; s < 2; ++s) {
        const std::complex<double> cot_val = 1.0 / std::tan(a[s] / (2.0 * T));
        const int idx = phys.mode_idx(n, k_plus + s);
        phys.nu_bath[idx] = a[s];
        phys.c_bath[idx]  = 0.5 * b[s] * (cot_val - I);
    }
}

// The vibrational-mode contribution to a real (Matsubara/Padé) pole ν of site n:
//   c_pole = χ · 2iT · J(iν),   J_BO(iν) = 4iλγω₀²ν / ((ω₀²+ν²)² − 4γ²ν²)
//   ⇒ Σ_m −8 λ_m γ_m ω₀m² T ν / ((ω₀m²+ν²)² − 4γ_m²ν²)       (before the χ).
// The Drude part is written by the caller token-for-token as before; this is
// an ADD term — no modes ⇒ 0.0 ⇒ the pole coefficient is bit-identical.
double bo_pole_add(const PhysicalParameters& phys, int n, double nu) {
    double sum = 0.0;
    for (const VibMode& v : phys.bath_sites[n].modes) {
        const double w0 = v.omega_au, g = v.gamma_au;
        const double p = w0 * w0 + nu * nu;
        sum += -8.0 * v.lambda_au * g * w0 * w0 * phys.temperature * nu
             / (p * p - 4.0 * g * g * nu * nu);
    }
    return sum;
}

// Exact ∫₀^∞ Re C(t) dt = T·lim_{ω→0} J(ω)/ω = T[η/γ + Σ_m 4λ_mγ_m/ω₀m²].
// Drude part token-for-token as before, BO part an add term (0 without modes).
double exact_area(const PhysicalParameters& phys, int n) {
    const BathSite& s = phys.bath_sites[n];
    double area = s.eta_au * phys.temperature / s.gamma_au;
    double bo = 0.0;
    for (const VibMode& v : s.modes)
        bo += 4.0 * v.lambda_au * v.gamma_au / (v.omega_au * v.omega_au);
    area += phys.temperature * bo;
    return area;
}

// Every ladder of a site must carry a distinct frequency: two equal ν's are
// one ladder written twice (the ±1-jump bookkeeping and the c̄ pairing both
// assume distinct ν). The Drude-vs-pole 1e-3 resonance gate above is stricter
// for the real poles; this catches duplicated vibrational modes.
void check_distinct_poles(const PhysicalParameters& phys, int n) {
    const heom::ModeLayout ml = phys.mode_layout();
    const int kc = ml.count(n);
    for (int j = 0; j < kc; ++j)
        for (int k = j + 1; k < kc; ++k) {
            const std::complex<double> nj = phys.nu_bath[ml.slot(n, j)];
            const std::complex<double> nk = phys.nu_bath[ml.slot(n, k)];
            if (std::abs(nj - nk) < 1e-9 * std::abs(nk))
                throw std::runtime_error(
                    "bath site " + std::to_string(n + 1) + ": slots " + std::to_string(j)
                    + " and " + std::to_string(k) + " have the same ν = ("
                    + std::to_string(nk.real() / CM_TO_AU) + ", "
                    + std::to_string(nk.imag() / CM_TO_AU)
                    + ") cm⁻¹ — duplicate ladder (same vibrational mode listed twice?)");
        }
}

// Right-action coefficients for site n. C*(t) = Σ_k c_k* e^{-ν_k* t}: the
// ladder that decays with ν_k carries, on the right, the conjugate of the
// coefficient whose frequency is ν_k*. ONE rule, no slot-order assumption:
//   c̄_k = c_j*,  j = the slot of this site with ν_j == ν_k*.
// Real ν ⇒ j == k (c̄ = c*, exactly what the kernel used to conj on the fly).
// Conjugate pair ⇒ j = the partner. No partner, or two, ⇒ the expansion is
// not closed under conjugation and the HEOM would silently break Hermiticity
// — throw.
void set_cbar(PhysicalParameters& phys, int n) {
    const heom::ModeLayout ml = phys.mode_layout();
    const int kc = ml.count(n);
    for (int k = 0; k < kc; ++k) {
        const int ik = ml.slot(n, k);
        const std::complex<double> target = std::conj(phys.nu_bath[ik]);
        const double tol = 1e-12 * std::abs(target);
        int partner = -1;
        for (int j = 0; j < kc; ++j) {
            const int ij = ml.slot(n, j);
            if (std::abs(phys.nu_bath[ij] - target) <= tol) {
                if (partner >= 0) {
                    throw std::runtime_error(
                        "bath site " + std::to_string(n) + ": slots "
                        + std::to_string(partner) + " and " + std::to_string(j)
                        + " both have ν = conj(ν_" + std::to_string(k)
                        + "); duplicate ladder, c̄ pairing is ambiguous");
                }
                partner = j;
            }
        }
        if (partner < 0) {
            throw std::runtime_error(
                "bath site " + std::to_string(n) + ", slot " + std::to_string(k)
                + ": no slot with ν = conj(ν_k) — the exponential expansion is "
                  "not closed under conjugation (C*(t) needs it)");
        }
        phys.cbar_bath[ik] = std::conj(phys.c_bath[ml.slot(n, partner)]);
    }
}

// Per-site tail, run AFTER c_{nk}/ν_{nk} for every slot of site n are filled:
//   c̄_{nk} by the pairing rule above, then
//   markov_residue[n] = η_n T/γ_n − Σ_k Re(c_{nk}/ν_{nk})  (ν=0 skipped)
// Identical form for Matsubara and Padé — only the c_{nk}/ν_{nk} contents differ.
// Markov residue = exact real area minus the retained finite sum.
void finalize_site(PhysicalParameters& phys, int n) {
    const BathSite& s = phys.bath_sites[n];
    const int k_bo = 1 + s.n_extra_poles;
    for (size_t m = 0; m < s.modes.size(); ++m)
        set_bo_pair(phys, n, static_cast<int>(m), k_bo + 2 * static_cast<int>(m));
    check_distinct_poles(phys, n);
    set_cbar(phys, n);
    const heom::ModeLayout ml = phys.mode_layout();
    double retained = 0.0;
    for (int k = 0; k < ml.count(n); ++k) {
        const int idx = ml.slot(n, k);
        if (phys.nu_bath[idx] == 0.0) continue;
        retained += (phys.c_bath[idx] / phys.nu_bath[idx]).real();
    }
    const double residue = exact_area(phys, n) - retained;

    // The Markov terminator enters as -residue·[Q,[Q,ρ]], a positive-definite
    // dissipator ONLY while residue ≥ 0. A negative residue flips its sign into
    // anti-diffusion (pumps the hierarchy) and can drive an otherwise-stable run
    // to divergence with no other symptom. It means the retained finite sum
    // Σ_{k≤K} Re(c_k/ν_k) overshot the exact area ηT/γ — usually too few poles
    // for the temperature. Warn; don't silently ship an unphysical sign.
    if (residue < 0.0) {
        std::cerr << "[WARN] Markov residue for site " << n << " is negative ("
                  << residue << "): the truncation terminator becomes "
                  "anti-dissipative and may drive divergence. Increase "
                  "n_extra_poles or raise the temperature.\n";
    }
    phys.markov_residue[n] = residue;
}

}  // namespace

// ---------------------------------------------------------------------------
// Padé [K-1/K] poles ζ_l (dimensionless, ascending) and weights χ_l.
//
// Padé [K-1/K] poles, transcribed rather than re-derived. Two symmetric
// tridiagonal matrices (dim 2K and
// 2K-1) with zero diagonal and off-diagonals 1/sqrt((2i+3)(2i+5)) resp.
// 1/sqrt((2i+5)(2i+7)); eval = 4/λ²; χ via the double product with the e^(1)
// index remap that skips l-1. Eigen's SelfAdjointEigenSolver eigenvalues are
// ascending, matching numpy.linalg.eigvalsh.
//
// Returned arrays have length K+1 with index [0]=0 (Drude placeholder), so
// they align index-for-index with c_bath / nu_bath. K=0 returns {0.0} only
// (no PSD poles): the dim 2K-1 = -1 matrix must NOT be built.
void boson_pade_NmN(int K, std::vector<double>& zeta, std::vector<double>& chi) {
    if (K <= 0) {
        zeta.assign(1, 0.0);
        chi.assign(1, 0.0);
        return;
    }

    const int n0 = K;
    const int n1 = K - 1;
    const int dim0 = 2 * K;
    const int dim1 = 2 * K - 1;

    Eigen::MatrixXd dmat0 = Eigen::MatrixXd::Zero(dim0, dim0);
    for (int i = 0; i < dim0 - 1; ++i) {
        const double off = 1.0 / std::sqrt((2.0 * i + 3.0) * (2.0 * i + 5.0));
        dmat0(i, i + 1) = off;
        dmat0(i + 1, i) = off;
    }
    Eigen::VectorXd dval0 =
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(dmat0).eigenvalues();

    Eigen::VectorXd dval1;
    if (dim1 >= 1) {
        Eigen::MatrixXd dmat1 = Eigen::MatrixXd::Zero(dim1, dim1);
        for (int i = 0; i < dim1 - 1; ++i) {
            const double off = 1.0 / std::sqrt((2.0 * i + 5.0) * (2.0 * i + 7.0));
            dmat1(i, i + 1) = off;
            dmat1(i + 1, i) = off;
        }
        dval1 = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(dmat1).eigenvalues();
    }

    // eval = 4/λ² over the n0 (resp. n1) smallest (most negative) eigenvalues.
    std::vector<double> eval0(n0), eval1(n1);
    for (int j = 0; j < n0; ++j) eval0[j] = 4.0 / (dval0[j] * dval0[j]);
    for (int k = 0; k < n1; ++k) eval1[k] = 4.0 / (dval1[k] * dval1[k]);

    zeta.assign(K + 1, 0.0);
    chi.assign(K + 1, 0.0);
    const double pref = 0.5 * n0 * (2.0 * n0 + 3.0);
    for (int j = 0; j < n0; ++j) {
        double tmpa = 1.0;
        for (int k = 0; k < j; ++k)
            tmpa *= (eval1[k] - eval0[j]) / (eval0[k] - eval0[j]);
        for (int k = j + 1; k < n0; ++k)
            tmpa *= (eval1[k - 1] - eval0[j]) / (eval0[k] - eval0[j]);
        zeta[j + 1] = std::sqrt(eval0[j]);
        chi[j + 1] = pref * tmpa;
    }
}

void compute_matsubara_drude(PhysicalParameters& phys) {
    const double T_au = phys.temperature;
    setup_bath_arrays(phys);
    const int n_bath = phys.n_bath();

    for (int n = 0; n < n_bath; ++n) {
        const double gamma_au = phys.bath_sites[n].gamma_au;
        const double eta_au   = phys.bath_sites[n].eta_au;
        const int    K_n      = phys.bath_sites[n].n_extra_poles;
        set_drude_pole(phys, n);

        // Matsubara poles: ν_k = 2πkT (site-independent), c_{nk} = 2η_nγ_n·T·ν_k/(ν_k²−γ_n²)
        // + the vibrational-mode residues at the same pole (bo_pole_add).
        for (int k = 1; k <= K_n; ++k) {
            const double omg_k = 2.0 * k * PI * T_au;
            check_bath_resonance(omg_k, gamma_au, k, "Matsubara");
            const int idx = phys.mode_idx(n, k);
            phys.nu_bath[idx] = omg_k;
            const double denom = omg_k * omg_k - gamma_au * gamma_au;
            phys.c_bath[idx] = std::complex<double>(
                2.0 * eta_au * gamma_au * T_au * omg_k / denom + bo_pole_add(phys, n, omg_k), 0.0
            );
        }

        finalize_site(phys, n);
    }
}

void compute_pade_drude(PhysicalParameters& phys) {
    const double T_au = phys.temperature;
    setup_bath_arrays(phys);
    const int n_bath = phys.n_bath();

    // Padé poles ζ_l / weights χ_l depend only on K_n, so they are computed per
    // site (a few µs of Eigen; a cache would be a second code path for nothing).
    std::vector<double> zeta, chi;
    for (int n = 0; n < n_bath; ++n) {
        const double gamma_au = phys.bath_sites[n].gamma_au;
        const double eta_au   = phys.bath_sites[n].eta_au;
        const int    K_n      = phys.bath_sites[n].n_extra_poles;
        boson_pade_NmN(K_n, zeta, chi);
        set_drude_pole(phys, n);

        // ν_l = ζ_l·T (ℏ=1, β=1/T ⇒ ζ·T is the a.u. frequency),
        // c_{nl} = χ_l · [2η_nγ_n·ν_l·T/(ν_l²−γ_n²) + Σ_m BO_m(ν_l)]. Factor 2ηγ
        // Factor 2*eta*gamma, not 4*eta*gamma, plus underdamped-mode residues.
        for (int l = 1; l <= K_n; ++l) {
            const double nu_l = zeta[l] * T_au;
            check_bath_resonance(nu_l, gamma_au, l, "Padé");
            const int idx = phys.mode_idx(n, l);
            phys.nu_bath[idx] = nu_l;
            const double denom = nu_l * nu_l - gamma_au * gamma_au;
            phys.c_bath[idx] = std::complex<double>(
                chi[l] * 2.0 * eta_au * gamma_au * nu_l * T_au / denom
                    + chi[l] * bo_pole_add(phys, n, nu_l), 0.0
            );
        }

        finalize_site(phys, n);
    }
}

// Single entry point: dispatch on the (required) expansion method. parse_bath
// calls this instead of a method-specific builder so the rest of the pipeline
// stays method-agnostic.
void compute_bath_correlation(PhysicalParameters& phys) {
    switch (phys.expansion) {
        case BathExpansion::PADE:      compute_pade_drude(phys);      break;
        case BathExpansion::MATSUBARA: compute_matsubara_drude(phys); break;
    }
}

// d^(n)_i = 1 iff Hilbert state i has pigment n excited: its own single
// excitation in the e manifold, plus every |m,p⟩ (m<p) with m==n or p==n in
// the f manifold when the basis has one. Where e and f START is not decided
// here — ManifoldLayout / DoubleExcitedIndex are the single source — so the
// three bases collapse into one code path.
std::vector<int8_t> build_d_coupling(const PhysicalParameters& phys) {
    const int n_states = phys.n_states;
    const int n_sites  = phys.n_sites;
    const auto lay = heom::ManifoldLayout::from(phys.basis, n_states, n_sites);
    std::vector<int8_t> d(n_states * n_sites, 0);

    for (int n = 0; n < n_states; ++n) {
        int8_t* row = &d[n * n_sites];
        row[lay.lo(1) + n] = 1;
        if (lay.empty(2)) continue;
        const DoubleExcitedIndex didx(n_states);
        for (int m = 0; m < n_states; ++m)
            for (int p = m + 1; p < n_states; ++p)
                if (m == n || p == n) row[didx.double_index(m + 1, p + 1)] = 1;
    }
    return d;
}
