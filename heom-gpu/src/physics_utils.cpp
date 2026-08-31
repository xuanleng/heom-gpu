#include "types.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace {

// Relative-gap threshold for declaring a pole degenerate with the Drude pole
// (ν_k ≈ γ). Matches bath_correlation RESONANCE_REL_TOL so the
// C++ throws on exactly the same configs the Python reference rejects.
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

// Pre-size the flat per-site coefficient arrays and pin the final mode layout.
//
// final K (= n_bath_modes_total) = n_extra_poles. Arrays are zero-filled. Returns
// K_drude (= n_extra_poles), the last pole index.
int setup_bath_arrays(PhysicalParameters& phys) {
    const int K_drude = phys.n_extra_poles;
    phys.n_bath_modes_total = K_drude;
    const size_t n = static_cast<size_t>(phys.n_bath());
    phys.c_bath.assign(n * (K_drude + 1), std::complex<double>(0.0, 0.0));
    phys.nu_bath.assign(n * (K_drude + 1), 0.0);
    phys.markov_residue.assign(phys.n_bath(), 0.0);
    return K_drude;
}

// Per-site tail, run AFTER c_{nk}/ν_{nk} for k=0..K_drude are filled:
//   markov_residue[n] = η_n T/γ_n − Σ_{k≤K_drude} Re(c_{nk}/ν_{nk})  (ν=0 skipped)
// Identical form for Matsubara and Padé — only the c_{nk}/ν_{nk} contents differ.
// Markov residue = exact real area minus the retained finite sum.
void finalize_site(PhysicalParameters& phys, int n, int K_drude) {
    double retained = 0.0;
    for (int k = 0; k <= K_drude; ++k) {
        const int idx = phys.mode_idx(n, k);
        if (phys.nu_bath[idx] == 0.0) continue;
        retained += (phys.c_bath[idx] / phys.nu_bath[idx]).real();
    }
    const BathSite& s = phys.bath_sites[n];
    const double residue = s.eta_au * phys.temperature / s.gamma_au - retained;

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
// ALGORITHM IS NOT RE-DERIVED HERE — transcribed line-for-line from the
// authoritative reference: bath_correlation compute_pade_poles
// (tested, regression CSV). Two symmetric tridiagonal matrices (dim 2K and
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
    const int K_drude = setup_bath_arrays(phys);
    const int n_bath  = phys.n_bath();

    for (int n = 0; n < n_bath; ++n) {
        const double gamma_au = phys.bath_sites[n].gamma_au;
        const double eta_au   = phys.bath_sites[n].eta_au;
        set_drude_pole(phys, n);

        // Matsubara poles: ν_k = 2πkT (site-independent), c_{nk} = 2η_nγ_n·T·ν_k/(ν_k²−γ_n²).
        for (int k = 1; k <= K_drude; ++k) {
            const double omg_k = 2.0 * k * PI * T_au;
            check_bath_resonance(omg_k, gamma_au, k, "Matsubara");
            const int idx = phys.mode_idx(n, k);
            phys.nu_bath[idx] = omg_k;
            const double denom = omg_k * omg_k - gamma_au * gamma_au;
            phys.c_bath[idx] = std::complex<double>(
                2.0 * eta_au * gamma_au * T_au * omg_k / denom, 0.0
            );
        }

        finalize_site(phys, n, K_drude);
    }
}

void compute_pade_drude(PhysicalParameters& phys) {
    const double T_au = phys.temperature;
    const int K_drude = setup_bath_arrays(phys);
    const int n_bath  = phys.n_bath();

    // Padé poles ζ_l / weights χ_l are site-independent (depend only on K_drude),
    // so compute them once. Cached on phys for HDF5 /derived cross-checks.
    boson_pade_NmN(K_drude, phys.pade_zeta, phys.pade_chi);

    for (int n = 0; n < n_bath; ++n) {
        const double gamma_au = phys.bath_sites[n].gamma_au;
        const double eta_au   = phys.bath_sites[n].eta_au;
        set_drude_pole(phys, n);

        // ν_l = ζ_l·T (ℏ=1, β=1/T ⇒ ζ·T is the a.u. frequency),
        // c_{nl} = χ_l · 2η_nγ_n·ν_l·T/(ν_l²−γ_n²). Factor 2ηγ (NOT 4ηγ) — see
        // Factor 2*eta*gamma, not 4*eta*gamma.
        for (int l = 1; l <= K_drude; ++l) {
            const double nu_l = phys.pade_zeta[l] * T_au;
            check_bath_resonance(nu_l, gamma_au, l, "Padé");
            const int idx = phys.mode_idx(n, l);
            phys.nu_bath[idx] = nu_l;
            const double denom = nu_l * nu_l - gamma_au * gamma_au;
            phys.c_bath[idx] = std::complex<double>(
                phys.pade_chi[l] * 2.0 * eta_au * gamma_au * nu_l * T_au / denom, 0.0
            );
        }

        finalize_site(phys, n, K_drude);
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

namespace {

// Mark the double-excited columns that pigment n participates in.
//
// Hilbert layout for DOUBLE_EXCITED: column 0 = ground, columns 1..n_states =
// single excitations, columns 1+n_states.. = double excitations |m,p⟩ (m<p)
// enumerated in lexicographic order. Pigment n is excited in |m,p⟩ iff m==n
// or p==n. The double-excited block index is computed in closed form so the
// caller carries no running counter.
void mark_double_excited_columns(int8_t* row, int n, int n_states) {
    const int block_base = 1 + n_states;
    int col = block_base;
    for (int m = 0; m < n_states - 1; ++m) {
        for (int p = m + 1; p < n_states; ++p, ++col) {
            if (m == n || p == n) row[col] = 1;
        }
    }
}

}  // namespace

std::vector<int8_t> build_d_coupling(const PhysicalParameters& phys) {
    const int n_states = phys.n_states;
    const int n_sites  = phys.n_sites;
    std::vector<int8_t> d(n_states * n_sites, 0);

    for (int n = 0; n < n_states; ++n) {
        int8_t* row = &d[n * n_sites];
        switch (phys.basis) {
            case BasisType::EXCITED_ONLY:
                row[n] = 1;
                break;
            case BasisType::WITH_GROUND:
                row[n + 1] = 1;
                break;
            case BasisType::DOUBLE_EXCITED:
                row[n + 1] = 1;  // single-excitation column (shared with WITH_GROUND)
                mark_double_excited_columns(row, n, n_states);
                break;
        }
    }
    return d;
}
