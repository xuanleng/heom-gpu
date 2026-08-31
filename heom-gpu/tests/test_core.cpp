#include "types.hpp"
#include "heom_config.hpp"
#include "hamiltonian_expansion.hpp"
#include "hierarchy_truncation.hpp"
#include "ado_filter.cuh"
#include "heom_derivatives.cuh"
#include "memory_budget.hpp"
#include "gpu_alloc_plan.hpp"
#include "output/spectrum_writer.hpp"
#include "spectra/dipole.hpp"          // DipoleOperator + factory
#include "output/console_printer.hpp"  // format_computation_time tiers
#include "util/stream_format.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cassert>
#include <complex>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#include <type_traits>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    ++tests_run; \
    std::cout << "  " << name << "... " << std::flush; \
} while(0)

#define PASS() do { ++tests_passed; std::cout << "PASS\n"; } while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    double _a = (a), _b = (b); \
    if (std::abs(_a - _b) > (tol)) { \
        std::cout << "FAIL\n    expected " << _b << " got " << _a \
                  << " (diff=" << std::abs(_a-_b) << ")\n"; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        std::cout << "FAIL\n    expected " << _b << " got " << _a << "\n"; \
        return; \
    } \
} while(0)

#define ASSERT_NE(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a == _b) { \
        std::cout << "FAIL\n    expected != " << _b << " but both equal\n"; \
        return; \
    } \
} while(0)

#define ASSERT_THROWS(expr) do { \
    bool caught = false; \
    try { expr; } catch (const std::exception&) { caught = true; } \
    if (!caught) { \
        std::cout << "FAIL\n    expected exception, none thrown\n"; \
        return; \
    } \
} while(0)

// Dual of ASSERT_THROWS (in-house style instead of pulling in Catch2). Used by the
// in-memory Config api-stability test below; reusable by future tests.
#define ASSERT_NOTHROW(expr) do { \
    try { expr; } catch (const std::exception& _e) { \
        std::cout << "FAIL\n    threw std::exception: " << _e.what() << "\n"; \
        return; \
    } catch (...) { \
        std::cout << "FAIL\n    threw unknown exception\n"; \
        return; \
    } \
} while(0)

// ============================================================================
// 6.2 Matsubara expansion
// ============================================================================

// Per-site upgrade: PhysicalParameters no longer carries scalar eta/gamma/sigma.
// Fill p.bath_sites (length n_states — set it FIRST) with a homogeneous Drude
// bath from atomic-unit inputs. These low-level physics tests use n_states=1, so
// the flat [n][k] coefficient layout collapses to [k] (site 0 ⇒ index k) and the
// legacy [k] accesses / size assertions stay valid while exercising the new code.
static void set_homog_bath(PhysicalParameters& p, double eta_au, double gamma_au) {
    BathSite s;
    s.eta_au           = eta_au;
    s.eta_cm           = eta_au / CM_TO_AU;
    s.gamma_au         = gamma_au;
    s.gamma_inverse_fs = 1.0 / (gamma_au * FS_TO_AU);
    p.bath_sites.assign(p.n_states, s);
}

static PhysicalParameters make_fmo7_params() {
    PhysicalParameters p;
    p.temperature = 300.0 * BOLTZMANN;
    p.n_bath_modes_total = 5;
    p.n_extra_poles = 5;   // Snapshot user input
    p.n_states = 1;        // single-bath: flat layout == [K+1]
    p.n_sites = 1;
    p.basis = BasisType::EXCITED_ONLY;
    set_homog_bath(p, 35.0 * CM_TO_AU, 1.0 / (166.0 * FS_TO_AU));
    compute_matsubara_drude(p);
    return p;
}

static void test_bath_modes_no_disorder_slot() {
    TEST("bath modes: n_bath_modes_total == n_extra_poles (no disorder slot)");
    PhysicalParameters p;
    p.temperature = 77.0 * BOLTZMANN; p.n_states = 2; p.n_sites = 2;
    p.basis = BasisType::EXCITED_ONLY; p.n_extra_poles = 3; p.n_bath_modes_total = 3;
    set_homog_bath(p, 35.0 * CM_TO_AU, 1.0 / (100.0 * FS_TO_AU));
    compute_matsubara_drude(p);
    ASSERT_EQ(p.n_bath_modes_total, 3);   // locks: no disorder slot ⇒ always == K_drude
    PASS();
}

static void test_matsubara_nu0() {
    TEST("matsubara ν_0 = γ");
    auto p = make_fmo7_params();
    ASSERT_NEAR(p.nu_bath[0], p.bath_sites[0].gamma_au, 1e-15);
    PASS();
}

static void test_matsubara_nuk() {
    TEST("matsubara ν_k = 2πkT (k≥1)");
    auto p = make_fmo7_params();
    for (int k = 1; k <= 5; ++k)
        ASSERT_NEAR(p.nu_bath[k], 2.0 * k * PI * p.temperature, 1e-15);
    PASS();
}

static void test_matsubara_c0_imaginary() {
    TEST("matsubara Im(c_0) = -η·γ/2");
    auto p = make_fmo7_params();
    ASSERT_NEAR(p.c_bath[0].imag(),
                -0.5 * p.bath_sites[0].eta_au * p.bath_sites[0].gamma_au, 1e-15);
    PASS();
}

static void test_matsubara_ck_real() {
    TEST("matsubara c_k (k≥1) is real");
    auto p = make_fmo7_params();
    for (int k = 1; k <= 5; ++k)
        ASSERT_NEAR(p.c_bath[k].imag(), 0.0, 1e-30);
    PASS();
}

static void test_matsubara_markov_positive() {
    TEST("matsubara Markov term ≥ 0");
    auto p = make_fmo7_params();
    if (p.markov_residue[0] < -1e-15) {
        std::cout << "FAIL\n    markov_residue = " << p.markov_residue[0] << " < 0\n";
        return;
    }
    PASS();
}

// Negative test: ω_k = 2πkT and γ collapse to the same frequency, so
// Drude–Matsubara expansion has both a vanishing denominator (numerical)
// and a degenerate ν_0=γ vs ν_k=ω_k pair (physical). compute_matsubara_drude
// must throw before c[k] gets to NaN-poison every ADO via cudaMemcpy.
// Construct the exact-resonance case for k=1 by setting γ = 2π·T directly.
static void test_matsubara_resonance_k1_throws() {
    TEST("matsubara resonance ω_1 ≈ γ throws (no NaN c[1])");
    PhysicalParameters p;
    p.temperature_K    = 77.0;
    p.temperature      = 77.0 * BOLTZMANN;
    p.n_bath_modes_total      = 5;
    p.n_extra_poles = 5;
    p.n_states = 1;
    p.n_sites  = 1;
    p.basis    = BasisType::EXCITED_ONLY;
    set_homog_bath(p, 35.0 * CM_TO_AU, 2.0 * PI * p.temperature);  // exact ω_1 = γ
    ASSERT_THROWS(compute_matsubara_drude(p));
    PASS();
}

// k=2 resonance: γ = 4π·T. Same expansion failure but at a higher index;
// proves the loop's check covers the whole k=1..K sweep, not just k=1.
static void test_matsubara_resonance_k2_throws() {
    TEST("matsubara resonance ω_2 ≈ γ throws (loop coverage)");
    PhysicalParameters p;
    p.temperature_K    = 77.0;
    p.temperature      = 77.0 * BOLTZMANN;
    p.n_bath_modes_total      = 5;
    p.n_extra_poles = 5;
    p.n_states = 1;
    p.n_sites  = 1;
    p.basis    = BasisType::EXCITED_ONLY;
    set_homog_bath(p, 35.0 * CM_TO_AU, 4.0 * PI * p.temperature);  // exact ω_2 = γ
    ASSERT_THROWS(compute_matsubara_drude(p));
    PASS();
}

// Positive test: FMO-7 default parameters (T=300K, γ_inv=166fs) sit far
// from any resonance — relative gap at k=1 is ~70×, well above the 1e-3
// physical-separation threshold. This guards against the resonance check
// accidentally tightening the threshold and rejecting valid configs.
static void test_matsubara_no_resonance_default_params() {
    TEST("matsubara: default FMO-7 params do not trigger resonance check");
    ASSERT_NOTHROW(make_fmo7_params());
    PASS();
}

// ============================================================================
// Padé [K-1/K] expansion. The pole/weight algorithm is NOT
// re-derived here — these gates pin boson_pade_NmN / compute_pade_drude to the
// authoritative reference bath_correlation helpers and the shared
// Drude pole + Markov-residue contract with the Matsubara path.
// ============================================================================

// Builds a Padé-mode phys with K = n_extra_poles = n_bath_modes_total. Disorder
// off (so n_bath_modes_total stays K). Drude bath in atomic units.
static PhysicalParameters make_pade_params(int K, double T_K,
                                           double eta_cm, double ginv_fs) {
    PhysicalParameters p;
    p.temperature_K    = T_K;
    p.temperature      = T_K * BOLTZMANN;
    p.n_bath_modes_total = K;
    p.n_extra_poles      = K;
    p.n_states = 1;
    p.n_sites  = 1;
    p.basis    = BasisType::EXCITED_ONLY;
    set_homog_bath(p, eta_cm * CM_TO_AU, 1.0 / (ginv_fs * FS_TO_AU));
    p.expansion = BathExpansion::PADE;
    return p;
}

// K=1 anchor: ζ_1²=60, χ_1=5/2 (HEOM-QUICK2 / tanimura-gpu reference). Exact
// rationals, so 1e-9 is generous.
static void test_pade_poles_match_reference_K1() {
    TEST("Padé poles K=1 match anchor (ζ_1²=60, χ_1=5/2)");
    std::vector<double> z, c;
    boson_pade_NmN(1, z, c);
    ASSERT_EQ((int)z.size(), 2);   // length K+1, [0]=Drude placeholder
    ASSERT_EQ((int)c.size(), 2);
    ASSERT_NEAR(z[0], 0.0, 1e-15);
    ASSERT_NEAR(z[1] * z[1], 60.0, 1e-9);
    ASSERT_NEAR(c[1], 2.5, 1e-9);
    PASS();
}

// K=2 anchor (8 digits).
static void test_pade_poles_match_reference_K2() {
    TEST("Padé poles K=2 match the reference anchors");
    std::vector<double> z, c;
    boson_pade_NmN(2, z, c);
    ASSERT_EQ((int)z.size(), 3);
    ASSERT_NEAR(z[1] * z[1], 39.76486849, 1e-6);
    ASSERT_NEAR(z[2] * z[2], 380.23513151, 1e-6);
    ASSERT_NEAR(c[1], 1.03282418, 1e-6);
    ASSERT_NEAR(c[2], 5.96717582, 1e-6);
    PASS();
}

// K=5, T=77K, γ=53 cm⁻¹: every bath frequency (ν_0=γ plus the 5 PSD poles)
// must be pairwise distinct AND separated from γ. Guards both the per-pole
// resonance check and the eigenvalue distinctness the kernel's ±1-jump relies on.
static void test_pade_poles_distinct() {
    TEST("Padé K=5 poles distinct & ≠ γ (T=77K, γ=53cm⁻¹)");
    PhysicalParameters p = make_pade_params(5, 77.0, 35.0, 100.0);
    set_homog_bath(p, 35.0 * CM_TO_AU, 53.0 * CM_TO_AU);  // override: γ fixed in cm⁻¹
    ASSERT_NOTHROW(compute_pade_drude(p));
    const auto& nu = p.nu_bath;
    double maxnu = 0.0;
    for (double v : nu) maxnu = std::max(maxnu, v);
    for (size_t i = 0; i < nu.size(); ++i)
        for (size_t j = i + 1; j < nu.size(); ++j) {
            double gap = std::abs(nu[i] - nu[j]) / maxnu;
            if (gap < 1e-3) {
                std::cout << "FAIL\n    ν[" << i << "]=" << nu[i]
                          << " ≈ ν[" << j << "]=" << nu[j]
                          << " (rel gap " << gap << ")\n";
                return;
            }
        }
    PASS();
}

// Builds a Drude bath under the requested expansion with K poles. Used by the
// K=0 degeneracy and high-K convergence gates.
static PhysicalParameters build_bath(BathExpansion e, int K, double T_K) {
    PhysicalParameters p = make_pade_params(K, T_K, 35.0, 100.0);
    p.expansion = e;
    compute_bath_correlation(p);
    return p;
}

// K=0 leaves only the Drude pole + Markov residue, identical code path for both
// expansions — must be byte-equal (exact ==), not merely close.
static void test_pade_K0_equals_matsubara_K0() {
    TEST("Padé K=0 == Matsubara K=0 (Drude-only, byte-equal)");
    auto pm = build_bath(BathExpansion::MATSUBARA, 0, 77.0);
    auto pp = build_bath(BathExpansion::PADE, 0, 77.0);
    ASSERT_EQ((int)pp.c_bath.size(), (int)pm.c_bath.size());
    ASSERT_EQ((int)pp.nu_bath.size(), (int)pm.nu_bath.size());
    ASSERT_EQ(pp.nu_bath[0], pm.nu_bath[0]);
    ASSERT_EQ(pp.c_bath[0].real(), pm.c_bath[0].real());
    ASSERT_EQ(pp.c_bath[0].imag(), pm.c_bath[0].imag());
    ASSERT_EQ(pp.markov_residue[0], pm.markov_residue[0]);
    PASS();
}

// Both schemes expand the SAME exact C(t). On the window where BOTH have
// converged they must agree to 1e-6 RELATIVE. The window starts at t=5 fs, not
// t=0, on purpose and this is the whole point of Padé:
//   - At fixed K, Matsubara converges only as ~1/K and its error is largest at
//     SHORT time (high-frequency content). At K=20, T=300K it still carries
//     ~1e-4 relative error near t≈1 fs, whereas the Padé [K-1/K] sum is already
//     essentially exact there. The short-time gap is Matsubara's deficiency,
//     not a discrepancy to mask — so the agreement gate lives on the converged
//     tail (t≥5 fs), where "both expand the same bath" is the real claim.
//   - t=0 is additionally ill-posed: the finite sum is a partial sum of the
//     (log-divergent) Matsubara tail whose remainder is carried by the separate
//     Markov δ-term, so C(0) is scheme-defined, not a physical observable.
// Relative norm only — the absolute scale drifts with ηT/γ.
static void test_pade_K_high_converges_to_matsubara_K_high() {
    TEST("Padé vs Matsubara C(t) agree on converged window K=20 (rel ≤ 1e-6)");
    auto pm = build_bath(BathExpansion::MATSUBARA, 20, 300.0);
    auto pp = build_bath(BathExpansion::PADE, 20, 300.0);
    auto Ct = [](const PhysicalParameters& p, double t) {
        std::complex<double> s(0.0, 0.0);
        for (size_t k = 0; k < p.c_bath.size(); ++k)
            s += p.c_bath[k] * std::exp(-p.nu_bath[k] * t);
        return s;
    };
    double max_diff = 0.0, max_ref = 0.0;
    for (double t_fs = 5.0; t_fs <= 200.0 + 1e-9; t_fs += 1.0) {
        const double t = t_fs * FS_TO_AU;
        max_diff = std::max(max_diff, std::abs(Ct(pp, t) - Ct(pm, t)));
        max_ref  = std::max(max_ref, std::abs(Ct(pm, t)));
    }
    ASSERT_NEAR(max_diff / max_ref, 0.0, 1e-6);
    PASS();
}

// T-0 physics proof, now permanently owned by C++ (replaces the throwaway Python
// prototype). (a) markov_residue is real (double) — compile-time; (b) equals
// ηT/γ − Σ Re(c_k/ν_k); (c) the Drude pole's imaginary part Im(c_0/ν_0) = −η/2
// is fully absorbed by the Drude term (never leaks into Markov); (d) |residue|
// shrinks toward 0 as K grows. This is the compliance gate against any future PR
// smuggling a complex 2ηT/γ−iη residue into the kernel.
static void test_pade_markov_residue_real_only() {
    TEST("Padé Markov residue: real, formula, Drude Im=−η/2, K-converges");
    static_assert(
        std::is_same<decltype(PhysicalParameters::markov_residue)::value_type,
                     double>::value,
        "per-site markov_residue elements must stay real (double) — never complex");
    const int Ks[] = {0, 1, 2, 5};
    for (int K : Ks) {
        PhysicalParameters p = make_pade_params(K, 77.0, 35.0, 100.0);
        compute_pade_drude(p);
        double retained = 0.0;
        for (int k = 0; k <= K; ++k) {
            if (p.nu_bath[k] == 0.0) continue;
            retained += (p.c_bath[k] / p.nu_bath[k]).real();
        }
        const double eta = p.bath_sites[0].eta_au;
        const double gamma = p.bath_sites[0].gamma_au;
        const double expect = eta * p.temperature / gamma - retained;
        ASSERT_NEAR(p.markov_residue[0], expect, 1e-15);
        const double drude_im = (p.c_bath[0] / p.nu_bath[0]).imag();
        ASSERT_NEAR(drude_im, -0.5 * eta, 1e-15);
    }
    PhysicalParameters p0 = make_pade_params(0, 77.0, 35.0, 100.0);
    PhysicalParameters p5 = make_pade_params(5, 77.0, 35.0, 100.0);
    compute_pade_drude(p0);
    compute_pade_drude(p5);
    if (std::abs(p5.markov_residue[0]) > std::abs(p0.markov_residue[0]) + 1e-12) {
        std::cout << "FAIL\n    |residue(K=5)|=" << std::abs(p5.markov_residue[0])
                  << " > |residue(K=0)|=" << std::abs(p0.markov_residue[0]) << "\n";
        return;
    }
    PASS();
}

// ============================================================================
// 6.3 Hamiltonian expansion
// ============================================================================

static void test_excited_only() {
    TEST("expand_hamiltonian EXCITED_ONLY: identity transform");
    Eigen::MatrixXcd H(3, 3);
    H << 1.0, 0.1, 0.2,
         0.1, 2.0, 0.3,
         0.2, 0.3, 3.0;
    auto H_out = expand_hamiltonian(H, BasisType::EXCITED_ONLY);
    ASSERT_EQ(H_out.rows(), 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(std::abs(H_out(i,j) - H(i,j)), 0.0, 1e-15);
    PASS();
}

static void test_with_ground() {
    TEST("expand_hamiltonian WITH_GROUND: dim = n+1, ground row/col zero");
    Eigen::MatrixXcd H(3, 3);
    H << 1.0, 0.1, 0.2,
         0.1, 2.0, 0.3,
         0.2, 0.3, 3.0;
    auto H_out = expand_hamiltonian(H, BasisType::WITH_GROUND);
    ASSERT_EQ(H_out.rows(), 4);
    ASSERT_EQ(H_out.cols(), 4);
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(std::abs(H_out(0,i)), 0.0, 1e-15);
        ASSERT_NEAR(std::abs(H_out(i,0)), 0.0, 1e-15);
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(std::abs(H_out(i+1,j+1) - H(i,j)), 0.0, 1e-15);
    PASS();
}

static void test_double_excited_dim() {
    TEST("expand_hamiltonian DOUBLE_EXCITED: dim = 1 + n + n(n-1)/2");
    Eigen::MatrixXcd H(3, 3);
    H << 1.0, 0.1, 0.2,
         0.1, 2.0, 0.3,
         0.2, 0.3, 3.0;
    auto H_out = expand_hamiltonian(H, BasisType::DOUBLE_EXCITED);
    int expected = 1 + 3 + 3 * 2 / 2;  // 1 + 3 + 3 = 7
    ASSERT_EQ(H_out.rows(), expected);
    ASSERT_EQ(H_out.cols(), expected);
    ASSERT_NEAR(std::abs(H_out(0,0)), 0.0, 1e-15);
    PASS();
}

// ============================================================================
// 6.4 Hierarchy index generation
// ============================================================================

static void test_hierarchy_count_trivial() {
    TEST("hierarchy n_bath=2,K=0,L=2: count = C(2+2,2) = 6");
    PhysicalParameters p;
    p.n_states = 2;
    p.n_sites = 2;
    p.basis = BasisType::EXCITED_ONLY;
    p.n_bath_modes_total = 0;
    HierarchyTruncation ht(p, 2);
    auto indices = ht.generate_hierarchy_indices();
    ASSERT_EQ(static_cast<int>(indices.size()), 6);
    PASS();
}

static void test_hierarchy_simple_cutoff_count() {
    TEST("hierarchy: N=2 K_drude=0 L_cutoff=2 → Σα≤2 gives exactly 6 ADOs");
    PhysicalParameters p;
    p.n_states = 2; p.n_sites = 2; p.basis = BasisType::EXCITED_ONLY;
    p.n_bath_modes_total = 0; p.n_extra_poles = 0;
    HierarchyTruncation ht(p, /*L_cutoff=*/2);
    auto idx = ht.generate_hierarchy_indices();
    ASSERT_EQ((int)idx.size(), 6);
    PASS();
}

static void test_hierarchy_zero_index() {
    TEST("hierarchy: first index is zero vector");
    PhysicalParameters p;
    p.n_states = 3;
    p.n_sites = 3;
    p.basis = BasisType::EXCITED_ONLY;
    p.n_bath_modes_total = 1;
    HierarchyTruncation ht(p, 1);
    auto indices = ht.generate_hierarchy_indices();
    auto zero = indices[0];
    ASSERT_EQ(zero.sum(), 0);
    for (int i = 0; i < zero.get_size(); ++i)
        ASSERT_EQ(zero[i], 0);
    PASS();
}

static void test_hierarchy_neighbor_maps() {
    TEST("hierarchy neighbors: α=0 plus exists, minus = -1");
    PhysicalParameters p;
    p.n_states = 2;
    p.n_sites = 2;
    p.basis = BasisType::EXCITED_ONLY;
    p.n_bath_modes_total = 0;
    HierarchyTruncation ht(p, 2);
    auto indices = ht.generate_hierarchy_indices();
    auto [plus_map, minus_map] = ht.build_neighbor_maps(indices);

    int n_bath = 2, K = 0;
    bool any_plus = false, all_minus_neg = true;
    for (int n = 0; n < n_bath; ++n) {
        for (int k = 0; k <= K; ++k) {
            int idx = 0 * n_bath * (K + 1) + n * (K + 1) + k;
            if (plus_map[idx] >= 0) any_plus = true;
            if (minus_map[idx] >= 0) all_minus_neg = false;
        }
    }
    if (!any_plus) { std::cout << "FAIL\n    no plus neighbor for α=0\n"; return; }
    if (!all_minus_neg) { std::cout << "FAIL\n    minus neighbor exists for α=0\n"; return; }
    PASS();
}

static void test_hierarchy_sorted() {
    TEST("hierarchy: indices are sorted");
    PhysicalParameters p;
    p.n_states = 3;
    p.n_sites = 3;
    p.basis = BasisType::EXCITED_ONLY;
    p.n_bath_modes_total = 1;
    HierarchyTruncation ht(p, 2);
    auto indices = ht.generate_hierarchy_indices();
    for (size_t i = 1; i < indices.size(); ++i) {
        if (!(indices[i-1] < indices[i]) && !(indices[i-1] == indices[i])) {
            std::cout << "FAIL\n    indices not sorted at position " << i << "\n";
            return;
        }
    }
    PASS();
}

// ============================================================================
// 6.5 d_coupling construction (n_hat_n bath coupling)
// ============================================================================

static void test_d_coupling_excited_only() {
    TEST("d_coupling EXCITED_ONLY: d^(n)_i = δ_{ni}");
    PhysicalParameters p;
    p.n_states = 3;
    p.n_sites = 3;
    p.basis = BasisType::EXCITED_ONLY;
    auto d = build_d_coupling(p);
    ASSERT_EQ(static_cast<int>(d.size()), 9);
    for (int n = 0; n < 3; ++n)
        for (int i = 0; i < 3; ++i)
            ASSERT_EQ(d[n * 3 + i], (n == i) ? 1 : 0);
    PASS();
}

static void test_d_coupling_with_ground() {
    TEST("d_coupling WITH_GROUND: d^(n)_i = δ_{i,n+1}");
    PhysicalParameters p;
    p.n_states = 3;
    p.n_sites = 4;
    p.basis = BasisType::WITH_GROUND;
    auto d = build_d_coupling(p);
    ASSERT_EQ(static_cast<int>(d.size()), 12);
    for (int n = 0; n < 3; ++n) {
        ASSERT_EQ(d[n * 4 + 0], 0);
        for (int i = 1; i < 4; ++i)
            ASSERT_EQ(d[n * 4 + i], (i == n + 1) ? 1 : 0);
    }
    PASS();
}

static void test_d_coupling_double_excited() {
    TEST("d_coupling DOUBLE_EXCITED 3-site: verify d vectors");
    PhysicalParameters p;
    p.n_states = 3;
    p.n_sites = 7;
    p.basis = BasisType::DOUBLE_EXCITED;
    auto d = build_d_coupling(p);
    ASSERT_EQ(static_cast<int>(d.size()), 21);

    // Basis: |g>=0, |1>=1, |2>=2, |3>=3, |1,2>=4, |1,3>=5, |2,3>=6
    // d^(0) (site 0): excited in |1>=1, |1,2>=4, |1,3>=5
    ASSERT_EQ(d[0*7+0], 0); ASSERT_EQ(d[0*7+1], 1); ASSERT_EQ(d[0*7+2], 0);
    ASSERT_EQ(d[0*7+3], 0); ASSERT_EQ(d[0*7+4], 1); ASSERT_EQ(d[0*7+5], 1);
    ASSERT_EQ(d[0*7+6], 0);
    // d^(1) (site 1): excited in |2>=2, |1,2>=4, |2,3>=6
    ASSERT_EQ(d[1*7+0], 0); ASSERT_EQ(d[1*7+1], 0); ASSERT_EQ(d[1*7+2], 1);
    ASSERT_EQ(d[1*7+3], 0); ASSERT_EQ(d[1*7+4], 1); ASSERT_EQ(d[1*7+5], 0);
    ASSERT_EQ(d[1*7+6], 1);
    // d^(2) (site 2): excited in |3>=3, |1,3>=5, |2,3>=6
    ASSERT_EQ(d[2*7+0], 0); ASSERT_EQ(d[2*7+1], 0); ASSERT_EQ(d[2*7+2], 0);
    ASSERT_EQ(d[2*7+3], 1); ASSERT_EQ(d[2*7+4], 0); ASSERT_EQ(d[2*7+5], 1);
    ASSERT_EQ(d[2*7+6], 1);
    PASS();
}

static void test_hierarchy_with_ground_uses_n_states() {
    TEST("hierarchy WITH_GROUND: dim = n_states*(K+1), not n_sites*(K+1)");
    PhysicalParameters p;
    p.n_states = 3;
    p.n_sites = 4;
    p.basis = BasisType::WITH_GROUND;
    p.n_bath_modes_total = 0;
    HierarchyTruncation ht(p, 1);
    auto indices = ht.generate_hierarchy_indices();
    // n_bath = n_states = 3, K=0, L=1 → 1 + 3 = 4 ADOs
    ASSERT_EQ(static_cast<int>(indices.size()), 4);
    // Each index vector has size n_states*(K+1) = 3
    ASSERT_EQ(indices[0].get_size(), 3);
    PASS();
}

// ============================================================================
// Sink site mapping tests
// ============================================================================

static std::string write_temp_yaml(const std::string& basis, bool sink_enabled,
                                    int target_site, int output_interval = 10,
                                    int print_interval = 100,
                                    int excited_site = 1,
                                    double sink_rate = 0.001) {
    std::string path = "/tmp/test_heom_config.yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 7\n"
      << "  basis: \"" << basis << "\"\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12400, -106,    8,   -5,    6,   -8,   -4]\n"
      << "    - [ -106, 12540,  28,    6,    2,   13,    1]\n"
      << "    - [    8,   28, 12120, -62,   -1,   -9,   17]\n"
      << "    - [   -5,    6,  -62, 12295, -70,  -19,  -57]\n"
      << "    - [    6,    2,   -1,  -70, 12440,  40,   -2]\n"
      << "    - [   -8,   13,   -9,  -19,   40, 12480,  32]\n"
      << "    - [   -4,    1,   17,  -57,   -2,   32, 12380]\n"
      << "  w0: 12000.0\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 70.0\n  gamma_inverse: 100.0\n  expansion_method: matsubara\n  n_extra_poles: 1\n"
      << "  L_cutoff: 1\n"
      << "filter:\n"
      << "  enabled: true\n  tolerance: 1.0e-7\n  start_step: 10\n  warmup_factor: 0.001\n"
      << "dynamics:\n"
      << "  integrator: rk4\n"
      << "  dt: 2.0\n  n_steps: 100\n"
      << "  output_interval: " << output_interval << "\n"
      << "  print_interval: " << print_interval << "\n"
      << "  sink:\n"
      << "    enabled: " << (sink_enabled ? "true" : "false") << "\n"
      << "    target_site: " << target_site << "\n"
      << "    rate: " << sink_rate << "\n"
      << "  initial_state:\n    type: \"site_excitation\"\n    excited_site: " << excited_site << "\n"
      << "  output:\n    save_path: \"/tmp/\"\n    use_timestamp: false\n";
    f.close();
    return path;
}

using HEOMConfig::Config;

// Task blocks (dynamics/absorption/2des) are parsed lazily by their
// resolve_<task>_config(): load_yaml() only fills the shared blocks. A test
// inspecting cfg.time / cfg.initial / cfg.output (or exercising a dynamics-block
// throw path — sink / initial_state / time grid) must run both stages. This
// helper mirrors main.cpp's dynamics dispatch (load then resolve).
static Config load_dyn(const std::string& path) {
    auto cfg = Config::load_yaml(path);
    resolve_dynamics_config(cfg);
    return cfg;
}

// ============================================================================
// Padé / expansion_method YAML contract. expansion_method is a
// hard-required field with no implicit default and no legacy n_matsubara shim.
// ============================================================================

// Minimal valid dynamics yaml whose bath-correlation lines are injected raw, so
// negative tests can drop/mangle expansion_method and n_extra_poles at will.
static std::string write_corr_yaml(const std::string& corr_block) {
    std::string path = "/tmp/test_heom_corr.yaml";
    std::ofstream f(path);
    f << "system:\n  n_states: 2\n  basis: \"excited_only\"\n  temperature: 77.0\n"
      << "  hamiltonian:\n    - [12100, -50]\n    - [-50, 12300]\n  w0: 12000.0\n"
      << "bath:\n  spectral_density: \"Drude-Lorentz\"\n  eta: 35.0\n  gamma_inverse: 100.0\n"
      << corr_block         // injects expansion_method / n_extra_poles (bath fields)
      << "  L_cutoff: 1\n"
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

// Minimal with_ground absorption yaml (mirrors config/abs_dimer.yaml), filename
// auto so resolve_absorption_config exercises the method suffix.
static std::string write_abs_yaml(const std::string& method) {
    std::string path = "/tmp/test_heom_abs.yaml";
    std::ofstream f(path);
    f << "system:\n  n_states: 2\n  basis: with_ground\n  temperature: 77.0\n"
      << "  hamiltonian:\n    - [12100.0, -50.0]\n    - [-50.0, 12300.0]\n  w0: 12000.0\n"
      << "  dipoles:\n    - [1.0, 0.0, 0.0]\n    - [0.0, 1.0, 0.0]\n"
      << "  strengths: [4.5, 4.5]\n"
      << "bath:\n  spectral_density: \"Drude-Lorentz\"\n  eta: 35.0\n  gamma_inverse: 100.0\n"
      << "  expansion_method: " << method << "\n  n_extra_poles: 1\n  L_cutoff: 3\n"
      << "filter:\n  enabled: true\n  tolerance: 1.0e-7\n"
      << "absorption:\n  integrator: rk4\n  dt: 1.0\n  n_steps: 200\n  output_interval: 1\n"
      << "  print_interval: 200\n"
      << "  output:\n    save_path: \"/tmp/\"\n    filename: \"auto\"\n    use_timestamp: false\n";
    f.close();
    return path;
}

static void test_pade_yaml_invalid_value_throws() {
    TEST("expansion_method: gauss → throws");
    ASSERT_THROWS(Config::load_yaml(
        write_corr_yaml("  expansion_method: gauss\n  n_extra_poles: 1\n")));
    PASS();
}

// Missing field must throw with a message naming the requirement and both legal
// values — guards against a future PR silently restoring an implicit default.
static void test_expansion_method_missing_throws() {
    TEST("missing expansion_method → throws (is required / pade / matsubara)");
    bool ok = false;
    try {
        Config::load_yaml(write_corr_yaml("  n_extra_poles: 1\n"));
    } catch (const std::exception& e) {
        const std::string m = e.what();
        ok = m.find("is required") != std::string::npos
          && m.find("pade") != std::string::npos
          && m.find("matsubara") != std::string::npos;
    }
    if (!ok) {
        std::cout << "FAIL\n    expected 'is required'+'pade'+'matsubara' message\n";
        return;
    }
    PASS();
}

// Legacy bath.n_matsubara (no n_extra_poles) is rejected — no fallback shim.
static void test_n_matsubara_field_rejected() {
    TEST("legacy bath.n_matsubara rejected ('n_extra_poles is required')");
    bool ok = false;
    try {
        Config::load_yaml(
            write_corr_yaml("  expansion_method: matsubara\n  n_matsubara: 1\n"));
    } catch (const std::exception& e) {
        ok = std::string(e.what()).find("n_extra_poles is required")
             != std::string::npos;
    }
    if (!ok) {
        std::cout << "FAIL\n    expected 'n_extra_poles is required'\n";
        return;
    }
    PASS();
}

// Range [0,50] enforced in parse_bath, BEFORE compute_bath_correlation — so an
// out-of-range count never reaches the Padé matrix builder.
static void test_n_extra_poles_range_checked() {
    TEST("n_extra_poles out of [0,50] throws at parse");
    ASSERT_THROWS(Config::load_yaml(
        write_corr_yaml("  expansion_method: matsubara\n  n_extra_poles: -1\n")));
    ASSERT_THROWS(Config::load_yaml(
        write_corr_yaml("  expansion_method: pade\n  n_extra_poles: 51\n")));
    PASS();
}

// ============================================================================
// integrator YAML contract: REQUIRED per-task field, no silent default. The key
// lives in the task time-grid block, so the throw fires at resolve_<task>(),
// not at load_yaml(). "commented-out pseudo-option" is not allowed — a missing
// integrator must force an explicit user choice.
// ============================================================================

// Minimal dynamics config whose dynamics-block `integrator` line is injectable
// (pass "" to omit the key entirely).
static std::string write_integrator_dyn_yaml(const std::string& integ_line) {
    std::string path = "/tmp/test_heom_integrator.yaml";
    std::ofstream f(path);
    f << "system:\n  n_states: 2\n  basis: \"excited_only\"\n  temperature: 77.0\n"
      << "  hamiltonian:\n    - [12100, -50]\n    - [-50, 12300]\n  w0: 12000.0\n"
      << "bath:\n  spectral_density: \"Drude-Lorentz\"\n  eta: 35.0\n  gamma_inverse: 100.0\n"
      << "  expansion_method: matsubara\n  n_extra_poles: 1\n  L_cutoff: 1\n"
      << "filter:\n  enabled: false\n"
      << "dynamics:\n"
      << integ_line                        // e.g. "  integrator: rk4\n" or ""
      << "  dt: 2.0\n  n_steps: 100\n  output_interval: 10\n  print_interval: 100\n"
      << "  initial_state:\n    type: \"site_excitation\"\n    excited_site: 1\n"
      << "  output:\n    save_path: \"/tmp/\"\n    use_timestamp: false\n";
    f.close();
    return path;
}

// Missing integrator must throw at resolve, naming the field and both values.
static void test_integrator_missing_throws() {
    TEST("missing integrator → throws (requires 'rk4' or 'if-rk4')");
    bool ok = false;
    try {
        auto cfg = Config::load_yaml(write_integrator_dyn_yaml(""));
        resolve_dynamics_config(cfg);
    } catch (const std::exception& e) {
        const std::string m = e.what();
        ok = m.find("integrator") != std::string::npos
          && m.find("rk4") != std::string::npos
          && m.find("if-rk4") != std::string::npos;
    }
    if (!ok) {
        std::cout << "FAIL\n    expected 'integrator'+'rk4'+'if-rk4' message\n";
        return;
    }
    PASS();
}

// Explicit rk4 / if-rk4 parse to the matching enum.
static void test_integrator_explicit_values_parse() {
    TEST("integrator: rk4 → Rk4 ; if-rk4 → IfRk4");
    auto cfg_r = load_dyn(write_integrator_dyn_yaml("  integrator: rk4\n"));
    ASSERT_EQ(cfg_r.time.integrator == Integrator::Rk4, true);
    auto cfg_i = load_dyn(write_integrator_dyn_yaml("  integrator: if-rk4\n"));
    ASSERT_EQ(cfg_i.time.integrator == Integrator::IfRk4, true);
    PASS();
}

// An unknown integrator string is rejected.
static void test_integrator_invalid_value_throws() {
    TEST("integrator: rk5 → throws");
    ASSERT_THROWS({
        auto cfg = Config::load_yaml(write_integrator_dyn_yaml("  integrator: rk5\n"));
        resolve_dynamics_config(cfg);
    });
    PASS();
}

// Matsubara auto-names carry a trailing _matsubara; Padé carries no method
// marker. Verified for both dynamics and absorption auto-naming paths so a Padé
// re-run never silently overwrites a Matsubara baseline (and vice versa).
static void test_auto_filename_marks_matsubara() {
    TEST("auto-name: Matsubara→_matsubara, Padé→none (dynamics+absorption)");
    {
        auto cfg_m = load_dyn(
            write_corr_yaml("  expansion_method: matsubara\n  n_extra_poles: 1\n"));
        const std::string nm = resolve_dynamics_auto_filename(cfg_m);
        if (nm.size() < 10 || nm.substr(nm.size() - 10) != "_matsubara") {
            std::cout << "FAIL\n    dynamics Matsubara name '" << nm
                      << "' lacks _matsubara suffix\n";
            return;
        }
        auto cfg_p = load_dyn(
            write_corr_yaml("  expansion_method: pade\n  n_extra_poles: 1\n"));
        const std::string np = resolve_dynamics_auto_filename(cfg_p);
        if (np.find("_matsubara") != std::string::npos) {
            std::cout << "FAIL\n    dynamics Padé name '" << np
                      << "' wrongly carries _matsubara\n";
            return;
        }
    }
    {
        auto cfg_m = Config::load_yaml(write_abs_yaml("matsubara"));
        resolve_absorption_config(cfg_m);
        if (cfg_m.output.filename.find("_matsubara") == std::string::npos) {
            std::cout << "FAIL\n    absorption Matsubara name '"
                      << cfg_m.output.filename << "' lacks _matsubara\n";
            return;
        }
        auto cfg_p = Config::load_yaml(write_abs_yaml("pade"));
        resolve_absorption_config(cfg_p);
        if (cfg_p.output.filename.find("_matsubara") != std::string::npos) {
            std::cout << "FAIL\n    absorption Padé name '"
                      << cfg_p.output.filename << "' wrongly carries _matsubara\n";
            return;
        }
    }
    PASS();
}

static void test_sink_site_excited_only() {
    TEST("sink_site EXCITED_ONLY: target_site=3 → index 2");
    auto path = write_temp_yaml("excited_only", true, 3);
    auto cfg = load_dyn(path);
    ASSERT_EQ(cfg.physics.sink.site, 2);
    PASS();
}

static void test_sink_site_with_ground() {
    TEST("sink_site WITH_GROUND: target_site=3 → index 3 (ground at 0)");
    auto path = write_temp_yaml("with_ground", true, 3);
    auto cfg = load_dyn(path);
    ASSERT_EQ(cfg.physics.sink.site, 3);
    PASS();
}

static void test_sink_site_double_excited() {
    TEST("sink_site DOUBLE_EXCITED: target_site=3 → index 3 (ground at 0)");
    auto path = write_temp_yaml("double_excited", true, 3);
    auto cfg = load_dyn(path);
    ASSERT_EQ(cfg.physics.sink.site, 3);
    PASS();
}

static void test_sink_site_out_of_range() {
    TEST("sink_site: target_site=0 throws");
    ASSERT_THROWS(load_dyn(write_temp_yaml("excited_only", true, 0)));
    PASS();
}

static void test_sink_site_out_of_range_high() {
    TEST("sink_site: target_site=8 throws (n_states=7)");
    ASSERT_THROWS(load_dyn(write_temp_yaml("excited_only", true, 8)));
    PASS();
}

static void test_sink_enabled_zero_rate_throws() {
    TEST("sink: enabled sink with rate=0 throws");
    ASSERT_THROWS(load_dyn(
        write_temp_yaml("excited_only", true, 3, 10, 100, 1, /*sink_rate=*/0.0)));
    PASS();
}

static void test_sink_enabled_negative_rate_throws() {
    TEST("sink: enabled sink with negative rate throws");
    ASSERT_THROWS(load_dyn(
        write_temp_yaml("excited_only", true, 3, 10, 100, 1, /*sink_rate=*/-0.001)));
    PASS();
}

// ============================================================================
// excited_site basis-aware mapping (Solution A — pigment 1-based)
// ============================================================================

static void test_excited_site_excited_only() {
    TEST("excited_site EXCITED_ONLY: pigment 3 → Hilbert index 2");
    auto path = write_temp_yaml("excited_only", false, 3, 10, 100, /*excited_site=*/3);
    auto cfg = load_dyn(path);
    // cfg.initial is now std::optional<InitialState>, filled by
    // resolve_dynamics_config(); write_temp_yaml always emits initial_state.
    ASSERT_EQ(cfg.initial.has_value(), true);
    ASSERT_EQ(cfg.initial->excited_site, 2);
    PASS();
}

static void test_excited_site_with_ground() {
    TEST("excited_site WITH_GROUND: pigment 3 → Hilbert index 3 (ground at 0)");
    auto path = write_temp_yaml("with_ground", false, 3, 10, 100, /*excited_site=*/3);
    auto cfg = load_dyn(path);
    ASSERT_EQ(cfg.initial.has_value(), true);
    ASSERT_EQ(cfg.initial->excited_site, 3);
    PASS();
}

static void test_excited_site_double_excited() {
    TEST("excited_site DOUBLE_EXCITED: pigment 3 → Hilbert index 3 (ground at 0)");
    auto path = write_temp_yaml("double_excited", false, 3, 10, 100, /*excited_site=*/3);
    auto cfg = load_dyn(path);
    ASSERT_EQ(cfg.initial.has_value(), true);
    ASSERT_EQ(cfg.initial->excited_site, 3);
    PASS();
}

static void test_excited_site_out_of_range_low() {
    TEST("excited_site: pigment=0 throws");
    ASSERT_THROWS(load_dyn(
        write_temp_yaml("excited_only", false, 3, 10, 100, /*excited_site=*/0)));
    PASS();
}

static void test_excited_site_out_of_range_high() {
    TEST("excited_site: pigment=8 throws (n_states=7)");
    ASSERT_THROWS(load_dyn(
        write_temp_yaml("excited_only", false, 3, 10, 100, /*excited_site=*/8)));
    PASS();
}

static std::string write_temp_exciton_yaml(const std::string& basis, int exciton_index) {
    std::string path = "/tmp/test_heom_exciton_config.yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  basis: \"" << basis << "\"\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100, -50]\n"
      << "    - [-50, 12300]\n"
      << "  w0: 12000.0\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 70.0\n  gamma_inverse: 100.0\n  expansion_method: matsubara\n  n_extra_poles: 1\n"
      << "  L_cutoff: 1\n"
      << "filter:\n"
      << "  enabled: true\n  tolerance: 1.0e-7\n  start_step: 10\n  warmup_factor: 0.001\n"
      << "dynamics:\n"
      << "  integrator: rk4\n"
      << "  dt: 2.0\n  n_steps: 100\n"
      << "  output_interval: 10\n"
      << "  print_interval: 100\n"
      << "  initial_state:\n"
      << "    type: \"exciton_excitation\"\n"
      << "    exciton_index: " << exciton_index << "\n"
      << "  output:\n    save_path: \"/tmp/\"\n    use_timestamp: false\n";
    f.close();
    return path;
}

static void test_exciton_index_excited_only_maps_to_zero_based() {
    TEST("exciton_index EXCITED_ONLY: user 1 → internal eigenvector 0");
    auto cfg = load_dyn(write_temp_exciton_yaml("excited_only", 1));
    ASSERT_EQ(cfg.initial.has_value(), true);
    ASSERT_EQ(cfg.initial->exciton_index, 0);
    ASSERT_EQ(single_exciton_user_index_1based(cfg.initial->exciton_index,
                                               cfg.physics.basis), 1);
    PASS();
}

static void test_exciton_index_with_ground_skips_ground() {
    TEST("exciton_index WITH_GROUND: user 1 skips ground → internal eigenvector 1");
    auto cfg = load_dyn(write_temp_exciton_yaml("with_ground", 1));
    ASSERT_EQ(cfg.initial.has_value(), true);
    ASSERT_EQ(cfg.initial->exciton_index, 1);
    ASSERT_EQ(single_exciton_user_index_1based(cfg.initial->exciton_index,
                                               cfg.physics.basis), 1);
    PASS();
}

static void test_exciton_index_with_ground_auto_filename_uses_user_index() {
    TEST("exciton_index WITH_GROUND: auto filename uses user-facing index");
    auto cfg = Config::load_yaml(write_temp_exciton_yaml("with_ground", 1));
    HEOMConfig::resolve_dynamics_config(cfg);
    if (cfg.output.filename.find("exciton1") == std::string::npos ||
        cfg.output.filename.find("exciton2") != std::string::npos) {
        std::cout << "FAIL\n    expected filename to contain exciton1 only, got: "
                  << cfg.output.filename << "\n";
        return;
    }
    PASS();
}

static void test_exciton_index_with_ground_high_throws() {
    TEST("exciton_index WITH_GROUND: user n_states+1 throws");
    ASSERT_THROWS(load_dyn(write_temp_exciton_yaml("with_ground", 3)));
    PASS();
}

static void test_exciton_index_double_excited_throws() {
    TEST("exciton_index DOUBLE_EXCITED: unsupported basis throws");
    ASSERT_THROWS(load_dyn(write_temp_exciton_yaml("double_excited", 1)));
    PASS();
}

static std::string write_temp_custom_matrix_yaml(const std::string& matrix_lines) {
    std::string path = "/tmp/test_heom_custom_matrix_config.yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  basis: \"excited_only\"\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100, -50]\n"
      << "    - [-50, 12300]\n"
      << "  w0: 12000.0\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 70.0\n  gamma_inverse: 100.0\n  expansion_method: matsubara\n  n_extra_poles: 1\n"
      << "  L_cutoff: 1\n"
      << "filter:\n"
      << "  enabled: true\n  tolerance: 1.0e-7\n  start_step: 10\n  warmup_factor: 0.001\n"
      << "dynamics:\n"
      << "  integrator: rk4\n"
      << "  dt: 2.0\n  n_steps: 100\n"
      << "  output_interval: 10\n"
      << "  print_interval: 100\n"
      << "  initial_state:\n"
      << "    type: \"custom_matrix\"\n"
      << "    matrix_elements:\n"
      << matrix_lines
      << "  output:\n    save_path: \"/tmp/\"\n    use_timestamp: false\n";
    f.close();
    return path;
}

static void test_custom_matrix_non_hermitian_throws() {
    TEST("custom_matrix: non-Hermitian density matrix throws");
    auto path = write_temp_custom_matrix_yaml(
        "      - [1, 1, 1.0]\n"
        "      - [1, 2, 0.5]\n");
    ASSERT_THROWS(load_dyn(path));
    PASS();
}

static void test_custom_matrix_trace_not_one_throws() {
    TEST("custom_matrix: trace != 1 throws");
    auto path = write_temp_custom_matrix_yaml(
        "      - [1, 1, 0.5]\n");
    ASSERT_THROWS(load_dyn(path));
    PASS();
}

// ============================================================================
// Interval validation tests
// ============================================================================

static void test_output_interval_zero_throws() {
    TEST("validate: output_interval=0 throws");
    ASSERT_THROWS(load_dyn(write_temp_yaml("excited_only", false, 3, 0)));
    PASS();
}

static void test_output_interval_greater_than_n_steps_throws() {
    TEST("validate: output_interval > n_steps throws");
    ASSERT_THROWS(load_dyn(write_temp_yaml("excited_only", false, 3, 101)));
    PASS();
}

static void test_print_interval_zero_disables_printing() {
    // print_interval == 0 is the sentinel for "no statistics output" — the
    // trajectory loop already guards each call site with `print_interval > 0`,
    // so 0 must round-trip through the per-task time grid parse. Negative
    // values still belong to test_print_interval_negative_throws below.
    TEST("validate: print_interval=0 is accepted (disables printing)");
    ASSERT_NOTHROW(load_dyn(write_temp_yaml("excited_only", false, 3, 10, 0)));
    PASS();
}

static void test_print_interval_negative_throws() {
    TEST("validate: print_interval<0 throws");
    ASSERT_THROWS(load_dyn(write_temp_yaml("excited_only", false, 3, 10, -1)));
    PASS();
}

// U-list single-knob schema: `filter.ulist` carries the value — 0 (or absent)
// = off; a value in (0,1] = on and is the adaptive gate τ. It lives in the
// filter block (parsed at load_yaml, only when filter.enabled is true). Helper
// injects the raw ulist line into the filter block so each case can exercise
// the parser.
static std::string write_temp_ulist_yaml(const std::string& ulist_line) {
    std::string path = "/tmp/test_heom_ulist_config.yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 3\n"
      << "  basis: \"excited_only\"\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12400, -106,     8]\n"
      << "    - [ -106, 12540,   28]\n"
      << "    - [    8,    28, 12120]\n"
      << "  w0: 12000.0\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 70.0\n  gamma_inverse: 100.0\n  expansion_method: matsubara\n  n_extra_poles: 1\n"
      << "  L_cutoff: 1\n"
      << "filter:\n"
      << "  enabled: true\n  tolerance: 1.0e-7\n  start_step: 10\n  warmup_factor: 0.001\n"
      << ulist_line
      << "dynamics:\n"
      << "  integrator: rk4\n"
      << "  dt: 2.0\n  n_steps: 100\n  output_interval: 10\n  print_interval: 100\n"
      << "  initial_state:\n    type: \"site_excitation\"\n    excited_site: 1\n"
      << "  output:\n    save_path: \"/tmp/\"\n    use_timestamp: false\n";
    f.close();
    return path;
}

static void test_ulist_zero_is_off() {
    TEST("ulist: 0 parses as off (ulist_tau == 0)");
    auto cfg = Config::load_yaml(write_temp_ulist_yaml("  ulist: 0\n"));
    ASSERT_NEAR(cfg.filter.ulist_tau, 0.0, 1e-12);
    PASS();
}

static void test_ulist_value_is_on() {
    TEST("ulist: 0.9 parses as on with tau=0.9");
    auto cfg = Config::load_yaml(write_temp_ulist_yaml("  ulist: 0.9\n"));
    ASSERT_NEAR(cfg.filter.ulist_tau, 0.9, 1e-12);
    PASS();
}

static void test_ulist_absent_defaults_off() {
    TEST("ulist absent defaults to off (ulist_tau == 0)");
    auto cfg = Config::load_yaml(write_temp_ulist_yaml(""));
    ASSERT_NEAR(cfg.filter.ulist_tau, 0.0, 1e-12);
    PASS();
}

static void test_ulist_out_of_range_throws() {
    TEST("ulist > 1 throws");
    ASSERT_THROWS(Config::load_yaml(write_temp_ulist_yaml("  ulist: 1.5\n")));
    PASS();
}

static void test_disorder_off_uses_legacy_path() {
    // Disorder-OFF smoke check: a K_drude=1 hierarchy at L_cutoff=2 is
    // non-empty and every ADO respects the single occupation cutoff Σα ≤ 2.
    TEST("hierarchy disorder OFF: simple Σα ≤ L_cutoff cutoff holds");
    PhysicalParameters p;
    p.n_states = 3;
    p.n_sites = 3;
    p.basis = BasisType::EXCITED_ONLY;
    p.n_bath_modes_total = 1;
    p.n_extra_poles = 1;

    HierarchyTruncation ht(p, /*L_cutoff=*/2);
    auto indices = ht.generate_hierarchy_indices();
    if (indices.empty()) {
        std::cout << "FAIL\n    hierarchy unexpectedly empty\n";
        return;
    }
    for (const auto& idx : indices) {
        if (idx.sum() > 2) {
            std::cout << "FAIL\n    ADO with Σα > L_cutoff escaped the cutoff\n";
            return;
        }
    }
    PASS();
}

// yaml schema tests for the bath-block occupation cutoff. `write_temp_disorder_yaml`
// emits an optional `static_disorder_fwhm` line (so its rejection can be
// exercised) plus an explicit L_cutoff line under bath — a test can omit it
// (missing-key) or supply a deliberately-invalid value (int8 overflow). Cannot
// reuse `write_temp_yaml()` because that helper hard-codes L_cutoff.
namespace {
std::string write_temp_disorder_yaml(bool emit_fwhm,
                                     const std::string& L_cutoff_line)
{
    std::string path = "/tmp/test_heom_disorder_config.yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  basis: \"excited_only\"\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12400, -100]\n"
      << "    - [-100, 12500]\n"
      << "  w0: 12000.0\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 70.0\n  gamma_inverse: 100.0\n  expansion_method: matsubara\n  n_extra_poles: 1\n";
    if (emit_fwhm)
        f << "  static_disorder_fwhm: 0\n";
    f << L_cutoff_line   // bath.L_cutoff — may be empty (missing-key test)
      << "filter:\n"
      << "  enabled: true\n  tolerance: 1.0e-7\n  start_step: 10\n  warmup_factor: 0.001\n"
      << "dynamics:\n  integrator: rk4\n  dt: 2.0\n  n_steps: 100\n  output_interval: 10\n  print_interval: 100\n"
      << "  initial_state:\n    type: \"site_excitation\"\n    excited_site: 1\n"
      << "  output:\n    save_path: \"/tmp/\"\n    use_timestamp: false\n";
    f.close();
    return path;
}
}  // namespace

// ---------- yaml schema tests (bath.L_cutoff) ----------

static void test_yaml_static_disorder_fwhm_rejected() {
    TEST("yaml: bath.static_disorder_fwhm is rejected (disorder-in-HEOM removed)");
    // any value (including 0) must deprecation-throw, guiding users to delete
    // the key.
    auto path = write_temp_disorder_yaml(/*emit_fwhm=*/true, "  L_cutoff: 1\n");
    ASSERT_THROWS(Config::load_yaml(path));
    PASS();
}

static void test_yaml_bath_missing_L_cutoff_throws() {
    TEST("yaml: missing bath.L_cutoff throws");
    auto path = write_temp_disorder_yaml(/*emit_fwhm=*/false, /*L_cutoff_line=*/"");
    ASSERT_THROWS(Config::load_yaml(path));
    PASS();
}

static void test_yaml_L_cutoff_int8_limit_throws() {
    TEST("yaml: L_cutoff >= 127 throws (int8_t limit)");
    auto path = write_temp_disorder_yaml(/*emit_fwhm=*/false, "  L_cutoff: 127\n");
    ASSERT_THROWS(Config::load_yaml(path));
    PASS();
}

// ============================================================================
// Memory Budget Tests
// ============================================================================

// Build a Config with explicit knobs so we can vary M, basis, K, filtering
// independently. n_states=7 matches FMO-7; tests exercise all three bases.
static HEOMConfig::Config make_budget_config(const std::string& basis,
                                             int n_bath_modes_total,
                                             int L_cutoff,
                                             double filter_tolerance) {
    auto path = write_temp_yaml(basis, /*sink_enabled=*/false, /*target_site=*/3);
    auto cfg = Config::load_yaml(path);
    cfg.physics.n_bath_modes_total = n_bath_modes_total;
    cfg.L_cutoff                   = L_cutoff;
    // filtering on/off is now the explicit switch; tolerance==0 means "off" in
    // this test helper's contract (budget buckets keyed on filter.enabled).
    cfg.filter.enabled   = (filter_tolerance > 0.0);
    cfg.filter.tolerance = filter_tolerance;
    return cfg;
}

static void test_budget_bucket_consistency() {
    TEST("budget: gpu_total = sum of GPU buckets");
    auto cfg = make_budget_config("excited_only", 1, 1, 1e-7);
    auto b = heom::estimate_budget(cfg, /*M=*/100);
    size_t sum = b.rho_tensor + b.rk4_workspace + b.hierarchy_maps
               + b.filter_buffers + b.auxiliary;
    ASSERT_EQ(b.gpu_total, sum);
    sum = b.host_hierarchy + b.host_neighbor_maps + b.host_density_buffer;
    ASSERT_EQ(b.host_total, sum);
    PASS();
}

static void test_budget_manual_excited_only() {
    TEST("budget: manual byte-count matches estimate (EXCITED_ONLY)");
    auto cfg = make_budget_config("excited_only", 1, 1, 1e-7);
    const size_t M = 50;
    auto b = heom::estimate_budget(cfg, M);
    // EXCITED_ONLY n_states=7 → n_sites=7. K=1 → K+1=2. n_bath=7.
    const size_t N = 7, n_bath = 7, Kp1 = 2;
    const size_t cd = 16;
    const size_t rho_size = M * N * N;
    const size_t map_size = M * n_bath * Kp1;
    // d_rho_next deleted: RK4 final combine is in-place, single rho tensor.
    ASSERT_EQ(b.rho_tensor,     1 * rho_size * cd);
    // RK4 workspace shrunk from 5 (k1..k4 + temp) to 3 buffers
    // (k_curr + temp + acc). Single largest GPU allocation, so this
    // assertion guards the saving — flipping back to 5 would silently
    // re-add ~2 * M * N^2 * 16B per run.
    ASSERT_EQ(b.rk4_workspace,  3 * rho_size * cd);
    // HierarchyMap bucket: d_plus_map(int) + d_minus_map(int) +
    // d_level_indices(int8) + d_coeff_plus(double) + d_coeff_minus(double).
    // The two coeff arrays hold the precomputed Term 4/5 sqrt magnitudes
    // (per-ADO, map layout) that the kernel used to recompute on the fly.
    ASSERT_EQ(b.hierarchy_maps, map_size * (4 + 4 + 1 + 8 + 8));
    PASS();
}

static void test_budget_filter_off_zero_buffers() {
    TEST("budget: filter_tolerance=0 → filter_buffers=0");
    auto cfg = make_budget_config("excited_only", 1, 1, /*filter_tolerance=*/0.0);
    auto b = heom::estimate_budget(cfg, 100);
    ASSERT_EQ(b.filter_buffers, size_t{0});
    ASSERT_EQ(b.use_filtering, false);
    PASS();
}

static void test_budget_filter_on_includes_buffers() {
    TEST("budget: filter_tolerance>0 → filter_buffers>0");
    auto cfg = make_budget_config("excited_only", 1, 1, 1e-7);
    auto b = heom::estimate_budget(cfg, 100);
    ASSERT_EQ(b.use_filtering, true);
    if (b.filter_buffers == 0) {
        std::cout << "FAIL\n    expected filter_buffers > 0\n";
        return;
    }
    PASS();
}

static void test_budget_basis_scales_N_squared() {
    TEST("budget: WITH_GROUND has N=8 → ρ tensor 64/49 of EXCITED_ONLY");
    auto cfg_excited = make_budget_config("excited_only", 1, 1, 1e-7);
    auto cfg_ground  = make_budget_config("with_ground",  1, 1, 1e-7);
    auto b_e = heom::estimate_budget(cfg_excited, 100);
    auto b_g = heom::estimate_budget(cfg_ground,  100);
    ASSERT_EQ(b_e.N, 7);
    ASSERT_EQ(b_g.N, 8);
    // ρ tensor scales as N²; ratio should be exactly 64/49 (integer math).
    ASSERT_EQ(b_g.rho_tensor * 49ULL, b_e.rho_tensor * 64ULL);
    PASS();
}

static void test_budget_check_fits() {
    TEST("budget: gpu_total < gpu_free → check returns Fits");
    auto cfg = make_budget_config("excited_only", 1, 1, 1e-7);
    auto b = heom::estimate_budget(cfg, 100);
    std::string err;
    auto result = heom::check_budget(
        b, cfg, /*gpu_free=*/size_t{4} * 1024 * 1024 * 1024, err);
    ASSERT_EQ(static_cast<int>(result),
              static_cast<int>(heom::BudgetCheck::Fits));
    if (!err.empty()) {
        std::cout << "FAIL\n    expected empty err, got: " << err << "\n";
        return;
    }
    PASS();
}

static void test_budget_check_exceeds() {
    TEST("budget: gpu_total > gpu_free → check returns Exceeds with hints");
    auto cfg = make_budget_config("excited_only", 1, 1, 1e-7);
    auto b = heom::estimate_budget(cfg, 100);
    std::string err;
    auto result = heom::check_budget(b, cfg, /*gpu_free=*/size_t{1024}, err);
    ASSERT_EQ(static_cast<int>(result),
              static_cast<int>(heom::BudgetCheck::Exceeds));
    if (err.find("Suggestions") == std::string::npos) {
        std::cout << "FAIL\n    expected 'Suggestions' in error message, got: " << err << "\n";
        return;
    }
    if (err.find("L_cutoff") == std::string::npos) {
        std::cout << "FAIL\n    expected L_cutoff hint, got: " << err << "\n";
        return;
    }
    PASS();
}

// The old API conflated "fits" and "could not check" — both returned
// true. The new tri-state API returns Unknown so callers see the difference.
static void test_budget_check_unknown_gpu() {
    TEST("budget: gpu_free=0 (unknown device) → check returns Unknown");
    auto cfg = make_budget_config("excited_only", 1, 1, 1e-7);
    auto b = heom::estimate_budget(cfg, 100);
    std::string err;
    auto result = heom::check_budget(b, cfg, /*gpu_free=*/0, err);
    ASSERT_EQ(static_cast<int>(result),
              static_cast<int>(heom::BudgetCheck::Unknown));
    PASS();
}

static void test_budget_safety_margin() {
    TEST("budget: 96% of gpu_free fails 95% margin, passes 100%");
    auto cfg = make_budget_config("excited_only", 1, 1, 1e-7);
    auto b = heom::estimate_budget(cfg, 100);
    // Concoct a free value such that gpu_total = 96% of it.
    size_t gpu_free = static_cast<size_t>(b.gpu_total / 0.96);
    std::string err;
    auto strict = heom::check_budget(
        b, cfg, gpu_free, err, /*safety_margin=*/0.95);
    ASSERT_EQ(static_cast<int>(strict),
              static_cast<int>(heom::BudgetCheck::Exceeds));
    err.clear();
    auto loose = heom::check_budget(
        b, cfg, gpu_free, err, /*safety_margin=*/1.0);
    ASSERT_EQ(static_cast<int>(loose),
              static_cast<int>(heom::BudgetCheck::Fits));
    PASS();
}

// Bad safety_margin must throw, not silently clamp.
static void test_budget_safety_margin_invalid_throws() {
    TEST("budget: safety_margin out of (0,1] throws std::invalid_argument");
    auto cfg = make_budget_config("excited_only", 1, 1, 1e-7);
    auto b = heom::estimate_budget(cfg, 100);
    std::string err;
    bool caught_zero = false;
    try { heom::check_budget(b, cfg, 1024, err, /*safety_margin=*/0.0); }
    catch (const std::invalid_argument&) { caught_zero = true; }
    if (!caught_zero) {
        std::cout << "FAIL\n    safety_margin=0.0 did not throw\n";
        return;
    }
    bool caught_high = false;
    try { heom::check_budget(b, cfg, 1024, err, /*safety_margin=*/1.5); }
    catch (const std::invalid_argument&) { caught_high = true; }
    if (!caught_high) {
        std::cout << "FAIL\n    safety_margin=1.5 did not throw\n";
        return;
    }
    PASS();
}

static void test_budget_hierarchy_real_M() {
    TEST("budget: hierarchy-generated M is non-trivial (FMO-7 K=1 L=2)");
    auto cfg = make_budget_config("excited_only", 1, 2, 1e-7);
    HierarchyTruncation hierarchy(cfg.physics, cfg.L_cutoff);
    auto indices = hierarchy.generate_hierarchy_indices();
    auto b = heom::estimate_budget(cfg, indices.size());
    if (b.M == 0 || b.gpu_total == 0) {
        std::cout << "FAIL\n    expected non-zero M and gpu_total, got M="
                  << b.M << " gpu_total=" << b.gpu_total << "\n";
        return;
    }
    PASS();
}

// h_rho0_ host buffer is N² (not M·N²). Regression guard against
// re-introducing the old M·N² cargo cult sizing.
static void test_budget_host_buffer_is_N_squared() {
    TEST("budget: host_density_buffer is N² (not M·N²)");
    auto cfg = make_budget_config("excited_only", 1, 1, 1e-7);
    auto b = heom::estimate_budget(cfg, /*M=*/100);
    // EXCITED_ONLY n_states=7 → N=7 → 49 * 16B = 784B
    ASSERT_EQ(b.host_density_buffer, size_t{7 * 7 * 16});
    PASS();
}

// ============================================================================
// GPU buffer plan single-source-of-truth tests
// ============================================================================

// Sum of plan bytes must equal budget.gpu_total. If someone later forgets to
// update plan_gpu_buffers() when adding a buffer, this test catches the drift.
static void test_plan_total_matches_budget() {
    TEST("plan: sum of bytes matches budget.gpu_total");
    auto cfg = make_budget_config("excited_only", 1, 1, 1e-7);
    auto b = heom::estimate_budget(cfg, /*M=*/100);
    auto plan = heom::plan_gpu_buffers(
        100, b.N, b.K, b.n_bath, b.L, b.use_filtering);
    size_t sum = 0;
    for (const auto& s : plan) sum += s.bytes();
    ASSERT_EQ(sum, b.gpu_total);
    PASS();
}

// Filtering off means the four FilterBuf entries must be omitted.
static void test_plan_filter_off_buffer_count() {
    TEST("plan: filter off → no FilterBuf entries");
    auto plan = heom::plan_gpu_buffers(
        /*M=*/100, /*N=*/7, /*K=*/1, /*n_bath=*/7, /*L=*/1,
        /*use_filtering=*/false);
    int filter_count = 0;
    for (const auto& s : plan) {
        if (s.bucket == heom::GPUBufferSpec::Bucket::FilterBuf) ++filter_count;
    }
    ASSERT_EQ(filter_count, 0);
    PASS();
}

// Filtering on means exactly three FilterBuf entries: ado_norms, active_flags,
// level_counts. (|A| is derived from level_counts, so no partial_sums buffer.)
static void test_plan_filter_on_buffer_count() {
    TEST("plan: filter on → 3 FilterBuf entries");
    auto plan = heom::plan_gpu_buffers(
        /*M=*/100, /*N=*/7, /*K=*/1, /*n_bath=*/7, /*L=*/1,
        /*use_filtering=*/true);
    int filter_count = 0;
    for (const auto& s : plan) {
        if (s.bucket == heom::GPUBufferSpec::Bucket::FilterBuf) ++filter_count;
    }
    ASSERT_EQ(filter_count, 3);
    PASS();
}

// Plan must include all canonical buffer names. Catches typos and accidental
// renames between plan and allocator.
static void test_plan_canonical_names_present() {
    TEST("plan: canonical buffer names all present");
    auto plan = heom::plan_gpu_buffers(
        /*M=*/100, /*N=*/7, /*K=*/1, /*n_bath=*/7, /*L=*/1,
        /*use_filtering=*/true);
    const std::vector<std::string> required = {
        "d_rho", "d_workspace",
        "d_hamiltonian", "d_c_bath", "d_markov_residue",
        "d_plus_map", "d_minus_map", "d_level_indices",
        "d_coeff_plus", "d_coeff_minus",
        "d_total_decay", "d_d_coupling",
        "d_ado_norms", "d_active_flags", "d_level_counts"
    };
    for (const auto& name : required) {
        bool found = false;
        for (const auto& s : plan) {
            if (name == s.name) { found = true; break; }
        }
        if (!found) {
            std::cout << "FAIL\n    missing buffer in plan: " << name << "\n";
            return;
        }
    }
    PASS();
}

static void test_derivative_total_elements_overflow_throws() {
    TEST("cuda launch: M*N*N overflow is rejected before kernel launch");
    ASSERT_THROWS(heom::checked_total_elements(/*M=*/300000000, /*N=*/100));
    ASSERT_NOTHROW(heom::checked_total_elements(/*M=*/1000, /*N=*/8));
    PASS();
}

static void test_derivative_shared_memory_limit_throws() {
    TEST("cuda launch: derivative shared memory over device limit is rejected");
    ASSERT_THROWS(heom::checked_derivative_smem_bytes(
        /*N=*/106, /*n_bath=*/14, /*max_smem_bytes=*/49152));
    ASSERT_NOTHROW(heom::checked_derivative_smem_bytes(
        /*N=*/8, /*n_bath=*/7, /*max_smem_bytes=*/49152));
    PASS();
}

static void test_derivative_context_requires_total_decay() {
    TEST("cuda launch: total_decay pointer is required");
    HEOMContext ctx{};
    ctx.rho_in = reinterpret_cast<const cuDoubleComplex*>(0x1);
    ctx.drho_out = reinterpret_cast<cuDoubleComplex*>(0x1);
    ctx.hamiltonian = reinterpret_cast<const cuDoubleComplex*>(0x1);
    ctx.c_bath = reinterpret_cast<const cuDoubleComplex*>(0x1);
    ctx.coeff_plus = reinterpret_cast<const double*>(0x1);
    ctx.coeff_minus = reinterpret_cast<const double*>(0x1);
    ctx.neighbor_map_plus = reinterpret_cast<const int*>(0x1);
    ctx.neighbor_map_minus = reinterpret_cast<const int*>(0x1);
    ctx.level_indices = reinterpret_cast<const int8_t*>(0x1);
    ctx.total_decay = nullptr;
    ctx.d_coupling = reinterpret_cast<const int8_t*>(0x1);
    ctx.M = 1;
    ctx.N = 1;
    ctx.K = 0;
    ctx.n_bath = 1;
    ctx.row_off = 0; ctx.n_row = 1;   // valid block, so the throw below can
    ctx.col_off = 0; ctx.n_col = 1;   // only be about total_decay
    ASSERT_THROWS(heom::validate_derivative_context(ctx));
    PASS();
}

// The manifold block is a claim the kernel obeys literally, so the shapes it
// refuses matter as much as the ones it accepts. In particular a partially
// overlapping row/col range is meaningless (manifolds partition the basis) and
// would break the Term 1 dispatch's "at most two contiguous runs" assumption.
static void test_derivative_context_manifold_block_rules() {
    TEST("cuda launch: manifold block must be non-empty, in range, and "
         "identical-or-disjoint");
    HEOMContext base{};
    base.rho_in = reinterpret_cast<const cuDoubleComplex*>(0x1);
    base.drho_out = reinterpret_cast<cuDoubleComplex*>(0x1);
    base.hamiltonian = reinterpret_cast<const cuDoubleComplex*>(0x1);
    base.c_bath = reinterpret_cast<const cuDoubleComplex*>(0x1);
    base.coeff_plus = reinterpret_cast<const heom::real_t*>(0x1);
    base.coeff_minus = reinterpret_cast<const heom::real_t*>(0x1);
    base.markov_residue = reinterpret_cast<const heom::real_t*>(0x1);
    base.neighbor_map_plus = reinterpret_cast<const int*>(0x1);
    base.neighbor_map_minus = reinterpret_cast<const int*>(0x1);
    base.level_indices = reinterpret_cast<const int8_t*>(0x1);
    base.total_decay = reinterpret_cast<const double*>(0x1);
    base.d_coupling = reinterpret_cast<const int8_t*>(0x1);
    base.M = 1; base.N = 8; base.K = 0; base.n_bath = 1;

    auto with_block = [&](int ro, int nr, int co, int nc) {
        HEOMContext c = base;
        c.row_off = ro; c.n_row = nr; c.col_off = co; c.n_col = nc;
        return c;
    };

    // Whole matrix, and the (f,e)-shaped disjoint block the design targets.
    ASSERT_NOTHROW(heom::validate_derivative_context(with_block(0, 8, 0, 8)));
    ASSERT_NOTHROW(heom::validate_derivative_context(with_block(4, 4, 1, 3)));
    ASSERT_NOTHROW(heom::validate_derivative_context(with_block(1, 3, 4, 4)));
    ASSERT_NOTHROW(heom::validate_derivative_context(with_block(1, 3, 1, 3)));

    // Default-constructed (0,0,0,0): the whole point of having no sentinel.
    ASSERT_THROWS(heom::validate_derivative_context(base));
    ASSERT_THROWS(heom::validate_derivative_context(with_block(0, 0, 0, 8)));
    ASSERT_THROWS(heom::validate_derivative_context(with_block(0, 8, 0, 0)));
    // Out of range, negative offset.
    ASSERT_THROWS(heom::validate_derivative_context(with_block(0, 9, 0, 8)));
    ASSERT_THROWS(heom::validate_derivative_context(with_block(-1, 4, 0, 8)));
    ASSERT_THROWS(heom::validate_derivative_context(with_block(6, 4, 0, 8)));
    // Partial overlap: neither identical nor disjoint.
    ASSERT_THROWS(heom::validate_derivative_context(with_block(0, 4, 2, 4)));
    ASSERT_THROWS(heom::validate_derivative_context(with_block(2, 4, 0, 4)));
    ASSERT_THROWS(heom::validate_derivative_context(with_block(0, 8, 1, 3)));
    PASS();
}

static void test_spectrum_filtering_metadata_uses_config() {
    TEST("spectrum metadata: filtering attrs come from config.filter");
    HEOMConfig::Config cfg{};

    cfg.filter.enabled   = true;
    cfg.filter.tolerance = 1.0e-6;
    auto meta = heom::output::spectrum_filtering_metadata(cfg);
    ASSERT_EQ(meta.use_filtering, true);
    ASSERT_NEAR(meta.filter_threshold, 1.0e-6, 1.0e-18);

    cfg.filter.enabled = false;
    meta = heom::output::spectrum_filtering_metadata(cfg);
    ASSERT_EQ(meta.use_filtering, false);
    PASS();
}

// typed_alloc must throw std::logic_error before any cudaMalloc when the
// caller's GPUMemory<T> template arg disagrees with the plan's elem_bytes.
// This is the whole point of the helper — without it, a future commit that
// changes one side but not the other silently under/over-allocates GPU
// memory. Use a hand-rolled spec list (not plan_gpu_buffers()) so the test
// keeps catching the helper's invariant even if the canonical plan is
// later refactored to remove a particular entry.
static void test_typed_alloc_type_mismatch_throws() {
    TEST("typed_alloc: sizeof(T) != plan.elem_bytes throws (no cudaMalloc)");
    using B = heom::GPUBufferSpec::Bucket;
    std::vector<heom::GPUBufferSpec> bogus_plan = {
        {"d_test", /*count=*/100, /*elem_bytes=*/sizeof(double), B::Auxiliary}
    };
    bool caught = false;
    try {
        // Caller asks for int8_t (1 byte), plan says double (8 bytes).
        // typed_alloc must throw *before* GPUMemory<int8_t>(100) runs.
        auto buf = heom::typed_alloc<int8_t>(bogus_plan, "d_test");
        (void)buf;
    } catch (const std::logic_error& e) {
        caught = true;
        const std::string msg = e.what();
        if (msg.find("d_test") == std::string::npos
            || msg.find("type mismatch") == std::string::npos) {
            std::cout << "FAIL\n    error message lacks diagnostic info: "
                      << msg << "\n";
            return;
        }
    }
    if (!caught) {
        std::cout << "FAIL\n    typed_alloc<int8_t> did not throw on "
                     "elem_bytes=8 mismatch\n";
        return;
    }
    PASS();
}

// typed_alloc must throw std::logic_error when the caller asks for a name
// the plan never produced. Same fail-loud-at-startup story as the type
// mismatch case, but for the typo-in-name failure mode.
static void test_typed_alloc_missing_buffer_throws() {
    TEST("typed_alloc: missing buffer name throws (no cudaMalloc)");
    using B = heom::GPUBufferSpec::Bucket;
    std::vector<heom::GPUBufferSpec> bogus_plan = {
        {"d_present", /*count=*/100, /*elem_bytes=*/sizeof(double), B::Auxiliary}
    };
    bool caught = false;
    try {
        auto buf = heom::typed_alloc<double>(bogus_plan, "d_absent");
        (void)buf;
    } catch (const std::logic_error& e) {
        caught = true;
        const std::string msg = e.what();
        if (msg.find("d_absent") == std::string::npos
            || msg.find("missing buffer") == std::string::npos) {
            std::cout << "FAIL\n    error message lacks diagnostic info: "
                      << msg << "\n";
            return;
        }
    }
    if (!caught) {
        std::cout << "FAIL\n    typed_alloc did not throw on missing buffer\n";
        return;
    }
    PASS();
}

// ============================================================================
// In-memory Config api-stability
// ============================================================================
//
// Mechanical guard for the ML-fitting contract: HEOMConfig::Config must
// stay constructible without touching the filesystem. A future commit that
// silently moves a required field into load_yaml()'s private side effects
// will trip this test long before any Python binding ever runs.
//
// We deliberately do NOT construct a SystemContext here — that ctor would
// allocate / upload GPU buffers and pull this out of test_core's no-GPU
// budget. The SystemContext + compute_absorption forward path lives in
// the GPU/integration smoke.

static void test_in_memory_config_construction() {
    TEST("Config: in-memory construction without filesystem");

    using namespace HEOMConfig;
    Config cfg{};

    cfg.physics.n_states = 2;
    cfg.physics.basis    = BasisType::WITH_GROUND;
    cfg.physics.n_sites  = compute_hilbert_dim(cfg.physics.n_states, cfg.physics.basis);
    cfg.physics.hamiltonian =
        expand_hamiltonian(Eigen::MatrixXcd::Identity(2, 2) * 12000.0 * CM_TO_AU,
                           cfg.physics.basis);

    cfg.physics.temperature_K    = 77.0;
    cfg.physics.temperature      = cfg.physics.temperature_K * BOLTZMANN;
    set_homog_bath(cfg.physics, 35.0 * CM_TO_AU, 1.0 / (100.0 * FS_TO_AU));

    cfg.physics.n_bath_modes_total       = 0;
    cfg.physics.n_extra_poles = 0;
    compute_matsubara_drude(cfg.physics);

    // Guard the dual-field invariant. Without static
    // disorder, compute_matsubara_drude() must not bump n_extra_poles
    // (that's the user-input snapshot); n_bath_modes_total is the runtime array
    // length. A future commit that re-merges the two fields, or makes
    // compute_matsubara_drude mutate n_extra_poles as a side effect, will
    // immediately fail one of these two assertions.
    ASSERT_EQ(cfg.physics.n_extra_poles, 0);
    ASSERT_EQ(cfg.physics.n_bath_modes_total, cfg.physics.n_extra_poles);

    cfg.L_cutoff         = 1;
    cfg.filter.enabled   = false;   // filtering OFF: knobs left at defaults

    cfg.time.dt                   = 1.0;
    cfg.time.dt_au                = cfg.time.dt * FS_TO_AU;
    cfg.time.n_steps              = 1;
    cfg.time.output_interval      = 1;
    cfg.time.print_interval       = 0;

    // cfg.initial is std::optional<InitialState>. Build a
    // local InitialState first, then commit — same path as load_yaml(), so a
    // future commit that adds another required field will trip both tests
    // together.
    InitialState init0{};
    init0.type         = InitialState::Type::SITE_EXCITATION;
    init0.excited_site = 1;   // WITH_GROUND: pigment 1 → Hilbert idx 1
    cfg.initial = std::move(init0);

    cfg.output.save_path     = "";
    cfg.output.filename      = "in_memory_config_test";
    cfg.output.use_timestamp = false;

    cfg.yaml_path = "<in-memory>";

    ASSERT_NOTHROW(cfg.validate());
    PASS();
}

static void test_load_yaml_preserves_source_snapshot() {
    TEST("Config::load_yaml preserves raw yaml source snapshot");

    auto path = write_temp_yaml("excited_only", false, 3);
    auto cfg = Config::load_yaml(path);

    std::ifstream in(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string expected_yaml = buffer.str();

    // config_yaml is the EXACT input text (raw provenance). config_hash is no
    // longer a hash of this text — it is the physics identity (see below).
    ASSERT_EQ(cfg.yaml_path, path);
    ASSERT_EQ(cfg.config_yaml, expected_yaml);
    PASS();
}

// Write an arbitrary yaml string to a temp file and return the path.
static std::string write_raw_yaml(const std::string& text,
                                  const std::string& path = "/tmp/test_heom_physhash.yaml") {
    std::ofstream f(path);
    f << text;
    f.close();
    return path;
}

// The canonical physics vector, mirrored byte-for-byte by the Python
// test_config_hash golden. If you edit this string, update BOTH goldens.
static const char* kPhysicsHashVectorYaml =
    "system:\n"
    "  n_states: 2\n"
    "  basis: with_ground\n"
    "  temperature: 77.0\n"
    "  hamiltonian:\n"
    "    - [12100.0, -50.0]\n"
    "    - [-50.0, 12300.0]\n"
    "  w0: 12000.0\n"
    "  dipoles:\n"
    "    - [1.0, 0.0, 0.0]\n"
    "    - [0.0, 1.0, 0.0]\n"
    "  strengths: [4.5, 4.5]\n"
    "bath:\n"
    "  spectral_density: \"Drude-Lorentz\"\n"
    "  eta: 35.0\n"
    "  gamma_inverse: 100.0\n"
    "  expansion_method: matsubara\n"
    "  n_extra_poles: 1\n"
    "  L_cutoff: 3\n"
    "filter:\n"
    "  enabled: true\n"
    "  tolerance: 1.0e-7\n"
    "  start_step: 10\n"
    "  warmup_factor: 0.001\n"
    "absorption:\n"
    "  integrator: rk4\n"
    "  dt: 1.0\n"
    "  n_steps: 200\n"
    "  output_interval: 1\n"
    "  print_interval: 200\n"
    "  output:\n"
    "    save_path: \"/tmp/\"\n"
    "    filename: \"auto\"\n"
    "    use_timestamp: false\n"
    "static_disorder:\n"
    "  draws_file: \"nodes.h5\"\n";

// Cross-language golden: the SAME string in the Python golden must hash to this.
// Recomputed for the task-scoped schema (see tools test_config_hash.py::_GOLDEN).
static const char* kPhysicsHashVectorExpected = "7e9a4ebcf3c349b7";

static void test_config_hash_physics_golden() {
    TEST("config_hash: physics golden vector (cross-language pin)");
    auto cfg = Config::load_yaml(write_raw_yaml(kPhysicsHashVectorYaml));
    ASSERT_EQ(cfg.config_hash, std::string(kPhysicsHashVectorExpected));
    PASS();
}

static void test_config_hash_ignores_comments_and_format() {
    TEST("config_hash: stable across comments / whitespace / key order / output+static_disorder");
    const std::string base = kPhysicsHashVectorYaml;
    const std::string ref = Config::load_yaml(
        write_raw_yaml(base, "/tmp/test_ph_ref.yaml")).config_hash;

    // (1) A pure comment edit + blank line: config_hash must NOT change (this is
    // exactly the incremental-merge trap that motivated the physics hash).
    const std::string commented =
        "# a leading comment\n" + base + "\n# trailing note about r-range\n";
    ASSERT_EQ(Config::load_yaml(
        write_raw_yaml(commented, "/tmp/test_ph_cmt.yaml")).config_hash, ref);

    // (2) Editing the non-physical output block must NOT change it.
    std::string out_edited = base;
    auto pos = out_edited.find("/tmp/");
    out_edited.replace(pos, 5, "/other/dir/");
    ASSERT_EQ(Config::load_yaml(
        write_raw_yaml(out_edited, "/tmp/test_ph_out.yaml")).config_hash, ref);

    // (3) A real physical edit (temperature) MUST change it.
    std::string phys_edited = base;
    auto tp = phys_edited.find("temperature: 77.0");
    phys_edited.replace(tp, std::string("temperature: 77.0").size(),
                        "temperature: 100.0");
    ASSERT_NE(Config::load_yaml(
        write_raw_yaml(phys_edited, "/tmp/test_ph_phys.yaml")).config_hash, ref);
    PASS();
}

// ============================================================================
// spectra/dipole: basis-aware dipole-operator factory
// ============================================================================
//
// Stage 1 absorption projects per-pigment dipole vectors onto the WITH_GROUND
// Hilbert basis (DipoleOperator). The tests below pin that factory's output
// layout for the WITH_GROUND / DOUBLE_EXCITED bases and its input validation.
// spectra/ must stay CUDA-free: no cuComplex / d_* device symbols anywhere
// under src/spectra/ or include/spectra/.

static void test_dipole_with_ground_basic() {
    TEST("dipole: from_pigment_dipoles WITH_GROUND basic");

    using namespace heom::spectra;
    const std::vector<std::array<double, 3>> dvec = {
        {{1.0, 0.0, 0.0}},   // pigment 1: x-polarised
        {{0.0, 1.0, 0.0}},   // pigment 2: y-polarised
    };
    const std::vector<double> str = {1.0, 1.0};
    const auto op = DipoleOperator::from_pigment_dipoles(
        dvec, str, BasisType::WITH_GROUND, /*N=*/3);

    ASSERT_EQ(op.mu_x.rows(), 3);
    ASSERT_EQ(op.mu_x.cols(), 3);

    // μ_x: only the (k=1, ground=0) entry is non-zero (= 1).
    ASSERT_NEAR(op.mu_x(1, 0).real(), 1.0, 1e-15);
    ASSERT_NEAR(op.mu_x(1, 0).imag(), 0.0, 1e-15);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            if (i == 1 && j == 0) continue;
            ASSERT_NEAR(std::abs(op.mu_x(i, j)), 0.0, 1e-15);
        }

    // μ_y: only the (k=2, ground=0) entry is non-zero (= 1).
    ASSERT_NEAR(op.mu_y(2, 0).real(), 1.0, 1e-15);
    ASSERT_NEAR(op.mu_y(2, 0).imag(), 0.0, 1e-15);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            if (i == 2 && j == 0) continue;
            ASSERT_NEAR(std::abs(op.mu_y(i, j)), 0.0, 1e-15);
        }

    // μ_z: identically zero (no z-polarised pigments in this test).
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(std::abs(op.mu_z(i, j)), 0.0, 1e-15);

    PASS();
}

static void test_dipole_with_ground_strengths() {
    TEST("dipole: from_pigment_dipoles WITH_GROUND strengths");

    using namespace heom::spectra;
    const std::vector<std::array<double, 3>> dvec = {
        {{1.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
    };
    const std::vector<double> str = {2.0, 3.0};
    const auto op = DipoleOperator::from_pigment_dipoles(
        dvec, str, BasisType::WITH_GROUND, /*N=*/3);

    // strengths must scale the matrix elements linearly (μ⁺_k = s_k · d̂_k |k⟩⟨g|).
    ASSERT_NEAR(op.mu_x(1, 0).real(), 2.0, 1e-15);
    ASSERT_NEAR(op.mu_x(2, 0).real(), 3.0, 1e-15);
    // y / z components are zero (input dipoles point along x).
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            ASSERT_NEAR(std::abs(op.mu_y(i, j)), 0.0, 1e-15);
            ASSERT_NEAR(std::abs(op.mu_z(i, j)), 0.0, 1e-15);
        }

    PASS();
}

static void test_dipole_excited_only_throws() {
    TEST("dipole: from_pigment_dipoles EXCITED_ONLY throws std::invalid_argument");

    using namespace heom::spectra;
    const std::vector<std::array<double, 3>> dvec = {{{1.0, 0.0, 0.0}}};
    const std::vector<double> str = {1.0};

    // Both the exception type and the diagnostic substring are part of the
    // contract — the message helps the YAML user diagnose "wrong basis"
    // without spelunking through compute_linear_response().
    bool caught = false;
    std::string what;
    try {
        DipoleOperator::from_pigment_dipoles(
            dvec, str, BasisType::EXCITED_ONLY, /*N=*/1);
    } catch (const std::invalid_argument& e) {
        caught = true;
        what = e.what();
    }
    if (!caught) {
        std::cout << "FAIL\n    expected std::invalid_argument, none thrown\n";
        return;
    }
    if (what.find("requires basis=with_ground") == std::string::npos) {
        std::cout << "FAIL\n    expected message to contain 'requires basis=with_ground', "
                  << "got: " << what << "\n";
        return;
    }

    PASS();
}

static void test_dipole_double_excited_block() {
    TEST("dipole DOUBLE_EXCITED: e<->f raising block (2-site)");

    using namespace heom::spectra;
    // 2 pigments: n_states=2, DOUBLE_EXCITED dim = 1 + 2 + 1 = 4
    // Hilbert: 0=g, 1=|1>, 2=|2>, 3=|12>
    const std::vector<std::array<double, 3>> dirs = {
        {{1.0, 0.0, 0.0}},   // pigment 1: x-polarised, strength 2
        {{0.0, 1.0, 0.0}},   // pigment 2: y-polarised, strength 3
    };
    const std::vector<double> str = {2.0, 3.0};
    const int N = 4;
    const auto op = DipoleOperator::from_pigment_dipoles(
        dirs, str, BasisType::DOUBLE_EXCITED, N);

    ASSERT_EQ(op.mu_x.rows(), 4);
    ASSERT_EQ(op.mu_x.cols(), 4);

    // g->e block (column 0): mu_x(1,0)=2.0, mu_y(2,0)=3.0
    ASSERT_NEAR(std::abs(op.mu_x(1, 0) - std::complex<double>(2.0, 0)), 0.0, 1e-15);
    ASSERT_NEAR(std::abs(op.mu_y(2, 0) - std::complex<double>(3.0, 0)), 0.0, 1e-15);
    // e->f block: |12>(idx3) <- |1>(idx1) excites pigment 2 (y, s=3);
    //             |12>(idx3) <- |2>(idx2) excites pigment 1 (x, s=2).
    ASSERT_NEAR(std::abs(op.mu_y(3, 1) - std::complex<double>(3.0, 0)), 0.0, 1e-15);
    ASSERT_NEAR(std::abs(op.mu_x(3, 2) - std::complex<double>(2.0, 0)), 0.0, 1e-15);
    // Raising operator only: double->single columns are 0 (that is μ⁻'s job).
    ASSERT_NEAR(std::abs(op.mu_y(1, 3)), 0.0, 1e-15);
    ASSERT_NEAR(std::abs(op.mu_x(2, 3)), 0.0, 1e-15);

    PASS();
}

// Helper: write a minimal absorption-style yaml (modelled on abs_dimer.yaml).
// Transition dipoles/strengths are now a system property (parsed at load_yaml),
// so the strength-validation tests can throw straight out of Config::load_yaml.
static std::string write_temp_absorption_yaml(const std::string& strengths = "[4.5, 4.5]") {
    std::string path = "/tmp/test_heom_absorption_config.yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  basis: with_ground\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100.0, -50.0]\n"
      << "    - [ -50.0, 12300.0]\n"
      << "  w0: 12000.0\n"
      << "  dipoles:\n"
      << "    - [1.0, 0.0, 0.0]\n"
      << "    - [0.0, 1.0, 0.0]\n"
      << "  strengths: " << strengths << "\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 35.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: matsubara\n"
      << "  n_extra_poles: 1\n"
      << "  L_cutoff: 3\n"
      << "filter:\n"
      << "  enabled: true\n  tolerance: 1.0e-7\n"
      << "absorption:\n"
      << "  integrator: rk4\n"
      << "  dt: 1.0\n"
      << "  n_steps: 100\n"
      << "  output_interval: 1\n"
      << "  print_interval: 50\n"
      << "  output:\n"
      << "    save_path: \"/tmp/\"\n"
      << "    filename: \"absorption_smoke\"\n"
      << "    use_timestamp: false\n";
    f.close();
    return path;
}

// Positive test: a well-formed absorption yaml (dipoles under system) loads
// cleanly. Guards against dipole/strength validation tightening to reject a
// valid unit-vector set.
static void test_absorption_yaml_loads_clean() {
    TEST("absorption yaml: valid system.dipoles loads (no false positive)");
    ASSERT_NOTHROW(Config::load_yaml(write_temp_absorption_yaml()));
    PASS();
}

static void test_dipoles_all_zero_strengths_throw() {
    TEST("system.strengths: all-zero throws");
    ASSERT_THROWS(Config::load_yaml(write_temp_absorption_yaml(/*strengths=*/"[0.0, 0.0]")));
    PASS();
}

static void test_dipoles_negative_strength_throws() {
    TEST("system.strengths: negative throws");
    ASSERT_THROWS(Config::load_yaml(write_temp_absorption_yaml(/*strengths=*/"[4.5, -1.0]")));
    PASS();
}

static void test_linear_response_rejects_non_ground_basis() {
    TEST("linear_response: rejects basis without explicit ground state");
    using namespace heom::spectra;
    const std::vector<std::array<double, 3>> dvec = {{{1.0, 0.0, 0.0}}};
    const std::vector<double> str = {1.0};
    const auto dipole = DipoleOperator::from_pigment_dipoles(
        dvec, str, BasisType::WITH_GROUND, /*N=*/2);

    ASSERT_THROWS(validate_linear_response_inputs(
        BasisType::EXCITED_ONLY, dipole, /*N=*/2));
    ASSERT_NOTHROW(validate_linear_response_inputs(
        BasisType::WITH_GROUND, dipole, /*N=*/2));
    PASS();
}

static void test_linear_response_rejects_all_zero_dipoles() {
    TEST("linear_response: rejects all-zero dipole directions");
    using namespace heom::spectra;
    DipoleOperator dipole;
    dipole.mu_x = Eigen::MatrixXcd::Zero(3, 3);
    dipole.mu_y = Eigen::MatrixXcd::Zero(3, 3);
    dipole.mu_z = Eigen::MatrixXcd::Zero(3, 3);

    ASSERT_THROWS(validate_linear_response_inputs(
        BasisType::WITH_GROUND, dipole, /*N=*/3));
    PASS();
}

static void test_spectrum_result_shape_validation() {
    TEST("spectrum writer: rejects malformed LinearResponseResult shape");
    heom::spectra::LinearResponseResult out;
    out.N = 3;
    out.n_outputs = 2;
    out.time_fs = {0.0};
    out.rho_xyz.resize(static_cast<size_t>(3) * 2 * 3 * 3);

    ASSERT_THROWS(heom::output::validate_spectrum_result_shape(
        /*expected_N=*/3, /*expected_n_outputs=*/2, out));

    out.time_fs = {0.0, 1.0};
    ASSERT_NOTHROW(heom::output::validate_spectrum_result_shape(
        /*expected_N=*/3, /*expected_n_outputs=*/2, out));

    out.N = 4;
    ASSERT_THROWS(heom::output::validate_spectrum_result_shape(
        /*expected_N=*/3, /*expected_n_outputs=*/2, out));
    out.N = 3;

    out.n_outputs = 3;
    ASSERT_THROWS(heom::output::validate_spectrum_result_shape(
        /*expected_N=*/3, /*expected_n_outputs=*/2, out));
    out.n_outputs = 2;

    out.rho_xyz.pop_back();
    ASSERT_THROWS(heom::output::validate_spectrum_result_shape(
        /*expected_N=*/3, /*expected_n_outputs=*/2, out));
    PASS();
}

// ============================================================================

// heom::StreamFormat, the RAII guard for sticky stream state.
//
// Poison the stream with a state that is NOT the library default, so a guard
// that merely resets to defaults (rather than restoring what it found) fails
// here too.
static void test_stream_format_guard() {
    TEST("StreamFormat: restores flags/precision/fill, incl. on unwind");
    std::ostringstream s;
    s << std::scientific << std::setprecision(9) << std::setfill('#');
    const auto flags = s.flags();
    const auto prec  = s.precision();
    const char fill  = s.fill();

    {
        const heom::StreamFormat guard(s);
        s << std::fixed << std::setprecision(1) << std::setfill('@');
    }
    ASSERT_EQ(static_cast<unsigned>(s.flags()), static_cast<unsigned>(flags));
    ASSERT_EQ(s.precision(), prec);
    ASSERT_EQ(s.fill(), fill);

    // Destructors run during unwinding; a manual save/restore pair does not.
    // That is the whole reason this is a class and not two statements.
    try {
        const heom::StreamFormat guard(s);
        s << std::fixed << std::setprecision(2);
        throw std::runtime_error("unwind");
    } catch (const std::runtime_error&) {}
    ASSERT_EQ(static_cast<unsigned>(s.flags()), static_cast<unsigned>(flags));
    ASSERT_EQ(s.precision(), prec);

    // nullptr sink (HierarchyTruncation::log_ when silent) must be a no-op,
    // so call sites need no conditional.
    { const heom::StreamFormat guard(static_cast<std::ostream*>(nullptr)); }
    PASS();
}

// The invariant the guard exists to enforce: a printer must not change the
// caller's stream state. Before the guard, print_density_matrix_summary left
// `fixed`+precision(3) on std::cout, which reformatted every double printed
// later in the process -- including test diagnostics, where a 3.33e-03
// discrepancy rendered as "0.0" and read like a passing value.
static void test_console_printers_restore_stream_state() {
    TEST("console printers leave std::cout formatting untouched");
    std::ostringstream sink;
    std::streambuf* const real_buf = std::cout.rdbuf(sink.rdbuf());
    const heom::StreamFormat outer(std::cout);   // keep the suite's own cout clean

    std::cout << std::scientific << std::setprecision(9) << std::setfill('#');
    const auto flags = std::cout.flags();
    const auto prec  = std::cout.precision();
    const char fill  = std::cout.fill();

    Eigen::VectorXd eigenvalues(2);
    eigenvalues << 0.0, 1.0e-3;
    const Eigen::MatrixXcd eigenvectors = Eigen::MatrixXcd::Identity(2, 2);
    const Eigen::MatrixXcd rho = Eigen::MatrixXcd::Identity(2, 2) * 0.5;

    heom::console::print_eigenspectrum(eigenvalues, 0, 2, 12000.0);
    heom::console::print_exciton_composition(eigenvectors, 0, 2);
    heom::console::print_density_matrix_summary(rho, 2);

    const bool flags_ok = std::cout.flags() == flags;
    const bool prec_ok  = std::cout.precision() == prec;
    const bool fill_ok  = std::cout.fill() == fill;
    std::cout.rdbuf(real_buf);

    ASSERT_EQ(flags_ok, true);
    ASSERT_EQ(prec_ok, true);
    ASSERT_EQ(fill_ok, true);
    PASS();
}

static void test_format_computation_time_tiers() {
    using heom::console::format_computation_time;
    TEST("format_computation_time: tier boundaries (s / m / h / d)");
    // < 1 min → 3-decimal seconds (byte-identical to the original format).
    ASSERT_EQ(format_computation_time(0.5),    std::string("0.500s"));
    ASSERT_EQ(format_computation_time(45.0),   std::string("45.000s"));
    ASSERT_EQ(format_computation_time(59.999), std::string("59.999s"));
    // [1 min, 1 h) → "Mm Ss".
    ASSERT_EQ(format_computation_time(60.0),   std::string("1m 0s"));
    ASSERT_EQ(format_computation_time(3599.0), std::string("59m 59s"));
    // [1 h, 24 h) → "Hh Mm Ss".
    ASSERT_EQ(format_computation_time(3600.0),       std::string("1h 0m 0s"));
    ASSERT_EQ(format_computation_time(3661.0),       std::string("1h 1m 1s"));
    ASSERT_EQ(format_computation_time(86399.0),      std::string("23h 59m 59s"));
    // >= 24 h → "Dd Hh Mm Ss".
    ASSERT_EQ(format_computation_time(86400.0),      std::string("1d 0h 0m 0s"));
    ASSERT_EQ(format_computation_time(90061.0),      std::string("1d 1h 1m 1s"));
    ASSERT_EQ(format_computation_time(2.0 * 86400.0 + 3.0 * 3600.0 + 4.0 * 60.0 + 5.0),
              std::string("2d 3h 4m 5s"));
    PASS();
}

int main() {
    std::cout << "\n=== HEOM Core Unit Tests ===\n\n";

    std::cout << "[Matsubara Expansion]\n";
    test_bath_modes_no_disorder_slot();
    test_matsubara_nu0();
    test_matsubara_nuk();
    test_matsubara_c0_imaginary();
    test_matsubara_ck_real();
    test_matsubara_markov_positive();
    test_matsubara_resonance_k1_throws();
    test_matsubara_resonance_k2_throws();
    test_matsubara_no_resonance_default_params();

    std::cout << "\n[Padé Expansion]\n";
    test_pade_poles_match_reference_K1();
    test_pade_poles_match_reference_K2();
    test_pade_poles_distinct();
    test_pade_K0_equals_matsubara_K0();
    test_pade_K_high_converges_to_matsubara_K_high();
    test_pade_markov_residue_real_only();

    std::cout << "\n[Hamiltonian Expansion]\n";
    test_excited_only();
    test_with_ground();
    test_double_excited_dim();

    std::cout << "\n[Hierarchy Index Generation]\n";
    test_hierarchy_count_trivial();
    test_hierarchy_simple_cutoff_count();
    test_hierarchy_zero_index();
    test_hierarchy_neighbor_maps();
    test_hierarchy_sorted();
    test_hierarchy_with_ground_uses_n_states();

    std::cout << "\n[d_coupling Construction (n_hat_n)]\n";
    test_d_coupling_excited_only();
    test_d_coupling_with_ground();
    test_d_coupling_double_excited();

    std::cout << "\n[Sink Site Mapping]\n";
    test_sink_site_excited_only();
    test_sink_site_with_ground();
    test_sink_site_double_excited();
    test_sink_site_out_of_range();
    test_sink_site_out_of_range_high();
    test_sink_enabled_zero_rate_throws();
    test_sink_enabled_negative_rate_throws();

    std::cout << "\n[Excited Site Mapping]\n";
    test_excited_site_excited_only();
    test_excited_site_with_ground();
    test_excited_site_double_excited();
    test_excited_site_out_of_range_low();
    test_excited_site_out_of_range_high();
    test_exciton_index_excited_only_maps_to_zero_based();
    test_exciton_index_with_ground_skips_ground();
    test_exciton_index_with_ground_auto_filename_uses_user_index();
    test_exciton_index_with_ground_high_throws();
    test_exciton_index_double_excited_throws();
    test_custom_matrix_non_hermitian_throws();
    test_custom_matrix_trace_not_one_throws();

    std::cout << "\n[Padé / expansion_method YAML Contract]\n";
    test_pade_yaml_invalid_value_throws();
    test_expansion_method_missing_throws();
    test_n_matsubara_field_rejected();
    test_n_extra_poles_range_checked();
    test_auto_filename_marks_matsubara();

    std::cout << "\n[integrator YAML Contract]\n";
    test_integrator_missing_throws();
    test_integrator_explicit_values_parse();
    test_integrator_invalid_value_throws();

    std::cout << "\n[Config Validation]\n";
    test_output_interval_zero_throws();
    test_output_interval_greater_than_n_steps_throws();
    test_print_interval_zero_disables_printing();
    test_print_interval_negative_throws();
    test_ulist_zero_is_off();
    test_ulist_value_is_on();
    test_ulist_absent_defaults_off();
    test_ulist_out_of_range_throws();

    std::cout << "\n[Disorder Removal / Hierarchy]\n";
    test_disorder_off_uses_legacy_path();
    // yaml bath schema tests (reject removed disorder-in-HEOM keys, require L_cutoff).
    test_yaml_static_disorder_fwhm_rejected();
    test_yaml_bath_missing_L_cutoff_throws();
    test_yaml_L_cutoff_int8_limit_throws();

    std::cout << "\n[Memory Budget]\n";
    test_budget_bucket_consistency();
    test_budget_manual_excited_only();
    test_budget_filter_off_zero_buffers();
    test_budget_filter_on_includes_buffers();
    test_budget_basis_scales_N_squared();
    test_budget_check_fits();
    test_budget_check_exceeds();
    test_budget_check_unknown_gpu();
    test_budget_safety_margin();
    test_budget_safety_margin_invalid_throws();
    test_budget_hierarchy_real_M();
    test_budget_host_buffer_is_N_squared();

    std::cout << "\n[GPU Alloc Plan]\n";
    test_plan_total_matches_budget();
    test_plan_filter_off_buffer_count();
    test_plan_filter_on_buffer_count();
    test_plan_canonical_names_present();
    test_derivative_total_elements_overflow_throws();
    test_derivative_shared_memory_limit_throws();
    test_derivative_context_requires_total_decay();
    test_derivative_context_manifold_block_rules();
    test_spectrum_filtering_metadata_uses_config();
    test_typed_alloc_type_mismatch_throws();
    test_typed_alloc_missing_buffer_throws();

    std::cout << "\n[Config API Stability]\n";
    test_in_memory_config_construction();
    test_load_yaml_preserves_source_snapshot();
    test_config_hash_ignores_comments_and_format();
    test_config_hash_physics_golden();

    std::cout << "\n[Spectra]\n";
    test_dipole_with_ground_basic();
    test_dipole_with_ground_strengths();
    test_dipole_excited_only_throws();
    test_dipole_double_excited_block();
    test_absorption_yaml_loads_clean();
    test_dipoles_all_zero_strengths_throw();
    test_dipoles_negative_strength_throws();
    test_linear_response_rejects_non_ground_basis();
    test_linear_response_rejects_all_zero_dipoles();
    test_spectrum_result_shape_validation();

    std::cout << "\n[Console Formatters]\n";
    test_stream_format_guard();
    test_console_printers_restore_stream_state();
    test_format_computation_time_tiers();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
