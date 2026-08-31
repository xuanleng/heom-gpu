// Minimal GPU correctness test for compute_heom_derivatives_kernel.
//
// Configures the smallest hierarchy that can still exercise the kernel:
//
//   M = 1, N = 2, K = 0, n_bath = 1, no filtering, no sink.
//
// In this regime every HEOM term except the commutator (Term 1) vanishes:
//   - Term 2 wants total_decay[0] = sum alpha_{n,k} nu_k. We set
//     total_decay[0] = 0 explicitly (sole ADO has alpha = 0).
//   - Term 3 wants markov_residue != 0 AND d_i^(n) != d_j^(n). We set
//     markov_residue = 0.
//   - Term 4 / Term 5 read neighbours; with M = 1 the maps are -1
//     everywhere so the (alpha_plus >= 0) / (alpha_minus >= 0) guards
//     skip the whole loop body.
//   - Sink is disabled.
//
// The expected output is therefore exactly drho/dt = -i [H, rho], which we
// compute analytically on the host and compare element by element. A 1-ULP
// disagreement here would mean the kernel's commutator path, the
// shared-memory H/d cache, the CUDA_CHECK_KERNEL plumbing
//, or the HEOM_BLOCK_SIZE definition is broken — none of
// which the existing CPU-only test_core can catch.

#include <stdexcept>
#include <sstream>
#include <iomanip>
#include "core/heom_derivatives.cuh"
#include "core/rk4_integrator.cuh"
#include "core/ado_filter.cuh"
#include "core/cuda_common.cuh"
#include "core/dipole_apply.cuh"
#include "core/precision.hpp"
#include "heom_config.hpp"
#include "heom_trajectory.hpp"
#include "gpu_alloc_plan.hpp"
#include "system_context.hpp"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

// ρ-side buffers follow the library's hier precision (FP64 vs mixed FP32); the
// same source builds test_gpu_kernel (heom_core) and test_gpu_kernel_mp
// (heom_core_mp). H / c_bath stay cuDoubleComplex (FP64) as in HEOMContext.
using heom::complex_t;
using heom::c_make;
using heom::c_re;
using heom::c_im;

// Tolerance keys off the build's precision: a few ULP for
// FP64, FP32 machine-eps scale for mixed. The reference is exact, so this is
// purely the pipeline's own rounding.
constexpr double kKernelTol = heom::kMixedPrecision ? 1e-4 : 1e-15;
constexpr double kCarryTol  = heom::kMixedPrecision ? 1e-4 : 1e-10;
constexpr double kMuTol     = heom::kMixedPrecision ? 1e-4 : 1e-12;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "FAIL: " << what << " -- " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

#define CHK(call) check_cuda((call), #call)

std::string write_tiny_trajectory_yaml() {
    const std::string path = "/tmp/test_heom_trajectory_guard.yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 1\n"
      << "  basis: \"excited_only\"\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100.0]\n"
      << "  w0: 12000.0\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 35.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: matsubara\n"
      << "  n_extra_poles: 0\n"
      << "  L_cutoff: 0\n"
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
      << "    save_path: \"/tmp/\"\n"
      << "    filename: \"trajectory_guard\"\n"
      << "    use_timestamp: false\n";
    return path;
}

// Dimer (N=2), L=2, filtering OFF — small but multi-ADO, so carry-through
// snapshot/restore actually exercises α>0 slots. Filtering off keeps
// snapshot/restore byte-exact (full M, no norm-based drop).
std::string write_carry_through_yaml() {
    const std::string path = "/tmp/test_heom_carry_through.yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  basis: \"excited_only\"\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100.0, 100.0]\n"
      << "    - [100.0, 12000.0]\n"
      << "  w0: 12000.0\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 35.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: matsubara\n"
      << "  n_extra_poles: 0\n"
      << "  L_cutoff: 2\n"
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
      << "    save_path: \"/tmp/\"\n"
      << "    filename: \"carry_through\"\n"
      << "    use_timestamp: false\n";
    return path;
}

// Doubly-excited dimer: N = 1(g) + 2(e) + 1(f) = 4, the smallest basis that
// has all three manifolds and therefore all three shapes of manifold block —
// (g,e) with row_off < col_off, (f,e) with row_off > col_off, and a diagonal
// (e,e). Filtering off, L=2, no extra poles: a handful of ADOs, still with real
// +/- neighbours.
std::string write_double_excited_dimer_yaml() {
    const std::string path = "/tmp/test_heom_double_excited_dimer.yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  basis: \"double_excited\"\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100.0, 100.0]\n"
      << "    - [100.0, 12000.0]\n"
      << "  w0: 12000.0\n"
      << "  dipoles:\n"
      << "    - [1.0, 0.0, 0.0]\n"
      << "    - [0.0, 1.0, 0.0]\n"
      << "  strengths: [4.5, 4.5]\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 35.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: matsubara\n"
      << "  n_extra_poles: 0\n"
      << "  L_cutoff: 2\n"
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
      << "    save_path: \"/tmp/\"\n"
      << "    filename: \"double_excited_dimer\"\n"
      << "    use_timestamp: false\n";
    return path;
}

bool test_trajectory_solve_without_initial_throws() {
    auto cfg = HEOMConfig::Config::load_yaml(write_tiny_trajectory_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);  // dynamics block: time grid + initial
    SystemContext sys(cfg);
    HEOMTrajectory traj(sys, /*enable_hdf5=*/false);

    try {
        traj.solve();
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        if (what.find("before set_initial") != std::string::npos) {
            return true;
        }
        std::cerr << "FAIL: unexpected trajectory guard error: " << what << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: unexpected exception type: " << e.what() << "\n";
        return false;
    }

    std::cerr << "FAIL: HEOMTrajectory::solve() without set_initial() did not throw\n";
    return false;
}

// RAII wrapper: cudaMalloc on construction, cudaFree on destruction.
template <typename T>
struct DevBuf {
    T* ptr = nullptr;
    explicit DevBuf(size_t n) { CHK(cudaMalloc(&ptr, n * sizeof(T))); }
    ~DevBuf() { if (ptr) cudaFree(ptr); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

// commit b.1: full-ADO single-side dipole kernel vs CPU reference.
// M ADOs, N×N each; deterministic complex ρ and μ; check μ·ρ_α and ρ_α·μ.
bool test_single_side_mu() {
    using cplx = std::complex<double>;
    const int M = 3, N = 4;

    std::vector<complex_t> h_rho(M * N * N), h_mu(N * N);
    std::vector<cplx> rho(M * N * N), mu(N * N);
    for (int a = 0; a < M; ++a)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const double re = 0.1 * (a + 1) + 0.3 * i - 0.2 * j;
                const double im = 0.05 * (a + 1) - 0.1 * i + 0.15 * j;
                rho[a * N * N + i * N + j] = cplx(re, im);
                h_rho[a * N * N + i * N + j] = c_make(re, im);
            }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const double re = 0.2 * i + 0.1 * j + 0.05;
            const double im = -0.1 * i + 0.07 * j;
            mu[i * N + j] = cplx(re, im);
            h_mu[i * N + j] = c_make(re, im);
        }

    DevBuf<complex_t> d_rho(M * N * N), d_out(M * N * N), d_mu(N * N);
    CHK(cudaMemcpy(d_rho.ptr, h_rho.data(), M * N * N * sizeof(complex_t),
                   cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_mu.ptr, h_mu.data(), N * N * sizeof(complex_t),
                   cudaMemcpyHostToDevice));

    auto run_and_check = [&](int side) -> bool {
        launch_apply_single_side_mu(d_rho.ptr, d_out.ptr, d_mu.ptr, side, M, N);
        CHK(cudaDeviceSynchronize());
        std::vector<complex_t> h_out(M * N * N);
        CHK(cudaMemcpy(h_out.data(), d_out.ptr, M * N * N * sizeof(complex_t),
                       cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int a = 0; a < M; ++a)
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    cplx acc(0.0, 0.0);
                    for (int k = 0; k < N; ++k) {
                        if (side == 0)
                            acc += mu[i * N + k] * rho[a * N * N + k * N + j];  // μ·ρ
                        else
                            acc += rho[a * N * N + i * N + k] * mu[k * N + j];  // ρ·μ
                    }
                    const complex_t g = h_out[a * N * N + i * N + j];
                    if (std::abs(c_re(g) - acc.real()) > kMuTol ||
                        std::abs(c_im(g) - acc.imag()) > kMuTol) {
                        std::cout << "  single-side mu side=" << side << " a=" << a
                                  << " (" << i << "," << j << "): got ("
                                  << c_re(g) << "," << c_im(g) << ") exp ("
                                  << acc.real() << "," << acc.imag() << ")\n";
                        ok = false;
                    }
                }
        return ok;
    };

    const bool left_ok = run_and_check(0);
    const bool right_ok = run_and_check(1);
    if (left_ok && right_ok) {
        std::cout << "single-side mu kernel (left/right vs CPU): PASS\n";
        return true;
    }
    std::cout << "single-side mu kernel: FAIL\n";
    return false;
}

// Task A1: batched per-slot single-side dipole. A B=2 launch with a DISTINCT μ
// per slot (μ0/μ1) and distinct ρ per slot must reproduce, slot-for-slot, two
// independent single-slot launch_apply_single_side_mu calls (the single-slot path).
// Bit-exact in FP64 (slot b runs identical arithmetic on its own data), kMuTol
// in mixed. Guards per-slot μ addressing (mu[b·N²]) and slot independence (boff).
bool test_batch_single_side_mu() {
    const int M = 3, N = 4;
    const size_t N2 = static_cast<size_t>(N) * N;
    const size_t stride = static_cast<size_t>(M) * N2;

    std::vector<complex_t> h_rho(2 * stride), h_mu(2 * N2);
    for (int b = 0; b < 2; ++b)
        for (int a = 0; a < M; ++a)
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    const double re = 0.1 * (a + 1) + 0.3 * i - 0.2 * j + 0.5 * b;
                    const double im = 0.05 * (a + 1) - 0.1 * i + 0.15 * j - 0.3 * b;
                    h_rho[b * stride + a * N2 + i * N + j] = c_make(re, im);
                }
    for (int b = 0; b < 2; ++b)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const double re = 0.2 * i + 0.1 * j + 0.05 + 0.4 * b;
                const double im = -0.1 * i + 0.07 * j + 0.2 * b;
                h_mu[b * N2 + i * N + j] = c_make(re, im);
            }

    DevBuf<complex_t> d_rho(2 * stride), d_mu(2 * N2);
    DevBuf<complex_t> d_out_batch(2 * stride), d_out_ref(2 * stride);
    CHK(cudaMemcpy(d_rho.ptr, h_rho.data(), 2 * stride * sizeof(complex_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_mu.ptr, h_mu.data(), 2 * N2 * sizeof(complex_t), cudaMemcpyHostToDevice));

    const double tol = heom::kMixedPrecision ? kMuTol : 0.0;

    auto run_and_check = [&](int side) -> bool {
        CHK(cudaMemset(d_out_batch.ptr, 0, 2 * stride * sizeof(complex_t)));
        CHK(cudaMemset(d_out_ref.ptr, 0, 2 * stride * sizeof(complex_t)));
        // Batched: B=2, per-slot μ in one launch.
        launch_apply_single_side_mu_batch(d_rho.ptr, d_out_batch.ptr, d_mu.ptr,
                                          side, M, N, 2);
        // Reference: two independent single-slot launches (bit-exact target).
        launch_apply_single_side_mu(d_rho.ptr, d_out_ref.ptr, d_mu.ptr,
                                    side, M, N);
        launch_apply_single_side_mu(d_rho.ptr + stride, d_out_ref.ptr + stride,
                                    d_mu.ptr + N2, side, M, N);
        CHK(cudaDeviceSynchronize());

        std::vector<complex_t> ob(2 * stride), orf(2 * stride);
        CHK(cudaMemcpy(ob.data(), d_out_batch.ptr, 2 * stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(orf.data(), d_out_ref.ptr, 2 * stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < 2 * stride; ++k)
            if (std::abs(c_re(ob[k]) - c_re(orf[k])) > tol ||
                std::abs(c_im(ob[k]) - c_im(orf[k])) > tol) {
                std::cout << "  batch mu side=" << side << " k=" << k
                          << ": batched (" << c_re(ob[k]) << "," << c_im(ob[k])
                          << ") ref (" << c_re(orf[k]) << "," << c_im(orf[k]) << ")\n";
                return false;
            }
        return true;
    };

    const bool ok = run_and_check(0) && run_and_check(1);
    if (ok) std::cout << "batch single-side mu (B=2 per-slot == 2x single): PASS\n";
    else    std::cout << "batch single-side mu: FAIL\n";
    return ok;
}

// Task A2: batched gather/scatter over slots. B=2, multi-ADO, gather the active
// blocks (shared ids) of [B·M·N²] into staging [B·n_active·N²], then scatter
// back into a ZEROED [B·M·N²]. The round-trip must reproduce every active block
// bit-for-bit (pure copy, no arithmetic ⇒ exact both builds) and leave inactive
// blocks zero, in every slot. Guards slot stride (b·M·N² for ρ, b·n_active·N²
// for staging) and shared-ids addressing.
bool test_batch_gather_scatter() {
    const int M = 6, N = 3;
    const size_t N2 = static_cast<size_t>(N) * N;
    const size_t stride = static_cast<size_t>(M) * N2;
    const std::vector<int> ids = {0, 2, 3, 5};   // n_active=4, shared by both slots
    const int n_active = static_cast<int>(ids.size());

    std::vector<complex_t> h_rho(2 * stride);
    for (int b = 0; b < 2; ++b)
        for (int a = 0; a < M; ++a)
            for (size_t t = 0; t < N2; ++t) {
                const double re = 0.1 * (a + 1) + 0.01 * static_cast<double>(t) + 0.5 * b;
                const double im = -0.05 * (a + 1) + 0.02 * static_cast<double>(t) - 0.3 * b;
                h_rho[b * stride + a * N2 + t] = c_make(re, im);
            }

    DevBuf<complex_t> d_rho(2 * stride), d_stage(2 * n_active * N2), d_rho2(2 * stride);
    DevBuf<int> d_ids(n_active);
    CHK(cudaMemcpy(d_rho.ptr, h_rho.data(), 2 * stride * sizeof(complex_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_ids.ptr, ids.data(), n_active * sizeof(int), cudaMemcpyHostToDevice));
    CHK(cudaMemset(d_rho2.ptr, 0, 2 * stride * sizeof(complex_t)));

    launch_gather_ado_blocks_batch(d_rho.ptr, d_stage.ptr, d_ids.ptr, n_active, N, M, 2);
    launch_scatter_ado_blocks_batch(d_stage.ptr, d_rho2.ptr, d_ids.ptr, n_active, N, M, 2);
    CHK(cudaDeviceSynchronize());

    std::vector<complex_t> out(2 * stride);
    CHK(cudaMemcpy(out.data(), d_rho2.ptr, 2 * stride * sizeof(complex_t), cudaMemcpyDeviceToHost));

    std::vector<char> is_active(M, 0);
    for (int id : ids) is_active[id] = 1;
    bool ok = true;
    for (int b = 0; b < 2 && ok; ++b)
        for (int a = 0; a < M && ok; ++a)
            for (size_t t = 0; t < N2; ++t) {
                const complex_t g = out[b * stride + a * N2 + t];
                const complex_t e = is_active[a] ? h_rho[b * stride + a * N2 + t]
                                                 : c_make(0.0, 0.0);
                if (c_re(g) != c_re(e) || c_im(g) != c_im(e)) {
                    std::cout << "  gather/scatter b=" << b << " a=" << a << " t=" << t
                              << ": got (" << c_re(g) << "," << c_im(g) << ") exp ("
                              << c_re(e) << "," << c_im(e) << ")\n";
                    ok = false;
                    break;
                }
            }

    if (ok) std::cout << "batch gather/scatter (B=2 round-trip, shared ids): PASS\n";
    else    std::cout << "batch gather/scatter: FAIL\n";
    return ok;
}

// Task A3: batched per-slot trace detection. B=2, each slot has a DISTINCT ρ
// (α=0 block) and DISTINCT μ_det; the batched launch must write out[b·ns+idx]
// == the single-slot launch_trace_BR for that slot (identical tree reduction ⇒
// bit-exact FP64, 1e-4 mixed). Guards ρ slot stride (b·M·N², reads α=0 only,
// α>0 junk ignored), per-slot μ_det (b·N²) and per-slot output (b·ns+idx).
bool test_batch_trace_BR() {
    const int M = 2, N = 3, B = 2;
    const size_t N2 = static_cast<size_t>(N) * N;
    const size_t stride = static_cast<size_t>(M) * N2;
    const int ns = 5, idx = 2;

    std::vector<complex_t> h_rho(B * stride), h_mu(B * N2);
    for (int b = 0; b < B; ++b)
        for (int a = 0; a < M; ++a)          // a=1 is junk that MUST be ignored
            for (size_t t = 0; t < N2; ++t) {
                const double re = 0.2 * (a + 1) + 0.03 * static_cast<double>(t) + 0.7 * b;
                const double im = -0.1 * (a + 1) + 0.05 * static_cast<double>(t) - 0.4 * b;
                h_rho[b * stride + a * N2 + t] = c_make(re, im);
            }
    for (int b = 0; b < B; ++b)
        for (size_t t = 0; t < N2; ++t)
            h_mu[b * N2 + t] = c_make(0.15 * static_cast<double>(t) + 0.3 * b,
                                      -0.08 * static_cast<double>(t) + 0.2 * b);

    DevBuf<complex_t> d_rho(B * stride), d_mu(B * N2);
    DevBuf<complex_t> d_out_batch(B * ns), d_out_ref(B * ns);
    CHK(cudaMemcpy(d_rho.ptr, h_rho.data(), B * stride * sizeof(complex_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_mu.ptr, h_mu.data(), B * N2 * sizeof(complex_t), cudaMemcpyHostToDevice));
    CHK(cudaMemset(d_out_batch.ptr, 0, B * ns * sizeof(complex_t)));
    CHK(cudaMemset(d_out_ref.ptr, 0, B * ns * sizeof(complex_t)));

    launch_trace_BR_batch(d_mu.ptr, d_rho.ptr, d_out_batch.ptr, idx, N, M, B, ns);
    for (int b = 0; b < B; ++b)
        launch_trace_BR(d_mu.ptr + b * N2, d_rho.ptr + b * stride,
                        d_out_ref.ptr + b * ns, idx, N);
    CHK(cudaDeviceSynchronize());

    std::vector<complex_t> ob(B * ns), orf(B * ns);
    CHK(cudaMemcpy(ob.data(), d_out_batch.ptr, B * ns * sizeof(complex_t), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(orf.data(), d_out_ref.ptr, B * ns * sizeof(complex_t), cudaMemcpyDeviceToHost));

    const double tol = heom::kMixedPrecision ? kMuTol : 0.0;
    bool ok = true;
    for (int b = 0; b < B; ++b) {
        const complex_t g = ob[b * ns + idx];
        const complex_t e = orf[b * ns + idx];
        if (std::abs(c_re(g) - c_re(e)) > tol || std::abs(c_im(g) - c_im(e)) > tol) {
            std::cout << "  trace_BR b=" << b << ": batched (" << c_re(g) << ","
                      << c_im(g) << ") ref (" << c_re(e) << "," << c_im(e) << ")\n";
            ok = false;
        }
    }

    if (ok) std::cout << "batch trace_BR (B=2 per-slot mu_det/out == single): PASS\n";
    else    std::cout << "batch trace_BR: FAIL\n";
    return ok;
}

// U-local filter: the U-local norm scan + mark must equal the
// full-M reference on the scan domain and leave flags outside it untouched.
// Also locks the scan-domain rule for the dense→compact transition: when the
// previous write set was the full grid, a full-M scan (scan_list = nullptr)
// must reproduce the reference everywhere.
// Runs both norm kernels (per-thread N*N<64 and warp N*N>=64).
bool test_ulocal_filter_marking() {
    bool all_ok = true;

    for (const int N : {3, 10}) {   // 9 < 64 <= 100: covers both kernel paths
        const int M = 8;
        const int NN = N * N;
        const double threshold = 0.3;

        // Block a has max |element| = 1/(a+1): active set under 0.3 is
        // {0, 1, 2} (norms 1.0, 0.5, 0.333…) plus the unconditional α=0.
        std::vector<complex_t> h_rho(M * NN, c_make(0.0, 0.0));
        for (int a = 0; a < M; ++a)
            h_rho[a * NN + (a % NN)] = c_make(1.0 / (a + 1), 0.0);

        DevBuf<complex_t>       d_rho(M * NN);
        DevBuf<double>          d_norms(M);
        DevBuf<int8_t>          d_flags(M);
        CHK(cudaMemcpy(d_rho.ptr, h_rho.data(),
                       M * NN * sizeof(complex_t), cudaMemcpyHostToDevice));

        // Full-M reference (also the dense→compact transition case: previous
        // write set = full grid ⇒ scan_list = nullptr must be used).
        std::vector<int8_t> flags_ref(M);
        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N, 0, N, 0, N);
        filter_ados_mark_only(d_norms.ptr, d_flags.ptr, threshold, M);
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(flags_ref.data(), d_flags.ptr, M, cudaMemcpyDeviceToHost));

        // U-local pass. Write set U = {0,1,2,3,4}: superset of the active set,
        // as the previous whole-step's closure always is. Pre-state: garbage
        // (1) inside U to prove the kernel rewrites it, 0 outside U (A ⊆ U ⇒
        // outside is inactive) to prove the kernel never touches it.
        const std::vector<int> h_ulist = {0, 1, 2, 3, 4};
        const int u_count = static_cast<int>(h_ulist.size());
        DevBuf<int> d_ulist(u_count);
        CHK(cudaMemcpy(d_ulist.ptr, h_ulist.data(), u_count * sizeof(int),
                       cudaMemcpyHostToDevice));

        std::vector<int8_t> pre(M, 0);
        for (int a : h_ulist) pre[a] = 1;
        CHK(cudaMemcpy(d_flags.ptr, pre.data(), M, cudaMemcpyHostToDevice));
        // Poison the norms so a full-M mark (wrong domain) would misflag.
        {
            std::vector<double> junk(M, 1.0e9);
            CHK(cudaMemcpy(d_norms.ptr, junk.data(), M * sizeof(double),
                           cudaMemcpyHostToDevice));
        }

        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N, 0, N, 0, N,
                          d_ulist.ptr, u_count);
        filter_ados_mark_only(d_norms.ptr, d_flags.ptr, threshold, M,
                              d_ulist.ptr, u_count);
        CHK(cudaDeviceSynchronize());

        std::vector<int8_t> flags_got(M);
        CHK(cudaMemcpy(flags_got.data(), d_flags.ptr, M, cudaMemcpyDeviceToHost));

        std::vector<int8_t> in_u(M, 0);
        for (int a : h_ulist) in_u[a] = 1;
        for (int a = 0; a < M; ++a) {
            const int8_t expect = in_u[a] ? flags_ref[a] : 0;
            if (flags_got[a] != expect) {
                std::cout << "  ulocal filter N=" << N << " a=" << a
                          << ": got " << int(flags_got[a])
                          << " expect " << int(expect)
                          << (in_u[a] ? " (in U)" : " (outside U)") << "\n";
                all_ok = false;
            }
        }
    }

    if (all_ok) {
        std::cout << "ulocal filter marking (== full-M on U, untouched outside): PASS\n";
        return true;
    }
    std::cout << "ulocal filter marking: FAIL\n";
    return false;
}

// Manifold-blocked norm scan. Three properties, and the middle one is the only
// one with teeth: an implementation that accepts the block parameters and then
// ignores them still passes (1), because out-of-block is zero there.
//   (1) equivalence : out-of-block zero  ⇒ blocked norms == whole-matrix norms
//   (2) confinement : out-of-block LARGE ⇒ blocked norms still ignore it
//   (3) rejection   : empty / out-of-range blocks throw
// Both kernels are covered: N=4 (16 < 64, per-thread) and N=29 (warp, and the
// production ESA size, whose odd row stride is the interesting alignment case).
bool test_manifold_blocked_norms() {
    bool all_ok = true;

    for (const int N : {4, 29}) {
        const int M  = 8;
        const int NN = N * N;
        // Non-square, offset block: a square or origin-anchored one would let a
        // row/col transposition slip through.
        const int row_off = 1, n_row = N - 1;
        const int col_off = 0, n_col = 1;

        std::vector<complex_t> h_rho(M * NN, c_make(0.0, 0.0));
        for (int a = 0; a < M; ++a)
            for (int r = 0; r < n_row; ++r)
                for (int c = 0; c < n_col; ++c) {
                    // Distinct magnitudes so the max is a specific element.
                    const double v = 1.0 / (a + 1) - 0.001 * (r * n_col + c);
                    h_rho[a * NN + (row_off + r) * N + col_off + c] =
                        c_make(v, 0.0);
                }

        DevBuf<complex_t> d_rho(M * NN);
        DevBuf<double>    d_norms(M);
        CHK(cudaMemcpy(d_rho.ptr, h_rho.data(), M * NN * sizeof(complex_t),
                       cudaMemcpyHostToDevice));

        // (1) equivalence
        std::vector<double> full(M), blocked(M);
        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N, 0, N, 0, N);
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(full.data(), d_norms.ptr, M * sizeof(double),
                       cudaMemcpyDeviceToHost));

        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N,
                          row_off, n_row, col_off, n_col);
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(blocked.data(), d_norms.ptr, M * sizeof(double),
                       cudaMemcpyDeviceToHost));

        for (int a = 0; a < M; ++a)
            if (blocked[a] != full[a]) {          // bit-equal, not approximate
                std::cout << "  blocked norms N=" << N << " a=" << a
                          << ": blocked " << blocked[a]
                          << " != whole-matrix " << full[a] << "\n";
                all_ok = false;
            }

        // (2) confinement: flood everything OUTSIDE the block with a value far
        // above anything inside it. The blocked scan must not see it.
        std::vector<complex_t> h_poison = h_rho;
        for (int a = 0; a < M; ++a)
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    const bool inside = (i >= row_off && i < row_off + n_row &&
                                         j >= col_off && j < col_off + n_col);
                    if (!inside) h_poison[a * NN + i * N + j] = c_make(1.0e3, 0.0);
                }
        CHK(cudaMemcpy(d_rho.ptr, h_poison.data(), M * NN * sizeof(complex_t),
                       cudaMemcpyHostToDevice));

        std::vector<double> confined(M);
        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N,
                          row_off, n_row, col_off, n_col);
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(confined.data(), d_norms.ptr, M * sizeof(double),
                       cudaMemcpyDeviceToHost));

        for (int a = 0; a < M; ++a)
            if (confined[a] != blocked[a]) {
                std::cout << "  blocked norms N=" << N << " a=" << a
                          << ": out-of-block 1e3 leaked in (" << confined[a]
                          << " vs " << blocked[a] << ")\n";
                all_ok = false;
            }
    }

    // (3) rejection
    {
        const int N = 8, M = 2;
        DevBuf<complex_t> d_rho(M * N * N);
        DevBuf<double>    d_norms(M);
        struct Bad { const char* what; int ro, nr, co, nc; };
        const Bad bad[] = {
            {"empty n_row",      0, 0, 0, N},
            {"empty n_col",      0, N, 0, 0},
            {"row past the end", 1, N, 0, N},
            {"negative offset", -1, N, 0, N},
        };
        for (const Bad& b : bad) {
            bool threw = false;
            try {
                compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N,
                                  b.ro, b.nr, b.co, b.nc);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            if (!threw) {
                std::cout << "  blocked norms: " << b.what
                          << " was accepted, must throw\n";
                all_ok = false;
            }
        }
    }

    if (all_ok) {
        std::cout << "manifold blocked norms (equivalence + confinement + rejection): PASS\n";
        return true;
    }
    std::cout << "manifold blocked norms: FAIL\n";
    return false;
}

// Read ρ_0 element-by-element through expectation(): expectation(E_ab) =
// Tr(E_ab·ρ0) = ρ0(b,a). Independent of the host mirror (propagate_steps does
// not refresh it) — this is exactly the path 2DES drivers use.
Eigen::MatrixXcd read_rho0_via_expectation(const HEOMTrajectory& traj, int N) {
    Eigen::MatrixXcd rho(N, N);
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b) {
            Eigen::MatrixXcd E = Eigen::MatrixXcd::Zero(N, N);
            E(a, b) = std::complex<double>(1.0, 0.0);
            rho(b, a) = traj.expectation(E);
        }
    return rho;
}

// commit b.2 (b2-5): carry-through primitives — snapshot round-trip,
// segmented==continuous, single-side μ on the physical layer.
bool test_carry_through_primitives() {
    using namespace heom::spectra;
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);  // dynamics block: time grid (dt) + initial
    SystemContext sys(cfg);
    const int N = sys.N();

    Eigen::MatrixXcd rho0 = Eigen::MatrixXcd::Zero(N, N);
    rho0(0, 0) = std::complex<double>(1.0, 0.0);   // site-1 population

    bool ok = true;

    // Test: segmented == continuous (reuses one trajectory; also checks that
    // set_initial() resets internal_step_ so the second run starts clean).
    {
        HEOMTrajectory traj(sys, /*enable_hdf5=*/false);
        const int A = 7, B = 5;
        traj.set_initial(rho0);
        traj.propagate_steps(A + B);
        const Eigen::MatrixXcd rho_cont = read_rho0_via_expectation(traj, N);

        traj.set_initial(rho0);
        traj.propagate_steps(A);
        traj.propagate_steps(B);
        const Eigen::MatrixXcd rho_seg = read_rho0_via_expectation(traj, N);

        const double err = (rho_cont - rho_seg).cwiseAbs().maxCoeff();
        if (err > kCarryTol) {
            std::cout << "  segmented!=continuous: max|delta|=" << err << "\n";
            ok = false;
        }
    }

    // Test: snapshot round-trip — propagate K, snapshot, perturb, restore,
    // physical layer must match the snapshot point. Exercises the SAME device
    // snapshot/restore primitives the production 2DES carry-through driver uses
    // (snapshot_device/restore_device), not a test-only host mirror of them.
    {
        HEOMTrajectory traj(sys, /*enable_hdf5=*/false);
        const int K = 6, K2 = 4;
        traj.set_initial(rho0);
        traj.propagate_steps(K);
        const Eigen::MatrixXcd rho_ref = read_rho0_via_expectation(traj, N);
        DeviceHierarchySnapshot snap;
        traj.snapshot_device(snap);

        traj.propagate_steps(K2);                  // perturb d_rho_
        traj.restore_device(snap);
        const Eigen::MatrixXcd rho_after = read_rho0_via_expectation(traj, N);

        const double err = (rho_ref - rho_after).cwiseAbs().maxCoeff();
        if (err > kCarryTol) {
            std::cout << "  snapshot round-trip: max|delta|=" << err << "\n";
            ok = false;
        }
    }

    // Test: single-side μ on the physical layer — apply Left ⇒ α=0 block = μ·ρ0.
    {
        HEOMTrajectory traj(sys, /*enable_hdf5=*/false);
        traj.set_initial(rho0);
        Eigen::MatrixXcd mu(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                mu(i, j) = std::complex<double>(0.2 * i - 0.1 * j + 0.3,
                                                0.05 * i + 0.07 * j);
        traj.apply_dipole_full_hierarchy(mu, DipoleSide::Left);
        const Eigen::MatrixXcd got = read_rho0_via_expectation(traj, N);
        const Eigen::MatrixXcd exp = mu * rho0;
        const double err = (got - exp).cwiseAbs().maxCoeff();
        if (err > kMuTol) {
            std::cout << "  single-side mu physical layer: max|delta|=" << err << "\n";
            ok = false;
        }
    }

    if (ok) std::cout << "carry-through primitives (snapshot/segment/mu): PASS\n";
    else    std::cout << "carry-through primitives: FAIL\n";
    return ok;
}

// Multi-ADO read-only context borrowing SystemContext's device tables. Only the
// ρ buffers (rho_in/drho_out) and batch are left to the caller. Filtering off
// (active_flags = nullptr, u_list = nullptr): full-M launch, so every ADO —
// including α>0 slots with real +/- neighbours — is exercised.
HEOMContext build_multi_ado_ctx(const SystemContext& sys) {
    HEOMContext c{};
    c.hamiltonian         = sys.d_hamiltonian();
    c.c_bath              = sys.d_c_bath();
    c.coeff_plus          = sys.d_coeff_plus();
    c.coeff_minus         = sys.d_coeff_minus();
    c.markov_residue      = sys.d_markov_residue();
    c.neighbor_map_plus   = sys.d_plus_map();
    c.neighbor_map_minus  = sys.d_minus_map();
    c.level_indices       = sys.d_level_indices();
    c.total_decay         = sys.d_total_decay();
    c.active_flags        = nullptr;
    c.d_coupling          = sys.d_d_coupling();
    c.M = sys.M(); c.N = sys.N(); c.K = sys.K(); c.n_bath = sys.n_bath();
    c.sink_enabled = false; c.sink_site = 0; c.gamma_sink = 0.0;
    c.u_list = nullptr; c.u_count = 0;
    c.batch = 1;
    c.row_off = 0; c.n_row = sys.N();
    c.col_off = 0; c.n_col = sys.N();
    return c;
}

// grid.y batch dimension. A B=2 launch over IDENTICAL ρ replicas must
// reproduce, in EVERY slot, the B=1 reference bit-for-bit (FP64) / within tol
// (mixed). Uses the dimer L=2 config (M=6, real +/- neighbours) so Term 4/5's
// neighbour reads are indexed through boff — a batch bug that dropped boff on
// the neighbour read would corrupt slot 1 and fail here, unlike M=1.
bool test_batch_derivatives_symmetry() {
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int M = sys.M(), N = sys.N();
    const size_t stride = static_cast<size_t>(M) * N * N;

    std::vector<complex_t> h_rho(stride);
    for (int a = 0; a < M; ++a)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const double re = 0.13 * (a + 1) + 0.31 * i - 0.22 * j;
                const double im = 0.07 * (a + 1) - 0.11 * i + 0.17 * j;
                h_rho[static_cast<size_t>(a) * N * N + i * N + j] = c_make(re, im);
            }

    DevBuf<complex_t> d_rho1(stride), d_drho1(stride);
    DevBuf<complex_t> d_rho2(2 * stride), d_drho2(2 * stride);
    CHK(cudaMemcpy(d_rho1.ptr, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_rho2.ptr, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_rho2.ptr + stride, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
    CHK(cudaMemset(d_drho2.ptr, 0, 2 * stride * sizeof(complex_t)));

    HEOMContext c1 = build_multi_ado_ctx(sys);
    c1.rho_in = d_rho1.ptr; c1.drho_out = d_drho1.ptr; c1.batch = 1;
    HEOMContext c2 = build_multi_ado_ctx(sys);
    c2.rho_in = d_rho2.ptr; c2.drho_out = d_drho2.ptr; c2.batch = 2;

    compute_heom_derivatives(c1);
    compute_heom_derivatives(c2);
    CHK(cudaDeviceSynchronize());

    std::vector<complex_t> ref(stride), s0(stride), s1(stride);
    CHK(cudaMemcpy(ref.data(), d_drho1.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(s0.data(), d_drho2.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(s1.data(), d_drho2.ptr + stride, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t k = 0; k < stride; ++k) {
        for (int slot = 0; slot < 2; ++slot) {
            const complex_t g = slot ? s1[k] : s0[k];
            if (std::abs(c_re(g) - c_re(ref[k])) > kKernelTol ||
                std::abs(c_im(g) - c_im(ref[k])) > kKernelTol) {
                std::cout << "  batch symmetry slot=" << slot << " k=" << k
                          << ": got (" << c_re(g) << "," << c_im(g) << ") ref ("
                          << c_re(ref[k]) << "," << c_im(ref[k]) << ")\n";
                ok = false;
            }
        }
    }

    if (ok) std::cout << "batch derivatives symmetry (B=2 == B=1 per slot): PASS\n";
    else    std::cout << "batch derivatives symmetry: FAIL\n";
    return ok;
}

// Term 6 (Lindblad sink). Everything else about the sink is covered — test_core
// checks the yaml and the basis-dependent site mapping — but nothing ran the
// KERNEL with sink_enabled = true, so a break in Term 6 itself was invisible.
// Population dynamics uses this routinely (2DES cannot: parse_sink() is only
// called from resolve_dynamics_config).
//
// Asserted, by differencing an on-run against an off-run:
//   (a) the contribution is exactly -gamma*(delta_is + delta_js)/2 * rho[i][j],
//       i.e. gamma on the (s,s) element and gamma/2 on the rest of row/column s;
//   (b) it is applied to EVERY ADO, not only rho_0. This is the claim of the
//       "sink acts on the whole hierarchy" section, and getting it wrong would
//       leave the physics plausible but the relaxation rate wrong;
//   (c) elements untouched by the sink are bit-identical to the off-run.
// Failure messages must not depend on ambient std::cout state. Several solver
// paths set std::fixed/setprecision on std::cout and never restore it (the
// flags are sticky), which rounds a 1e-3 discrepancy to "0.0" and turns the
// diagnostic into "got (0.0,0.0) want (0.0,0.0)".
static std::string fmt_c(double re, double im) {
    std::ostringstream s;
    s << std::scientific << std::setprecision(6) << "(" << re << ", " << im << ")";
    return s.str();
}

bool test_sink_term() {
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int M = sys.M(), N = sys.N();
    const size_t stride = static_cast<size_t>(M) * N * N;

    const int    sink_site  = 1;      // not 0: a bug that hardcodes index 0 shows up
    const double gamma_sink = 0.037;  // arbitrary, just not a round number

    std::vector<complex_t> h_rho(stride);
    for (int a = 0; a < M; ++a)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                h_rho[static_cast<size_t>(a) * N * N + i * N + j] =
                    c_make(0.13 * (a + 1) + 0.31 * i - 0.22 * j,
                           0.07 * (a + 1) - 0.11 * i + 0.17 * j);

    DevBuf<complex_t> d_rho(stride), d_off(stride), d_on(stride);
    CHK(cudaMemcpy(d_rho.ptr, h_rho.data(), stride * sizeof(complex_t),
                   cudaMemcpyHostToDevice));

    HEOMContext c_off = build_multi_ado_ctx(sys);
    c_off.rho_in = d_rho.ptr; c_off.drho_out = d_off.ptr;
    HEOMContext c_on = build_multi_ado_ctx(sys);
    c_on.rho_in = d_rho.ptr; c_on.drho_out = d_on.ptr;
    c_on.sink_enabled = true;
    c_on.sink_site    = sink_site;
    c_on.gamma_sink   = gamma_sink;

    compute_heom_derivatives(c_off);
    compute_heom_derivatives(c_on);
    CHK(cudaDeviceSynchronize());

    std::vector<complex_t> off(stride), on(stride);
    CHK(cudaMemcpy(off.data(), d_off.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(on.data(), d_on.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));

    bool ok = true;
    int touched_ados = 0;
    for (int a = 0; a < M; ++a) {
        bool this_ado_touched = false;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const size_t k = static_cast<size_t>(a) * N * N + i * N + j;
                const double factor = (i == sink_site && j == sink_site) ? gamma_sink
                                    : (i == sink_site || j == sink_site) ? 0.5 * gamma_sink
                                    : 0.0;
                const double want_re = -factor * c_re(h_rho[k]);
                const double want_im = -factor * c_im(h_rho[k]);
                const double got_re  = c_re(on[k]) - c_re(off[k]);
                const double got_im  = c_im(on[k]) - c_im(off[k]);

                // Untouched elements must match bit-for-bit: the same arithmetic
                // ran on both sides, so anything else means Term 6 leaked.
                const double tol = (factor == 0.0)
                                 ? 0.0
                                 : (heom::kMixedPrecision ? 1e-4 : 1e-12);
                if (std::abs(got_re - want_re) > tol ||
                    std::abs(got_im - want_im) > tol) {
                    std::cout << "  sink term a=" << a << " (" << i << "," << j
                              << "): got " << fmt_c(got_re, got_im)
                              << " want " << fmt_c(want_re, want_im) << "\n";
                    ok = false;
                }
                if (factor != 0.0 && (want_re != 0.0 || want_im != 0.0))
                    this_ado_touched = true;
            }
        if (this_ado_touched) ++touched_ados;
    }

    // (b) the whole hierarchy, not just rho_0.
    if (touched_ados != M) {
        std::cout << "  sink reached " << touched_ados << " of " << M
                  << " ADOs; it must act on the whole hierarchy\n";
        ok = false;
    }

    if (ok) std::cout << "sink term (Term 6 == -gamma/2*(d_is+d_js)*rho, all "
                      << M << " ADOs): PASS\n";
    else    std::cout << "sink term: FAIL\n";
    return ok;
}

// Manifold blocking: a blocked launch must reproduce the full launch EXACTLY
// on the elements it covers, and must not touch the ones it does not.
//
// Bit-exactness, not a tolerance. The blocked launch drops k-sum terms the full
// launch computed, and the whole design rests on those terms being exactly zero
// — so anything but equality means either ρ was not really confined to the
// block or the surviving terms got reordered. A tolerance would hide both.
// Equality is spelled with ==, which admits the one documented deviation:
// a result that lands on -0.0 in one path can be +0.0 in the other, since the
// skipped iterations used to add an exact +0.0.
//
// Four blocks. The Term 1 dispatch has three branches which differ only in the
// ORDER the two half-sums are accumulated, and each needs a case where that
// order is observable:
//   (f,e) 3,1×1,2 : row_off > col_off  → B-loop first
//   (e,f) 1,2×3,1 : row_off < col_off  → A-loop first
//   (e,e) 1,2×1,2 : diagonal           → interleaved, as the original
//   (g,e) 0,1×1,2 : row_off < col_off, and the shape SE/GSB actually run in
//
// (g,e) does NOT cover the row<col branch for ordering purposes, which is why
// (e,f) is here as well: the A-half of a (g,e) block is scaled by H[0,0], the
// ground-state energy, which is zero — the term vanishes identically and moving
// it makes no difference. Only a block whose BOTH manifolds carry a nonzero
// diagonal H block (e and f do; g does not) can detect a swapped loop.
//
// MUTATION CHECK (must be re-run by hand whenever the dispatch is touched):
// swap the two loops in EITHER disjoint branch of heom_derivatives.cu; this
// test must fail. If it passes, the ρ below is no longer order-sensitive and
// the magnitudes need re-tuning — the test is then worthless as written.
bool test_manifold_block_matches_full_launch() {
    auto cfg = HEOMConfig::Config::load_yaml(write_double_excited_dimer_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int M = sys.M(), N = sys.N();
    const size_t stride = static_cast<size_t>(M) * N * N;

    bool ok = true;
    if (N != 4) {
        std::cout << "  expected N=4 for the doubly-excited dimer, got " << N << "\n";
        return false;
    }

    struct Case { const char* name; int ro, nr, co, nc; };
    const Case cases[] = {
        {"(f,e) row_off>col_off", 3, 1, 1, 2},
        {"(e,f) row_off<col_off", 1, 2, 3, 1},
        {"(e,e) diagonal",        1, 2, 1, 2},
        {"(g,e) SE/GSB shape",    0, 1, 1, 2},
    };

    for (const Case& cs : cases) {
        // ρ is EXACTLY zero outside the block — the premise the kernel is
        // allowed to assume. Inside, magnitudes span ~1e-6 to ~1e6 so that the
        // half-sums cannot be reordered without changing the low bits: two
        // small terms that each vanish against a large one still shift it when
        // they are added together first.
        std::vector<complex_t> h_rho(stride, c_make(0.0, 0.0));
        for (int a = 0; a < M; ++a)
            for (int di = 0; di < cs.nr; ++di)
                for (int dj = 0; dj < cs.nc; ++dj) {
                    const int i = cs.ro + di, j = cs.co + dj;
                    const double mag = std::pow(10.0, 6.0 - 4.0 * ((a + di + dj) % 4));
                    h_rho[static_cast<size_t>(a) * N * N + i * N + j] =
                        c_make(mag * (1.0 + 0.3 * di), mag * (-0.7 + 0.11 * dj));
                }

        DevBuf<complex_t> d_rho(stride), d_ref(stride), d_blk(stride);
        CHK(cudaMemcpy(d_rho.ptr, h_rho.data(), stride * sizeof(complex_t),
                       cudaMemcpyHostToDevice));
        CHK(cudaMemset(d_ref.ptr, 0, stride * sizeof(complex_t)));
        CHK(cudaMemset(d_blk.ptr, 0, stride * sizeof(complex_t)));

        HEOMContext full = build_multi_ado_ctx(sys);
        full.rho_in = d_rho.ptr; full.drho_out = d_ref.ptr;
        HEOMContext blocked = build_multi_ado_ctx(sys);
        blocked.rho_in = d_rho.ptr; blocked.drho_out = d_blk.ptr;
        blocked.row_off = cs.ro; blocked.n_row = cs.nr;
        blocked.col_off = cs.co; blocked.n_col = cs.nc;

        compute_heom_derivatives(full);
        compute_heom_derivatives(blocked);
        CHK(cudaDeviceSynchronize());

        std::vector<complex_t> ref(stride), blk(stride);
        CHK(cudaMemcpy(ref.data(), d_ref.ptr, stride * sizeof(complex_t),
                       cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(blk.data(), d_blk.ptr, stride * sizeof(complex_t),
                       cudaMemcpyDeviceToHost));

        int n_in_bad = 0, n_out_written = 0, n_out_premise = 0;
        for (int a = 0; a < M; ++a)
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    const size_t e = static_cast<size_t>(a) * N * N + i * N + j;
                    const bool inside = (i >= cs.ro && i < cs.ro + cs.nr &&
                                         j >= cs.co && j < cs.co + cs.nc);
                    if (inside) {
                        if (!(c_re(blk[e]) == c_re(ref[e]) &&
                              c_im(blk[e]) == c_im(ref[e]))) {
                            if (n_in_bad < 3)
                                std::cout << "    in-block mismatch a=" << a
                                          << " (" << i << "," << j << "): blocked ("
                                          << c_re(blk[e]) << "," << c_im(blk[e])
                                          << ") full (" << c_re(ref[e]) << ","
                                          << c_im(ref[e]) << ")\n";
                            ++n_in_bad;
                        }
                    } else {
                        // The blocked launch must not have written here at all.
                        if (c_re(blk[e]) != 0.0 || c_im(blk[e]) != 0.0) ++n_out_written;
                        // ...and the full launch must agree that the derivative
                        // out there is zero, or ρ would leak out of the block on
                        // the next step and the whole premise is void. This holds
                        // because H_S is manifold-block-diagonal; if it ever stops
                        // being so, this is where we find out.
                        if (c_re(ref[e]) != 0.0 || c_im(ref[e]) != 0.0) ++n_out_premise;
                    }
                }

        if (n_in_bad || n_out_written || n_out_premise) {
            std::cout << "  " << cs.name << ": in-block mismatches=" << n_in_bad
                      << " out-of-block written=" << n_out_written
                      << " out-of-block nonzero derivative=" << n_out_premise << "\n";
            ok = false;
        }
    }

    if (ok) std::cout << "manifold block == full launch (4 blocks, bitwise): PASS\n";
    else    std::cout << "manifold block == full launch: FAIL\n";
    return ok;
}

// Manifold blocking, the part that is an ARGUMENT rather than an equality.
//
// A blocked rk4_step writes its stage buffers only inside the block, so from
// stage 2 onwards temp_a/temp_b hold whatever was in the workspace outside it.
// The design claims that cannot matter, because the only cross-element reads in
// the kernel are Term 1's two half-sums and both stay inside the block. That is
// a claim about every read in a 500-line kernel, and reviewing it is not the
// same as testing it — so: fill the workspace with NaN and see.
//
// NaN is the right poison. Zero would be absorbed by any multiply and prove
// nothing; NaN propagates through every arithmetic path, so a single stray read
// anywhere in four stages lands in the result. DevBuf hands back raw cudaMalloc
// memory, which is neither zeroed nor poisoned, hence both fills are explicit.
//
// MUTATION CHECK (re-run by hand when the dispatch changes): restore Term 1's
// original `for (k = 0; k < N; ++k)` loop; the NaN run must then produce NaN
// and this test must fail.
bool test_manifold_block_ignores_stale_stage_buffer() {
    auto cfg = HEOMConfig::Config::load_yaml(write_double_excited_dimer_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int M = sys.M(), N = sys.N();
    const size_t stride = static_cast<size_t>(M) * N * N;
    const double dt = 0.5;

    // (f,e): the block the ESA t-segment actually runs in.
    const int ro = 3, nr = 1, co = 1, nc = 2;

    std::vector<complex_t> h_rho(stride, c_make(0.0, 0.0));
    for (int a = 0; a < M; ++a)
        for (int dj = 0; dj < nc; ++dj)
            h_rho[static_cast<size_t>(a) * N * N + ro * N + (co + dj)] =
                c_make(0.4 - 0.05 * a + 0.2 * dj, 0.1 * a - 0.3 * dj);

    const size_t ws_len = 3 * stride;
    std::vector<complex_t> h_nan(ws_len,
        c_make(std::numeric_limits<double>::quiet_NaN(),
               std::numeric_limits<double>::quiet_NaN()));

    DevBuf<complex_t> d_rho_clean(stride), d_rho_nan(stride);
    DevBuf<complex_t> d_ws_clean(ws_len), d_ws_nan(ws_len);
    CHK(cudaMemcpy(d_rho_clean.ptr, h_rho.data(), stride * sizeof(complex_t),
                   cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_rho_nan.ptr, h_rho.data(), stride * sizeof(complex_t),
                   cudaMemcpyHostToDevice));
    CHK(cudaMemset(d_ws_clean.ptr, 0, ws_len * sizeof(complex_t)));
    CHK(cudaMemcpy(d_ws_nan.ptr, h_nan.data(), ws_len * sizeof(complex_t),
                   cudaMemcpyHostToDevice));

    HEOMContext base = build_multi_ado_ctx(sys);
    base.drho_out = nullptr;   // fused path never writes a k buffer
    base.row_off = ro; base.n_row = nr; base.col_off = co; base.n_col = nc;

    HEOMContext c_clean = base; c_clean.rho_in = d_rho_clean.ptr;
    HEOMContext c_nan   = base; c_nan.rho_in   = d_rho_nan.ptr;
    rk4_step(c_clean, dt, d_ws_clean.ptr, nullptr);
    rk4_step(c_nan,   dt, d_ws_nan.ptr,   nullptr);
    CHK(cudaDeviceSynchronize());

    std::vector<complex_t> clean(stride), nan_run(stride);
    CHK(cudaMemcpy(clean.data(), d_rho_clean.ptr, stride * sizeof(complex_t),
                   cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(nan_run.data(), d_rho_nan.ptr, stride * sizeof(complex_t),
                   cudaMemcpyDeviceToHost));

    bool ok = true;
    int n_nan = 0, n_diff = 0;
    for (int a = 0; a < M; ++a)
        for (int dj = 0; dj < nc; ++dj) {
            const size_t e = static_cast<size_t>(a) * N * N + ro * N + (co + dj);
            if (std::isnan(c_re(nan_run[e])) || std::isnan(c_im(nan_run[e]))) ++n_nan;
            if (!(c_re(nan_run[e]) == c_re(clean[e]) &&
                  c_im(nan_run[e]) == c_im(clean[e]))) ++n_diff;
        }
    if (n_nan || n_diff) {
        std::cout << "  poisoned workspace leaked: in-block NaN=" << n_nan
                  << " mismatches vs clean workspace=" << n_diff << "\n";
        ok = false;
    }

    // Guard against the test passing because nothing happened at all: the step
    // has to have moved ρ somewhere inside the block.
    bool moved = false;
    for (int a = 0; a < M && !moved; ++a)
        for (int dj = 0; dj < nc && !moved; ++dj) {
            const size_t e = static_cast<size_t>(a) * N * N + ro * N + (co + dj);
            if (c_re(clean[e]) != c_re(h_rho[e]) || c_im(clean[e]) != c_im(h_rho[e]))
                moved = true;
        }
    if (!moved) {
        std::cout << "  rk4_step left rho untouched — test proves nothing\n";
        ok = false;
    }

    if (ok) std::cout << "manifold block ignores NaN-poisoned stage buffers: PASS\n";
    else    std::cout << "manifold block ignores NaN-poisoned stage buffers: FAIL\n";
    return ok;
}

// Batched RK4 workspace. A full rk4_step with B=2 identical replicas
// must land, in EVERY slot, on the B=1 result. rk4_step ping-pongs temp_a/temp_b
// and accumulates in acc — all three sub-buffers span batch·M·N², and the final
// combine writes ρ in place. A batch stride bug (int overflow, or sub-buffers
// spaced by M·N² instead of batch·M·N²) would cross-contaminate the slots.
bool test_batch_rk4_symmetry() {
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int M = sys.M(), N = sys.N();
    const size_t stride = static_cast<size_t>(M) * N * N;
    const double dt = 0.5;

    std::vector<complex_t> h_rho(stride);
    for (int a = 0; a < M; ++a)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const double re = 0.09 * (a + 1) + 0.21 * i - 0.13 * j;
                const double im = 0.04 * (a + 1) - 0.08 * i + 0.11 * j;
                h_rho[static_cast<size_t>(a) * N * N + i * N + j] = c_make(re, im);
            }

    // B=1 reference.
    DevBuf<complex_t> d_rho1(stride), d_ws1(3 * stride);
    CHK(cudaMemcpy(d_rho1.ptr, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
    HEOMContext c1 = build_multi_ado_ctx(sys);
    c1.rho_in = d_rho1.ptr; c1.batch = 1;
    rk4_step(c1, dt, d_ws1.ptr, nullptr);

    // B=2 over identical replicas.
    DevBuf<complex_t> d_rho2(2 * stride), d_ws2(3 * 2 * stride);
    CHK(cudaMemcpy(d_rho2.ptr, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_rho2.ptr + stride, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
    HEOMContext c2 = build_multi_ado_ctx(sys);
    c2.rho_in = d_rho2.ptr; c2.batch = 2;
    rk4_step(c2, dt, d_ws2.ptr, nullptr);
    CHK(cudaDeviceSynchronize());

    std::vector<complex_t> ref(stride), s0(stride), s1(stride);
    CHK(cudaMemcpy(ref.data(), d_rho1.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(s0.data(), d_rho2.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(s1.data(), d_rho2.ptr + stride, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t k = 0; k < stride; ++k) {
        for (int slot = 0; slot < 2; ++slot) {
            const complex_t g = slot ? s1[k] : s0[k];
            if (std::abs(c_re(g) - c_re(ref[k])) > kKernelTol ||
                std::abs(c_im(g) - c_im(ref[k])) > kKernelTol) {
                std::cout << "  rk4 batch symmetry slot=" << slot << " k=" << k
                          << ": got (" << c_re(g) << "," << c_im(g) << ") ref ("
                          << c_re(ref[k]) << "," << c_im(ref[k]) << ")\n";
                ok = false;
            }
        }
    }

    if (ok) std::cout << "batch RK4 symmetry (B=2 step == B=1 per slot): PASS\n";
    else    std::cout << "batch RK4 symmetry: FAIL\n";
    return ok;
}

// Union norm. norms[α] must equal max over batch slots of ‖ρ_α^(b)‖_max.
// Per α we plant one element per slot with a known magnitude, arranged so slot 0
// wins for small α and slot 1 for large α — so a bug that reads only slot 0 (or
// only the last slot) misflags roughly half the ADOs. Both N=3 (per-thread) and
// N=10 (warp) kernel paths are covered.
bool test_batch_union_norms() {
    bool all_ok = true;
    // Precision-scaled (mixed stores ρ as float); still << the ~0.1 crossover
    // gap, so a slot-selection bug is caught either way.
    const double kTol = heom::kMixedPrecision ? 1e-4 : 1e-12;

    for (const int N : {3, 10}) {   // 9 < 64 <= 100: both norm kernels
        const int M = 10, B = 2;
        const size_t NN = static_cast<size_t>(N) * N;
        const size_t total = static_cast<size_t>(B) * M * NN;

        std::vector<complex_t> h(total, c_make(0.0, 0.0));
        auto mag = [](int b, int a) {
            return (b == 0) ? (0.5 + 0.10 * a)    // slot 0 wins for a < 5
                            : (0.4 + 0.12 * a);   // slot 1 wins for a >= 5
        };
        for (int b = 0; b < B; ++b)
            for (int a = 0; a < M; ++a)
                h[static_cast<size_t>(b) * M * NN + static_cast<size_t>(a) * NN + (a % NN)] =
                    c_make(mag(b, a), 0.0);

        DevBuf<complex_t> d_rho(total);
        DevBuf<double>    d_norms(M);
        CHK(cudaMemcpy(d_rho.ptr, h.data(), total * sizeof(complex_t), cudaMemcpyHostToDevice));

        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N, 0, N, 0, N,
                          nullptr, 0, B);
        CHK(cudaDeviceSynchronize());

        std::vector<double> got(M);
        CHK(cudaMemcpy(got.data(), d_norms.ptr, M * sizeof(double), cudaMemcpyDeviceToHost));

        for (int a = 0; a < M; ++a) {
            const double expect = std::max(mag(0, a), mag(1, a));
            if (std::abs(got[a] - expect) > kTol) {
                std::cout << "  union norm N=" << N << " a=" << a
                          << ": got " << got[a] << " expect " << expect << "\n";
                all_ok = false;
            }
        }
    }

    if (all_ok) std::cout << "batch union norms (max over slots, both kernels): PASS\n";
    else        std::cout << "batch union norms: FAIL\n";
    return all_ok;
}

// Batch-aware alloc plan. ONLY d_rho and d_workspace scale by batch; all
// read-only structure (maps, coeffs, H, filter flags) is shared across slots and
// must stay ×1. Pure host function — no GPU needed. Locks the invariant so a
// future edit that accidentally batches a shared buffer (or forgets to batch a
// mutable one) fails loud.
bool test_batch_alloc_plan() {
    const size_t M = 100;
    const int N = 4, K = 1, nb = 4, L = 3;
    const auto p1 = heom::plan_gpu_buffers(M, N, K, nb, L, /*use_filtering=*/true, 1);
    const auto p4 = heom::plan_gpu_buffers(M, N, K, nb, L, /*use_filtering=*/true, 4);

    auto count = [](const std::vector<heom::GPUBufferSpec>& p,
                    const char* nm) -> size_t {
        for (const auto& s : p)
            if (std::string(s.name) == nm) return s.count;
        return SIZE_MAX;   // not found
    };

    bool ok = true;
    auto expect = [&](const char* nm, size_t got, size_t want) {
        if (got != want) {
            std::cout << "  alloc plan " << nm << ": got " << got
                      << " want " << want << "\n";
            ok = false;
        }
    };

    // Mutable big arrays scale ×4.
    expect("d_rho", count(p4, "d_rho"), 4 * count(p1, "d_rho"));
    expect("d_workspace", count(p4, "d_workspace"), 4 * count(p1, "d_workspace"));

    // Everything else is a single shared copy — invariant under batch.
    for (const char* nm : {"d_hamiltonian", "d_c_bath", "d_markov_residue",
                           "d_plus_map", "d_minus_map", "d_level_indices",
                           "d_coeff_plus", "d_coeff_minus", "d_total_decay",
                           "d_d_coupling", "d_ado_norms", "d_active_flags",
                           "d_level_counts"}) {
        expect(nm, count(p4, nm), count(p1, nm));
    }

    if (ok) std::cout << "batch alloc plan (only d_rho/d_workspace scale): PASS\n";
    else    std::cout << "batch alloc plan: FAIL\n";
    return ok;
}

// Task A4: variable active batch width on HEOMTrajectory. Two properties:
//  (A) active=1 on a batch=4 trajectory == a batch=1 trajectory, bit-for-bit
//      (reduced-width launch + set_initial_batch at width 1 are physics-correct).
//  (B) reducing active 4→2 FREEZES slots 2/3: after propagating at active=4,
//      a second propagate at active=2 must leave slots 2/3 untouched. This is
//      the idempotency-trap catcher — if set_active_batch failed to update
//      ctx_.batch, the launch would stay grid.y=4 and advance slots 2/3 (they
//      hold distinct nonzero state), so the freeze check would fail.
bool test_active_batch_width() {
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int N = sys.N();
    const int nsteps = 4;

    auto make_rho = [&](double seed) {
        Eigen::MatrixXcd r(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r(i, j) = std::complex<double>(0.1 * (i + 1) + 0.2 * j + seed,
                                               0.05 * i - 0.1 * j + 0.3 * seed);
        return r;
    };
    auto max_diff = [](const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
        return (a - b).cwiseAbs().maxCoeff();
    };

    // B=1 reference: nsteps from rho0(seed=0).
    Eigen::MatrixXcd ref0;
    {
        HEOMTrajectory t1(sys, /*enable_hdf5=*/false, 1);
        t1.set_initial(make_rho(0.0));
        t1.propagate_steps(nsteps, /*check_div=*/false);
        ref0 = t1.slot_rho0(0);
    }

    // (A) active=1 on a batch=4 trajectory must match the B=1 reference. Same
    // kernel arithmetic + order ⇒ bit-exact (FP64 and mixed); tol only guards
    // against a genuine precision surprise.
    const double tolA = heom::kMixedPrecision ? kCarryTol : 0.0;
    bool okA = true;
    {
        HEOMTrajectory t4(sys, /*enable_hdf5=*/false, 4);
        t4.set_active_batch(1);
        t4.set_initial_batch({make_rho(0.0)});
        t4.propagate_steps(nsteps, /*check_div=*/false);
        const double d = max_diff(t4.slot_rho0(0), ref0);
        if (d > tolA) {
            std::cout << "  active=1 vs B=1 ref: max|delta|=" << d << "\n";
            okA = false;
        }
    }

    // (B) freeze check: slots 2/3 must not move when active drops 4→2.
    bool okB = true;
    {
        HEOMTrajectory t4(sys, /*enable_hdf5=*/false, 4);
        t4.set_active_batch(4);
        t4.set_initial_batch({make_rho(0.0), make_rho(1.0),
                              make_rho(2.0), make_rho(3.0)});
        t4.propagate_steps(nsteps, /*check_div=*/false);  // ctx built at B=4
        const Eigen::MatrixXcd s2b = t4.slot_rho0(2), s3b = t4.slot_rho0(3);
        t4.set_active_batch(2);
        t4.propagate_steps(nsteps, /*check_div=*/false);  // must touch slots 0,1 only
        const double d2 = max_diff(t4.slot_rho0(2), s2b);
        const double d3 = max_diff(t4.slot_rho0(3), s3b);
        if (d2 != 0.0 || d3 != 0.0) {
            std::cout << "  active 4->2 leaked into frozen slots: |d slot2|=" << d2
                      << " |d slot3|=" << d3 << " (idempotency trap?)\n";
            okB = false;
        }
    }

    if (okA && okB) std::cout << "active batch width (A: active=1==B=1, B: 4->2 freeze): PASS\n";
    else            std::cout << "active batch width: FAIL\n";
    return okA && okB;
}

// Task A5: batched snapshot/restore + expand_slots. Filter off ⇒ n_active=M,
// full-block copy, so every check is bit-exact.
//  (a) snapshot at active=3 (batch=4 traj) → perturb → change width → restore:
//      must restore all 3 slots AND active_batch_ back to 3 (width-restore
//      teeth).
//  (b) expand_slots uneven fanout 3→6 with map {0,0,1,1,1,2}: child c's α=0 must
//      equal parent slot map[c]'s α=0 (general/uneven child→parent, shared ids).
bool test_batch_snapshot_expand() {
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int N = sys.N();
    const int nsteps = 3;

    auto make_rho = [&](double seed) {
        Eigen::MatrixXcd r(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r(i, j) = std::complex<double>(0.1 * (i + 1) + 0.2 * j + seed,
                                               0.05 * i - 0.1 * j + 0.3 * seed);
        return r;
    };
    auto max_diff = [](const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
        return (a - b).cwiseAbs().maxCoeff();
    };

    // (a) snapshot(3) → perturb → set_active_batch(1) → restore → back to 3 slots.
    bool okA = true;
    {
        HEOMTrajectory t(sys, /*enable_hdf5=*/false, 4);
        t.set_active_batch(3);
        t.set_initial_batch({make_rho(0.0), make_rho(1.0), make_rho(2.0)});
        DeviceHierarchySnapshot snap;
        t.snapshot_device(snap);
        if (snap.batch != 3) {
            std::cout << "  snapshot.batch=" << snap.batch << " expected 3\n";
            okA = false;
        }
        // Capture the snapshot-time slot state (already storage-precision rounded)
        // so the restore round-trip is a pure copy ⇒ bit-exact in FP64 AND mixed.
        Eigen::MatrixXcd ref[3];
        for (int b = 0; b < 3; ++b) ref[b] = t.slot_rho0(b);
        t.propagate_steps(nsteps, /*check_div=*/false);   // perturb slots 0-2
        t.set_active_batch(1);                             // change width
        t.restore_device(snap);
        if (t.active_batch() != 3) {
            std::cout << "  restore did not reset active_batch_ (got "
                      << t.active_batch() << ", expected 3)\n";
            okA = false;
        }
        for (int b = 0; b < 3; ++b) {
            const double d = max_diff(t.slot_rho0(b), ref[b]);
            if (d != 0.0) {
                std::cout << "  restore slot " << b << " max|delta|=" << d << "\n";
                okA = false;
            }
        }
    }

    // (b) expand_slots uneven 3→6.
    bool okB = true;
    {
        HEOMTrajectory t(sys, /*enable_hdf5=*/false, 6);
        t.set_active_batch(3);
        t.set_initial_batch({make_rho(0.0), make_rho(1.0), make_rho(2.0)});
        // Storage-precision reference for each parent slot (see note in (a)).
        DeviceHierarchySnapshot parent;
        t.snapshot_device(parent);
        Eigen::MatrixXcd pref[3];
        for (int p = 0; p < 3; ++p) pref[p] = t.slot_rho0(p);
        const std::vector<int> map = {0, 0, 1, 1, 1, 2};
        t.expand_slots(parent, map);
        if (t.active_batch() != 6) {
            std::cout << "  expand did not set active_batch_=6 (got "
                      << t.active_batch() << ")\n";
            okB = false;
        }
        for (int c = 0; c < 6; ++c) {
            const double d = max_diff(t.slot_rho0(c), pref[map[c]]);
            if (d != 0.0) {
                std::cout << "  expand child " << c << " (parent " << map[c]
                          << ") max|delta|=" << d << "\n";
                okB = false;
            }
        }
    }

    if (okA && okB) std::cout << "batch snapshot/restore + expand_slots (uneven 3->6): PASS\n";
    else            std::cout << "batch snapshot/restore + expand_slots: FAIL\n";
    return okA && okB;
}

// Task A6: batched detection. B_det=3 with per-slot (ρ, μ_det) must equal three
// independent B=1 trajectories running the same sample/propagate sequence. Slots
// carry distinct ρ and μ_det, so a trace kernel that dropped a per-slot offset
// (μ or ρ) would collide slots and mismatch. Independent slots ⇒ bit-exact vs
// single in FP64 and mixed.
bool test_batch_detect() {
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int N = sys.N();
    const int ns = 3;

    auto make_rho = [&](double seed) {
        Eigen::MatrixXcd r(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r(i, j) = std::complex<double>(0.1 * (i + 1) + 0.2 * j + seed,
                                               0.05 * i - 0.1 * j + 0.3 * seed);
        return r;
    };
    auto make_mu = [&](double seed) {
        Eigen::MatrixXcd m(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m(i, j) = std::complex<double>(0.3 * i - 0.15 * j + 0.2 * seed,
                                               0.11 * (j + 1) - 0.07 * i + seed);
        return m;
    };

    const Eigen::MatrixXcd rhos[3] = {make_rho(0.0), make_rho(1.0), make_rho(2.0)};
    const Eigen::MatrixXcd mus[3]  = {make_mu(0.0),  make_mu(1.0),  make_mu(2.0)};

    // Reference: three independent B=1 trajectories.
    std::vector<std::complex<double>> ref(3 * ns);
    for (int b = 0; b < 3; ++b) {
        HEOMTrajectory t1(sys, /*enable_hdf5=*/false, 1);
        t1.set_initial(rhos[b]);
        t1.detect_begin(mus[b], ns);
        t1.detect_sample(0);
        t1.propagate_steps(1, /*check_div=*/false);
        t1.detect_sample(1);
        t1.propagate_steps(1, /*check_div=*/false);
        t1.detect_sample(2);
        t1.detect_flush(&ref[b * ns], ns);
    }

    // Batched: one B=3 trajectory, same sequence.
    std::vector<std::complex<double>> out(3 * ns);
    {
        HEOMTrajectory t3(sys, /*enable_hdf5=*/false, 3);
        t3.set_active_batch(3);
        t3.set_initial_batch({rhos[0], rhos[1], rhos[2]});
        t3.detect_begin_batch({mus[0], mus[1], mus[2]}, ns);
        t3.detect_sample_batch(0);
        t3.propagate_steps(1, /*check_div=*/false);
        t3.detect_sample_batch(1);
        t3.propagate_steps(1, /*check_div=*/false);
        t3.detect_sample_batch(2);
        t3.detect_flush_batch(out.data(), ns);
        if (t3.detect_batch() != 3) {
            std::cout << "  detect_batch()=" << t3.detect_batch() << " expected 3\n";
            return false;
        }
    }

    const double tol = heom::kMixedPrecision ? kCarryTol : 0.0;
    bool ok = true;
    for (int k = 0; k < 3 * ns; ++k) {
        const double d = std::abs(out[k] - ref[k]);
        if (d > tol) {
            std::cout << "  detect batch slot=" << (k / ns) << " idx=" << (k % ns)
                      << " |delta|=" << d << "\n";
            ok = false;
        }
    }

    if (ok) std::cout << "batch detection (B_det=3 == 3x B=1, per-slot mu/rho): PASS\n";
    else    std::cout << "batch detection: FAIL\n";
    return ok;
}

// Task A7: per-slot dipole. apply_dipole_per_slot with distinct μ per slot must
// equal three independent B=1 apply_dipole_full_hierarchy calls. Covers both
// sides (left μ·ρ / right ρ·μ). Independent slots ⇒ bit-exact vs single.
bool test_batch_dipole_per_slot() {
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int N = sys.N();

    auto make_rho = [&](double seed) {
        Eigen::MatrixXcd r(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r(i, j) = std::complex<double>(0.1 * (i + 1) + 0.2 * j + seed,
                                               0.05 * i - 0.1 * j + 0.3 * seed);
        return r;
    };
    auto make_mu = [&](double seed) {
        Eigen::MatrixXcd m(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m(i, j) = std::complex<double>(0.3 * i - 0.15 * j + 0.2 * seed,
                                               0.11 * (j + 1) - 0.07 * i + seed);
        return m;
    };
    auto max_diff = [](const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
        return (a - b).cwiseAbs().maxCoeff();
    };

    const Eigen::MatrixXcd rhos[3] = {make_rho(0.0), make_rho(1.0), make_rho(2.0)};
    const Eigen::MatrixXcd mus[3]  = {make_mu(0.0),  make_mu(1.0),  make_mu(2.0)};
    const double tol = heom::kMixedPrecision ? kCarryTol : 0.0;

    bool ok = true;
    for (const auto side : {heom::spectra::DipoleSide::Left,
                            heom::spectra::DipoleSide::Right}) {
        Eigen::MatrixXcd ref[3];
        for (int b = 0; b < 3; ++b) {
            HEOMTrajectory t1(sys, /*enable_hdf5=*/false, 1);
            t1.set_initial(rhos[b]);
            t1.apply_dipole_full_hierarchy(mus[b], side);
            ref[b] = t1.slot_rho0(0);
        }
        HEOMTrajectory t3(sys, /*enable_hdf5=*/false, 3);
        t3.set_active_batch(3);
        t3.set_initial_batch({rhos[0], rhos[1], rhos[2]});
        t3.apply_dipole_per_slot({mus[0], mus[1], mus[2]}, side);
        for (int b = 0; b < 3; ++b) {
            const double d = max_diff(t3.slot_rho0(b), ref[b]);
            if (d > tol) {
                std::cout << "  dipole per-slot side="
                          << (side == heom::spectra::DipoleSide::Left ? "L" : "R")
                          << " slot " << b << " max|delta|=" << d << "\n";
                ok = false;
            }
        }
    }

    if (ok) std::cout << "batch per-slot dipole (== 3x B=1, both sides): PASS\n";
    else    std::cout << "batch per-slot dipole: FAIL\n";
    return ok;
}

// slot_active_stats().per_slot must size to the live width active_batch_,
// not the allocated batch_. A batch=4 trajectory run at active=3 must report
// exactly 3 slots — reading slot 3 (stale/idle) would poison the union-
// inflation measurement. Also checks the union invariant union_count ≥ max_b.
bool test_per_slot_active_counts() {
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    // per_slot_active_counts() lives on the filtering path (d_ado_norms_ is only
    // allocated when filtering is on, matching its sole production caller
    // emit_ado_stats). Enable filtering so the helper has its scratch buffer.
    cfg.filter.enabled   = true;
    cfg.filter.tolerance = 1e-8;
    SystemContext sys(cfg);
    const int N = sys.N();
    const int M = sys.M();

    auto make_rho = [&](double seed) {
        Eigen::MatrixXcd r(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r(i, j) = std::complex<double>(0.1 * (i + 1) + 0.2 * j + seed,
                                               0.05 * i - 0.1 * j + 0.3 * seed);
        return r;
    };

    bool ok = true;
    HEOMTrajectory t4(sys, /*enable_hdf5=*/false, 4);  // allocated batch_=4
    t4.set_active_batch(3);
    t4.set_initial_batch({make_rho(0.0), make_rho(1.0), make_rho(2.0)});

    // Right after init only α=0 is populated per slot ⇒ size==3, each count==1,
    // union==1. (size==4 would mean the helper walked the stale 4th slot: T0a bug.)
    const auto s0 = t4.slot_active_stats();
    if (static_cast<int>(s0.per_slot.size()) != 3) {
        std::cout << "  per_slot size=" << s0.per_slot.size()
                  << " (expected active_batch_=3, not batch_=4)\n";
        ok = false;
    }
    for (size_t b = 0; b < s0.per_slot.size(); ++b)
        if (s0.per_slot[b] != 1) {
            std::cout << "  slot " << b << " count=" << s0.per_slot[b]
                      << " (expected 1 at init: ADOs zero)\n";
            ok = false;
        }
    if (s0.union_count != 1) {
        std::cout << "  union_count=" << s0.union_count << " (expected 1 at init)\n";
        ok = false;
    }

    // Propagate a few steps: ADOs fill. Sizes stay 3, all in [1, M], and the
    // core invariant holds: union ≥ every per-slot count (union is the OR).
    t4.propagate_steps(4, /*check_div=*/false);
    const auto s1 = t4.slot_active_stats();
    if (static_cast<int>(s1.per_slot.size()) != 3) {
        std::cout << "  post-prop size=" << s1.per_slot.size() << " (expected 3)\n";
        ok = false;
    }
    for (int cnt : s1.per_slot) {
        if (cnt < 1 || cnt > M) {
            std::cout << "  post-prop count=" << cnt << " out of [1," << M << "]\n";
            ok = false;
        }
        if (s1.union_count < cnt) {
            std::cout << "  union_count=" << s1.union_count
                      << " < per-slot " << cnt << " (union must dominate)\n";
            ok = false;
        }
    }
    if (s1.union_count > M) {
        std::cout << "  union_count=" << s1.union_count << " > M=" << M << "\n";
        ok = false;
    }

    // Width change tracks live width: active=1 ⇒ exactly one slot reported.
    t4.set_active_batch(1);
    if (static_cast<int>(t4.slot_active_stats().per_slot.size()) != 1) {
        std::cout << "  active=1 per_slot size != 1\n";
        ok = false;
    }

    if (ok) std::cout << "slot_active_stats (size==active_batch_, union≥max): PASS\n";
    else    std::cout << "slot_active_stats: FAIL\n";
    return ok;
}

// Carry-through-style dimer config with a chosen L_cutoff, for the IF-RK4 test:
// L=2 gives deep ADOs with g_α>0 (a real IF workout); L=0 gives M=1/g≡0 (the
// bit-identical wiring case). Identical to write_carry_through_yaml() otherwise.
std::string write_if_rk4_yaml(int L_cutoff) {
    const std::string path = "/tmp/test_if_rk4_L" + std::to_string(L_cutoff) + ".yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  basis: \"excited_only\"\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100.0, 100.0]\n"
      << "    - [100.0, 12000.0]\n"
      << "  w0: 12000.0\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 35.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: matsubara\n"
      << "  n_extra_poles: 0\n"
      << "  L_cutoff: " << L_cutoff << "\n"
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
      << "    save_path: \"/tmp/\"\n"
      << "    filename: \"if_rk4\"\n"
      << "    use_timestamp: false\n";
    return path;
}

// T6: IF-RK4 must reproduce plain RK4. Both integrate the SAME equation; their
// only difference is HOW the (here small) step is taken, so at small dt they
// converge to the same trajectory with an O(dt⁴) gap.
//   (A) coupled L=2 dimer (M=6, deep ADOs carry g>0): 20 steps @ dt=0.5fs,
//       random full-hierarchy ρ ⇒ relative error < tol (FP64 1e-6 / mixed 1e-4).
//   (B) L=0 (M=1 ⇒ g≡0 ⇒ E=E2=1): IF must be BIT-IDENTICAL to plain (1.0·x is
//       exact). Only valid where NO ADO is rescaled — a coupled config is not.
bool test_if_rk4_matches_plain_small_dt() {
    // ---- (A) coupled: relative-error convergence ----
    {
        auto cfg = HEOMConfig::Config::load_yaml(write_if_rk4_yaml(2));
        HEOMConfig::resolve_dynamics_config(cfg);
        SystemContext sys(cfg);
        const int M = sys.M(), N = sys.N();
        const size_t stride = static_cast<size_t>(M) * N * N;
        const double dt = 0.5 * FS_TO_AU;   // 0.5 fs in a.u.

        std::vector<complex_t> h_rho(stride);
        for (int a = 0; a < M; ++a)
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    const double re = 0.11 * (a + 1) + 0.23 * i - 0.17 * j;
                    const double im = 0.05 * (a + 1) - 0.09 * i + 0.13 * j;
                    h_rho[static_cast<size_t>(a) * N * N + i * N + j] = c_make(re, im);
                }

        DevBuf<complex_t> d_rp(stride), d_wp(3 * stride);
        DevBuf<complex_t> d_ri(stride), d_wi(3 * stride);
        CHK(cudaMemcpy(d_rp.ptr, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_ri.ptr, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
        HEOMContext cp = build_multi_ado_ctx(sys); cp.rho_in = d_rp.ptr; cp.batch = 1;
        HEOMContext ci = build_multi_ado_ctx(sys); ci.rho_in = d_ri.ptr; ci.batch = 1;

        DevBuf<heom::real_t> d_eh(M), d_ef(M);
        compute_if_factors(sys.d_total_decay(), dt, M, d_eh.ptr, d_ef.ptr);

        for (int s = 0; s < 20; ++s) {
            rk4_step(cp, dt, d_wp.ptr, nullptr);
            rk4_step_if(ci, dt, d_wi.ptr, nullptr, d_eh.ptr, d_ef.ptr);
        }
        CHK(cudaDeviceSynchronize());

        std::vector<complex_t> ref(stride), got(stride);
        CHK(cudaMemcpy(ref.data(), d_rp.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(got.data(), d_ri.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));

        double max_ref = 0.0, max_diff = 0.0;
        for (size_t k = 0; k < stride; ++k) {
            const double ar = std::abs(c_re(ref[k])), ai = std::abs(c_im(ref[k]));
            if (ar > max_ref) max_ref = ar;
            if (ai > max_ref) max_ref = ai;
            const double dr = std::abs(c_re(got[k]) - c_re(ref[k]));
            const double di = std::abs(c_im(got[k]) - c_im(ref[k]));
            if (dr > max_diff) max_diff = dr;
            if (di > max_diff) max_diff = di;
        }
        const double rel = max_diff / (max_ref > 0.0 ? max_ref : 1.0);
        const double tol = heom::kMixedPrecision ? 1e-4 : 1e-6;
        if (rel > tol) {
            std::cout << "  IF-RK4 vs plain: rel err=" << rel << " > tol=" << tol << "\n";
            std::cout << "if_rk4_matches_plain_small_dt: FAIL\n";
            return false;
        }
    }

    // ---- (B) L=0: bit-identical wiring (g≡0 ⇒ E=E2=1) ----
    {
        auto cfg = HEOMConfig::Config::load_yaml(write_if_rk4_yaml(0));
        HEOMConfig::resolve_dynamics_config(cfg);
        SystemContext sys(cfg);
        const int M = sys.M(), N = sys.N();
        const size_t stride = static_cast<size_t>(M) * N * N;
        const double dt = 1.0 * FS_TO_AU;

        std::vector<complex_t> h_rho(stride);
        for (size_t k = 0; k < stride; ++k)
            h_rho[k] = c_make(0.07 * (k + 1), -0.03 * (k + 1));

        DevBuf<complex_t> d_rp(stride), d_wp(3 * stride);
        DevBuf<complex_t> d_ri(stride), d_wi(3 * stride);
        CHK(cudaMemcpy(d_rp.ptr, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_ri.ptr, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
        HEOMContext cp = build_multi_ado_ctx(sys); cp.rho_in = d_rp.ptr; cp.batch = 1;
        HEOMContext ci = build_multi_ado_ctx(sys); ci.rho_in = d_ri.ptr; ci.batch = 1;

        DevBuf<heom::real_t> d_eh(M), d_ef(M);
        compute_if_factors(sys.d_total_decay(), dt, M, d_eh.ptr, d_ef.ptr);

        rk4_step(cp, dt, d_wp.ptr, nullptr);
        rk4_step_if(ci, dt, d_wi.ptr, nullptr, d_eh.ptr, d_ef.ptr);
        CHK(cudaDeviceSynchronize());

        std::vector<complex_t> ref(stride), got(stride);
        CHK(cudaMemcpy(ref.data(), d_rp.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(got.data(), d_ri.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < stride; ++k) {
            if (c_re(got[k]) != c_re(ref[k]) || c_im(got[k]) != c_im(ref[k])) {
                std::cout << "  L=0 IF not bit-identical at k=" << k << ": got ("
                          << c_re(got[k]) << "," << c_im(got[k]) << ") ref ("
                          << c_re(ref[k]) << "," << c_im(ref[k]) << ")\n";
                std::cout << "if_rk4_matches_plain_small_dt: FAIL\n";
                return false;
            }
        }
    }

    std::cout << "if_rk4_matches_plain_small_dt (coupled rel<tol + L=0 bit-identical): PASS\n";
    return true;
}

}  // namespace

int main() {
    constexpr int M = 1;
    constexpr int N = 2;
    constexpr int K = 0;
    constexpr int n_bath = 1;

    // Hermitian, real H = [[1, 0.1], [0.1, 2]].
    const std::vector<complex_t> h_H = {
        c_make(1.0, 0.0), c_make(0.1, 0.0),
        c_make(0.1, 0.0), c_make(2.0, 0.0),
    };
    // rho = |0><0|  (hier precision).
    const std::vector<complex_t> h_rho = {
        c_make(1.0, 0.0), c_make(0.0, 0.0),
        c_make(0.0, 0.0), c_make(0.0, 0.0),
    };
    // Sole ADO (alpha = 0) has no neighbours.
    const std::vector<int>    h_plus_map  = {-1};
    const std::vector<int>    h_minus_map = {-1};
    const std::vector<int8_t> h_level     = {0};

    // Matsubara: K+1 = 1 entry, value irrelevant — Term 2 is zero because
    // total_decay = 0; Term 3 is zero because markov_residue = 0.
    const std::vector<complex_t> h_c_mats  = {c_make(0, 0)};
    // Precomputed Term 4/5 magnitudes [M*n_bath*(K+1)]. The sole ADO has no
    // neighbours (maps = -1) and α=0, so neither is read; zero them.
    const std::vector<heom::real_t>          h_coeff_plus  = {0};
    const std::vector<heom::real_t>          h_coeff_minus = {0};

    // d_coupling: identical across (i, j) so even if Term 3 fired its
    // diff term would be zero. Belt-and-braces.
    const std::vector<int8_t> h_dcoup  = {1, 1};
    const std::vector<double> h_decay  = {0.0};
    const std::vector<heom::real_t> h_markov = {0};     // [n_bath] per-site Markov residue

    DevBuf<complex_t>       d_rho   (M * N * N);
    DevBuf<complex_t>       d_drho  (M * N * N);
    DevBuf<complex_t>       d_H     (N * N);
    DevBuf<int>             d_plus  (M * n_bath * (K + 1));
    DevBuf<int>             d_minus (M * n_bath * (K + 1));
    DevBuf<complex_t>       d_c     (n_bath * (K + 1));
    DevBuf<heom::real_t>          d_coeff_plus (M * n_bath * (K + 1));
    DevBuf<heom::real_t>          d_coeff_minus(M * n_bath * (K + 1));
    DevBuf<int8_t>          d_level (M * n_bath * (K + 1));
    DevBuf<double>          d_decay (M);
    DevBuf<int8_t>          d_dcoup (n_bath * N);
    DevBuf<heom::real_t>          d_markov(n_bath);

    CHK(cudaMemcpy(d_rho.ptr,   h_rho.data(),       M*N*N*sizeof(complex_t),             cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_H.ptr,     h_H.data(),         N*N*sizeof(complex_t),         cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_plus.ptr,  h_plus_map.data(),  M*n_bath*(K+1)*sizeof(int),          cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_minus.ptr, h_minus_map.data(), M*n_bath*(K+1)*sizeof(int),          cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_c.ptr,     h_c_mats.data(),    n_bath*(K+1)*sizeof(complex_t),       cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_coeff_plus.ptr,  h_coeff_plus.data(),  M*n_bath*(K+1)*sizeof(heom::real_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_coeff_minus.ptr, h_coeff_minus.data(), M*n_bath*(K+1)*sizeof(heom::real_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_level.ptr, h_level.data(),     M*n_bath*(K+1)*sizeof(int8_t),       cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_decay.ptr, h_decay.data(),     M*sizeof(double),                    cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_dcoup.ptr, h_dcoup.data(),     n_bath*N*sizeof(int8_t),             cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_markov.ptr, h_markov.data(),   n_bath*sizeof(heom::real_t),               cudaMemcpyHostToDevice));

    HEOMContext ctx{};
    ctx.rho_in            = d_rho.ptr;
    ctx.drho_out          = d_drho.ptr;
    ctx.hamiltonian       = d_H.ptr;
    ctx.c_bath       = d_c.ptr;
    ctx.coeff_plus   = d_coeff_plus.ptr;
    ctx.coeff_minus  = d_coeff_minus.ptr;
    ctx.markov_residue     = d_markov.ptr;
    ctx.neighbor_map_plus = d_plus.ptr;
    ctx.neighbor_map_minus= d_minus.ptr;
    ctx.level_indices     = d_level.ptr;
    ctx.total_decay       = d_decay.ptr;
    ctx.active_flags      = nullptr;     // no filtering
    ctx.d_coupling        = d_dcoup.ptr;
    ctx.M = M; ctx.N = N; ctx.K = K; ctx.n_bath = n_bath;
    ctx.sink_enabled = false; ctx.sink_site = 0; ctx.gamma_sink = 0.0;
    ctx.row_off = 0; ctx.n_row = N;
    ctx.col_off = 0; ctx.n_col = N;

    compute_heom_derivatives(ctx);
    CHK(cudaDeviceSynchronize());

    std::vector<complex_t> h_drho(M * N * N);
    CHK(cudaMemcpy(h_drho.data(), d_drho.ptr,
                   M * N * N * sizeof(complex_t),
                   cudaMemcpyDeviceToHost));

    // Analytic reference: -i [H, rho] for the values above.
    //   [H, rho] = H rho - rho H = [[0, -0.1], [0.1, 0]]
    //   -i [H, rho] = [[0, 0.1 i], [-0.1 i, 0]]
    const complex_t expected[4] = {
        c_make(0.0,  0.0),
        c_make(0.0,  0.1),
        c_make(0.0, -0.1),
        c_make(0.0,  0.0),
    };

    const double TOL = kKernelTol;
    bool ok = true;
    for (int i = 0; i < M * N * N; ++i) {
        const double dr = std::abs(c_re(h_drho[i]) - c_re(expected[i]));
        const double di = std::abs(c_im(h_drho[i]) - c_im(expected[i]));
        if (dr > TOL || di > TOL) {
            std::cout << "  element " << i << ": got ("
                      << c_re(h_drho[i]) << ", " << c_im(h_drho[i]) << "i)"
                      << ", expected ("
                      << c_re(expected[i]) << ", " << c_im(expected[i]) << "i)"
                      << "  diff=(" << dr << ", " << di << ")\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "test_gpu_kernel: PASS"
                  << "  (commutator vs CPU reference, tol=" << TOL << ")\n";
        if (!test_trajectory_solve_without_initial_throws()) {
            return 1;
        }
        std::cout << "trajectory initial-state guard: PASS\n";
        if (!test_single_side_mu()) {
            return 1;
        }
        if (!test_batch_single_side_mu()) {
            return 1;
        }
        if (!test_batch_gather_scatter()) {
            return 1;
        }
        if (!test_batch_trace_BR()) {
            return 1;
        }
        if (!test_ulocal_filter_marking()) {
            return 1;
        }
        if (!test_manifold_blocked_norms()) {
            return 1;
        }
        if (!test_carry_through_primitives()) {
            return 1;
        }
        if (!test_batch_derivatives_symmetry()) {
            return 1;
        }
        if (!test_sink_term()) {
            return 1;
        }
        if (!test_manifold_block_matches_full_launch()) {
            return 1;
        }
        if (!test_manifold_block_ignores_stale_stage_buffer()) {
            return 1;
        }
        if (!test_batch_rk4_symmetry()) {
            return 1;
        }
        if (!test_batch_union_norms()) {
            return 1;
        }
        if (!test_batch_alloc_plan()) {
            return 1;
        }
        if (!test_active_batch_width()) {
            return 1;
        }
        if (!test_batch_snapshot_expand()) {
            return 1;
        }
        if (!test_batch_detect()) {
            return 1;
        }
        if (!test_batch_dipole_per_slot()) {
            return 1;
        }
        if (!test_per_slot_active_counts()) {
            return 1;
        }
        if (!test_if_rk4_matches_plain_small_dt()) {
            return 1;
        }
        return 0;
    }
    std::cout << "test_gpu_kernel: FAIL\n";
    return 1;
}
