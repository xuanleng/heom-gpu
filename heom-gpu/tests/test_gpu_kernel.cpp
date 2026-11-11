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
#include "harness.hpp"
#include "core/rk4_integrator.cuh"
#include "core/ado_filter.cuh"
#include "core/cuda_common.cuh"
#include "core/dipole_apply.cuh"
#include "core/precision.hpp"
#include "heom_config.hpp"
#include "heom_trajectory.hpp"
#include "dynamics_run.hpp"
#include "detector.hpp"
#include "gpu_alloc_plan.hpp"
#include "memory_budget.hpp"
#include "output/hdf5_writer.hpp"
#include "system_context.hpp"

#include <hdf5.h>

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

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
    const std::string path = heom_test::tmp_dir() + "/test_heom_trajectory_guard.yaml";
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
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
      << "    filename: \"trajectory_guard\"\n"
      << "    use_timestamp: false\n";
    return path;
}

// Dimer (N=2), L=2, filtering OFF — small but multi-ADO, so carry-through
// snapshot/restore actually exercises α>0 slots. Filtering off keeps
// snapshot/restore byte-exact (full M, no norm-based drop).
std::string write_carry_through_yaml() {
    const std::string path = heom_test::tmp_dir() + "/test_heom_carry_through.yaml";
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
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
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
    const std::string path = heom_test::tmp_dir() + "/test_heom_double_excited_dimer.yaml";
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
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
      << "    filename: \"double_excited_dimer\"\n"
      << "    use_timestamp: false\n";
    return path;
}

bool test_trajectory_solve_without_initial_throws() {
    auto cfg = HEOMConfig::Config::load_yaml(write_tiny_trajectory_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);  // dynamics block: time grid + initial
    SystemContext sys(cfg);
    HEOMTrajectory traj(sys);

    try {
        heom::run_output_loop(traj, [](int, double, const std::complex<double>*, int, int) {});
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

    std::cerr << "FAIL: run_output_loop without set_initial() did not throw\n";
    return false;
}

// exciton_index k is the k-th exciton of the single-excitation manifold,
// the same physical state in every basis. Dimer with w0 = 12300: after the
// reorganisation shift the lower exciton sits ≈ -177 cm⁻¹ below |g⟩ (E = 0).
// A global eigen-sort put |g⟩ at column 1, so with_ground exciton 1 used to
// build ρ0 = |g⟩⟨g|.
std::string write_exciton_below_w0_yaml(const char* basis, int exciton_index) {
    const std::string path = heom_test::tmp(std::string("test_exciton_below_w0_") + basis
                                            + std::to_string(exciton_index) + ".yaml");
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  basis: " << basis << "\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100.0, -50.0]\n"
      << "    - [-50.0, 12300.0]\n"
      << "  w0: 12300.0\n"
      << "bath:\n"
      << "  eta: 70.0\n"
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
      << "  initial_state:\n"
      << "    type: exciton_excitation\n"
      << "    exciton_index: " << exciton_index << "\n"
      << "  output:\n"
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
      << "    filename: exciton_below_w0\n"
      << "    use_timestamp: false\n";
    return path;
}

bool test_exciton_index_is_basis_independent() {
    for (int k = 1; k <= 2; ++k) {
        auto build = [&](const char* basis) {
            auto cfg = HEOMConfig::Config::load_yaml(write_exciton_below_w0_yaml(basis, k));
            HEOMConfig::resolve_dynamics_config(cfg);
            SystemContext sys(cfg);
            return std::make_pair(build_initial_state(sys, *cfg.initial),
                                  sys.eigenvalues());
        };
        const auto [exc, e_exc] = build("excited_only");
        const auto [wg, e_wg]   = build("with_ground");
        const double d_rho = (wg.block(1, 1, 2, 2) - exc).cwiseAbs().maxCoeff();
        const double d_e   = std::abs(e_wg(k) - e_exc(k - 1));
        if (std::abs(wg(0, 0)) != 0.0 || d_rho > 1e-12 || d_e > 1e-15 || !(e_exc(0) < 0.0)) {
            std::cerr << "FAIL: exciton " << k << ": with_ground rho(g,g) = " << std::abs(wg(0, 0))
                      << ", max|Δρ_ee| = " << d_rho << ", |ΔE| = " << d_e
                      << ", E_1 = " << e_exc(0) << " (must be < 0 for this test to bite)\n";
            return false;
        }
    }
    return true;
}

// Root `status` says how a file ended, written LAST so a reader can
// trust whatever it finds. Four endings of the same one-record run:
//   killed     (no destructor runs: _exit in a forked child) -> "running"
//   unwound    (destroyed without close(): an exception)     -> "aborted"
//   close(Diverged)                                           -> "diverged"
//   close(Complete)                                           -> "complete"
// n_records_written is 1 whenever the writer got to say anything. Earlier
// the destructor silently called close(0.0), so a crashed run looked like a
// finished run that took 0 seconds, and a diverged run looked complete.
bool test_hdf5_writer_status() {
    using heom::hdf5::RunStatus;
    auto cfg = HEOMConfig::Config::load_yaml(write_tiny_trajectory_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    cfg.output.save_path = heom_test::tmp_dir() + "/";
    cfg.output.filename  = "test_heom_status";
    cfg.output.use_timestamp = false;
    SystemContext sys(cfg);
    const int N = sys.N();
    const int n_bath = sys.n_bath();
    Eigen::MatrixXcd rho0 = Eigen::MatrixXcd::Zero(N, N);
    rho0(0, 0) = 1.0;
    std::vector<std::complex<double>> flat(static_cast<size_t>(N) * N, 0.0);
    flat[0] = 1.0;
    const std::vector<std::complex<double>> zeros(static_cast<size_t>(n_bath), 0.0);
    const std::string path = heom_test::tmp_dir() + "/test_heom_status.h5";

    auto read_str = [](hid_t file, const char* name) -> std::string {
        if (H5Aexists(file, name) <= 0) return "<absent>";
        hid_t a = H5Aopen(file, name, H5P_DEFAULT);
        hid_t t = H5Aget_type(a);
        char* v = nullptr;
        H5Aread(a, t, &v);
        std::string out = v ? v : "";
        H5free_memory(v);
        H5Tclose(t);
        H5Aclose(a);
        return out;
    };
    auto read_int = [](hid_t file, const char* name) {
        if (H5Aexists(file, name) <= 0) return -1;
        hid_t a = H5Aopen(file, name, H5P_DEFAULT);
        int v = -1;
        H5Aread(a, H5T_NATIVE_INT, &v);
        H5Aclose(a);
        return v;
    };
    auto check = [&](const char* label, const std::string& want_status,
                     bool want_secs, int want_records) {
        hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) {
            std::cerr << "FAIL(" << label << "): cannot reopen " << path << "\n";
            return false;
        }
        const std::string st = read_str(file, "status");
        const bool secs = H5Aexists(file, "simulation_time_seconds") > 0;
        const int  nrec = read_int(file, "n_records_written");
        const bool inc  = H5Aexists(file, "incomplete") > 0;
        H5Fclose(file);
        std::cout << "  " << label << ": status=" << st << " n_records_written=" << nrec
                  << " simulation_time_seconds=" << (secs ? "present" : "absent") << "\n";
        if (st != want_status || secs != want_secs || nrec != want_records || inc) {
            std::cerr << "FAIL(" << label << "): want status=" << want_status
                      << " n_records_written=" << want_records << " secs="
                      << (want_secs ? "present" : "absent") << ", no `incomplete`\n";
            return false;
        }
        return true;
    };

    std::cout.flush();
    const pid_t pid = fork();
    if (pid == 0) {   // killed mid-run: nothing after write_timestep runs
        HDF5Writer w(false);
        w.open(sys, rho0, false, 0.0);
        w.write_timestep(0.0, flat.data(), N, zeros.data(), n_bath);
        _exit(0);
    }
    int wstatus = 0;
    waitpid(pid, &wstatus, 0);
    if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) {
        std::cerr << "FAIL(killed): child did not exit cleanly\n";
        return false;
    }
    if (!check("killed", "running", false, -1)) return false;

    {   // destroyed without close(): what an unwinding exception does
        HDF5Writer w(false);
        w.open(sys, rho0, false, 0.0);
        w.write_timestep(0.0, flat.data(), N, zeros.data(), n_bath);
    }
    if (!check("unwound", "aborted", false, 1)) return false;

    for (const auto [st, name] : {std::pair{RunStatus::Diverged, "diverged"},
                                  std::pair{RunStatus::Complete, "complete"}}) {
        HDF5Writer w(false);
        w.open(sys, rho0, false, 0.0);
        w.write_timestep(0.0, flat.data(), N, zeros.data(), n_bath);
        w.close(2.5, st);
        if (!check(name, name, true, 1)) return false;
    }

    std::remove(path.c_str());
    return true;
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

    const auto full = heom::ManifoldBlock::full(N);
    auto run_and_check = [&](int side) -> bool {
        launch_apply_single_side_mu(d_rho.ptr, d_out.ptr, d_mu.ptr, side, M, N,
                                    full, full, /*batch=*/1);
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
// independent B=1 launches on per-slot base pointers.
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
    const auto full = heom::ManifoldBlock::full(N);

    auto run_and_check = [&](int side) -> bool {
        CHK(cudaMemset(d_out_batch.ptr, 0, 2 * stride * sizeof(complex_t)));
        CHK(cudaMemset(d_out_ref.ptr, 0, 2 * stride * sizeof(complex_t)));
        // Batched: B=2, per-slot μ in one launch.
        launch_apply_single_side_mu(d_rho.ptr, d_out_batch.ptr, d_mu.ptr,
                                    side, M, N, full, full, 2);
        // Reference: two independent B=1 launches (bit-exact target).
        launch_apply_single_side_mu(d_rho.ptr, d_out_ref.ptr, d_mu.ptr,
                                    side, M, N, full, full, 1);
        launch_apply_single_side_mu(d_rho.ptr + stride, d_out_ref.ptr + stride,
                                    d_mu.ptr + N2, side, M, N, full, full, 1);
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
    if (ok) std::cout << "batch single-side mu (B=2 per-slot == 2x B=1): PASS\n";
    else    std::cout << "batch single-side mu: FAIL\n";
    return ok;
}

// Task A3: batched per-slot trace detection. B=2, each slot has a DISTINCT ρ
// (α=0 block) and DISTINCT μ_det; the batched launch must write out[b·ns+idx]
// == a B=1 launch_trace_BR on that slot's base pointers (identical tree reduction ⇒
// bit-exact FP64, 1e-4 mixed). Guards ρ slot stride (b·M·Nb, reads α=0 only,
// α>0 junk ignored), per-slot μ_det (b·N²) and per-slot output (b·ns+idx).
// ρ is stored compacted to `blk`; run at full(N) and at the (e,g)-shaped
// block {rows [1,3) × col [0,1)}, and in both cases also check against the
// host Tr(B·ρ) over the expanded matrix (summation order differs ⇒ kMuTol).
bool test_batch_trace_BR() {
    using cplx = std::complex<double>;
    const int M = 2, N = 3, B = 2;
    const size_t N2 = static_cast<size_t>(N) * N;
    const int ns = 5, idx = 2;
    bool all_ok = true;

    for (const heom::ManifoldBlock blk : {heom::ManifoldBlock::full(N),
                                          heom::ManifoldBlock{1, 2, 0, 1}}) {
        const int Nb = blk.elems();
        const size_t stride = static_cast<size_t>(M) * Nb;

        std::vector<complex_t> h_rho(B * stride), h_mu(B * N2);
        for (int b = 0; b < B; ++b)
            for (int a = 0; a < M; ++a)          // a=1 is junk that MUST be ignored
                for (int t = 0; t < Nb; ++t) {
                    const double re = 0.2 * (a + 1) + 0.03 * t + 0.7 * b;
                    const double im = -0.1 * (a + 1) + 0.05 * t - 0.4 * b;
                    h_rho[b * stride + a * Nb + t] = c_make(re, im);
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

        launch_trace_BR(d_mu.ptr, d_rho.ptr, d_out_batch.ptr, idx, N, M, B, ns, blk);
        for (int b = 0; b < B; ++b)
            launch_trace_BR(d_mu.ptr + b * N2, d_rho.ptr + b * stride,
                            d_out_ref.ptr + b * ns, idx, N, M, 1, ns, blk);
        CHK(cudaDeviceSynchronize());

        std::vector<complex_t> ob(B * ns), orf(B * ns);
        CHK(cudaMemcpy(ob.data(), d_out_batch.ptr, B * ns * sizeof(complex_t), cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(orf.data(), d_out_ref.ptr, B * ns * sizeof(complex_t), cudaMemcpyDeviceToHost));

        const double tol = heom::kMixedPrecision ? kMuTol : 0.0;
        for (int b = 0; b < B; ++b) {
            const complex_t g = ob[b * ns + idx];
            const complex_t e = orf[b * ns + idx];
            if (std::abs(c_re(g) - c_re(e)) > tol || std::abs(c_im(g) - c_im(e)) > tol) {
                std::cout << "  trace_BR Nb=" << Nb << " b=" << b << ": batched (" << c_re(g)
                          << "," << c_im(g) << ") ref (" << c_re(e) << "," << c_im(e) << ")\n";
                all_ok = false;
            }
            // Host oracle on the expanded α=0 matrix: Tr(B·ρ) = Σ_i Σ_k B[i,k]·ρ[k,i].
            cplx acc(0.0, 0.0);
            for (int i = 0; i < N; ++i)
                for (int k = 0; k < N; ++k) {
                    const int lk = k - blk.row_off, li = i - blk.col_off;
                    if (lk < 0 || lk >= blk.n_row || li < 0 || li >= blk.n_col) continue;
                    const complex_t Bik = h_mu[b * N2 + i * N + k];
                    const complex_t rki = h_rho[b * stride + lk * blk.n_col + li];
                    acc += cplx(c_re(Bik), c_im(Bik)) * cplx(c_re(rki), c_im(rki));
                }
            if (std::abs(c_re(g) - acc.real()) > kMuTol || std::abs(c_im(g) - acc.imag()) > kMuTol) {
                std::cout << "  trace_BR Nb=" << Nb << " b=" << b << ": got (" << c_re(g)
                          << "," << c_im(g) << ") host (" << acc.real() << "," << acc.imag() << ")\n";
                all_ok = false;
            }
        }
    }

    if (all_ok) std::cout << "batch trace_BR (B=2 per-slot == single, full + (e,g) block): PASS\n";
    else        std::cout << "batch trace_BR: FAIL\n";
    return all_ok;
}

// Rectangular dipole apply, S ≠ T. 3-site double-excited basis (N = 1+3+3
// = 7): (g,g) →Left→ (e,g), then (e,g) →Right→ (e,e). Input ρ is stored on S,
// output on T; the oracle is the host full-matrix product (zero outside S)
// restricted to T, element-wise within kMuTol (host vs device rounding, as in
// test_single_side_mu). μ is a random-ish dense matrix, so the check is
// "compact rectangle == full product on T", not a physics claim.
bool test_rect_single_side_mu() {
    using cplx = std::complex<double>;
    const int M = 3, N = 7, B = 2;
    const size_t N2 = static_cast<size_t>(N) * N;
    const heom::ManifoldBlock gg{0, 1, 0, 1}, eg{1, 3, 0, 1}, ee{1, 3, 1, 3};
    struct Case { heom::ManifoldBlock S, T; int side; const char* name; };
    const Case cases[] = { {gg, eg, 0, "gg->eg Left"}, {eg, ee, 1, "eg->ee Right"} };

    std::vector<complex_t> h_mu(B * N2);
    std::vector<cplx> mu(B * N2);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const cplx v(0.2 * i + 0.1 * j + 0.05 + 0.4 * b, -0.1 * i + 0.07 * j + 0.2 * b);
                mu[b * N2 + i * N + j] = v;
                h_mu[b * N2 + i * N + j] = c_make(v.real(), v.imag());
            }
    DevBuf<complex_t> d_mu(B * N2);
    CHK(cudaMemcpy(d_mu.ptr, h_mu.data(), B * N2 * sizeof(complex_t), cudaMemcpyHostToDevice));

    const double tol = kMuTol;
    bool all_ok = true;
    for (const Case& c : cases) {
        const int NbS = c.S.elems(), NbT = c.T.elems();
        std::vector<complex_t> h_in(B * M * NbS);
        std::vector<cplx> full_in(B * M * N2, cplx(0.0, 0.0));   // expanded oracle input
        for (int b = 0; b < B; ++b)
            for (int a = 0; a < M; ++a)
                for (int r = 0; r < c.S.n_row; ++r)
                    for (int cc = 0; cc < c.S.n_col; ++cc) {
                        const cplx v(0.1 * (a + 1) + 0.3 * r - 0.2 * cc + 0.5 * b,
                                     0.05 * (a + 1) - 0.1 * r + 0.15 * cc - 0.3 * b);
                        h_in[(b * M + a) * NbS + r * c.S.n_col + cc] = c_make(v.real(), v.imag());
                        full_in[(b * M + a) * N2 + (c.S.row_off + r) * N + (c.S.col_off + cc)] = v;
                    }
        DevBuf<complex_t> d_in(B * M * NbS), d_out(B * M * NbT);
        CHK(cudaMemcpy(d_in.ptr, h_in.data(), B * M * NbS * sizeof(complex_t), cudaMemcpyHostToDevice));
        launch_apply_single_side_mu(d_in.ptr, d_out.ptr, d_mu.ptr, c.side, M, N, c.S, c.T, B);
        CHK(cudaDeviceSynchronize());
        std::vector<complex_t> h_out(B * M * NbT);
        CHK(cudaMemcpy(h_out.data(), d_out.ptr, B * M * NbT * sizeof(complex_t), cudaMemcpyDeviceToHost));

        for (int b = 0; b < B; ++b)
            for (int a = 0; a < M; ++a)
                for (int r = 0; r < c.T.n_row; ++r)
                    for (int cc = 0; cc < c.T.n_col; ++cc) {
                        const int i = c.T.row_off + r, j = c.T.col_off + cc;
                        cplx acc(0.0, 0.0);
                        for (int k = 0; k < N; ++k)
                            acc += c.side == 0
                                 ? mu[b * N2 + i * N + k] * full_in[(b * M + a) * N2 + k * N + j]
                                 : full_in[(b * M + a) * N2 + i * N + k] * mu[b * N2 + k * N + j];
                        const complex_t g = h_out[(b * M + a) * NbT + r * c.T.n_col + cc];
                        if (std::abs(c_re(g) - acc.real()) > tol || std::abs(c_im(g) - acc.imag()) > tol) {
                            std::cout << "  rect mu " << c.name << " b=" << b << " a=" << a << " ("
                                      << i << "," << j << "): got (" << c_re(g) << "," << c_im(g)
                                      << ") exp (" << acc.real() << "," << acc.imag() << ")\n";
                            all_ok = false;
                        }
                    }
    }
    if (all_ok) std::cout << "rectangular single-side mu (gg->eg Left, eg->ee Right; == full product on T): PASS\n";
    else        std::cout << "rectangular single-side mu: FAIL\n";
    return all_ok;
}

// U-local filter: the U-local norm scan + mark must equal the
// full-M reference on the scan domain and leave flags outside it untouched.
// Also locks the scan-domain rule for the dense→compact transition: when the
// previous write set was the full grid, a full-M scan (scan_list = nullptr)
// must reproduce the reference everywhere.
// Runs both norm kernels (per-thread Nb<64 and warp Nb>=64; here Nb = N²).
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
        DevBuf<int>             d_nan(1);
        CHK(cudaMemset(d_nan.ptr, 0, sizeof(int)));
        CHK(cudaMemcpy(d_rho.ptr, h_rho.data(),
                       M * NN * sizeof(complex_t), cudaMemcpyHostToDevice));

        // Full-M reference (also the dense→compact transition case: previous
        // write set = full grid ⇒ scan_list = nullptr must be used).
        std::vector<int8_t> flags_ref(M);
        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N, heom::ManifoldBlock::full(N));
        filter_ados_mark_only(d_norms.ptr, d_flags.ptr, d_nan.ptr, threshold, M);
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

        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N, heom::ManifoldBlock::full(N),
                          d_ulist.ptr, u_count);
        filter_ados_mark_only(d_norms.ptr, d_flags.ptr, d_nan.ptr, threshold, M,
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
        int nan_flag = -1;
        CHK(cudaMemcpy(&nan_flag, d_nan.ptr, sizeof(int), cudaMemcpyDeviceToHost));
        if (nan_flag != 0) {
            std::cout << "  ulocal filter N=" << N
                      << ": nan_flag raised on all-finite input\n";
            all_ok = false;
        }
    }

    if (all_ok) {
        std::cout << "ulocal filter marking (== full-M on U, untouched outside): PASS\n";
        return true;
    }
    std::cout << "ulocal filter marking: FAIL\n";
    return false;
}

// A NaN anywhere in an ADO must surface, not vanish. The old norm
// kernel's `abs_sq > max` skipped NaN, reported the max of the finite
// remainder (0 for a fully-NaN block), and the filter then marked the ADO
// inactive — freezing the blow-up behind a flag ‖ρ₀‖_F never sees.
// Checks: norms[α]=NaN sticks through both norm kernels (per-thread N=3,
// warp N=10) and the warp shuffle; finite blocks are untouched; the NaN ADO
// is marked ACTIVE; nan_flag is raised; and a re-run on finite data does
// not raise it.
bool test_filter_nan_surfaces() {
    bool all_ok = true;
    const double kNaN = std::numeric_limits<double>::quiet_NaN();

    for (const int N : {3, 10}) {
        const int M = 4;
        const int NN = N * N;
        const double threshold = 1.0e-2;

        // Every block 1e-3 (below threshold) except α=2 which carries ONE NaN
        // at the last element (so the sticky max must survive the whole scan
        // and, for N=10, the cross-lane reduction).
        std::vector<complex_t> h_rho(M * NN, c_make(1.0e-3, 0.0));
        h_rho[2 * NN + (NN - 1)] = c_make(kNaN, 0.0);

        DevBuf<complex_t> d_rho(M * NN);
        DevBuf<double>    d_norms(M);
        DevBuf<int8_t>    d_flags(M);
        DevBuf<int>       d_nan(1);
        CHK(cudaMemset(d_nan.ptr, 0, sizeof(int)));
        CHK(cudaMemcpy(d_rho.ptr, h_rho.data(),
                       M * NN * sizeof(complex_t), cudaMemcpyHostToDevice));

        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N, heom::ManifoldBlock::full(N));
        filter_ados_mark_only(d_norms.ptr, d_flags.ptr, d_nan.ptr, threshold, M);
        CHK(cudaDeviceSynchronize());

        std::vector<double> norms(M);
        std::vector<int8_t> flags(M);
        int nan_flag = -1;
        CHK(cudaMemcpy(norms.data(), d_norms.ptr, M * sizeof(double), cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(flags.data(), d_flags.ptr, M, cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(&nan_flag, d_nan.ptr, sizeof(int), cudaMemcpyDeviceToHost));

        if (!std::isnan(norms[2])) {
            std::cout << "  nan filter N=" << N << ": norms[2]=" << norms[2]
                      << " (NaN was swallowed by the norm kernel)\n";
            all_ok = false;
        }
        for (int a : {1, 3}) {
            if (!(std::abs(norms[a] - 1.0e-3) < 1.0e-9)) {
                std::cout << "  nan filter N=" << N << ": norms[" << a << "]="
                          << norms[a] << " expect 1e-3\n";
                all_ok = false;
            }
        }
        const int8_t expect_flags[4] = {1, 0, 1, 0};
        for (int a = 0; a < M; ++a) {
            if (flags[a] != expect_flags[a]) {
                std::cout << "  nan filter N=" << N << ": flags[" << a << "]="
                          << int(flags[a]) << " expect " << int(expect_flags[a]) << "\n";
                all_ok = false;
            }
        }
        if (nan_flag != 1) {
            std::cout << "  nan filter N=" << N << ": nan_flag=" << nan_flag
                      << " expect 1\n";
            all_ok = false;
        }

        // Finite re-run on a fresh flag must stay quiet.
        h_rho[2 * NN + (NN - 1)] = c_make(1.0e-3, 0.0);
        CHK(cudaMemcpy(d_rho.ptr, h_rho.data(),
                       M * NN * sizeof(complex_t), cudaMemcpyHostToDevice));
        CHK(cudaMemset(d_nan.ptr, 0, sizeof(int)));
        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N, heom::ManifoldBlock::full(N));
        filter_ados_mark_only(d_norms.ptr, d_flags.ptr, d_nan.ptr, threshold, M);
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(&nan_flag, d_nan.ptr, sizeof(int), cudaMemcpyDeviceToHost));
        if (nan_flag != 0) {
            std::cout << "  nan filter N=" << N << ": nan_flag=" << nan_flag
                      << " on finite re-run\n";
            all_ok = false;
        }
    }

    if (all_ok) {
        std::cout << "filter NaN surfaces (sticky norm, active, flag): PASS\n";
        return true;
    }
    std::cout << "filter NaN surfaces: FAIL\n";
    return false;
}

// Host-side compaction: full [M][N²] hierarchy → compact [M][Nb] on `blk`.
// Elements outside the block are DROPPED — that is the storage contract,
// and the kernel-level tests below feed the block launch exactly this while
// the full launch (oracle) keeps the full matrix.
std::vector<complex_t> compact(const std::vector<complex_t>& full, int M, int N,
                               const heom::ManifoldBlock& blk) {
    const int Nb = blk.elems();
    std::vector<complex_t> out(static_cast<size_t>(M) * Nb);
    for (int a = 0; a < M; ++a)
        for (int r = 0; r < blk.n_row; ++r)
            for (int c = 0; c < blk.n_col; ++c)
                out[static_cast<size_t>(a) * Nb + r * blk.n_col + c] =
                    full[static_cast<size_t>(a) * N * N + (blk.row_off + r) * N + blk.col_off + c];
    return out;
}

// Manifold-blocked norm scan on COMPACT input. Properties:
//   (1) equivalence : norms of the compact [M][Nb] block == norms of the full
//                     matrix that is zero outside the block, bit-equal
//   (2) drop        : compact() discards out-of-block elements — poisoning the
//                     exterior of the full matrix changes nothing the kernel
//                     sees, so the norms are again bit-equal
//   (3) rejection   : empty / out-of-range blocks throw
// Both kernels are covered on compact data: the (e,g)-shaped column (Nb = N−1
// < 64, per-thread) and the (e,e) square (N=29 ⇒ Nb = 784 ≥ 64, warp); the
// full-matrix oracle runs per-thread at N=4 and warp at N=29.
bool test_manifold_blocked_norms() {
    bool all_ok = true;

    for (const int N : {4, 29}) {
        const int M  = 8;
        const int NN = N * N;
        for (const heom::ManifoldBlock blk : {heom::ManifoldBlock{1, N - 1, 0, 1},
                                              heom::ManifoldBlock{1, N - 1, 1, N - 1}}) {
            const int Nb = blk.elems();
            std::vector<complex_t> h_rho(M * NN, c_make(0.0, 0.0));
            for (int a = 0; a < M; ++a)
                for (int r = 0; r < blk.n_row; ++r)
                    for (int c = 0; c < blk.n_col; ++c) {
                        // Distinct magnitudes so the max is a specific element.
                        const double v = 1.0 / (a + 1) - 0.0001 * (r * blk.n_col + c);
                        h_rho[a * NN + (blk.row_off + r) * N + blk.col_off + c] = c_make(v, 0.0);
                    }

            DevBuf<complex_t> d_full(M * NN), d_cmp(M * Nb);
            DevBuf<double>    d_norms(M);
            CHK(cudaMemcpy(d_full.ptr, h_rho.data(), M * NN * sizeof(complex_t),
                           cudaMemcpyHostToDevice));
            const auto h_cmp = compact(h_rho, M, N, blk);
            CHK(cudaMemcpy(d_cmp.ptr, h_cmp.data(), M * Nb * sizeof(complex_t),
                           cudaMemcpyHostToDevice));

            // (1) equivalence
            std::vector<double> full(M), blocked(M);
            compute_ado_norms(d_full.ptr, d_norms.ptr, M, N, heom::ManifoldBlock::full(N));
            CHK(cudaDeviceSynchronize());
            CHK(cudaMemcpy(full.data(), d_norms.ptr, M * sizeof(double), cudaMemcpyDeviceToHost));

            compute_ado_norms(d_cmp.ptr, d_norms.ptr, M, N, blk);
            CHK(cudaDeviceSynchronize());
            CHK(cudaMemcpy(blocked.data(), d_norms.ptr, M * sizeof(double), cudaMemcpyDeviceToHost));

            for (int a = 0; a < M; ++a)
                if (blocked[a] != full[a]) {          // bit-equal, not approximate
                    std::cout << "  blocked norms N=" << N << " Nb=" << Nb << " a=" << a
                              << ": compact " << blocked[a]
                              << " != whole-matrix " << full[a] << "\n";
                    all_ok = false;
                }

            // (2) drop: poison the exterior, compact again, same norms.
            std::vector<complex_t> h_poison = h_rho;
            for (int a = 0; a < M; ++a)
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < N; ++j) {
                        const bool inside = (i >= blk.row_off && i < blk.row_off + blk.n_row &&
                                             j >= blk.col_off && j < blk.col_off + blk.n_col);
                        if (!inside) h_poison[a * NN + i * N + j] = c_make(1.0e3, 0.0);
                    }
            const auto h_cmp2 = compact(h_poison, M, N, blk);
            CHK(cudaMemcpy(d_cmp.ptr, h_cmp2.data(), M * Nb * sizeof(complex_t),
                           cudaMemcpyHostToDevice));
            std::vector<double> dropped(M);
            compute_ado_norms(d_cmp.ptr, d_norms.ptr, M, N, blk);
            CHK(cudaDeviceSynchronize());
            CHK(cudaMemcpy(dropped.data(), d_norms.ptr, M * sizeof(double), cudaMemcpyDeviceToHost));
            for (int a = 0; a < M; ++a)
                if (dropped[a] != blocked[a]) {
                    std::cout << "  blocked norms N=" << N << " Nb=" << Nb << " a=" << a
                              << ": exterior 1e3 survived compaction (" << dropped[a]
                              << " vs " << blocked[a] << ")\n";
                    all_ok = false;
                }
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
                                  heom::ManifoldBlock{b.ro, b.nr, b.co, b.nc});
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
        std::cout << "manifold blocked norms (compact == full, exterior dropped, rejection): PASS\n";
        return true;
    }
    std::cout << "manifold blocked norms: FAIL\n";
    return false;
}

// Slot 0's physical layer, straight from the device.
Eigen::MatrixXcd read_rho0_via_expectation(const HEOMTrajectory& traj, int /*N*/) {
    return traj.slot_rho0(0);
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
        HEOMTrajectory traj(sys);
        const int A = 7, B = 5;
        traj.set_initial({rho0}, heom::ManifoldBlock::full(N));
        traj.propagate_steps(A + B);
        const Eigen::MatrixXcd rho_cont = read_rho0_via_expectation(traj, N);

        traj.set_initial({rho0}, heom::ManifoldBlock::full(N));
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
        HEOMTrajectory traj(sys);
        const int K = 6, K2 = 4;
        traj.set_initial({rho0}, heom::ManifoldBlock::full(N));
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
        HEOMTrajectory traj(sys);
        traj.set_initial({rho0}, heom::ManifoldBlock::full(N));
        Eigen::MatrixXcd mu(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                mu(i, j) = std::complex<double>(0.2 * i - 0.1 * j + 0.3,
                                                0.05 * i + 0.07 * j);
        traj.apply_dipole({mu}, DipoleSide::Left, heom::ManifoldBlock::full(N));
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

// The support block is state that travels with ρ, and the two calls that
// can carry a nonzero outside it (set_initial, apply_dipole) check the claim on
// the host before touching the device. Doubly-excited dimer, N=4: g={0},
// e={1,2}, f={3}.
//   (i)   set_initial: a nonzero outside the declared block throws.
//   (ii)  apply_dipole: a μ that maps the current support outside the declared
//         one throws; so does a declared block that moves the fixed axis.
//   (iii) the legal sequence g → (e,g) → (f,g) passes, support() follows, and
//         a snapshot/restore round-trip re-installs the captured block.
bool test_support_validation() {
    using namespace heom::spectra;
    using heom::ManifoldBlock;
    auto cfg = HEOMConfig::Config::load_yaml(write_double_excited_dimer_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int N = sys.N();
    if (N != 4) {
        std::cout << "  expected N=4 for the doubly-excited dimer, got " << N << "\n";
        return false;
    }
    const ManifoldBlock gg{0, 1, 0, 1}, eg{1, 2, 0, 1}, fg{3, 1, 0, 1}, ee{1, 2, 1, 2};

    Eigen::MatrixXcd ground = Eigen::MatrixXcd::Zero(N, N);
    ground(0, 0) = 1.0;
    // μ⁺ shaped like DipoleOperator::from_pigment_dipoles: g→e column, e→f block.
    Eigen::MatrixXcd mu_up = Eigen::MatrixXcd::Zero(N, N);
    mu_up(1, 0) = 1.0; mu_up(2, 0) = 0.5;
    mu_up(3, 1) = 0.5; mu_up(3, 2) = 1.0;

    auto expect_throw = [](const char* what, auto&& fn) {
        try { fn(); }
        catch (const std::invalid_argument&) { return true; }
        catch (const std::exception& e) {
            std::cout << "  " << what << ": wrong exception type: " << e.what() << "\n";
            return false;
        }
        std::cout << "  " << what << ": did not throw\n";
        return false;
    };
    bool ok = true;

    // (i) ρ0 = |g><g| declared as (e,e): (0,0) is outside.
    {
        HEOMTrajectory t(sys);
        ok &= expect_throw("set_initial outside block",
                           [&] { t.set_initial({ground}, ee); });
    }
    // (ii-a) μ⁺ on (g,g) reaches e only, but a μ with a direct g→f entry
    // declared as (e,g) would freeze the f row.
    {
        HEOMTrajectory t(sys);
        t.set_initial({ground}, gg);
        Eigen::MatrixXcd bad = mu_up;
        bad(3, 0) = 0.25;                            // g→f directly
        ok &= expect_throw("apply_dipole mu maps outside declared block",
                           [&] { t.apply_dipole({bad}, DipoleSide::Left, eg); });
    }
    // (ii-b) a Left dipole keeps the columns; declaring (e,e) after (g,g) moves them.
    {
        HEOMTrajectory t(sys);
        t.set_initial({ground}, gg);
        ok &= expect_throw("apply_dipole Left changes columns",
                           [&] { t.apply_dipole({mu_up}, DipoleSide::Left, ee); });
    }
    // (iii) legal ESA-like sequence + snapshot carries the block.
    {
        HEOMTrajectory t(sys);
        t.set_initial({ground}, gg);
        t.apply_dipole({mu_up}, DipoleSide::Left, eg);
        if (t.support() != eg) { std::cout << "  support after mu0 != (e,g)\n"; ok = false; }
        DeviceHierarchySnapshot snap;
        t.snapshot_device(snap);
        if (snap.support != eg) { std::cout << "  snapshot did not capture (e,g)\n"; ok = false; }
        t.apply_dipole({mu_up}, DipoleSide::Left, fg);
        if (t.support() != fg) { std::cout << "  support after mu1 != (f,g)\n"; ok = false; }
        const Eigen::MatrixXcd got = read_rho0_via_expectation(t, N);
        const Eigen::MatrixXcd exp = mu_up * mu_up * ground;
        if ((got - exp).cwiseAbs().maxCoeff() > kMuTol) {
            std::cout << "  (f,g) physical layer != mu*mu*rho0\n"; ok = false;
        }
        t.restore_device(snap);
        if (t.support() != eg) { std::cout << "  restore did not re-install (e,g)\n"; ok = false; }
        DeviceHierarchySnapshot empty;
        ok &= expect_throw("restore from empty snapshot",
                           [&] { t.restore_device(empty); });
    }

    if (ok) std::cout << "support validation (set_initial/apply_dipole/snapshot): PASS\n";
    else    std::cout << "support validation: FAIL\n";
    return ok;
}

// rho_capacity per task on the doubly-excited dimer (n_states=2),
// and the capacity check with teeth — a trajectory built for absorption (cap =
// n_e = 2) must refuse set_initial(full(N)) BEFORE any write, naming the
// capacity. Also: the compact absorption trajectory must round-trip its (e,g)
// block through slot_rho0 (expanded with exact zeros outside).
bool test_rho_capacity() {
    using heom::ManifoldBlock;
    bool ok = true;

    // dynamics (resolver sets Task::Dynamics): N=4 ⇒ 16
    auto cfg_dyn = HEOMConfig::Config::load_yaml(write_double_excited_dimer_yaml());
    HEOMConfig::resolve_dynamics_config(cfg_dyn);
    if (heom::rho_capacity(cfg_dyn) != 16) {
        std::cout << "  dynamics capacity != 16: " << heom::rho_capacity(cfg_dyn) << "\n"; ok = false;
    }
    // 2DES in the double basis: n_e=2, n_f=1 ⇒ max(4, 2) = 4; with_ground ⇒ 4.
    auto cfg_2d = cfg_dyn;
    cfg_2d.task = HEOMConfig::Config::Task::TwoDES;
    if (heom::rho_capacity(cfg_2d) != 4) {
        std::cout << "  2des double capacity != 4: " << heom::rho_capacity(cfg_2d) << "\n"; ok = false;
    }
    auto cfg_2d_wg = cfg_2d;
    HEOMConfig::apply_basis(cfg_2d_wg, BasisType::WITH_GROUND);
    if (heom::rho_capacity(cfg_2d_wg) != 4) {
        std::cout << "  2des with_ground capacity != 4: " << heom::rho_capacity(cfg_2d_wg) << "\n"; ok = false;
    }
    // absorption: with_ground, (e,g) ⇒ n_e·1 = 2.
    auto cfg_abs = cfg_dyn;
    cfg_abs.task = HEOMConfig::Config::Task::Absorption;
    HEOMConfig::apply_basis(cfg_abs, BasisType::WITH_GROUND);
    if (heom::rho_capacity(cfg_abs) != 2) {
        std::cout << "  absorption capacity != 2: " << heom::rho_capacity(cfg_abs) << "\n"; ok = false;
    }

    // The trajectory built on the absorption config: full(N) must be refused.
    {
        SystemContext sys(cfg_abs);
        const int N = sys.N();
        HEOMTrajectory t(sys);
        Eigen::MatrixXcd full_rho = Eigen::MatrixXcd::Zero(N, N);
        full_rho(1, 1) = 1.0;
        bool threw = false;
        try { t.set_initial({full_rho}, ManifoldBlock::full(N)); }
        catch (const std::invalid_argument& e) {
            threw = true;
            if (std::string(e.what()).find("capacity") == std::string::npos) {
                std::cout << "  capacity error does not name the capacity: " << e.what() << "\n";
                ok = false;
            }
        }
        if (!threw) { std::cout << "  set_initial(full) on an absorption trajectory did not throw\n"; ok = false; }

        // The block it WAS built for round-trips (compact store, expanded read).
        const ManifoldBlock eg{1, N - 1, 0, 1};
        Eigen::MatrixXcd r0 = Eigen::MatrixXcd::Zero(N, N);
        r0(1, 0) = std::complex<double>(0.25, -0.5);
        r0(2, 0) = std::complex<double>(-0.75, 0.125);
        t.set_initial({r0}, eg);
        if (t.support() != eg) { std::cout << "  support after set_initial != (e,g)\n"; ok = false; }
        const Eigen::MatrixXcd back = t.slot_rho0(0);
        if ((back - r0).cwiseAbs().maxCoeff() != 0.0) {
            std::cout << "  (e,g) block did not round-trip through compact storage\n"; ok = false;
        }
        // A snapshot claiming a block beyond this capacity can only be forged:
        // a real one comes from this context, whose trajectories share it.
        DeviceHierarchySnapshot big;
        big.support = ManifoldBlock::full(N);
        threw = false;
        try { t.restore_device(big); } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) { std::cout << "  restore_device(forged full snapshot) on absorption trajectory did not throw\n"; ok = false; }
    }

    if (ok) std::cout << "rho_capacity (dynamics 16 / 2des 4 / absorption 2) + capacity guard: PASS\n";
    else    std::cout << "rho_capacity: FAIL\n";
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
    c.cbar_bath           = sys.d_cbar_bath();
    c.coeff_plus          = sys.d_coeff_plus();
    c.coeff_minus         = sys.d_coeff_minus();
    c.markov_residue      = sys.d_markov_residue();
    c.neighbor_map_plus   = sys.d_plus_map();
    c.neighbor_map_minus  = sys.d_minus_map();
    c.level_indices       = sys.d_level_indices();
    c.total_decay         = sys.d_total_decay();
    c.active_flags        = nullptr;
    c.d_coupling          = sys.d_d_coupling();
    c.mode_off            = sys.d_mode_off();
    c.M = sys.M(); c.N = sys.N(); c.D = sys.D(); c.n_bath = sys.n_bath();
    c.sink_enabled = false; c.sink_site = 0; c.gamma_sink = 0.0;
    c.u_list = nullptr; c.u_count = 0;
    c.batch = 1;
    c.block = heom::ManifoldBlock::full(sys.N());
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

        // Oracle: full launch on the full matrix. Under test: block launch on
        // the COMPACT [M][Nb] input (compact storage) writing a compact output.
        const heom::ManifoldBlock cblk{cs.ro, cs.nr, cs.co, cs.nc};
        const int Nb = cblk.elems();
        const size_t cstride = static_cast<size_t>(M) * Nb;
        const auto h_cmp = compact(h_rho, M, N, cblk);

        DevBuf<complex_t> d_rho(stride), d_ref(stride), d_cmp(cstride), d_blk(cstride);
        CHK(cudaMemcpy(d_rho.ptr, h_rho.data(), stride * sizeof(complex_t),
                       cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_cmp.ptr, h_cmp.data(), cstride * sizeof(complex_t),
                       cudaMemcpyHostToDevice));
        CHK(cudaMemset(d_ref.ptr, 0, stride * sizeof(complex_t)));
        CHK(cudaMemset(d_blk.ptr, 0, cstride * sizeof(complex_t)));

        HEOMContext full = build_multi_ado_ctx(sys);
        full.rho_in = d_rho.ptr; full.drho_out = d_ref.ptr;
        HEOMContext blocked = build_multi_ado_ctx(sys);
        blocked.rho_in = d_cmp.ptr; blocked.drho_out = d_blk.ptr;
        blocked.block = cblk;

        compute_heom_derivatives(full);
        compute_heom_derivatives(blocked);
        CHK(cudaDeviceSynchronize());

        std::vector<complex_t> ref(stride), blk(cstride);
        CHK(cudaMemcpy(ref.data(), d_ref.ptr, stride * sizeof(complex_t),
                       cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(blk.data(), d_blk.ptr, cstride * sizeof(complex_t),
                       cudaMemcpyDeviceToHost));

        int n_in_bad = 0, n_out_premise = 0;
        for (int a = 0; a < M; ++a)
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    const size_t e = static_cast<size_t>(a) * N * N + i * N + j;
                    const bool inside = (i >= cs.ro && i < cs.ro + cs.nr &&
                                         j >= cs.co && j < cs.co + cs.nc);
                    if (inside) {
                        const size_t ce = static_cast<size_t>(a) * Nb
                                        + (i - cs.ro) * cs.nc + (j - cs.co);
                        if (!(c_re(blk[ce]) == c_re(ref[e]) &&
                              c_im(blk[ce]) == c_im(ref[e]))) {
                            if (n_in_bad < 3)
                                std::cout << "    in-block mismatch a=" << a
                                          << " (" << i << "," << j << "): compact ("
                                          << c_re(blk[ce]) << "," << c_im(blk[ce])
                                          << ") full (" << c_re(ref[e]) << ","
                                          << c_im(ref[e]) << ")\n";
                            ++n_in_bad;
                        }
                    } else {
                        // The full launch must agree that the derivative out
                        // there is zero, or ρ would leak out of the block on the
                        // next step and compact storage would be dropping real
                        // data. This holds because H_S is manifold-block-
                        // diagonal; if it ever stops being so, this is where we
                        // find out.
                        if (c_re(ref[e]) != 0.0 || c_im(ref[e]) != 0.0) ++n_out_premise;
                    }
                }

        if (n_in_bad || n_out_premise) {
            std::cout << "  " << cs.name << ": in-block mismatches=" << n_in_bad
                      << " out-of-block nonzero derivative=" << n_out_premise << "\n";
            ok = false;
        }
    }

    if (ok) std::cout << "manifold block (compact) == full launch restricted (4 blocks, bitwise): PASS\n";
    else    std::cout << "manifold block == full launch: FAIL\n";
    return ok;
}

// lhc2-smem: the kernel stages only the H rows a 256-thread block
// touches (sA: R cyclic rows from li0) for a DISJOINT manifold block. The
// N=4 dimer above never exercises that path — its Nb ≤ 2 makes every block
// cover all rows. This synthetic case uses the LHC-II ESA shape (N = 1+14+91
// = 106; f×e = 91×14, Nb = 1274 > 256) and the transposed e×f, with M = 5
// ADOs so blocks start mid-row and straddle ADO boundaries. Term 4/5 are off
// (no neighbours); Term 1 reads the staged H, Term 3 (two baths with nonzero
// Markov residues and a varied d_coupling pattern) reads the staged d
// columns (step 2). H is real, symmetric
// and manifold-block-diagonal with entries spanning 1e-3..1e3 so a staging
// bug (wrong row, wrong cyclic wrap) changes values. Oracle: the whole-matrix
// launch (diagonal path, all of H staged) restricted to the block, bitwise.
bool test_disjoint_block_partial_row_staging() {
    const int ng = 1, ne = 14, nf = 91, N = ng + ne + nf, M = 5;
    const int n_bath = 2, D = 2;
    const size_t NN = static_cast<size_t>(N) * N, stride = static_cast<size_t>(M) * NN;
    auto manifold = [&](int s) { return s < ng ? 0 : (s < ng + ne ? 1 : 2); };
    std::vector<complex_t> h_H(NN, c_make(0.0, 0.0));
    for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) {
            if (manifold(i) != manifold(j)) continue;
            const double v = (i == j) ? 1.0e3 * (1.0 + 0.01 * i)
                                      : std::pow(10.0, ((i * 7 + j * 3) % 5) - 3.0) * (((i + j) % 2) ? 1.0 : -1.0);
            h_H[i * N + j] = c_make(v, 0.0);
            h_H[j * N + i] = c_make(v, 0.0);
        }
    const std::vector<int> h_map(M * D, -1);
    const std::vector<int8_t> h_level(M * D, 0);
    const std::vector<heom::real_t> h_coeff(M * D, 0);
    const std::vector<complex_t> h_c(D, c_make(0.0, 0.0));
    const std::vector<cuDoubleComplex> h_decay(M, make_cuDoubleComplex(0.0, 0.0));
    const std::vector<heom::real_t> h_markov = {0.37, 1.9};
    const std::vector<int> h_mode_off = {0, 1, 2};
    std::vector<int8_t> h_dc(n_bath * N, 0);
    for (int n = 0; n < n_bath; ++n)
        for (int s = 0; s < N; ++s) h_dc[n * N + s] = ((s * (n + 3) + n) % 3 == 0) ? 1 : 0;

    DevBuf<complex_t> d_H(NN), d_c(D);
    DevBuf<int> d_map(M * D), d_mode_off(n_bath + 1);
    DevBuf<int8_t> d_level(M * D), d_dc(n_bath * N);
    DevBuf<heom::real_t> d_coeff(M * D), d_markov(n_bath);
    DevBuf<cuDoubleComplex> d_decay(M);
    CHK(cudaMemcpy(d_H.ptr, h_H.data(), NN * sizeof(complex_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_c.ptr, h_c.data(), D * sizeof(complex_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_map.ptr, h_map.data(), M * D * sizeof(int), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_mode_off.ptr, h_mode_off.data(), (n_bath + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_level.ptr, h_level.data(), M * D * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_dc.ptr, h_dc.data(), n_bath * N * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_coeff.ptr, h_coeff.data(), M * D * sizeof(heom::real_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_markov.ptr, h_markov.data(), n_bath * sizeof(heom::real_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_decay.ptr, h_decay.data(), M * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    auto make_ctx = [&]() {
        HEOMContext c{};
        c.hamiltonian = d_H.ptr; c.c_bath = d_c.ptr; c.cbar_bath = d_c.ptr;
        c.coeff_plus = d_coeff.ptr; c.coeff_minus = d_coeff.ptr; c.markov_residue = d_markov.ptr;
        c.neighbor_map_plus = d_map.ptr; c.neighbor_map_minus = d_map.ptr;
        c.level_indices = d_level.ptr; c.total_decay = d_decay.ptr; c.active_flags = nullptr;
        c.d_coupling = d_dc.ptr; c.mode_off = d_mode_off.ptr;
        c.M = M; c.N = N; c.D = D; c.n_bath = n_bath;
        c.sink_enabled = false; c.sink_site = 0; c.gamma_sink = 0.0;
        c.u_list = nullptr; c.u_count = 0; c.batch = 1;
        c.block = heom::ManifoldBlock::full(N);
        return c;
    };

    struct Case { const char* name; heom::ManifoldBlock blk; };
    const Case cases[] = {
        {"(f,e) 91x14 LHC-II ESA", {ng + ne, nf, ng, ne}},
        {"(e,f) 14x91",            {ng, ne, ng + ne, nf}},
        {"(f,f) 91x91 diagonal",   {ng + ne, nf, ng + ne, nf}},
    };
    bool ok = true;
    for (const Case& cs : cases) {
        const heom::ManifoldBlock& b = cs.blk;
        std::vector<complex_t> h_rho(stride, c_make(0.0, 0.0));
        for (int a = 0; a < M; ++a)
            for (int r = 0; r < b.n_row; ++r)
                for (int q = 0; q < b.n_col; ++q) {
                    const double mag = std::pow(10.0, 3.0 - 2.0 * ((a + r + 2 * q) % 4));
                    h_rho[a * NN + (b.row_off + r) * N + b.col_off + q] =
                        c_make(mag * (1.0 + 0.013 * r), mag * (-0.7 + 0.011 * q));
                }
        const int Nb = b.elems();
        const size_t cstride = static_cast<size_t>(M) * Nb;
        const auto h_cmp = compact(h_rho, M, N, b);
        DevBuf<complex_t> d_rho(stride), d_ref(stride), d_cmp(cstride), d_blk(cstride);
        CHK(cudaMemcpy(d_rho.ptr, h_rho.data(), stride * sizeof(complex_t), cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_cmp.ptr, h_cmp.data(), cstride * sizeof(complex_t), cudaMemcpyHostToDevice));
        HEOMContext full = make_ctx();  full.rho_in = d_rho.ptr; full.drho_out = d_ref.ptr;
        HEOMContext blkc = make_ctx();  blkc.rho_in = d_cmp.ptr; blkc.drho_out = d_blk.ptr; blkc.block = b;
        compute_heom_derivatives(full);
        compute_heom_derivatives(blkc);
        CHK(cudaDeviceSynchronize());
        std::vector<complex_t> ref(stride), got(cstride);
        CHK(cudaMemcpy(ref.data(), d_ref.ptr, stride * sizeof(complex_t), cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(got.data(), d_blk.ptr, cstride * sizeof(complex_t), cudaMemcpyDeviceToHost));
        int bad = 0, nonzero = 0;
        for (int a = 0; a < M; ++a)
            for (int r = 0; r < b.n_row; ++r)
                for (int q = 0; q < b.n_col; ++q) {
                    const complex_t x = got[a * Nb + r * b.n_col + q];
                    const complex_t y = ref[a * NN + (b.row_off + r) * N + b.col_off + q];
                    if (c_re(x) != 0.0 || c_im(x) != 0.0) ++nonzero;
                    if (!(c_re(x) == c_re(y) && c_im(x) == c_im(y))) {
                        if (bad < 3)
                            std::cout << "    " << cs.name << " mismatch a=" << a << " r=" << r
                                      << " q=" << q << " got (" << c_re(x) << "," << c_im(x)
                                      << ") ref (" << c_re(y) << "," << c_im(y) << ")\n";
                        ++bad;
                    }
                }
        // A test whose derivative is all zero would pass trivially.
        if (bad || nonzero < static_cast<int>(cstride) / 2) {
            std::cout << "  " << cs.name << ": mismatches=" << bad << " nonzero=" << nonzero
                      << "/" << cstride << "\n";
            ok = false;
        }
    }
    if (ok) std::cout << "disjoint block partial-row H staging == full launch (N=106, 3 blocks, bitwise): PASS\n";
    else    std::cout << "disjoint block partial-row H staging: FAIL\n";
    return ok;
}

// Manifold blocking, the part that is an ARGUMENT rather than an equality.
//
// The block IS the storage, so the workspace a blocked rk4_step uses is
// exactly 3·M·Nb — there is no "outside the block" region any more. What is
// left to test is the ALLOCATED-vs-USED gap: the trajectory allocates 3·B·M·
// Nb_cap and a smaller block uses a prefix of it, with whatever the previous
// (larger) block left behind sitting past the prefix. The design claims the
// kernel never reads past M·Nb of any sub-buffer; the sub-buffers are spaced
// by M·Nb (not by the capacity), so a stale tail can only be read through an
// addressing bug. That is a claim about every read in a 500-line kernel, and
// reviewing it is not the same as testing it — so: allocate the workspace at
// the FULL-matrix capacity, fill all of it with NaN, run on the compact block,
// and see.
//
// NaN is the right poison. Zero would be absorbed by any multiply and prove
// nothing; NaN propagates through every arithmetic path, so a single stray read
// anywhere in four stages lands in the result. DevBuf hands back raw cudaMalloc
// memory, which is neither zeroed nor poisoned, hence both fills are explicit.
//
// MUTATION CHECK (re-run by hand when the addressing changes): make rk4_step's
// per_slot use N² instead of ctx.block.elems(); the NaN run must then produce
// NaN (or garbage) and this test must fail.
bool test_manifold_block_ignores_stale_stage_buffer() {
    auto cfg = HEOMConfig::Config::load_yaml(write_double_excited_dimer_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int M = sys.M(), N = sys.N();
    const double dt = 0.5;

    // (f,e): the block the ESA t-segment actually runs in.
    const int ro = 3, nr = 1, co = 1, nc = 2;
    const heom::ManifoldBlock cblk{ro, nr, co, nc};
    const int Nb = cblk.elems();
    const size_t stride = static_cast<size_t>(M) * Nb;      // compact ρ

    std::vector<complex_t> h_rho(stride, c_make(0.0, 0.0));
    for (int a = 0; a < M; ++a)
        for (int dj = 0; dj < nc; ++dj)
            h_rho[static_cast<size_t>(a) * Nb + dj] =
                c_make(0.4 - 0.05 * a + 0.2 * dj, 0.1 * a - 0.3 * dj);

    // Workspace at full-matrix capacity (what a dynamics-capacity trajectory
    // would hand a blocked step); the step must use only its 3·M·Nb prefix.
    const size_t ws_len = 3 * static_cast<size_t>(M) * N * N;
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
    base.block = cblk;

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
    for (size_t e = 0; e < stride; ++e) {
        if (std::isnan(c_re(nan_run[e])) || std::isnan(c_im(nan_run[e]))) ++n_nan;
        if (!(c_re(nan_run[e]) == c_re(clean[e]) &&
              c_im(nan_run[e]) == c_im(clean[e]))) ++n_diff;
    }
    if (n_nan || n_diff) {
        std::cout << "  poisoned workspace leaked: NaN=" << n_nan
                  << " mismatches vs clean workspace=" << n_diff << "\n";
        ok = false;
    }

    // Guard against the test passing because nothing happened at all: the step
    // has to have moved ρ somewhere inside the block.
    bool moved = false;
    for (size_t e = 0; e < stride && !moved; ++e)
        if (c_re(clean[e]) != c_re(h_rho[e]) || c_im(clean[e]) != c_im(h_rho[e]))
            moved = true;
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

// Per-slot norms: norms[b·M + α] must
// equal ‖ρ_α^(b)‖_max of slot b ALONE — no max over slots. Per α we plant one
// element per slot with a known magnitude, arranged so slot 0 is larger for
// small α and slot 1 for large α — so a kernel that still took the union (or
// read only one slot) gets roughly half the entries of some row wrong. Both
// N=3 (per-thread) and N=10 (warp) kernel paths are covered, full-M and with
// per-slot scan segments (scan_off).
bool test_batch_per_slot_norms() {
    bool all_ok = true;
    const double kTol = heom::kMixedPrecision ? 1e-4 : 1e-12;

    for (const int N : {3, 10}) {   // 9 < 64 <= 100: both norm kernels
        const int M = 10, B = 2;
        const size_t NN = static_cast<size_t>(N) * N;
        const size_t total = static_cast<size_t>(B) * M * NN;

        std::vector<complex_t> h(total, c_make(0.0, 0.0));
        auto mag = [](int b, int a) {
            return (b == 0) ? (0.5 + 0.10 * a)    // slot 0 larger for a < 5
                            : (0.4 + 0.12 * a);   // slot 1 larger for a >= 5
        };
        for (int b = 0; b < B; ++b)
            for (int a = 0; a < M; ++a)
                h[static_cast<size_t>(b) * M * NN + static_cast<size_t>(a) * NN + (a % NN)] =
                    c_make(mag(b, a), 0.0);

        DevBuf<complex_t> d_rho(total);
        DevBuf<double>    d_norms(B * M);
        CHK(cudaMemcpy(d_rho.ptr, h.data(), total * sizeof(complex_t), cudaMemcpyHostToDevice));

        // (1) full M per slot.
        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N, heom::ManifoldBlock::full(N),
                          nullptr, 0, B);
        CHK(cudaDeviceSynchronize());
        std::vector<double> got(B * M);
        CHK(cudaMemcpy(got.data(), d_norms.ptr, B * M * sizeof(double), cudaMemcpyDeviceToHost));
        for (int b = 0; b < B; ++b)
            for (int a = 0; a < M; ++a)
                if (std::abs(got[b * M + a] - mag(b, a)) > kTol) {
                    std::cout << "  per-slot norm N=" << N << " b=" << b << " a=" << a
                              << ": got " << got[b * M + a] << " expect " << mag(b, a) << "\n";
                    all_ok = false;
                }

        // (2) per-slot scan segments: slot 0 scans {1,3}, slot 1 scans {2,4,6};
        // every other entry must keep its poison.
        const std::vector<int> h_list = {1, 3, 2, 4, 6};
        const std::vector<int> h_off  = {0, 2, 5};
        DevBuf<int> d_list(5), d_off(3);
        CHK(cudaMemcpy(d_list.ptr, h_list.data(), 5 * sizeof(int), cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(d_off.ptr, h_off.data(), 3 * sizeof(int), cudaMemcpyHostToDevice));
        std::vector<double> poison(B * M, -1.0);
        CHK(cudaMemcpy(d_norms.ptr, poison.data(), B * M * sizeof(double), cudaMemcpyHostToDevice));
        compute_ado_norms(d_rho.ptr, d_norms.ptr, M, N, heom::ManifoldBlock::full(N),
                          d_list.ptr, /*scan_count = longest segment*/3, B, d_off.ptr);
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(got.data(), d_norms.ptr, B * M * sizeof(double), cudaMemcpyDeviceToHost));
        for (int b = 0; b < B; ++b)
            for (int a = 0; a < M; ++a) {
                bool in = false;
                for (int k = h_off[b]; k < h_off[b + 1]; ++k) in |= (h_list[k] == a);
                const double want = in ? mag(b, a) : -1.0;
                if (std::abs(got[b * M + a] - want) > kTol) {
                    std::cout << "  per-slot segment norm N=" << N << " b=" << b << " a=" << a
                              << ": got " << got[b * M + a] << " expect " << want << "\n";
                    all_ok = false;
                }
            }
    }

    if (all_ok) std::cout << "batch per-slot norms (no union, both kernels, segments): PASS\n";
    else        std::cout << "batch per-slot norms: FAIL\n";
    return all_ok;
}

// Fused norm (kFinal epilogue). The bits the final stage folds into
// norm_sq_bits must be exactly the |z|² max the standalone compute_ado_norms
// takes the sqrt of: sqrt(fused) == standalone bitwise, per ADO, as a union
// over B=2 slots that differ, on the full launch and on a compact u_list
// launch (unlisted ADOs stay at the zeroed value), for plain and IF RK4, and
// with a NaN planted in one slot (it must stick).
bool test_fused_final_norms() {
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int M = sys.M(), N = sys.N(), B = 2;
    const size_t stride = static_cast<size_t>(M) * N * N;
    const double dt = 0.5;

    std::vector<complex_t> h_rho(B * stride);
    for (int b = 0; b < B; ++b)
        for (int a = 0; a < M; ++a)
            for (int e = 0; e < N * N; ++e) {
                const double re = 0.07 * (a + 1) * (b ? 1.3 : 0.6) + 0.19 * e - 0.05 * b;
                const double im = 0.03 * (a + 1) - 0.09 * e * (b ? -1.0 : 1.0);
                h_rho[b * stride + static_cast<size_t>(a) * N * N + e] = c_make(re, im);
            }

    DevBuf<complex_t> d_eh(M), d_ef(M);
    compute_if_factors(sys.d_total_decay(), dt, M, d_eh.ptr, d_ef.ptr);

    // Compact launch over every other ADO (α=0 included).
    std::vector<int> h_u;
    for (int a = 0; a < M; a += 2) h_u.push_back(a);
    DevBuf<int> d_u(h_u.size());
    CHK(cudaMemcpy(d_u.ptr, h_u.data(), h_u.size() * sizeof(int), cudaMemcpyHostToDevice));

    bool ok = true;
    for (const bool use_if : {false, true})
    for (const bool compact : {false, true})
    for (const bool plant_nan : {false, true}) {
        std::vector<complex_t> h = h_rho;
        if (plant_nan)   // ADO 2 is in both launch sets; plant in slot 1
            h[stride + 2 * N * N + 1] = c_make(std::numeric_limits<double>::quiet_NaN(), 0.0);
        DevBuf<complex_t> d_rho(B * stride), d_ws(3 * B * stride);
        // Per-slot fused rows [B×M] (batch-slot-flat layout).
        DevBuf<unsigned long long> d_bits(static_cast<size_t>(B) * M);
        DevBuf<double> d_ref(static_cast<size_t>(B) * M);
        CHK(cudaMemcpy(d_rho.ptr, h.data(), B * stride * sizeof(complex_t), cudaMemcpyHostToDevice));
        CHK(cudaMemset(d_bits.ptr, 0, static_cast<size_t>(B) * M * sizeof(unsigned long long)));
        HEOMContext c = build_multi_ado_ctx(sys);
        c.rho_in = d_rho.ptr; c.batch = B;
        if (compact) { c.u_list = d_u.ptr; c.u_count = static_cast<int>(h_u.size()); }
        if (use_if) rk4_step_if(c, dt, d_ws.ptr, nullptr, d_eh.ptr, d_ef.ptr, d_bits.ptr);
        else        rk4_step(c, dt, d_ws.ptr, nullptr, d_bits.ptr);
        compute_ado_norms(d_rho.ptr, d_ref.ptr, M, N, heom::ManifoldBlock::full(N),
                          nullptr, 0, B);
        CHK(cudaDeviceSynchronize());
        std::vector<unsigned long long> bits(static_cast<size_t>(B) * M);
        std::vector<double> ref(static_cast<size_t>(B) * M);
        CHK(cudaMemcpy(bits.data(), d_bits.ptr, bits.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(ref.data(), d_ref.ptr, ref.size() * sizeof(double), cudaMemcpyDeviceToHost));
        int n_nan = 0;
        for (int b = 0; b < B; ++b) {
            for (int a = 0; a < M; ++a) {
                const size_t at = static_cast<size_t>(b) * M + a;
                double sq; std::memcpy(&sq, &bits[at], sizeof(double));
                const bool launched = !compact || a % 2 == 0;
                const double got = std::sqrt(sq);
                bool good;
                if (!launched)               good = (bits[at] == 0ull);
                else if (std::isnan(ref[at])) good = std::isnan(got);
                else good = std::memcmp(&got, &ref[at], sizeof(double)) == 0;
                if (launched && std::isnan(got)) ++n_nan;
                if (!good) {
                    std::cout << "  fused norm if=" << use_if << " compact=" << compact
                              << " nan=" << plant_nan << " b=" << b << " a=" << a
                              << ": got " << got << " ref " << ref[at] << "\n";
                    ok = false;
                }
            }
        }
        if (plant_nan && n_nan == 0) {
            std::cout << "  planted NaN did not reach the fused norm\n";
            ok = false;
        }
    }
    if (ok) std::cout << "fused kFinal norms == compute_ado_norms per-slot (bitwise): PASS\n";
    else    std::cout << "fused kFinal norms: FAIL\n";
    return ok;
}


bool test_batch_alloc_plan() {
    const size_t M = 100;
    const int N = 4, D = 8, nb = 4, L = 3;   // 4 baths x (Drude + 1 pole)
    const size_t Nb = static_cast<size_t>(N) * N;
    const auto p1 = heom::plan_gpu_buffers(M, N, D, nb, L, /*use_filtering=*/true, 1, Nb);
    const auto p4 = heom::plan_gpu_buffers(M, N, D, nb, L, /*use_filtering=*/true, 4, Nb);
    // rho_elems sizes d_rho / d_workspace per ADO; a smaller block shrinks
    // exactly those two and nothing else (d_hamiltonian still goes by N).
    const auto pb = heom::plan_gpu_buffers(M, N, D, nb, L, /*use_filtering=*/true, 1, 6);

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
    expect("d_rho(N^2)", count(p1, "d_rho"), M * Nb);
    expect("d_rho(Nb=6)", count(pb, "d_rho"), M * 6);
    expect("d_workspace(Nb=6)", count(pb, "d_workspace"), 3 * M * 6);
    expect("d_hamiltonian(Nb=6)", count(pb, "d_hamiltonian"), Nb);
    // Out-of-range rho_elems is a programming error, not a silent clamp.
    bool threw = false;
    try { (void)heom::plan_gpu_buffers(M, N, D, nb, L, true, 1, 0); }
    catch (const std::invalid_argument&) { threw = true; }
    expect("rho_elems=0 throws", threw ? 1 : 0, 1);
    threw = false;
    try { (void)heom::plan_gpu_buffers(M, N, D, nb, L, true, 1, Nb + 1); }
    catch (const std::invalid_argument&) { threw = true; }
    expect("rho_elems>N^2 throws", threw ? 1 : 0, 1);

    // Per-slot filter rows (one active set per slot) scale with batch too.
    for (const char* nm : {"d_ado_norms", "d_active_flags"})
        expect(nm, count(p4, nm), 4 * count(p1, nm));

    // Everything else is a single shared copy — invariant under batch.
    for (const char* nm : {"d_hamiltonian", "d_c_bath", "d_markov_residue",
                           "d_plus_map", "d_minus_map", "d_level_indices",
                           "d_coeff_plus", "d_coeff_minus", "d_total_decay",
                           "d_d_coupling", "d_level_counts"}) {
        expect(nm, count(p4, nm), count(p1, nm));
    }

    if (ok) std::cout << "batch alloc plan (d_rho/d_workspace/filter rows scale): PASS\n";
    else    std::cout << "batch alloc plan: FAIL\n";
    return ok;
}

// Task A4: variable active batch width on HEOMTrajectory. Two properties:
//  (A) active=1 on a batch=4 trajectory == a batch=1 trajectory, bit-for-bit
//      (reduced-width launch + set_initial at width 1 are physics-correct).
//  (B) narrowing 4→2 FREEZES slots 2/3: after propagating at active=4, a
//      second propagate at active=2 must leave slots 2/3 untouched. The width
//      is a property of ρ, so the narrowing goes through the production
//      path that does it — restore_device of a B=2 snapshot (set_initial would
//      zero the whole buffer and prove nothing). This is the idempotency-
//      trap catcher — if the width setter failed to update ctx_.batch, the
//      launch would stay grid.y=4 and advance slots 2/3 (they hold distinct
//      nonzero state), so the freeze check would fail.
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
        HEOMTrajectory t1(sys, 1);
        t1.set_initial({make_rho(0.0)}, heom::ManifoldBlock::full(N));
        t1.propagate_steps(nsteps, /*check_div=*/false);
        ref0 = t1.slot_rho0(0);
    }

    // (A) active=1 on a batch=4 trajectory must match the B=1 reference. Same
    // kernel arithmetic + order ⇒ bit-exact (FP64 and mixed); tol only guards
    // against a genuine precision surprise.
    const double tolA = heom::kMixedPrecision ? kCarryTol : 0.0;
    bool okA = true;
    {
        HEOMTrajectory t4(sys, 4);
        t4.set_initial({make_rho(0.0)}, heom::ManifoldBlock::full(N));
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
        HEOMTrajectory t4(sys, 4);
        DeviceHierarchySnapshot snap2;                     // a B=2 state to narrow to
        t4.set_initial({make_rho(5.0), make_rho(6.0)}, heom::ManifoldBlock::full(N));
        t4.snapshot_device(snap2);
        t4.set_initial({make_rho(0.0), make_rho(1.0),
                              make_rho(2.0), make_rho(3.0)}, heom::ManifoldBlock::full(N));
        t4.propagate_steps(nsteps, /*check_div=*/false);  // ctx built at B=4
        const Eigen::MatrixXcd s2b = t4.slot_rho0(2), s3b = t4.slot_rho0(3);
        t4.restore_device(snap2);                          // width 4 -> 2, slots 2/3 untouched
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

// Task A5: batched snapshot/restore + expand_slots. Snapshots are verbatim
// copies, so every check is bit-exact.
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

    // (a) snapshot(3) → perturb → set_initial at width 1 → restore → back to 3 slots.
    bool okA = true;
    {
        HEOMTrajectory t(sys, 4);
        t.set_initial({make_rho(0.0), make_rho(1.0), make_rho(2.0)}, heom::ManifoldBlock::full(N));
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
        t.set_initial({make_rho(9.0)}, heom::ManifoldBlock::full(N));  // width 3 -> 1
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
        HEOMTrajectory t(sys, 6);
        t.set_initial({make_rho(0.0), make_rho(1.0), make_rho(2.0)}, heom::ManifoldBlock::full(N));
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

// A snapshot is replayed only on the SystemContext that took it, at
// a width the target holds, through an in-range fan-out map — and a refused
// call leaves the target exactly as it was. sysB is built from the SAME yaml
// as sysA: structurally identical, still refused (identity, not shape).
// Comparing only N would let a snapshot from a context with a larger M
// write past the target's ρ, and a bad map must throw before ρ is
// zeroed and the width changed.
bool test_snapshot_replayed_only_where_taken() {
    auto cfg = HEOMConfig::Config::load_yaml(write_carry_through_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sysA(cfg), sysB(cfg);
    const int N = sysA.N();
    const auto full = heom::ManifoldBlock::full(N);
    auto make_rho = [&](double seed) {
        Eigen::MatrixXcd r(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r(i, j) = std::complex<double>(0.1 * (i + 1) + 0.2 * j + seed,
                                               0.05 * i - 0.1 * j + 0.3 * seed);
        return r;
    };

    HEOMTrajectory a(sysA, 4);
    a.set_initial({make_rho(0.0), make_rho(1.0), make_rho(2.0)}, full);
    a.propagate_steps(2, /*check_div=*/false);
    DeviceHierarchySnapshot snap;
    a.snapshot_device(snap);

    bool ok = true;
    auto refused = [&](const char* what, HEOMTrajectory& t, auto&& fn) {
        const int w = t.active_batch();
        std::vector<Eigen::MatrixXcd> before;
        for (int b = 0; b < w; ++b) before.push_back(t.slot_rho0(b));
        bool threw = false;
        try { fn(); } catch (const std::invalid_argument&) { threw = true; }
        bool same = t.active_batch() == w;
        for (int b = 0; same && b < w; ++b)
            same = (t.slot_rho0(b) - before[b]).cwiseAbs().maxCoeff() == 0.0;
        if (!threw || !same) {
            std::cout << "  " << what << ":" << (threw ? "" : " no invalid_argument;")
                      << (same ? "" : " target modified") << "\n";
            ok = false;
        }
    };

    HEOMTrajectory b(sysB, 4);
    b.set_initial({make_rho(5.0)}, full);
    b.propagate_steps(1, /*check_div=*/false);
    refused("restore from another context", b, [&] { b.restore_device(snap); });
    refused("expand from another context", b, [&] { b.expand_slots(snap, {0, 1}); });

    HEOMTrajectory narrow(sysA, 2);
    narrow.set_initial({make_rho(5.0)}, full);
    refused("restore 3 slots into a batch-2 trajectory", narrow,
            [&] { narrow.restore_device(snap); });

    refused("expand with child_to_parent out of range", a,
            [&] { a.expand_slots(snap, {0, 3}); });
    DeviceHierarchySnapshot never;
    refused("restore a never-filled snapshot", a, [&] { a.restore_device(never); });

    // The legal replay: same context, another trajectory, no set_initial.
    HEOMTrajectory c(sysA, 4);
    c.restore_device(snap);
    for (int s = 0; s < 3; ++s)
        if ((c.slot_rho0(s) - a.slot_rho0(s)).cwiseAbs().maxCoeff() != 0.0) {
            std::cout << "  restore into a sibling trajectory: slot " << s << " differs\n";
            ok = false;
        }

    if (ok) std::cout << "snapshot replayed only where taken, refusals touch nothing: PASS\n";
    else    std::cout << "snapshot identity / refusal: FAIL\n";
    return ok;
}

// FMO-7 (the production Hamiltonian) with the filter and U-list on, IF-RK4 at
// 10 fs.
std::string write_fmo_filtered_yaml(int L, const std::string& integrator = "if-rk4",
                                    const std::string& ulist = "0.9",
                                    const std::string& tolerance = "1e-6",
                                    bool filter_on = true) {
    const std::string path = heom_test::tmp("test_fmo_filtered_L" + std::to_string(L) + "_"
                                            + integrator + "_u" + ulist + "_t" + tolerance
                                            + (filter_on ? "" : "_off") + ".yaml");
    std::ofstream f(path);
    f << "system:\n  n_states: 7\n  basis: excited_only\n  temperature: 77.0\n  hamiltonian:\n"
         "    - [12415.0,  -87.7,    5.5,   -5.9,    6.7,  -13.7,   -9.9]\n"
         "    - [ -87.7, 12479.7,   30.8,    8.2,    0.7,   11.8,    4.3]\n"
         "    - [   5.5,   30.8, 12134.1,  -53.5,   -2.2,   -9.6,    6.0]\n"
         "    - [  -5.9,    8.2,  -53.5, 12317.4,  -70.7,  -17.0,  -63.3]\n"
         "    - [   6.7,    0.7,   -2.2,  -70.7, 12445.0,   81.1,   -1.3]\n"
         "    - [ -13.7,   11.8,   -9.6,  -17.0,   81.1, 12610.0,   39.7]\n"
         "    - [  -9.9,    4.3,    6.0,  -63.3,   -1.3,   39.7, 12417.0]\n"
         "  w0: 12300.0\n"
         "bath:\n  eta: 70.0\n  gamma_inverse: 100.0\n"
         "  expansion_method: pade\n  n_extra_poles: 1\n  L_cutoff: " << L << "\n"
         "filter:\n  enabled: " << (filter_on ? "true" : "false") << "\n"
         "  tolerance: " << tolerance << "\n  ulist: " << ulist << "\n"
         "dynamics:\n  integrator: " << integrator << "\n  dt: 10.0\n  n_steps: 1\n"
         "  output_interval: 1\n"
         "  print_interval: 0\n  initial_state:\n    type: site_excitation\n    excited_site: 1\n"
         "  output:\n    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
         "    filename: fmo_filtered\n    use_timestamp: false\n";
    return path;
}

// restore(snapshot()) is the identity. A trajectory that before EVERY
// step is snapshotted, sent down a branch (two steps and a dipole) and
// restored must end exactly where the uninterrupted one does — the 2DES τ and
// T chains at their production sampling. L=4/5 is where the filter really
// prunes; the lossy snapshot (active ADOs only, the rest zeroed on restore)
// ended 7.8e-9 / 1.6e-8 off. Exact in both builds: the mixed build's Kahan
// carry is part of the snapshot, and restarting it at zero instead cost
// 6e-8 / 1.9e-7 — more than the build's own 3.7e-8 error against FP64.
bool test_snapshot_restore_is_identity() {
    using namespace heom::spectra;
    const int n_steps = 200;
    bool ok = true;
    for (const int L : {4, 5}) {
        auto cfg = HEOMConfig::Config::load_yaml(write_fmo_filtered_yaml(L));
        HEOMConfig::resolve_dynamics_config(cfg);
        SystemContext sys(cfg);
        const int N = sys.N();
        const auto full = heom::ManifoldBlock::full(N);
        Eigen::MatrixXcd rho0 = Eigen::MatrixXcd::Zero(N, N);
        rho0(0, 0) = 1.0;
        const Eigen::MatrixXcd mu = Eigen::MatrixXcd::Ones(N, N);

        HEOMTrajectory cont(sys), tree(sys);
        cont.set_initial({rho0}, full);
        tree.set_initial({rho0}, full);
        DeviceHierarchySnapshot snap;
        for (int s = 0; s < n_steps; ++s) {
            tree.snapshot_device(snap);
            tree.propagate_steps(2, /*check_div=*/false);
            tree.apply_dipole({mu}, DipoleSide::Left, full);
            tree.restore_device(snap);
            cont.propagate_steps(1, /*check_div=*/false);
            tree.propagate_steps(1, /*check_div=*/false);
        }
        const double d = (cont.slot_rho0(0) - tree.slot_rho0(0)).cwiseAbs().maxCoeff();
        std::cout << "  L=" << L << " M=" << sys.M() << ": max|rho0(continuous) - "
                  << "rho0(restored every step)| = " << d << "\n";
        if (d != 0.0) ok = false;
    }
    if (ok) std::cout << "snapshot/restore is the identity (FMO-7 L=4/5, filter + U-list on): PASS\n";
    else    std::cout << "snapshot/restore identity: FAIL (must be bit-exact)\n";
    return ok;
}

// At tolerance → 0 the filter must reproduce filter-off bit for bit, the
// whole hierarchy and the Kahan carry included. FMO-7 L=5 takes three full
// steps to fill from a site excitation, so a fill window of one step (the old
// fresh_state_) or any fixed two leaves exact zeros behind; they went inactive
// and the neighbor gate throttled the growth: 5e-7 off at IF dt=10 (the 3-site
// L=2 2DES gate fills in one step and never saw it). The 2DES side is
// test_vanishing_tolerance_fmo_2des in test_2des_reuse.
bool test_vanishing_tolerance_fmo_dynamics() {
    using namespace heom::spectra;
    const int L = 5, n_steps = 30;
    auto run = [&](bool on) {
        auto cfg = HEOMConfig::Config::load_yaml(
            write_fmo_filtered_yaml(L, "if-rk4", "0.9", "1e-300", on));
        HEOMConfig::resolve_dynamics_config(cfg);
        SystemContext sys(cfg);
        const int N = sys.N();
        Eigen::MatrixXcd rho0 = Eigen::MatrixXcd::Zero(N, N);
        rho0(0, 0) = 1.0;
        HEOMTrajectory t(sys);
        t.set_initial({rho0}, heom::ManifoldBlock::full(N));
        t.propagate_steps(n_steps, /*check_div=*/false);
        DeviceHierarchySnapshot s;
        t.snapshot_device(s);
        std::vector<char> h(s.rho.size() * sizeof(heom::complex_t)
                            + s.kahan.size() * sizeof(heom::complex_t));
        CHK(cudaMemcpy(h.data(), s.rho.get(), s.rho.size() * sizeof(heom::complex_t),
                       cudaMemcpyDeviceToHost));
        if (s.kahan.size() > 0)
            CHK(cudaMemcpy(h.data() + s.rho.size() * sizeof(heom::complex_t), s.kahan.get(),
                           s.kahan.size() * sizeof(heom::complex_t), cudaMemcpyDeviceToHost));
        return h;
    };
    const bool same = run(false) == run(true);
    if (same) std::cout << "tolerance 1e-300 == filter off (FMO-7 L=5, IF dt=10, "
                           "whole hierarchy): PASS\n";
    else      std::cout << "tolerance 1e-300 vs filter off (FMO-7 L=5): FAIL "
                           "(must be byte-identical)\n";
    return same;
}

// The launch domain is a speed choice, never a result. ulist 1.0 runs every
// step compact, ulist 0 every step on the full grid, ulist 0.01 reaches the
// full grid through the adaptive gate. All three must leave the whole
// hierarchy (and the mixed build's Kahan carry) byte-identical. The full grid
// used to rescale the ADOs outside U (and their carry) by E2 under IF-RK4; the
// compact launch never touched them. Plain RK4 adds an exact zero there, so it
// agreed already — kept as a guard.
bool test_dense_launch_equals_compact() {
    using namespace heom::spectra;
    const int L = 5, n_steps = 100;
    bool ok = true;
    for (const std::string integ : {"if-rk4", "rk4"}) {
        std::vector<std::vector<char>> rho, kahan;
        for (const std::string u : {"1.0", "0", "0.01"}) {
            auto cfg = HEOMConfig::Config::load_yaml(write_fmo_filtered_yaml(L, integ, u));
            HEOMConfig::resolve_dynamics_config(cfg);
            SystemContext sys(cfg);
            const int N = sys.N();
            const auto full = heom::ManifoldBlock::full(N);
            Eigen::MatrixXcd rho0 = Eigen::MatrixXcd::Zero(N, N);
            rho0(0, 0) = 1.0;
            HEOMTrajectory t(sys);
            t.set_initial({rho0}, full);
            t.propagate_steps(n_steps, /*check_div=*/false);
            DeviceHierarchySnapshot s;
            t.snapshot_device(s);
            auto host = [](const GPUMemory<heom::complex_t>& d) {
                std::vector<char> h(d.size() * sizeof(heom::complex_t));
                if (!h.empty())
                    CHK(cudaMemcpy(h.data(), d.get(), h.size(), cudaMemcpyDeviceToHost));
                return h;
            };
            rho.push_back(host(s.rho));
            kahan.push_back(host(s.kahan));
        }
        for (size_t k = 1; k < rho.size(); ++k) {
            const bool same = rho[k] == rho[0] && kahan[k] == kahan[0];
            std::cout << "  " << integ << " L=" << L << ": full launch (" << (k == 1 ? "ulist 0" : "gate")
                      << ") vs compact: " << (same ? "byte-identical" : "DIFFERS") << "\n";
            if (!same) ok = false;
        }
    }
    if (ok) std::cout << "dense launch == compact launch (FMO-7, filter on, IF + plain): PASS\n";
    else    std::cout << "dense launch == compact launch: FAIL (must be byte-identical)\n";
    return ok;
}

// Uncoupled dimer, rk4 at dt = 8 fs: the 2000 cm⁻¹ coherence lies outside
// RK4's stability region and grows ~1.5× per step, while a diagonal ρ0
// commutes with H and with every V, so its α=0 block never moves.
std::string write_unstable_coherence_yaml() {
    const std::string path = heom_test::tmp("test_unstable_coherence.yaml");
    std::ofstream f(path);
    f << "system:\n  n_states: 2\n  basis: excited_only\n  temperature: 77.0\n"
         "  hamiltonian:\n    - [14000.0, 0.0]\n    - [0.0, 12000.0]\n  w0: 12000.0\n"
         "bath:\n  eta: 35.0\n  gamma_inverse: 100.0\n  expansion_method: matsubara\n"
         "  n_extra_poles: 0\n  L_cutoff: 2\n"
         "filter:\n  enabled: false\n"
         "dynamics:\n  integrator: rk4\n  dt: 8.0\n  n_steps: 1\n  output_interval: 1\n"
         "  print_interval: 0\n  initial_state:\n    type: site_excitation\n    excited_site: 1\n"
         "  output:\n    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
         "    filename: unstable_coherence\n    use_timestamp: false\n";
    return path;
}

// Slot b's divergence ceiling is 10·‖ρ_0^(b)‖ at its last
// set_initial / dipole, whatever happened to the slot since. X (the unstable
// coherence) must alarm on the same step in every arrangement:
//   batched next to Y = 10·|1><1| — the shared ceiling (10 × max over slots)
//     let X grow ~7× further unnoticed;
//   snapshot + restore two steps before the alarm — restore used to re-base
//     on the current norm;
//   expand {1, 0} from a {X, Y} snapshot — expand used to give every child
//     the parent's max.
bool test_divergence_baseline_per_slot() {
    auto cfg = HEOMConfig::Config::load_yaml(write_unstable_coherence_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const auto full = heom::ManifoldBlock::full(2);
    Eigen::MatrixXcd X = Eigen::MatrixXcd::Zero(2, 2), Y = Eigen::MatrixXcd::Zero(2, 2);
    X(0, 1) = X(1, 0) = 1.0;
    Y(0, 0) = 10.0;

    auto first_alarm = [&](HEOMTrajectory& t, int restore_at) {
        DeviceHierarchySnapshot s;
        for (int k = 1; k <= 60; ++k) {
            t.propagate_steps(1);
            if (t.diverged()) return k;
            if (k == restore_at) { t.snapshot_device(s); t.restore_device(s); }
        }
        return -1;
    };

    HEOMTrajectory solo(sys, 1), calm(sys, 1), both(sys, 2), again(sys, 1), fan(sys, 2);
    solo.set_initial({X}, full);
    const int ref = first_alarm(solo, -1);
    calm.set_initial({Y}, full);
    const int k_calm = first_alarm(calm, -1);
    both.set_initial({X, Y}, full);
    const int k_batch = first_alarm(both, -1);
    again.set_initial({X}, full);
    const int k_restore = first_alarm(again, ref - 2);
    fan.set_initial({X, Y}, full);
    DeviceHierarchySnapshot parent;
    fan.snapshot_device(parent);
    fan.expand_slots(parent, {1, 0});
    const int k_expand = first_alarm(fan, -1);

    const bool ok = ref >= 4 && k_calm == -1 && k_batch == ref && k_restore == ref
                 && k_expand == ref;
    std::cout << "  first alarm: X alone " << ref << ", Y alone " << k_calm
              << " (must be none), batched " << k_batch << ", restored " << k_restore
              << ", expanded " << k_expand << "\n";
    if (ok) std::cout << "divergence baseline per slot (batched / restore / expand == alone): PASS\n";
    else    std::cout << "divergence baseline per slot: FAIL\n";
    return ok;
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
    const size_t ns = 3;

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
        HEOMTrajectory t1(sys, 1);
        heom::Detector d1(t1);
        t1.set_initial({rhos[b]}, heom::ManifoldBlock::full(N));
        d1.begin({mus[b]}, ns);
        d1.sample(0);
        t1.propagate_steps(1, /*check_div=*/false);
        d1.sample(1);
        t1.propagate_steps(1, /*check_div=*/false);
        d1.sample(2);
        d1.flush(&ref[b * ns]);
    }

    // Batched: one B=3 trajectory, same sequence.
    std::vector<std::complex<double>> out(3 * ns);
    {
        HEOMTrajectory t3(sys, 3);
        heom::Detector d3(t3);
        t3.set_initial({rhos[0], rhos[1], rhos[2]}, heom::ManifoldBlock::full(N));
        d3.begin({mus[0], mus[1], mus[2]}, ns);
        d3.sample(0);
        t3.propagate_steps(1, /*check_div=*/false);
        d3.sample(1);
        t3.propagate_steps(1, /*check_div=*/false);
        d3.sample(2);
        d3.flush(out.data());
        if (d3.batch() != 3 || d3.n_samples() != ns) {
            std::cout << "  Detector batch()=" << d3.batch()
                      << " n_samples()=" << d3.n_samples()
                      << " expected 3, " << ns << "\n";
            return false;
        }
    }

    const double tol = heom::kMixedPrecision ? kCarryTol : 0.0;
    bool ok = true;
    for (size_t k = 0; k < 3 * ns; ++k) {
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

// Task A7: per-slot dipole. apply_dipole with distinct μ per slot must equal
// three independent B=1 apply_dipole calls. Covers both sides (left μ·ρ /
// right ρ·μ). Independent slots ⇒ bit-exact vs B=1.
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
            HEOMTrajectory t1(sys, 1);
            t1.set_initial({rhos[b]}, heom::ManifoldBlock::full(N));
            t1.apply_dipole({mus[b]}, side, heom::ManifoldBlock::full(N));
            ref[b] = t1.slot_rho0(0);
        }
        HEOMTrajectory t3(sys, 3);
        t3.set_initial({rhos[0], rhos[1], rhos[2]}, heom::ManifoldBlock::full(N));
        t3.apply_dipole({mus[0], mus[1], mus[2]}, side, heom::ManifoldBlock::full(N));
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
    HEOMTrajectory t4(sys, 4);  // allocated batch_=4
    t4.set_initial({make_rho(0.0), make_rho(1.0), make_rho(2.0)}, heom::ManifoldBlock::full(N));

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
    t4.set_initial({make_rho(0.0)}, heom::ManifoldBlock::full(N));
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
    const std::string path = heom_test::tmp_dir() + "/test_if_rk4_L" + std::to_string(L_cutoff) + ".yaml";
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
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
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

        DevBuf<complex_t> d_eh(M), d_ef(M);
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

        DevBuf<complex_t> d_eh(M), d_ef(M);
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


// 2.2.1: HDF5 system_bath_energy round-trip + attrs (filter off).
bool test_esb_hdf5_readback() {
    using heom::hdf5::RunStatus;
    auto cfg = HEOMConfig::Config::load_yaml(write_tiny_trajectory_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    // Need L>=1 so first-tier exists; rewrite L via a dedicated yaml.
    const std::string path_yaml = heom_test::tmp_dir() + "/test_esb_hdf5.yaml";
    {
        std::ofstream f(path_yaml);
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
          << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
          << "    filename: \"test_esb_hdf5\"\n"
          << "    use_timestamp: false\n";
    }
    cfg = HEOMConfig::Config::load_yaml(path_yaml);
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int N = sys.N();
    const int n_bath = sys.n_bath();
    Eigen::MatrixXcd rho0 = build_initial_state(sys, *cfg.initial);
    HEOMTrajectory traj(sys);
    traj.set_initial({rho0}, heom::ManifoldBlock::full(N));
    traj.propagate_steps(1);
    std::vector<std::complex<double>> rho_flat(static_cast<size_t>(N) * N);
    traj.read_alpha0(rho_flat);
    std::vector<std::complex<double>> esb;
    traj.read_system_bath_energy(esb);
    if (static_cast<int>(esb.size()) != n_bath) {
        std::cerr << "FAIL(esb_hdf5): esb size " << esb.size() << " != n_bath " << n_bath << "\n";
        return false;
    }

    const std::string h5path = heom_test::tmp_dir() + "/test_esb_hdf5.h5";
    std::remove(h5path.c_str());
    {
        HDF5Writer w(false);
        w.open(sys, rho0, /*use_filtering=*/false, 0.0);
        w.write_timestep(0.0, rho_flat.data(), N, esb.data(), n_bath);
        w.close(0.1, RunStatus::Complete);
    }

    hid_t file = H5Fopen(h5path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "FAIL(esb_hdf5): cannot open " << h5path << "\n";
        return false;
    }
    hid_t dset = H5Dopen2(file, "/results/system_bath_energy", H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "FAIL(esb_hdf5): missing /results/system_bath_energy\n";
        H5Fclose(file);
        return false;
    }
    auto read_str_attr = [](hid_t loc, const char* name) -> std::string {
        if (H5Aexists(loc, name) <= 0) return "<absent>";
        hid_t a = H5Aopen(loc, name, H5P_DEFAULT);
        hid_t ty = H5Aget_type(a);
        char* v = nullptr;
        H5Aread(a, ty, &v);
        std::string out = v ? v : "";
        H5free_memory(v);
        H5Tclose(ty);
        H5Aclose(a);
        return out;
    };
    const std::string units = read_str_attr(dset, "units");
    const std::string phys  = read_str_attr(dset, "physical_part");
    const std::string axis1 = read_str_attr(dset, "axis1");
    const std::string inac  = read_str_attr(dset, "includes_inactive_ados");
    const std::string sem   = read_str_attr(dset, "filter_semantics");
    if (units != "Eh" || phys != "real" || axis1 != "bath_index"
        || inac != "false"
        || sem != "physical_system_bath_interaction_energy") {
        std::cerr << "FAIL(esb_hdf5): attrs units=" << units << " physical_part=" << phys
                  << " axis1=" << axis1 << " includes_inactive_ados=" << inac
                  << " filter_semantics=" << sem << "\n";
        H5Dclose(dset);
        H5Fclose(file);
        return false;
    }

    hid_t space = H5Dget_space(dset);
    if (space < 0) {
        std::cerr << "FAIL(esb_hdf5): H5Dget_space failed\n";
        H5Dclose(dset);
        H5Fclose(file);
        return false;
    }
    const int ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    H5Sclose(space);
    if (ndims != 2 || dims[0] < 1 || static_cast<int>(dims[1]) != n_bath) {
        std::cerr << "FAIL(esb_hdf5): unexpected shape ndims=" << ndims
                  << " dims=[" << dims[0] << "," << dims[1] << "]\n";
        H5Dclose(dset);
        H5Fclose(file);
        return false;
    }
    hid_t ctype = heom::hdf5::make_complex_compound_type();
    std::vector<std::complex<double>> got(static_cast<size_t>(dims[0]) * dims[1]);
    if (H5Dread(dset, ctype, H5S_ALL, H5S_ALL, H5P_DEFAULT, got.data()) < 0) {
        std::cerr << "FAIL(esb_hdf5): H5Dread failed\n";
        H5Tclose(ctype);
        H5Dclose(dset);
        H5Fclose(file);
        return false;
    }
    H5Tclose(ctype);
    H5Dclose(dset);
    H5Fclose(file);

    // Compare the first written record (we only wrote one timestep).
    for (int i = 0; i < n_bath; ++i) {
        if (std::abs(got[static_cast<size_t>(i)] - esb[i]) > 1e-14) {
            std::cerr << "FAIL(esb_hdf5): bath " << i << " wrote " << esb[i]
                      << " read " << got[i] << "\n";
            return false;
        }
        if (!std::isfinite(got[i].real()) || !std::isfinite(got[i].imag())) {
            std::cerr << "FAIL(esb_hdf5): non-finite at bath " << i << "\n";
            return false;
        }
    }
    std::remove(h5path.c_str());
    std::cout << "esb HDF5 readback + attrs (filter off): PASS\n";
    return true;
}

// 2.2.1: DOUBLE_EXCITED must not throw; ESB finite and matches host rebuild.
bool test_esb_double_excited_finite() {
    auto cfg = HEOMConfig::Config::load_yaml(write_double_excited_dimer_yaml());
    HEOMConfig::resolve_dynamics_config(cfg);
    SystemContext sys(cfg);
    const int N = sys.N();
    const int n_bath = sys.n_bath();
    if (N != 4 || n_bath != 2) {
        std::cerr << "FAIL(esb_dex): unexpected N=" << N << " n_bath=" << n_bath << "\n";
        return false;
    }
    if (sys.first_tier().empty()) {
        std::cerr << "FAIL(esb_dex): empty first_tier\n";
        return false;
    }
    Eigen::MatrixXcd rho0 = build_initial_state(sys, *cfg.initial);
    HEOMTrajectory traj(sys);
    traj.set_initial({rho0}, heom::ManifoldBlock::full(N));
    traj.propagate_steps(5);

    std::vector<std::complex<double>> esb;
    try {
        traj.read_system_bath_energy(esb);
    } catch (const std::exception& e) {
        std::cerr << "FAIL(esb_dex): read_system_bath_energy threw: " << e.what() << "\n";
        return false;
    }
    if (static_cast<int>(esb.size()) != n_bath) {
        std::cerr << "FAIL(esb_dex): size " << esb.size() << "\n";
        return false;
    }
    for (int n = 0; n < n_bath; ++n) {
        if (!std::isfinite(esb[n].real()) || !std::isfinite(esb[n].imag())) {
            std::cerr << "FAIL(esb_dex): non-finite E[" << n << "]=" << esb[n] << "\n";
            return false;
        }
    }
    // Second readout must match bitwise (no hidden scratch corruption).
    std::vector<std::complex<double>> esb2;
    traj.read_system_bath_energy(esb2);
    for (int n = 0; n < n_bath; ++n) {
        if (esb2[n] != esb[n]) {
            std::cerr << "FAIL(esb_dex): non-deterministic readout at bath " << n << "\n";
            return false;
        }
    }
    // Spot-check: d_coupling rows include f-state occupations on DOUBLE_EXCITED
    // (e-site projector + at least one double-excited state with that pigment).
    const auto& dcoup = sys.d_coupling_host();
    bool saw_f = false;
    for (int bath = 0; bath < n_bath; ++bath) {
        int nonzero = 0;
        for (int i = 0; i < N; ++i) {
            if (dcoup[static_cast<size_t>(bath) * N + i] != 0) ++nonzero;
        }
        if (nonzero >= 2) saw_f = true;
    }
    if (!saw_f) {
        std::cerr << "FAIL(esb_dex): expected f-state occupations in d_coupling\n";
        return false;
    }
    std::cout << "esb DOUBLE_EXCITED finite + d_coupling f-states: PASS\n";
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
    const std::vector<cuDoubleComplex> h_decay = {make_cuDoubleComplex(0.0, 0.0)};
    const std::vector<heom::real_t> h_markov = {0};     // [n_bath] per-site Markov residue
    const std::vector<int>    h_mode_off = {0, K + 1};  // [n_bath+1] ladder prefix sums

    DevBuf<complex_t>       d_rho   (M * N * N);
    DevBuf<complex_t>       d_drho  (M * N * N);
    DevBuf<complex_t>       d_H     (N * N);
    DevBuf<int>             d_plus  (M * n_bath * (K + 1));
    DevBuf<int>             d_minus (M * n_bath * (K + 1));
    DevBuf<complex_t>       d_c     (n_bath * (K + 1));
    DevBuf<heom::real_t>          d_coeff_plus (M * n_bath * (K + 1));
    DevBuf<heom::real_t>          d_coeff_minus(M * n_bath * (K + 1));
    DevBuf<int8_t>          d_level (M * n_bath * (K + 1));
    DevBuf<cuDoubleComplex> d_decay (M);
    DevBuf<int8_t>          d_dcoup (n_bath * N);
    DevBuf<heom::real_t>          d_markov(n_bath);
    DevBuf<int>             d_mode_off(n_bath + 1);

    CHK(cudaMemcpy(d_rho.ptr,   h_rho.data(),       M*N*N*sizeof(complex_t),             cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_H.ptr,     h_H.data(),         N*N*sizeof(complex_t),         cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_plus.ptr,  h_plus_map.data(),  M*n_bath*(K+1)*sizeof(int),          cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_minus.ptr, h_minus_map.data(), M*n_bath*(K+1)*sizeof(int),          cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_c.ptr,     h_c_mats.data(),    n_bath*(K+1)*sizeof(complex_t),       cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_coeff_plus.ptr,  h_coeff_plus.data(),  M*n_bath*(K+1)*sizeof(heom::real_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_coeff_minus.ptr, h_coeff_minus.data(), M*n_bath*(K+1)*sizeof(heom::real_t), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_level.ptr, h_level.data(),     M*n_bath*(K+1)*sizeof(int8_t),       cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_decay.ptr, h_decay.data(),     M*sizeof(cuDoubleComplex),           cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_dcoup.ptr, h_dcoup.data(),     n_bath*N*sizeof(int8_t),             cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_markov.ptr, h_markov.data(),   n_bath*sizeof(heom::real_t),               cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_mode_off.ptr, h_mode_off.data(), (n_bath+1)*sizeof(int),               cudaMemcpyHostToDevice));

    HEOMContext ctx{};
    ctx.rho_in            = d_rho.ptr;
    ctx.drho_out          = d_drho.ptr;
    ctx.hamiltonian       = d_H.ptr;
    ctx.c_bath       = d_c.ptr;
    ctx.cbar_bath    = d_c.ptr;   // c = 0 here, so c̄ = conj(c) = c: reuse the buffer
    ctx.coeff_plus   = d_coeff_plus.ptr;
    ctx.coeff_minus  = d_coeff_minus.ptr;
    ctx.markov_residue     = d_markov.ptr;
    ctx.neighbor_map_plus = d_plus.ptr;
    ctx.neighbor_map_minus= d_minus.ptr;
    ctx.level_indices     = d_level.ptr;
    ctx.total_decay       = d_decay.ptr;
    ctx.active_flags      = nullptr;     // no filtering
    ctx.d_coupling        = d_dcoup.ptr;
    ctx.mode_off          = d_mode_off.ptr;
    ctx.M = M; ctx.N = N; ctx.D = n_bath * (K + 1); ctx.n_bath = n_bath;
    ctx.sink_enabled = false; ctx.sink_site = 0; ctx.gamma_sink = 0.0;
    ctx.block = heom::ManifoldBlock::full(N);

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

    // The commutator check above is the suite's first test; the rest only
    // make sense once the kernel plumbing itself is right.
    ++heom_test::run();
    if (ok) {
        ++heom_test::passed();
        std::cout << "  commutator vs CPU reference (tol=" << TOL << ")... PASS\n";
        RUN_BOOL(test_trajectory_solve_without_initial_throws);
        RUN_BOOL(test_exciton_index_is_basis_independent);
        RUN_BOOL(test_hdf5_writer_status);
        RUN_BOOL(test_esb_hdf5_readback);
        RUN_BOOL(test_esb_double_excited_finite);
        RUN_BOOL(test_single_side_mu);
        RUN_BOOL(test_batch_single_side_mu);
        RUN_BOOL(test_batch_trace_BR);
        RUN_BOOL(test_rect_single_side_mu);
        RUN_BOOL(test_ulocal_filter_marking);
        RUN_BOOL(test_filter_nan_surfaces);
        RUN_BOOL(test_manifold_blocked_norms);
        RUN_BOOL(test_carry_through_primitives);
        RUN_BOOL(test_support_validation);
        RUN_BOOL(test_rho_capacity);
        RUN_BOOL(test_batch_derivatives_symmetry);
        RUN_BOOL(test_sink_term);
        RUN_BOOL(test_manifold_block_matches_full_launch);
        RUN_BOOL(test_disjoint_block_partial_row_staging);
        RUN_BOOL(test_manifold_block_ignores_stale_stage_buffer);
        RUN_BOOL(test_batch_rk4_symmetry);
        RUN_BOOL(test_batch_per_slot_norms);
        RUN_BOOL(test_fused_final_norms);
        RUN_BOOL(test_batch_alloc_plan);
        RUN_BOOL(test_active_batch_width);
        RUN_BOOL(test_batch_snapshot_expand);
        RUN_BOOL(test_snapshot_replayed_only_where_taken);
        RUN_BOOL(test_snapshot_restore_is_identity);
        RUN_BOOL(test_dense_launch_equals_compact);
        RUN_BOOL(test_vanishing_tolerance_fmo_dynamics);
        RUN_BOOL(test_divergence_baseline_per_slot);
        RUN_BOOL(test_batch_detect);
        RUN_BOOL(test_batch_dipole_per_slot);
        RUN_BOOL(test_per_slot_active_counts);
        RUN_BOOL(test_if_rk4_matches_plain_small_dt);
    } else {
        std::cout << "  commutator vs CPU reference... FAIL\n";
    }
    return heom_test_summary("test_gpu_kernel");
}
