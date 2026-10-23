// End-to-end batched 3-polarization absorption.
//
// Same small system, run compute_linear_response twice on the SAME
// SystemContext: (a) absorption.batch=false (serial x/y/z loop) and (b)
// absorption.batch=true (one B-wide propagation). Assert the three-direction
// ρ_xyz tensors agree:
//   - filtering OFF : bit-identical (atol == 0) — grid.y batching does not
//                     change per-slot arithmetic order.
//   - filtering ON  : bit-identical too — every slot keeps its OWN active set
//                     (2026-10-22-batch-per-slot; it used to be the union of
//                     the three, which was only within filter tolerance).

#include "spectra/linear_response.hpp"
#include "harness.hpp"
#include "spectra/dipole.hpp"
#include "heom_config.hpp"
#include "system_context.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

std::string write_abs_yaml(bool filter_on) {
    const std::string path = heom_test::tmp("test_abs_batch_")
                           + (filter_on ? "filt" : "nofilt") + ".yaml";
    std::ofstream f(path);
    f << "system:\n"
      << "  n_states: 2\n"
      << "  temperature: 77.0\n"
      << "  hamiltonian:\n"
      << "    - [12100.0, 100.0]\n"
      << "    - [100.0, 12000.0]\n"
      << "  w0: 12000.0\n"
      << "  dipoles:\n"
      << "    - [1.0, 0.2, 0.0]\n"
      << "    - [0.3, -0.8, 0.5]\n"
      << "  strengths: [1.0, 1.0]\n"
      << "bath:\n"
      << "  spectral_density: \"Drude-Lorentz\"\n"
      << "  eta: 35.0\n"
      << "  gamma_inverse: 100.0\n"
      << "  expansion_method: matsubara\n"
      << "  n_extra_poles: 0\n"
      << "  L_cutoff: 3\n"
      << "filter:\n"
      << "  enabled: " << (filter_on ? "true" : "false") << "\n"
      << "  tolerance: 1.0e-6\n"
      << "  ulist: 0.0\n"
      << "absorption:\n"
      << "  integrator: rk4\n"
      << "  dt: 2\n"
      << "  n_steps: 30\n"
      << "  output_interval: 1\n"
      << "  print_interval: 0\n"
      << "  output:\n"
      << "    save_path: \"" << heom_test::tmp_dir() << "/\"\n"
      << "    filename: \"abs_batch_test\"\n"
      << "    use_timestamp: false\n";
    return path;
}

// Run serial then batched on one SystemContext; return max |Δ| over ρ_xyz.
double run_and_diff(bool filter_on) {
    HEOMConfig::Config cfg =
        HEOMConfig::Config::load_yaml(write_abs_yaml(filter_on));
    HEOMConfig::resolve_absorption_config(cfg);

    SystemContext sys(cfg);
    auto mu = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths,
        cfg.physics.basis, cfg.physics.n_sites);

    heom::spectra::LinearResponseRequest req;
    req.dipole = mu;

    cfg.linear_batch = false;
    const auto serial = heom::spectra::compute_linear_response(sys, req);

    cfg.linear_batch = true;
    const auto batch = heom::spectra::compute_linear_response(sys, req);

    if (serial.rho_xyz.size() != batch.rho_xyz.size()) {
        std::cout << "  rho_xyz size mismatch: serial=" << serial.rho_xyz.size()
                  << " batch=" << batch.rho_xyz.size() << "\n";
        return 1e30;
    }
    double max_abs = 0.0;
    for (size_t k = 0; k < serial.rho_xyz.size(); ++k) {
        max_abs = std::max(max_abs, std::abs(serial.rho_xyz[k] - batch.rho_xyz[k]));
    }
    return max_abs;
}

bool test_filter_off_bitexact() {
    const double d = run_and_diff(/*filter_on=*/false);
    if (d != 0.0) {
        std::cout << "absorption batch (filter off): FAIL — max|Δ|=" << d
                  << " (expected 0)\n";
        return false;
    }
    std::cout << "absorption batch (filter off, serial==batch bit-exact): PASS\n";
    return true;
}

bool test_filter_on_tol() {
    const double d = run_and_diff(/*filter_on=*/true);
    if (d != 0.0) {
        std::cout << "absorption batch (filter on): FAIL — max|Δ|=" << d
                  << " (expected 0: per-slot active sets)\n";
        return false;
    }
    std::cout << "absorption batch (filter on, per-slot, serial==batch bit-exact): PASS\n";
    return true;
}

// linear_response.cpp declares the e×g manifold block (rows [1,N), column 0)
// for every absorption propagation. That declaration is only legitimate because
// ρ(0) = μ⁺|g⟩⟨g| has support nowhere else and the HEOM generator preserves the
// block. Guard the premise, not the optimisation: if ρ_eq ever stops being the
// pure ground projector (a thermal state populating e, say), or the basis stops
// putting g at index 0, the block silently starts truncating real amplitude and
// no bitwise A/B gate can see it — both branches would be equally wrong.
//
// Exact zeros, not a tolerance: elements outside the block are never written.
bool test_eg_block_support() {
    HEOMConfig::Config cfg =
        HEOMConfig::Config::load_yaml(write_abs_yaml(/*filter_on=*/false));
    HEOMConfig::resolve_absorption_config(cfg);

    SystemContext sys(cfg);
    auto mu = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths,
        cfg.physics.basis, cfg.physics.n_sites);

    heom::spectra::LinearResponseRequest req;
    req.dipole = mu;
    cfg.linear_batch = false;
    const auto res = heom::spectra::compute_linear_response(sys, req);

    const int N = res.N;
    double in_block_max = 0.0;
    for (int dir = 0; dir < 3; ++dir) {
        for (int rec = 0; rec < res.n_outputs; ++rec) {
            const size_t off =
                (static_cast<size_t>(dir) * res.n_outputs + rec)
                * static_cast<size_t>(N) * N;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    const auto v = res.rho_xyz[off + static_cast<size_t>(i) * N + j];
                    const bool in_block = (i >= 1 && j == 0);
                    if (in_block) {
                        in_block_max = std::max(in_block_max, std::abs(v));
                    } else if (v != std::complex<double>(0.0, 0.0)) {
                        std::cout << "absorption e×g block support: FAIL — "
                                  << "dir=" << dir << " rec=" << rec
                                  << " element (" << i << "," << j
                                  << ") = " << v.real() << std::showpos
                                  << v.imag() << "i" << std::noshowpos
                                  << " is outside the declared block\n";
                        return false;
                    }
                }
            }
        }
    }
    if (in_block_max == 0.0) {
        std::cout << "absorption e×g block support: FAIL — block is entirely "
                     "zero, so the support check proves nothing\n";
        return false;
    }
    std::cout << "absorption e×g block support (exterior exactly zero): PASS"
                 "  (in-block max=" << in_block_max << ")\n";
    return true;
}

}  // namespace

int main() {
    RUN_BOOL(test_filter_off_bitexact);
    RUN_BOOL(test_filter_on_tol);
    RUN_BOOL(test_eg_block_support);
    return heom_test_summary("test_absorption_batch");
}
