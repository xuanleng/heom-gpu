#include "dynamics_run.hpp"
#include "heom_trajectory.hpp"
#include "system_context.hpp"
#include "output/hdf5_writer.hpp"
#include "output/console_printer.hpp"
#include "util/stream_format.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace heom {

double run_output_loop(HEOMTrajectory& traj, const StepCallback& sink) {
    // The sink is handed batch() slots, but read_alpha0 fills only the active
    // ones; a narrowed width would feed it stale bytes for slots ≥ active.
    if (traj.active_batch() != traj.batch()) {
        throw std::runtime_error(
            "run_output_loop: active_batch (" + std::to_string(traj.active_batch())
            + ") != batch (" + std::to_string(traj.batch())
            + "); set_initial with batch() matrices");
    }

    std::cout << "=== Time Evolution ===\n";

    const auto& cfg = traj.sys().config();
    const double dt_output_fs    = cfg.time.dt;
    const int    n_output_steps  = cfg.time.n_steps;
    const int    output_interval = cfg.time.output_interval;

    // Formatting stated here, not inherited. This line used to print "1000.0 fs"
    // or "1000.000 fs" for the SAME run length depending on which initial-state
    // branch had printed last and what precision it leaked.
    {
        const heom::StreamFormat fmt_guard(std::cout);
        std::cout << "Running " << n_output_steps << " steps ("
                  << std::fixed << std::setprecision(1)
                  << (n_output_steps * dt_output_fs) << " fs)...\n";
    }

    const auto start_time = std::chrono::high_resolution_clock::now();

    const int N = traj.sys().N();
    const int B = traj.batch();
    std::vector<std::complex<double>> buf;
    auto emit = [&](int rec, double t_fs) {
        traj.read_alpha0(buf);
        sink(rec, t_fs, buf.data(), N, B);
    };

    // t=0 goes through emit() so the record 0 / sink contract is identical to
    // every later record.
    emit(0, 0.0);

    // One propagate_steps per record: the internal step counter starts at 0
    // after set_initial, so cadence printing / ulist gating see the same step
    // index the single-loop version fed them. The config
    // guarantees n_steps is a whole number of records, so the last step is
    // the last record.
    if (n_output_steps % output_interval != 0)
        throw std::invalid_argument(
            "run_output_loop: n_steps (" + std::to_string(n_output_steps)
            + ") is not a multiple of output_interval ("
            + std::to_string(output_interval) + ")");
    const int n_records = n_output_steps / output_interval;
    for (int rec = 1; rec <= n_records; ++rec) {
        traj.propagate_steps(output_interval, /*check_div=*/false);
        const int    step_done      = rec * output_interval;
        const double output_time_fs = step_done * dt_output_fs;
        emit(rec, output_time_fs);
        // Norm-based divergence guard: catches both trace-violating exponential
        // growth (dynamics: ρ(0) trace=1) and traceless off-diagonal explosion
        // (spectroscopy: ρ(0)=μ⁺ρ_eq, trace≡0). Runs AFTER the record is
        // delivered, as it always has.
        if (traj.check_divergence_now(output_time_fs, step_done, n_output_steps))
            break;
    }

    const std::chrono::duration<double> elapsed =
        std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "done. (computation time: "
              << heom::console::format_computation_time(elapsed.count()) << ")\n\n";
    return elapsed.count();
}

namespace {

// cfg.initial → ρ(0), with the human-readable summary the dynamics CLI has
// always printed. Lives here, not in build_initial_state(), so 2DES drivers
// that build ρ(0) themselves stay quiet.
Eigen::MatrixXcd describe_and_build_initial(const SystemContext& sys) {
    const heom::StreamFormat fmt_guard(std::cout);
    if (!sys.config().initial.has_value()) {
        throw std::runtime_error(
            "run_dynamics: cfg.initial is empty; the dynamics task requires "
            "'initial_state:' in yaml.");
    }
    const InitialState& init = *sys.config().initial;

    switch (init.type) {
        case InitialState::Type::SITE_EXCITATION:
            std::cout << " pigment "
                      << hilbert_to_pigment(init.excited_site,
                                            sys.config().physics.basis)
                      << " excited";
            break;
        case InitialState::Type::EXCITON_EXCITATION: {
            const auto& phys = sys.config().physics;
            const int lo = ground_slots(phys.basis);
            heom::console::print_eigenspectrum(
                sys.eigenvalues(), lo, phys.n_states, init.exciton_index, phys.w0_cm);
            heom::console::print_exciton_composition(
                sys.eigenvectors(), init.exciton_index, lo, phys.n_states);
            break;
        }
        case InitialState::Type::COHERENT:
            std::cout << " coherent superposition";
            break;
        case InitialState::Type::CUSTOM_MATRIX:
            std::cout << " custom matrix";
            break;
    }

    Eigen::MatrixXcd rho0 = build_initial_state(sys, init);

    if (init.type == InitialState::Type::EXCITON_EXCITATION) {
        // print_density_matrix_summary needs the actual built matrix
        heom::console::print_density_matrix_summary(rho0, sys.N());
    } else if (init.type == InitialState::Type::COHERENT) {
        // Per-site populations need init.coherent_coeffs
        for (int i = 0; i < sys.N(); ++i) {
            const double pop = std::norm(init.coherent_coeffs[i]);
            if (pop > 1e-6)
                std::cout << " |c" << i << "|^2="
                          << std::fixed << std::setprecision(3) << pop;
        }
    }
    return rho0;
}

}  // namespace

bool run_dynamics(const SystemContext& sys) {
    const auto& cfg = sys.config();
    const int N = sys.N();
    HEOMTrajectory traj(sys);   // B=1

    std::cout << "Initializing density matrix..." << std::flush;
    const Eigen::MatrixXcd rho0 = describe_and_build_initial(sys);
    // Dynamics runs the whole matrix: a config-built ρ0 may sit anywhere.
    traj.set_initial({rho0}, heom::ManifoldBlock::full(N));
    std::cout << " done.\n\n";

    HDF5Writer writer(cfg.output.use_timestamp);
    writer.open(sys, rho0, cfg.filter.enabled, cfg.filter.tolerance);
    const double seconds = run_output_loop(
        traj, [&](int, double t_fs, const std::complex<double>* rho, int, int) {
            writer.write_timestep(t_fs, rho, N);
        });
    writer.close(seconds,
                 traj.diverged() ? heom::hdf5::RunStatus::Diverged
                                 : heom::hdf5::RunStatus::Complete,
                 cfg.filter.enabled ? &traj.filtering_history() : nullptr);

    // Tr(ρ_0) of the final state. One more N² D2H; the bytes are the ones the
    // last record already saw (or, on divergence, the state the check fired on).
    std::vector<std::complex<double>> buf;
    traj.read_alpha0(buf);
    double trace = 0.0;
    for (int i = 0; i < N; ++i) trace += buf[static_cast<size_t>(i) * N + i].real();

    const heom::StreamFormat fmt_guard(std::cout);
    if (traj.diverged()) {
        // `fixed` stated explicitly: setprecision alone means SIGNIFICANT
        // digits, so an exact trace of 1 printed as "1", not "1.0000000000".
        std::cerr << "Final trace: " << std::fixed << std::setprecision(10) << trace
                  << " (DIVERGED)\n";
    } else {
        std::cout << "Final trace: " << std::fixed << std::setprecision(6) << trace << "\n";
    }
    std::cout << "Results saved to: " << writer.get_output_path() << "\n";
    return traj.diverged();
}

}  // namespace heom
