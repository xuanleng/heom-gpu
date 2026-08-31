#include "heom_config.hpp"
#include "system_context.hpp"
#include "heom_trajectory.hpp"
#include "spectra/absorption.hpp"
#include "spectra/dipole.hpp"
#include "spectra/two_dim.hpp"
#include "mc/mc_driver.hpp"
#include "output/spectrum_writer.hpp"
#include "output/console_printer.hpp"
#include "heom_version.hpp"

#include <chrono>
#include <cmath>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

/**
 * Task-table dispatch for spectroscopy subcommands.
 *
 * The CLI is a thin shell over a `kTasks[]` table. Each row supplies:
 *   - `name`        : the subcommand string (`"dynamics"`, `"absorption"`,
 *                     ...). The user types `heom_solver <name> <yaml>`.
 *   - `resolve`     : task-aware config check + auto-filename resolution
 *                     (called BEFORE `SystemContext sys(cfg)`).
 *   - `run`         : compute + write_results entry. Read-only on `cfg`.
 *   - `description` : one-liner for `--help`.
 *
 * Adding a new spectroscopy mode is three steps:
 *   1. write `resolve_<task>_config()` and `run_<task>()` (the resolver
 *      typically lives in src/heom_config.cpp next to its peers; the run
 *      function lives next to its compute kernel),
 *   2. add a row to `kTasks[]` below,
 *   3. done. `print_help` / dispatch are not touched.
 *
 * Architectural note: `SystemContext` is built once, then handed to
 * the task's `run` function. For 1D this is "one ctx, one trajectory". For
 * future 2DES the same SystemContext drives N trajectories with dipole-
 * applied ρ(0)s — no need to rebuild the hierarchy.
 */

namespace {

// ---------------------------------------------------------------------------
// Task entry points (resolve_* live in src/heom_config.cpp; run_* live here).
// ---------------------------------------------------------------------------

int run_dynamics(const SystemContext& sys, const HEOMConfig::Config& /*cfg*/) {
    HEOMTrajectory traj(sys);
    std::cout << "Initializing density matrix..." << std::flush;
    traj.set_initial_from_config();
    std::cout << " done.\n\n";

    traj.solve();
    traj.save_results();
    return traj.diverged() ? 2 : 0;
}

int run_absorption(const SystemContext& sys, const HEOMConfig::Config& cfg) {
    // resolve_absorption_config has already verified dipoles present and
    // basis == WITH_GROUND, so this is safe here.
    auto mu = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles,
        cfg.physics.strengths,
        cfg.physics.basis,
        cfg.physics.n_sites);

    // Total wall-clock across the x/y/z propagations. Each direction already
    // prints its own "computation time" from inside HEOMTrajectory::solve();
    // this brackets all three (the same pattern future 2DES run_* reuse).
    const auto t_start = std::chrono::steady_clock::now();
    auto out = heom::spectra::compute_absorption(sys, mu);
    const double total_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cout << "Total absorption computation time (x+y+z): "
              << heom::console::format_computation_time(total_s) << "\n\n";

    // HDF5 output is the spectroscopy writer;
    // compute_absorption itself is host-only and never opens files. The total
    // x/y/z wall-clock goes into the file as `simulation_time_seconds`.
    heom::output::write_spectrum(cfg, sys, out, total_s);

    // Linear response cannot "diverge" via the same mechanism as a populated
    // ρ_eq dynamics run (compute_linear_response throws on divergence
    // instead of returning a half-built result), so reaching here means OK.
    return 0;
}

int run_2des(const SystemContext& sys, const HEOMConfig::Config& cfg) {
    // resolve_2des_config validated axis presence, dt divisibility, and forced
    // basis = double_excited (so `sys` is the ESA context).
    const auto& spec = *cfg.twodes;
    heom::spectra::TwoDESRequest req =
        heom::spectra::build_2des_request(spec, cfg.time.dt);

    // ESA context = the double_excited `sys` built by main. SE/GSB context =
    // a with_ground clone (apply_basis re-expands H on a copied config). cfg_wg
    // must outlive sys_single (SystemContext holds a const Config&).
    HEOMConfig::Config cfg_wg = cfg;
    HEOMConfig::apply_basis(cfg_wg, BasisType::WITH_GROUND);
    SystemContext sys_single(cfg_wg);

    auto mu_double = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths, BasisType::DOUBLE_EXCITED, sys.N());
    auto mu_single = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths, BasisType::WITH_GROUND, sys_single.N());
    req.dipole_double = &mu_double;
    req.dipole_single = &mu_single;

    const auto t_start = std::chrono::steady_clock::now();
    auto out = heom::spectra::compute_2des(sys_single, sys, req);
    const double total_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cout << "Total 2DES computation time: "
              << heom::console::format_computation_time(total_s) << "\n\n";

    heom::output::write_2des(cfg, sys, out, total_s);
    return 0;
}

// ---------------------------------------------------------------------------
// Task table — single source of truth for what subcommands exist.
// ---------------------------------------------------------------------------

struct TaskEntry {
    std::string_view name;
    void (*resolve)(HEOMConfig::Config&);
    int  (*run)(const SystemContext&, const HEOMConfig::Config&);
    std::string_view description;
};

constexpr TaskEntry kTasks[] = {
    {"dynamics",   HEOMConfig::resolve_dynamics_config,   run_dynamics,
     "Population dynamics (HEOM time evolution)"},
    {"absorption", HEOMConfig::resolve_absorption_config, run_absorption,
     "Linear absorption spectroscopy"},
    {"2des",       HEOMConfig::resolve_2des_config,       run_2des,
     "Two-dimensional electronic spectroscopy (carry-through HEOM)"},
    // Future: emission / cd / fluorescence / ta — add a row here.
};

const TaskEntry* find_task(std::string_view name) {
    for (const auto& t : kTasks) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// Full help — usage, the MC-worker plumbing flags, and the task list, all in
// one shot (no separate --list-tasks; --help shows everything). Streamed so the
// error paths can send it to stderr and --help to stdout.
void print_help(std::ostream& os, const char* prog) {
    os << "Usage:\n"
       << "  " << prog << " <task> <config.yaml>\n"
       << "      Single-shot run of <task> on <config.yaml>.\n"
       << "\n"
       << "  " << prog << " <task> <config.yaml> --rank <N> [--run-id <TS>] "
          "[--r-begin <A> --r-end <B>]\n"
       << "      MC static-disorder worker N. Low-level plumbing — normally\n"
       << "      launched by `mc-disorder-run <task> <config.yaml> --workers K`.\n"
       << "        --rank <N>              1-based worker index (N >= 1).\n"
       << "        --run-id <TS>           shared stem for a worker set "
          "(YYYYMMDD_HHMM or YYYYMMDD_HHMMSS)\n"
       << "                                (optional; a lone worker mints its own).\n"
       << "        --r-begin <A> --r-end <B>\n"
       << "                                this worker's 1-based INCLUSIVE realization\n"
       << "                                range (both together, or both omitted =\n"
       << "                                whole node file). High-level equivalent:\n"
       << "                                `mc-disorder-run ... --range A B`.\n"
       << "\n"
       << "  " << prog << " --help\n"
       << "      Show this message (usage + supported tasks).\n"
       << "  " << prog << " --version\n"
       << "      Print " << heom_software_name() << " "
       << heom_software_version() << ".\n"
       << "\n"
       << "Supported tasks:\n";
    for (const auto& t : kTasks) {
        os << "  " << std::left << std::setw(14) << t.name << t.description << "\n";
    }
}

// Parse the optional MC worker knobs from argv[3..]. Returns -1 when `--rank`
// is absent (single-shot). `--run-id TS` (optional, only meaningful with
// --rank) is the shared timestamp the launcher hands to every worker so their
// rank files share one prefix; captured into `run_id`. `--r-begin N` /
// `--r-end M` give this worker's 1-based INCLUSIVE realization interval (must be
// given together or both omitted ⇒ 0/0 sentinel ⇒ driver uses the whole file);
// the driver does the single 1-based→0-based conversion. The disorder node file
// and flock queue are still derived from the config. Unknown flags / missing
// values throw so a typo cannot silently mis-run.
int parse_worker_rank(int argc, char** argv, std::string& run_id,
                      int& r_begin, int& r_end) {
    int rank = -1;
    r_begin = 0;
    r_end = 0;
    bool has_begin = false, has_end = false;
    for (int i = 3; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--rank") {
            if (i + 1 >= argc) throw std::invalid_argument("--rank requires a value");
            rank = std::stoi(argv[++i]);
            if (rank < 1) {
                throw std::invalid_argument(
                    "--rank must be >= 1 (workers are 1-based); "
                    "use `mc-disorder-run <task> <cfg> --workers K` to launch workers");
            }
        } else if (flag == "--run-id") {
            if (i + 1 >= argc) throw std::invalid_argument("--run-id requires a value");
            run_id = argv[++i];
        } else if (flag == "--r-begin") {
            if (i + 1 >= argc) throw std::invalid_argument("--r-begin requires a value");
            r_begin = std::stoi(argv[++i]);
            has_begin = true;
        } else if (flag == "--r-end") {
            if (i + 1 >= argc) throw std::invalid_argument("--r-end requires a value");
            r_end = std::stoi(argv[++i]);
            has_end = true;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(flag));
        }
    }
    if (has_begin != has_end) {
        throw std::invalid_argument(
            "--r-begin and --r-end must be given together (1-based inclusive "
            "range) or both omitted (whole node file)");
    }
    return rank;
}

// Timestamp stem matching resolve_output_path's `YYYYMMDD_HHMM`, so a worker
// with no --run-id names its shard exactly like a single-shot would.
std::string local_timestamp() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream ts;
    ts << std::setfill('0')
       << std::setw(4) << (tm.tm_year + 1900)
       << std::setw(2) << (tm.tm_mon + 1)
       << std::setw(2) << tm.tm_mday << "_"
       << std::setw(2) << tm.tm_hour
       << std::setw(2) << tm.tm_min;
    return ts.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        print_help(std::cout, argv[0]);
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << heom_software_name() << " " << heom_software_version()
                  << "\n";
        return 0;
    }

    if (argc < 3) {
        // Deliberately do NOT default to dynamics. Print a clear
        // error so callers know to upgrade their command line. The
        // `Did you mean: ... dynamics ...` hint catches the most common
        // old muscle-memory invocation `heom_solver foo.yaml`.
        print_help(std::cerr, argv[0]);
        if (argc == 2) {
            std::cerr << "\nDid you mean: " << argv[0]
                      << " dynamics " << argv[1] << " ?\n";
        }
        return 1;
    }

    const std::string_view task_name = argv[1];
    const std::string yaml = argv[2];

    const TaskEntry* task = find_task(task_name);
    if (!task) {
        std::cerr << "Unknown task: '" << task_name
                  << "'. Try: " << argv[0] << " --help\n";
        return 1;
    }

    try {
        std::string run_id;
        int r_begin = 0, r_end = 0;
        const int rank = parse_worker_rank(argc, argv, run_id, r_begin, r_end);

        auto cfg = HEOMConfig::Config::load_yaml(yaml);
        // resolve_<task>_config(cfg) MUST run BEFORE SystemContext
        // sys(cfg). The latter holds a `const Config&`, so any task-level
        // mutation of cfg (output.filename auto-resolution, in particular)
        // performed inside run_*(sys, cfg) would be invisible to the
        // writer. The dispatch below is the single place we get to mutate.
        // It also fills cfg.output (save_path) — the MC paths below need it.
        task->resolve(cfg);
        cfg.print_summary();

        const bool have_draws = !cfg.disorder_draws_file.empty();

        // Mode is chosen by the invocation, not a config switch: no `--rank`
        // ⇒ single-shot (byte-for-byte the pre-MC dispatch). `--rank N` ⇒ this
        // process is MC worker N; it hands off to the MC shell which builds its
        // own per-realization contexts.
        if (rank < 0) {
            if (have_draws) {
                std::cerr << "[note] static_disorder.draws_file is set but no "
                             "--rank given; running single-shot. Use "
                             "`mc-disorder-run " << task_name << " " << yaml
                          << " --workers K` for disorder averaging.\n";
            }
            SystemContext sys(cfg);
            return task->run(sys, cfg);
        }

        if (!have_draws) {
            throw std::invalid_argument(
                "--rank given but config has no `static_disorder.draws_file` "
                "(add the block, or drop --rank for a single-shot run)");
        }
        // Workers of one launch share the TS (launcher passes --run-id); a lone
        // worker mints its own. The driver builds the shard stem itself — it
        // needs disorder_identity (from the node file) for the merge key.
        const std::string ts = run_id.empty() ? local_timestamp() : run_id;
        heom::mc::MCOptions opts;
        opts.draws_file = cfg.disorder_draws_file;
        opts.queue      = cfg.output.save_path + "mc.queue";
        opts.r_begin    = r_begin;   // 1-based inclusive (0 = sentinel ⇒ whole file)
        opts.r_end      = r_end;
        opts.run_id     = ts;
        opts.rank       = rank;
        return heom::mc::run_mc(std::string(task_name), cfg, opts);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
