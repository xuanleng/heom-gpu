#include "heom_config.hpp"
#include "system_context.hpp"
#include "dynamics_run.hpp"
#include "spectra/absorption.hpp"
#include "spectra/dipole.hpp"
#include "spectra/fluorescence.hpp"
#include "spectra/two_dim.hpp"
#include "mc/mc_driver.hpp"
#include "output/spectrum_writer.hpp"
#include "output/console_printer.hpp"
#include "heom_version.hpp"
#include "util/timestamp.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
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
    return heom::run_dynamics(sys) ? 2 : 0;
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
    // prints its own "computation time" from inside heom::run_output_loop;
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

int run_fluorescence(const SystemContext& sys, const HEOMConfig::Config& cfg) {
    auto mu = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths, cfg.physics.basis, cfg.physics.n_sites);
    const auto t_start = std::chrono::steady_clock::now();
    auto out = heom::spectra::compute_fluorescence(sys, mu);
    const double total_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cout << "Total fluorescence computation time (equilibration + emission): "
              << heom::console::format_computation_time(total_s) << "\n\n";
    heom::output::write_fluorescence(cfg, sys, out, total_s);
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
    // MC static-disorder worker (`--rank`); nullptr ⇒ MC does not apply.
    int  (*mc_run)(const HEOMConfig::Config&, const heom::mc::MCOptions&);
    std::string_view description;
};

constexpr TaskEntry kTasks[] = {
    {"dynamics",   HEOMConfig::resolve_dynamics_config,   run_dynamics,    nullptr,
     "Population dynamics (HEOM time evolution)"},
    {"absorption", HEOMConfig::resolve_absorption_config, run_absorption,  heom::mc::run_absorption_mc,
     "Linear absorption spectroscopy"},
    {"fluorescence", HEOMConfig::resolve_fluorescence_config, run_fluorescence, heom::mc::run_fluorescence_mc,
     "Steady-state fluorescence (equilibrated excited-state hierarchy)"},
    {"2des",       HEOMConfig::resolve_2des_config,       run_2des,        heom::mc::run_2des_mc,
     "Two-dimensional electronic spectroscopy (carry-through HEOM)"},
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
       << "      launched by `mc-disorder-run <task> <config.yaml> [--gpus 0,1]`.\n"
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

// Parse the optional MC worker knobs from argv[3..]. `rank` stays nullopt when
// `--rank` is absent (single-shot). `--run-id TS` (optional, only meaningful
// with --rank) is the shared timestamp the launcher hands to every worker so
// their rank files share one prefix; captured into `run_id`. `--r-begin N` /
// `--r-end M` give this worker's 1-based INCLUSIVE realization interval (must be
// given together, or both omitted ⇒ `range` nullopt ⇒ driver uses the whole
// file); the driver does the single 1-based→0-based conversion. The disorder
// node file and flock queue are still derived from the config. Unknown flags,
// missing or non-integer values ("1x"), and worker flags without --rank all
// throw so a typo cannot silently mis-run.
struct WorkerArgs {
    std::optional<int> rank;
    std::string run_id;
    std::optional<heom::mc::Interval> range;
};

int parse_int_arg(std::string_view flag, std::string_view text) {
    int v = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), v);
    if (ec != std::errc() || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string(flag) + " expects an integer, got '" +
                                    std::string(text) + "'");
    }
    return v;
}

WorkerArgs parse_worker_args(int argc, char** argv) {
    WorkerArgs w;
    std::optional<int> r_begin, r_end;
    for (int i = 3; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--rank") {
            if (i + 1 >= argc) throw std::invalid_argument("--rank requires a value");
            const int rank = parse_int_arg(flag, argv[++i]);
            if (rank < 1) {
                throw std::invalid_argument(
                    "--rank must be >= 1 (workers are 1-based); "
                    "use `mc-disorder-run <task> <cfg> [--gpus 0,1]` to launch workers");
            }
            w.rank = rank;
        } else if (flag == "--run-id") {
            if (i + 1 >= argc) throw std::invalid_argument("--run-id requires a value");
            w.run_id = argv[++i];
        } else if (flag == "--r-begin") {
            if (i + 1 >= argc) throw std::invalid_argument("--r-begin requires a value");
            r_begin = parse_int_arg(flag, argv[++i]);
        } else if (flag == "--r-end") {
            if (i + 1 >= argc) throw std::invalid_argument("--r-end requires a value");
            r_end = parse_int_arg(flag, argv[++i]);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(flag));
        }
    }
    if (!w.rank && (r_begin || r_end || !w.run_id.empty())) {
        throw std::invalid_argument(
            "--run-id / --r-begin / --r-end only apply to an MC worker (--rank N); "
            "a single-shot run would ignore them");
    }
    if (r_begin.has_value() != r_end.has_value()) {
        throw std::invalid_argument(
            "--r-begin and --r-end must be given together (1-based inclusive "
            "range) or both omitted (whole node file)");
    }
    if (r_begin) w.range = heom::mc::Interval{*r_begin, *r_end};
    return w;
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
        const WorkerArgs w = parse_worker_args(argc, argv);

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
        if (!w.rank) {
            if (have_draws) {
                std::cerr << "[note] static_disorder.draws_file is set but no "
                             "--rank given; running single-shot. Use "
                             "`mc-disorder-run " << task_name << " " << yaml
                          << "` (add `--gpus` for multi-GPU).\n";
            }
            SystemContext sys(cfg);
            return task->run(sys, cfg);
        }

        if (!have_draws) {
            throw std::invalid_argument(
                "--rank given but config has no `static_disorder.draws_file` "
                "(add the block, or drop --rank for a single-shot run)");
        }
        if (!task->mc_run) {
            throw std::invalid_argument(
                "MC static disorder does not apply to task '" + std::string(task_name) +
                "' (dynamics runs single-shot; drop --rank)");
        }
        // Workers of one launch share the TS (launcher passes --run-id); a lone
        // worker mints its own. The driver builds the shard stem itself — it
        // needs disorder_identity (from the node file) for the merge key.
        heom::mc::MCOptions opts;
        opts.draws_file = cfg.disorder_draws_file;
        opts.range      = w.range;
        opts.run_id     = w.run_id.empty() ? heom::local_timestamp() : w.run_id;
        opts.rank       = *w.rank;
        return task->mc_run(cfg, opts);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
