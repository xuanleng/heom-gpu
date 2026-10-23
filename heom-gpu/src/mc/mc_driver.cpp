#include "mc/mc_driver.hpp"

#include "mc/mc_disorder.hpp"
#include "mc/mc_partial.hpp"
#include "mc/mc_queue.hpp"
#include "mc/mc_realizations.hpp"

#include "system_context.hpp"
#include "core/hamiltonian_expansion.hpp"   // compute_hilbert_dim
#include "output/console_printer.hpp"       // format_computation_time
#include "output/spectrum_writer.hpp"       // write_spectrum / write_2des
#include "spectra/absorption.hpp"
#include "spectra/dipole.hpp"
#include "spectra/fluorescence.hpp"
#include "spectra/two_dim.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace heom::mc {

namespace {

// A shard is built in two steps — the ordinary spectrum writer emits a
// single-shot-format file, then append_mc_provenance adds /mc — and the merge
// tools glob save_path for *_rank*.h5. A worker killed between the two steps
// would leave a well-formed spectrum with no /mc where the merge can find it.
// So the writer targets this hidden subdirectory (the tools' globs and the
// shard regex never look inside a dot-dir) and publish_shard renames the
// finished file into save_path — same filesystem, so the rename is atomic and
// the shard is either complete or invisible.
constexpr const char* kPartialDir = ".partial";

// A copy of cfg whose output writes exactly `stem` (the shard name — base + key
// + TS + 1-based interval + rank, already assembled) under
// <save_path>/.partial/, with the timestamp writer off.
HEOMConfig::Config output_config(const HEOMConfig::Config& cfg,
                                 const std::string& stem) {
    HEOMConfig::Config out = cfg;
    // Trailing separator: resolve_output_path concatenates dir + filename.
    out.output.save_path =
        (std::filesystem::path(cfg.output.save_path) / kPartialDir / "").string();
    out.output.filename = stem;
    out.output.use_timestamp = false;
    return out;
}

// Move a finished shard from .partial/ into save_path. Returns the final path.
std::string publish_shard(const std::string& partial_path,
                          const HEOMConfig::Config& cfg) {
    const auto final_path = std::filesystem::path(cfg.output.save_path) /
                            std::filesystem::path(partial_path).filename();
    std::filesystem::rename(partial_path, final_path);
    return final_path.string();
}

// The 1-based INCLUSIVE realization interval this worker covers; no --r-begin/
// --r-end on the command line means the whole file [1, n_node].
Interval resolve_interval(const MCOptions& opts, int n_node) {
    const Interval iv = opts.range.value_or(Interval{1, n_node});
    const int b = iv.begin1, e = iv.end1;
    if (b < 1 || e < b || e > n_node) {
        throw std::runtime_error(
            "MC realization range [" + std::to_string(b) + "," +
            std::to_string(e) + "] out of bounds 1.." + std::to_string(n_node) +
            "; extend the node file (mc-disorder-gen --extend) or fix --range");
    }
    return {b, e};
}

// <base>_<key>_<TS>_r<begin>-<end>_rank<N>. Interval is 1-based (matches the
// user's --range); key + TS group shards across batches. base = the
// resolved single-shot stem (e.g. fmo_absorption), read BEFORE it is overwritten
// by output_config.
std::string shard_stem(const HEOMConfig::Config& cfg, const MCOptions& opts,
                       const std::string& disorder_identity, const Interval& iv) {
    const std::string key = mc_merge_key(cfg.config_hash, disorder_identity);
    return cfg.output.filename + "_" + key + "_" + opts.run_id +
           "_r" + std::to_string(iv.begin1) + "-" + std::to_string(iv.end1) +
           "_rank" + std::to_string(opts.rank);
}

// One MC worker, for any spectroscopy. `Task` supplies the five things that
// differ between absorption and 2DES and nothing else:
//   label                      console tag
//   Result compute(cfg_r)      run the spectroscopy on ONE disordered config
//   shape(O)                   observable shape for Accumulator::init
//   payload(O)                 the observable vector inside a Result (mutable)
//   write(cfg, sys, O, sec)    single-shot-format HDF5 writer; returns path
//
// Everything else — realization file, interval, shard naming, flock queue,
// per-node clone + disorder, weighting, the empty-worker note, provenance,
// atomic publish — is written once here. The HEOM kernel is untouched.
template <class Task>
int run_worker(const HEOMConfig::Config& cfg, const MCOptions& opts, const Task& task) {
    using Result = decltype(task.compute(cfg));

    // The queue counter and shard files live under output.<task>.save_path
    // (derived in main). Create it up front so the flock open and the spectrum
    // writer both land (the writer resolves the same save_path).
    const std::filesystem::path parent(cfg.output.save_path);
    if (!parent.empty()) std::filesystem::create_directories(parent);

    RealizationSet R = load_realizations(opts.draws_file);   // rejects non-mean
    if (R.n_site != cfg.physics.n_states) {
        throw std::runtime_error(
            std::string("run_mc(") + task.label + "): draws n_site=" +
            std::to_string(R.n_site) + " != cfg n_states=" +
            std::to_string(cfg.physics.n_states));
    }

    const Interval iv = resolve_interval(opts, R.n_node);
    const std::string stem = shard_stem(cfg, opts, R.disorder_identity, iv);
    // Shared flock queue over the 0-based half-open interval [begin0, end0):
    // the single 1-based→0-based conversion. Every worker of a batch draws
    // disjoint r ids from the same counter, so ids are unique with no gaps.
    // The counter file is named by the batch (run-id + interval), never reused.
    Queue queue((parent / queue_file_name(opts.run_id, iv)).string(),
                iv.begin1 - 1, iv.end1);
    Accumulator acc;
    Result last;                                // template for shape/time axes
    bool inited = false;
    const auto t_worker = std::chrono::steady_clock::now();
    while (const auto claimed = queue.claim()) {
        const int r = *claimed;                 // 0-based (array/queue truth)
        std::cout << "\n=== MC disorder node " << (r + 1) << " / ["
                  << iv.begin1 << "," << iv.end1 << "]  (" << task.label << ") ===\n";
        const auto t_node = std::chrono::steady_clock::now();
        HEOMConfig::Config cr = cfg;            // deep value copy (Config is value-semantic)
        apply_disorder(cr, R.row(r));           // site diagonal + re-expand
        Result O = task.compute(cr);
        if (!inited) {
            acc.init(task.shape(O), R.estimator);
            inited = true;
        }
        // mean folds a bare sum (scale=1, exact no-op); quadrature folds w_r·O_r.
        const double scale =
            (R.estimator == Estimator::Quadrature) ? R.weight[r] : 1.0;
        acc.add(task.payload(O), scale, r);
        last = std::move(O);                    // keep the last for its axes/shape
        const double sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_node).count();
        std::cout << "=== node " << (r + 1) << " done in "
                  << heom::console::format_computation_time(sec) << " ===\n";
    }
    const double worker_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_worker).count();

    // Legitimate when workers > nodes, or when a worker joins a batch (same
    // --run-id and range) whose queue is already drained; the launcher treats
    // non-zero as FAILED.
    if (acc.count == 0) {
        std::cout << "\n[note] this worker claimed 0 nodes (workers > nodes, or "
                     "run-id " << opts.run_id << " r" << iv.begin1 << "-" << iv.end1
                  << " is already drained); nothing written.\n";
        return 0;
    }

    // Write a single-shot-format file whose /results holds THIS worker's
    // disorder mean, then append the /mc provenance the merge step needs. Base
    // context (σ=0) supplies the disorder-invariant metadata. Quadrature has no
    // per-worker mean (Σw_claimed≠1) → /results carries the partial weighted sum
    // as a placeholder; the authoritative payload is /mc/sum_obs.
    task.payload(last) = (R.estimator == Estimator::Quadrature) ? acc.sum_obs
                                                                : acc.mean();
    SystemContext sys0(cfg);
    const HEOMConfig::Config cfg_out = output_config(cfg, stem);
    const std::string partial = task.write(cfg_out, sys0, last, worker_sec);
    append_mc_provenance(partial, acc, cfg.config_hash, R.disorder_identity);
    const std::string path = publish_shard(partial, cfg);
    std::cout << "\nMC " << task.label << " shard written: " << path
              << " (count=" << acc.count << ", worker time "
              << heom::console::format_computation_time(worker_sec) << ")\n";
    return 0;
}

} // namespace

int run_absorption_mc(const HEOMConfig::Config& cfg, const MCOptions& opts) {
    // Dipoles are disorder-invariant → build once.
    const auto mu = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths,
        cfg.physics.basis, cfg.physics.n_sites);

    struct Absorption {
        const char* label;
        const heom::spectra::DipoleOperator& mu;
        heom::spectra::LinearResponseResult compute(const HEOMConfig::Config& cr) const {
            SystemContext sys(cr);              // cr outlives sys (same scope)
            return heom::spectra::compute_absorption(sys, mu);
        }
        static std::vector<size_t> shape(const heom::spectra::LinearResponseResult& O) {
            return {3u, static_cast<size_t>(O.n_outputs),
                    static_cast<size_t>(O.N), static_cast<size_t>(O.N)};
        }
        static std::vector<std::complex<double>>& payload(heom::spectra::LinearResponseResult& O) {
            return O.rho_xyz;
        }
        static std::string write(const HEOMConfig::Config& c, const SystemContext& s,
                                 const heom::spectra::LinearResponseResult& O, double sec) {
            return heom::output::write_spectrum(c, s, O, sec);
        }
    };
    return run_worker(cfg, opts, Absorption{"absorption x/y/z", mu});
}

// Each realization equilibrates on its own; one that does not converge aborts
// the worker before any shard is written.
int run_fluorescence_mc(const HEOMConfig::Config& cfg, const MCOptions& opts) {
    const auto mu = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths,
        cfg.physics.basis, cfg.physics.n_sites);

    struct Fluorescence {
        const char* label;
        const heom::spectra::DipoleOperator& mu;
        heom::spectra::FluorescenceResult compute(const HEOMConfig::Config& cr) const {
            SystemContext sys(cr);
            return heom::spectra::compute_fluorescence(sys, mu);
        }
        static std::vector<size_t> shape(const heom::spectra::FluorescenceResult& O) {
            return {3u, static_cast<size_t>(O.linear.n_outputs),
                    static_cast<size_t>(O.linear.N), static_cast<size_t>(O.linear.N)};
        }
        static std::vector<std::complex<double>>& payload(heom::spectra::FluorescenceResult& O) {
            return O.linear.rho_xyz;
        }
        // The ensemble file carries no single realization's equilibrium state.
        static std::string write(const HEOMConfig::Config& c, const SystemContext& s,
                                 const heom::spectra::FluorescenceResult& O, double sec) {
            const heom::spectra::FluorescenceResult ens{O.linear, {}, {}};
            return heom::output::write_fluorescence(c, s, ens, sec);
        }
    };
    return run_worker(cfg, opts, Fluorescence{"fluorescence x/y/z", mu});
}

int run_2des_mc(const HEOMConfig::Config& cfg, const MCOptions& opts) {
    const auto& spec = *cfg.twodes;   // resolve_2des_config asserted; basis=DOUBLE_EXCITED
    heom::spectra::TwoDESRequest req =
        heom::spectra::build_2des_request(spec, cfg.time.dt);

    // Dipoles are disorder-invariant → build once. Double dim = cfg.physics.n_sites
    // (basis already DOUBLE_EXCITED); single dim from the WITH_GROUND expansion.
    const int single_N = compute_hilbert_dim(cfg.physics.n_states, BasisType::WITH_GROUND);
    const auto mu_double = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths, BasisType::DOUBLE_EXCITED, cfg.physics.n_sites);
    const auto mu_single = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths, BasisType::WITH_GROUND, single_N);
    req.dipole_double = &mu_double;
    req.dipole_single = &mu_single;

    struct TwoDES {
        const char* label;
        const heom::spectra::TwoDESRequest& req;
        heom::spectra::TwoDESResult compute(const HEOMConfig::Config& cr_double) const {
            // The same disorder drives both contexts. ESA = double-excited
            // context (the disordered config as handed in); SE/GSB = the same
            // disordered site diagonal re-expanded WITH_GROUND.
            SystemContext sys_double(cr_double);
            HEOMConfig::Config cr_single = cr_double;
            HEOMConfig::apply_basis(cr_single, BasisType::WITH_GROUND);
            SystemContext sys_single(cr_single);
            return heom::spectra::compute_2des(sys_single, sys_double, req);
        }
        static std::vector<size_t> shape(const heom::spectra::TwoDESResult& O) {
            return {static_cast<size_t>(O.n_component), static_cast<size_t>(O.n_signal),
                    static_cast<size_t>(O.n_pathway),   static_cast<size_t>(O.n_tau),
                    static_cast<size_t>(O.n_tw),        static_cast<size_t>(O.n_t)};
        }
        static std::vector<std::complex<double>>& payload(heom::spectra::TwoDESResult& O) {
            return O.resp;
        }
        static std::string write(const HEOMConfig::Config& c, const SystemContext& s,
                                 const heom::spectra::TwoDESResult& O, double sec) {
            return heom::output::write_2des(c, s, O, sec);
        }
    };
    return run_worker(cfg, opts, TwoDES{"2des", req});
}

} // namespace heom::mc
