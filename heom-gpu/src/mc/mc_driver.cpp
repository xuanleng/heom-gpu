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
#include "spectra/two_dim.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace heom::mc {

namespace {

// A copy of cfg whose output writes exactly `stem` (the shard name — base + key
// + TS + 1-based interval + rank, already assembled) under the task save_path,
// with the timestamp writer off. The ordinary spectrum writer then emits a
// single-shot-format file; the driver appends the /mc provenance block after.
HEOMConfig::Config output_config(const HEOMConfig::Config& cfg,
                                 const std::string& stem) {
    HEOMConfig::Config out = cfg;
    out.output.filename = stem;
    out.output.use_timestamp = false;
    return out;
}

// The 1-based INCLUSIVE realization interval this worker covers. The 0/0
// sentinel (main passed no --r-begin/--r-end) means the whole file [1, n_node].
struct Interval { int begin1; int end1; };

Interval resolve_interval(const MCOptions& opts, int n_node) {
    int b = opts.r_begin, e = opts.r_end;
    if (b == 0 && e == 0) { b = 1; e = n_node; }   // sentinel ⇒ whole file
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

// Realization source: a shared flock queue over the 0-based half-open interval
// [begin0, end0). The single 1-based→0-based conversion lives at the call site
// (begin1-1, end1). Every worker of a batch draws disjoint r ids from the same
// counter, so ids are unique with no gaps.
class ClaimSource {
public:
    ClaimSource(const std::string& queue, int begin0, int end0)
        : queue_(queue, begin0, end0) {}
    int next() { return queue_.claim(); }
private:
    Queue queue_;
};

int run_absorption_mc(const HEOMConfig::Config& cfg, const MCOptions& opts) {
    RealizationSet R = load_realizations(opts.draws_file);   // rejects non-mean
    if (R.n_site != cfg.physics.n_states) {
        throw std::runtime_error(
            "run_absorption_mc: draws n_site=" + std::to_string(R.n_site) +
            " != cfg n_states=" + std::to_string(cfg.physics.n_states));
    }

    const auto mu = heom::spectra::DipoleOperator::from_pigment_dipoles(
        cfg.physics.dipoles, cfg.physics.strengths,
        cfg.physics.basis, cfg.physics.n_sites);

    const Interval iv = resolve_interval(opts, R.n_node);
    const std::string stem = shard_stem(cfg, opts, R.disorder_identity, iv);
    ClaimSource src(opts.queue, iv.begin1 - 1, iv.end1);   // 1-based → 0-based half-open
    Accumulator acc;
    heom::spectra::LinearResponseResult last;   // template for shape/time axes
    bool inited = false;
    const auto t_worker = std::chrono::steady_clock::now();
    for (int r; (r = src.next()) >= 0; ) {      // r is 0-based (array/queue truth)
        std::cout << "\n=== MC disorder node " << (r + 1) << " / ["
                  << iv.begin1 << "," << iv.end1 << "]  (absorption x/y/z) ===\n";
        const auto t_node = std::chrono::steady_clock::now();
        HEOMConfig::Config cr = cfg;           // deep value copy (Config is value-semantic)
        apply_disorder(cr, R.row(r));          // site diagonal + re-expand (WITH_GROUND)
        SystemContext sys(cr);                 // cr outlives sys (same scope)
        auto O = heom::spectra::compute_absorption(sys, mu);
        if (!inited) {
            acc.init({3u, static_cast<size_t>(O.n_outputs),
                      static_cast<size_t>(O.N), static_cast<size_t>(O.N)},
                     R.estimator);
            inited = true;
        }
        // mean folds a bare sum (scale=1, exact no-op); quadrature folds w_r·O_r.
        const double scale =
            (R.estimator == Estimator::Quadrature) ? R.weight[r] : 1.0;
        acc.add(O.rho_xyz, scale, r);
        last = std::move(O);                   // keep the last for its axes/shape
        const double sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_node).count();
        std::cout << "=== node " << (r + 1) << " done in "
                  << heom::console::format_computation_time(sec) << " ===\n";
    }
    const double worker_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_worker).count();

    if (acc.count == 0) {
        std::cout << "\n[note] this worker claimed 0 nodes (workers > nodes?); "
                     "nothing written.\n";
        return 0;
    }

    // Write a single-shot-format spectrum whose /results holds THIS worker's
    // disorder mean, then append the /mc provenance the merge step needs. Base
    // context (σ=0) supplies the disorder-invariant metadata. Quadrature has no
    // per-worker mean (Σw_claimed≠1) → /results carries the partial weighted sum
    // as a placeholder; the authoritative payload is /mc/sum_obs.
    last.rho_xyz = (R.estimator == Estimator::Quadrature) ? acc.sum_obs
                                                          : acc.mean();
    SystemContext sys0(cfg);
    const HEOMConfig::Config cfg_out = output_config(cfg, stem);
    const std::string path =
        heom::output::write_spectrum(cfg_out, sys0, last, worker_sec);
    append_mc_provenance(path, acc, cfg.config_hash, R.disorder_identity);
    std::cout << "\nMC shard written: " << path
              << " (count=" << acc.count << ", worker time "
              << heom::console::format_computation_time(worker_sec) << ")\n";
    return 0;
}

int run_2des_mc(const HEOMConfig::Config& cfg, const MCOptions& opts) {
    RealizationSet R = load_realizations(opts.draws_file);   // rejects non-mean
    if (R.n_site != cfg.physics.n_states) {
        throw std::runtime_error(
            "run_2des_mc: draws n_site=" + std::to_string(R.n_site) +
            " != cfg n_states=" + std::to_string(cfg.physics.n_states));
    }

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

    const Interval iv = resolve_interval(opts, R.n_node);
    const std::string stem = shard_stem(cfg, opts, R.disorder_identity, iv);
    ClaimSource src(opts.queue, iv.begin1 - 1, iv.end1);   // 1-based → 0-based half-open
    Accumulator acc;
    heom::spectra::TwoDESResult last;           // template for shape/time axes
    bool inited = false;
    const auto t_worker = std::chrono::steady_clock::now();
    for (int r; (r = src.next()) >= 0; ) {      // r is 0-based (array/queue truth)
        std::cout << "\n=== MC disorder node " << (r + 1) << " / ["
                  << iv.begin1 << "," << iv.end1 << "]  (2des) ===\n";
        const auto t_node = std::chrono::steady_clock::now();
        // The same disorder drives both contexts. ESA uses the double-excited context.
        HEOMConfig::Config cr_double = cfg;
        apply_disorder(cr_double, R.row(r));               // basis stays DOUBLE_EXCITED
        SystemContext sys_double(cr_double);
        // SE/GSB = with-ground context, same δ_r on the site diagonal.
        HEOMConfig::Config cr_single = cfg;
        apply_disorder(cr_single, R.row(r));
        HEOMConfig::apply_basis(cr_single, BasisType::WITH_GROUND);
        SystemContext sys_single(cr_single);

        auto O = heom::spectra::compute_2des(sys_single, sys_double, req);
        if (!inited) {
            acc.init({static_cast<size_t>(O.n_component),
                      static_cast<size_t>(O.n_signal),
                      static_cast<size_t>(O.n_pathway),
                      static_cast<size_t>(O.n_tau),
                      static_cast<size_t>(O.n_tw),
                      static_cast<size_t>(O.n_t)},
                     R.estimator);
            inited = true;
        }
        // mean folds a bare sum (scale=1, exact no-op); quadrature folds w_r·O_r.
        const double scale =
            (R.estimator == Estimator::Quadrature) ? R.weight[r] : 1.0;
        acc.add(O.resp, scale, r);
        last = std::move(O);                   // keep the last for its axes/shape
        const double sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_node).count();
        std::cout << "=== node " << (r + 1) << " done in "
                  << heom::console::format_computation_time(sec) << " ===\n";
    }
    const double worker_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_worker).count();

    if (acc.count == 0) {
        std::cout << "\n[note] this worker claimed 0 nodes (workers > nodes?); "
                     "nothing written.\n";
        return 0;
    }

    // Same treatment as absorption: single-shot-format 2des file (mean in
    // /results/resp) + /mc provenance. The double-excited base context carries
    // the metadata superset (matches write_2des's single-shot contract).
    // Quadrature: /results/resp is the partial weighted-sum placeholder;
    // /mc/sum_obs is authoritative.
    last.resp = (R.estimator == Estimator::Quadrature) ? acc.sum_obs
                                                       : acc.mean();
    SystemContext sys0(cfg);
    const HEOMConfig::Config cfg_out = output_config(cfg, stem);
    const std::string path =
        heom::output::write_2des(cfg_out, sys0, last, worker_sec);
    append_mc_provenance(path, acc, cfg.config_hash, R.disorder_identity);
    std::cout << "\nMC 2des shard written: " << path
              << " (count=" << acc.count << ", worker time "
              << heom::console::format_computation_time(worker_sec) << ")\n";
    return 0;
}

} // namespace

int run_mc(const std::string& task_name,
           const HEOMConfig::Config& cfg,
           const MCOptions& opts) {
    // The queue counter and shard files live under output.<task>.save_path
    // (derived in main). Create it up front so the flock open and the spectrum
    // writer both land (the writer resolves the same save_path).
    const std::filesystem::path parent(cfg.output.save_path);
    if (!parent.empty()) std::filesystem::create_directories(parent);

    if (task_name == "absorption") {
        return run_absorption_mc(cfg, opts);
    }
    if (task_name == "2des") {
        return run_2des_mc(cfg, opts);
    }
    throw std::runtime_error(
        "MC static disorder does not apply to task '" + task_name +
        "' (dynamics runs single-shot; drop --rank)");
}

} // namespace heom::mc
