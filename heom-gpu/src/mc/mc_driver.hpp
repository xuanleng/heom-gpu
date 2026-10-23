#ifndef HEOM_MC_MC_DRIVER_HPP
#define HEOM_MC_MC_DRIVER_HPP

#include "heom_config.hpp"
#include "util/fnv.hpp"

#include <optional>
#include <string>

namespace heom::mc {

/**
 * @brief Resolved knobs for the MC static-disorder path.
 *
 * Derived in main.cpp from the config + the worker CLI flags. `draws_file` =
 * static_disorder.draws_file; `range` is the 1-based INCLUSIVE realization
 * interval this worker covers
 * (nullopt ⇒ the whole file [1, n_node]); `run_id` is
 * the shared timestamp (one per launch); `rank` is this worker's index. The
 * driver builds the output stem itself (it needs disorder_identity for the merge
 * key) and the queue path (see queue_file_name). main only reaches an MC worker
 * when `--rank` is given and a draws file exists.
 */
// 1-based INCLUSIVE realization interval, as the CLI speaks it (--r-begin /
// --r-end). The driver converts once to the queue's 0-based half-open form.
struct Interval { int begin1; int end1; };

struct MCOptions {
    std::string draws_file;        // disorder node file
    std::optional<Interval> range; // nullopt ⇒ whole node file [1, n_node]
    std::string run_id;            // shared timestamp TS (YYYYMMDD_HHMMSS)
    int rank = 0;                  // this worker's rank (>= 1)
};

// The flock counter shared by the workers of ONE batch, under save_path. The
// batch identity (run-id + resolved 1-based interval) is in the name, so a
// counter left by another batch can never be read: a fresh batch always starts
// at its own begin, and the workers of one batch (same --run-id, same range)
// always share. The heom-tools launcher builds the same name.
inline std::string queue_file_name(const std::string& run_id, const Interval& iv) {
    return "mc." + run_id + "_r" + std::to_string(iv.begin1) + "-" +
           std::to_string(iv.end1) + ".queue";
}

// Stable per-parameter merge key: fnv1a64_hex(config_hash + "|" + identity),
// 16 hex chars. Same physics ⇒ same key ⇒ shards group/merge across batches and
// timestamps; any parameter change flips the key. Inline (header-only fnv) so
// tests can pin its stability without linking the driver's translation unit.
inline std::string mc_merge_key(const std::string& config_hash,
                                const std::string& disorder_identity) {
    return heom::fnv1a64_hex(config_hash + "|" + disorder_identity);
}

/**
 * @brief MC worker for one spectroscopy. Clones the (already resolved) config
 * per realization, freezes disorder, runs the compute, and folds the result
 * into a shard (single-shot-format file + /mc provenance). The HEOM kernel is
 * untouched. Which one runs is a column of main.cpp's kTasks table; dynamics
 * has none (MC does not apply to it).
 */
int run_absorption_mc(const HEOMConfig::Config& cfg, const MCOptions& opts);
int run_fluorescence_mc(const HEOMConfig::Config& cfg, const MCOptions& opts);
int run_2des_mc(const HEOMConfig::Config& cfg, const MCOptions& opts);

} // namespace heom::mc

#endif // HEOM_MC_MC_DRIVER_HPP
