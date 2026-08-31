#ifndef HEOM_MC_MC_DRIVER_HPP
#define HEOM_MC_MC_DRIVER_HPP

#include "heom_config.hpp"
#include "util/fnv.hpp"

#include <string>

namespace heom::mc {

/**
 * @brief Resolved knobs for the MC static-disorder path.
 *
 * Derived in main.cpp from the config + the worker CLI flags. `draws_file` =
 * static_disorder.draws_file; `queue` = output.<task>.save_path + "mc.queue";
 * `[r_begin, r_end]` is the 1-based INCLUSIVE realization interval this worker
 * covers (0/0 sentinel ⇒ driver uses the whole file [1, n_node]); `run_id` is
 * the shared timestamp (one per launch); `rank` is this worker's index. The
 * driver builds the output stem itself (it needs disorder_identity for the merge
 * key). main only reaches run_mc when `--rank` is given and a draws file exists.
 */
struct MCOptions {
    std::string draws_file;  // disorder node file
    std::string queue;       // flock counter file (shared by all workers)
    int r_begin = 0;         // 1-based inclusive start (0 = sentinel ⇒ whole file)
    int r_end = 0;           // 1-based inclusive end   (0 = sentinel ⇒ whole file)
    std::string run_id;      // shared timestamp TS (YYYYMMDD_HHMM)
    int rank = 0;            // this worker's rank
};

// Stable per-parameter merge key: fnv1a64_hex(config_hash + "|" + identity),
// 16 hex chars. Same physics ⇒ same key ⇒ shards group/merge across batches and
// timestamps; any parameter change flips the key. Inline (header-only fnv) so
// tests can pin its stability without linking the driver's translation unit.
inline std::string mc_merge_key(const std::string& config_hash,
                                const std::string& disorder_identity) {
    return heom::fnv1a64_hex(config_hash + "|" + disorder_identity);
}

/**
 * @brief MC dispatch by task name. Clones the (already resolved) config per
 * realization, freezes disorder, runs the spectroscopy compute, and folds the
 * result into a bare-sum partial file. The HEOM kernel is untouched.
 *
 * @throws std::runtime_error for dynamics (MC does not apply to dynamics) or
 *         any unknown task.
 */
int run_mc(const std::string& task_name,
           const HEOMConfig::Config& cfg,
           const MCOptions& opts);

} // namespace heom::mc

#endif // HEOM_MC_MC_DRIVER_HPP
