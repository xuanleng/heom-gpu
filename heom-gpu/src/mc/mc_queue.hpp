#ifndef HEOM_MC_MC_QUEUE_HPP
#define HEOM_MC_MC_QUEUE_HPP

#include <optional>
#include <string>

namespace heom::mc {

/**
 * @brief Dumb counter queue guarded by POSIX flock.
 *
 * The queue file holds a single int64 = the next realization index to hand out.
 * `claim()` takes an exclusive lock, reads `next` (seeding it to `begin0` when
 * the file is empty), and — if `next < end0` — increments+writes it back and
 * returns the old value; otherwise nullopt (drained). The whole read-modify-write is
 * under LOCK_EX, so any number of processes pointing at the same file partition
 * the 0-based half-open interval [begin0, end0) with no gaps and no dups.
 *
 * The interval is the machine-internal truth: the CLI speaks 1-based inclusive
 * `[begin, end]` and the driver converts once via [begin-1, end).
 */
class Queue {
public:
    // Creates the file (empty) if it does not exist. First claim() seeds begin0.
    Queue(const std::string& path, int begin0, int end0);

    // Next index in [begin0, end0); nullopt when the queue is drained.
    std::optional<int> claim();

private:
    std::string path_;
    int begin0_;
    int end0_;
};

} // namespace heom::mc

#endif // HEOM_MC_MC_QUEUE_HPP
