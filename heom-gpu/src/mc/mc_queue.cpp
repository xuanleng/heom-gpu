#include "mc/mc_queue.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace heom::mc {

namespace {

[[noreturn]] void fail(const std::string& what, const std::string& path) {
    throw std::runtime_error("mc::Queue " + what + " '" + path + "': " +
                             std::strerror(errno));
}

// Not an OS error (errno is meaningless here): the file has bytes in it that
// cannot be a counter we wrote. Refuse rather than guess — a guessed `next`
// hands the same realization to two workers or skips a run of them, and that
// only surfaces at merge time as duplicate/missing r_done, after the GPU hours
// are spent.
[[noreturn]] void corrupt(const std::string& why, const std::string& path) {
    throw std::runtime_error("mc::Queue file corrupt (" + why + "): '" + path +
                             "' -- delete it to restart the interval");
}

} // namespace

Queue::Queue(const std::string& path, int begin0, int end0)
    : path_(path), begin0_(begin0), end0_(end0) {
    if (begin0_ < 0 || end0_ < begin0_) {
        throw std::invalid_argument("mc::Queue requires 0 <= begin0 <= end0");
    }
    // Create-if-absent; leave an existing (resumed) counter untouched.
    int fd = ::open(path_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) fail("cannot open", path_);
    ::close(fd);
}

std::optional<int> Queue::claim() {
    int fd = ::open(path_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) fail("cannot open", path_);

    if (::flock(fd, LOCK_EX) != 0) { ::close(fd); fail("flock failed on", path_); }

    std::int64_t next = 0;
    ssize_t got = ::pread(fd, &next, sizeof(next), 0);
    if (got < 0) { ::flock(fd, LOCK_UN); ::close(fd); fail("pread failed on", path_); }
    // Exactly two legal shapes: empty (fresh) or one whole int64. A short read
    // leaves the high bytes of `next` at whatever they were — silently wrong.
    if (got != 0 && got != static_cast<ssize_t>(sizeof(next))) {
        ::flock(fd, LOCK_UN); ::close(fd);
        corrupt("short read, " + std::to_string(got) + " of " +
                std::to_string(sizeof(next)) + " bytes", path_);
    }
    if (got == 0) next = begin0_;   // fresh/empty file ⇒ start at begin0
    // next < end0_ (an int) is checked below, so the only unrepresentable
    // value left is a negative one.
    if (next < 0) {
        ::flock(fd, LOCK_UN); ::close(fd);
        corrupt("negative counter " + std::to_string(next), path_);
    }

    std::optional<int> result;
    if (next < end0_) {
        std::int64_t updated = next + 1;
        if (::pwrite(fd, &updated, sizeof(updated), 0) != (ssize_t)sizeof(updated)) {
            ::flock(fd, LOCK_UN); ::close(fd); fail("pwrite failed on", path_);
        }
        result = static_cast<int>(next);
    }

    ::flock(fd, LOCK_UN);
    ::close(fd);
    return result;
}

} // namespace heom::mc
