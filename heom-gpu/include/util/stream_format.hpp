#ifndef HEOM_UTIL_STREAM_FORMAT_HPP
#define HEOM_UTIL_STREAM_FORMAT_HPP

#include <ios>
#include <ostream>

namespace heom {

// Restores a stream's sticky formatting state on scope exit.
//
// Format flags (fixed/scientific/hex/left/boolalpha/...), precision and fill
// are STICKY: set them on a shared stream and every later insertion by anyone
// inherits them. std::setw is the exception -- it resets itself after one
// insertion -- which is why the leak is easy to miss in review.
//
// A function that sets them on std::cout without restoring silently reformats
// the rest of the process's output. The failure mode is nastier than a crash:
// with `fixed` + precision(1) still in effect, a test printing a 3.33e-03
// discrepancy reports "0.0", which reads like a passing value. That cost real
// debugging time on the sink kernel test.
//
// Usage: declare one at the top of any function that sets flags on a stream it
// does not own.
//
//     void print_summary() {
//         const heom::StreamFormat guard(std::cout);
//         std::cout << std::fixed << std::setprecision(3) << x << "\n";
//     }   // <- cout's previous flags/precision/fill are back
//
// Prefer formatting into a local std::ostringstream where practical: state
// that is never shared cannot leak, and needs no guard.
// An optional log sink (`std::ostream* log_`, nullptr => silent) is a common
// shape here, so the pointer overload treats nullptr as "nothing to guard".
// Callers never need to wrap the guard in a conditional or a smart pointer.
class StreamFormat {
public:
    explicit StreamFormat(std::ostream& os) : StreamFormat(&os) {}

    explicit StreamFormat(std::ostream* os)
        : os_(os),
          flags_(os ? os->flags() : std::ios_base::fmtflags{}),
          precision_(os ? os->precision() : 0),
          fill_(os ? os->fill() : ' ') {}

    ~StreamFormat() {
        if (!os_) return;
        os_->flags(flags_);
        os_->precision(precision_);
        os_->fill(fill_);
    }

    StreamFormat(const StreamFormat&) = delete;
    StreamFormat& operator=(const StreamFormat&) = delete;

private:
    std::ostream* os_;
    std::ios_base::fmtflags flags_;
    std::streamsize precision_;
    char fill_;
};

} // namespace heom

#endif // HEOM_UTIL_STREAM_FORMAT_HPP
