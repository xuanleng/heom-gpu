#ifndef HEOM_TESTS_HARNESS_HPP
#define HEOM_TESTS_HARNESS_HPP

// The one test harness for every C++ test binary. In-house on purpose
// (in-house harness over Catch2): a test is a `void f()` that calls TEST() once,
// any number of ASSERT_*(), and PASS() at the end; an ASSERT_* that fails
// prints why and returns from the test without PASS(), so the test counts as
// run-but-not-passed. Suites that keep the older `bool test_x()` shape are
// counted through RUN_BOOL(). main() ends with `return heom_test_summary(...)`.
// One failure semantics everywhere: exit 0 iff passed == run.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

#include <unistd.h>

namespace heom_test {
inline int& run()    { static int n = 0; return n; }
inline int& passed() { static int n = 0; return n; }

// Every file a test writes goes under this directory: one per process, so two
// binaries (gpu_kernel / gpu_kernel_mp) or two `ctest -j` jobs never share a
// path. Removed at normal exit; an aborted run leaves it for inspection.
inline const std::string& tmp_dir() {
    static const struct Dir {
        std::string path;
        Dir() {
            const char* base = std::getenv("TMPDIR");
            path = std::string(base && *base ? base : "/tmp") + "/heom_test_XXXXXX";
            if (!mkdtemp(path.data())) { std::perror("mkdtemp"); std::abort(); }
        }
        ~Dir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    } dir;
    return dir.path;
}
inline std::string tmp(const std::string& name) { return tmp_dir() + "/" + name; }
} // namespace heom_test

#define TEST(name) do { \
    ++heom_test::run(); \
    std::cout << "  " << name << "... " << std::flush; \
} while (0)

#define PASS() do { ++heom_test::passed(); std::cout << "PASS\n"; } while (0)

// A test that cannot run in this environment (missing tool, no GPU) says so
// and is not counted — a skip is neither a pass nor a failure.
#define SKIP(reason) do { \
    --heom_test::run(); \
    std::cout << "SKIP (" << reason << ")\n"; \
    return; \
} while (0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { std::cout << "FAIL\n    " #cond " is false\n"; return; } \
} while (0)

// Both sides must be finite: NaN fails every `>` and Inf - Inf is NaN, so a
// bare `|a-b| > tol` lets both through. Test non-finite values with ASSERT_TRUE.
#define ASSERT_NEAR(a, b, tol) do { \
    double _a = (a), _b = (b); \
    if (!std::isfinite(_a) || !std::isfinite(_b) || !(std::abs(_a - _b) <= (tol))) { \
        std::cout << "FAIL\n    expected " << _b << " got " << _a \
                  << " (diff=" << std::abs(_a - _b) << ")\n"; \
        return; \
    } \
} while (0)

#define ASSERT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        std::cout << "FAIL\n    expected " << _b << " got " << _a << "\n"; \
        return; \
    } \
} while (0)

#define ASSERT_NE(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a == _b) { \
        std::cout << "FAIL\n    expected != " << _b << " but both equal\n"; \
        return; \
    } \
} while (0)

#define ASSERT_THROWS(expr) do { \
    bool _caught = false; \
    try { expr; } catch (const std::exception&) { _caught = true; } \
    if (!_caught) { \
        std::cout << "FAIL\n    expected exception, none thrown\n"; \
        return; \
    } \
} while (0)

#define ASSERT_NOTHROW(expr) do { \
    try { expr; } catch (const std::exception& _e) { \
        std::cout << "FAIL\n    threw std::exception: " << _e.what() << "\n"; \
        return; \
    } catch (...) { \
        std::cout << "FAIL\n    threw unknown exception\n"; \
        return; \
    } \
} while (0)

// Adapter for `bool test_x()` suites: the function prints its own diagnostics
// and returns pass/fail; the harness only counts. Runs every test even after a
// failure so one run reports all failures.
#define RUN_BOOL(fn) do { \
    ++heom_test::run(); \
    std::cout << "  " #fn "... " << std::flush; \
    if (fn()) { ++heom_test::passed(); std::cout << "PASS\n"; } \
    else      { std::cout << "FAIL\n"; } \
} while (0)

// One exit code policy for every suite.
inline int heom_test_summary(const char* suite) {
    const int r = heom_test::run(), p = heom_test::passed();
    std::cout << "\n=== " << suite << ": " << p << "/" << r << " passed ===\n\n";
    return (r > 0 && p == r) ? 0 : 1;
}

#endif // HEOM_TESTS_HARNESS_HPP
