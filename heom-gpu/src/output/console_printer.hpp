#ifndef HEOM_OUTPUT_CONSOLE_PRINTER_HPP
#define HEOM_OUTPUT_CONSOLE_PRINTER_HPP

// Console pretty-printers extracted from heom_solver.cpp.
//
// These helpers were previously file-static functions in heom_solver.cpp.
// They are pure formatters with no solver state, so isolating them into
// their own translation unit:
//   - keeps heom_solver.cpp focused on numerical orchestration,
//   - makes the formatters reusable from other entry points (e.g. tools/),
//   - lets the unit-test target avoid recompiling them when only solver
//     orchestration changes.
//
// All functions write to std::cout and never throw. Numeric formatting
// (precision, units cm^-1, w0 offset) matches the pre-extraction output
// byte-for-byte so that existing log scrapers / golden-stdout checks
// continue to work.

#include "types.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <string>

namespace heom::console {

// Human-readable wall-clock duration. Tiers, by magnitude:
//   < 1 min   : "S.SSSs"          (sub-minute, 3-decimal seconds)
//   < 1 hour  : "Mm Ss"
//   < 24 hour : "Hh Mm Ss"
//   >= 24 hour: "Dd Hh Mm Ss"
// Pure formatter, no state. Shared by the per-run solver summary and by
// task-level totals (absorption x/y/z, and 2DES later). The sub-hour output
// is byte-identical to the original (hms / ms / %.3fs) so existing log
// scrapers keep working; only the day tier is new.
inline std::string format_computation_time(double seconds) {
    char buf[80];
    const long total = static_cast<long>(seconds);
    const int  d = static_cast<int>(total / 86400);
    const int  h = static_cast<int>((total % 86400) / 3600);
    const int  m = static_cast<int>((total % 3600) / 60);
    const int  s = static_cast<int>(total % 60);

    if (d > 0) {
        snprintf(buf, sizeof(buf), "%dd %dh %dm %ds", d, h, m, s);
    } else if (h > 0) {
        snprintf(buf, sizeof(buf), "%dh %dm %ds", h, m, s);
    } else if (m > 0) {
        snprintf(buf, sizeof(buf), "%dm %ds", m, s);
    } else {
        snprintf(buf, sizeof(buf), "%.3fs", seconds);
    }
    return buf;
}

// Print the single-exciton block of the eigenspectrum, numbered 1..n like
// the YAML exciton_index.
//
//   eigenvalues  : SystemContext::eigenvalues() (a.u., relative to w0),
//                  ascending within [lo, lo + n).
//   lo, n        : the single-excitation manifold's Hilbert range.
//   selected_idx : Hilbert index marked with "<- Selected" (-1: none).
//   w0_cm        : reference frequency w0 in cm^-1, used for the absolute
//                  energy column (E_abs = E_rel + w0).
void print_eigenspectrum(const Eigen::VectorXd& eigenvalues,
                         int lo, int n, int selected_idx, double w0_cm);

// Print the pigment composition of eigenvector column `exciton_idx` over the
// single-excitation rows [lo, lo + n), listing only pigments whose
// probability exceeds 0.1%.
void print_exciton_composition(const Eigen::MatrixXcd& eigenvectors,
                               int exciton_idx, int lo, int n);

// Print a one-line summary of the initial site-basis density matrix:
// non-trivial diagonals (populations > 0.01) and the first few
// off-diagonal coherences (>0.01, capped at 5).
void print_density_matrix_summary(const Eigen::MatrixXcd& rho, int N);

}  // namespace heom::console

#endif  // HEOM_OUTPUT_CONSOLE_PRINTER_HPP
