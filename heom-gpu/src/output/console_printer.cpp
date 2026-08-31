#include "console_printer.hpp"

#include "util/stream_format.hpp"

#include <complex>
#include <iomanip>
#include <iostream>

namespace heom::console {

void print_eigenspectrum(const Eigen::VectorXd& eigenvalues,
                         int selected_idx, int N, double w0_cm) {
    const heom::StreamFormat fmt_guard(std::cout);
    std::cout << "\n  Eigenspectrum:\n"
              << "    (Energies: absolute = E + w0, relative = E - w0, w0 = "
              << std::fixed << std::setprecision(1) << w0_cm << " cm^-1)\n\n";
    for (int i = 0; i < N; ++i) {
        double E_rel_cm = eigenvalues(i) / CM_TO_AU;
        double E_abs_cm = E_rel_cm + w0_cm;
        std::cout << "    [" << i << "] "
                  << "E_abs = " << std::setw(8) << std::setprecision(1) << std::fixed << E_abs_cm << " cm^-1, "
                  << "E_rel = " << std::setw(7) << std::setprecision(1) << std::fixed << E_rel_cm << " cm^-1";
        if (i == 0) std::cout << " (lowest)";
        if (i == N - 1) std::cout << " (highest)";
        if (i == selected_idx) std::cout << " <- Selected";
        std::cout << "\n";
    }
}

void print_exciton_composition(const Eigen::MatrixXcd& eigenvectors,
                               int exciton_idx, int N) {
    const heom::StreamFormat fmt_guard(std::cout);
    std::cout << "\n  Exciton #" << exciton_idx << " composition:\n";
    for (int i = 0; i < N; ++i) {
        std::complex<double> amp = eigenvectors(i, exciton_idx);
        double prob = std::norm(amp);
        if (prob > 0.001) {
            std::cout << "    site" << (i + 1) << ": amplitude = "
                      << std::setw(8) << std::setprecision(4) << std::fixed << amp.real();
            if (std::abs(amp.imag()) > 1e-6)
                std::cout << " + " << amp.imag() << "i";
            std::cout << ", probability = " << std::setw(5) << std::setprecision(1)
                      << std::fixed << (prob * 100) << "%\n";
        }
    }
}

void print_density_matrix_summary(const Eigen::MatrixXcd& rho, int N) {
    const heom::StreamFormat fmt_guard(std::cout);
    std::cout << "\n  Initial density matrix rho(0) in site basis:";
    std::cout << "\n    Diagonal (populations):";
    for (int i = 0; i < N; ++i) {
        double pop = std::abs(rho(i, i));
        if (pop > 0.01)
            std::cout << " rho(" << i << "," << i << ")=" << std::fixed << std::setprecision(3) << pop;
    }
    std::cout << "\n    Off-diagonal (coherences):";
    int count = 0;
    for (int i = 0; i < N && count < 5; ++i) {
        for (int j = i + 1; j < N && count < 5; ++j) {
            double coh = std::abs(rho(i, j));
            if (coh > 0.01) {
                std::cout << " |rho(" << i << "," << j << ")|=" << std::fixed << std::setprecision(3) << coh;
                if (++count >= 5) std::cout << " ...";
            }
        }
    }
}

}  // namespace heom::console
