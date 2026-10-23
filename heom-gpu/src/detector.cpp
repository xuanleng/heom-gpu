#include "detector.hpp"
#include "heom_trajectory.hpp"
#include "core/cuda_common.cuh"
#include "core/dipole_apply.cuh"

#include <climits>
#include <stdexcept>
#include <string>

namespace heom {

Detector::Detector(const HEOMTrajectory& traj)
    : traj_(traj),
      N_(traj.sys().N()),
      M_(traj.sys().M()),
      max_batch_(traj.batch()) {}

void Detector::begin(const std::vector<Eigen::MatrixXcd>& mus_det,
                     size_t n_samples) {
    const int B_det = static_cast<int>(mus_det.size());
    if (B_det < 1 || B_det > max_batch_) {
        throw std::invalid_argument(
            "Detector::begin: B_det=" + std::to_string(B_det)
            + " out of range [1, " + std::to_string(max_batch_) + "]");
    }
    if (n_samples < 1 || n_samples > static_cast<size_t>(INT_MAX)) {
        throw std::invalid_argument(
            "Detector::begin: n_samples=" + std::to_string(n_samples)
            + " out of range [1, INT_MAX]");
    }
    for (int b = 0; b < B_det; ++b) {
        if (mus_det[b].rows() != N_ || mus_det[b].cols() != N_) {
            throw std::runtime_error(
                "Detector::begin: mu_det[" + std::to_string(b)
                + "] size mismatch (expected " + std::to_string(N_) + "x"
                + std::to_string(N_) + ")");
        }
    }

    batch_     = B_det;
    n_samples_ = n_samples;

    // Per-slot μ_det, packed slot-major [B_det·N²].
    const size_t N2 = static_cast<size_t>(N_) * N_;
    const size_t mu_need = static_cast<size_t>(B_det) * N2;
    d_mu_det_.ensure(mu_need);
    h_stage_.resize(mu_need);
    for (int b = 0; b < B_det; ++b) {
        const auto& mu = mus_det[b];
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < N_; ++j)
                h_stage_[static_cast<size_t>(b) * N2 + i * N_ + j] = c_make(
                    static_cast<real_t>(mu(i, j).real()),
                    static_cast<real_t>(mu(i, j).imag()));
    }
    d_mu_det_.copy_from_host(h_stage_.data(), mu_need);

    // Result grid [B_det·n_samples], slot-major (slot b at [b·n_samples..)).
    d_out_.ensure(static_cast<size_t>(B_det) * n_samples);
}

void Detector::sample(size_t idx) {
    if (idx >= n_samples_) {
        throw std::out_of_range(
            "Detector::sample: idx=" + std::to_string(idx)
            + " >= n_samples=" + std::to_string(n_samples_));
    }
    // grid.y = batch_; slot b traces its own α=0 block against its own μ_det.
    // The block is read NOW: it is the storage shape of the ρ being sampled.
    launch_trace_BR(d_mu_det_.get(), traj_.device_rho(), d_out_.get(),
                    static_cast<int>(idx), N_, M_, batch_,
                    static_cast<int>(n_samples_), traj_.support());
}

void Detector::flush(std::complex<double>* out_host) {
    const size_t total = static_cast<size_t>(batch_) * n_samples_;
    h_stage_.resize(total);
    d_out_.copy_to_host(h_stage_.data(), total);
    for (size_t i = 0; i < total; ++i)
        out_host[i] = std::complex<double>(c_re(h_stage_[i]), c_im(h_stage_[i]));
}

}  // namespace heom
