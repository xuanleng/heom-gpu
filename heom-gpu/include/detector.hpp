#ifndef HEOM_DETECTOR_HPP
#define HEOM_DETECTOR_HPP

#include "gpu_memory.hpp"
#include "core/precision.hpp"   // heom::complex_t (hier precision buffers)

#include <Eigen/Dense>
#include <complex>
#include <cstddef>
#include <vector>

class HEOMTrajectory;

namespace heom {

/**
 * @brief Device-side 2DES detection: Tr(μ_det^(b)·ρ_0^(b)) per slot, sampled
 * into a device grid, flushed to host once.
 *
 * Self-contained: owns the μ_det upload and the result grid, reads the
 * hierarchy through the trajectory's device_rho() + support() at sample time
 * (the compact layout depends on the current block, and in the batched walk
 * the block changes twice between begin() and the first sample()), and
 * touches nothing else — not propagation, not filtering. The trajectory it
 * was built from must outlive it.
 *
 * Usage (one tree):
 *   Detector det(traj);
 *   det.begin(mus_det, n_samples);   // upload B_det dipoles, size the grid
 *   ... det.sample(idx) after each t-point (async, no D2H) ...
 *   det.flush(out);                  // ONE D2H; out holds batch()·n_samples()
 *
 * Buffers are grow-only across begin() calls, so a detector reused over many
 * trees (the 2DES drivers do this) allocates once.
 */
class Detector {
public:
    explicit Detector(const HEOMTrajectory& traj);

    /// Upload B_det = mus_det.size() detection dipoles (one per slot) and size
    /// the result grid to [B_det·n_samples]. The detection width is DECOUPLED
    /// from the trajectory's active batch: sample() always launches at B_det.
    /// B_det must be in [1, trajectory batch]; n_samples in [1, INT_MAX] (the
    /// kernel indexes with int).
    void begin(const std::vector<Eigen::MatrixXcd>& mus_det, size_t n_samples);

    /// Sample every slot's trace into column idx (async, no sync / no D2H).
    /// Slot b reads its own compact α=0 block (b·M·Nb, Nb = traj.support().
    /// elems()) and its own μ_det (b·N²), writes out[b·n_samples+idx]. idx
    /// must be < n_samples().
    void sample(size_t idx);

    /// Copy the [B_det·n_samples] grid to host in ONE D2H. out_host must hold
    /// batch()·n_samples() scalars, slot-major: slot b at [b·n_samples..).
    void flush(std::complex<double>* out_host);

    /// Detection width / column count set by the last begin().
    int    batch()     const { return batch_; }
    size_t n_samples() const { return n_samples_; }

private:
    const HEOMTrajectory& traj_;   // device_rho() + support(), read at sample()
    const int N_, M_, max_batch_;

    GPUMemory<complex_t>   d_mu_det_;   // [B_det·N²] per-slot μ_det
    GPUMemory<complex_t>   d_out_;      // [B_det·n_samples] result grid
    std::vector<complex_t> h_stage_;    // host staging: μ_det pack (H2D), grid (D2H)

    int    batch_     = 1;
    size_t n_samples_ = 0;
};

}  // namespace heom

#endif  // HEOM_DETECTOR_HPP
