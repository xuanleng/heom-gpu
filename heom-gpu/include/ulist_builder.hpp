#ifndef HEOM_ULIST_BUILDER_HPP
#define HEOM_ULIST_BUILDER_HPP

#include "gpu_memory.hpp"

#include <cstdint>

/**
 * @brief Builds the compact launch set U = E = A ∪ neighbors(A) on the
 *        device and exposes it as a device int array for the U-list kernels.
 *
 * U-list whole-step launch
 * The launch
 * domain of compute_heom_derivatives is shrunk from M to |U|; ρ storage and
 * neighbor addressing are untouched (the kernel maps slot → α = u_list[slot]).
 *
 * Correctness contract (why U = E, not just A):
 *   The full kernel writes a nonzero derivative for α iff α is active (self
 *   terms) OR α has an active neighbor (Term 4/5). The latter set is exactly
 *   neighbors(A). So {α : dρ_α ≠ 0} ⊆ A ∪ neighbors(A) = E. Launching U = E
 *   covers every α whose derivative is nonzero; the whole-step path then folds
 *   the RK4 stages in-place over U and freezes α∉U (their ρ is provably
 *   unchanged this interval), so no full-grid zeroing is needed.
 *
 * Construction is device-side: a scatter kernel marks E from active_flags
 * through the neighbor maps (which already live on the GPU, borrowed from
 * SystemContext — no host mirror is kept), then cub::DeviceSelect::Flagged
 * compacts the flags into ascending ADO ids (stable, preserves per-ADO
 * memory locality). The only per-build host traffic is one 4-byte D2H of
 * |U| — the previous host-side builder round-tripped the full M-byte flags
 * array, ran an O(M·channels) serial loop, uploaded the list back, and held
 * permanent host copies of both neighbor maps (2·M·channels·4 B of RAM).
 *
 * Opt-in via the time_evolution.ulist config flag. Until the first build()
 * the list is the identity 0..M-1 (count = M), i.e. a full launch — so
 * warmup steps before the first filter pass stay byte-equal to the original
 * path.
 */
class UListBuilder {
public:
    // d_plus_map / d_minus_map: DEVICE pointers to the static neighbor maps
    // owned by SystemContext, each [M * n_bath * (K+1)] with -1 for a missing
    // neighbor, identical indexing to compute_heom_derivatives_kernel:
    //   idx = alpha * n_bath * (K+1) + n * (K+1) + k
    // Borrowed, not owned — SystemContext outlives every HEOMTrajectory and
    // therefore this builder.
    UListBuilder(int M, int n_bath, int K,
                 const int* d_plus_map, const int* d_minus_map);

    // Recompute U from one device active_flags snapshot (all on-device) and
    // return |U|. The 4-byte count D2H is the only host synchronisation.
    int build(const int8_t* d_active_flags);

    const int* device_list() const { return d_u_list_.get(); }
    int        count() const { return count_; }

private:
    int M_;
    int channels_;                  // n_bath * (K+1)
    int count_;                     // current |U|
    const int* d_plus_map_;         // borrowed from SystemContext
    const int* d_minus_map_;        // borrowed from SystemContext
    GPUMemory<uint8_t> d_e_flags_;  // [M] closure marks, reused per build()
    GPUMemory<int>     d_u_list_;   // [M] device compact list
    GPUMemory<int>     d_ids_;      // [M] constant 0..M-1 (DeviceSelect input)
    GPUMemory<int>     d_num_;      // [1] |U| written by DeviceSelect
    GPUMemory<uint8_t> d_temp_;     // cub temp storage (size queried once)
};

#endif // HEOM_ULIST_BUILDER_HPP
