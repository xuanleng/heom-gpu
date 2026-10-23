#ifndef HEOM_ULIST_BUILDER_HPP
#define HEOM_ULIST_BUILDER_HPP

#include "gpu_memory.hpp"

#include <cstdint>
#include <vector>

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
 *   the RK4 stages in-place over U and freezes α∉U, so no full-grid zeroing
 *   is needed. (Freezing is exact for plain RK4; under IF-RK4 it skips the E2
 *   decay of those ADOs — a filter choice, applied identically by the full
 *   launch through device_flags().)
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
 * Built whenever the filter is on: the membership flags gate the full launch
 * too (HEOMContext::u_flags), so both launch domains freeze the same ADOs.
 * The compact list is used only when filter.ulist is on. Until the first
 * build() the list is the identity 0..M-1 (count = M). No step ever launches
 * through it: the fill-window steps after any ρ replacement run dense, and
 * every later step filters and calls build() before it launches.
 *
 * Per slot (2026-10-22-batch-per-slot): a batched trajectory keeps one flag
 * row per slot ([batch][M]), so build() marks one closure U_b per slot and
 * compacts them into ONE slot-major list of local ADO ids; slot b's segment
 * is [off[b], off[b+1]) (device_offsets()). Each slot then launches over its
 * own closure instead of the union of all slots. batch == 1 is the old
 * builder exactly.
 */
class UListBuilder {
public:
    // d_plus_map / d_minus_map: DEVICE pointers to the static neighbor maps
    // owned by SystemContext, each [M * D] with -1 for a missing neighbor,
    // identical indexing to compute_heom_derivatives_kernel:
    //   idx = alpha * D + mode_off[n] + k
    // The closure walk only needs the row width D, not the (n,k) structure.
    // Borrowed, not owned — SystemContext outlives every HEOMTrajectory and
    // therefore this builder.
    UListBuilder(int M, int D, int batch,
                 const int* d_plus_map, const int* d_minus_map);

    // Recompute U_b for the first `width` slots from their device flag rows
    // (active_flags[b·M..], all on-device) and return Σ_b |U_b|. The
    // (width+1)-int offset D2H is the only host synchronisation.
    int build(const int8_t* d_active_flags, int width);

    const int* device_list() const { return d_u_list_.get(); }
    // [width+1] segment offsets of the last build(): slot b's closure is
    // device_list()[off[b] .. off[b+1]).
    // nullptr after a width-1 build: the single segment is the whole list.
    const int* device_offsets() const { return width_ > 1 ? d_off_.get() : nullptr; }
    // [batch][M] U membership from the last build(): what a full launch gates
    // its epilogues on (HEOMContext::u_flags, row b for slot b).
    const uint8_t* device_flags() const { return d_e_flags_.get(); }
    int        count() const { return count_; }          // Σ_b |U_b|
    int        max_count() const { return max_count_; }  // max_b |U_b|
    // Build host+device block prefixes for flat launch: block_off[b+1] =
    // block_off[b] + ceil(|U_b|·Nb/block_size). Returns Σ (= grid.x).
    int        prepare_launch(int Nb, int block_size = 256);
    // [width+1] device; valid after prepare_launch. nullptr when width==1.
    const int* device_block_off() const {
        return width_ > 1 ? d_block_off_.get() : nullptr;
    }

private:
    int M_;
    int channels_;                  // D, row width of the neighbor maps
    int batch_;                     // slots the buffers are sized for
    int count_;                     // current Σ_b |U_b|
    int width_ = 1;                 // width of the last build()
    int max_count_ = 0;             // current max_b |U_b| (launch grid.x)
    const int* d_plus_map_;         // borrowed from SystemContext
    const int* d_minus_map_;        // borrowed from SystemContext
    GPUMemory<uint8_t> d_e_flags_;  // [batch·M] closure marks, reused per build()
    GPUMemory<int>     d_u_list_;   // [batch·M] device compact list (slot-major)
    GPUMemory<int>     d_ids_;      // [batch·M] constant 0..M-1 per slot (DeviceSelect input)
    GPUMemory<int>     d_num_;      // [1] Σ|U_b| written by DeviceSelect
    GPUMemory<int>     d_count_;    // [batch] |U_b|
    GPUMemory<int>     d_off_;      // [batch+1] segment offsets
    std::vector<int>   h_off_;      // host twin of d_off_
    GPUMemory<int>     d_block_off_; // [batch+1] flat-launch block prefixes
    GPUMemory<uint8_t> d_temp_;     // cub temp storage (size queried once)
};

#endif // HEOM_ULIST_BUILDER_HPP
