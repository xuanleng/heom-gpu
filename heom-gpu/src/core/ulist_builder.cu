#include "ulist_builder.hpp"
#include "cuda_common.cuh"
#include "core/mode_layout.hpp"

// CUB ships with the CUDA toolkit (CCCL). NOTE: cub's CountingInputIterator
// header was removed in CCCL 3.x / CUDA 13, so the identity input for
// DeviceSelect is a plain device array (d_ids_, filled 0..M-1 once in the
// ctor) instead of a fancy iterator. Reading M ints per build is noise next
// to the scatter kernel.
#include <cub/device/device_select.cuh>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

// E = A ∪ neighbors(A). One thread per ADO: an active α marks itself and
// every existing plus/minus neighbor. Multiple threads may write the same
// e_flags byte concurrently — all of them write 1, so whichever store lands
// is correct (benign same-value race; no atomics needed).
//
// Per slot: grid.y = b reads slot b's flags active[b·M..] and marks its own
// closure e_flags[b·M..] — each slot launches over ITS closure only.
__global__ void mark_closure_kernel(
    const int8_t* __restrict__ active_all,
    const int*    __restrict__ plus_map,
    const int*    __restrict__ minus_map,
    uint8_t* e_flags_all,
    int M,
    int channels)
{
    const size_t boff = static_cast<size_t>(blockIdx.y) * M;
    const int8_t* active = active_all + boff;
    uint8_t* e_flags = e_flags_all + boff;
    int alpha = blockIdx.x * blockDim.x + threadIdx.x;
    if (alpha >= M || !active[alpha]) return;

    e_flags[alpha] = 1u;
    const size_t base = static_cast<size_t>(alpha) * channels;
    for (int c = 0; c < channels; ++c) {
        const int p = plus_map[base + c];
        if (p >= 0) e_flags[p] = 1u;
        const int m = minus_map[base + c];
        if (m >= 0) e_flags[m] = 1u;
    }
}

// count[b] = |U_b| = Σ_α e_flags[b·M + α] (grid.y = b).
__global__ void count_closure_kernel(const uint8_t* __restrict__ e_flags,
                                     int M, int* __restrict__ count)
{
    // One atomic per block (block-wide count), not per element: all slots'
    // counts sit in 21 words, so per-element atomics would serialise there.
    const int b = blockIdx.y;
    const int alpha = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = __syncthreads_count(
        alpha < M && e_flags[static_cast<size_t>(b) * M + alpha]);
    if (threadIdx.x == 0 && c > 0) atomicAdd(&count[b], c);
}

// off[0] = 0, off[b+1] = off[b] + count[b]: slot b's segment of the
// concatenated list. One thread; width ≤ 21.
__global__ void closure_offsets_kernel(const int* __restrict__ count,
                                       int width, int* __restrict__ off)
{
    off[0] = 0;
    for (int b = 0; b < width; ++b) off[b + 1] = off[b] + count[b];
}

}  // namespace

UListBuilder::UListBuilder(int M, int D, int batch,
                           const int* d_plus_map, const int* d_minus_map)
    : M_(M)
    , channels_(D)
    , batch_(batch)
    , count_(M)
    , d_plus_map_(d_plus_map)
    , d_minus_map_(d_minus_map)
    , d_e_flags_(static_cast<size_t>(batch) * M)
    , d_u_list_(static_cast<size_t>(batch) * M)
    , d_ids_(static_cast<size_t>(batch) * M)
    , d_num_(1)
    , d_count_(static_cast<size_t>(batch))
    , d_off_(static_cast<size_t>(batch) + 1)
    , d_block_off_(static_cast<size_t>(batch) + 1)
    , h_off_(static_cast<size_t>(batch) + 1, 0)
{
    // cub temp-storage size query (nullptr pass does no work, only sizes).
    // Sized for the widest build, batch·M flags.
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_bytes, d_ids_.get(), d_e_flags_.get(),
        d_u_list_.get(), d_num_.get(), batch_ * M_));
    d_temp_ = GPUMemory<uint8_t>(temp_bytes);

    // d_ids_ = [0..M-1] repeated per slot: the constant input for
    // DeviceSelect, so the compacted list holds LOCAL ADO ids, slot-major and
    // ascending inside each slot. The first M also serve as the
    // pre-first-build u_list (identity ⇒ full launch; no step launches
    // through it, see ulist_builder.hpp).
    std::vector<int> ids(static_cast<size_t>(batch_) * M_);
    for (int b = 0; b < batch_; ++b)
        std::iota(ids.begin() + static_cast<size_t>(b) * M_,
                  ids.begin() + static_cast<size_t>(b + 1) * M_, 0);
    d_ids_.copy_from_host(ids.data(), ids.size());
    d_u_list_.copy_from_host(ids.data(), ids.size());
}

int UListBuilder::build(const int8_t* d_active_flags, int width) {
    if (width < 1 || width > batch_) {
        throw std::invalid_argument("UListBuilder::build: width out of range");
    }
    const size_t n = static_cast<size_t>(width) * M_;
    CUDA_CHECK(cudaMemset(d_e_flags_.get(), 0, n));

    const dim3 grid((M_ + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE, width);
    mark_closure_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
        d_active_flags, d_plus_map_, d_minus_map_,
        d_e_flags_.get(), M_, channels_);
    CUDA_CHECK_KERNEL();

    // Stable compaction over all width·M flags → slot-major, ascending
    // local ADO ids inside each slot (preserves per-ADO memory locality in
    // the strided U-list kernels, same order as the old host loop produced).
    size_t temp_bytes = d_temp_.bytes();
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        d_temp_.get(), temp_bytes, d_ids_.get(), d_e_flags_.get(),
        d_u_list_.get(), d_num_.get(), static_cast<int>(n)));

    width_ = width;
    if (width == 1) {
        // One slot: |U| is the select count; no segment kernels, one 4-byte
        // D2H as before (device_offsets() is nullptr ⇒ shared-list launch).
        CUDA_CHECK(cudaMemcpy(&count_, d_num_.get(), sizeof(int),
                              cudaMemcpyDeviceToHost));
        max_count_ = count_;
        h_off_[0] = 0; h_off_[1] = count_;
        return count_;
    }

    CUDA_CHECK(cudaMemset(d_count_.get(), 0, width * sizeof(int)));
    count_closure_kernel<<<grid, HEOM_BLOCK_SIZE>>>(
        d_e_flags_.get(), M_, d_count_.get());
    CUDA_CHECK_KERNEL();
    closure_offsets_kernel<<<1, 1>>>(d_count_.get(), width, d_off_.get());
    CUDA_CHECK_KERNEL();

    // The only host traffic per build: one (width+1)-int D2H of the segment
    // offsets (they feed ctx.u_count = max_b |U_b| and the adaptive gate's
    // Σ|U_b| / (width·M) fraction host-side). width == 1 ⇒ off = {0, |U|}.
    CUDA_CHECK(cudaMemcpy(h_off_.data(), d_off_.get(), (width + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    count_ = h_off_[width];
    max_count_ = 0;
    for (int b = 0; b < width; ++b)
        max_count_ = std::max(max_count_, h_off_[b + 1] - h_off_[b]);
    return count_;
}

int UListBuilder::prepare_launch(int Nb, int block_size)
{
    if (Nb < 1 || block_size < 1 || width_ < 1) return 0;
    std::vector<int> h(static_cast<size_t>(width_) + 1, 0);
    for (int b = 0; b < width_; ++b) {
        const int n = h_off_[b + 1] - h_off_[b];
        const int blocks_b = static_cast<int>(
            (static_cast<long long>(n) * Nb + block_size - 1) / block_size);
        h[b + 1] = h[b] + blocks_b;
    }
    CUDA_CHECK(cudaMemcpy(d_block_off_.get(), h.data(),
                          (width_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    return h[width_];
}

