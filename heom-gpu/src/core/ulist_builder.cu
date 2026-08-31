#include "ulist_builder.hpp"
#include "cuda_common.cuh"

// CUB ships with the CUDA toolkit (CCCL). NOTE: cub's CountingInputIterator
// header was removed in CCCL 3.x / CUDA 13, so the identity input for
// DeviceSelect is a plain device array (d_ids_, filled 0..M-1 once in the
// ctor) instead of a fancy iterator. Reading M ints per build is noise next
// to the scatter kernel.
#include <cub/device/device_select.cuh>

#include <numeric>
#include <vector>

namespace {

// E = A ∪ neighbors(A). One thread per ADO: an active α marks itself and
// every existing plus/minus neighbor. Multiple threads may write the same
// e_flags byte concurrently — all of them write 1, so whichever store lands
// is correct (benign same-value race; no atomics needed).
__global__ void mark_closure_kernel(
    const int8_t* __restrict__ active,
    const int*    __restrict__ plus_map,
    const int*    __restrict__ minus_map,
    uint8_t* e_flags,
    int M,
    int channels)
{
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

}  // namespace

UListBuilder::UListBuilder(int M, int n_bath, int K,
                           const int* d_plus_map, const int* d_minus_map)
    : M_(M)
    , channels_(n_bath * (K + 1))
    , count_(M)
    , d_plus_map_(d_plus_map)
    , d_minus_map_(d_minus_map)
    , d_e_flags_(static_cast<size_t>(M))
    , d_u_list_(static_cast<size_t>(M))
    , d_ids_(static_cast<size_t>(M))
    , d_num_(1)
{
    // cub temp-storage size query (nullptr pass does no work, only sizes).
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_bytes, d_ids_.get(), d_e_flags_.get(),
        d_u_list_.get(), d_num_.get(), M_));
    d_temp_ = GPUMemory<uint8_t>(temp_bytes);

    // d_ids_ = 0..M-1, the constant identity input for DeviceSelect.
    // The same array also serves as the pre-first-build u_list (identity ⇒
    // full launch, byte-equal to the original path during warmup steps).
    std::vector<int> identity(static_cast<size_t>(M_));
    std::iota(identity.begin(), identity.end(), 0);
    d_ids_.copy_from_host(identity.data(), identity.size());
    d_u_list_.copy_from_host(identity.data(), identity.size());
}

int UListBuilder::build(const int8_t* d_active_flags) {
    CUDA_CHECK(cudaMemset(d_e_flags_.get(), 0, static_cast<size_t>(M_)));

    const int blocks = (M_ + HEOM_BLOCK_SIZE - 1) / HEOM_BLOCK_SIZE;
    mark_closure_kernel<<<blocks, HEOM_BLOCK_SIZE>>>(
        d_active_flags, d_plus_map_, d_minus_map_,
        d_e_flags_.get(), M_, channels_);
    CUDA_CHECK_KERNEL();

    // Stable compaction → ascending ADO ids (preserves per-ADO memory
    // locality in the strided U-list kernels, same order as the old host
    // loop produced).
    size_t temp_bytes = d_temp_.bytes();
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        d_temp_.get(), temp_bytes, d_ids_.get(), d_e_flags_.get(),
        d_u_list_.get(), d_num_.get(), M_));

    // The only host traffic per build: one 4-byte D2H of |U| (the count
    // feeds ctx.u_count and the adaptive gate's |E|/M fraction host-side).
    CUDA_CHECK(cudaMemcpy(&count_, d_num_.get(), sizeof(int),
                          cudaMemcpyDeviceToHost));
    return count_;
}
