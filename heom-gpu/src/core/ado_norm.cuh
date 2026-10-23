#ifndef ADO_NORM_CUH
#define ADO_NORM_CUH

// Device-only pieces of the ADO max-norm, shared by the standalone norm
// kernels (ado_filter.cu) and the fused kFinal epilogue
// (heom_derivatives.cu). One definition of |z|² so both paths compile the
// same expression and stay bit-identical. Include from .cu files only.

#include "core/precision.hpp"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace heom {

// |z|² in FP64 (a mixed-precision element promotes before squaring).
__device__ __forceinline__ double ado_abs_sq(complex_t z) {
    const double re = c_re(z);
    const double im = c_im(z);
    return re * re + im * im;
}

// Fused-norm accumulation: fold one element's |z|² into norm_sq_bits[idx]
// (max over the ADO's elements). idx = b·M + α for per-slot rows.
//
// |z|² is +0, positive or NaN, so its IEEE bits order like the values when
// read as uint64; the max is then an integer max — exact and
// order-independent, so the atomics cannot change the result. NaN is mapped
// to the largest positive NaN pattern: it beats +inf and every finite value,
// so a NaN element sticks exactly like nan_sticky_max in the standalone
// kernels. Threads of a warp holding the same α (one ADO's Nb elements are
// consecutive global_idx) are reduced first, so one atomic per (warp, ADO)
// reaches memory instead of one per element. Exited threads are simply
// absent from coalesced_threads().
__device__ __forceinline__ void fused_norm_accumulate(
    unsigned long long* norm_sq_bits, int alpha, complex_t z)
{
    namespace cg = cooperative_groups;
    const double s = ado_abs_sq(z);
    const unsigned long long bits = (s != s)
        ? 0x7FFFFFFFFFFFFFFFull
        : static_cast<unsigned long long>(__double_as_longlong(s));
    const cg::coalesced_group same = cg::labeled_partition(cg::coalesced_threads(), alpha);
    const unsigned long long m = cg::reduce(same, bits, cg::greater<unsigned long long>());
    if (same.thread_rank() == 0) atomicMax(&norm_sq_bits[alpha], m);
}

}  // namespace heom

#endif  // ADO_NORM_CUH
