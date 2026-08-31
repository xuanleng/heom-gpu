#ifndef HEOM_CORE_PRECISION_HPP
#define HEOM_CORE_PRECISION_HPP

// Mixed-precision types, selected by HEOM_MIXED_PRECISION.
//
// Undefined: FP64 (complex_t = cuDoubleComplex); wrappers forward to cuC*.
// Defined:   the large O(M*N^2) arrays are FP32 (cuComplex).
//
// Coefficients, the Hamiltonian, and setup stay FP64 and are narrowed once
// on the hot-path read. Narrowing is explicit.
//
// Wrappers are __host__ __device__, shared by host staging and kernels.

#include <cuComplex.h>

namespace heom {

#ifdef HEOM_MIXED_PRECISION
using real_t    = float;
using complex_t = cuComplex;
inline constexpr bool kMixedPrecision = true;
#else
using real_t    = double;
using complex_t = cuDoubleComplex;
inline constexpr bool kMixedPrecision = false;
#endif

// Complex ops on complex_t. FP32 uses cuC*f, FP64 uses cuC*.

#ifdef HEOM_MIXED_PRECISION

__host__ __device__ __forceinline__ complex_t c_make(real_t r, real_t i) { return make_cuComplex(r, i); }
__host__ __device__ __forceinline__ real_t    c_re(complex_t z)          { return cuCrealf(z); }
__host__ __device__ __forceinline__ real_t    c_im(complex_t z)          { return cuCimagf(z); }
__host__ __device__ __forceinline__ complex_t c_add(complex_t a, complex_t b) { return cuCaddf(a, b); }
__host__ __device__ __forceinline__ complex_t c_sub(complex_t a, complex_t b) { return cuCsubf(a, b); }
__host__ __device__ __forceinline__ complex_t c_mul(complex_t a, complex_t b) { return cuCmulf(a, b); }
__host__ __device__ __forceinline__ complex_t c_conj(complex_t z)             { return cuConjf(z); }

#else

__host__ __device__ __forceinline__ complex_t c_make(real_t r, real_t i) { return make_cuDoubleComplex(r, i); }
__host__ __device__ __forceinline__ real_t    c_re(complex_t z)          { return cuCreal(z); }
__host__ __device__ __forceinline__ real_t    c_im(complex_t z)          { return cuCimag(z); }
__host__ __device__ __forceinline__ complex_t c_add(complex_t a, complex_t b) { return cuCadd(a, b); }
__host__ __device__ __forceinline__ complex_t c_sub(complex_t a, complex_t b) { return cuCsub(a, b); }
__host__ __device__ __forceinline__ complex_t c_mul(complex_t a, complex_t b) { return cuCmul(a, b); }
__host__ __device__ __forceinline__ complex_t c_conj(complex_t z)             { return cuConj(z); }

#endif

// Narrow an FP64 coefficient to hierarchy precision on the hot-path read.
// Identity when complex_t is cuDoubleComplex.
__host__ __device__ __forceinline__ complex_t c_from_double(cuDoubleComplex z) {
    return c_make(static_cast<real_t>(cuCreal(z)),
                  static_cast<real_t>(cuCimag(z)));
}

// z * s, s real
__host__ __device__ __forceinline__ complex_t c_scale(real_t s, complex_t z) {
    return c_make(s * c_re(z), s * c_im(z));
}

// z * (i*s): (re, im) -> (-s*im, s*re)
__host__ __device__ __forceinline__ complex_t c_scale_i(real_t s, complex_t z) {
    return c_make(-s * c_im(z), s * c_re(z));
}

// Host copy-out promotes with std::complex<double>(c_re(v), c_im(v)).

}  // namespace heom

#endif  // HEOM_CORE_PRECISION_HPP
