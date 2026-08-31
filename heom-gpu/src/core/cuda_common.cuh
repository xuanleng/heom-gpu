#ifndef CUDA_COMMON_CUH
#define CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Threads per CUDA block for every kernel in the project. Must be a
// power of 2 (the trace/reduction kernels use a halving reduction), at
// least 32 (warp size), at most 1024 (CUDA hardware limit).
//
// Override at configure time with:
//     cmake -DHEOM_BLOCK_SIZE=128 ...
// CMake forwards the value via target_compile_definitions; the guard
// here provides the default when no -D is supplied.
#ifndef HEOM_BLOCK_SIZE
#define HEOM_BLOCK_SIZE 256
#endif
static_assert((HEOM_BLOCK_SIZE & (HEOM_BLOCK_SIZE - 1)) == 0,
              "HEOM_BLOCK_SIZE must be a power of 2 for reduction kernels");
static_assert(HEOM_BLOCK_SIZE >= 32 && HEOM_BLOCK_SIZE <= 1024,
              "HEOM_BLOCK_SIZE must be in [32, 1024]");

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + \
                cudaGetErrorString(err)); \
        } \
    } while(0)

// Use after a kernel launch. cudaGetLastError() catches *launch* failures
// (bad grid/block, invalid arguments). Kernel *execution* failures
// (out-of-bounds, illegal memory) only surface on the next blocking CUDA
// call or on cudaDeviceSynchronize(). In debug builds we therefore force
// the synchronize so a kernel bug is reported at the responsible call
// site instead of silently corrupting downstream state and blowing up
// hundreds of lines later. Release builds keep the launch-error-only path
// — synchronizing every kernel would serialise the whole pipeline.
#ifndef NDEBUG
  #define CUDA_CHECK_KERNEL() \
      do { \
          CUDA_CHECK(cudaGetLastError()); \
          CUDA_CHECK(cudaDeviceSynchronize()); \
      } while(0)
#else
  #define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif

#endif // CUDA_COMMON_CUH
