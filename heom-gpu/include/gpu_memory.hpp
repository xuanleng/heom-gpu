#ifndef GPU_MEMORY_HPP
#define GPU_MEMORY_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/**
 * @brief RAII wrapper for GPU memory
 * 
 * LINUS REFACTOR: Automatic memory management.
 * - Constructor allocates
 * - Destructor frees (compiler does it for you)
 * - No manual cudaFree() needed
 * - No memory leaks possible
 * 
 * Rule of 5:
 * - Delete copy (prevent double-free)
 * - Implement move (transfer ownership)
 */
template<typename T>
class GPUMemory {
public:
    /**
     * @brief Allocate GPU memory
     * @param count Number of elements (not bytes!)
     * 
     * LINUS STYLE: Standard GPU memory allocation
     * - Allocates physical GPU memory immediately
     * - User must ensure system fits in GPU RAM by adjusting K/L parameters
     */
    explicit GPUMemory(size_t count = 0) 
        : ptr_(nullptr), size_(count) 
    {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMalloc failed: ") + cudaGetErrorString(err)
                );
            }
        }
    }
    
    /**
     * @brief Destructor - automatic cleanup
     */
    ~GPUMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    
    // Delete copy constructor/assignment (prevent double-free)
    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;
    
    // Move constructor
    GPUMemory(GPUMemory&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) 
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    // Move assignment
    GPUMemory& operator=(GPUMemory&& other) noexcept {
        if (this != &other) {
            // Free our current memory
            if (ptr_) {
                cudaFree(ptr_);
            }
            // Take ownership of other's memory
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    /**
     * @brief Get raw device pointer
     */
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    
    /**
     * @brief Get number of elements
     */
    size_t size() const { return size_; }
    
    /**
     * @brief Get size in bytes
     */
    size_t bytes() const { return size_ * sizeof(T); }
    
    /**
     * @brief Copy from host to device
     */
    void copy_from_host(const T* host_data, size_t count) {
        if (count > size_) {
            throw std::runtime_error("copy_from_host: count exceeds allocated size");
        }
        cudaError_t err = cudaMemcpy(ptr_, host_data, count * sizeof(T), 
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(err)
            );
        }
    }
    
    /**
     * @brief Copy from device to host
     */
    void copy_to_host(T* host_data, size_t count) const {
        if (count > size_) {
            throw std::runtime_error("copy_to_host: count exceeds allocated size");
        }
        cudaError_t err = cudaMemcpy(host_data, ptr_, count * sizeof(T), 
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(err)
            );
        }
    }
    
    /**
     * @brief Grow to hold at least `count` elements; no-op if already big enough.
     *
     * The scratch buffers reused across a 2DES tree all want this: reallocate
     * only when the requirement grows, never shrink. Contents are NOT preserved
     * when a reallocation happens — every caller overwrites the buffer before
     * reading it — so the old block is freed BEFORE the new one is allocated:
     * the peak is the larger size, not the sum the budget never counted.
     */
    void ensure(size_t count) {
        if (count > size_) {
            *this = GPUMemory<T>();
            *this = GPUMemory<T>(count);
        }
    }

    /**
     * @brief Set memory to zero
     */
    void zero() {
        if (ptr_) {
            cudaError_t err = cudaMemset(ptr_, 0, bytes());
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMemset failed: ") + cudaGetErrorString(err)
                );
            }
        }
    }

private:
    T* ptr_;
    size_t size_;
};

#endif // GPU_MEMORY_HPP

