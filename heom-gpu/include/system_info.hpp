#ifndef SYSTEM_INFO_HPP
#define SYSTEM_INFO_HPP

#include <string>

/**
 * @brief System hardware and software information
 * 
 * Captures system configuration for reproducibility and performance analysis.
 * Query once at initialization (~2ms overhead, negligible).
 */
struct SystemInfo {
    std::string cpu_model;
    int cpu_cores;
    std::string gpu_model;
    size_t gpu_memory_mb;
    int gpu_compute_major;
    int gpu_compute_minor;
    std::string os_name;
    std::string os_version;
    std::string cuda_version;
    std::string compiler_version;
    std::string hdf5_version;
    
    /**
     * @brief Query system information
     * 
     * Queries hardware and software configuration.
     * Gracefully handles failures (returns "Unknown" on error).
     * 
     * @return SystemInfo structure with all available information
     */
    static SystemInfo query();
    
private:
    static std::string query_cpu_model();
    static int query_cpu_cores();
    static void query_gpu_info(std::string& model, size_t& memory_mb, 
                                int& major, int& minor);
    static void query_os_info(std::string& name, std::string& version);
    static std::string query_cuda_version();
    static std::string query_compiler_version();
    static std::string query_hdf5_version();
};

#endif // SYSTEM_INFO_HPP

