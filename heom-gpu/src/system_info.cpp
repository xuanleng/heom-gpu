#include "../include/system_info.hpp"
#include <cuda_runtime.h>
#include <hdf5.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <sys/utsname.h>

SystemInfo SystemInfo::query() {
    SystemInfo info;
    
    info.cpu_model = query_cpu_model();
    info.cpu_cores = query_cpu_cores();
    query_gpu_info(info.gpu_model, info.gpu_memory_mb, 
                   info.gpu_compute_major, info.gpu_compute_minor);
    query_os_info(info.os_name, info.os_version);
    info.cuda_version = query_cuda_version();
    info.compiler_version = query_compiler_version();
    info.hdf5_version = query_hdf5_version();
    
    return info;
}

std::string SystemInfo::query_cpu_model() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        std::cerr << "[WARN] system_info: /proc/cpuinfo not readable ("
                  << std::strerror(errno)
                  << "); CPU model will be reported as 'Unknown'.\n";
        return "Unknown";
    }

    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos && pos + 2 < line.size()) {
                std::string model = line.substr(pos + 2);
                size_t start = model.find_first_not_of(" \t");
                size_t end = model.find_last_not_of(" \t");
                if (start != std::string::npos && end != std::string::npos) {
                    return model.substr(start, end - start + 1);
                }
                return model;
            }
        }
    }
    std::cerr << "[WARN] system_info: no 'model name' line in /proc/cpuinfo;"
                 " CPU model will be reported as 'Unknown'.\n";
    return "Unknown";
}

int SystemInfo::query_cpu_cores() {
    int cores = std::thread::hardware_concurrency();
    return cores > 0 ? cores : 0;
}

void SystemInfo::query_gpu_info(std::string& model, size_t& memory_mb,
                                 int& major, int& minor) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) {
        model = prop.name;
        memory_mb = prop.totalGlobalMem / (1024 * 1024);
        major = prop.major;
        minor = prop.minor;
    } else {
        // Fall through to Unknown but tell the user WHY: most often the
        // process doesn't see a GPU at all (no driver, no device, container
        // missing /dev/nvidia*). Silent 'Unknown' here used to mask
        // "running on the wrong host" until a much later cudaMalloc.
        std::cerr << "[WARN] system_info: cudaGetDeviceProperties(0) failed: "
                  << cudaGetErrorString(err)
                  << "; GPU info will be reported as 'Unknown'.\n";
        model = "Unknown";
        memory_mb = 0;
        major = 0;
        minor = 0;
    }
}

void SystemInfo::query_os_info(std::string& name, std::string& version) {
    struct utsname info;
    if (uname(&info) == 0) {
        name = info.sysname;
        version = info.release;
    } else {
        std::cerr << "[WARN] system_info: uname() failed ("
                  << std::strerror(errno)
                  << "); OS info will be reported as 'Unknown'.\n";
        name = "Unknown";
        version = "Unknown";
    }
}

std::string SystemInfo::query_cuda_version() {
    int version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&version);
    if (err == cudaSuccess) {
        int major = version / 1000;
        int minor = (version % 1000) / 10;
        return std::to_string(major) + "." + std::to_string(minor);
    }
    std::cerr << "[WARN] system_info: cudaRuntimeGetVersion failed: "
              << cudaGetErrorString(err)
              << "; CUDA runtime version will be reported as 'Unknown'.\n";
    return "Unknown";
}

std::string SystemInfo::query_compiler_version() {
#ifdef __GNUC__
    return "GCC " __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#else
    return "Unknown";
#endif
}

std::string SystemInfo::query_hdf5_version() {
    unsigned int major = 0, minor = 0, release = 0;
    if (H5get_libversion(&major, &minor, &release) >= 0) {
        return std::to_string(major) + "." +
               std::to_string(minor) + "." +
               std::to_string(release);
    }
    std::cerr << "[WARN] system_info: H5get_libversion failed;"
                 " HDF5 version will be reported as 'Unknown'.\n";
    return "Unknown";
}

