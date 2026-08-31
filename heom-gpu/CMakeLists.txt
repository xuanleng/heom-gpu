cmake_minimum_required(VERSION 3.18)
project(HEOM_GPU_Solver VERSION 1.0.0 LANGUAGES C CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture: allow override via CMAKE_CUDA_ARCHITECTURES
# Users can specify: cmake -DCMAKE_CUDA_ARCHITECTURES=86
#                    cmake -DCMAKE_CUDA_ARCHITECTURES=120  (Blackwell; CUDA >= 12.8)
# Or set multiple: cmake -DCMAKE_CUDA_ARCHITECTURES="86;120"
# If not set, defaults to common architectures for maximum compatibility
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING 
        "CUDA architectures to build for (e.g., '86' for RTX 3060, '75;80;86' for multiple)")
    message(STATUS "CUDA architectures not specified, using defaults: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  Override with: cmake -DCMAKE_CUDA_ARCHITECTURES=<arch>")
endif()

# Build type defaults to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA threads-per-block, override with: cmake -DHEOM_BLOCK_SIZE=128 ...
# Must be a power of 2 in [32, 1024]; static_asserts in cuda_common.cuh
# enforce this at compile time.
set(HEOM_BLOCK_SIZE 256 CACHE STRING
    "CUDA threads per block (power of 2 in [32, 1024]; default 256)")

# Single source of truth for the per-language optimisation flags shared by
# every core/spectra library (FP64 and mixed-precision alike). One function so
# the FP64 and _mp builds cannot drift in optimisation level / fast-math (which
# would poison an FP64-vs-mixed comparison). NOTE: --use_fast_math hits FP32
# harder than FP64 (div/sqrt approx, subnormal flush); if the mixed-precision
# regression test fails, the first knob is dropping it from the _mp libs only.
function(heom_apply_build_flags tgt)
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            --expt-relaxed-constexpr
        >
    )
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            $<$<CONFIG:Debug>:-g -Wall -Wextra>
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g>
            $<$<CONFIG:MinSizeRel>:-Os>
        >
    )
endfunction()

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED)

# Find yaml-cpp: try pkg-config first (more reliable on Ubuntu), then find_package
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(YAML_CPP yaml-cpp)
    if(YAML_CPP_FOUND)
        add_library(yaml-cpp INTERFACE IMPORTED)
        target_include_directories(yaml-cpp INTERFACE ${YAML_CPP_INCLUDE_DIRS})
        target_link_directories(yaml-cpp INTERFACE ${YAML_CPP_LIBRARY_DIRS})
        target_link_libraries(yaml-cpp INTERFACE ${YAML_CPP_LINK_LIBRARIES})
        target_compile_options(yaml-cpp INTERFACE ${YAML_CPP_CFLAGS_OTHER})
        message(STATUS "Found yaml-cpp via pkg-config: ${YAML_CPP_VERSION}")
        message(STATUS "  Include dirs: ${YAML_CPP_INCLUDE_DIRS}")
        message(STATUS "  Library dirs: ${YAML_CPP_LIBRARY_DIRS}")
        message(STATUS "  Libraries: ${YAML_CPP_LINK_LIBRARIES}")
    endif()
endif()

# Fallback to find_package if pkg-config didn't work
if(NOT TARGET yaml-cpp)
    find_package(yaml-cpp QUIET)
    if(yaml-cpp_FOUND)
        message(STATUS "Found yaml-cpp via find_package")
    endif()
endif()

if(NOT TARGET yaml-cpp)
    message(FATAL_ERROR "yaml-cpp not found. Please install yaml-cpp development package.")
endif()

# Enable OpenMP support (platform-aware)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found. Performance may be degraded.")
endif()

# ============================================================================
# heom_core: STATIC library containing every translation unit that makes a
# HEOM solver (CUDA kernels + host orchestration + HDF5/console output +
# config parsing + physics utilities). Only main.cpp is kept out so that
# alternative entry points (a 2DES driver, a Python binding, a benchmark
# runner) can link the same object archive instead of re-listing sources
# and risking drift.
# ============================================================================
set(CORE_SOURCES
    src/core/hierarchy_truncation.cpp
    src/core/heom_derivatives.cu
    src/core/rk4_integrator.cu
    src/core/ado_filter.cu
    src/core/ulist_builder.cu
    src/core/dipole_apply.cu
    src/core/hamiltonian_expansion.cpp
)

set(UTILS_SOURCES
    src/heom_config.cpp
    src/system_context.cpp
    src/heom_trajectory.cpp
    src/system_info.cpp
    src/physics_utils.cpp
    src/memory_budget.cpp
)

set(OUTPUT_SOURCES
    src/output/hdf5_writer.cpp
    src/output/hdf5_metadata.cpp        # shared metadata writers
    src/output/spectrum_writer.cpp      # spectroscopy HDF5 schema
    src/output/console_printer.cpp
)

# MC static-disorder shell: a thin host-side layer that freezes one
# disorder realization onto the Hamiltonian and re-expands. The HEOM
# kernel is untouched. Pure C++/Eigen — no CUDA in this file.
set(MC_SOURCES
    src/mc/mc_disorder.cpp
    src/mc/mc_realizations.cpp
    src/mc/mc_partial.cpp
    src/mc/mc_queue.cpp
)

add_library(heom_core STATIC
    ${CORE_SOURCES}
    ${UTILS_SOURCES}
    ${OUTPUT_SOURCES}
    ${MC_SOURCES}
)

# include/ and src/ are PUBLIC so any executable that links heom_core
# (heom_solver today, future drivers tomorrow) automatically sees the
# headers without re-declaring the path.
target_include_directories(heom_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(heom_core PUBLIC
    CUDA::cudart
    Eigen3::Eigen
    HDF5::HDF5
    yaml-cpp
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(heom_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Thread-block size is part of the public ABI of the CUDA kernels (kernel
# templates and shared-memory sizes depend on it), so consumers see it too.
target_compile_definitions(heom_core PUBLIC HEOM_BLOCK_SIZE=${HEOM_BLOCK_SIZE})

# CUDA -O3/fast-math + per-config CXX flags (shared with the _mp build so they
# cannot drift — see heom_apply_build_flags above).
heom_apply_build_flags(heom_core)

# ============================================================================
# spectra: STATIC library for the spectroscopy module (Stage 1 = absorption).
#   - PUBLIC depends on heom_core (uses SystemContext / HEOMTrajectory).
#   - The whole module is host-side: NO CUDA includes (cuComplex.h etc.) by
#     design. This module stays free of CUDA headers.
#   - PUBLIC includes already come transitively from heom_core (which exposes
#     ${CMAKE_SOURCE_DIR}/include); spectra just adds source files.
# ============================================================================
set(SPECTRA_SOURCES
    src/spectra/dipole.cpp
    src/spectra/linear_response.cpp   # linear-response driver, depends on heom_core
    src/spectra/absorption.cpp        # thin wrapper around linear response
    src/spectra/two_dim.cpp           # carry-through 2DES driver
    src/mc/mc_driver.cpp              # MC static-disorder shell (uses compute_*)
)

add_library(spectra STATIC ${SPECTRA_SOURCES})
target_link_libraries(spectra PUBLIC heom_core)
heom_apply_build_flags(spectra)

# ============================================================================
# heom_solver: the production CLI executable. Just main.cpp + heom_core +
# spectra (main.cpp dispatches into compute_absorption()).
# ============================================================================
add_executable(heom_solver src/main.cpp)
target_link_libraries(heom_solver PRIVATE heom_core spectra)

if(UNIX)
    set_target_properties(heom_solver PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# ============================================================================
# Mixed-precision twin: same sources, HEOM_MIXED_PRECISION defined.
#   heom_solver     -> full FP64, byte-identical to history (Never break userspace)
#   heom_solver_mp  -> FP32 hierarchy / RK4 workspace (consumer-GPU production)
# Switching = choosing the binary (mc-disorder-run --solver / $HEOM_SOLVER /
# PATH); zero runtime branch, zero yaml change. The macro is PUBLIC so every TU
# that sees HEOMContext's pointer types (including main.cpp) gets the same
# `heom::complex_t`, or ODR tears. We duplicate the LIBRARIES (heom_core lives in
# STATIC libs, not in the thin main.cpp), not the executable source list.
# ============================================================================
add_library(heom_core_mp STATIC
    ${CORE_SOURCES}
    ${UTILS_SOURCES}
    ${OUTPUT_SOURCES}
    ${MC_SOURCES}
)
target_include_directories(heom_core_mp PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(heom_core_mp PUBLIC
    CUDA::cudart
    Eigen3::Eigen
    HDF5::HDF5
    yaml-cpp
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(heom_core_mp PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_definitions(heom_core_mp PUBLIC
    HEOM_BLOCK_SIZE=${HEOM_BLOCK_SIZE}
    HEOM_MIXED_PRECISION)
heom_apply_build_flags(heom_core_mp)

add_library(spectra_mp STATIC ${SPECTRA_SOURCES})
target_link_libraries(spectra_mp PUBLIC heom_core_mp)
heom_apply_build_flags(spectra_mp)

add_executable(heom_solver_mp src/main.cpp)
target_link_libraries(heom_solver_mp PRIVATE heom_core_mp spectra_mp)

if(UNIX)
    set_target_properties(heom_solver_mp PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# ============================================================================
# Tests
# ============================================================================
enable_testing()

# heom-tools owns the MC generator and the tooling tests. Monorepo default is
# the sibling checkout; a solver-only staged tree has none, so absence of the
# default is a loud warning (MC py tests are not registered), while an
# explicitly given path that is wrong is a configure error.
set(_heom_tools_default "${CMAKE_SOURCE_DIR}/../heom-tools")
set(HEOM_TOOLS_DIR "" CACHE PATH "heom-tools checkout (default: ../heom-tools if present)")
if(HEOM_TOOLS_DIR)
    set(_heom_tools "${HEOM_TOOLS_DIR}")
    if(NOT EXISTS "${_heom_tools}/heom_tools/mc_disorder/generate.py")
        message(FATAL_ERROR "HEOM_TOOLS_DIR=${HEOM_TOOLS_DIR} has no "
                            "heom_tools/mc_disorder/generate.py")
    endif()
elseif(EXISTS "${_heom_tools_default}/heom_tools/mc_disorder/generate.py")
    get_filename_component(_heom_tools "${_heom_tools_default}" REALPATH)
else()
    set(_heom_tools "")
    message(WARNING "heom-tools not found at ${_heom_tools_default}: the MC "
                    "generator / tooling tests are NOT registered. Set "
                    "-DHEOM_TOOLS_DIR=<checkout> to enable them.")
endif()

# Unit tests (Matsubara, Hamiltonian expansion, hierarchy generation, spectra).
# spectra/dipole.cpp is listed inline rather than via the `spectra` library so
# test_core stays a self-contained no-GPU build that does not need to drag in
# heom_core's CUDA dependency.
add_executable(test_core
    tests/test_core.cpp
    src/core/hamiltonian_expansion.cpp
    src/core/hierarchy_truncation.cpp
    src/heom_config.cpp
    src/physics_utils.cpp
    src/memory_budget.cpp
    src/spectra/dipole.cpp
    # Pure formatter (Eigen only, no CUDA): lets the suite assert that the
    # printers restore std::cout's sticky format state.
    src/output/console_printer.cpp
)
target_include_directories(test_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(test_core PRIVATE CUDA::cudart Eigen3::Eigen yaml-cpp)
add_test(NAME unit_tests COMMAND test_core)

# MC static-disorder unit tests. Host-only by design: apply_disorder just
# touches the site Hamiltonian + re-expands, so this target does NOT need CUDA
# runtime or SystemContext (GPU). Sources listed inline (mirrors test_core) to
# keep it a fast, GPU-free unit build. Rebuild-timing (needs SystemContext/GPU)
# is a one-off measurement, deliberately NOT wired here.
add_executable(test_mc
    tests/test_mc.cpp
    src/mc/mc_disorder.cpp
    src/mc/mc_realizations.cpp
    src/mc/mc_queue.cpp
    src/core/hamiltonian_expansion.cpp
    src/heom_config.cpp
    src/physics_utils.cpp
    src/memory_budget.cpp
)
target_include_directories(test_mc PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src
)
# HDF5 for the disorder-node reader; MC_GEN_SCRIPT lets the C++ reader test round-trip
# against the real Python generator output (cross-language schema contract).
target_link_libraries(test_mc PRIVATE CUDA::cudart Eigen3::Eigen yaml-cpp HDF5::HDF5)
if(_heom_tools)
    target_compile_definitions(test_mc PRIVATE
        MC_GEN_SCRIPT="${_heom_tools}/heom_tools/mc_disorder/generate.py")
endif()
add_test(NAME mc_tests COMMAND test_mc)

# HDF5 metadata smoke test: exercises config snapshot writing without starting
# a HEOM trajectory or touching CUDA runtime allocation.
add_executable(test_hdf5_config_metadata
    tests/test_hdf5_config_metadata.cpp
)
target_include_directories(test_hdf5_config_metadata PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(test_hdf5_config_metadata PRIVATE heom_core)
add_test(NAME hdf5_config_metadata COMMAND test_hdf5_config_metadata)

# GPU-side smoke test: tiny system that compares kernel output to a hand-
# computed -i [H, rho] reference. Runs through the same launcher
# as production code, so it implicitly checks shared-memory layout, the
# CUDA_CHECK_KERNEL plumbing, and HEOM_BLOCK_SIZE wiring. Requires a GPU.
add_executable(test_gpu_kernel tests/test_gpu_kernel.cpp)
target_link_libraries(test_gpu_kernel PRIVATE heom_core)
add_test(NAME gpu_kernel COMMAND test_gpu_kernel)

# Same kernel smoke test against the mixed-precision library. The test's own
# tolerance must be FP32-scale (~1e-4) not the FP64 few-ULP bound — the test
# source keys its tolerance off HEOM_MIXED_PRECISION.
add_executable(test_gpu_kernel_mp tests/test_gpu_kernel.cpp)
target_link_libraries(test_gpu_kernel_mp PRIVATE heom_core_mp)
add_test(NAME gpu_kernel_mp COMMAND test_gpu_kernel_mp)

# End-to-end batched 3-polarization absorption (serial vs batch).
# Links spectra (compute_linear_response) + heom_core. Requires a GPU.
add_executable(test_absorption_batch tests/test_absorption_batch.cpp)
target_link_libraries(test_absorption_batch PRIVATE heom_core spectra)
add_test(NAME absorption_batch COMMAND test_absorption_batch)

# Pure-host grouping check for the 3->9->21 orientation-reuse tree.
# Links spectra (group_isotropic). No GPU needed.
add_executable(test_2des_reuse tests/test_2des_reuse.cpp)
target_link_libraries(test_2des_reuse PRIVATE heom_core spectra)
add_test(NAME 2des_reuse COMMAND test_2des_reuse)

# Integration test: FMO-7 end-to-end benchmark comparison
add_test(NAME integration_fmo7
    COMMAND ${CMAKE_COMMAND} -E env
        HEOM_SOLVER=$<TARGET_FILE:heom_solver>
        HEOM_ROOT=${CMAKE_SOURCE_DIR}
        python3 ${CMAKE_SOURCE_DIR}/tests/test_integration.py
)

# Benchmark infrastructure: template/compare self-consistency (all 3 tasks).
# pytest's prepend import mode inserts tests/ (no __init__.py) onto sys.path,
# so `import run_all_benchmarks` resolves without a WORKING_DIRECTORY hack.
# The two MC cases skip themselves when HEOM_TOOLS_DIR is empty.
add_test(NAME benchmark_infra
    COMMAND ${CMAKE_COMMAND} -E env
        HEOM_SOLVER=$<TARGET_FILE:heom_solver>
        HEOM_ROOT=${CMAKE_SOURCE_DIR}
        HEOM_TOOLS_DIR=${_heom_tools}
        python3 -m pytest ${CMAKE_SOURCE_DIR}/tests/test_benchmark_infra.py -v
)

if(_heom_tools)
    # MC static-disorder tests (Python): σ>0 reproducibility + error-estimator
    # well-formedness. These drive the GPU solver, hence the HEOM_SOLVER env
    # (same pattern as benchmark_infra). Pure generator schema tests live with
    # the tools package (heom-tools/tests/mc_disorder).
    add_test(NAME mc_disorder_py
        COMMAND ${CMAKE_COMMAND} -E env
            HEOM_SOLVER=$<TARGET_FILE:heom_solver>
            HEOM_ROOT=${CMAKE_SOURCE_DIR}
            HEOM_TOOLS_DIR=${_heom_tools}
            python3 -m pytest ${CMAKE_SOURCE_DIR}/tests/test_mc_disorder.py -v
    )

    # Pure MC tooling tests (Python, NO GPU): sparse_gh quadrature generator
    # moment checks (test_generate), quadrature shard merge (test_mc_merge), and
    # the smoothness-probe L² metric (test_probe). These live with the
    # distributable packages and need no solver, so they run in any environment
    # (CI, sandbox) — kept separate from the GPU-driven mc_disorder_py above.
    add_test(NAME mc_tooling_py
        COMMAND python3 -m pytest
            ${_heom_tools}/tests/mc_disorder
            ${_heom_tools}/tests/common
            -v
    )
endif()

# Full regression net: dynamics×3 + absorption + 2des, bit-for-bit vs baselines.
add_test(NAME regression_all
    COMMAND ${CMAKE_COMMAND} -E env
        HEOM_SOLVER=$<TARGET_FILE:heom_solver>
        HEOM_ROOT=${CMAKE_SOURCE_DIR}
        python3 ${CMAKE_SOURCE_DIR}/tests/compare_benchmark.py --run-all --tol 1e-12
)

# Install rules
install(TARGETS heom_solver heom_solver_mp DESTINATION bin)
install(DIRECTORY config/ DESTINATION share/heom_solver/config)
install(FILES README.md BUILD.md LICENSE DESTINATION share/heom_solver/)

# Print configuration summary
message(STATUS "=== HEOM GPU Solver Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "HDF5 version: ${HDF5_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "HEOM_BLOCK_SIZE: ${HEOM_BLOCK_SIZE}")
message(STATUS "=====================================")

