cmake_minimum_required(VERSION 3.18)

# CUDA architecture. build.sh (the documented entry point) detects the GPU and
# passes -DCMAKE_CUDA_ARCHITECTURES; the site scripts pass it explicitly. This
# default is for a bare `cmake` and MUST be set before project() enables CUDA —
# after that point CMake has already cached its own default (sm_75 with CMake
# 3.22 + CUDA 13), and a later `if(NOT DEFINED ...)` block never runs; that is
# exactly how this file silently built sm_75-only binaries.
#   cmake -DCMAKE_CUDA_ARCHITECTURES=86
#   cmake -DCMAKE_CUDA_ARCHITECTURES=120       (Blackwell; CUDA >= 12.8)
#   cmake -DCMAKE_CUDA_ARCHITECTURES="86;120"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        # Compile for the GPU present on the build host.
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
    else()
        # No detection available: a broad list. sm_120 is deliberately absent —
        # nvcc < 12.8 rejects it and would break configure on older sites.
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures")
    endif()
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not given; using ${CMAKE_CUDA_ARCHITECTURES} "
                   "(pass -DCMAKE_CUDA_ARCHITECTURES=120 for Blackwell, or use ./build.sh)")
endif()

project(HEOM_GPU_Solver VERSION 2.2.0 LANGUAGES C CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
    src/core/hierarchy_distance.cu      # relative_distance (fluorescence equilibration)
    src/core/dressed_start.cu           # dress_ados (fluorescence start)
    src/core/hamiltonian_expansion.cpp
    src/core/manifold_layout.cpp
)

set(UTILS_SOURCES
    src/heom_config.cpp
    src/system_context.cpp
    src/heom_trajectory.cpp
    src/detector.cpp                    # Device-side detection, split off HEOMTrajectory
    src/dynamics_run.cpp                # Output loop + dynamics CLI, split off HEOMTrajectory
    src/system_info.cpp
    src/physics_utils.cpp
    src/memory_budget.cpp
)

set(OUTPUT_SOURCES
    src/output/hdf5_writer.cpp
    src/output/hdf5_metadata.cpp        # Shared metadata writers (free fns)
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

# ============================================================================
# spectra: STATIC library for the spectroscopy module.
#   - PUBLIC depends on heom_core (uses SystemContext / HEOMTrajectory).
#   - The whole module is host-side: NO CUDA includes (cuComplex.h etc.) by
#     design.
# ============================================================================
set(SPECTRA_SOURCES
    src/spectra/dipole.cpp
    src/spectra/linear_record.cpp     # direction grouping + ρ_α(t) recording (abs / fl)
    src/spectra/fluorescence.cpp      # bare Boltzmann start, equilibration, emission
    src/spectra/linear_response.cpp   # linear-response driver, depends on heom_core
    src/spectra/absorption.cpp        # thin wrapper around linear response
    src/spectra/two_dim.cpp           # carry-through 2DES driver
    src/mc/mc_driver.cpp              # MC static-disorder shell (uses compute_*)
)

# ============================================================================
# One solver stack = heom_core<suffix> + spectra<suffix> + heom_solver<suffix>.
# Called twice:
#   heom_add_solver_stack("")                        FP64, byte-identical to history
#   heom_add_solver_stack("_mp" HEOM_MIXED_PRECISION) FP32 hierarchy / RK4 workspace
# Switching = choosing the binary (mc-disorder-run --solver / $HEOM_SOLVER /
# PATH); zero runtime branch, zero yaml change. Extra args are PUBLIC compile
# definitions: every TU that sees HEOMContext's pointer types (including
# main.cpp) must get the same `heom::complex_t`, or ODR tears. Everything that
# could drift between the twins (includes, links, flags, RPATH) is written once
# here — the two stacks cannot disagree by omission.
#
# include/ and src/ are PUBLIC so any executable that links the core library
# (the CLI, the tests, future drivers) sees the headers without re-declaring
# the path. HEOM_BLOCK_SIZE is PUBLIC because it is part of the CUDA kernels'
# ABI (kernel templates and shared-memory sizes depend on it).
# ============================================================================
function(heom_add_solver_stack suffix)
    set(core heom_core${suffix})
    set(spec spectra${suffix})
    set(cli  heom_solver${suffix})

    add_library(${core} STATIC
        ${CORE_SOURCES} ${UTILS_SOURCES} ${OUTPUT_SOURCES} ${MC_SOURCES})
    target_include_directories(${core} PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${core} PUBLIC
        CUDA::cudart Eigen3::Eigen HDF5::HDF5 yaml-cpp)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${core} PUBLIC OpenMP::OpenMP_CXX)
    endif()
    target_compile_definitions(${core} PUBLIC HEOM_BLOCK_SIZE=${HEOM_BLOCK_SIZE} ${ARGN})
    heom_apply_build_flags(${core})

    add_library(${spec} STATIC ${SPECTRA_SOURCES})
    target_link_libraries(${spec} PUBLIC ${core})
    heom_apply_build_flags(${spec})

    add_executable(${cli} src/main.cpp)
    target_link_libraries(${cli} PRIVATE ${core} ${spec})
    if(UNIX)
        set_target_properties(${cli} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    endif()
endfunction()

heom_add_solver_stack("")
heom_add_solver_stack("_mp" HEOM_MIXED_PRECISION)

# ============================================================================
# Tests
# ============================================================================
# A solver tree copied without tests/ (release / staged checkout)
# still configures and builds heom_solver / heom_solver_mp. The full tree
# has tests/test_core.cpp and registers the same targets as before.
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_core.cpp")
enable_testing()

# heom-tools owns the MC generator and the tooling tests. In the lab tree it is
# the sibling checkout; a solver-only staged tree
# has none, so absence of the default is a loud warning (tests that need it are
# not registered, test_mc SKIPs its generator cases), while an explicitly given
# path that is wrong is a configure error rather than a mid-suite abort.
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

# Unit tests (Matsubara, Hamiltonian expansion, hierarchy generation, spectra,
# config hash, budget). Host-only at RUN time: nothing here launches a kernel,
# so the binary runs on a GPU-less host — linking heom_core only registers the
# fatbins, it does not need a device. Linking the library (rather than the old
# inline source list) means a file added to the library is in the test build
# automatically instead of surfacing as an undefined symbol weeks later.
add_executable(test_core tests/test_core.cpp)
target_include_directories(test_core PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
target_link_libraries(test_core PRIVATE heom_core spectra)
add_test(NAME unit_tests COMMAND test_core)

# MC static-disorder unit tests. Host-only at run time: apply_disorder just
# touches the site Hamiltonian + re-expands; no SystemContext, no GPU needed.
# MC_GEN_SCRIPT lets the C++ reader test round-trip against the real Python
# generator output (cross-language schema contract).
add_executable(test_mc tests/test_mc.cpp)
target_include_directories(test_mc PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
target_link_libraries(test_mc PRIVATE heom_core)
if(_heom_tools)
    target_compile_definitions(test_mc PRIVATE
        MC_GEN_SCRIPT="${_heom_tools}/heom_tools/mc_disorder/generate.py")
endif()
add_test(NAME mc_tests COMMAND test_mc)

# HDF5 metadata smoke test: exercises config snapshot writing without starting
# a HEOM trajectory or touching CUDA runtime allocation.
add_executable(test_hdf5_config_metadata tests/test_hdf5_config_metadata.cpp)
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

# Fluorescence primitives and gates (relative_distance, dress_ados, G1–G3, G7).
# Twin binaries like gpu_kernel: the source keys tolerances off HEOM_MIXED_PRECISION.
add_executable(test_fluorescence tests/test_fluorescence.cpp)
target_link_libraries(test_fluorescence PRIVATE heom_core spectra)
add_test(NAME fluorescence COMMAND test_fluorescence)
add_executable(test_fluorescence_mp tests/test_fluorescence.cpp)
target_link_libraries(test_fluorescence_mp PRIVATE heom_core_mp spectra_mp)
add_test(NAME fluorescence_mp COMMAND test_fluorescence_mp)

# Pure-host grouping check for the 3->9->21 orientation-reuse tree.
# Links spectra (group_isotropic). No GPU needed.
add_executable(test_2des_reuse tests/test_2des_reuse.cpp)
target_link_libraries(test_2des_reuse PRIVATE heom_core spectra)
add_test(NAME 2des_reuse COMMAND test_2des_reuse)

# Integration test: FMO-7 end-to-end benchmark comparison. Exit 77 = genuine
# environment skip (missing Python module), shown as "Skipped", not "Passed".
add_test(NAME integration_fmo7
    COMMAND ${CMAKE_COMMAND} -E env
        HEOM_SOLVER=$<TARGET_FILE:heom_solver>
        HEOM_ROOT=${CMAKE_SOURCE_DIR}
        python3 ${CMAKE_SOURCE_DIR}/tests/test_integration.py
)
set_tests_properties(integration_fmo7 PROPERTIES SKIP_RETURN_CODE 77)

# Benchmark infrastructure: template/compare self-consistency (all 3 tasks).
# pytest's prepend import mode inserts tests/ (no __init__.py) onto sys.path,
# so `import regress` resolves without a WORKING_DIRECTORY hack. The two MC
# cases skip themselves when HEOM_TOOLS_DIR is empty.
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
    # Fluorescence gates G4/G5: solver vs NumPy monomer hierarchy and the
    # cumulant reference (heom_tools.common.cumulant).
    add_test(NAME fluorescence_gates_py
        COMMAND ${CMAKE_COMMAND} -E env
            HEOM_SOLVER=$<TARGET_FILE:heom_solver>
            HEOM_ROOT=${CMAKE_SOURCE_DIR}
            PYTHONPATH=${_heom_tools}
            python3 -m pytest ${CMAKE_SOURCE_DIR}/tests/test_fluorescence_gates.py -v
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

# Full regression net: dynamics×3 + absorption + 2des vs committed baselines,
# |Δ| < 1e-12. NOT bitwise: baselines come from another build and carry a
# ulp-scale codegen residual. The bitwise gate is `regress.py bitwise`, run by
# hand across a refactor (--emit before, --emit after, then compare). Fails
# (does not skip) when a baseline is missing — they are tracked files.
add_test(NAME regression_all
    COMMAND ${CMAKE_COMMAND} -E env
        HEOM_SOLVER=$<TARGET_FILE:heom_solver>
        HEOM_ROOT=${CMAKE_SOURCE_DIR}
        python3 ${CMAKE_SOURCE_DIR}/tests/regress.py check --tol 1e-12
)

# The mixed-precision binary is the default production solver; the same set
# against the FP64 baselines at an FP32-scale tolerance. 1e-5 still catches
# any indexing / sign / wiring bug, which moves results by >= 1e-3.
add_test(NAME regression_mp
    COMMAND ${CMAKE_COMMAND} -E env
        HEOM_SOLVER=$<TARGET_FILE:heom_solver_mp>
        HEOM_ROOT=${CMAKE_SOURCE_DIR}
        python3 ${CMAKE_SOURCE_DIR}/tests/regress.py check --tol 1e-5
)

# ctest runs binaries; it does not build them. A plain `ctest` after a source
# edit tests yesterday's solver and reports green — this happened twice.
# Every test above requires this fixture, so a stale
# binary can no longer produce a passing run: the build either succeeds (a
# no-op costs ~1 s) or the whole suite reads "Not Run".
add_test(NAME build_all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --parallel)
set_tests_properties(build_all PROPERTIES FIXTURES_SETUP build)
get_property(_heom_tests DIRECTORY PROPERTY TESTS)
list(REMOVE_ITEM _heom_tests build_all)
set_tests_properties(${_heom_tests} PROPERTIES FIXTURES_REQUIRED build)
else()
message(STATUS "tests/test_core.cpp not found: building heom_solver and heom_solver_mp only")
endif()

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

