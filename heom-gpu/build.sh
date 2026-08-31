#!/usr/bin/env bash
# HEOM GPU Solver - Build Script
# Usage: ./build.sh [clean|debug|release|help]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
HEOM GPU Solver - Build Script

Usage: ./build.sh [OPTION]

Options:
    clean           Remove build directory and start fresh
    debug           Build with debug symbols and no optimization
    release         Build with full optimization (default)
    install         Build and install to system
    help            Show this help message

Examples:
    ./build.sh                  # Build with auto-detected GPU architecture
    ./build.sh clean            # Clean and rebuild
    ./build.sh debug            # Build with debug info
    
GPU Architecture:
    The script automatically detects your GPU and selects the optimal
    CUDA architecture (sm_120 for RTX 5060/50-series Blackwell, sm_89 for
    RTX 4090/40-series Ada, sm_86 for RTX 3060, etc.).
    
    To manually override:
    CMAKE_CUDA_ARCHITECTURES=120 ./build.sh   # RTX 5060 / Blackwell
    CMAKE_CUDA_ARCHITECTURES=86 ./build.sh    # RTX 3060
    
Environment Variables:
    CMAKE_CUDA_ARCHITECTURES    Override auto-detected CUDA architecture
    CMAKE_INSTALL_PREFIX        Installation prefix (default: /usr/local)
    
EOF
}

# Parse command line arguments
BUILD_TYPE="Release"
CLEAN=0
INSTALL=0

case "${1:-}" in
    clean)
        CLEAN=1
        ;;
    debug)
        BUILD_TYPE="Debug"
        ;;
    release)
        BUILD_TYPE="Release"
        ;;
    install)
        INSTALL=1
        ;;
    help|--help|-h)
        show_help
        exit 0
        ;;
    "")
        # Default: release build
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac

# Check for required tools
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Please install CMake 3.18 or later."
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    print_error "NVCC not found. Please install CUDA Toolkit."
    exit 1
fi

# Auto-detect CUDA architecture based on available GPU
auto_detect_cuda_arch() {
    # If manually specified, respect user's choice
    if [ -n "$CMAKE_CUDA_ARCHITECTURES" ]; then
        print_info "Using manually specified CUDA architecture: $CMAKE_CUDA_ARCHITECTURES"
        return
    fi
    
    # Try to detect GPU and auto-select architecture
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        
        if [ -n "$GPU_NAME" ]; then
            print_info "Detected GPU: $GPU_NAME"
            
            # Map GPU to CUDA compute capability (specific names before broad patterns).
            case "$GPU_NAME" in
                *"RTX 5090"*|*"RTX 5080"*|*"RTX 5070"*|*"RTX 5060"*|*"RTX 5050"*|*"RTX 50"*|*"Blackwell"*)
                    export CMAKE_CUDA_ARCHITECTURES="120"
                    print_info "→ Auto-selected: sm_120 (Blackwell)"
                    ;;
                *"RTX 4090"*|*"RTX 4080"*|*"RTX 4070"*|*"RTX 5880"*|*"L40"*|*"L4"*)
                    export CMAKE_CUDA_ARCHITECTURES="89"
                    print_info "→ Auto-selected: sm_89 (Ada Lovelace)"
                    ;;
                *"RTX 40"*|*"Ada"*)
                    export CMAKE_CUDA_ARCHITECTURES="89"
                    print_info "→ Auto-selected: sm_89 (Ada Lovelace)"
                    ;;
                *"A100"*|*"A30"*)
                    export CMAKE_CUDA_ARCHITECTURES="80"
                    print_info "→ Auto-selected: sm_80 (Ampere A100)"
                    ;;
                *"RTX 30"*|*"RTX 3090"*|*"RTX 3080"*|*"RTX 3070"*|*"RTX 3060"*|*"A10"*|*"A16"*|*"A2"*)
                    export CMAKE_CUDA_ARCHITECTURES="86"
                    print_info "→ Auto-selected: sm_86 (Ampere GA10x)"
                    ;;
                *"V100"*)
                    export CMAKE_CUDA_ARCHITECTURES="70"
                    print_info "→ Auto-selected: sm_70 (Volta)"
                    ;;
                *"RTX 20"*|*"T4"*|*"Turing"*)
                    export CMAKE_CUDA_ARCHITECTURES="75"
                    print_info "→ Auto-selected: sm_75 (Turing)"
                    ;;
                *"GTX 10"*|*"Pascal"*)
                    export CMAKE_CUDA_ARCHITECTURES="61"
                    print_info "→ Auto-selected: sm_61 (Pascal)"
                    ;;
                *)
                    print_warn "Unknown GPU: $GPU_NAME"
                    print_warn "Using CMakeLists.txt defaults. For optimal performance, manually specify:"
                    print_warn "  CMAKE_CUDA_ARCHITECTURES=<arch> ./build.sh"
                    ;;
            esac
        else
            print_warn "Could not detect GPU name"
            print_warn "Using CMakeLists.txt defaults"
        fi
    else
        print_warn "nvidia-smi not found, cannot auto-detect GPU"
        print_warn "Using CMakeLists.txt defaults"
    fi
}

# Auto-detect CUDA architecture
auto_detect_cuda_arch

# Clean build if requested
if [ $CLEAN -eq 1 ]; then
    print_info "Cleaning build directory..."
    rm -rf build/ CMakeCache.txt
fi

# Create build directory
print_info "Configuring project (Build type: $BUILD_TYPE)..."
cmake -B build \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    ${CMAKE_CUDA_ARCHITECTURES:+-DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES} \
    ${CMAKE_INSTALL_PREFIX:+-DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX}

# Build
print_info "Building project..."
if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
    NUM_CORES="$CMAKE_BUILD_PARALLEL_LEVEL"
else
    NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi
print_info "Parallel jobs: $NUM_CORES"
cmake --build build -j$NUM_CORES

if [ $? -eq 0 ]; then
    print_info "Build successful!"
    print_info "Binaries: ./build/heom_solver ./build/heom_solver_mp"
    
    # Install if requested
    if [ $INSTALL -eq 1 ]; then
        print_info "Installing..."
        if [ -w "${CMAKE_INSTALL_PREFIX:-/usr/local}" ]; then
            cmake --install build
        else
            print_warn "Installation requires root privileges"
            sudo cmake --install build
        fi
    fi
else
    print_error "Build failed!"
    exit 1
fi

