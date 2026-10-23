# heom-gpu 2.2.0

GPU solver for hierarchical equations of motion (HEOM) with a Drude–Lorentz
bath. Released under the **MIT License** (`LICENSE`).

Figure scripts and production data for the accompanying article are on
Zenodo and are not needed to build this tree. The post-processing package
is in `../heom-tools/`.

## Requirements

- NVIDIA GPU (compute capability 6.1 or newer)
- CUDA Toolkit 11 or newer (12.8 or newer for Blackwell, e.g. RTX 50-series)
- CMake 3.18 or newer
- C++17 / CUDA 17 compiler (GCC 9 or newer is known to work)
- Eigen 3.3, HDF5 (C), yaml-cpp
- OpenMP (optional)

Debian/Ubuntu:

```bash
sudo apt install -y cmake build-essential libeigen3-dev libhdf5-dev libyaml-cpp-dev
```

## Build

```bash
./build.sh
```

Override the architecture when the compile node has no GPU, or when you
target a card the probe does not know:

```bash
CMAKE_CUDA_ARCHITECTURES=89 ./build.sh    # Ada (RTX 4090 / 4090 D)
CMAKE_CUDA_ARCHITECTURES=120 ./build.sh   # Blackwell (RTX 5060); needs CUDA >= 12.8
```

Binaries land in `build/`:

| Binary            | Precision                         |
|-------------------|-----------------------------------|
| `heom_solver`     | FP64                              |
| `heom_solver_mp`  | mixed FP32 (manuscript production) |

```bash
./build/heom_solver --version
```

Optional checks: `cd build && ctest --output-on-failure`.

## Run

Tasks are `dynamics`, `absorption`, `fluorescence`, and `2des`. Each run reads one YAML.

```bash
./build/heom_solver_mp dynamics     config/fmo.yaml
./build/heom_solver_mp absorption   config/fmo.yaml
./build/heom_solver_mp 2des         config/fmo.yaml
./build/heom_solver    fluorescence config/fmo_fluorescence.yaml
```

HDF5 output carries `/provenance` attributes `software_name=heom-gpu` and
`software_version=2.2.0`. Do not strip or alter them.

## Cite

Any publication that uses this solver, or numbers obtained with it, must
cite the article and name the software:

> X.-l. Ren, T. Zhang, J. Cheng, W.-z. Zhang, X. Leng,
> GPU-Accelerated Hierarchical Equations of Motion for Polarization-Resolved
> Two-Dimensional Electronic Spectra of Disordered Excitonic Systems,
> *J. Chem. Theory Comput.* **2026**. Use the published DOI when available.

Identify the code as **heom-gpu 2.2.0**. Questions about the release:
lengxuan@nbu.edu.cn.
