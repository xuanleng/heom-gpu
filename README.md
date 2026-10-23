# heom-gpu

GPU solver for the hierarchical equations of motion (HEOM) of excitonic
systems coupled to harmonic baths, with Python post-processing for
population dynamics, linear absorption and polarization-resolved
two-dimensional electronic spectra (2DES) of disordered systems.

This snapshot is tag `v2.2.0`.

The code accompanies

> X.-l. Ren, T. Zhang, J. Cheng, W.-z. Zhang and X. Leng,
> *GPU-Accelerated Hierarchical Equations of Motion for Polarization-Resolved Two-Dimensional Electronic Spectra of Disordered Excitonic Systems*, J. Chem. Theory Comput., submitted.

## Versions

| Tag | Use |
|-----|-----|
| `v1.0.0` | Solver of the original submission; also the manifold-blocking comparison of the revised manuscript |
| `v2.2.0` | All other results of the revised manuscript |
| `v2.2.1` | `v2.2.0` plus the first-tier system-bath energy output (Supporting Information, Section S1); density matrices are bit-identical to `v2.2.0` |

`git diff v2.2.0 v2.2.1` shows that one change.

## Layout

| Directory | Contents |
|-----------|----------|
| `heom-gpu/` | CUDA/C++ solver: usage in `heom-gpu/README.md`, build in `heom-gpu/BUILD.md` |
| `heom-tools/` | Python post-processing and static-disorder Monte Carlo (`heom-tools/README.md`) |

## Build

Requirements: an NVIDIA GPU, CUDA Toolkit 11 or newer (12.8 or newer for
RTX 50-series), CMake 3.18 or newer, a C++17 compiler, Eigen 3.3, HDF5 and
yaml-cpp.

```bash
cd heom-gpu
./build.sh
```

This produces `build/heom_solver` (FP64) and `build/heom_solver_mp` (mixed
FP32, used for the production runs). `ctest` in `heom-gpu/build` runs the
test suite.

## License

MIT, see `LICENSE`. To cite the software, see `CITATION.cff`.
