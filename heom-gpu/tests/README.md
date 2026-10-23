# Tests

Build first (`../BUILD.md`). The Python tests need `numpy`, `h5py`, `pyyaml`
and `pytest`, and find `../../heom-tools` on their own. The `heom-post`
environment of `heom-tools` has everything.

```bash
cd build && ctest --output-on-failure
```

`ctest` runs the C++ unit tests (`test_core`, CPU only), the GPU kernel tests, task-level tests, the Python tests of the Monte
Carlo tooling, and end-to-end regression against `benchmark_data/` (FP64
binary at 1e-12, mixed-precision binary at 1e-5).

The regression scripts take the solver from `$HEOM_SOLVER` (default
`build/heom_solver`):

| Command | Purpose |
|---------|---------|
| `regress.py check [--tol 1e-12]` | Run the benchmarks and compare with `benchmark_data/` |
| `regress.py diff REF.h5 TEST.h5 [--tol]` | Compare two output files within a tolerance |
| `regress.py inspect FILE.h5` | Shapes, trace and final populations of one file |
| `regress.py bitwise --emit DIR`, `regress.py bitwise A B` | Run the benchmarks into DIR, or compare two such directories byte for byte |
| `regress.py regen` | Overwrite the reference files with the current binary |
