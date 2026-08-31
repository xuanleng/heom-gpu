# Tests

Build first (`../BUILD.md`). The Python tests need `numpy`, `h5py`, `pyyaml`
and `pytest`, and find `../../heom-tools` on their own. The `heom-post`
environment of `heom-tools` has everything.

```bash
cd build && ctest --output-on-failure
```

`ctest` runs the C++ unit tests (`test_core`, CPU only), the GPU kernel tests, task-level tests, the Python tests of the Monte
Carlo tooling, and end-to-end regression of the FP64 binary against
`benchmark_data/` at 1e-12.

The regression scripts take the solver from `$HEOM_SOLVER` (default
`build/heom_solver`):

| Script | Purpose |
|--------|---------|
| `compare_benchmark.py --run-all [--tol 1e-12]` | Run the benchmarks and compare with `benchmark_data/` |
| `compare_benchmark.py REF.h5 TEST.h5 [--tol]` | Compare two output files within a tolerance |
| `bitcmp_h5.py --emit DIR`, `bitcmp_h5.py A B` | Run the benchmarks into DIR, or compare two such directories byte for byte |
| `run_all_benchmarks.py` | Generate the per-site FMO-7 benchmark files |
