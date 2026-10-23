# heom-gpu

CUDA/C++ solver for the hierarchical equations of motion (HEOM) of
excitonic systems. Each pigment couples to its own Drude–Lorentz bath, optionally with underdamped Brownian-oscillator modes,
expanded in Matsubara or Padé exponentials. The hierarchy is pruned on the
fly by dynamic ADO filtering and propagated with RK4 or an
integrating-factor RK4. Tasks: population dynamics, linear absorption, fluorescence
and polarization-resolved two-dimensional electronic spectra (2DES).

## Build

See `BUILD.md`. In short:

```bash
./build.sh                                # Release build, GPU architecture detected
CMAKE_CUDA_ARCHITECTURES=89 ./build.sh    # or name the architecture
```

| Binary | Precision |
|--------|-----------|
| `build/heom_solver` | FP64 throughout; reference for the regression tests |
| `build/heom_solver_mp` | Mixed FP32/FP64; production runs, default of `mc-disorder-run` |

## Run

```bash
./build/heom_solver    dynamics     config/fmo.yaml
./build/heom_solver_mp absorption   config/fmo.yaml
./build/heom_solver_mp 2des         config/fmo.yaml
./build/heom_solver    fluorescence config/fmo_fluorescence.yaml
```

- A YAML file holds one block per task; a run executes the task named on the
  command line.
- `config/fmo.yaml` is the commented reference configuration (FMO, 7 pigments,
  77 K). `config/fmo_fluorescence.yaml` adds
  fluorescence, and `config/fmo_vib.yaml` an underdamped mode per pigment. The keys are parsed in `src/heom_config.cpp`.
- Energies are in cm⁻¹, times in fs, and pigment indices start at 1.
- Every task block must set `integrator` (`rk4` or `if-rk4`); there is no
  default. Fluorescence requires
  `rk4` and `filter.enabled: false`.
- `filter.tolerance` is the ADO filtering threshold. `filter.enabled: false`
  keeps the full hierarchy.
- The task fixes the basis: `dynamics` uses the single-excitation manifold,
  `absorption` and `fluorescence` add the ground state, and `2des` adds the
  double-excitation manifold.

## Output

Each run writes one HDF5 file to `output.save_path` of its task block.
Dynamics files hold the site-basis reduced density matrix at every output
time, spectroscopy files the response functions. Every file also stores the
Hamiltonian, the bath expansion, the hierarchy parameters, the configuration
hash and the precision mode. `h5ls -r` shows the layout, and
`heom_tools.common.heom_h5` reads it from Python.

## Static disorder

Absorption, fluorescence and 2DES are averaged over static disorder with the Monte
Carlo driver of `../heom-tools`:

```bash
mc-disorder-gen --out config/nodes_r0.h5 --method iid --n-node 64 --n-site 7 --fwhm 100 --seed 1
# set static_disorder.draws_file in the YAML to nodes_r0.h5
mc-disorder-run absorption config/fmo.yaml --gpus 0,1
las-post <save_path>/*_rank1.h5           # merge the shards, then the spectrum
```

`--method rqmc --n-replicas R` writes R randomized quasi-Monte Carlo replicas
and a manifest for `mc-disorder-run --manifest`. `mc-disorder-merge
--combine-replicas` combines the replicas and `mc-disorder-status` reports the
error estimate. Shards from FP64 and mixed-precision runs are not merged
together.

## Tests

```bash
cd build && ctest --output-on-failure
```

See `tests/README.md`.

## License

MIT, see `LICENSE`. To cite, see `../CITATION.cff`.
