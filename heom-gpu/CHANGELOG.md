# Changelog

Public releases are git tags on this repository (`v1.0.0`, `v2.2.0`); the other
versions below were internal. The version
string lives in three places that must agree: `VERSION`,
`include/heom_version.hpp`, and `project(... VERSION ...)` in `CMakeLists.txt`.
Every HDF5 file records the string it was written by in
`/provenance/software_version`.

## 2.2.0 — 2026-10-23

Solver physics stays bit-identical to 2.1.2 on the validated cases. Filtered
batched 2DES now keeps one active set per slot, so those runs are not
compared bitwise against the old shared active set.

### Added / Changed

- **Faster derivative kernel.** Term-4/5 bath couplings are evaluated with
  bit masks, and the Hamiltonian is staged in shared memory (part or whole
  matrix). LHC-II L3 ESA mixed-precision derivative kernel **~2.88×** vs
  2.1.2, FMO ~1.08–1.17×. Bit-identical to 2.1.2 on the validated cases.

- **`2des.batch: auto`.** Chooses 21→7→1 by GPU budget (`usable = free×0.95`);
  writes choice into HDF5. Explicit 21/7/1 unchanged. Auto→21 bit-exact vs
  explicit 21 on K1L3.

- **Per-slot active sets + flat derivative launch.** Filtered batched 2DES
  keeps one active set per slot; compact launch uses `block_off` binary search.
  FMO K2L5 filtered: ~1.03× vs the previous launch; still slightly slower than
  B=1; bit-identical across B=21/B=1.

- **Fused ADO max-norm in kFinal** when filtering (past the fill window).
  Skips the standalone norm kernel; filter takes `sqrt` of IEEE bits.
  Production filtered K1L5/K2L5 wall ~−3%; K1L3 filter-on was an artificial
  test point (~−14%). Bit-identical to standalone norms.

## 2.1.2 — 2026-10-09

2DES post-processing fix and a CLI rename. Solver output is bit-identical to
2.1.1; only the version string changes.

### Changed

- **`fl-post` / `fl-plot` are renamed `flu-post` / `flu-plot`.** The old
  names are gone, no aliases. Processed fluorescence HDF5 now records
  `entry_point = "flu-post"`; files written by 2.1.1 (`"fl-post"`) still
  load in `flu-plot`, which checks `spectrum_type` only. After pulling,
  reinstall heom-tools (`pip install -e heom-tools`) so the console
  scripts are regenerated.

- **`twodes-post` weights the τ = 0 row and the t = 0 column by one half
  (corner ¼)**, the trapezoid rule 2.1.1 applied to the linear spectra, on
  both one-sided axes. rp and nr coincide at τ = 0 and are summed into the
  2DES total, so full weight counted that row twice. The old maps carried
  stripes through every peak along both frequency axes of ≈ Γ·Δ/4 of the
  peak and a peak lift of about twice that: 8 fs grid, 100 cm⁻¹ line →
  ~4 % stripes, peak +8 %. Weak features sitting on those stripes move the
  most: on the FMO maps of the JCTC paper (10 fs grid) the absorptive
  cross peaks drop by 0.01–0.02 of the map maximum, 40–65 % of their own
  height at T = 0, and hierarchy-truncation ε rises by 0.1–0.2 percentage
  points because the old stripes were common to test and reference. `fft_method: check`
  never flagged it because the legacy reference had the same weights.
  Peak positions, time-domain plots and solver output are unchanged; rerun
  `twodes-post` on the existing HDF5. Do not compare old `*_post.h5`
  spectra with spectra computed by the 2.1.2 `two_d_fft`: the endpoint
  weights differ, and the difference is the artifact itself.

## 2.1.1 — 2026-10-09

Steady-state fluorescence.
Solver output of the existing tasks is bit-identical to 2.1.0 apart from two
added datasets; processed absorption spectra move (see Changed).

### Added

- **`heom_solver fluorescence <cfg>`.** Equilibrates the (e,e) block of the
  whole hierarchy in real time (bare-H_S Boltzmann σ_0, higher ADOs from the
  displaced-start closed form, exact for one pigment), right-multiplies every
  ADO by μ⁺ and records ρ_α(t) of the (e,g) block exactly as absorption does.
  Config block `fluorescence:` (keys as `absorption:` plus a required
  `equilibration: {t_max, check_interval = 50, tolerance = 1e-6}`). Stop rule
  max(δ, ε) ≤ tolerance on the whole-hierarchy relative distance; a run that
  does not converge by `t_max` is an error that prints the last (δ, r, ε) and
  whether δ is at a floor or still decaying.
- Fluorescence HDF5: `rho_xyz` as absorption, plus `/results/rho_eq` and
  `/results/equilibration/{time_fs,delta,ratio,estimate}` (attrs `t_eq_fs`,
  `tolerance`, `check_interval_fs`, `t_max_fs`).
- Every task's HDF5: `/parameters/reorg_cm` (λ per pigment) and
  `/derived/markov_residue_cm`.
- MC static disorder: `mc-disorder-run fluorescence <cfg>` (worker
  `--rank N`); shards merge on the existing `rho_xyz` path.
- heom-tools: `fl-post` (spectrum, per-exciton split, sticks p_k·f_k with
  HEOM / Boltzmann populations side by side, `--omega3`, MC shard folding,
  `--absorption <abs.h5>` KMS fit → T_eff after checking both files share
  system/bath/filter) and `fl-plot`. The linear-spectrum core is shared with
  `las-post` (`heom_tools.common.linear_spectrum`);
  `heom_tools.common.cumulant` gives the exact monomer C_A / C_F on the
  solver's own expansion.
- Gates: `test_fluorescence` (C++), `fluorescence_gates_py` (solver vs NumPy
  hierarchy to 1e-16, vs the cumulant closed form, Stokes shift, FMO-7 KMS),
  regression baseline `benchmark_fluorescence_fmo7`, example
  `config/fmo_fluorescence.yaml` (T_eff −0.05 % at 77 K). README / BUILD.md
  list the new task.

### Rejected configurations (fluorescence only)

- `integrator: if-rk4`: its fixed point is biased against the steady state
  by O(dt^5) (one pigment: 3.5e-2 at dt = 2 fs; FMO with modes: 1.7e-3).
- `filter.enabled: true`: the ADO filter pins δ at a floor 0.2–100×
  `filter.tolerance` depending on the system (LHC-II: ~1e-4 at 1e-6), so the
  stop rule cannot be trusted.

### Changed

- **heom-tools linear spectra (`las-post`, `fl-post`) weight C(t_0) by one
  half** (trapezoid). The Fortran-reference rectangle added Re C(0)/(2·nfft)
  to every frequency, a flat baseline of 0.6–1.1 % of the peak on a dimer; it biased the KMS
  fit by several percent. Processed absorption spectra therefore shift by that
  constant; line shapes and peak positions do not change.
- `Config::absorption_batch` → `linear_batch` (internal; the YAML key is
  still `batch:`).

### Fixed

- `mc-disorder-run 2des` printed `las-post` as the next step; it now names
  `twodes-post`.

### Usage notes (measured)

- Fluorescence needs a deeper L than absorption: its start fills every tier.
  One pigment at 77 K, same L: the fluorescence error is 4–100× the
  absorption error. Check L on the fluorescence itself.
- Padé K = 1 caps the FMO-7 KMS temperature at ~ −4 % for any L; K = 2,
  L = 5 gives −0.05 %.
- Equilibration time is set by the slowest relaxation, often exciton
  population transfer rather than the bath: FMO-7 τ ≈ 2.7 ps (21–26 ps at
  1e-6 depending on K, L; the example config takes under 4 min on an RTX 3060), LHC-II with three modes τ ≈ 11 ps (~85 ps extrapolated).
- Mixed precision: the steady state is good to ~5e-6 of FP64 and the ratio r
  is noisy near the floor; tolerance below ~1e-5 buys nothing and slow
  systems can stop early.

## 2.1.0 — 2026-10-05

The last release of the dense-hierarchy line: every ADO with Σα ≤ L is
stored. 3.0.0 is reserved for the sparse-storage rework (only the active ADOs
stored), so this release is a minor bump even though it carries the breaking
changes below; check them before rerunning a 2.0.0 config.

### Breaking

- **The config parser rejects what it used to ignore**. An
  unknown key in any block is an error that lists the legal keys; a value that
  does not decode (`enabled: ture`, `n_steps: 20.0`) is an error instead
  of the default; every number must be finite (`tolerance: .nan` dropped the
  whole hierarchy). `bath.spectral_density`, if given, must be
  `Drude-Lorentz`. Configs that loaded before still load under this change
  alone (the warmup removal below is separate).
- **`2des.tau_max / dtau / t_max / dt_detect` are required**. A missing
  axis used to become a one-point axis.
- **`n_steps` must be a multiple of `output_interval`** for dynamics and
  absorption. The steps after the last record were never written and never
  checked for divergence.
- **Every result file carries a root `status` attribute**:
  `running` from the moment the file is created, then exactly one of
  `complete`, `diverged`, `aborted` as the very last write, so a killed run
  is left as `running`. Dynamics files also carry `n_records_written` (rows
  past it are preallocated padding); the old `incomplete = 1` attribute is
  gone. Spectrum files write `simulation_time_seconds` after `/results`, not
  before. heom-tools readers (population, spectrum, 2DES post-processing,
  2DES merge, MC shard merge) raise `IncompleteRunError` on any `status`
  other than `complete`; files without the attribute (written before it
  existed) read as before.
- **Worker flags are strict**. `--rank`, `--r-begin`, `--r-end` must be
  whole integers (`--rank 1x` used to mean 1), and `--run-id` / `--r-begin` /
  `--r-end` without `--rank` are an error instead of being ignored.
- **Timestamps have seconds; a solver-chosen name never overwrites**.
  `use_timestamp: true` names end in `_YYYYMMDD_HHMMSS` (was `_HHMM`) and
  the file is created exclusively: if it already exists (the same run
  finished in the same second) the solver stops with an error. Spectrum
  files are created after the compute, so that error comes at the end.
  `use_timestamp: false` still overwrites, with an `[note] overwriting ...`
  line. The default MC run-id and the heom-tools launcher run-ids are
  `YYYYMMDD_HHMMSS` too; `SHARD_RE` already accepted both.
- **The MC queue file is per batch**: `mc.<run_id>_r<A>-<B>.queue`
  instead of one `mc.queue` per `save_path`, and `MCOptions::queue` is gone.
  Launchers delete it after every worker has exited instead of resetting it
  before the batch.
- **The filter warmup is gone**. `filter.start_step` and
  `filter.warmup_factor` are rejected with an error that names them, even with
  the filter off; `filter.tolerance` is the one threshold for the whole run.
  With the fill-window fix below, the warmup changed nothing for 2DES (byte-identical) or
  absorption; for dynamics it only tightened the first ~300 fs, by less than
  1e-7 absolute. Lower `tolerance` about 10x to get that back. Configs that
  set the keys (the archived paper and release data) need heom-gpu <= 2.0.0;
  every live config in the repository has been updated. `config_hash` moves
  for every config that had the keys. FP64 baselines regenerated: relative to the filter-off
  result, the distance moves from 1.8e-9 to 3.2e-9 for `benchmark_fmo7`, from
  2.4e-8 to 3.7e-8 for `_n29` and from 3.06e-8 to 3.00e-8 for `_with_ground`;
  absorption and 2DES do not change at that level (4.6e-7 and 5.5e-8).

### Fixed

- **The filter waits until the hierarchy has filled**. After a ρ
  replacement (trajectory start, dipole) every ADO used to be active for one
  step only. One RK4 step does not fill an FMO hierarchy (it takes ⌈L/2⌉), so
  the ADOs still exactly 0 went inactive at any tolerance and the neighbor
  gate let the hierarchy grow one level per step. At IF dt=10 this set an
  error floor that no tolerance could lower: 3.7–7.4e-7 for dynamics
  (L = 4–6), 3.5e-8 / 1.5e-7 for 2DES (L = 4 / 5). Now every ADO stays active
  for as long as each full step lowers the number of exact-zero ADOs and some
  are left; the count rides the 2DES snapshots. Tolerance 1e-300 is
  byte-identical to the filter off for FMO-7 L=5 dynamics and 2DES (new tests
  `test_vanishing_tolerance_fmo_dynamics`, `test_vanishing_tolerance_fmo_2des`).
  When the hierarchy is already full (most 2DES dipoles) the cost is one norm
  scan. The FP64 baselines move at the filter-error level and are equally far
  from the filter-off results as before (`benchmark_fmo7` 2.8e-11,
  `_n29` 4.1e-9, `_with_ground` 1.8e-9, `benchmark_absorption_fmo7` 1.7e-9,
  `benchmark_3sites_2des` 1.5e-7 with max |R| = 9); they are regenerated.
- **`exciton_index` with `basis: with_ground`** selected the ground
  state whenever an exciton lay below w0. The eigenbasis is now built per
  excitation manifold, so exciton k is the same state in every basis.
  `derived/eigenvalues_relative_cm` and `derived/eigenvectors` of with_ground
  / double_excited files are ordered by manifold (ascending within each), not
  globally.
- **2DES checks every (signal, pathway) before the first one runs**: a
  bad name, a duplicate or an over-budget ESA no longer discards finished
  pathways.
- **2DES snapshots are lossless**. A snapshot used to keep only the
  ADOs active at capture and zero the rest on restore, so with the filter on
  the τ/T chains (which snapshot at every sample) depended on the sampling
  grid. It is now a verbatim copy of the whole state — ρ, flags, the mixed
  build's Kahan carry, the fresh flag, the divergence baselines — and
  propagating on from a restore is bit-identical to never having branched,
  in both builds. The snapshot buffer is sized once. 2DES results with
  the filter on move at the filter-error level: `benchmark_3sites_2des` by
  9e-8 (max |R| = 9), equally far from the filter-off result as before; its
  baseline is regenerated. The mixed build's 2DES budget reserves twice the
  snapshot memory for the carry.
- **Divergence is judged per slot**. Each slot is held to 10× its own
  ‖ρ₀‖ from the last `set_initial` / dipole, restores and fan-outs inherit
  it, and a weak slot batched with a strong one alarms on the same step it
  would alone (it used to share the largest slot's ceiling). The report names
  the slot and its baseline.
- **A snapshot is replayed only where it was taken**. Restore and
  fan-out reject a snapshot from another `SystemContext`, an unfilled one, one
  wider than the target and an out-of-range slot map with
  `std::invalid_argument`, before writing anything; a foreign snapshot used to
  scatter out of bounds.
- **IF-RK4 dense launches freeze ADOs outside U**, as compact launches
  always did; the dense path used to decay them by E2 every step. Dense and
  compact (`filter.ulist` 0, 1 or adaptive) are now byte-identical for ρ and
  the Kahan carry. Plain RK4 was already identical. FMO-7 L=6 IF dynamics
  with `ulist: 0` moves by 1.7e-16.
- **The 2DES memory guard no longer counts the system tables twice**:
  the hierarchy maps already resident in `SystemContext` are shown in the
  budget printout but not added to the total, and the 2DES snapshots get
  their own row. The U-closure arrays are counted whenever the filter is on.
  The effect is about 0.2% of the FMO 2DES budget.
- **`load_realizations` checks attribute shapes**: a `sigma_cm` whose
  length is not `n_site`, an array-valued scalar attribute, or a `weight` /
  `group` of the wrong shape is an error instead of an out-of-bounds read.
- **Batched absorption logs `Step N (t=…)` lines again**; with
  `absorption.batch: true` the filter statistics had lost their step and
  time.
- **Launch index ceiling**: `M·Nb` must leave one block of headroom
  below `INT_MAX`; within the last 255 values the rounded-up grid and the last
  block's thread index wrapped negative and passed the bounds check.
- **A new MC batch can no longer inherit a drained queue**. Workers
  started with `--rank` directly (no launcher reset) used to read the
  previous batch's exhausted counter, claim 0 nodes and exit 0. With the
  run-id and interval in the queue name, only a re-join of the same batch
  sees its own counter.

### Changed

- **`tools/` is now `heom-tools`**, the post-processing package.
- **The shipped `config/fmo.yaml` is a cheap demo**: `L_cutoff: 3`,
  `n_extra_poles: 1`, `integrator: if-rk4`. Results from it differ from the
  2.0.0 copy; it is not a production hierarchy.
- **The test suite is a gate**: `ASSERT_NEAR` rejects NaN/Inf,
  `HEOM_TOOLS_DIR` wires heom-tools into the C++ and Python tests (skipped
  with a warning in a solver-only tree), every test writes under its own
  temporary directory, and `regression_mp` covers the mixed-precision binary.
- **The site reorganization energy** λ_n = η/2 + Σ Sω₀ is printed as such
  (the summary used to say η/2) and pinned by a test; the shift itself was
  already applied, so no result moves.

## 2.0.0 — 2026-09-09

Everything after the JCTC manuscript freeze. The physics of a 1.0.0 config
is unchanged — FP64 `regression_all` holds at |Δ| < 1e-12 through every
refactor below — but the config schema grew, one knob was removed, the HDF5
`config_hash` means something different, and the solver core was rebuilt.
Hence a major bump.

One fixture note: the 3-site filter-on 2DES baseline was regenerated.
The 1.0.0 tree shipped a stale copy — a change of 2026-08-24, in 1.0.0,
had already changed filter-on ADO growth at segment starts and the
fixture was never re-recorded; a stale `build/` kept the gate green. The new
baseline is what 1.0.0's own source produces when actually rebuilt.

### Breaking

- **`config_hash` is task-scoped physics only**. It hashes the keys
  that determine the numbers a task produces and nothing else (no output
  paths, no unrelated task blocks). Hashes written by 1.0.0 do not compare
  equal to 2.0.0 hashes of the same config; `twodes-merge` refuses to mix
  shards across the boundary, which is the intended behaviour.
- **`HEOM_ULIST_TAU` environment override removed**. `filter.ulist`
  in the YAML is the only source of τ, and it is part of `config_hash`.
- **Required config keys fail with a path**. A missing or mistyped
  `system.temperature` says `system.temperature is required` /
  `cannot read 'x' as a number`. Configs that relied on a silently
  defaulted required key stop loading.
- **Vibrational-mode `gamma_inverse` is the C(t) decay time**. For an
  underdamped Brownian oscillator `gamma_inverse` is 1/γ in fs with the same
  meaning as the Drude key; the Lorentzian FWHM is 2γ. Papers that write
  Γ (FWHM) in the denominator need `gamma_inverse = 5308.8 / (Γ_cm / 2)`.
  New key, so no 1.0.0 config is affected, but the factor of 2 is a trap.
- **Dense-ρ fallback and the flat 21-orientation 2DES walk removed**. ρ is always stored compact; 2DES always runs the 3→9→21
  reuse walk and fails loud on budget instead of falling back.
- **Test and script entry points renamed**.
  `tests/compare_benchmark.py`, `tests/bitcmp_h5.py`,
  `tests/run_all_benchmarks.py` → `tests/regress.py {check,diff,inspect,
  bitwise,regen,sites}`.

### Added

- **Per-pigment baths**. `bath.sites.<n>` overrides `eta`,
  `gamma_inverse`, `n_extra_poles` (K_n) per pigment; hierarchy width is
  D = Σ_n (1 + K_n + 2·modes_n). Example: `config/fmo_vib.yaml`.
- **Underdamped vibrational modes**. `vibrational_modes:
  [{omega_cm, huang_rhys, gamma_inverse}]` at bath or site level; complex
  pole pipeline (ν, E complex; s_k = sqrt|c c̄|).
- **Manifold compaction**. ρ stores only its own manifold block
  `[B][M][Nb]`. LHC-II N=106 ESA D=28: GPU total 1.88 GiB → 220.65 MiB
  (8.72×) at no per-element cost; D=42/56/70 now fit one 8 GiB card.
  Ceiling is `M·Nb < 2^31`.
- **Real-part H in shared memory**. 1.88× on LHC-II ESA; FP32 results
  differ from 1.0.0 at the 1e-8 relative level (FMA order), FP64 unchanged.
- **HDF5 `incomplete` attribute**. A file whose writer did not reach
  close-out carries `incomplete=1` at the root; readers can tell a crashed
  run from a finished one without counting records.
- **Full memory budget**. `[MEMORY BUDGET]` counts every device and
  host allocation the solver makes, including the ones 1.0.0 omitted.
- Unified test harness `tests/harness.hpp`; one `regress.py`; CLI
  `--r-begin/--r-end` and `--rank` carried as `std::optional`.

### Fixed (silent-error batch, 2026-09-04)

- ADO filter lets NaN through the norm and raises the divergence flag
  instead of masking it.
- MC queue treats short or negative reads as corruption; shards are built
  under `.partial/` and renamed into place once `/mc` is appended.
- The 6-digit sigma identity for MC realizations is an explicit on-disk
  contract with a pinned locale.
- Regression gate fails on zero comparisons; ctest builds before testing.
- Every int launch/index product goes through one checked helper.

### Internal

- One `ManifoldLayout` / `ModeLayout` (prefix table) for every index. `HEOMTrajectory` keeps only the batched API; detection lives in
  `heom::Detector`; the output loop and HDF5 live outside the core.
- One MC worker template with a `kTasks.mc_run` column; one
  `require<T>` for config keys; one CMake solver stack.

## 1.0.0 — 2026-08-29

The solver cited by the JCTC manuscript (tag `v1.0.0`).
Produced the data in the Zenodo record 10.5281/zenodo.22156602.
