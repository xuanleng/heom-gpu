# Benchmark Data

Generated: 2026-04-02 (n_hat_n bath coupling upgrade)

## Files

### FMO-7 with ADO filtering (tol=1e-7)

| File | Site | Steps | dt(fs) | Time(s) | Tr(ρ) |
|------|------|-------|--------|---------|-------|
| benchmark_fmo7_filtered_site{1-7}.h5 | 1-7 | 500 | 2.0 | ~6s | 1.0 (max\|Tr-1\| ~ 1e-15) |

### FMO-7 without filtering (tol=0)

| File | Site | Steps | dt(fs) | Time(s) | Tr(ρ) |
|------|------|-------|--------|---------|-------|
| benchmark_fmo7_nofilter_site{1-7}.h5 | 1-7 | 500 | 2.0 | ~11.5s | 1.0 (max\|Tr-1\| ~ 1e-15) |

### Large system (FMO-7 double_excited, N=29 Hilbert dim)

| File | Config | L | K | Steps | M (ADOs) | Time(s) | GPU mem |
|------|--------|---|---|-------|----------|---------|---------|
| benchmark_fmo7_n29.h5 | benchmark_fmo7_n29.yaml | 3 | 2 | 100 | 2024 | 3.7 | 182 MB |

**Note**: With n_hat_n bath coupling, hierarchy dimension = n_states×(K+1) = 7×3 = 21 (not n_sites×(K+1) = 29×3 = 87), giving M=2024 ADOs instead of ~117K. This is physically correct: each chromophore site has one independent bath, regardless of basis size.

### Fluorescence (2.1.1)

| File | Config | L | K | Steps | dt(fs) | t_eq (fs) | Time(s) |
|------|--------|---|---|-------|--------|-----------|---------|
| benchmark_fluorescence_fmo7.h5 | benchmark_fluorescence_fmo7.yaml | 3 | 2 | 1500 | 2.0 | 22250 (tol 1e-6) | ~7 |

FMO-7, with_ground, 77 K, filter off (the fluorescence task rejects filtering). FP64 baseline; `regression_mp` checks the MP build against it at `--tol 1e-5` (measured Δ 3e-6).

### Legacy

| File | Note |
|------|------|
| benchmark_fmo7.h5 | Same params as filtered_site1, kept for backward compat |

## Filtered vs Nofilter comparison

Max element-wise difference (tol=1e-7 filtering vs exact):
- max |Δρ_ij| ~ 1e-3 (across all 7 sites, 1000 fs simulation)
- Tr(ρ) conserved identically in both cases
- Filtering provides ~2x speedup (6s vs 11.5s)

## Common parameters

- System: FMO-7, excited_only basis (7 sites)
- Temperature: 77 K
- Bath: Drude-Lorentz, η=70 cm⁻¹, 1/γ=100 fs
- Hierarchy: L=3, K=5 (14190 ADOs)
- GPU: NVIDIA GeForce RTX 3060 (12GB)

## Usage

```bash
# Self-check single file
python tests/regress.py inspect tests/benchmark_data/benchmark_fmo7_filtered_site1.h5

# Compare filtered vs nofilter for same initial state
python tests/regress.py diff \
  tests/benchmark_data/benchmark_fmo7_filtered_site1.h5 \
  tests/benchmark_data/benchmark_fmo7_nofilter_site1.h5 --tol 1e-2

# Regression test: compare same config before/after code change
python tests/regress.py diff \
  tests/benchmark_data/benchmark_fmo7_filtered_site1.h5 <new_result.h5> --tol 1e-12

# Generate the 14 per-site runs (skips existing files)
python tests/regress.py sites

# Rewrite the committed gate baselines (deliberate re-baseline only)
python tests/regress.py regen [benchmark_fmo7 ...]
```
