# heom-tools

Python package (`import heom_tools`) that post-processes `heom-gpu` output and
runs static-disorder Monte Carlo averages on top of the solver.

## Install

From the repository root:

```bash
conda env create -f heom-tools/environment-heom-post.yml
conda activate heom-post
pip install -e "heom-tools[rqmc]"
```

Tests: `cd heom-tools && pytest`.

## Commands

| Command | Purpose |
|---------|---------|
| `pop-plot`, `pop-compare` | Population dynamics: plots and comparison |
| `las-post`, `las-plot`, `las-compare` | Linear absorption: spectrum from the solver output, plots, comparison |
| `twodes-post`, `twodes-plot`, `twodes-animate`, `twodes-compare`, `twodes-merge` | 2DES: spectra, plots, animations, comparison, merging along the waiting time |
| `mc-disorder-gen`, `mc-disorder-run`, `mc-disorder-merge`, `mc-disorder-status` | Static disorder: draws, GPU workers, merging, error estimate |
| `mc-probe` | Sampling-error studies of the Monte Carlo average |

Every command takes `--help`. `twodes-animate` needs `ffmpeg` for MP4 output.
The solver build is described in `../heom-gpu/README.md`.
