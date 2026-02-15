---
sidebar_position: 5
---

# Reproducing results

All benchmarks are run through the `benchmarks` CLI. To get started, run:

```bash
uv run benchmarks --help
```

This will show you all available commands. Each command is documented with `--help` and the source code is available in `packages/benchmarks/src/benchmarks/commands/`. The commands are designed to be modular, e.g.

```
uv run benchmarks download --help
```

will show you how to download the data.

## Step 0: Download data

```bash
uv run benchmarks download
```

Downloads and caches the GRS test field FITS cube (copied from the GaussPy+ repository and persisted in this project for reproducibility)

and VizieR catalog to `/tmp/phspectra/`. The FITS cube is large and the GaussPy+ decompositions take a long time to run, so this only needs to be done once.

## Step 1: GaussPy+ comparison (generates all shared data)

```bash
uv run benchmarks compare --n-spectra 1000 --extra-pixels 31,40
```

Runs phspectra and GaussPy+ (Docker) on 1000 real GRS spectra (plus any extra pixels requested). This runs in serial, so expect it to take a while. Produces:

- `/tmp/phspectra/compare-docker/spectra.npz` -- the spectra
- `/tmp/phspectra/compare-docker/results.json` -- GaussPy+ decompositions + timing
- `/tmp/phspectra/compare-docker/phspectra_results.json` -- PHSpectra decompositions + timing
- `/tmp/phspectra/compare-docker/comparison_docker.json` -- summary statistics

## Step 2: Generate plots

All plot commands read from the saved data in `/tmp/phspectra/compare-docker/` and save figures directly to the docs static image directory via the `@docs_figure` decorator. No manual copying is needed.

### Comparison plots (accuracy section)

```bash
uv run benchmarks compare-plot
```

Produces: `rms-distribution.png`, `rms-scatter.png`, `compare-disagreements.png`, `width-comparison.png`

### N components vs RMS

```bash
uv run benchmarks ncomp-rms-plot
```

Produces: `ncomp-vs-rms.png`

### Performance histogram

```bash
uv run benchmarks performance-plot
```

Produces: `performance-benchmark.png`

## Step 3: Beta training sweep

```bash
uv run benchmarks train-beta
```

Sweeps beta values against the GaussPy+ Docker decompositions from step 1. Produces: `f1-beta-sweep.png`

## Step 4: Synthetic benchmark

```bash
uv run benchmarks synthetic --n-per-category 50
```

Runs PHSpectra on 350 synthetic spectra with known ground truth across a grid of beta values. Produces $F_1$ scores, CSV results, and plots: `synthetic-f1.png`, `synthetic-errors.png`

## Step 5: Persistence diagrams

```bash
uv run benchmarks persistence-plot
```

Generates the water-level-stages and persistence-diagram illustrations used in the [Persistent Homology](../idea-and-plan/persistent-homology-primer) page. These use a synthetic signal and do not depend on any downloaded data.

Produces: `water-level-stages.png`, `persistence-diagram.png`

## Additional commands

### Inspect a single pixel

```bash
uv run benchmarks inspect PX PY
```

Shows data + GaussPy+ + PHSpectra at multiple ($\beta$, $\mathrm{MF}_{\min}$) combinations for one pixel.

### Survey map

```bash
uv run benchmarks survey-map
```

Generates a 2x2 survey visualisation from full-field decomposition results.
