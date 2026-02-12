---
sidebar_position: 5
---

# Reproducing results

All benchmarks are run through the `benchmarks` CLI. Docker must be running for the GaussPy+ comparison. Run them in order — each step depends on the previous one.

```bash
uv run benchmarks --help
```

## Step 0: Download data

```bash
uv run benchmarks download
```

Downloads and caches the GRS test field FITS cube and VizieR catalog to `/tmp/phspectra/`. That can be changed with the `--cache-dir` option. The FITS cube is large and the GaussPy+ decompositions take a long time to run, so this only needs to be done once.

## Step 1: GaussPy+ comparison (generates all shared data)

```bash
uv run benchmarks compare
```

Runs phspectra and GaussPy+ (Docker) on 400 real GRS spectra. Produces:

- `/tmp/phspectra/compare-docker/spectra.npz` — the 400 spectra
- `/tmp/phspectra/compare-docker/results.json` — GaussPy+ decompositions + timing
- Plots: `compare-disagreements.png`, `compare-narrower-widths.png`, `compare-rms.png`

## Step 2: Beta training sweep

```bash
uv run benchmarks train-beta
```

Sweeps beta values against the GaussPy+ Docker decompositions from step 1. Produces `train-beta-docker.png`.

## Step 3: Width comparison

```bash
uv run benchmarks width
```

Matches components between phspectra and GaussPy+ and plots width distributions. Produces `width-comparison.png`.

## Step 4: Copy plots to docs

```bash
cp /tmp/phspectra/compare-docker/compare-disagreements-docker.png packages/docs/static/img/results/
cp /tmp/phspectra/compare-docker/compare-narrower-widths-docker.png packages/docs/static/img/results/
cp /tmp/phspectra/compare-docker/compare-rms-docker.png packages/docs/static/img/results/
cp /tmp/phspectra/compare-docker/training-output/train-beta-docker.png packages/docs/static/img/results/train-beta-docker.png
cp /tmp/phspectra/compare-docker/width-comparison.png packages/docs/static/img/results/width-comparison.png
```

## Additional commands

### Inspect a single pixel

```bash
uv run benchmarks inspect PX PY
```

Shows data + GaussPy+ + phspectra at multiple (beta, sig_min) combinations for one pixel.

### Performance benchmark

```bash
uv run benchmarks performance
```

Runs a timing benchmark comparing phspectra vs GaussPy+ (Docker).

### Synthetic benchmark

```bash
uv run benchmarks synthetic
```

Runs phspectra on synthetic spectra with known ground truth across a grid of beta values. Produces F1 scores, CSV results, and plots.
