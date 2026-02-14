---
sidebar_position: 4
---

# Training tool

The `train-gui` package provides an interactive matplotlib viewer for curating a hand-labeled training set of Gaussian components. The curated training set is used by `benchmarks train-beta` to measure $F_1$ against human-verified ground truth instead of raw GaussPy+ output.

## Prerequisites

Run `benchmarks compare` first to generate the comparison data that the viewer loads:

```bash
uv run benchmarks download
uv run benchmarks compare --n-spectra 1000 --extra-pixels 31,40
```

## Launching the viewer

```bash
uv run train-gui
```

The viewer opens a two-panel matplotlib window. The upper panel shows the spectrum and fitted components; the lower panel lists all available components with their key bindings.

### Options

| Flag             | Default                                     | Description                                |
| ---------------- | ------------------------------------------- | ------------------------------------------ |
| `--data-dir`     | `/tmp/phspectra/compare-docker`             | Directory with `benchmarks compare` output |
| `--training-set` | `packages/train-gui/data/training_set.json` | Path to the JSON file                      |
| `--start-index`  | `0`                                         | Pixel index to start at                    |
| `--survey`       | `GRS`                                       | Survey name stored in each entry           |

## Controls

| Key           | Action                                                   |
| ------------- | -------------------------------------------------------- |
| `a`&ndash;`z` | Toggle the component with that label (shown on the plot) |
| Left / Right  | Navigate to the previous / next pixel                    |
| `s`           | Save the training set to disk                            |
| `c`           | Clear all selected components for the current pixel      |
| `q`           | Save and quit                                            |

## Residual view

The plot shows two signals:

- **Dashed gray line** &mdash; the original spectrum, for context.
- **Solid gray line** &mdash; the residual after subtracting all selected (green) components.

As you toggle components on and off, the residual updates immediately. This lets you judge whether the selected set accounts for all visible features in the spectrum: if the residual is flat, the decomposition is complete.

## Manual Gaussian fitting

When neither GaussPy+ nor PHSpectra detected a component that you can see in the residual, you can fit one manually:

1. Click and drag horizontally across the feature in the residual to define a channel range.
2. On release, `scipy.optimize.curve_fit` fits a single Gaussian to the residual within that range. Initial guesses are derived from the span (peak amplitude, center position, width / 4).
3. The fitted component is added to the training set with `"source": "manual"` and drawn immediately in orange with a `MAN` label.

If the fit does not converge, a message is printed to the terminal. Try a wider or narrower span.

## Training set format

The training set is a JSON array of pixel entries:

```json
[
  {
    "survey": "GRS",
    "pixel": [8, 43],
    "components": [
      {
        "source": "gausspyplus",
        "amplitude": 0.497,
        "mean": 2.986,
        "stddev": 1.422
      },
      {
        "source": "manual",
        "amplitude": 0.31,
        "mean": 185.2,
        "stddev": 2.05
      }
    ]
  }
]
```

Each component records its origin (`gausspyplus`, `phspectra`, or `manual`) so downstream analysis can distinguish curated algorithmic components from hand-fitted ones.

## Using the training set for beta optimisation

Once you have curated enough pixels, run the beta sweep against your training set:

```bash
uv run benchmarks train-beta --training-set packages/train-gui/data/training_set.json
```

This replaces the default GaussPy+ reference with your curated components. The sweep reports $F_1$, precision, and recall at each $\beta$ value, and writes the results to `f1-beta-sweep.{csv,json,png}`.
