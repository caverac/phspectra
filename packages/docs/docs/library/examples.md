---
sidebar_position: 4
---

# Examples

## Basic decomposition

```python
import numpy as np
from phspectra import fit_gaussians

# Create a synthetic spectrum: two Gaussians + noise
rng = np.random.default_rng(42)
x = np.arange(200, dtype=np.float64)
signal = (
    3.0 * np.exp(-0.5 * ((x - 60) / 4.0) ** 2)
    + 1.5 * np.exp(-0.5 * ((x - 130) / 8.0) ** 2)
    + rng.normal(0, 0.2, size=200)
)

# Decompose
components = fit_gaussians(signal, beta=3.5)

for c in components:
    print(f"  amplitude={c.amplitude:.2f}  mean={c.mean:.1f}  stddev={c.stddev:.2f}")
```

**Output**

```
  amplitude=3.00  mean=60.1  stddev=3.97
  amplitude=1.48  mean=129.9  stddev=8.12
```

The library recovers both components with accurate amplitude, position, and width estimates. The number of components is determined automatically -- no need to specify it in advance.

## Reconstructing the model

Each `GaussianComponent` defines a Gaussian $G(x) = A \exp(-(x - \mu)^2 / (2 \sigma^2))$. To reconstruct the full model:

```python
x = np.arange(len(signal), dtype=np.float64)
model = np.zeros_like(x)
for c in components:
    model += c.amplitude * np.exp(-0.5 * ((x - c.mean) / c.stddev) ** 2)

residual = signal - model
```

## Accessing peak persistence directly

For diagnostic purposes, you can inspect the raw persistence diagram before fitting:

```python
from phspectra import find_peaks_by_persistence, estimate_rms

rms = estimate_rms(signal)
peaks = find_peaks_by_persistence(signal, min_persistence=3.5 * rms)

for pk in peaks:
    print(f"  channel={pk.index}  persistence={pk.persistence:.3f}  birth={pk.birth:.3f}  death={pk.death:.3f}")
```

```
  channel=59  persistence=3.101  birth=3.101  death=0.000
  channel=126  persistence=1.954  birth=1.616  death=-0.337
  channel=162  persistence=0.727  birth=0.426  death=-0.301
  channel=190  persistence=0.618  birth=0.345  death=-0.274
```
