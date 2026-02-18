<p align="center">
  <img src="https://caverac.github.io/phspectra/img/logo.svg" width="128" alt="phspectra logo">
</p>

<h1 align="center">phspectra</h1>

<p align="center">
  Persistent homology spectral line decomposition.
</p>

<p align="center">
  <a href="https://caverac.github.io/phspectra/">Documentation</a>
</p>

---

A Python library for decomposing 1-D astronomical spectra into Gaussian components using **0-dimensional persistent homology** for peak detection. Instead of derivative-based methods (GaussPy) or brute-force parameter sweeps, it ranks peaks by their topological persistence and uses a single tuning parameter.

## Installation

```bash
pip install phspectra
```

**Requirements:** Python >= 3.11, NumPy >= 1.26, SciPy >= 1.12.

An optional C extension is compiled automatically from source when available, providing ~2x faster fitting. If compilation fails, the library falls back to SciPy's `curve_fit` transparently.

## Quick start

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

```
  amplitude=3.00  mean=60.1  stddev=3.97
  amplitude=1.48  mean=129.9  stddev=8.12
```

The number of components is determined automatically -- no need to specify it in advance.

## API

The public API consists of three functions:

| Symbol                                                      | Description                                       |
| ----------------------------------------------------------- | ------------------------------------------------- |
| `fit_gaussians(signal, *, beta=3.5, ...)`                   | Decompose a 1-D spectrum into Gaussian components |
| `find_peaks_by_persistence(signal, *, min_persistence=0.0)` | Low-level peak detection via persistent homology  |
| `estimate_rms(signal, *, mask_pad=2, mad_clip=5.0)`         | Signal-masked noise estimation                    |

`fit_gaussians` returns a list of `GaussianComponent` dataclasses, each with `amplitude`, `mean`, and `stddev` fields.

See the [full API reference](https://caverac.github.io/phspectra/library/api) for parameter details, types, and error handling.

## How it works

1. **Noise estimation** -- signal-masked MAD estimator (Riener et al. 2019)
2. **Peak detection** -- 0-dim persistent homology ranks peaks by significance
3. **Curve fitting** -- bounded Levenberg-Marquardt with initial guesses from persistence
4. **Validation** -- SNR, matched-filter SNR, and FWHM checks discard unphysical components
5. **Refinement** -- iterative residual search, negative-dip splitting, and blended-pair merging, accepted only when AICc improves

See the [algorithm overview](https://caverac.github.io/phspectra/library/algorithm) for details.

## License

MIT
