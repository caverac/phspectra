---
sidebar_position: 1
---

# Installation

<div style={{textAlign: 'center', marginBottom: '2rem'}}>
  <img src="/img/logo.svg" alt="phspectra logo" style={{maxWidth: '360px', width: '100%'}} />
</div>

`phspectra` is a Python library for decomposing 1-D astronomical spectra into Gaussian components using **0-dimensional persistent homology** for peak detection. Instead of derivative-based methods (GaussPy) or brute-force parameter sweeps, it ranks peaks by their topological persistence and uses a single parameter ($\beta$) to threshold noise from real structure.

## Requirements

- Python >= 3.11
- NumPy >= 1.26
- SciPy >= 1.12

## Install from PyPI

```bash
pip install phspectra
```

An optional C extension (`_gaussfit`) is compiled automatically from source when building from a checkout that includes `src/phspectra/_gaussfit.c`. If compilation fails, the library falls back to SciPy's `curve_fit` transparently.

## Publishing

The package is published to [PyPI](https://pypi.org/project/phspectra/) via [python-semantic-release](https://python-semantic-release.readthedocs.io/). On every merge to `main`, the release pipeline:

1. Parses conventional commits scoped to `packages/phspectra/`.
2. Bumps the version in `pyproject.toml` and `__init__.py`.
3. Tags the release as `phspectra-v{version}`.
4. Builds an sdist + wheel with `python -m build`.
5. Publishes to PyPI via trusted publishing (OIDC, no API tokens).
