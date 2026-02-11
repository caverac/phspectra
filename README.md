# PHSpectra

**Persistent homology for spectral line decomposition.**

> **[Read the full documentation](https://caverac.github.io/phspectra/)**

Radio-astronomical spectra — particularly HI 21-cm and ${}^{13}\mathrm{CO}$ emission — are typically a superposition of multiple Gaussian components. Recovering these individual components is essential for understanding the interstellar medium.

This project uses **0-dimensional persistent homology** to detect and rank peaks by their topological persistence, providing a natural measure of significance without a training step. The only free parameter is $\beta$ — a persistence threshold in units of noise ($\beta \times \sigma_{\mathrm{rms}}$). Detected peaks seed a sum-of-Gaussians fit, yielding a full decomposition of the spectrum.

## Table of Contents

- [Packages](#packages)
- [Getting Started](#getting-started)
- [License](#license)

## Packages

This is a monorepo with three packages:

| Package                                     | Description                                                                 |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| [`phspectra`](packages/phspectra)           | Core Python library — persistence-based peak detection and Gaussian fitting |
| [`docs`](packages/docs)                     | Project documentation ([live site](https://caverac.github.io/phspectra/))   |
| [`infrastructure`](packages/infrastructure) | AWS CDK stack for large-scale processing                                    |

See the [documentation site](https://caverac.github.io/phspectra/) for motivation, algorithm details, and the project roadmap.

## Getting Started

This project uses [mise](https://mise.jdx.dev/) to manage tool versions (Node 22, Python 3.11, uv).

```bash
mise install
yarn install
yarn dev:docs
```

## License

MIT
