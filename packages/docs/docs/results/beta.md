---
sidebar_position: 1
---

# Beta parameter sensitivity

The main tuning parameter in phspectra is $\beta$, the persistence threshold in units of noise $\sigma$. A peak must have topological persistence exceeding $\beta \cdot \sigma_\mathrm{rms}$ to be retained as a candidate component. In practice, $\beta$ controls the trade-off between detecting faint features (low $\beta$) and rejecting noise artifacts (high $\beta$).

## Insensitivity across a wide range

We evaluated $\beta$ on two independent benchmarks:

1. **Synthetic spectra** (350 spectra with known ground-truth components across 7 difficulty categories)
2. **Real GRS spectra** (200 pixels from the Galactic Ring Survey test field, scored against GaussPy+ catalog decompositions)

### Synthetic data: controlled benchmark

The real-data benchmark above compares phspectra against GaussPy+ (another algorithm), not against ground truth. To isolate $\beta$ sensitivity from algorithmic disagreement, we constructed a synthetic benchmark with **known true components**.

**Test design.** We generate 350 spectra across seven categories of increasing difficulty:

| Category | Label | Components | Amplitudes (K) | Widths $\sigma$ (ch) | Constraint |
|---|---|---|---|---|---|
| Single Bright | SB | 1 | 1.0&ndash;5.0 | 3&ndash;10 | SNR > 7 |
| Single Faint | SF | 1 | 0.3&ndash;0.8 | 3&ndash;10 | SNR 2&ndash;6 |
| Single Narrow | SN | 1 | 1.0&ndash;5.0 | 1&ndash;2.5 | Sub-resolution widths |
| Single Broad | SBd | 1 | 0.5&ndash;3.0 | 10&ndash;20 | Extended features |
| Multi Separated | MS | 2&ndash;3 | 0.5&ndash;4.0 | 2&ndash;8 | Separation > $4\sigma$ |
| Multi Blended | MB | 2&ndash;3 | 0.5&ndash;4.0 | 3&ndash;8 | Separation $1.5$&ndash;$3\sigma$ |
| Crowded | C | 4&ndash;5 | 0.3&ndash;3.0 | 2&ndash;6 | Mixed separations |

All spectra use GRS-realistic parameters: 424 channels with additive Gaussian noise at $\sigma = 0.13$ K. Because the true components are known exactly, $F_1$ measures *true accuracy* rather than agreement with another algorithm.

For each spectrum we sweep $\beta$ from 3.8 to 4.5, decompose with phspectra, and score using Hungarian matching with the [Lindner et al. (2015)](https://arxiv.org/abs/1409.2840) criteria.

**Results.** The figure below shows $F_1$ vs $\beta$ for each category and overall:

![Synthetic $F_1$ vs beta](/img/results/synthetic-f1.png)

The key observations:

1. **$F_1$ varies by only 0.005** across the full $\beta$ sweep (0.916 at $\beta=3.8$ to 0.911 at $\beta=4.5$). This confirms that $\beta$ sensitivity is negligible on ground-truth data.

2. **The difficulty gradient follows expectations.** Multi-component separated spectra score highest ($F_1$ = 0.965), while blended multi-component spectra are the hardest ($F_1$ = 0.849). This validates that the benchmark categories genuinely span the difficulty spectrum.

3. **Parameter recovery is accurate.** The error plots below show amplitude relative error, position error (in channels), and width relative error across the $\beta$ sweep:

![Synthetic errors](/img/results/synthetic-errors.png)

Position errors are sub-channel for most categories, and amplitude and width relative errors are small and stable across the tested $\beta$ range.

The per-category $F_1$ at the optimal $\beta$:

| Category | Label | $F_1$ |
|---|---|---|
| Multi Separated | MS | 0.965 |
| Single Broad | SBd | 0.935 |
| Crowded | C | 0.933 |
| Single Bright | SB | 0.926 |
| Single Narrow | SN | 0.909 |
| Single Faint | SF | 0.863 |
| Multi Blended | MB | 0.849 |

### Real data: beta training

Sweeping $\beta$ from 3.8 to 4.5 on 200 GRS spectra shows a nearly flat $F_1$ curve:

![Beta training on GRS spectra](/img/results/f1-beta-sweep.png)

The optimal $\beta = 4.0$ achieves $F_1$ = 0.847 against GaussPy+ reference decompositions. However, the variation across the entire sweep is only $\Delta F_1 \approx 0.01$ &mdash; the algorithm is remarkably stable.

Note that the $F_1$ ceiling here does not reflect a limitation of phspectra &mdash; it reflects *disagreement between two different decomposition strategies*. See the [Accuracy](accuracy) section for a detailed analysis of where and why the decompositions differ.



## Why this matters

GaussPy requires a trained smoothing parameter $\alpha$ that is sensitive to the noise properties and spectral structure of each survey. The training procedure ([Lindner et al. 2015](https://arxiv.org/abs/1409.2840)) requires labeled decompositions and can produce different optimal values for different regions of the same survey.

In contrast, phspectra's $\beta$ parameter is:

- **Survey-agnostic**: values in the range $\beta = 3.8$&ndash;$4.0$ work well across both real and synthetic data with fundamentally different noise structures.
- **Robust to perturbation**: performance degrades gracefully rather than collapsing at non-optimal values. There is no cliff &mdash; the $F_1$ curve is flat.
- **Physically interpretable**: $\beta$ directly controls the minimum significance (in $\sigma$) for a peak to be considered real. A value of $\beta = 4.0$ means "reject anything less significant than a $4\sigma$ fluctuation," which is a natural and intuitive threshold.

The default value of $\beta = 4.0$ is recommended for general use.
