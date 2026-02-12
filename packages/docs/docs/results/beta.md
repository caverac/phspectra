---
sidebar_position: 1
---

# Beta parameter sensitivity

The only free parameter in phspectra is $\beta$, the persistence threshold in units of noise $\sigma$. A peak must have topological persistence exceeding $\beta \cdot \sigma_\mathrm{rms}$ to be retained as a candidate component. In practice, $\beta$ controls the trade-off between detecting faint features (low $\beta$) and rejecting noise artifacts (high $\beta$).

## Insensitivity across a wide range

We evaluated $\beta$ on two independent benchmarks:

1. **Real GRS spectra** (200 pixels from the Galactic Ring Survey test field, scored against GaussPy+ catalog decompositions)
2. **Synthetic spectra** (350 spectra with known ground-truth components across 7 difficulty categories)

### Real data: beta training

Sweeping $\beta$ from 3.8 to 4.5 on 200 GRS spectra shows a nearly flat F1 curve:

![Beta training on GRS spectra](/img/results/train-beta.png)

The optimal $\beta = 4.0$ achieves F1 = 0.847 against GaussPy+ reference decompositions. However, the variation across the entire sweep is only $\Delta$F1 $\approx$ 0.01 &mdash; the algorithm is remarkably stable.

Note that the F1 ceiling here does not reflect a limitation of phspectra &mdash; it reflects *disagreement between two different decomposition strategies*. See the [Accuracy](accuracy) section for a detailed analysis of where and why the decompositions differ.

### Synthetic data: controlled benchmark

The real-data benchmark above compares phspectra against GaussPy+ (another algorithm), not against ground truth. To isolate $\beta$ sensitivity from algorithmic disagreement, we constructed a synthetic benchmark with **known true components**.

**Test design.** We generate 350 spectra across seven categories of increasing difficulty:

| Category | Components | Amplitudes (K) | Widths $\sigma$ (ch) | Constraint |
|---|---|---|---|---|
| `single_bright` | 1 | 1.0&ndash;5.0 | 3&ndash;10 | SNR > 7 |
| `single_faint` | 1 | 0.3&ndash;0.8 | 3&ndash;10 | SNR 2&ndash;6 |
| `single_narrow` | 1 | 1.0&ndash;5.0 | 1&ndash;2.5 | Sub-resolution widths |
| `single_broad` | 1 | 0.5&ndash;3.0 | 10&ndash;20 | Extended features |
| `multi_separated` | 2&ndash;3 | 0.5&ndash;4.0 | 2&ndash;8 | Separation > $4\sigma$ |
| `multi_blended` | 2&ndash;3 | 0.5&ndash;4.0 | 3&ndash;8 | Separation $1.5$&ndash;$3\sigma$ |
| `crowded` | 4&ndash;5 | 0.3&ndash;3.0 | 2&ndash;6 | Mixed separations |

All spectra use GRS-realistic parameters: 424 channels with additive Gaussian noise at $\sigma = 0.13$ K. Because the true components are known exactly, F1 measures *true accuracy* rather than agreement with another algorithm.

For each spectrum we sweep $\beta$ from 3.8 to 4.5, decompose with phspectra, and score using Hungarian matching with the Lindner et al. (2015) criteria.

**Results.** The top-left panel below shows F1 vs $\beta$ for each category and overall:

![Synthetic benchmark](/img/results/synthetic-benchmark.png)

The key observations:

1. **F1 varies by only 0.003** across the full $\beta$ sweep (0.947 at $\beta=3.8$ to 0.944 at $\beta=4.5$). This confirms that $\beta$ sensitivity is negligible on ground-truth data.

2. **The difficulty gradient is exactly as expected.** Isolated bright components are solved perfectly (F1 = 1.0), while blended multi-component spectra are the hardest (F1 = 0.819). This validates that the benchmark categories genuinely span the difficulty spectrum.

3. **Precision and recall are both high** (top-center panel). The algorithm neither hallucinates spurious components nor misses real ones across the tested range.

4. **Component count error is near zero** (top-right panel) for all single-component categories. The `multi_blended` and `crowded` categories show slight undercounting at higher $\beta$, consistent with the algorithm conservatively merging ambiguous overlapping features.

5. **Parameter recovery is accurate** (bottom row). At the optimal $\beta$, position errors are sub-channel for bright isolated components, and amplitude and width relative errors are small across all categories.

The per-category F1 at the optimal $\beta$:

| Category | F1 |
|---|---|
| Single bright | 1.000 |
| Single narrow | 1.000 |
| Single broad | 0.990 |
| Multi-component (separated) | 0.996 |
| Single faint | 0.980 |
| Crowded (4&ndash;5 components) | 0.945 |
| Multi-component (blended) | 0.819 |

## Why this matters

GaussPy requires a trained smoothing parameter $\alpha$ that is sensitive to the noise properties and spectral structure of each survey. The training procedure (Lindner et al. 2015) requires labeled decompositions and can produce different optimal values for different regions of the same survey.

In contrast, phspectra's $\beta$ parameter is:

- **Survey-agnostic**: the same value ($\beta = 4.0$) works well across both real and synthetic data with fundamentally different noise structures.
- **Robust to perturbation**: performance degrades gracefully rather than collapsing at non-optimal values. There is no cliff &mdash; the F1 curve is flat.
- **Physically interpretable**: $\beta$ directly controls the minimum significance (in $\sigma$) for a peak to be considered real. A value of $\beta = 4.0$ means "reject anything less significant than a $4\sigma$ fluctuation," which is a natural and intuitive threshold.

The default value of $\beta = 4.0$ is recommended for general use.
