---
sidebar_position: 2
---

# Accuracy

## True accuracy on synthetic data

When ground truth is known exactly (synthetic spectra with prescribed Gaussian components), phspectra achieves an overall **F1 = 0.947**, with perfect scores on isolated bright and narrow components. The only challenging regime is heavily blended multi-component spectra (F1 = 0.819), where any algorithm faces fundamental ambiguity.

## Comparison with GaussPy+

We run both phspectra and GaussPy+ on 400 randomly selected GRS spectra. GaussPy+ is run in Docker using `GaussPyDecompose` with the trained parameters from Riener et al. (2019): $\alpha_1 = 2.89$, $\alpha_2 = 6.65$, two-phase decomposition, SNR threshold = 3.0.

### Fit quality (RMS)

| Metric | phspectra | GaussPy+ |
|---|---|---|
| Mean RMS (K) | 0.1299 | 0.1273 |
| Lower RMS wins | **235 / 400** (59%) | 165 / 400 (41%) |

phspectra achieves lower residual RMS on the majority of spectra (59%), despite GaussPy+ having a slightly lower mean RMS. The difference in means is driven by a few spectra where GaussPy+ fits many more components (up to 14), reducing RMS at the cost of potential overfitting.

![RMS comparison](/img/results/compare-rms-docker.png)

The left panel shows the RMS distributions overlap heavily &mdash; both tools fit most spectra near the noise floor ($\sigma = 0.13$ K). The center scatter plot shows phspectra wins the majority of head-to-head comparisons. The right panel shows the RMS difference distribution is shifted positive (GaussPy+ RMS minus phspectra RMS), confirming phspectra's advantage on most spectra.

### Where decompositions differ

A systematic comparison reveals several recurring patterns of disagreement:

![Disagreement cases](/img/results/compare-disagreements-docker.png)

The six panels show representative cases:

- **PH fewer components**: GaussPy+ sometimes fits many components (up to 14) where phspectra finds fewer, better-constrained ones
- **PH more components**: phspectra resolves blended features that GaussPy+ misses entirely
- **PH lower / GP+ lower RMS**: each tool wins on different spectra, with different decomposition strategies
- **Same N, different positions**: even with the same component count, the two algorithms place components differently
- **Different widths**: the two algorithms sometimes assign different widths to the same feature

### Component widths

A population-level comparison of fitted widths shows **no systematic bias** between the two tools. Matching 644 component pairs across 400 spectra (Hungarian algorithm, position tolerance $< 2\sigma$), the median width ratio (GaussPy+ / phspectra) is **1.00** and the split is near even: GaussPy+ fits wider profiles in 55% of pairs, phspectra in 45%.

![Width comparison](/img/results/width-comparison.png)

The scatter plot (left) shows matched widths cluster tightly around the 1:1 line. The histogram (center) confirms the ratio distribution is sharply peaked at 1.0, with a tail of outliers pulling the mean to 1.25. The right panel shows the width difference is uncorrelated with amplitude &mdash; neither tool is biased toward wider or narrower profiles in any signal regime.

While individual spectra can show large width differences (the disagreement panel above includes such cases), these are isolated instances driven by different decomposition strategies, not a systematic effect.
