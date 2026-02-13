---
sidebar_position: 2
---

# Accuracy

## True accuracy on synthetic data

When ground truth is known exactly (synthetic spectra with prescribed Gaussian components), phspectra achieves an overall **$F_1$ = 0.916**. The only challenging regime is heavily blended multi-component spectra ($F_1$ = 0.849), where any algorithm faces fundamental ambiguity. See the [Beta parameter sensitivity](beta) section for the full breakdown.

## Comparison with GaussPy+

We run both phspectra and GaussPy+ on 1000 randomly selected GRS spectra. GaussPy+ is run in Docker using `GaussPyDecompose` with the trained parameters from [Riener et al. (2019)](https://arxiv.org/abs/1906.10506): $\alpha_1 = 2.89$, $\alpha_2 = 6.65$, two-phase decomposition, SNR threshold = 3.0.

### Fit quality (RMS)

| Metric | phspectra | GaussPy+ |
|---|---|---|
| Mean RMS (K) | 0.1303 | 0.1300 |
| Lower RMS wins | **683 / 1000** (68%) | 317 / 1000 (32%) |

phspectra achieves lower residual RMS on the majority of spectra (68%), despite GaussPy+ having a slightly lower mean RMS. The difference in means is driven by a few spectra where GaussPy+ fits many more components, reducing RMS at the cost of potential overfitting.

![RMS distribution](/img/results/rms-distribution.png)

The RMS distributions overlap heavily &mdash; both tools fit most spectra near the noise floor ($\sigma = 0.13$ K).

![RMS scatter](/img/results/rms-scatter.png)

The scatter plot shows phspectra wins the majority of head-to-head comparisons: points below the 1:1 line indicate lower phspectra RMS.

### Where decompositions differ

A systematic comparison reveals several recurring patterns of disagreement:

![Disagreement cases](/img/results/compare-disagreements.png)

The six panels show representative cases:

- **PH fewer components**: GaussPy+ sometimes fits many components (up to 14) where phspectra finds fewer, better-constrained ones
- **PH more components**: phspectra resolves blended features that GaussPy+ misses entirely
- **PH lower / GP+ lower RMS**: each tool wins on different spectra, with different decomposition strategies
- **Same N, different positions**: even with the same component count, the two algorithms place components differently
- **Different widths**: the two algorithms sometimes assign different widths to the same feature

### Component widths

A population-level comparison of fitted widths shows **no systematic bias** between the two tools. Matching 644 component pairs across 1000 spectra (Hungarian algorithm, position tolerance $< 2\sigma$), the median log-width ratio $\ln(\sigma_{\text{phspectra}} / \sigma_{\text{GaussPy+}})$ is near zero, and the split is near even: GaussPy+ fits wider profiles in 55% of pairs, phspectra in 45%.

![Width comparison](/img/results/width-comparison.png)

The histogram of $\ln(\sigma_{\text{phspectra}} / \sigma_{\text{GaussPy+}})$ is sharply peaked at zero, confirming that neither tool systematically favours wider or narrower profiles. While individual spectra can show large width differences (the disagreement panel above includes such cases), these are isolated instances driven by different decomposition strategies, not a systematic effect.
