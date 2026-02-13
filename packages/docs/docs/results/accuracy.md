---
sidebar_position: 2
---

# Accuracy

Plots in this section can be reproduced using, both execute in under a couple of seconds

```python
uv run benchmarks compare-plot
uv run benchmarks ncomp-rms-plot
```


## True accuracy on synthetic data

When ground truth is known exactly (synthetic spectra with prescribed Gaussian components), PHSpectra achieves an overall **$F_1$ = 0.941**. The only challenging regime is heavily blended multi-component spectra ($F_1$ = 0.862), where any algorithm faces fundamental ambiguity. See the [Beta parameter sensitivity](beta) section for the full breakdown.

## Comparison with GaussPy+

We run both PHSpectra and GaussPy+ on 1001 randomly selected GRS spectra. GaussPy+ is run in Docker using `GaussPyDecompose` with the trained parameters from [Riener et al. (2019)](https://arxiv.org/abs/1906.10506): $\alpha_1 = 2.89$, $\alpha_2 = 6.65$, two-phase decomposition, SNR threshold = 3.0.

### Fit quality (RMS)

| Metric | PHSpectra | GaussPy+ |
|---|---|---|
| Mean RMS (K) | 0.1312 | 0.1300 |
| Lower RMS wins | **633 / 1001** (63%) | 368 / 1001 (37%) |

PHSpectra achieves lower residual RMS on the majority of spectra (63%), despite GaussPy+ having a slightly lower mean RMS. The difference in means is driven by a few spectra where GaussPy+ fits many more components, reducing RMS at the cost of potential overfitting.

![RMS distribution](/img/results/rms-distribution.png)

The RMS distributions overlap heavily &mdash; both tools fit most spectra near the noise floor ($\sigma = 0.13$ K).

![RMS scatter](/img/results/rms-scatter.png)

The scatter plot shows PHSpectra wins the majority of head-to-head comparisons (633/1001): points below the 1:1 line indicate lower PHSpectra RMS.

### Where decompositions differ

A systematic comparison reveals several recurring patterns of disagreement:

![Disagreement cases](/img/results/compare-disagreements.png)

The six panels show representative cases:

- **PHS fewer components**: GaussPy+ sometimes fits many components (up to 14) where PHSpectra finds fewer, better-constrained ones
- **PHS more components**: PHSpectra resolves blended features that GaussPy+ misses entirely
- **PHS lower / GP+ lower RMS**: each tool wins on different spectra, with different decomposition strategies
- **Same N, different positions**: even with the same component count, the two algorithms place components differently
- **Different widths**: the two algorithms sometimes assign different widths to the same feature

### Component count vs RMS

The scatter plot below shows the number of fitted components against residual RMS for both methods. A clear pattern emerges above RMS $\approx 0.2$ K: GaussPy+ fits systematically more components to noisy spectra than PHSpectra does.

![N components vs RMS](/img/results/ncomp-vs-rms.png)

This is a potential overfitting problem. When a spectrum is noisy or has weak, ambiguous features, GaussPy+'s derivative-based detection can interpret noise fluctuations as real peaks, fitting many narrow components to chase down the residual. The result is a lower RMS &mdash; but at the cost of introducing spurious components that have no physical basis. This explains why GaussPy+ has a slightly lower *mean* RMS despite losing the majority of head-to-head comparisons: its mean is pulled down by a tail of high-noise spectra where it fits 10&ndash;15 components to what PHSpectra correctly identifies as noise.

PHSpectra's persistence threshold imposes a hard significance floor: no candidate peak survives unless its topological prominence exceeds $\beta \times \sigma_\mathrm{rms}$. On noisy spectra this means fewer (or zero) components are fitted, producing a higher RMS but a more physically defensible decomposition. In the low-RMS regime (RMS $\leq 0.2$ K), where both methods agree that real signal is present, the mean component counts are comparable.

### Component widths

A population-level comparison of fitted widths shows **no systematic bias** between the two tools. Matching 1722 component pairs across 1001 spectra (Hungarian algorithm, position tolerance $< 2\sigma$), the median log-width ratio $\ln(\sigma_{\text{PHSpectra}} / \sigma_{\text{GaussPy+}})$ is near zero, and the split is near even: GaussPy+ fits wider profiles in 54% of pairs, PHSpectra in 46%.

![Width comparison](/img/results/width-comparison.png)

The histogram of $\ln(\sigma_{\text{PHSpectra}} / \sigma_{\text{GaussPy+}})$ is sharply peaked at zero, confirming that neither tool systematically favours wider or narrower profiles. While individual spectra can show large width differences (the disagreement panel above includes such cases), these are isolated instances driven by different decomposition strategies, not a systematic effect.
