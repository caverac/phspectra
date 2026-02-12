---
sidebar_position: 3
---

# Performance

## Speed comparison

We benchmark the wall-clock time for decomposing 400 real GRS spectra (424 channels each) using both phspectra and GaussPy+ (Riener et al. 2019). Both algorithms are run on the same spectra with their recommended configurations:

- **phspectra**: $\beta = 4.0$ (default), pure Python
- **GaussPy+**: two-phase decomposition with $\alpha_1 = 2.89$, $\alpha_2 = 6.65$ (trained values from Riener et al. 2019, Sect. 4.1), SNR threshold = 3.0

### Results

| Metric | phspectra | GaussPy+ | Factor |
|---|---|---|---|
| Total time (400 spectra) | 44.0 s | 257.0 s | **5.8&times;** |
| Mean per spectrum | 110 ms | 643 ms | 5.8&times; |
| Mean components detected | 1.9 | 2.3 | &mdash; |

phspectra is **5.8&times; faster** than GaussPy+ on identical real survey data.

![Performance benchmark](/img/results/performance-benchmark.png)

### Why phspectra is faster

The speed advantage comes from algorithmic differences:

1. **No smoothing sweep.** GaussPy+ convolves the spectrum with a family of Gaussian kernels at each $\alpha$ scale, computing derivatives at every scale. The two-phase decomposition repeats this process twice (once per $\alpha$). phspectra skips smoothing entirely &mdash; it operates directly on the raw spectrum using persistence-based peak detection, which is $O(n)$ in the number of channels.

2. **Fewer optimization steps.** GaussPy+ fits all candidate components simultaneously using `scipy.optimize.leastsq`, sometimes iterating multiple times as it adds or removes components. phspectra fits each component in a localized window and performs a single global refinement pass.

3. **No training required.** GaussPy+'s $\alpha$ parameters must be trained per survey (or per survey region), which adds a separate computational cost not reflected in the per-spectrum timing. phspectra's $\beta$ parameter requires no training &mdash; the default value works across surveys.

### Benchmark details

- **Hardware**: single-core sequential processing for both tools (no parallelization)
- **phspectra**: native Python 3.14, run directly
- **GaussPy+**: Python 3.10 in Docker (required for compatibility with legacy numpy/scipy), batch decomposition via `GaussPyDecompose`, per-spectrum timing via GaussPy core decomposer
- **Spectra**: 400 randomly selected GRS pixels with at least one cataloged component, 424 velocity channels each
