---
sidebar_position: 6
---

# GRS survey

Full-survey decomposition of the [Galactic Ring Survey](../idea-and-plan/data-sources#grs---galactic-ring-survey) using the [AWS pipeline](../infrastructure/aws-pipeline).

## Five-tile strip (tiles 26--34)

Tiles 26, 28, 30, 32, and 34 cover $\ell \approx 25\degree$--$35\degree$ in the inner Galaxy, spanning the Scutum--Centaurus arm where molecular cloud complexes are densely packed along the line of sight.

### Submit

```bash
for l in 26 28 30 32 34; do
  uv run benchmarks pipeline "/tmp/phspectra/grs-full/grs-${l}-cube.fits" --survey "grs-${l}"
done
```

### Visualise

Once all five tiles are processed, generate the multi-tile strip:

```bash
uv run benchmarks grs-map-plot --input-dir /tmp/phspectra/grs-full
```

<figure class="scientific">
  <img src="/img/results/grs-map-plot.png" alt="GRS five-tile decomposition strip" />
  <figcaption>

**Figure.** Four-panel decomposition strip of GRS tiles 26--34 ($\ell \approx 25\degree$--$35\degree$). **(a)** Velocity RGB composite -- three velocity bins mapped to R, G, B from the decomposed Gaussians. **(b)** Topological complexity -- number of Gaussian components detected per pixel; highlights cloud boundaries, outflows, and shock fronts. **(c)** Amplitude--velocity bivariate colormap -- hue encodes centroid velocity, luminance encodes peak amplitude. **(d)** Dominant velocity field -- centroid velocity of the brightest component per pixel, revealing bulk gas motions hidden by moment-1 blending when multiple clouds overlap along the line of sight.

  </figcaption>
</figure>

## Spatial correlation of decomposition fields

The two-point autocorrelation function $\xi(\theta)$ measures how strongly a scalar field at two positions is correlated as a function of their angular separation $\theta$. We compute it across the five-tile strip ($\ell \approx 25\degree$--$35\degree$) using FFT-based estimation on the global pixel grid, with spatial-jackknife error bands (4$\times$4 block grid). Four fields are derived from the Gaussian decomposition:

- **$N_\mathrm{comp}$** -- number of components per pixel (topological complexity).
- **$I_\mathrm{tot}$** -- total integrated intensity, $\sum_i A_i \sigma_i$, proportional to column density.
- **$\bar{v}$** -- intensity-weighted mean velocity (first moment of fitted components).
- **$\sigma_v$** -- intensity-weighted velocity dispersion (second central moment of component centroids).

```bash
uv run benchmarks correlation-plot --input-dir /tmp/phspectra/grs-full
```

<figure class="scientific">
  <img src="/img/results/correlation-plot.png" alt="Two-point autocorrelation of decomposition fields" />
  <figcaption>

**Figure.** Two-point autocorrelation of four decomposition-derived scalar fields across GRS tiles 26--34. Shaded bands show 1$\sigma$ spatial-jackknife uncertainties. The dashed vertical line marks the correlation length $\theta_\mathrm{corr}$ where $\xi$ drops to $1/e$, when it falls within the plotted range.

  </figcaption>
</figure>

The velocity field $\bar{v}$ and dispersion $\sigma_v$ decorrelate faster than the structural fields ($N_\mathrm{comp}$, $I_\mathrm{tot}$), reflecting the smaller coherence scale of gas kinematics compared to the cloud-scale column density structure. The correlation lengths of $\sim 0.3\degree$--$0.6\degree$ for $N_\mathrm{comp}$ and $I_\mathrm{tot}$ are consistent with the angular extent of giant molecular cloud complexes in the inner Galaxy. [Roman-Duval et al. (2010)](https://doi.org/10.1088/0004-637X/723/1/492) measured physical radii of 1--40 pc for 580 molecular clouds identified in the GRS, at kinematic distances of 1--12 kpc ([Roman-Duval et al. 2009](https://doi.org/10.1088/0004-637X/699/2/1153)). At typical GRS distances of 3--8 kpc, clouds of 10--30 pc radius subtend $0.1\degree$--$0.6\degree$, matching the observed correlation scale.

## Velocity spacing distribution

The distribution of velocity separations between adjacent fitted Gaussian components reveals the characteristic velocity scales recovered by the decomposition. For each spectrum with two or more significant components (amplitude $\geq 3\sigma_\mathrm{rms}$), we sort the component centroids by velocity and compute adjacent spacings $\Delta v = v_{i+1} - v_i$.

```bash
uv run benchmarks velocity-spacing-plot --input-dir /tmp/phspectra/grs-full
```

<figure class="scientific">
  <img src="/img/results/velocity-spacing-plot.png" alt="Velocity spacing distribution" style={{width: '50%'}} />
  <figcaption>

**Figure.** Normalised distribution of adjacent velocity spacings $\Delta v$ across GRS tiles 26--34. The grey histogram shows all spectra combined; coloured step histograms split by the number of significant components per spectrum ($N = 2$, $N = 3$--$4$, $N \geq 5$).

  </figcaption>
</figure>

## Full survey (all 22 tiles)

See [Data Sources](../idea-and-plan/data-sources#downloading-the-full-survey) for download instructions. To process every tile, submit each one as a separate pipeline run:

```bash
for l in 15 17 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56; do
  uv run benchmarks pipeline "/tmp/phspectra/grs-full/grs-${l}-cube.fits" --survey "grs-${l}"
done
```

| Property       | Value  |
| -------------- | ------ |
| Tiles          | 22     |
| Total spectra  | ~2.3M  |
| Lambda chunks  | ~5,100 |
| Estimated cost | ~$40   |
