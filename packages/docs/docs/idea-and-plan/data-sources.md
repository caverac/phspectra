---
sidebar_position: 4
---

# Data Sources

Public spectral-line surveys we can use to develop and benchmark PHSpectra.
The primary reference dataset comes from the GaussPy+ paper ([Riener et al. 2019](https://arxiv.org/abs/1906.10506)),
which provides both raw data and a full Gaussian decomposition catalog for comparison.

## GRS --Galactic Ring Survey

The [Galactic Ring Survey](https://www.bu.edu/galacticring/) mapped
${}^{13}\mathrm{CO}$ ($J = 1 \to 0$) emission at 110.2 GHz across the inner Milky Way.

| Property              | Value                                                                   |
| --------------------- | ----------------------------------------------------------------------- |
| Tracer                | ${}^{13}\mathrm{CO}$ ($J = 1 \to 0$), 110.201 GHz                       |
| Coverage              | $18\degree \leq \ell \leq 55.7\degree$, $\lvert b \rvert \leq 1\degree$ |
| Angular resolution    | 46"                                                                     |
| Velocity range        | $-5$ to 135 km/s (VLSR)                                                 |
| Channel width         | 0.21 km/s                                                               |
| Channels per spectrum | ~670                                                                    |
| RMS noise             | ~0.13 K ($T_A^*$)                                                       |
| Reference             | [Jackson et al. 2006](https://doi.org/10.1086/508258)                   |

**Public access:** Full FITS cubes are available from the
[BU GRS archive](https://www.bu.edu/galacticring/new_data.html).

### Downloading the full survey

The survey is distributed as 22 FITS tiles (2&deg; &times; 2&deg; each, ~5.5 GB total) from the BU archive. Each tile is named `grs-{LL}-cube.fits` where `LL` is the central galactic longitude.

```bash
# Download all 22 tiles into a local directory
mkdir -p /tmp/phspectra/grs-full
for l in 15 17 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56; do
  curl -ksL -o "/tmp/phspectra/grs-full/grs-${l}-cube.fits" \
    "https://grunt.bu.edu/grs-stitch/source/grs-${l}-cube.fits"
done
```

| Tile range           | Tiles                   | Velocity range   | Size per tile |
| -------------------- | ----------------------- | ---------------- | ------------- |
| $\ell = 15$--40&deg; | 14 tiles (~290 MB each) | $-5$ to 135 km/s | ~290 MB       |
| $\ell = 42$--54&deg; | 7 tiles (~190 MB each)  | $-5$ to 85 km/s  | ~190 MB       |
| $\ell = 56$&deg;     | 1 tile                  | $-5$ to 85 km/s  | ~67 MB        |

## GaussPy+ test field

The GaussPy+ repository bundles a small FITS cube covering a test field from the GRS.
This is the easiest way to get started -- no large downloads required.

**Direct download:**

```
https://github.com/mriener/gausspyplus/raw/master/gausspyplus/data/grs-test_field.fits
```

The cube is ~4 MB and covers a $60 \times 70$ pixel spatial region (4,200 spectra) with 424 velocity channels.
The GaussPy+ tutorial uses pixel (y=31, x=40) as a representative spectrum.

| Property    | Value                                                                          |
| ----------- | ------------------------------------------------------------------------------ |
| File        | `grs-test_field.fits`                                                          |
| Dimensions  | 424 $\times$ 60 $\times$ 70 (vel $\times$ lat $\times$ lon)                    |
| Size        | ~4 MB                                                                          |
| Coverage    | $\ell = [55.27, 55.69]\degree$, $b = [0.01, 0.37]\degree$                      |
| Parent tile | `grs-56-cube.fits` ($\ell = [55.00, 55.71]\degree$) -- a sub-region of tile 56 |

**Reference:** [Riener et al. 2019, A&A 628, A78](https://doi.org/10.1051/0004-6361/201935519)
([arXiv:1906.10506](https://arxiv.org/abs/1906.10506))

## GaussPy+ full decomposition catalog

[Riener et al. (2020)](https://doi.org/10.1051/0004-6361/201936814) published a complete Gaussian decomposition of the GRS using GaussPy+.
This catalog provides ground-truth component parameters (amplitude, velocity, FWHM) for
every spectrum in the survey -- ideal for benchmarking.

| Property  | Value                                                                           |
| --------- | ------------------------------------------------------------------------------- |
| Catalog   | GRS GaussPy+ decomposition                                                      |
| Format    | FITS tables via CDS/VizieR                                                      |
| Reference | [Riener et al. 2020, A&A 633, A14](https://doi.org/10.1051/0004-6361/201936814) |

**Access:** [CDS VizieR catalog J/A+A/633/A14](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/633/A14)

## Downloading with the CLI

The benchmarks CLI downloads both the GRS test field FITS cube and the VizieR decomposition catalog in a single command:

```bash
uv run benchmarks download
```

Files are cached in `/tmp/phspectra/` by default (configurable with `--cache-dir`). Use `--force` to re-download. See [Reproducing results](../results/reproducing) for the full benchmark workflow.

## Other public surveys

These surveys use different tracers and resolutions. They are useful for testing
generalization beyond ${}^{13}\mathrm{CO}$:

- **GALFA-HI** -- HI 21-cm emission from Arecibo. High spectral resolution (0.18 km/s),
  covers the full Arecibo sky.
  [Peek et al. 2018](https://doi.org/10.3847/1538-4365/aac889)

- **THOR** -- The HI/OH/Recombination line survey of the inner Milky Way (VLA).
  [Beuther et al. 2016](https://doi.org/10.1051/0004-6361/201527108)

- **SEDIGISM** -- ${}^{13}\mathrm{CO}$/$\mathrm{C}{}^{18}\mathrm{O}$ ($J = 2 \to 1$) from the APEX telescope, covering $\ell = -60\degree$ to $+18\degree$.
  [Schuller et al. 2021](https://doi.org/10.1093/mnras/stab1227)

## References

1. Jackson, J. M. et al. 2006, ApJS 163, 145 --[DOI](https://doi.org/10.1086/508258)
2. Riener, M. et al. 2019, A&A 628, A78 --[DOI](https://doi.org/10.1051/0004-6361/201935519) / [arXiv:1906.10506](https://arxiv.org/abs/1906.10506)
3. Riener, M. et al. 2020, A&A 633, A14 --[DOI](https://doi.org/10.1051/0004-6361/201936814)
4. Peek, J. E. G. et al. 2018, ApJS 234, 2 --[DOI](https://doi.org/10.3847/1538-4365/aac889)
5. Beuther, H. et al. 2016, A&A 595, A32 --[DOI](https://doi.org/10.1051/0004-6361/201527108)
6. Schuller, F. et al. 2021, MNRAS 500, 3064 --[DOI](https://doi.org/10.1093/mnras/stab1227)
