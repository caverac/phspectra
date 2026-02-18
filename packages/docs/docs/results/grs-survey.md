---
sidebar_position: 6
---

# GRS survey

Full-survey decomposition of the [Galactic Ring Survey](../idea-and-plan/data-sources#grs----galactic-ring-survey) using the [AWS pipeline](../infrastructure/aws-pipeline).

## Tile 56 (contains the test field)

The GRS test field ($\ell = [55.27, 55.69]\degree$, $b = [0.01, 0.37]\degree$) is a sub-region of tile 56. Running the full tile is a good first step before processing the entire survey.

| Property       | Value              |
| -------------- | ------------------ |
| File           | `grs-56-cube.fits` |
| Spatial pixels | 116 $\times$ 357   |
| Spectra        | 41,412             |
| Channels       | 424                |
| Lambda chunks  | 83                 |
| Estimated cost | ~$0.02             |
| Wall-clock     | ~1--2 minutes      |

### Submit

```bash
uv run benchmarks pipeline /tmp/phspectra/grs-full/grs-56-cube.fits --survey grs-56
```

### Visualise

Once the pipeline completes, generate the survey map plot:

```bash
uv run benchmarks survey-map-plot --survey grs-56 --fits-file /tmp/phspectra/grs-full/grs-56-cube.fits
```

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
| Lambda chunks  | ~4,600 |
| Estimated cost | ~$1    |
