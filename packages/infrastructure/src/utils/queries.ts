export const BetaComparisonQuery = `
SELECT
  beta,
  COUNT(*) AS n_spectra,
  AVG(n_components) AS avg_components,
  APPROX_PERCENTILE(n_components, 0.5) AS median_components,
  STDDEV(CAST(n_components AS DOUBLE)) AS std_components
FROM
  phspectra.decompositions
WHERE
  survey = 'grs'
GROUP BY beta
ORDER BY beta;
`

export const RmsDistributionQuery = `
SELECT
  survey,
  COUNT(*) AS n_spectra,
  AVG(rms) AS avg_rms,
  APPROX_PERCENTILE(rms, 0.5) AS median_rms,
  MIN(rms) AS min_rms,
  MAX(rms) AS max_rms
FROM
  phspectra.decompositions
GROUP BY survey
ORDER BY survey;
`
