WITH table_lengths AS (
  SELECT
    tableid,
    COUNT(DISTINCT rowid) AS table_length
  FROM main_tokenized
  GROUP BY 1
)
SELECT
  COUNT(*) AS table_count,
  MIN(table_length) AS min_length,
  MAX(table_length) AS max_length,
  AVG(table_length) AS avg_length,
  STDDEV_SAMP(table_length) AS std_length,
  APPROXIMATE_MEDIAN(table_length) AS median_length,
  APPROXIMATE_PERCENTILE(table_length USING PARAMETERS percentile = 0.25) AS p25,
  APPROXIMATE_PERCENTILE(table_length USING PARAMETERS percentile = 0.75) AS p75,
  APPROXIMATE_PERCENTILE(table_length USING PARAMETERS percentile = 0.90) AS p90,
  APPROXIMATE_PERCENTILE(table_length USING PARAMETERS percentile = 0.99) AS p99
FROM table_lengths;
