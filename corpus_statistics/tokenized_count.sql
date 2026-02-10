SELECT
  tokenized,
  COUNT(*) AS term_count
FROM main_tokenized /* + PROJS('public.inv_index_proj') */
GROUP BY tokenized
ORDER BY term_count DESC
LIMIT 10000;