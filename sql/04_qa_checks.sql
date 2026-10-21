-- 1. Uniqueness & Duplicate Check
-- Criteria: agg_rows = agg_distinct AND nppes_rows = nppes_distinct
SELECT
    (SELECT COUNT(*) FROM mart_hcp_aggregated) AS agg_rows,
    (SELECT COUNT(DISTINCT hcp_id) FROM mart_hcp_aggregated) AS agg_distinct,
    (SELECT COUNT(*) FROM raw_nppes) AS nppes_rows,
    (SELECT COUNT(DISTINCT NPI) FROM raw_nppes) AS nppes_distinct;

-- 2. Valid ZIP Match Rate
-- Criteria: nppes_valid_zip_rate >= 98%
SELECT
    SUM(CASE WHEN n.zip_raw IS NOT NULL AND regexp_matches(TRIM(n.zip_raw), '^[0-9]{5}') THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS nppes_valid_zip_rate
FROM mart_hcp_aggregated d LEFT JOIN raw_nppes n ON d.hcp_id = n.NPI;

-- 3. ZCTA Match Rate (Primary Targets only)
-- Criteria: zcta_match_rate >= 95%
SELECT
    SUM(CASE WHEN z.zip_code IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS zcta_match_rate
FROM mart_hcp_deciled d LEFT JOIN zip_lat_lon z ON d.zip_code = z.zip_code
WHERE d.target_status = 'Primary Target';

-- 4. Staging Drug Audit
-- Criteria: Verify every intended statin name returned rows
SELECT generic_drug_name, COUNT(*)
FROM stg_rx_base GROUP BY 1 ORDER BY 2 DESC;

-- 5. Decile Sanity Check
-- Criteria: Each decile should represent roughly 10% of the volume within the specialty
SELECT specialty, decile, SUM(total_fills) * 100.0 / SUM(SUM(total_fills)) OVER(PARTITION BY specialty) AS pct_vol
FROM mart_hcp_deciled
GROUP BY 1, 2 ORDER BY 1, 2;

-- 6. Unassigned Targets
-- Criteria: dropped_pct < 5%
SELECT
    COUNT(*) AS unassigned_primary_targets,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {roster_table} WHERE target_status = 'Primary Target') AS dropped_pct,
    SUM(CASE decile WHEN 10 THEN 24 WHEN 9 THEN 18 WHEN 8 THEN 12 ELSE 0 END) AS dropped_visits
FROM {roster_table}
WHERE target_status = 'Primary Target' AND territory_id IS NULL;

-- 6b. Primary Target Count Integrity Check (must equal mart count)
SELECT
    (SELECT COUNT(*) FROM {roster_table} WHERE target_status = 'Primary Target') AS roster_primary_count,
    (SELECT COUNT(*) FROM mart_hcp_deciled WHERE target_status = 'Primary Target') AS mart_primary_count,
    CASE WHEN (SELECT COUNT(*) FROM {roster_table} WHERE target_status = 'Primary Target') = 
              (SELECT COUNT(*) FROM mart_hcp_deciled WHERE target_status = 'Primary Target') 
         THEN 'PASS' ELSE 'FAIL_COUNT_MISMATCH' END AS status_6b;

-- 6c. State Representation Check (all mart states represented)
SELECT
    (SELECT COUNT(DISTINCT state) FROM {roster_table} WHERE state IS NOT NULL) AS roster_states_count,
    (SELECT COUNT(DISTINCT state) FROM mart_hcp_deciled WHERE state IS NOT NULL) AS mart_states_count,
    CASE WHEN (SELECT COUNT(DISTINCT state) FROM {roster_table} WHERE state IS NOT NULL) = 
              (SELECT COUNT(DISTINCT state) FROM mart_hcp_deciled WHERE state IS NOT NULL) 
         THEN 'PASS' ELSE 'FAIL_STATES_MISMATCH' END AS status_6c;

-- 6d. ZIP Integrity Check (each ZIP in exactly one territory)
SELECT
    COUNT(*) AS total_assigned_zips,
    SUM(CASE WHEN territory_count > 1 THEN 1 ELSE 0 END) AS split_zips_count,
    CASE WHEN SUM(CASE WHEN territory_count > 1 THEN 1 ELSE 0 END) = 0 THEN 'PASS' ELSE 'FAIL_SPLIT_ZIPS' END AS status_6d
FROM (
    SELECT zip_code, COUNT(DISTINCT territory_id) AS territory_count
    FROM {roster_table}
    WHERE target_status = 'Primary Target' AND territory_id IS NOT NULL
    GROUP BY zip_code
);

-- 6e. Workload Band Integrity Check (all territories within 960-1440 visits - report count outside)
SELECT
    COUNT(*) AS total_territories,
    SUM(CASE WHEN total_visits < 960 OR total_visits > 1440 THEN 1 ELSE 0 END) AS territories_outside_band,
    MIN(total_visits) AS min_visits,
    MAX(total_visits) AS max_visits,
    CASE WHEN SUM(CASE WHEN total_visits < 960 OR total_visits > 1440 THEN 1 ELSE 0 END) = 0 THEN 'PASS' ELSE 'FAIL_OUTSIDE_BAND' END AS status_6e
FROM (
    SELECT territory_id, SUM(visits) AS total_visits
    FROM {roster_table}
    WHERE target_status = 'Primary Target' AND territory_id IS NOT NULL
    GROUP BY territory_id
);
