CREATE OR REPLACE TABLE mart_hcp_aggregated AS
SELECT
    hcp_id,
    ANY_VALUE(hcp_name) AS hcp_name,
    ANY_VALUE(specialty) AS specialty,
    ANY_VALUE(city) AS city,
    ANY_VALUE(state) AS state,
    ANY_VALUE(zip_code) AS zip_code,
    SUM(total_fills) AS total_fills,
    SUM(total_drug_cost) AS total_drug_cost
FROM stg_rx_base
GROUP BY hcp_id
HAVING SUM(total_fills) > 0;

CREATE OR REPLACE TABLE mart_hcp_deciled AS
WITH ranked AS (
    SELECT *,
        SUM(total_fills) OVER (
            PARTITION BY specialty
            ORDER BY total_fills DESC, hcp_id
        ) AS cum_vol,
        SUM(total_fills) OVER (PARTITION BY specialty) AS tot_vol
    FROM mart_hcp_aggregated
)
SELECT *,
    10 - LEAST(9, FLOOR(10 * (cum_vol - total_fills) / tot_vol))::INT AS decile,
    CASE
        WHEN 10 - LEAST(9, FLOOR(10 * (cum_vol - total_fills) / tot_vol)) >= 8 THEN 'Primary Target'
        WHEN 10 - LEAST(9, FLOOR(10 * (cum_vol - total_fills) / tot_vol)) >= 5 THEN 'Secondary Target'
        ELSE 'Low Priority'
    END AS target_status
FROM ranked;
