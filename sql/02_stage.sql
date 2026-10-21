CREATE OR REPLACE TABLE stg_rx_base AS
SELECT
    c.Prscrbr_NPI AS hcp_id,
    concat_ws(' ', c.Prscrbr_First_Name, c.Prscrbr_Last_Org_Name) AS hcp_name,
    c.Prscrbr_Type AS specialty,
    COALESCE(n.city, c.Prscrbr_City) AS city,
    COALESCE(n.state, c.Prscrbr_State_Abrvtn) AS state,
    LPAD(SUBSTRING(n.zip_raw, 1, 5), 5, '0') AS zip_code,
    c.Gnrc_Name AS generic_drug_name,
    CAST(c.Tot_30day_Fills AS DOUBLE) AS total_fills,
    CAST(c.Tot_Drug_Cst AS DOUBLE) AS total_drug_cost
FROM raw_cms_part_d c
LEFT JOIN raw_nppes n ON c.Prscrbr_NPI = n.NPI
WHERE UPPER(c.Gnrc_Name) IN (
    'ATORVASTATIN CALCIUM', 'ROSUVASTATIN CALCIUM', 'SIMVASTATIN',
    'PRAVASTATIN SODIUM', 'LOVASTATIN', 'PITAVASTATIN CALCIUM',
    'PITAVASTATIN MAGNESIUM', 'FLUVASTATIN SODIUM'
)
AND c.Prscrbr_Type IN ('Cardiology', 'Internal Medicine', 'Family Practice')
-- Explicit whitelist of the contiguous 48 states + DC (excludes AK, HI, PR, GU, ZZ, etc.)
AND COALESCE(n.state, c.Prscrbr_State_Abrvtn) IN (
    'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
);
