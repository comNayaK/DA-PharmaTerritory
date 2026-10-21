CREATE OR REPLACE TABLE raw_cms_part_d AS
SELECT * FROM
read_csv_auto('data/Medicare_Part_D_Prescribers_by_Provider_and_Drug_2023.csv',
all_varchar=true, normalize_names=true);

CREATE OR REPLACE TABLE raw_nppes AS
SELECT
    NPI,
    Entity_Type_Code,
    Provider_Business_Practice_Location_Address_City_Name AS city,
    Provider_Business_Practice_Location_Address_State_Name AS state,
    Provider_Business_Practice_Location_Address_Postal_Code AS zip_raw
FROM read_csv_auto('data/npidata_pfile_20050523-20260913.csv', all_varchar=true,
normalize_names=true)
WHERE Entity_Type_Code = '1'
-- Updated based on Section 0 audit
AND (provider_business_practice_location_address_country_code_if_outside_us IS NULL
OR provider_business_practice_location_address_country_code_if_outside_us = 'US');

CREATE OR REPLACE TABLE zip_lat_lon AS
SELECT
    GEOID AS zip_code,
    CAST(INTPTLAT AS DOUBLE) AS latitude,
    CAST(INTPTLONG AS DOUBLE) AS longitude
FROM read_csv_auto('data/2023_Gaz_zcta_national.txt', delim='\t', all_varchar=true,
normalize_names=true);
