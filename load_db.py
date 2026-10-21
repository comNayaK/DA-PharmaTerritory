import duckdb
import os

con = duckdb.connect("pharma_commercial.duckdb")

with open('sql/01_load.sql', 'r') as f:
    con.execute(f.read())

with open('sql/02_stage.sql', 'r') as f:
    con.execute(f.read())

with open('sql/03_mart.sql', 'r') as f:
    con.execute(f.read())

print("DB load complete")
