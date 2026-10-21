import pandas as pd
import numpy as np
import math
import sys
import os
import argparse
import json
import csv
from k_means_constrained import KMeansConstrained
import duckdb

def main():
    parser = argparse.ArgumentParser(description="Pharma Commercial HCP Territory Alignment (Naive Baseline)")
    parser.add_argument("--state", default=None, help="Optional two-letter state code (e.g. FL). Default is National.")
    parser.add_argument("--out-table", default=None, help="DuckDB table name to store the roster (default: roster_naive_<scope>)")
    args = parser.parse_args()

    scope = args.state.upper() if args.state else "NATIONAL"
    out_table = args.out_table if args.out_table else f"roster_naive_{scope}"

    if "HCP_SALT" not in os.environ and os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].strip()
                if line.startswith("HCP_SALT="):
                    os.environ["HCP_SALT"] = line.split("=", 1)[1].strip("'\"")

    salt = os.environ.get("HCP_SALT")
    assert salt is not None, "HCP_SALT environment variable must be set. Never hardcode the salt."
    assert salt.isalnum() and len(salt) >= 16, "HCP_SALT must be alphanumeric and at least 16 characters"

    con = duckdb.connect("pharma_commercial.duckdb")

    state_query = f"AND d.state = '{args.state.upper()}'" if args.state else ""

    # Primary targets in scope
    df_scope_targets = con.execute(f"""
        SELECT d.hcp_id, d.decile, d.state, d.zip_code
        FROM mart_hcp_deciled d
        WHERE d.target_status = 'Primary Target'
          {state_query}
    """).df()

    total_primary_targets = len(df_scope_targets)

    # Fetch valid Primary Targets with ZCTA coordinates
    df = con.execute(f"""
        SELECT d.hcp_id, d.total_fills, d.decile, d.state, d.zip_code, z.latitude, z.longitude 
        FROM mart_hcp_deciled d
        JOIN zip_lat_lon z ON d.zip_code = z.zip_code
        WHERE d.target_status = 'Primary Target'
          AND z.latitude IS NOT NULL
          {state_query}
    """).df()

    assigned_count = len(df)
    unassigned_count = total_primary_targets - assigned_count
    dropped_pct = (unassigned_count * 100.0 / total_primary_targets) if total_primary_targets > 0 else 0.0

    print(f"Target Scope: {scope} | Primary Targets in Scope: {total_primary_targets}")
    print(f"Geocoded Targets: {assigned_count} | Unassigned Targets: {unassigned_count} ({dropped_pct:.2f}%)")

    if assigned_count == 0:
        sys.exit("ERROR: No targets matched coordinates. Check ZCTA join.")

    FREQ = {10: 24, 9: 18, 8: 12}
    df["visits"] = df["decile"].map(FREQ).fillna(0)

    CALLS_PER_REP = 1200 
    total_calls_needed = df["visits"].sum()
    num_territories = max(1, math.ceil(total_calls_needed / CALLS_PER_REP))

    print(f"Capacity Math: {int(total_calls_needed)} total calls needed.")
    print(f"Divided by {CALLS_PER_REP} calls/rep = {num_territories} geographic territories.")

    if num_territories > 300:
        print("WARNING: Scope may be too massive. KMeansConstrained can be memory-heavy. Benchmark on a single state first if needed.")

    lat = np.radians(df['latitude'])
    lon = np.radians(df['longitude'])
    xyz = np.c_[np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]

    target_hcp_count = len(df) / num_territories
    kmc = KMeansConstrained(
        n_clusters=num_territories,
        size_min=int(target_hcp_count * 0.8), 
        size_max=int(target_hcp_count * 1.2),
        random_state=42
    )
    df["territory_id"] = kmc.fit_predict(xyz)

    con.register('df_territory', df)

    # Create scoped roster table in DuckDB
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS 
        SELECT 
            MD5(? || h.hcp_id) AS hcp_surrogate_id, 
            h.specialty,
            h.city,
            h.state,
            LPAD(h.zip_code, 5, '0') AS zip_code,
            h.decile,
            h.target_status,
            h.total_fills,
            t.territory_id,
            t.latitude,
            t.longitude,
            t.visits
        FROM mart_hcp_deciled h
        LEFT JOIN df_territory t ON h.hcp_id = t.hcp_id
        WHERE 1=1 {state_query}
    """, [salt])

    # Verify surrogate ID uses exactly env HCP_SALT
    match_count = con.execute(f"""
        SELECT COUNT(*)
        FROM {out_table} r
        JOIN mart_hcp_deciled h 
          ON r.hcp_surrogate_id = MD5(? || h.hcp_id)
        WHERE 1=1 {state_query}
    """, [salt]).fetchone()[0]
    total_roster_rows = con.execute(f"SELECT COUNT(*) FROM {out_table}").fetchone()[0]
    assert match_count == total_roster_rows, (
        f"Surrogate ID verification failed: matched {match_count} of {total_roster_rows} rows"
    )
    print(f"Surrogate ID verified: all {total_roster_rows} rows match MD5(HCP_SALT || hcp_id).")

    # Print NULL-territory count
    null_info = con.execute(f"""
        SELECT 
            COUNT(*) AS total_hcp,
            SUM(CASE WHEN territory_id IS NULL THEN 1 ELSE 0 END) AS null_territories,
            SUM(CASE WHEN target_status = 'Primary Target' AND territory_id IS NULL THEN 1 ELSE 0 END) AS null_primary_targets
        FROM {out_table}
    """).fetchone()
    print(f"Table {out_table} created. Total HCPs: {null_info[0]}, Total NULL territories: {null_info[1]} (Primary Targets with NULL territory: {null_info[2]})")

    # Check Workload Balance
    hcp_counts = df.groupby('territory_id').size()
    visit_sums = df.groupby('territory_id')['visits'].sum()

    print(f"Territory HCP Count (Constrained) - Max: {hcp_counts.max()}, Min: {hcp_counts.min()}")
    print(f"Territory Visit Workload (Unconstrained) - Max: {visit_sums.max()}, Min: {visit_sums.min()}")
    cv = visit_sums.std() / visit_sums.mean()
    print(f"Visit Workload CV: {cv:.2f}")
    within_tol_series = ((visit_sums >= CALLS_PER_REP * 0.8) & (visit_sums <= CALLS_PER_REP * 1.2))
    within_tolerance = within_tol_series.mean() * 100
    print(f"Territories within ±20% of {CALLS_PER_REP} visits: {within_tolerance:.1f}%")

    if within_tolerance >= 70.0:
        print("PASS: Workload variance is within acceptable target (>= 70%).")
    else:
        print("FAIL: Workload variance exceeds acceptable target (< 70%).")

    # Exports
    os.makedirs('output', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    # 1. Roster CSV: Primary Targets with assigned territory only, ZIP written as quoted string
    roster_csv_path = f"output/tableau_territory_roster_{scope}.csv"
    df_roster_export = con.execute(f"""
        SELECT 
            hcp_surrogate_id,
            specialty,
            city,
            state,
            zip_code,
            decile,
            target_status,
            total_fills,
            territory_id,
            latitude,
            longitude,
            visits
        FROM {out_table}
        WHERE target_status = 'Primary Target' AND territory_id IS NOT NULL
        ORDER BY territory_id, decile DESC
    """).df()

    df_roster_export["zip_code"] = df_roster_export["zip_code"].astype(str).str.zfill(5)
    df_roster_export.to_csv(roster_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Roster CSV exported to {roster_csv_path} ({len(df_roster_export)} rows)")

    # 2. Territory summary CSV: territory_id, hcp_count, total_visits, states_spanned, within_tolerance flag, pct_of_1200
    terr_summary = df.groupby('territory_id').agg(
        hcp_count=('hcp_id', 'count'),
        total_visits=('visits', 'sum'),
        states_spanned=('state', 'nunique')
    ).reset_index()
    terr_summary["within_tolerance"] = ((terr_summary["total_visits"] >= CALLS_PER_REP * 0.8) & (terr_summary["total_visits"] <= CALLS_PER_REP * 1.2))
    terr_summary["pct_of_1200"] = ((terr_summary["total_visits"] / CALLS_PER_REP) * 100.0).round(1)

    summary_csv_path = f"output/territory_summary_{scope}.csv"
    terr_summary.to_csv(summary_csv_path, index=False)
    print(f"Territory Summary CSV exported to {summary_csv_path} ({len(terr_summary)} rows)")

    # 3. Metrics JSON
    n_crossing = int((terr_summary["states_spanned"] > 1).sum())
    zip_terr = df.groupby('zip_code')['territory_id'].nunique()
    split_zips = zip_terr[zip_terr > 1]
    n_split_zips = int(len(split_zips))
    n_hcps_split = int(df[df['zip_code'].isin(split_zips.index)].shape[0])

    # Distance metrics from HCP to territory centroid (visit-weighted)
    hcp_terrs = df["territory_id"].values.astype(int)
    w_visits = df["visits"].values
    n_k = int(num_territories)
    centroids = np.zeros((n_k, 3))
    for t in range(n_k):
        pts = xyz[hcp_terrs == t]
        pts_w = w_visits[hcp_terrs == t]
        if len(pts) > 0:
            c = np.sum(pts * pts_w[:, None], axis=0)
            centroids[t] = c / np.linalg.norm(c)
    assigned_centroids = centroids[hcp_terrs]
    hcp_dots = np.clip(np.sum(xyz * assigned_centroids, axis=1), -1.0, 1.0)
    hcp_dist_km = 6371.0 * np.arccos(hcp_dots)

    mean_hcp_dist = round(float(np.mean(hcp_dist_km)), 2)
    p90_hcp_dist = round(float(np.percentile(hcp_dist_km, 90)), 2)
    max_radius = round(float(np.max(hcp_dist_km)), 2)

    metrics_data = {
        "k": int(num_territories),
        "hcp_count": int(total_primary_targets),
        "assigned": int(assigned_count),
        "unassigned": int(unassigned_count),
        "dropped_pct": round(float(dropped_pct), 4),
        "visits_min": float(visit_sums.min()),
        "visits_median": float(visit_sums.median()),
        "visits_max": float(visit_sums.max()),
        "visits_mean": round(float(visit_sums.mean()), 3),
        "cv": round(float(cv), 3),
        "pct_within_960_1440": round(float(within_tolerance), 1),
        "n_territories_crossing_state_lines": n_crossing,
        "n_zips_split_across_territories": n_split_zips,
        "n_hcps_in_split_zips": n_hcps_split,
        "mean_hcp_distance_to_centroid_km": mean_hcp_dist,
        "p90_hcp_distance_to_centroid_km": p90_hcp_dist,
        "max_territory_radius_km": max_radius
    }

    metrics_json_path = f"metrics/{scope}_naive.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to {metrics_json_path}")

    con.close()

if __name__ == "__main__":
    main()
