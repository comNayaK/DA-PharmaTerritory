import pandas as pd
import numpy as np
import math
import sys
import os
import argparse
import json
import csv
import time
import duckdb
import scipy.sparse as sp
from scipy.optimize import linprog
from sklearn.cluster import KMeans

def compute_distances_km(xyz, centroids):
    dots = np.clip(np.dot(xyz, centroids.T), -1.0, 1.0)
    return 6371.0 * np.arccos(dots)

def solve_lp(xyz, weights, centroids, penalty_weight=1200.0, initial_M=10, max_M=60):
    n_zips = len(weights)
    k = len(centroids)
    M = initial_M
    while M <= max_M:
        dist_km = compute_distances_km(xyz, centroids)
        cand = np.argsort(dist_km, axis=1)[:, :M]
        
        n_assign_vars = n_zips * M
        n_vars = n_assign_vars + k
        c = np.zeros(n_vars)
        row_eq, col_eq, data_eq = [], [], []
        row_ub, col_ub, data_ub = [], [], []
        idx = 0
        for i in range(n_zips):
            w = weights[i]
            for m in range(M):
                t = cand[i, m]
                c[idx] = w * dist_km[i, t]
                row_eq.append(i); col_eq.append(idx); data_eq.append(1.0)
                # Band bounds: 960 <= V_t <= 1440
                row_ub.append(t); col_ub.append(idx); data_ub.append(w)
                row_ub.append(k + t); col_ub.append(idx); data_ub.append(-w)
                # Auxiliary evenness bounds: V_t - 1200 <= d_t and 1200 - V_t <= d_t
                row_ub.append(2 * k + t); col_ub.append(idx); data_ub.append(w)
                row_ub.append(3 * k + t); col_ub.append(idx); data_ub.append(-w)
                idx += 1
                
        # Auxiliary variables d_t: objective coefficient is penalty_weight (lambda)
        c[n_assign_vars:] = penalty_weight
        for t in range(k):
            dt_col = n_assign_vars + t
            row_ub.append(2 * k + t); col_ub.append(dt_col); data_ub.append(-1.0)
            row_ub.append(3 * k + t); col_ub.append(dt_col); data_ub.append(-1.0)
            
        A_eq = sp.csr_matrix((data_eq, (row_eq, col_eq)), shape=(n_zips, n_vars))
        b_eq = np.ones(n_zips)
        A_ub = sp.csr_matrix((data_ub, (row_ub, col_ub)), shape=(4 * k, n_vars))
        b_ub = np.concatenate([
            np.full(k, 1440.0),
            np.full(k, -960.0),
            np.full(k, 1200.0),
            np.full(k, -1200.0)
        ])
        
        bounds = [(0, 1)] * n_assign_vars + [(0, None)] * k
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            print(f"LP solved successfully with M={M} ({res.nit} iterations).")
            return res.x[:n_assign_vars].reshape((n_zips, M)), cand, dist_km
        print(f"M={M} infeasible for candidate restriction, widening candidate neighbors to M={M+5}...")
        M += 5
    raise RuntimeError(f"Could not solve LP even after widening to M={max_M}. Capacity band [960, 1440] infeasible.")

def repair_pass(assign, weights, dist_km, cand_search, k, max_passes=250):
    n_zips = len(weights)
    v = np.bincount(assign, weights=weights, minlength=k).astype(float)
    
    def calc_violation(visits):
        return np.sum(np.maximum(0, 960.0 - visits)) + np.sum(np.maximum(0, visits - 1440.0))

    initial_viol = calc_violation(v)
    unbalanced_initial = np.sum((v < 960.0) | (v > 1440.0))
    print(f"Repair pass start: {unbalanced_initial} territories outside [960, 1440], total visit violation = {initial_viol:.1f}")

    if initial_viol == 0:
        print("All territories already in band [960, 1440].")
        return assign, v

    n_cands = cand_search.shape[1]

    for pass_num in range(max_passes):
        viol = calc_violation(v)
        if viol == 0:
            print(f"Repair pass complete: 100% of territories in band [960, 1440] at pass {pass_num}!")
            break
            
        best_move = None
        best_delta = 0.0
        best_penalty = float('inf')
        
        surplus_terrs = set(np.where(v > 1440.0)[0])
        deficit_terrs = set(np.where(v < 960.0)[0])
        
        # 1-move search
        for i in range(n_zips):
            src = assign[i]
            w = weights[i]
            # Only consider moving out of surplus, or moving into a deficit territory
            if src not in surplus_terrs and len(deficit_terrs.intersection(cand_search[i])) == 0:
                continue
                
            for m in range(n_cands):
                dst = cand_search[i, m]
                if dst == src:
                    continue
                
                old_viol = (max(0.0, 960.0 - v[src]) + max(0.0, v[src] - 1440.0) +
                            max(0.0, 960.0 - v[dst]) + max(0.0, v[dst] - 1440.0))
                new_src_v = v[src] - w
                new_dst_v = v[dst] + w
                new_viol = (max(0.0, 960.0 - new_src_v) + max(0.0, new_src_v - 1440.0) +
                            max(0.0, 960.0 - new_dst_v) + max(0.0, new_dst_v - 1440.0))
                delta = new_viol - old_viol
                if delta < 0:
                    pen = dist_km[i, dst] - dist_km[i, src]
                    if (delta < best_delta) or (delta == best_delta and pen < best_penalty):
                        best_delta = delta
                        best_penalty = pen
                        best_move = (i, src, dst, w)
                        
        if best_move is not None:
            i, src, dst, w = best_move
            assign[i] = dst
            v[src] -= w
            v[dst] += w
            continue
            
        # 2-swap search if 1-move is stuck
        best_swap = None
        best_swap_delta = 0.0
        best_swap_pen = float('inf')
        
        unbalanced = list(surplus_terrs | deficit_terrs)
        for u in unbalanced:
            zips_u = np.where(assign == u)[0]
            for z1 in zips_u:
                w1 = weights[z1]
                for m in range(min(15, n_cands)):
                    t2 = cand_search[z1, m]
                    if t2 == u:
                        continue
                    zips_t2 = np.where(assign == t2)[0]
                    for z2 in zips_t2:
                        w2 = weights[z2]
                        if w1 == w2:
                            continue
                        if u not in cand_search[z2]:
                            continue
                        
                        old_viol = (max(0.0, 960.0 - v[u]) + max(0.0, v[u] - 1440.0) +
                                    max(0.0, 960.0 - v[t2]) + max(0.0, v[t2] - 1440.0))
                        new_u_v = v[u] - w1 + w2
                        new_t2_v = v[t2] - w2 + w1
                        new_viol = (max(0.0, 960.0 - new_u_v) + max(0.0, new_u_v - 1440.0) +
                                    max(0.0, 960.0 - new_t2_v) + max(0.0, new_t2_v - 1440.0))
                        delta = new_viol - old_viol
                        if delta < 0:
                            pen = (dist_km[z1, t2] - dist_km[z1, u]) + (dist_km[z2, u] - dist_km[z2, t2])
                            if (delta < best_swap_delta) or (delta == best_swap_delta and pen < best_swap_pen):
                                best_swap_delta = delta
                                best_swap_pen = pen
                                best_swap = (z1, u, z2, t2)
                                
        if best_swap is not None:
            z1, u, z2, t2 = best_swap
            assign[z1] = t2
            assign[z2] = u
            v[u] = v[u] - weights[z1] + weights[z2]
            v[t2] = v[t2] - weights[z2] + weights[z1]
            continue
            
        print(f"Repair reached local minimum at pass {pass_num} with {calc_violation(v):.1f} violation remaining.")
        break
        
    return assign, v

def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Pharma Commercial ZIP-level Balanced Territory Alignment")
    parser.add_argument("--state", default=None, help="Optional two-letter state code (e.g. FL). Default is National.")
    parser.add_argument("--out-table", default=None, help="DuckDB table name to store the roster (default: roster_balanced_<scope>)")
    parser.add_argument("--penalty-weight", "--lambda", dest="penalty_weight", type=float, default=1200.0, help="Penalty weight lambda for workload evenness term (default: 1200.0)")
    args = parser.parse_args()

    scope = args.state.upper() if args.state else "NATIONAL"
    out_table = args.out_table if args.out_table else f"roster_balanced_{scope}"

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

    # Aggregate to ZIP units: weight = sum(visits), hcp_count, lat/lon (ZCTA centroid), state
    df_zip = df.groupby("zip_code").agg(
        weight=("visits", "sum"),
        hcp_count=("hcp_id", "count"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        state=("state", lambda s: s.mode()[0])
    ).reset_index()

    n_zips = len(df_zip)
    total_visits = df_zip["weight"].sum()
    CALLS_PER_REP = 1200
    k = max(1, math.ceil(total_visits / CALLS_PER_REP))
    print(f"Aggregated {assigned_count} HCPs into {n_zips} unique ZIP units.")
    print(f"Total visits: {int(total_visits)} | Capacity Math: k = {k} territories ([960, 1440] visits/territory).")

    # 3D unit-sphere coordinates
    lat = np.radians(df_zip["latitude"].values)
    lon = np.radians(df_zip["longitude"].values)
    xyz = np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])
    weights = df_zip["weight"].values.astype(float)

    # (a) Init with sklearn KMeans on 3D unit-sphere coords using sample_weight=weight, k clusters, random_state=42
    print("Initializing centroids with weighted KMeans (k=224, random_state=42)...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(xyz, sample_weight=weights)
    centroids = km.cluster_centers_
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids = centroids / norms

    # (b) & (c) Capacitated assignment LP iterations & centroid updates
    n_iterations = 4
    x = None
    cand = None
    dist_km = None
    for it in range(n_iterations):
        print(f"--- Iteration {it + 1}/{n_iterations} ---")
        x, cand, dist_km = solve_lp(xyz, weights, centroids, penalty_weight=args.penalty_weight, initial_M=10, max_M=60)
        
        # Recompute centroids from assignment
        new_centroids = np.zeros((k, 3))
        M = cand.shape[1]
        for i in range(n_zips):
            w = weights[i]
            for m in range(M):
                val = x[i, m]
                if val > 1e-6:
                    t = cand[i, m]
                    new_centroids[t] += (w * val) * xyz[i]
        norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_centroids = new_centroids / norms
        shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        print(f"Iteration {it + 1} max centroid shift: {shift:.6f}")
        centroids = new_centroids
        if shift < 1e-4:
            print("Centroids converged.")
            break

    # (d) Round any fractional ZIPs to their largest share
    print("Rounding fractional ZIP assignments to largest share...")
    zip_assignment = np.array([cand[i, np.argmax(x[i])] for i in range(n_zips)], dtype=int)
    
    # Candidate search space for repair
    cand_search = np.argsort(dist_km, axis=1)[:, :20]
    
    # Run repair pass moving boundary ZIPs to neighboring territories until all within band
    zip_assignment, terr_visits = repair_pass(zip_assignment, weights, dist_km, cand_search, k)

    # Verify rule: a ZIP is NEVER split. Assert every ZIP maps to exactly one territory.
    zip_to_terr = dict(zip(df_zip["zip_code"], zip_assignment))
    df["territory_id"] = df["zip_code"].map(zip_to_terr)

    split_check = df.groupby("zip_code")["territory_id"].nunique()
    assert (split_check == 1).all(), "CRITICAL ASSERTION FAILED: A ZIP is split across multiple territories!"
    print(f"ZIP Integrity verified: 0 split ZIPs across all {n_zips} ZIPs.")

    # Recompute final centroids from discrete assignment
    final_centroids = np.zeros((k, 3))
    for t in range(k):
        zips_in_t = np.where(zip_assignment == t)[0]
        if len(zips_in_t) > 0:
            c_vec = np.sum(weights[zips_in_t, np.newaxis] * xyz[zips_in_t], axis=0)
            c_norm = np.linalg.norm(c_vec)
            final_centroids[t] = c_vec / (c_norm if c_norm > 0 else 1.0)
        else:
            final_centroids[t] = centroids[t]

    centroid_lats = np.degrees(np.arcsin(final_centroids[:, 2]))
    centroid_lons = np.degrees(np.arctan2(final_centroids[:, 1], final_centroids[:, 0]))

    # Register with DuckDB and create balanced roster table
    con.register("df_territory", df)

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

    # Verify surrogate ID join against env salt
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

    # Exports
    os.makedirs("output", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # 1. Roster CSV: Primary Targets with assigned territory only, ZIP quoted
    roster_csv_path = f"output/tableau_territory_roster_balanced_{scope}.csv"
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
    print(f"Balanced Roster CSV exported to {roster_csv_path} ({len(df_roster_export)} rows)")

    # 2. Territory summary CSV: territory_id, hcp_count, zip_count, total_visits, states_spanned, centroid_lat, centroid_lon, within_tolerance, pct_of_1200
    terr_summary = df.groupby("territory_id").agg(
        hcp_count=("hcp_id", "count"),
        zip_count=("zip_code", "nunique"),
        total_visits=("visits", "sum"),
        states_spanned=("state", "nunique")
    ).reset_index()

    terr_summary["centroid_lat"] = terr_summary["territory_id"].map(lambda t: round(float(centroid_lats[t]), 5))
    terr_summary["centroid_lon"] = terr_summary["territory_id"].map(lambda t: round(float(centroid_lons[t]), 5))
    terr_summary["within_tolerance"] = ((terr_summary["total_visits"] >= 960) & (terr_summary["total_visits"] <= 1440))
    terr_summary["pct_of_1200"] = ((terr_summary["total_visits"] / CALLS_PER_REP) * 100.0).round(1)

    summary_csv_path = f"output/territory_summary_balanced_{scope}.csv"
    terr_summary.to_csv(summary_csv_path, index=False)
    print(f"Balanced Territory Summary CSV exported to {summary_csv_path} ({len(terr_summary)} rows)")

    # Distance metrics from HCP to territory centroid
    hcp_lat = np.radians(df["latitude"].values)
    hcp_lon = np.radians(df["longitude"].values)
    hcp_xyz = np.column_stack([
        np.cos(hcp_lat) * np.cos(hcp_lon),
        np.cos(hcp_lat) * np.sin(hcp_lon),
        np.sin(hcp_lat)
    ])
    hcp_terrs = df["territory_id"].values.astype(int)
    assigned_centroids = final_centroids[hcp_terrs]
    hcp_dots = np.clip(np.sum(hcp_xyz * assigned_centroids, axis=1), -1.0, 1.0)
    hcp_dist_km = 6371.0 * np.arccos(hcp_dots)

    mean_hcp_dist = round(float(np.mean(hcp_dist_km)), 2)
    p90_hcp_dist = round(float(np.percentile(hcp_dist_km, 90)), 2)
    max_radius = round(float(np.max(hcp_dist_km)), 2)

    runtime_sec = round(time.time() - t_start, 2)
    print(f"Distance metrics: Mean = {mean_hcp_dist} km, P90 = {p90_hcp_dist} km, Max Radius = {max_radius} km.")
    print(f"Total balanced alignment runtime: {runtime_sec:.2f} seconds.")

    # 3. Metrics JSON
    visit_sums = terr_summary["total_visits"]
    cv = round(float(visit_sums.std() / visit_sums.mean()), 3)
    within_tol_pct = round(float(terr_summary["within_tolerance"].mean() * 100.0), 1)
    pct_1080_1320 = round(float(((visit_sums >= 1080.0) & (visit_sums <= 1320.0)).mean() * 100.0), 1)
    n_crossing = int((terr_summary["states_spanned"] > 1).sum())

    n_lower = int((visit_sums == 960.0).sum())
    n_upper = int((visit_sums == 1440.0).sum())
    n_within_5pct = int(((visit_sums <= 960.0 * 1.05) | (visit_sums >= 1440.0 * 0.95)).sum())

    metrics_data = {
        "k": int(k),
        "hcp_count": int(total_primary_targets),
        "assigned": int(assigned_count),
        "unassigned": int(unassigned_count),
        "dropped_pct": round(float(dropped_pct), 4),
        "visits_min": float(visit_sums.min()),
        "visits_median": float(visit_sums.median()),
        "visits_max": float(visit_sums.max()),
        "visits_mean": round(float(visit_sums.mean()), 3),
        "cv": cv,
        "pct_within_960_1440": within_tol_pct,
        "pct_within_1080_1320": pct_1080_1320,
        "n_territories_crossing_state_lines": n_crossing,
        "n_zips_split_across_territories": 0,
        "n_hcps_in_split_zips": 0,
        "n_territories_at_lower_bound": n_lower,
        "n_at_upper_bound": n_upper,
        "n_within_5pct_of_a_bound": n_within_5pct,
        "mean_hcp_distance_to_centroid_km": mean_hcp_dist,
        "p90_hcp_distance_to_centroid_km": p90_hcp_dist,
        "max_territory_radius_km": max_radius,
        "penalty_weight": args.penalty_weight,
        "runtime_seconds": runtime_sec
    }

    metrics_json_path = f"metrics/{scope}_balanced.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to {metrics_json_path}")

    # 4. Comparison JSON (naive / band_only / balanced)
    naive_json_path = f"metrics/{scope}_naive.json"
    band_only_json_path = f"metrics/{scope}_band_only.json"
    
    naive_metrics = None
    if os.path.exists(naive_json_path):
        with open(naive_json_path) as f:
            naive_metrics = json.load(f)
            
    band_only_metrics = None
    if os.path.exists(band_only_json_path):
        with open(band_only_json_path) as f:
            band_only_metrics = json.load(f)

    if naive_metrics is not None:
        def make_comp_entry(k):
            entry = {"naive": naive_metrics.get(k)}
            if band_only_metrics is not None:
                entry["band_only"] = band_only_metrics.get(k)
            entry["balanced"] = metrics_data.get(k)
            return entry

        comparison_data = {
            "comparison": {
                "k": make_comp_entry("k"),
                "pct_within_960_1440": make_comp_entry("pct_within_960_1440"),
                "pct_within_1080_1320": make_comp_entry("pct_within_1080_1320"),
                "cv": make_comp_entry("cv"),
                "visits_min": make_comp_entry("visits_min"),
                "visits_max": make_comp_entry("visits_max"),
                "visits_median": make_comp_entry("visits_median"),
                "visits_mean": make_comp_entry("visits_mean"),
                "n_territories_at_lower_bound": make_comp_entry("n_territories_at_lower_bound"),
                "n_at_upper_bound": make_comp_entry("n_at_upper_bound"),
                "n_within_5pct_of_a_bound": make_comp_entry("n_within_5pct_of_a_bound"),
                "n_zips_split_across_territories": make_comp_entry("n_zips_split_across_territories"),
                "n_hcps_in_split_zips": make_comp_entry("n_hcps_in_split_zips"),
                "n_territories_crossing_state_lines": make_comp_entry("n_territories_crossing_state_lines"),
                "mean_distance_to_centroid_km": {
                    "naive": naive_metrics.get("mean_hcp_distance_to_centroid_km"),
                    **({"band_only": band_only_metrics.get("mean_hcp_distance_to_centroid_km")} if band_only_metrics else {}),
                    "balanced": mean_hcp_dist
                },
                "p90_distance_to_centroid_km": {
                    "naive": naive_metrics.get("p90_hcp_distance_to_centroid_km"),
                    **({"band_only": band_only_metrics.get("p90_hcp_distance_to_centroid_km")} if band_only_metrics else {}),
                    "balanced": p90_hcp_dist
                },
                "max_territory_radius_km": {
                    "naive": naive_metrics.get("max_territory_radius_km"),
                    **({"band_only": band_only_metrics.get("max_territory_radius_km")} if band_only_metrics else {}),
                    "balanced": max_radius
                }
            },
            "naive": naive_metrics,
            "balanced": metrics_data
        }
        if band_only_metrics is not None:
            comparison_data["band_only"] = band_only_metrics

        comp_json_path = "metrics/comparison_naive_vs_balanced.json"
        with open(comp_json_path, "w") as f:
            json.dump(comparison_data, f, indent=2)
        print(f"Comparison metrics saved to {comp_json_path}")

    con.close()

if __name__ == "__main__":
    main()
