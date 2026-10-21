import os
import duckdb
import numpy as np
import pandas as pd
import csv
import json

def dsatur_coloring(centroids, k_neighbors=8, max_colors=8):
    """
    Color territory centroids using the DSATUR (Degree of Saturation) heuristic.

    Adjacency is defined by each territory's k nearest centroids in spherical coordinates,
    symmetrized into an undirected graph. DSATUR dynamically selects uncolored territories
    with the highest saturation degree (count of unique colors among neighbors) to minimize
    monochromatic adjacent territory conflicts on map visualizations.
    """
    k = len(centroids)
    dots = np.clip(np.dot(centroids, centroids.T), -1.0, 1.0)
    dist = 6371.0 * np.arccos(dots)
    np.fill_diagonal(dist, np.inf)

    adj = {i: set(np.argsort(dist[i])[:k_neighbors]) for i in range(k)}
    undir = {i: set() for i in range(k)}
    for i in range(k):
        for j in adj[i]:
            undir[i].add(j)
            undir[j].add(i)

    degrees = {i: len(undir[i]) for i in range(k)}
    colors = {}
    neighbor_colors = {i: set() for i in range(k)}

    while len(colors) < k:
        uncolored = [i for i in range(k) if i not in colors]
        # Choose vertex with maximum saturation degree; tie-break by degree descending, then by territory_id
        node = max(uncolored, key=lambda x: (len(neighbor_colors[x]), degrees[x], -x))

        used = neighbor_colors[node]
        chosen = None
        for c in range(1, max_colors + 1):
            if c not in used:
                chosen = c
                break
        if chosen is None:
            c_conflicts = {c: sum(1 for nbr in undir[node] if colors.get(nbr) == c) for c in range(1, max_colors + 1)}
            chosen = min(c_conflicts, key=c_conflicts.get)

        colors[node] = chosen
        for nbr in undir[node]:
            neighbor_colors[nbr].add(chosen)

    conflicts = 0
    for u in range(k):
        for v in undir[u]:
            if u < v and colors[u] == colors[v]:
                conflicts += 1

    color_list = [colors[i] for i in range(k)]
    colors_used = len(set(color_list))
    return color_list, conflicts, colors_used

def main():
    os.makedirs("output/tableau", exist_ok=True)
    con = duckdb.connect("pharma_commercial.duckdb", read_only=True)

    # 1. Fetch Balanced Roster
    df_balanced = con.execute("""
        SELECT 
            hcp_surrogate_id,
            specialty,
            city,
            state,
            LPAD(zip_code, 5, '0') AS zip_code,
            decile,
            target_status,
            total_fills,
            visits,
            territory_id AS territory_balanced,
            latitude,
            longitude
        FROM roster_balanced_NATIONAL
        WHERE target_status = 'Primary Target' AND territory_id IS NOT NULL
        ORDER BY territory_balanced, decile DESC
    """).df()

    # 2. Fetch Naive Roster
    df_naive = con.execute("""
        SELECT 
            hcp_surrogate_id,
            territory_id AS territory_naive
        FROM roster_naive_NATIONAL
        WHERE target_status = 'Primary Target' AND territory_id IS NOT NULL
    """).df()

    con.close()

    # Merge balanced and naive
    df_hcp = pd.merge(df_balanced, df_naive, on="hcp_surrogate_id", how="inner")
    assert len(df_hcp) == 15793, f"Expected 15793 HCPs, got {len(df_hcp)}"

    # Compute territory stats and centroids for both schemes
    schemes_data = []
    summary_dfs = []

    for scheme, terr_col in [("Naive", "territory_naive"), ("Balanced", "territory_balanced")]:
        df_sub = df_hcp.copy()
        df_sub["territory_id"] = df_sub[terr_col].astype(int)

        lat = np.radians(df_sub["latitude"].values)
        lon = np.radians(df_sub["longitude"].values)
        xyz = np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        terrs = df_sub["territory_id"].values
        k = terrs.max() + 1

        centroids = np.zeros((k, 3))
        for t in range(k):
            pts = xyz[terrs == t]
            w = df_sub["visits"].values[terrs == t]
            c = np.sum(pts * w[:, None], axis=0)
            c_norm = np.linalg.norm(c)
            centroids[t] = c / (c_norm if c_norm > 0 else 1.0)

        # Distance to centroid for each HCP
        assigned_centroids = centroids[terrs]
        dots = np.clip(np.sum(xyz * assigned_centroids, axis=1), -1.0, 1.0)
        df_sub["dist_km"] = 6371.0 * np.arccos(dots)

        c_lats = np.degrees(np.arcsin(centroids[:, 2]))
        c_lons = np.degrees(np.arctan2(centroids[:, 1], centroids[:, 0]))

        # Coloring and conflict detection using DSATUR (max 8 colors, symmetrized 8 nearest neighbors)
        color_groups, conflicts, colors_used = dsatur_coloring(centroids, k_neighbors=8, max_colors=8)
        print(f"DSATUR coloring conflicts ({scheme} scheme, max 8 colors, k=8 neighbors): {conflicts} conflicts, {colors_used} colors used")

        # Territory label helper
        def get_top2_states(series):
            counts = series.value_counts()
            top = list(counts.index[:2])
            return "/".join(top)

        summary_rows = []
        for t in range(k):
            t_hcps = df_sub[df_sub["territory_id"] == t]
            hcp_cnt = len(t_hcps)
            zip_cnt = t_hcps["zip_code"].nunique()
            tot_vis = int(t_hcps["visits"].sum())
            states_spanned = t_hcps["state"].nunique()
            top2_states = get_top2_states(t_hcps["state"])
            label = f"T{t:03d} {top2_states}"
            c_lat = round(float(c_lats[t]), 5)
            c_lon = round(float(c_lons[t]), 5)
            radius = round(float(t_hcps["dist_km"].max()), 2)
            within_band = "Y" if (tot_vis >= 960 and tot_vis <= 1440) else "N"
            pct_1200 = round((tot_vis / 1200.0) * 100.0, 1)
            c_group = color_groups[t]

            summary_rows.append({
                "scheme": scheme,
                "territory_id": t,
                "territory_label": label,
                "hcp_count": hcp_cnt,
                "zip_count": zip_cnt,
                "total_visits": tot_vis,
                "states_spanned": states_spanned,
                "centroid_lat": c_lat,
                "centroid_lon": c_lon,
                "radius_km": radius,
                "within_band": within_band,
                "pct_of_1200": pct_1200,
                "color_group": c_group
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_dfs.append(summary_df)
        schemes_data.append((scheme, summary_df, centroids, color_groups))

    # A) territory_summary_combined.csv
    df_combined_summary = pd.concat(summary_dfs, ignore_index=True)
    summary_path = "output/tableau/territory_summary_combined.csv"
    df_combined_summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Generated {summary_path} ({len(df_combined_summary)} rows)")

    # B) hcp_roster.csv
    balanced_summary_map = {row["territory_id"]: row for row in schemes_data[1][1].to_dict("records")}
    df_hcp["territory_label_balanced"] = df_hcp["territory_balanced"].map(lambda t: balanced_summary_map[t]["territory_label"])
    df_hcp["color_group_balanced"] = df_hcp["territory_balanced"].map(lambda t: balanced_summary_map[t]["color_group"])

    roster_cols = [
        "hcp_surrogate_id",
        "specialty",
        "city",
        "state",
        "zip_code",
        "decile",
        "total_fills",
        "visits",
        "latitude",
        "longitude",
        "territory_balanced",
        "territory_naive",
        "territory_label_balanced",
        "color_group_balanced"
    ]
    df_hcp_export = df_hcp[roster_cols].copy()
    df_hcp_export["zip_code"] = df_hcp_export["zip_code"].astype(str).str.zfill(5)

    roster_path = "output/tableau/hcp_roster.csv"
    df_hcp_export.to_csv(roster_path, index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    print(f"Generated {roster_path} ({len(df_hcp_export)} rows)")

    # C) kpi_comparison.csv
    with open("metrics/comparison_naive_vs_balanced.json") as f:
        comp_json = json.load(f)
    comp = comp_json["comparison"]

    kpi_rows = [
        {"metric": "pct_within_960_1440", "naive": comp["pct_within_960_1440"]["naive"], "band_only": comp["pct_within_960_1440"].get("band_only"), "balanced": comp["pct_within_960_1440"]["balanced"]},
        {"metric": "pct_within_1080_1320", "naive": comp["pct_within_1080_1320"]["naive"], "band_only": comp["pct_within_1080_1320"].get("band_only"), "balanced": comp["pct_within_1080_1320"]["balanced"]},
        {"metric": "cv", "naive": comp["cv"]["naive"], "band_only": comp["cv"].get("band_only"), "balanced": comp["cv"]["balanced"]},
        {"metric": "visits_min", "naive": comp["visits_min"]["naive"], "band_only": comp["visits_min"].get("band_only"), "balanced": comp["visits_min"]["balanced"]},
        {"metric": "visits_max", "naive": comp["visits_max"]["naive"], "band_only": comp["visits_max"].get("band_only"), "balanced": comp["visits_max"]["balanced"]},
        {"metric": "split_zips", "naive": comp["n_zips_split_across_territories"]["naive"], "band_only": comp["n_zips_split_across_territories"].get("band_only"), "balanced": comp["n_zips_split_across_territories"]["balanced"]},
        {"metric": "hcps_in_split_zips", "naive": comp["n_hcps_in_split_zips"]["naive"], "band_only": comp["n_hcps_in_split_zips"].get("band_only"), "balanced": comp["n_hcps_in_split_zips"]["balanced"]},
        {"metric": "state_crossing_territories", "naive": comp["n_territories_crossing_state_lines"]["naive"], "band_only": comp["n_territories_crossing_state_lines"].get("band_only"), "balanced": comp["n_territories_crossing_state_lines"]["balanced"]},
        {"metric": "mean_dist_km", "naive": comp["mean_distance_to_centroid_km"]["naive"], "band_only": comp["mean_distance_to_centroid_km"].get("band_only"), "balanced": comp["mean_distance_to_centroid_km"]["balanced"]},
        {"metric": "p90_dist_km", "naive": comp["p90_distance_to_centroid_km"]["naive"], "band_only": comp["p90_distance_to_centroid_km"].get("band_only"), "balanced": comp["p90_distance_to_centroid_km"]["balanced"]}
    ]
    df_kpi = pd.DataFrame(kpi_rows)
    kpi_path = "output/tableau/kpi_comparison.csv"
    df_kpi.to_csv(kpi_path, index=False, encoding="utf-8")
    print(f"Generated {kpi_path} ({len(df_kpi)} rows)")

    # D) tableau_data_dictionary.md
    dict_content = """# Tableau Data Dictionary

## 1. territory_summary_combined.csv
- **scheme**: Alignment model used for the territory ('Naive' or 'Balanced').
- **territory_id**: Unique integer identifier of the territory (0 to 223) within the alignment scheme.
- **territory_label**: Human-readable territory label with 3-digit ID and top 2 states by HCP count (e.g. 'T017 PA/NJ').
- **hcp_count**: Total number of Primary Target HCPs assigned to this territory.
- **zip_count**: Total number of distinct 5-digit ZIP codes in this territory.
- **total_visits**: Annual scheduled sales visit workload for all Primary Target HCPs in the territory.
- **states_spanned**: Number of distinct US states intersected by the territory's assigned HCPs.
- **centroid_lat**: Latitude of territory centroid on 3D unit sphere (degrees).
- **centroid_lon**: Longitude of territory centroid on 3D unit sphere (degrees).
- **radius_km**: Maximum distance (in kilometers) from any assigned HCP to the territory centroid.
- **within_band**: Workload compliance flag ('Y' if total_visits is within [960, 1440], else 'N').
- **pct_of_1200**: Territory workload expressed as a percentage of the nominal 1,200 annual visit capacity.
- **color_group**: Integer palette group (1-8) derived from DSATUR graph coloring on territory adjacency (8 nearest neighbors symmetrized).

## 2. hcp_roster.csv
- **hcp_surrogate_id**: Anonymized surrogate identifier for the prescriber (MD5 hash of environment salt concatenated with NPI).
- **specialty**: Medical specialty of the prescriber ('Cardiology', 'Family Practice', or 'Internal Medicine').
- **city**: Practice city name from NPPES provider record.
- **state**: Two-letter US state code from NPPES provider record.
- **zip_code**: Quoted 5-digit US postal ZIP code formatted with leading zeros.
- **decile**: Prescribing decile (1 to 10) calculated within specialty based on total statin 30-day fills.
- **total_fills**: Total annual statin prescription fills (standardized 30-day claims) in CMS Medicare Part D 2023.
- **visits**: Required annual personal promotion calls based on decile tier (Decile 10: 24, Decile 9: 18, Decile 8: 12).
- **latitude**: Latitude coordinate of the matching Census 2023 ZCTA centroid.
- **longitude**: Longitude coordinate of the matching Census 2023 ZCTA centroid.
- **territory_balanced**: Territory ID (0-223) assigned under the ZIP-level balanced alignment scheme.
- **territory_naive**: Territory ID (0-223) assigned under the unconstrained HCP KMeans baseline scheme.
- **territory_label_balanced**: Territory label under the balanced scheme (e.g. 'T017 PA/NJ').
- **color_group_balanced**: Map visualization color palette index (1-8) for the balanced territory assignment.

## 3. kpi_comparison.csv
- **metric**: Performance metric name (e.g. pct_within_960_1440, cv, visits_min, visits_max, split_zips, hcps_in_split_zips, state_crossing_territories, mean_dist_km, p90_dist_km).
- **naive**: Value of the metric under the unconstrained naive baseline alignment.
- **balanced**: Value of the metric under the capacitated ZIP-level balanced alignment.
"""
    dict_path = "output/tableau/tableau_data_dictionary.md"
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write(dict_content)
    print(f"Generated {dict_path}")

    # Verifications
    print("\n--- Verifications ---")
    assert len(df_hcp_export) == 15793, f"Roster count expected 15793, got {len(df_hcp_export)}"
    assert len(df_combined_summary) == 448, f"Summary count expected 448, got {len(df_combined_summary)}"
    print("Row counts verified: Roster = 15,793, Summary = 448.")

    key_fields_roster = ["hcp_surrogate_id", "territory_balanced", "territory_naive", "zip_code", "visits"]
    for col in key_fields_roster:
        null_cnt = df_hcp_export[col].isnull().sum()
        assert null_cnt == 0, f"Nulls found in roster column {col}: {null_cnt}"
    print("No nulls in key roster fields verified.")

    key_fields_summary = ["scheme", "territory_id", "territory_label", "total_visits", "color_group"]
    for col in key_fields_summary:
        null_cnt = df_combined_summary[col].isnull().sum()
        assert null_cnt == 0, f"Nulls found in summary column {col}: {null_cnt}"
    print("No nulls in key summary fields verified.")

    # Territory uniqueness
    for s in ["Naive", "Balanced"]:
        s_terrs = df_combined_summary[df_combined_summary["scheme"] == s]["territory_id"]
        assert len(s_terrs) == 224 and s_terrs.nunique() == 224, f"territory_id not unique in scheme {s}"
    print("Each territory_id unique within scheme verified (224 distinct IDs per scheme).")

    # ZIP length
    zip_lens = df_hcp_export["zip_code"].str.len()
    assert (zip_lens == 5).all(), f"Found ZIP codes not 5 characters: {df_hcp_export[zip_lens != 5]['zip_code']}"
    print("All ZIP codes verified to be exactly 5 characters.")

if __name__ == "__main__":
    main()
