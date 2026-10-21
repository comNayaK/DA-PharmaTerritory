import os
import time
import math
import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog
from sklearn.cluster import KMeans

def compute_distances_km(xyz, centroids):
    """Compute great-circle spherical distance in kilometers between 3D unit coordinates."""
    dots = np.clip(np.dot(xyz, centroids.T), -1.0, 1.0)
    return 6371.0 * np.arccos(dots)

def solve_lp_with_evenness(xyz, weights, centroids, lam, initial_M=10, max_M=60):
    """
    Solve the capacitated LP relaxation with an L1 workload evenness penalty term.

    Minimizes total visit-weighted travel distance plus lambda * sum(d_t), where
    d_t >= |V_t - 1200| measures absolute workload deviation from the 1,200 nominal target.
    Candidate centroids are restricted to the nearest M neighbors for solver efficiency.
    """
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

        c[n_assign_vars:] = lam
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
            return res.x[:n_assign_vars].reshape((n_zips, M)), cand, dist_km
        M += 5
    raise RuntimeError(f"LP infeasible even at M={max_M}")

def repair_pass(assign, weights, dist_km, cand_search, k, max_passes=250):
    """
    Steepest-descent heuristic to resolve any capacity band violations after rounding.

    Evaluates 1-moves (reassigning a boundary ZIP) and 2-swaps between unbalanced territories,
    prioritizing moves that minimize increase in centroid distance until all territories fall strictly
    within [960, 1440] annual visits.
    """
    n_zips = len(weights)
    v = np.bincount(assign, weights=weights, minlength=k).astype(float)

    def calc_violation(visits):
        return np.sum(np.maximum(0, 960.0 - visits)) + np.sum(np.maximum(0, visits - 1440.0))

    if calc_violation(v) == 0:
        return assign, v

    n_cands = cand_search.shape[1]
    for _ in range(max_passes):
        if calc_violation(v) == 0:
            break
        best_move = None
        best_delta = 0.0
        best_penalty = float('inf')

        surplus = set(np.where(v > 1440.0)[0])
        deficit = set(np.where(v < 960.0)[0])

        for i in range(n_zips):
            src = assign[i]
            w = weights[i]
            if src not in surplus and len(deficit.intersection(cand_search[i])) == 0:
                continue
            for m in range(n_cands):
                dst = cand_search[i, m]
                if dst == src:
                    continue
                old_viol = (max(0.0, 960.0 - v[src]) + max(0.0, v[src] - 1440.0) +
                            max(0.0, 960.0 - v[dst]) + max(0.0, v[dst] - 1440.0))
                new_src = v[src] - w
                new_dst = v[dst] + w
                new_viol = (max(0.0, 960.0 - new_src) + max(0.0, new_src - 1440.0) +
                            max(0.0, 960.0 - new_dst) + max(0.0, new_dst - 1440.0))
                delta = new_viol - old_viol
                if delta < 0:
                    pen = dist_km[i, dst] - dist_km[i, src]
                    if delta < best_delta or (delta == best_delta and pen < best_penalty):
                        best_delta = delta; best_penalty = pen
                        best_move = (i, src, dst, w)
        if best_move is not None:
            i, src, dst, w = best_move
            assign[i] = dst
            v[src] -= w; v[dst] += w
            continue

        best_swap = None
        best_swap_delta = 0.0
        best_swap_pen = float('inf')
        unb = list(surplus | deficit)
        for u in unb:
            for z1 in np.where(assign == u)[0]:
                w1 = weights[z1]
                for m in range(min(15, n_cands)):
                    t2 = cand_search[z1, m]
                    if t2 == u:
                        continue
                    for z2 in np.where(assign == t2)[0]:
                        w2 = weights[z2]
                        if w1 == w2 or u not in cand_search[z2]:
                            continue
                        old_viol = (max(0.0, 960.0 - v[u]) + max(0.0, v[u] - 1440.0) +
                                    max(0.0, 960.0 - v[t2]) + max(0.0, v[t2] - 1440.0))
                        new_u = v[u] - w1 + w2
                        new_t2 = v[t2] - w2 + w1
                        new_viol = (max(0.0, 960.0 - new_u) + max(0.0, new_u - 1440.0) +
                                    max(0.0, 960.0 - new_t2) + max(0.0, new_t2 - 1440.0))
                        delta = new_viol - old_viol
                        if delta < 0:
                            pen = (dist_km[z1, t2] - dist_km[z1, u]) + (dist_km[z2, u] - dist_km[z2, t2])
                            if delta < best_swap_delta or (delta == best_swap_delta and pen < best_swap_pen):
                                best_swap_delta = delta; best_swap_pen = pen
                                best_swap = (z1, u, z2, t2)
        if best_swap is not None:
            z1, u, z2, t2 = best_swap
            assign[z1] = t2; assign[z2] = u
            v[u] = v[u] - weights[z1] + weights[z2]
            v[t2] = v[t2] - weights[z2] + weights[z1]
            continue
        break
    return assign, v

def run_single_lambda(xyz, weights, init_centroids, df_hcp_coords, hcp_w, hcp_zip_idx, k, lam):
    t0 = time.time()
    centroids = init_centroids.copy()

    for _ in range(4):
        x, cand, dist_km = solve_lp_with_evenness(xyz, weights, centroids, lam, initial_M=10, max_M=60)
        new_centroids = np.zeros((k, 3))
        M = cand.shape[1]
        for i in range(len(weights)):
            w = weights[i]
            for m in range(M):
                val = x[i, m]
                if val > 1e-6:
                    t = cand[i, m]
                    new_centroids[t] += (w * val) * xyz[i]
        norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
        centroids = new_centroids / np.maximum(norms, 1e-9)

    zip_assign = np.array([cand[i, np.argmax(x[i])] for i in range(len(weights))], dtype=int)
    cand_search = np.argsort(dist_km, axis=1)[:, :20]
    zip_assign, v = repair_pass(zip_assign, weights, dist_km, cand_search, k)

    # Centroids and distance metrics
    hcp_terrs = zip_assign[hcp_zip_idx]
    final_centroids = np.zeros((k, 3))
    for t in range(k):
        pts = df_hcp_coords[hcp_terrs == t]
        pw = hcp_w[hcp_terrs == t]
        if len(pts) > 0:
            c = np.sum(pts * pw[:, None], axis=0)
            final_centroids[t] = c / np.maximum(np.linalg.norm(c), 1e-9)

    dots = np.clip(np.sum(df_hcp_coords * final_centroids[hcp_terrs], axis=1), -1.0, 1.0)
    dists = 6371.0 * np.arccos(dots)

    cv = float(v.std() / v.mean())
    pct_1080_1320 = float(np.mean((v >= 1080.0) & (v <= 1320.0)) * 100.0)
    n_lower = int((v == 960.0).sum())
    n_upper = int((v == 1440.0).sum())
    mean_dist = float(np.mean(dists))
    p90_dist = float(np.percentile(dists, 90))
    max_rad = float(np.max(dists))
    runtime = time.time() - t0

    return {
        "lambda": lam,
        "pct_within_1080_1320": round(pct_1080_1320, 1),
        "cv": round(cv, 3),
        "visits_min": float(v.min()),
        "visits_max": float(v.max()),
        "n_lower_bound": n_lower,
        "n_upper_bound": n_upper,
        "mean_hcp_dist_km": round(mean_dist, 2),
        "p90_hcp_dist_km": round(p90_dist, 2),
        "max_radius_km": round(max_rad, 2),
        "runtime_sec": round(runtime, 2),
        "feasible_le_46km": "Yes" if mean_dist <= 46.0 else "No"
    }

def main():
    print("Loading data for lambda sweep...")
    con = duckdb.connect("pharma_commercial.duckdb", read_only=True)
    df = con.execute("""
        SELECT d.hcp_id, d.decile, d.state, d.zip_code, z.latitude, z.longitude 
        FROM mart_hcp_deciled d
        JOIN zip_lat_lon z ON d.zip_code = z.zip_code
        WHERE d.target_status = 'Primary Target' AND z.latitude IS NOT NULL
    """).df()
    con.close()

    FREQ = {10: 24, 9: 18, 8: 12}
    df["visits"] = df["decile"].map(FREQ).fillna(0)

    df_zip = df.groupby("zip_code").agg(
        weight=("visits", "sum"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first")
    ).reset_index()

    n_zips = len(df_zip)
    total_visits = df_zip["weight"].sum()
    k = math.ceil(total_visits / 1200)

    lat = np.radians(df_zip["latitude"].values)
    lon = np.radians(df_zip["longitude"].values)
    xyz = np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    weights = df_zip["weight"].values.astype(float)

    zip_to_idx = {z: i for i, z in enumerate(df_zip["zip_code"])}
    hcp_zip_idx = df["zip_code"].map(zip_to_idx).values
    hcp_lat = np.radians(df["latitude"].values)
    hcp_lon = np.radians(df["longitude"].values)
    df_hcp_coords = np.column_stack([np.cos(hcp_lat)*np.cos(hcp_lon), np.cos(hcp_lat)*np.sin(hcp_lon), np.sin(hcp_lat)])
    hcp_w = df["visits"].values

    print(f"Aggregated {len(df)} HCPs into {n_zips} ZIPs. Initializing base KMeans (k={k})...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(xyz, sample_weight=weights)
    init_centroids = km.cluster_centers_
    norms = np.linalg.norm(init_centroids, axis=1, keepdims=True)
    init_centroids = init_centroids / np.maximum(norms, 1e-9)

    lambdas = [0.0, 50.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1200.0, 1300.0, 1400.0, 1500.0]
    results = []

    print(f"Sweeping lambda across {len(lambdas)} values...")
    for lam in lambdas:
        res = run_single_lambda(xyz, weights, init_centroids, df_hcp_coords, hcp_w, hcp_zip_idx, k, lam)
        results.append(res)
        print(f"  lambda={lam:>6.1f} | % in ±10%: {res['pct_within_1080_1320']:>5.1f}% | CV: {res['cv']:.3f} | Mean Dist: {res['mean_hcp_dist_km']:>5.2f} km | Feasible: {res['feasible_le_46km']}")

    df_res = pd.DataFrame(results)
    os.makedirs("metrics", exist_ok=True)
    out_csv = "metrics/lambda_sweep.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nLambda sweep completed successfully. Output saved to {out_csv}.")

if __name__ == "__main__":
    main()
