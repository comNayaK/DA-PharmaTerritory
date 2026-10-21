# Pharma Commercial HCP Targeting & Territory Alignment

## 1. Business Problem
Commercial pharmaceutical sales planning requires deploying a specialized field force to engage high-value prescribing healthcare providers (HCPs) across the United States mainland. For this illustrative statin portfolio, the commercial objectives are:
- **Prioritize High-Volume Prescribers**: Focus in-person sales representative interactions on top-tier prescribers (Deciles 8–10) while covering lower-volume physicians through non-personal promotion (digital, omni-channel).
- **Equitably Balance Rep Workload**: Establish territory boundaries that respect field capacity constraints (nominal target: 1,200 annual sales calls; operational tolerance band: 960 to 1,440 calls, representing $\pm 20\%$).
- **Ensure Operational Feasibility**: Eliminate territory overlap and preserve postal geography by ensuring that individual ZIP codes are never split across multiple sales representatives.

---

## 2. Data Sources & Vintage Mismatch
The pipeline integrates three primary public datasets:
1. **CMS Medicare Part D Prescribers by Provider and Drug (2023)**:
   - Aggregated prescriber-level 30-day standardized claim fills across statin products.
   - CMS suppresses prescriber-drug rows with $< 11$ claims to protect patient privacy.
2. **Census ZCTA National Gazetteer (2023)**:
   - **Centroids only**: Geographic coordinates (latitude and longitude) of Zip Code Tabulation Area (ZCTA) centroids.
3. **NPPES NPI Registry Snapshot (September 2026)**:
   - **Address fields only**: Primary practice address, city, state, and postal code.
4. **Vintage Mismatch & Geographic Scope**:
   - Prescribing claims from **2023** are joined to provider practice addresses from **September 2026**. Physicians who relocated between 2023 and 2026 reflect their 2026 practice location. Scope is restricted to the continental United States (**48 states + DC**); non-mainland territories (AK, HI, PR, Guam, US Virgin Islands, Military) are excluded to ensure mainland contiguity.

---

## 3. HCP Deciling Methodology
- **Specialty Scope**: Analysis is restricted to three core specialties: Cardiology, Family Practice, and Internal Medicine. (Nurse Practitioners, Physician Assistants, and Endocrinologists are excluded for this baseline).
- **Volume Metric**: Total statin volume is calculated by aggregating standardized 30-day fills across all single-entity statins (Atorvastatin, Rosuvastatin, Simvastatin, Pravastatin, Lovastatin, Pitavastatin, and Fluvastatin). Combination products are excluded by design.
- **Per-Specialty Deciling**: Deciles (1 to 10) are computed independently within each specialty so that PCPs and Cardiologists are evaluated against peers in their respective volume distributions.
- **Targeting Tiers**:
  - **Primary Targets (Deciles 8–10)**: 15,919 HCPs selected for field force territory alignment.
  - **Secondary Targets (Deciles 5–7)**: 26,979 HCPs addressed via non-personal promotion.
  - **Low Priority (Deciles 1–4)**: 138,222 HCPs.

---

## 4. Field Capacity & Workload Model
- **Call Frequency Rules**:
  - Decile 10: 24 calls / year (bi-weekly)
  - Decile 9: 18 calls / year (every 3 weeks)
  - Decile 8: 12 calls / year (monthly)
- **Total Workload**: Across the 15,793 geocoded Primary Targets, the aggregate annual field workload equals **268,320 visits**.
- **Field Capacity Assumption**:
  - The 1,200 calls/rep/year target is an explicit operational planning assumption (5 calls/day $\times$ 240 field days), rather than an empirically measured figure.
  - Total geographic territories: $k = \lceil 268,320 / 1200 \rceil = \mathbf{224\text{ territories}}$.
  - Workload tolerance band: $[960, 1440]$ visits per territory ($1,200 \pm 20\%$) centered on the 1,200-visit assumption.

---

## 5. Alignment Methodology

### Naive Baseline (`src/territory_alignment.py`)
- Direct clustering of individual HCP coordinates on 3D unit-sphere coordinates using `KMeansConstrained`.
- Strictly constrains HCP headcount ($\approx 70 \pm 20\%$ HCPs/territory), but leaves annual call visits unconstrained.
- **Deficiencies**: Wide visit spread (834 to 1,704 calls), only 58.9% of territories within the capacity band, and 131 ZIP codes split across territories (affecting 697 HCPs).

### ZIP-Level Balanced Alignment (`src/balanced_alignment.py`)
A capacitated mathematical programming **heuristic** combining an LP relaxation with an $L_1$ workload evenness penalty, discrete rounding, and local boundary repair:
1. **ZIP Aggregation**: HCPs are aggregated to 5,798 unique 5-digit ZIP codes with weight $w_i = \sum \text{visits}$, HCP count, and ZCTA centroid coordinates.
2. **KMeans Initialization**: 3D unit-sphere coordinates initialized using weighted `sklearn.cluster.KMeans` ($k=224$, `random_state=42`).
3. **Capacitated LP Assignment with Evenness Penalty**: Restricts candidate centroids to the nearest $M$ neighbors and solves a continuous linear program using the HiGHS simplex solver (`scipy.optimize.linprog`). The objective minimizes travel distance plus an auxiliary $L_1$ evenness penalty:
   $$\min_{x, d} \sum_{i,m} w_i \cdot \text{dist}_{i,t} \cdot x_{i,m} + \lambda \sum_{t=1}^k d_t$$
   subject to:
   - $\sum_{m=1}^M x_{i,m} = 1 \quad \forall i$ (each ZIP fully assigned)
   - $960 \le V_t \le 1440 \quad \forall t$ (hard capacity band: $1,200 \pm 20\%$)
   - $d_t \ge |V_t - 1200| \quad \forall t$ (auxiliary $L_1$ deviation from the 1,200-visit assumption, with $\lambda = 1200.0$)
   - $0 \le x_{i,m} \le 1, \quad d_t \ge 0$
   *Selection Rule & Distance Cap*: We swept $\lambda \in [0, 1500]$ and selected $\lambda = 1200.0$ to minimize workload CV subject to a mean HCP-to-centroid distance $\le 46.0\text{ km}$. This 46 km mean-distance cap is an analyst-chosen constraint defining the maximum acceptable travel dispersion.
4. **Centroid Refinement**: Recomputes visit-weighted centroids over 4 iterations until stability.
5. **Rounding & Local-Repair Heuristic**: Fractional ZIP assignments are rounded to their largest share, followed by a deterministic 1-move / 2-swap steepest-descent boundary repair pass ensuring 100% of territories fall strictly within $[960, 1440]$ visits.
6. **Indivisible ZIP Rule**: Every ZIP maps to exactly one territory (0 split ZIPs).

### Map Graph Coloring (`src/generate_tableau_data.py`)
To distinguish adjacent territories on Tableau geographic maps:
- Adjacency is defined by each territory's 8 nearest visit-weighted centroids (symmetrized into an undirected graph).
- Colors are assigned using the DSATUR (Degree of Saturation) heuristic with a maximum of 8 colors.
- **Conflict Results**:
  - **Balanced scheme**: 1 conflict. The symmetrized adjacency graph contains a 9-clique ($\omega(G) = 9$), which makes 0 conflicts mathematically impossible with 8 colors by the Pigeonhole Principle ($\chi(G) \ge 9$). 1 conflict is therefore the proven theoretical minimum.
  - **Naive scheme**: 3 conflicts (near-minimal; optimality not proven).

---

## 6. Alignment Results: Naive Baseline vs. Balanced

| Metric | Naive Baseline | Band-Only | ZIP-Level Balanced ($\lambda=1200$) | Delta / Impact |
|:---|:---:|:---:|:---:|:---|
| **Territories ($k$)** | 224 | 224 | 224 | Identical head count / capacity math |
| **Territories in Band [960, 1440]** | 58.9% (132 / 224) | 100.0% (224 / 224) | **100.0% (224 / 224)** | Hard capacity satisfied |
| **Territories in $\pm 10\%$ [1080, 1320]** | 13.8% (31 / 224) | 8.5% (19 / 224) | **93.8% (210 / 224)** | **+85.3 percentage points** |
| **Territories at Lower Bound (960)** | — | 33 | **0** | No territories trapped at lower bound |
| **Territories at Upper Bound (1440)** | — | 28 | **0** | No territories trapped at upper bound |
| **Territories within 5% of a Bound** | — | 188 | **10** | Massive reduction in boundary bunching |
| **Visit Workload CV** | 0.195 | 0.181 | **0.050** | **CV reduced 74%** |
| **Min Territory Visits** | 834.0 | 960.0 | **966.0** | Min/Max within band |
| **Max Territory Visits** | 1704.0 | 1440.0 | **1434.0** | Min/Max within band |
| **Median Territory Visits** | 1197.0 | 1179.0 | **1200.0** | Median 1,200 |
| **Mean Territory Visits** | 1197.857 | 1197.857 | **1197.857** | Workload perfectly preserved (268,320) |
| **Split ZIP Codes** | 131 | 0 | **0** | Completely eliminated ZIP fragmentation |
| **HCPs in Split ZIPs** | 697 | 0 | **0** | Unambiguous territory assignment |
| **State-Crossing Territories** | 101 | 105 | **91** | — |
| **Mean HCP Distance to Centroid** | 36.66 km | 42.70 km | **44.45 km** | Satisfies $\le 46.0$ km constraint |
| **P90 HCP Distance to Centroid** | 78.72 km | 93.65 km | **96.42 km** | Controlled travel dispersion |
| **Max Territory Radius** | 557.12 km | 602.59 km | **868.39 km** | Rural Mountain West/Plains span |

### Key Trade-Offs & Geographic Dispersion
- **Analyst-Chosen Distance Cap**: The 46.0 km mean HCP-to-centroid distance cap is an analyst-chosen constraint established to balance workload equity against travel burden.
- **Trade-Off of Evenness on Distance and Radius**: Enforcing workload parity with $\lambda = 1200.0$ brought 93.8% of territories within $\pm 10\%$ of the 1,200 target (up from 8.5% in Band-Only), but necessitated trade-offs in geographic dispersion:
  - **Mean HCP distance to centroid** increased from **42.70 km** (Band-Only) to **44.45 km** (+1.75 km increase, or +7.79 km compared to the 36.66 km naive baseline).
  - **Max territory radius** increased from **602.59 km** (Band-Only) to **868.39 km** (+265.80 km increase, or +311.27 km compared to the 557.12 km naive baseline). This expansion occurs predominantly in sparse, low-density Mountain West territories (e.g. T087 covering ID/MT/ND/UT/WA) where reps must span broader geographies to aggregate required visit volume while strictly preserving indivisible ZIP codes.

---

## 7. QA Scorecard
The automated data and roster audit suite ([run_qa.py](run_qa.py) against [sql/04_qa_checks.sql](sql/04_qa_checks.sql)) verifies data integrity:

| Check | Name | Criteria | Result | Status |
|:---|:---|:---|:---:|:---:|
| **QA 1** | Uniqueness & Duplicate Check | Exact match between row & distinct counts | 181,120 = 181,120 | **PASS** |
| **QA 2** | Valid ZIP Match Rate | Valid 5-digit regex match $\ge 98\%$ | 99.26% | **PASS** |
| **QA 3** | ZCTA Match Rate | ZCTA geocoding match $\ge 95\%$ | 99.21% | **PASS** |
| **QA 4** | Staging Drug Audit | All 8 intended statin entities present | 603,514 staging rows | **PASS** |
| **QA 5** | Decile Sanity Check | ~10% volume per decile per specialty | 10.0% $\pm 0.01\%$ | **PASS** |
| **QA 6** | Unassigned Primary Targets | Coordinate match drop rate $< 5\%$ | 0.79% (126 HCPs) | **PASS** |
| **QA 6b** | Primary Target Count Integrity | Roster count equals Mart count | 15,919 = 15,919 | **PASS** |
| **QA 6c** | State Representation | 48 states + DC represented | 49 = 49 | **PASS** |
| **QA 6d** | ZIP Code Indivisibility | No ZIP mapped to $> 1$ territory | 0 split ZIPs | **PASS** |
| **QA 6e** | Workload Band Compliance | All territories in $[960, 1440]$ visits | 0 territories outside | **PASS** |

---

## 8. Model Limitations & Assumptions
1. **Medicare Part D Only**: Claims reflect Medicare beneficiaries aged 65+, not commercial or Medicaid populations. CMS suppresses prescribers with $< 11$ claims.
2. **Specialty Boundary**: Limited to Cardiology, Family Practice, and Internal Medicine. Nurse Practitioners (NPs) and Physician Assistants (PAs) are excluded.
3. **ZCTA Geographic Centroids**: Distances use spherical great-circle distance between Census ZCTA centroids, not actual drive times or road networks.
4. **Unmatched Providers**: 126 Primary Targets (0.79%) could not be geocoded to 2023 Census ZCTAs and remain unassigned.
5. **Vintage Mismatch**: Joining 2023 prescribing data with a September 2026 NPPES snapshot assigns providers to their 2026 practice location.
6. **State-Crossing Boundaries**: Mathematical optimization clusters by travel distance, allowing territories to cross state lines where contiguous prescribers span administrative borders.
7. **Assumed Field Capacity**: The 1,200 annual call benchmark is an operational assumption (5 calls/day $\times$ 240 days), not empirically measured.

---

## 9. Reproduction Steps

### Environment Configuration
Set the mandatory secret salt in your shell (do not commit or echo):
```bash
export HCP_SALT="<alphanumeric-salt-at-least-16-chars>"
```

### Execution Pipeline
Run the commands in order from the repository root:
```bash
# 1. Initialize DuckDB, load raw data, build staging and deciled marts
python load_db.py

# 2. Run naive baseline alignment (KMeansConstrained)
python src/territory_alignment.py

# 3. Run ZIP-level balanced alignment (HiGHS LP + repair heuristic)
python src/balanced_alignment.py

# 4. Execute QA audit checks on balanced national roster
python run_qa.py --roster-table roster_balanced_NATIONAL

# 5. Generate Tableau visualization datasets
python src/generate_tableau_data.py

# 6. (Optional) Run reproducible lambda sweep
python src/sweep_lambda.py
```
