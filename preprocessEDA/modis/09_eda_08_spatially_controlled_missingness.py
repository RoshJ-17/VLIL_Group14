import os
import numpy as np
import pandas as pd

INPUT = "modisPixel_with_GPM.csv"
OUT = "eda_results"
CHUNKSIZE = 500_000
GRID_DEG = 1.0
RAIN_THRESHOLD = 0.1

# Minimum records in EACH availability group for the primary comparison.
MIN_GROUP_N = 100

os.makedirs(OUT, exist_ok=True)

usecols = [
    "timestamp", "lat", "lon", "CTH", "CTT",
    "GPM_precipitation_rate"
]

# Key: (month, lat_bin, lon_bin, availability_status)
# Values: [records, rainy_records, sum_GPM_rate]
agg = {}
rows_scanned = 0
valid_rows = 0

print("Starting spatially controlled missingness analysis...", flush=True)

for chunk in pd.read_csv(INPUT, usecols=usecols, chunksize=CHUNKSIZE):
    rows_scanned += len(chunk)

    ts = pd.to_datetime(
        chunk["timestamp"].astype(str),
        format="%Y%j_%H%M",
        errors="coerce"
    )
    lat = pd.to_numeric(chunk["lat"], errors="coerce")
    lon = pd.to_numeric(chunk["lon"], errors="coerce")
    gpm = pd.to_numeric(chunk["GPM_precipitation_rate"], errors="coerce")
    cth = pd.to_numeric(chunk["CTH"], errors="coerce")
    ctt = pd.to_numeric(chunk["CTT"], errors="coerce")

    valid = (
        ts.notna()
        & lat.between(-90, 90)
        & lon.between(-180, 180)
        & gpm.notna()
        & (gpm >= 0)
    )

    if not valid.any():
        continue

    d = pd.DataFrame({
        "month": ts.loc[valid].dt.strftime("%Y-%m").to_numpy(),
        "lat": lat.loc[valid].to_numpy(),
        "lon": lon.loc[valid].to_numpy(),
        "gpm": gpm.loc[valid].to_numpy(),
        "cth_missing": cth.loc[valid].isna().to_numpy(),
        "ctt_missing": ctt.loc[valid].isna().to_numpy(),
    })

    d["lat_bin"] = np.floor(d["lat"] / GRID_DEG) * GRID_DEG
    d["lon_bin"] = np.floor(d["lon"] / GRID_DEG) * GRID_DEG

    # Earlier EDA found identical masks. Keep an explicit status check.
    d["status"] = np.select(
        [
            ~d["cth_missing"] & ~d["ctt_missing"],
            d["cth_missing"] & d["ctt_missing"],
            d["cth_missing"] & ~d["ctt_missing"],
            ~d["cth_missing"] & d["ctt_missing"],
        ],
        [
            "both_present",
            "both_missing",
            "cth_missing_only",
            "ctt_missing_only",
        ],
        default="unknown"
    )
    d["rainy"] = (d["gpm"] >= RAIN_THRESHOLD).astype("int64")

    grouped = d.groupby(
        ["month", "lat_bin", "lon_bin", "status"],
        sort=False
    ).agg(
        records=("gpm", "size"),
        rainy_records=("rainy", "sum"),
        sum_rate=("gpm", "sum")
    )

    for key, row in grouped.iterrows():
        a = agg.setdefault(key, [0, 0, 0.0])
        a[0] += int(row["records"])
        a[1] += int(row["rainy_records"])
        a[2] += float(row["sum_rate"])

    valid_rows += len(d)

    if rows_scanned // 5_000_000 != (rows_scanned - len(chunk)) // 5_000_000:
        print(f"Rows scanned: {rows_scanned:,}", flush=True)

# Convert accumulated aggregates to a table.
records = []
for key, values in agg.items():
    month, lat_bin, lon_bin, status = key
    n, rainy_n, rate_sum = values
    records.append({
        "month": month,
        "lat_bin": lat_bin,
        "lon_bin": lon_bin,
        "status": status,
        "records": n,
        "rainy_records": rainy_n,
        "rainy_percent": 100 * rainy_n / n if n else np.nan,
        "mean_gpm_rate_mm_hr": rate_sum / n if n else np.nan,
    })

df = pd.DataFrame(records)
df.to_csv(
    f"{OUT}/cth_ctt_grid_month_status.csv", index=False
)

# Pair the two statuses within each exact grid cell and month.
present = df[df["status"] == "both_present"].copy()
missing = df[df["status"] == "both_missing"].copy()

keys = ["month", "lat_bin", "lon_bin"]
present = present.rename(columns={
    "records": "n_present",
    "rainy_records": "rainy_present",
    "rainy_percent": "rainy_pct_present",
    "mean_gpm_rate_mm_hr": "mean_rate_present",
})
missing = missing.rename(columns={
    "records": "n_missing",
    "rainy_records": "rainy_missing",
    "rainy_percent": "rainy_pct_missing",
    "mean_gpm_rate_mm_hr": "mean_rate_missing",
})

paired = present[keys + [
    "n_present", "rainy_present", "rainy_pct_present", "mean_rate_present"
]].merge(
    missing[keys + [
        "n_missing", "rainy_missing", "rainy_pct_missing", "mean_rate_missing"
    ]],
    on=keys,
    how="inner"
)

paired["rainy_pct_difference_pp"] = (
    paired["rainy_pct_present"] - paired["rainy_pct_missing"]
)
paired["mean_rate_difference_mm_hr"] = (
    paired["mean_rate_present"] - paired["mean_rate_missing"]
)
paired["min_group_n"] = paired[["n_present", "n_missing"]].min(axis=1)

paired.to_csv(
    f"{OUT}/cth_ctt_paired_grid_month_comparison.csv", index=False
)

# Primary comparison uses only grid-month groups with >= MIN_GROUP_N
# observations in BOTH availability groups.
eligible = paired[paired["min_group_n"] >= MIN_GROUP_N].copy()
eligible.to_csv(
    f"{OUT}/cth_ctt_paired_grid_month_comparison_eligible.csv",
    index=False
)

# A transparent pooled comparison within eligible grid-month strata:
# aggregate the records separately across only the eligible strata.
eligible_keys = eligible[keys]
eligible_status = df.merge(eligible_keys, on=keys, how="inner")

pooled_rows = []
for status in ["both_present", "both_missing"]:
    sub = eligible_status[eligible_status["status"] == status]
    n = int(sub["records"].sum())
    rainy_n = int(sub["rainy_records"].sum())
    rate_sum = float((sub["mean_gpm_rate_mm_hr"] * sub["records"]).sum())
    pooled_rows.append({
        "status": status,
        "eligible_records": n,
        "rainy_records": rainy_n,
        "rainy_percent": 100 * rainy_n / n if n else np.nan,
        "mean_gpm_rate_mm_hr": rate_sum / n if n else np.nan,
    })

pd.DataFrame(pooled_rows).to_csv(
    f"{OUT}/cth_ctt_spatially_controlled_pooled_summary.csv",
    index=False
)

# Equal-weight average of within-stratum differences. This gives each
# eligible grid-month equal weight, regardless of its record count.
if len(eligible):
    equal_weight_summary = {
        "eligible_grid_month_strata": len(eligible),
        "mean_stratum_rainy_pct_difference_pp":
            eligible["rainy_pct_difference_pp"].mean(),
        "median_stratum_rainy_pct_difference_pp":
            eligible["rainy_pct_difference_pp"].median(),
        "mean_stratum_rate_difference_mm_hr":
            eligible["mean_rate_difference_mm_hr"].mean(),
        "median_stratum_rate_difference_mm_hr":
            eligible["mean_rate_difference_mm_hr"].median(),
    }
else:
    equal_weight_summary = {
        "eligible_grid_month_strata": 0,
        "mean_stratum_rainy_pct_difference_pp": np.nan,
        "median_stratum_rainy_pct_difference_pp": np.nan,
        "mean_stratum_rate_difference_mm_hr": np.nan,
        "median_stratum_rate_difference_mm_hr": np.nan,
    }

pd.DataFrame([equal_weight_summary]).to_csv(
    f"{OUT}/cth_ctt_equal_weight_stratum_summary.csv", index=False
)

# Report mask mismatches, including any unexpected groups.
print("\n========== SPATIALLY CONTROLLED MISSINGNESS ==========")
print(f"Rows scanned: {rows_scanned:,}")
print(f"Valid rows: {valid_rows:,}")
print(f"Grid-month-status groups: {len(df):,}")
print(f"Grid-months with both availability states: {len(paired):,}")
print(
    f"Grid-months with at least {MIN_GROUP_N} rows per state: "
    f"{len(eligible):,}"
)

print("\nStatus totals:")
print(
    df.groupby("status")["records"].sum()
    .sort_values(ascending=False).to_string()
)

print("\nPooled results within eligible grid-month strata:")
print(pd.DataFrame(pooled_rows).to_string(index=False))

print("\nEqual-weight differences across eligible grid-month strata:")
print(pd.DataFrame([equal_weight_summary]).to_string(index=False))

print(f"\nOutputs saved in {OUT}/")
