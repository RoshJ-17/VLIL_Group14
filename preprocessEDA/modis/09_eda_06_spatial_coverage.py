import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = "modisPixel_with_GPM.csv"
OUT = "eda_results"
CHUNKSIZE = 500_000
GRID_DEG = 1.0
RAIN_THRESHOLD = 0.1

os.makedirs(OUT, exist_ok=True)

cols = [
    "timestamp", "lat", "lon", "GPM_precipitation_rate"
]

# Aggregations at grid, month-grid, and timestamp levels.
spatial = {}
monthly_spatial = {}
timestamp_coverage = {}
total_rows = 0
valid_rows = 0

print("Starting spatial coverage EDA...", flush=True)

for chunk in pd.read_csv(INPUT, usecols=cols, chunksize=CHUNKSIZE):
    total_rows += len(chunk)

    lat = pd.to_numeric(chunk["lat"], errors="coerce")
    lon = pd.to_numeric(chunk["lon"], errors="coerce")
    gpm = pd.to_numeric(
        chunk["GPM_precipitation_rate"], errors="coerce"
    )
    ts = pd.to_datetime(
        chunk["timestamp"].astype(str),
        format="%Y%j_%H%M",
        errors="coerce"
    )

    valid = (
        lat.between(-90, 90)
        & lon.between(-180, 180)
        & gpm.notna()
        & (gpm >= 0)
        & ts.notna()
    )

    if not valid.any():
        continue

    d = pd.DataFrame({
        "lat": lat[valid].to_numpy(),
        "lon": lon[valid].to_numpy(),
        "gpm": gpm[valid].to_numpy(),
        "timestamp": ts[valid].to_numpy()
    })

    d["lat_bin"] = np.floor(d["lat"] / GRID_DEG) * GRID_DEG
    d["lon_bin"] = np.floor(d["lon"] / GRID_DEG) * GRID_DEG
    d["month"] = d["timestamp"].dt.strftime("%Y-%m")

    d["rainy"] = (d["gpm"] >= RAIN_THRESHOLD).astype("int64")
    valid_rows += len(d)

    # Overall spatial aggregation.
    g = d.groupby(["lat_bin", "lon_bin"], sort=False).agg(
        records=("gpm", "size"),
        rainy_records=("rainy", "sum"),
        sum_rate=("gpm", "sum")
    )

    for key, row in g.iterrows():
        a = spatial.setdefault(
            key, {"records": 0, "rainy_records": 0, "sum_rate": 0.0}
        )
        a["records"] += int(row["records"])
        a["rainy_records"] += int(row["rainy_records"])
        a["sum_rate"] += float(row["sum_rate"])

    # Monthly spatial aggregation.
    mg = d.groupby(
        ["month", "lat_bin", "lon_bin"], sort=False
    ).agg(
        records=("gpm", "size"),
        rainy_records=("rainy", "sum"),
        sum_rate=("gpm", "sum")
    )

    for key, row in mg.iterrows():
        a = monthly_spatial.setdefault(
            key, {"records": 0, "rainy_records": 0, "sum_rate": 0.0}
        )
        a["records"] += int(row["records"])
        a["rainy_records"] += int(row["rainy_records"])
        a["sum_rate"] += float(row["sum_rate"])

    # Coverage per timestamp.
    tc = d.groupby("timestamp", sort=False).size()
    for key, value in tc.items():
        k = pd.Timestamp(key).strftime("%Y-%m-%d %H:%M")
        timestamp_coverage[k] = timestamp_coverage.get(k, 0) + int(value)

    if total_rows // 5_000_000 != (total_rows - len(chunk)) // 5_000_000:
        print(f"Rows scanned: {total_rows:,}", flush=True)

def to_frame(store, keys, name):
    rows = []
    for key, v in store.items():
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row.update(v)
        row["rainy_percent"] = (
            100 * v["rainy_records"] / v["records"]
            if v["records"] else np.nan
        )
        row["mean_gpm_rate_mm_hr"] = (
            v["sum_rate"] / v["records"]
            if v["records"] else np.nan
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(f"{OUT}/{name}.csv", index=False)
    return df

overall = to_frame(
    spatial, ["lat_bin", "lon_bin"], "spatial_coverage_1deg"
)
monthly = to_frame(
    monthly_spatial, ["month", "lat_bin", "lon_bin"],
    "monthly_spatial_coverage_1deg"
)

tc = pd.DataFrame([
    {"timestamp": k, "records": v}
    for k, v in timestamp_coverage.items()
]).sort_values("timestamp")
tc.to_csv(f"{OUT}/spatial_timestamp_coverage.csv", index=False)

# Coverage map: number of records per grid cell.
if not overall.empty:
    for value_col, filename, title, cmap in [
        ("records", "spatial_record_count.png",
         "Observation count per 1-degree grid cell", "viridis"),
        ("rainy_percent", "spatial_rainy_percent.png",
         "Rainy-record percentage per 1-degree grid cell", "magma"),
        ("mean_gpm_rate_mm_hr", "spatial_mean_gpm_rate.png",
         "Mean GPM rate per 1-degree grid cell (mm/hr)", "plasma")
    ]:
        pivot = overall.pivot(
            index="lat_bin", columns="lon_bin", values=value_col
        ).sort_index()
        plt.figure(figsize=(12, 8))
        plt.imshow(
            pivot.to_numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                float(pivot.columns.min()),
                float(pivot.columns.max() + GRID_DEG),
                float(pivot.index.min()),
                float(pivot.index.max() + GRID_DEG)
            ],
            cmap=cmap
        )
        plt.colorbar(label=value_col)
        plt.xlabel("Longitude (degrees)")
        plt.ylabel("Latitude (degrees)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{OUT}/{filename}", dpi=160)
        plt.close()

# Monthly observation counts by grid cell.
if not monthly.empty:
    count_pivot = monthly.pivot_table(
        index=["lat_bin", "lon_bin"],
        columns="month",
        values="records",
        fill_value=0
    )
    count_pivot.to_csv(f"{OUT}/monthly_spatial_record_matrix.csv")

print("\n========== SPATIAL COVERAGE SUMMARY ==========")
print(f"Rows scanned: {total_rows:,}")
print(f"Valid spatial + timestamp + target rows: {valid_rows:,}")
print(f"Occupied 1-degree grid cells: {len(overall):,}")
print(f"Unique timestamps: {len(timestamp_coverage):,}")

if not overall.empty:
    print("\nMost frequently sampled grid cells:")
    print(
        overall.sort_values("records", ascending=False)
        .head(15)[[
            "lat_bin", "lon_bin", "records",
            "rainy_percent", "mean_gpm_rate_mm_hr"
        ]].to_string(index=False)
    )

    print("\nLeast frequently sampled occupied grid cells:")
    print(
        overall.sort_values("records")
        .head(15)[[
            "lat_bin", "lon_bin", "records",
            "rainy_percent", "mean_gpm_rate_mm_hr"
        ]].to_string(index=False)
    )

print(f"\nOutputs saved in {OUT}/")
