import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = "modisPixel_with_GPM.csv"
OUT = "eda_results"
CHUNKSIZE = 500_000
RAIN_THRESHOLD = 0.1
GRID_DEG = 1.0

os.makedirs(OUT, exist_ok=True)

cols = [
    "timestamp", "lat", "lon", "CTH", "CTT",
    "GPM_precipitation_rate"
]

total_rows = 0
valid_target_rows = 0
both_present = 0
both_missing = 0
cth_only_missing = 0
ctt_only_missing = 0

# Group summaries: count, rainy count, sum of GPM rates.
groups = {
    "both_present": [0, 0, 0.0],
    "both_missing": [0, 0, 0.0],
    "cth_missing_only": [0, 0, 0.0],
    "ctt_missing_only": [0, 0, 0.0],
}

by_month = {}
by_hour = {}
by_grid = {}

def add_summary(store, key, mask, gpm, rainy):
    if not mask.any():
        return
    n = int(mask.sum())
    nr = int(rainy[mask].sum())
    sr = float(gpm[mask].sum())
    a = store.setdefault(key, [0, 0, 0.0])
    a[0] += n
    a[1] += nr
    a[2] += sr

print("Starting CTH/CTT missingness analysis...", flush=True)

for chunk in pd.read_csv(INPUT, usecols=cols, chunksize=CHUNKSIZE):
    total_rows += len(chunk)

    cth = pd.to_numeric(chunk["CTH"], errors="coerce")
    ctt = pd.to_numeric(chunk["CTT"], errors="coerce")
    gpm = pd.to_numeric(
        chunk["GPM_precipitation_rate"], errors="coerce"
    )
    ts = pd.to_datetime(
        chunk["timestamp"].astype(str),
        format="%Y%j_%H%M",
        errors="coerce"
    )
    lat = pd.to_numeric(chunk["lat"], errors="coerce")
    lon = pd.to_numeric(chunk["lon"], errors="coerce")

    valid = (
        ts.notna()
        & gpm.notna()
        & (gpm >= 0)
        & lat.between(-90, 90)
        & lon.between(-180, 180)
    )
    if not valid.any():
        continue

    valid_target_rows += int(valid.sum())

    cth_miss = cth.isna()
    ctt_miss = ctt.isna()

    bp = valid & ~cth_miss & ~ctt_miss
    bm = valid & cth_miss & ctt_miss
    co = valid & cth_miss & ~ctt_miss
    to = valid & ~cth_miss & ctt_miss

    both_present += int(bp.sum())
    both_missing += int(bm.sum())
    cth_only_missing += int(co.sum())
    ctt_only_missing += int(to.sum())

    rates = gpm.to_numpy()
    rainy = (gpm >= RAIN_THRESHOLD).fillna(False).to_numpy()

    for label, mask in [
        ("both_present", bp),
        ("both_missing", bm),
        ("cth_missing_only", co),
        ("ctt_missing_only", to),
    ]:
        add_summary(groups, label, mask.to_numpy(), rates, rainy)

    # Aggregate by month, hour, and grid for joint missingness patterns.
    d = pd.DataFrame({
        "month": ts[valid].dt.strftime("%Y-%m").to_numpy(),
        "hour": ts[valid].dt.hour.astype(str).str.zfill(2).to_numpy(),
        "lat": lat[valid].to_numpy(),
        "lon": lon[valid].to_numpy(),
        "gpm": gpm[valid].to_numpy(),
        "cth_missing": cth_miss[valid].to_numpy(),
        "ctt_missing": ctt_miss[valid].to_numpy(),
    })

    d["lat_bin"] = np.floor(d["lat"] / GRID_DEG) * GRID_DEG
    d["lon_bin"] = np.floor(d["lon"] / GRID_DEG) * GRID_DEG
    d["joint_status"] = np.select(
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
    d["rainy"] = (d["gpm"] >= RAIN_THRESHOLD).astype(int)

    for group_col, store in [
        ("month", by_month),
        ("hour", by_hour),
        ("lat_bin", None),
    ]:
        if group_col == "lat_bin":
            # Geographic grid aggregation below uses both coordinates.
            continue
        for (group_key, status), grp in d.groupby(
            [group_col, "joint_status"], sort=False
        ):
            key = (str(group_key), str(status))
            a = store.setdefault(key, [0, 0, 0.0])
            a[0] += len(grp)
            a[1] += int(grp["rainy"].sum())
            a[2] += float(grp["gpm"].sum())

    for (la, lo, status), grp in d.groupby(
        ["lat_bin", "lon_bin", "joint_status"], sort=False
    ):
        key = (float(la), float(lo), str(status))
        a = by_grid.setdefault(key, [0, 0, 0.0])
        a[0] += len(grp)
        a[1] += int(grp["rainy"].sum())
        a[2] += float(grp["gpm"].sum())

    if total_rows // 5_000_000 != (total_rows - len(chunk)) // 5_000_000:
        print(f"Rows scanned: {total_rows:,}", flush=True)

def summary_frame(store, keys):
    rows = []
    for key, v in store.items():
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        n, rainy_n, rate_sum = v
        row.update({
            "records": n,
            "rainy_records": rainy_n,
            "rainy_percent": 100 * rainy_n / n if n else np.nan,
            "mean_gpm_rate_mm_hr": rate_sum / n if n else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)

pd.DataFrame([
    {"joint_status": k, "records": v[0],
     "rainy_records": v[1],
     "rainy_percent": 100 * v[1] / v[0] if v[0] else np.nan,
     "mean_gpm_rate_mm_hr": v[2] / v[0] if v[0] else np.nan}
    for k, v in groups.items()
]).to_csv(f"{OUT}/cth_ctt_joint_status_summary.csv", index=False)

summary_frame(
    by_month, ["month", "joint_status"]
).to_csv(f"{OUT}/cth_ctt_missingness_by_month.csv", index=False)

summary_frame(
    by_hour, ["hour", "joint_status"]
).to_csv(f"{OUT}/cth_ctt_missingness_by_hour.csv", index=False)

grid_df = summary_frame(
    by_grid, ["lat_bin", "lon_bin", "joint_status"]
)
grid_df.to_csv(f"{OUT}/cth_ctt_missingness_by_grid.csv", index=False)

# Overall exact mask agreement across valid rows.
mask_rows = [
    ("both_present", both_present),
    ("both_missing", both_missing),
    ("cth_missing_only", cth_only_missing),
    ("ctt_missing_only", ctt_only_missing),
]
mask_df = pd.DataFrame(mask_rows, columns=["joint_status", "records"])
mask_df["percent_of_valid_rows"] = (
    100 * mask_df["records"] / valid_target_rows
)
mask_df.to_csv(f"{OUT}/cth_ctt_mask_agreement.csv", index=False)

# Plot monthly missing percentages for each feature.
m = summary_frame(by_month, ["month", "joint_status"])
if not m.empty:
    wide = m.pivot(
        index="month", columns="joint_status", values="records"
    ).fillna(0)
    wide["total"] = wide.sum(axis=1)
    for status in ["both_present", "both_missing",
                   "cth_missing_only", "ctt_missing_only"]:
        if status not in wide:
            wide[status] = 0
    wide["both_missing_pct"] = 100 * wide["both_missing"] / wide["total"]
    wide["both_present_pct"] = 100 * wide["both_present"] / wide["total"]
    wide = wide.sort_index()

    plt.figure(figsize=(13, 5))
    plt.plot(wide.index, wide["both_missing_pct"], marker="o",
             label="Both CTH and CTT missing")
    plt.plot(wide.index, wide["both_present_pct"], marker="o",
             label="Both present")
    plt.ylabel("Percentage of valid records")
    plt.xlabel("Month")
    plt.title("Joint CTH/CTT availability by month")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/cth_ctt_missingness_by_month.png", dpi=150)
    plt.close()

# Map fraction with both CTH and CTT missing per grid cell.
if not grid_df.empty:
    pivot = grid_df.pivot_table(
        index=["lat_bin", "lon_bin"],
        columns="joint_status",
        values="records",
        aggfunc="sum",
        fill_value=0
    )
    total = pivot.sum(axis=1)
    both = pivot.get("both_missing", pd.Series(0, index=pivot.index))
    pct = (100 * both / total).unstack("lon_bin").sort_index()

    plt.figure(figsize=(12, 8))
    plt.imshow(
        pct.to_numpy(), origin="lower", aspect="auto",
        interpolation="nearest",
        extent=[
            float(pct.columns.min()), float(pct.columns.max() + GRID_DEG),
            float(pct.index.min()), float(pct.index.max() + GRID_DEG)
        ],
        vmin=0, vmax=100, cmap="magma"
    )
    plt.colorbar(label="Both CTH and CTT missing (%)")
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.title("Joint CTH/CTT missingness by 1-degree grid cell")
    plt.tight_layout()
    plt.savefig(f"{OUT}/cth_ctt_missingness_spatial.png", dpi=150)
    plt.close()

print("\n========== CTH/CTT MISSINGNESS SUMMARY ==========")
print(f"Rows scanned: {total_rows:,}")
print(f"Valid target/time/coordinate rows: {valid_target_rows:,}")
print("\nJoint missingness counts:")
print(mask_df.to_string(index=False))
print("\nPrecipitation by joint availability status:")
print(pd.read_csv(
    f"{OUT}/cth_ctt_joint_status_summary.csv"
).to_string(index=False))
print(f"\nOutputs saved in {OUT}/")
