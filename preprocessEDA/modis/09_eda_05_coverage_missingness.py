import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = "modisPixel_with_GPM.csv"
OUT = "eda_results"
CHUNKSIZE = 500_000

os.makedirs(OUT, exist_ok=True)

required = [
    "timestamp", "lat", "lon", "CTT", "CTH",
    "Cloud_Phase", "Cloud_Optical_Thickness",
    "GPM_precipitation_rate"
]

print("Starting coverage and missingness EDA...", flush=True)

# Exact aggregate counts by timestamp, date, month and hour.
timestamp_counts = {}
daily_counts = {}
monthly_counts = {}
hourly_counts = {}

# Retrieval availability and GPM comparisons.
features = [
    "CTT", "CTH", "Cloud_Phase", "Cloud_Optical_Thickness"
]
availability = {
    f: {"available": 0, "missing": 0}
    for f in features
}
availability_by_month = {}
availability_by_hour = {}
cot_by_target = {
    "COT_available": {"n": 0, "rainy": 0, "sum_rate": 0.0},
    "COT_missing": {"n": 0, "rainy": 0, "sum_rate": 0.0},
}
total_rows = 0
valid_target_rows = 0

for chunk in pd.read_csv(
    INPUT, usecols=required, chunksize=CHUNKSIZE
):
    total_rows += len(chunk)

    # Parse the timestamp strings, e.g. 2022182_0340.
    ts = pd.to_datetime(
        chunk["timestamp"].astype(str),
        format="%Y%j_%H%M",
        errors="coerce"
    )
    valid_ts = ts.notna()
    valid_target = (
        pd.to_numeric(chunk["GPM_precipitation_rate"], errors="coerce")
        .notna()
        & (pd.to_numeric(
            chunk["GPM_precipitation_rate"], errors="coerce"
        ) >= 0)
    )
    valid = valid_ts & valid_target

    if valid.any():
        t = ts[valid]
        dates = t.dt.strftime("%Y-%m-%d")
        months = t.dt.strftime("%Y-%m")
        hours = t.dt.hour.astype(str).str.zfill(2)

        for key, counts in [
            (t.dt.strftime("%Y-%m-%d %H:%M"), timestamp_counts),
            (dates, daily_counts),
            (months, monthly_counts),
            (hours, hourly_counts),
        ]:
            vc = key.value_counts()
            for k, v in vc.items():
                counts[k] = counts.get(k, 0) + int(v)

        gpm = pd.to_numeric(
            chunk.loc[valid, "GPM_precipitation_rate"],
            errors="coerce"
        )
        valid_target_rows += int(valid.sum())

        # COT available/missing comparisons with target.
        cot_present = chunk.loc[valid, "Cloud_Optical_Thickness"].notna()
        for label, mask in [
            ("COT_available", cot_present),
            ("COT_missing", ~cot_present),
        ]:
            rates = gpm[mask]
            cot_by_target[label]["n"] += len(rates)
            cot_by_target[label]["rainy"] += int((rates >= 0.1).sum())
            cot_by_target[label]["sum_rate"] += float(rates.sum())

        # Missingness by month and hour for valid timestamp rows.
        frame = chunk.loc[valid_ts, features].copy()
        frame["month"] = ts[valid_ts].dt.strftime("%Y-%m").to_numpy()
        frame["hour"] = (
            ts[valid_ts].dt.hour.astype(str).str.zfill(2).to_numpy()
        )

        for group_col, store in [
            ("month", availability_by_month),
            ("hour", availability_by_hour),
        ]:
            grouped = frame.groupby(group_col, sort=False)
            for group_key, group in grouped:
                d = store.setdefault(
                    str(group_key),
                    {
                        f"{f}_{state}": 0
                        for f in features
                        for state in ("available", "missing")
                    }
                )
                for f in features:
                    d[f + "_available"] += int(group[f].notna().sum())
                    d[f + "_missing"] += int(group[f].isna().sum())

        for f in features:
            availability[f]["available"] += int(
                chunk.loc[valid_ts, f].notna().sum()
            )
            availability[f]["missing"] += int(
                chunk.loc[valid_ts, f].isna().sum()
            )

    if total_rows % 5_000_000 < CHUNKSIZE:
        print(f"Rows scanned: {total_rows:,}", flush=True)

print("\nWriting summaries...", flush=True)

def make_coverage(counts, key_name):
    df = pd.DataFrame(
        [{key_name: k, "pixel_records": v} for k, v in counts.items()]
    )
    if not df.empty:
        df = df.sort_values(key_name)
    return df

make_coverage(timestamp_counts, "timestamp").to_csv(
    f"{OUT}/timestamp_coverage.csv", index=False
)
make_coverage(daily_counts, "date").to_csv(
    f"{OUT}/daily_coverage.csv", index=False
)
make_coverage(monthly_counts, "month").to_csv(
    f"{OUT}/monthly_coverage.csv", index=False
)
make_coverage(hourly_counts, "hour").to_csv(
    f"{OUT}/hourly_coverage.csv", index=False
)

def availability_table(store, group_name):
    rows = []
    for group_key, values in store.items():
        row = {group_name: group_key}
        for f in features:
            a = values[f + "_available"]
            m = values[f + "_missing"]
            row[f + "_available"] = a
            row[f + "_missing"] = m
            row[f + "_missing_pct"] = (
                100.0 * m / (a + m) if a + m else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_name)

availability_table(
    availability_by_month, "month"
).to_csv(f"{OUT}/missingness_by_month.csv", index=False)

availability_table(
    availability_by_hour, "hour"
).to_csv(f"{OUT}/missingness_by_hour.csv", index=False)

overall_rows = []
for f, counts in availability.items():
    n = counts["available"] + counts["missing"]
    overall_rows.append({
        "feature": f,
        "available": counts["available"],
        "missing": counts["missing"],
        "missing_pct": 100.0 * counts["missing"] / n if n else np.nan
    })
pd.DataFrame(overall_rows).to_csv(
    f"{OUT}/retrieval_missingness_overall.csv", index=False
)

cot_rows = []
for label, d in cot_by_target.items():
    n = d["n"]
    cot_rows.append({
        "cot_status": label,
        "records": n,
        "rainy_records": d["rainy"],
        "rainy_percent": 100.0 * d["rainy"] / n if n else np.nan,
        "mean_gpm_rate_mm_hr": d["sum_rate"] / n if n else np.nan
    })
pd.DataFrame(cot_rows).to_csv(
    f"{OUT}/cot_missingness_vs_precipitation.csv", index=False
)

# Plots
for filename, counts, key, title, ylabel in [
    ("daily_coverage.png", daily_counts, "date",
     "Valid pixel records per day", "Pixel records"),
    ("monthly_coverage.png", monthly_counts, "month",
     "Valid pixel records per month", "Pixel records"),
    ("timestamp_coverage.png", timestamp_counts, "timestamp",
     "Valid pixel records per timestamp", "Pixel records"),
]:
    df = make_coverage(counts, key)
    if df.empty:
        continue
    plt.figure(figsize=(14, 5))
    plt.plot(df[key], df["pixel_records"])
    plt.title(title)
    plt.xlabel(key.capitalize())
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{filename}", dpi=150)
    plt.close()

hour_df = availability_table(availability_by_hour, "hour")
plt.figure(figsize=(12, 6))
for f in features:
    plt.plot(
        hour_df["hour"],
        hour_df[f + "_missing_pct"],
        marker="o",
        label=f
    )
plt.title("MODIS feature missingness by UTC hour in timestamp")
plt.xlabel("Hour parsed from timestamp")
plt.ylabel("Missing values (%)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/missingness_by_hour.png", dpi=150)
plt.close()

month_df = availability_table(availability_by_month, "month")
plt.figure(figsize=(14, 6))
for f in features:
    plt.plot(
        month_df["month"],
        month_df[f + "_missing_pct"],
        marker="o",
        label=f
    )
plt.title("MODIS feature missingness by month")
plt.xlabel("Month")
plt.ylabel("Missing values (%)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/missingness_by_month.png", dpi=150)
plt.close()

print("\n========== COVERAGE SUMMARY ==========")
print(f"Rows scanned: {total_rows:,}")
print(f"Valid timestamp + nonnegative GPM rows: {valid_target_rows:,}")
print(f"Unique timestamps: {len(timestamp_counts):,}")
print(f"Unique dates: {len(daily_counts):,}")
print("\nOverall retrieval missingness:")
print(pd.DataFrame(overall_rows).to_string(index=False))
print("\nCOT missingness versus precipitation:")
print(pd.DataFrame(cot_rows).to_string(index=False))
print("\nMonthly coverage:")
print(make_coverage(monthly_counts, "month").to_string(index=False))
print(f"\nOutputs saved in {OUT}/")

