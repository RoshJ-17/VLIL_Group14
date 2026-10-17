
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path("modisPixel_with_GPM.csv")
OUT = Path("eda_results")
OUT.mkdir(exist_ok=True)

CHUNK_SIZE = 500_000

counts = {}
rain_counts = {}
rate_sums = {}
rate_sq_sums = {}
rate_max = {}
month_counts = {}
hour_counts = {}

rows = 0
valid_rows = 0
invalid_time = 0

print("Starting temporal EDA...")

for chunk in pd.read_csv(
    CSV,
    chunksize=CHUNK_SIZE,
    usecols=["timestamp", "GPM_precipitation_rate"],
    low_memory=False
):
    rows += len(chunk)

    # Parse timestamps in the existing YYYYJJJ_HHMM format.
    ts_text = chunk["timestamp"].astype("string")
    ts = pd.to_datetime(
        ts_text,
        format="%Y%j_%H%M",
        errors="coerce"
    )

    invalid_time += int(ts.isna().sum())

    rate = pd.to_numeric(
        chunk["GPM_precipitation_rate"],
        errors="coerce"
    )

    valid = ts.notna() & rate.notna() & np.isfinite(rate) & (rate >= 0)

    frame = pd.DataFrame({
        "timestamp": ts[valid],
        "rate": rate[valid].astype(float)
    })

    if frame.empty:
        continue

    valid_rows += len(frame)
    frame["rain"] = (frame["rate"] >= 0.1).astype("int8")
    frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    frame["hour"] = frame["timestamp"].dt.hour

    for date, group in frame.groupby("date"):
        counts[date] = counts.get(date, 0) + len(group)
        rain_counts[date] = rain_counts.get(date, 0) + int(group["rain"].sum())
        rate_sums[date] = rate_sums.get(date, 0.0) + group["rate"].sum()
        rate_sq_sums[date] = rate_sq_sums.get(date, 0.0) + (group["rate"] ** 2).sum()
        rate_max[date] = max(rate_max.get(date, 0.0), group["rate"].max())

    for month, group in frame.groupby("month"):
        if month not in month_counts:
            month_counts[month] = [0, 0, 0.0]
        month_counts[month][0] += len(group)
        month_counts[month][1] += int(group["rain"].sum())
        month_counts[month][2] += group["rate"].sum()

    for hour, group in frame.groupby("hour"):
        if hour not in hour_counts:
            hour_counts[hour] = [0, 0, 0.0]
        hour_counts[hour][0] += len(group)
        hour_counts[hour][1] += int(group["rain"].sum())
        hour_counts[hour][2] += group["rate"].sum()

    if rows % 5_000_000 < CHUNK_SIZE:
        print(f"Rows scanned: {rows:,}")

daily = pd.DataFrame([
    {
        "date": date,
        "pixel_records": counts[date],
        "rainy_records": rain_counts[date],
        "rainy_percent": 100 * rain_counts[date] / counts[date],
        "mean_rate_mm_hr": rate_sums[date] / counts[date],
        "max_rate_mm_hr": rate_max[date]
    }
    for date in sorted(counts)
])

monthly = pd.DataFrame([
    {
        "month": month,
        "pixel_records": v[0],
        "rainy_records": v[1],
        "rainy_percent": 100 * v[1] / v[0],
        "mean_rate_mm_hr": v[2] / v[0]
    }
    for month, v in sorted(month_counts.items())
])

hourly = pd.DataFrame([
    {
        "hour": hour,
        "pixel_records": v[0],
        "rainy_records": v[1],
        "rainy_percent": 100 * v[1] / v[0],
        "mean_rate_mm_hr": v[2] / v[0]
    }
    for hour, v in sorted(hour_counts.items())
])

daily.to_csv(OUT / "daily_rainfall_summary.csv", index=False)
monthly.to_csv(OUT / "monthly_rainfall_summary.csv", index=False)
hourly.to_csv(OUT / "hourly_rainfall_summary.csv", index=False)

for data, x, y, title, filename in [
    (daily, "date", "rainy_percent", "Daily rainy-pixel percentage", "daily_rainy_percent.png"),
    (monthly, "month", "rainy_percent", "Monthly rainy-pixel percentage", "monthly_rainy_percent.png"),
    (hourly, "hour", "rainy_percent", "Rainy-pixel percentage by UTC hour", "hourly_rainy_percent.png"),
]:
    plt.figure(figsize=(12, 5))
    plt.plot(data[x], data[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel("Rainy pixels (%)")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150)
    plt.close()

print("\n========== TEMPORAL EDA SUMMARY ==========")
print("Rows scanned:", f"{rows:,}")
print("Valid precipitation/time rows:", f"{valid_rows:,}")
print("Invalid timestamps:", f"{invalid_time:,}")
print("Unique dates:", daily["date"].nunique())
print("Unique months:", monthly["month"].nunique())
print("\nMonthly rainfall summary:")
print(monthly.to_string(index=False))
print("\nSaved temporal summaries and plots in:", OUT)
