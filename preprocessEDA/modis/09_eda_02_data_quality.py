
from pathlib import Path
import pandas as pd
import numpy as np
import json

CSV = Path("modisPixel_with_GPM.csv")
OUT = Path("eda_results")
OUT.mkdir(exist_ok=True)

CHUNK_SIZE = 500_000

numeric_cols = [
    "lat", "lon", "BT_11", "BT_12", "BTD",
    "Cloud_Mask_Binary", "Cloud_Mask_Raw",
    "CTT", "CTH", "Cloud_Phase",
    "Cloud_Optical_Thickness",
    "GPM_precipitation_rate",
    "GPM_time_diff_minutes"
]

count = 0
missing = {}
invalid_numeric = {}
minimum = {}
maximum = {}
negative_counts = {}
duplicate_full_rows = 0
status_counts = {}
rainy = 0
non_rainy = 0
valid_target = 0
feature_stats = {}
seen_keys = set()

print("Starting chunked data-quality audit...")

for chunk in pd.read_csv(CSV, chunksize=CHUNK_SIZE, low_memory=False):
    n = len(chunk)
    count += n

    for col in chunk.columns:
        missing[col] = missing.get(col, 0) + int(chunk[col].isna().sum())

    # Check numeric conversion and range summaries.
    for col in numeric_cols:
        if col not in chunk:
            continue

        x = pd.to_numeric(chunk[col], errors="coerce")
        invalid_numeric[col] = invalid_numeric.get(col, 0) + int(
            (x.isna() & chunk[col].notna()).sum()
        )
        xvalid = x.dropna()

        if len(xvalid):
            minimum[col] = min(minimum.get(col, float("inf")), float(xvalid.min()))
            maximum[col] = max(maximum.get(col, float("-inf")), float(xvalid.max()))

        if col in ["CTH", "Cloud_Optical_Thickness", "GPM_precipitation_rate"]:
            negative_counts[col] = negative_counts.get(col, 0) + int((x < 0).sum())

    if "GPM_precipitation_rate" in chunk:
        gpm = pd.to_numeric(chunk["GPM_precipitation_rate"], errors="coerce")
        valid = gpm.notna() & np.isfinite(gpm) & (gpm >= 0)
        valid_target += int(valid.sum())
        rainy += int((valid & (gpm >= 0.1)).sum())
        non_rainy += int((valid & (gpm < 0.1)).sum())

    if "GPM_match_status" in chunk:
        vals = chunk["GPM_match_status"].fillna("<MISSING>").astype(str).value_counts()
        for key, value in vals.items():
            status_counts[key] = status_counts.get(key, 0) + int(value)

    # Exact duplicate rows within each chunk.
    duplicate_full_rows += int(chunk.duplicated().sum())

    if count % 5_000_000 < CHUNK_SIZE:
        print(f"Rows scanned: {count:,}")

report = {
    "rows_scanned": count,
    "missing_values_by_column": missing,
    "numeric_conversion_failures": invalid_numeric,
    "numeric_minimum": minimum,
    "numeric_maximum": maximum,
    "negative_value_counts": negative_counts,
    "exact_duplicate_rows_within_chunks": duplicate_full_rows,
    "GPM_match_status_counts": status_counts,
    "valid_GPM_targets": valid_target,
    "rainy_pixels_threshold_0.1": rainy,
    "non_rainy_pixels_threshold_0.1": non_rainy,
    "rainy_percentage": round(100 * rainy / valid_target, 6) if valid_target else None,
    "non_rainy_percentage": round(100 * non_rainy / valid_target, 6) if valid_target else None,
}

with open(OUT / "data_quality_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n========== DATA QUALITY SUMMARY ==========")
print("Rows:", f"{count:,}")
print("\nMissing values:")
for col, value in missing.items():
    print(f"  {col}: {value:,}")

print("\nNumeric ranges:")
for col in minimum:
    print(f"  {col}: {minimum[col]} to {maximum[col]}")

print("\nGPM target distribution at 0.1 mm/hr:")
print(f"  Rain: {rainy:,} ({report['rainy_percentage']}%)")
print(f"  No rain: {non_rainy:,} ({report['non_rainy_percentage']}%)")
print("\nGPM match statuses:", status_counts)
print("Exact duplicate rows detected within chunks:", duplicate_full_rows)
print("\nReport saved to:", OUT / "data_quality_report.json")

