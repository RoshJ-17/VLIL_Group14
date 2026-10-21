
import os
import pandas as pd
import numpy as np

INPUT_CSV = "modisPixel_with_GPM.csv"
OUTPUT_DIR = "eda_results"
CHUNK_SIZE = 500_000
N_EXTREMES = 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

cols = [
    "timestamp", "lat", "lon",
    "BT_11", "BT_12", "BTD",
    "CTT", "CTH", "Cloud_Phase",
    "Cloud_Mask_Binary", "Cloud_Mask_Raw",
    "GPM_precipitation_rate",
]

numeric_cols = [
    "BT_11", "BT_12", "BTD", "CTT", "CTH",
    "GPM_precipitation_rate",
]

top_parts = {c: [] for c in numeric_cols}
bottom_parts = {c: [] for c in numeric_cols}
rows_scanned = 0

print("Scanning for extreme records...", flush=True)

for chunk in pd.read_csv(
    INPUT_CSV,
    usecols=cols,
    chunksize=CHUNK_SIZE,
):
    rows_scanned += len(chunk)

    for col in numeric_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        valid = chunk.loc[
            np.isfinite(chunk[col]),
            cols
        ]

        if valid.empty:
            continue

        top_parts[col].append(
            valid.nlargest(N_EXTREMES, col)
        )
        bottom_parts[col].append(
            valid.nsmallest(N_EXTREMES, col)
        )

    if rows_scanned % 5_000_000 < CHUNK_SIZE:
        print(f"Rows scanned: {rows_scanned:,}", flush=True)

all_extremes = []

for col in numeric_cols:
    top = pd.concat(top_parts[col], ignore_index=True)
    bottom = pd.concat(bottom_parts[col], ignore_index=True)

    top = top.nlargest(N_EXTREMES, col).copy()
    top["extreme_feature"] = col
    top["extreme_type"] = "highest"

    bottom = bottom.nsmallest(N_EXTREMES, col).copy()
    bottom["extreme_feature"] = col
    bottom["extreme_type"] = "lowest"

    all_extremes.extend([top, bottom])

result = pd.concat(all_extremes, ignore_index=True)

result.to_csv(
    os.path.join(OUTPUT_DIR, "extreme_records.csv"),
    index=False,
)

print("\n===== MOST EXTREME RECORDS =====")
for col in ["BT_11", "BT_12", "BTD", "CTT", "CTH",
            "GPM_precipitation_rate"]:
    print(f"\n--- {col}: highest values ---")
    print(
        result[
            (result["extreme_feature"] == col)
            & (result["extreme_type"] == "highest")
        ][[
            "timestamp", "lat", "lon", col,
            "BT_11", "BT_12", "BTD",
            "CTT", "CTH", "GPM_precipitation_rate"
        ]].head(10).to_string(index=False)
    )

print("\nSaved all extreme records to eda_results/extreme_records.csv")

