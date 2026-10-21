
import os
import json
import pandas as pd
import numpy as np

INPUT_CSV = "modisPixel_with_GPM.csv"
OUTPUT_DIR = "eda_results"
CHUNK_SIZE = 1_000_000
RAIN_THRESHOLD = 0.1

CATEGORICAL_COLS = [
    "Cloud_Mask_Binary",
    "Cloud_Mask_Raw",
    "Cloud_Phase",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

required = CATEGORICAL_COLS + ["GPM_precipitation_rate"]
counts = {
    col: {
        "category_counts": {},
        "category_rainy_counts": {},
        "category_gpm_sum": {},
        "category_gpm_count": {},
        "missing": 0,
        "total": 0,
    }
    for col in CATEGORICAL_COLS
}

rows_scanned = 0

print("Starting categorical EDA...", flush=True)

for chunk in pd.read_csv(
    INPUT_CSV,
    usecols=required,
    chunksize=CHUNK_SIZE,
):
    rows_scanned += len(chunk)

    gpm = pd.to_numeric(
        chunk["GPM_precipitation_rate"], errors="coerce"
    )
    valid_target = gpm.notna() & np.isfinite(gpm) & (gpm >= 0)
    rainy = valid_target & (gpm >= RAIN_THRESHOLD)

    for col in CATEGORICAL_COLS:
        values = chunk[col]
        counts[col]["total"] += len(chunk)
        counts[col]["missing"] += int(values.isna().sum())

        # Include missing categories explicitly.
        labels = values.astype("string").fillna("<MISSING>")

        grouped = pd.DataFrame({
            "category": labels,
            "valid_target": valid_target.astype("int8"),
            "rainy": rainy.astype("int8"),
            "gpm": gpm.where(valid_target),
        })

        summary = grouped.groupby("category", dropna=False).agg(
            records=("category", "size"),
            valid_target_records=("valid_target", "sum"),
            rainy_records=("rainy", "sum"),
            gpm_sum=("gpm", "sum"),
            gpm_count=("gpm", "count"),
        )

        for category, row in summary.iterrows():
            key = str(category)
            counts[col]["category_counts"][key] = (
                counts[col]["category_counts"].get(key, 0)
                + int(row["records"])
            )
            counts[col]["category_rainy_counts"][key] = (
                counts[col]["category_rainy_counts"].get(key, 0)
                + int(row["rainy_records"])
            )
            counts[col]["category_gpm_sum"][key] = (
                counts[col]["category_gpm_sum"].get(key, 0.0)
                + float(row["gpm_sum"])
            )
            counts[col]["category_gpm_count"][key] = (
                counts[col]["category_gpm_count"].get(key, 0)
                + int(row["gpm_count"])
            )

    if rows_scanned % 5_000_000 < CHUNK_SIZE:
        print(f"Rows scanned: {rows_scanned:,}", flush=True)

output = {}

for col in CATEGORICAL_COLS:
    result = counts[col]
    rows = []

    for category, n in result["category_counts"].items():
        rainy_n = result["category_rainy_counts"].get(category, 0)
        gpm_sum = result["category_gpm_sum"].get(category, 0.0)
        gpm_n = result["category_gpm_count"].get(category, 0)

        rows.append({
            "feature": col,
            "category": category,
            "records": n,
            "record_percent": 100 * n / result["total"],
            "valid_target_records": gpm_n,
            "rainy_records": rainy_n,
            "rainy_percent": 100 * rainy_n / gpm_n if gpm_n else np.nan,
            "mean_gpm_rate_mm_hr": gpm_sum / gpm_n if gpm_n else np.nan,
        })

    table = pd.DataFrame(rows).sort_values(
        "records", ascending=False
    )

    path = os.path.join(OUTPUT_DIR, f"{col.lower()}_category_summary.csv")
    table.to_csv(path, index=False)

    output[col] = {
        "total_records": result["total"],
        "missing_records": result["missing"],
        "missing_percent": 100 * result["missing"] / result["total"],
        "categories": table.to_dict(orient="records"),
        "output_file": path,
    }

    print(f"\n===== {col} =====")
    print(f"Missing: {result['missing']:,} "
          f"({100 * result['missing'] / result['total']:.4f}%)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

json_path = os.path.join(OUTPUT_DIR, "categorical_analysis_summary.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=2, allow_nan=True)

print("\nCategorical EDA complete.")
print(f"Summary: {json_path}")


