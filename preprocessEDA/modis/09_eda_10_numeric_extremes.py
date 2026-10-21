
import os
import json
import pandas as pd
import numpy as np

INPUT_CSV = "modisPixel_with_GPM.csv"
OUTPUT_DIR = "eda_results"
CHUNK_SIZE = 1_000_000

COLS = [
    "BT_11",
    "BT_12",
    "BTD",
    "CTT",
    "CTH",
    "Cloud_Optical_Thickness",
    "GPM_precipitation_rate",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

stats = {
    c: {
        "count": 0,
        "missing": 0,
        "min": None,
        "max": None,
        "sum": 0.0,
        "sum_sq": 0.0,
        "negative_count": 0,
        "zero_count": 0,
        "nonfinite_count": 0,
    }
    for c in COLS
}

# Deterministic sample for approximate percentiles and tail inspection.
rng = np.random.default_rng(20261028)
sample_parts = {c: [] for c in COLS}
SAMPLE_PER_CHUNK = 5000
rows_scanned = 0

print("Starting numeric range and extreme-value analysis...", flush=True)

for chunk in pd.read_csv(
    INPUT_CSV,
    usecols=COLS,
    chunksize=CHUNK_SIZE,
):
    rows_scanned += len(chunk)

    for col in COLS:
        raw = chunk[col]
        x = pd.to_numeric(raw, errors="coerce")
        finite = x[np.isfinite(x)].to_numpy(dtype=np.float64)

        s = stats[col]
        s["missing"] += int(x.isna().sum())
        s["nonfinite_count"] += int((x.notna() & ~np.isfinite(x)).sum())
        s["count"] += int(finite.size)

        if finite.size:
            mn = float(finite.min())
            mx = float(finite.max())
            s["min"] = mn if s["min"] is None else min(s["min"], mn)
            s["max"] = mx if s["max"] is None else max(s["max"], mx)
            s["sum"] += float(finite.sum(dtype=np.float64))
            s["sum_sq"] += float(np.square(finite).sum(dtype=np.float64))
            s["negative_count"] += int((finite < 0).sum())
            s["zero_count"] += int((finite == 0).sum())

            take = min(SAMPLE_PER_CHUNK, finite.size)
            idx = rng.choice(finite.size, size=take, replace=False)
            sample_parts[col].append(finite[idx])

    if rows_scanned % 5_000_000 < CHUNK_SIZE:
        print(f"Rows scanned: {rows_scanned:,}", flush=True)

summary_rows = []
extreme_rows = []

for col in COLS:
    s = stats[col]
    n = s["count"]
    mean = s["sum"] / n if n else np.nan
    variance = max(s["sum_sq"] / n - mean * mean, 0.0) if n else np.nan

    samples = (
        np.concatenate(sample_parts[col])
        if sample_parts[col]
        else np.array([], dtype=np.float64)
    )

    percentiles = (
        np.percentile(samples, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        if samples.size
        else [np.nan] * 9
    )

    summary_rows.append({
        "feature": col,
        "finite_count": n,
        "missing_count": s["missing"],
        "nonfinite_count": s["nonfinite_count"],
        "min": s["min"],
        "max": s["max"],
        "mean": mean,
        "std": float(np.sqrt(variance)) if n else np.nan,
        "negative_count": s["negative_count"],
        "zero_count": s["zero_count"],
        "sample_count_for_percentiles": int(samples.size),
        "sample_p01": percentiles[1],
        "sample_p05": percentiles[2],
        "sample_p25": percentiles[3],
        "sample_p50": percentiles[4],
        "sample_p75": percentiles[5],
        "sample_p95": percentiles[6],
        "sample_p99": percentiles[7],
    })

    if samples.size:
        lo, hi = np.percentile(samples, [1, 99])
        extreme_rows.append({
            "feature": col,
            "sample_below_p01": int((samples < lo).sum()),
            "sample_above_p99": int((samples > hi).sum()),
            "sample_min": float(samples.min()),
            "sample_max": float(samples.max()),
            "sample_p01_cutoff": float(lo),
            "sample_p99_cutoff": float(hi),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    os.path.join(OUTPUT_DIR, "numeric_range_summary.csv"),
    index=False,
)

pd.DataFrame(extreme_rows).to_csv(
    os.path.join(OUTPUT_DIR, "numeric_extreme_sample_summary.csv"),
    index=False,
)

with open(os.path.join(OUTPUT_DIR, "numeric_range_summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2, allow_nan=True)

print("\n===== NUMERIC RANGE SUMMARY =====")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\nSaved results in eda_results/")

