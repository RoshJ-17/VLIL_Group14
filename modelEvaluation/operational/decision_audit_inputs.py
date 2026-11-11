
import json
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path(
    "data/processed/january_2023/january_2023_predictions.csv"
)

OUTPUT_DIR = Path("decision_support_results")
OUTPUT_FILE = OUTPUT_DIR / "decision_input_audit.json"

REQUIRED_COLUMNS = [
    "timestamp",
    "lat",
    "lon",
    "cloud_type_id",
    "CBH_km",
    "rain_probability",
    "rain_detected",
    "rainfall_intensity_conditional_mm_hr",
    "final_rainfall_mm_hr",
    "bt_anomaly_flag",
]

RAIN_THRESHOLD = 0.33

CLOUD_CLASSES = {
    0: "clear",
    1: "liquid_low",
    2: "liquid_middle",
    3: "liquid_high",
    4: "ice_low",
    5: "ice_middle",
    6: "ice_high",
}


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("\nStarting decision-support input audit...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Prediction file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print(f"\nInput file: {INPUT_FILE}")
print(f"Total observations: {len(df):,}")
print(f"Total columns in file: {len(df.columns)}")


# --------------------------------------------------
# Data types and missing values
# --------------------------------------------------

print("\n========== COLUMN DATA TYPES ==========")

for col in REQUIRED_COLUMNS:
    print(f"{col}: {df[col].dtype}")

print("\n========== MISSING VALUES ==========")

missing_summary = df[REQUIRED_COLUMNS].isna().sum()

for col, count in missing_summary.items():
    print(f"{col}: {count:,}")


# --------------------------------------------------
# Geographic coverage
# --------------------------------------------------

print("\n========== GEOGRAPHIC COVERAGE ==========")

geo_summary = {
    "latitude_min": float(df["lat"].min()),
    "latitude_max": float(df["lat"].max()),
    "longitude_min": float(df["lon"].min()),
    "longitude_max": float(df["lon"].max()),
    "unique_coordinates": int(
        df[["lat", "lon"]].drop_duplicates().shape[0]
    ),
}

for key, value in geo_summary.items():
    print(f"{key}: {value}")


# --------------------------------------------------
# Timestamp distribution
# --------------------------------------------------

print("\n========== TEMPORAL COVERAGE ==========")

df["parsed_timestamp"] = pd.to_datetime(
    df["timestamp"].astype(str),
    format="%Y%j_%H%M",
    errors="coerce"
)

invalid_timestamps = int(df["parsed_timestamp"].isna().sum())

timestamp_counts = df.groupby("timestamp").size()

print("Unique timestamps:", len(timestamp_counts))
print("Invalid timestamps:", invalid_timestamps)

if len(timestamp_counts):
    print("First timestamp:", timestamp_counts.index.min())
    print("Last timestamp:", timestamp_counts.index.max())
    print("Minimum observations per timestamp:", int(timestamp_counts.min()))
    print("Maximum observations per timestamp:", int(timestamp_counts.max()))
    print("Mean observations per timestamp:", round(timestamp_counts.mean(), 2))


# --------------------------------------------------
# Cloud type distribution
# --------------------------------------------------

print("\n========== CLOUD TYPE DISTRIBUTION ==========")

cloud_counts = df["cloud_type_id"].value_counts().sort_index()

cloud_summary = {}

for class_id, class_name in CLOUD_CLASSES.items():
    count = int(cloud_counts.get(class_id, 0))
    percentage = count / len(df) * 100

    cloud_summary[str(class_id)] = {
        "class_name": class_name,
        "count": count,
        "percentage": round(percentage, 4),
    }

    print(
        f"{class_id} ({class_name}): "
        f"{count:,} ({percentage:.4f}%)"
    )


# --------------------------------------------------
# CBH analysis
# --------------------------------------------------

print("\n========== CBH ANALYSIS ==========")

cbh = df["CBH_km"]

cbh_summary = {
    "minimum": float(cbh.min()),
    "maximum": float(cbh.max()),
    "mean": float(cbh.mean()),
    "median": float(cbh.median()),
    "negative_count": int((cbh < 0).sum()),
    "zero_count": int((cbh == 0).sum()),
    "positive_count": int((cbh > 0).sum()),
    "negative_percentage": round(
        (cbh < 0).mean() * 100, 4
    ),
}

for key, value in cbh_summary.items():
    print(f"{key}: {value}")

print("\nCBH statistics by cloud class:")

cbh_by_cloud = df.groupby("cloud_type_id")["CBH_km"].agg(
    count="count",
    minimum="min",
    median="median",
    maximum="max",
    mean="mean",
)

cbh_by_cloud["negative_count"] = (
    df.assign(negative=cbh < 0)
      .groupby("cloud_type_id")["negative"]
      .sum()
)

print(cbh_by_cloud.to_string())


# --------------------------------------------------
# Precipitation analysis
# --------------------------------------------------

print("\n========== PRECIPITATION ANALYSIS ==========")

rain_probability = df["rain_probability"]
rain_detected = df["rain_detected"]

rain_summary = {
    "probability_min": float(rain_probability.min()),
    "probability_max": float(rain_probability.max()),
    "probability_mean": float(rain_probability.mean()),
    "probability_median": float(rain_probability.median()),
    "rain_detected_count": int((rain_detected == 1).sum()),
    "rain_not_detected_count": int((rain_detected == 0).sum()),
    "rain_detected_percentage": round(
        (rain_detected == 1).mean() * 100, 4
    ),
    "training_threshold": RAIN_THRESHOLD,
}

threshold_expected = (rain_probability >= RAIN_THRESHOLD).astype(int)

threshold_mismatches = int(
    (threshold_expected != rain_detected).sum()
)

rain_summary["threshold_mismatches"] = threshold_mismatches

print("Rain probability minimum:", rain_summary["probability_min"])
print("Rain probability maximum:", rain_summary["probability_max"])
print("Rain probability mean:", rain_summary["probability_mean"])
print("Rain detections:", rain_summary["rain_detected_count"])
print("Rain detection percentage:", rain_summary["rain_detected_percentage"])
print("Probability threshold:", RAIN_THRESHOLD)
print("Threshold mismatches:", threshold_mismatches)


# --------------------------------------------------
# Rainfall intensity analysis
# --------------------------------------------------

print("\n========== RAINFALL INTENSITY ANALYSIS ==========")

conditional = df["rainfall_intensity_conditional_mm_hr"]
final_rainfall = df["final_rainfall_mm_hr"]

rainfall_summary = {
    "conditional_min": float(conditional.min()),
    "conditional_max": float(conditional.max()),
    "conditional_mean": float(conditional.mean()),
    "conditional_median": float(conditional.median()),
    "final_min": float(final_rainfall.min()),
    "final_max": float(final_rainfall.max()),
    "final_mean": float(final_rainfall.mean()),
    "final_median": float(final_rainfall.median()),
    "negative_conditional_count": int((conditional < 0).sum()),
    "negative_final_count": int((final_rainfall < 0).sum()),
    "rainfall_gate_violations": int(
        ((rain_detected == 0) & (final_rainfall != 0)).sum()
    ),
}

for key, value in rainfall_summary.items():
    print(f"{key}: {value}")


# --------------------------------------------------
# Data quality flags
# --------------------------------------------------

print("\n========== DATA QUALITY FLAGS ==========")

anomaly_count = int((df["bt_anomaly_flag"] == 1).sum())

print("BT anomaly flags:", anomaly_count)

anomaly_summary = {
    "flagged_count": anomaly_count,
    "flagged_percentage": round(
        anomaly_count / len(df) * 100, 6
    ),
}


# --------------------------------------------------
# Summary and report
# --------------------------------------------------

audit_summary = {
    "input_file": str(INPUT_FILE),
    "total_observations": int(len(df)),
    "total_columns": int(len(df.columns)),
    "missing_values": {
        col: int(count)
        for col, count in missing_summary.items()
    },
    "geographic_coverage": geo_summary,
    "temporal_coverage": {
        "unique_timestamps": int(len(timestamp_counts)),
        "invalid_timestamps": invalid_timestamps,
        "first_timestamp": str(timestamp_counts.index.min()),
        "last_timestamp": str(timestamp_counts.index.max()),
        "minimum_rows_per_timestamp": int(timestamp_counts.min()),
        "maximum_rows_per_timestamp": int(timestamp_counts.max()),
        "mean_rows_per_timestamp": float(timestamp_counts.mean()),
    },
    "cloud_distribution": cloud_summary,
    "cbh": cbh_summary,
    "precipitation": rain_summary,
    "rainfall_intensity": rainfall_summary,
    "data_quality": anomaly_summary,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as file:
    json.dump(audit_summary, file, indent=4)

print("\n========== AUDIT COMPLETED ==========")
print("Audit report saved to:", OUTPUT_FILE)
print("No input data was modified.")

