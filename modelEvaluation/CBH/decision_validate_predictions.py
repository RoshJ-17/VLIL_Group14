
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/m_prathilothamai/aviation_weather_project")
DATA = ROOT / "data/processed/january_2023"

SOURCE = DATA / "january_2023_features.csv"
PREDICTIONS = DATA / "january_2023_predictions.csv"

CHUNK_SIZE = 100_000

prediction_columns = [
    "cloud_type_id",
    "CBH_km",
    "rain_probability",
    "rain_detected",
    "rainfall_intensity_conditional_mm_hr",
    "final_rainfall_mm_hr",
    "bt_anomaly_flag"
]

numeric_columns = [
    "CBH_km",
    "rain_probability",
    "rain_detected",
    "rainfall_intensity_conditional_mm_hr",
    "final_rainfall_mm_hr"
]

source_rows = 0
prediction_rows = 0
chunk_number = 0

missing_counts = {c: 0 for c in prediction_columns}
nonfinite_counts = {c: 0 for c in numeric_columns}

class_counts = {i: 0 for i in range(7)}
rain_count = 0
anomaly_count = 0
gate_violations = 0
invalid_class_count = 0

cbh_min = np.inf
cbh_max = -np.inf
prob_min = np.inf
prob_max = -np.inf
conditional_min = np.inf
conditional_max = -np.inf
final_min = np.inf
final_max = -np.inf

source_columns = ["timestamp", "lat", "lon", "BT_11", "BT_12", "BTD"]

print("Starting independent prediction validation...")

source_reader = pd.read_csv(
    SOURCE,
    usecols=source_columns,
    chunksize=CHUNK_SIZE
)

prediction_reader = pd.read_csv(
    PREDICTIONS,
    chunksize=CHUNK_SIZE
)

for source, pred in zip(source_reader, prediction_reader):

    chunk_number += 1
    source_rows += len(source)
    prediction_rows += len(pred)

    if len(source) != len(pred):
        raise RuntimeError(
            f"Row mismatch in chunk {chunk_number}"
        )

    # Verify source observations were preserved in order.
    for col in source_columns:
        if col not in pred.columns:
            raise RuntimeError(f"Missing original column: {col}")

        a = source[col]
        b = pred[col]

        if col == "timestamp":
            equal = a.astype(str).equals(b.astype(str))
        else:
            equal = np.allclose(
                pd.to_numeric(a, errors="coerce").to_numpy(dtype=np.float64),
                pd.to_numeric(b, errors="coerce").to_numpy(dtype=np.float64),
                rtol=0,
                atol=1e-6,
                equal_nan=True
            )

        if not equal:
            raise RuntimeError(
                f"Source column changed: {col}, chunk {chunk_number}"
            )
    # Required prediction columns.
    for col in prediction_columns:
        if col not in pred.columns:
            raise RuntimeError(
                f"Missing prediction column: {col}"
            )

        missing_counts[col] += int(pred[col].isna().sum())

    # Numeric prediction validity.
    for col in numeric_columns:
        values = pd.to_numeric(
            pred[col], errors="coerce"
        ).to_numpy(dtype=np.float64)

        nonfinite_counts[col] += int(
            (~np.isfinite(values)).sum()
        )

    # Cloud class validity.
    classes = pred["cloud_type_id"].to_numpy()

    invalid_class_count += int(
        ((classes < 0) | (classes > 6) | (classes != np.floor(classes))).sum()
    )

    for class_id in range(7):
        class_counts[class_id] += int(
            (classes == class_id).sum()
        )

    # Rain detection validity.
    rain = pred["rain_detected"].to_numpy()

    if not np.isin(rain, [0, 1]).all():
        raise RuntimeError("Invalid rain detection labels.")

    rain_count += int((rain == 1).sum())

    # Rainfall gate.
    final_rain = pred["final_rainfall_mm_hr"].to_numpy()
    conditional = pred[
        "rainfall_intensity_conditional_mm_hr"
    ].to_numpy()

    no_rain = rain == 0

    gate_violations += int(
        np.count_nonzero(final_rain[no_rain] != 0)
    )

    # Conditional rainfall must be nonnegative.
    if (conditional < 0).any():
        raise RuntimeError(
            "Negative conditional rainfall found."
        )

    # Probability bounds.
    probabilities = pred["rain_probability"].to_numpy()

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError(
            "Rain probabilities outside [0,1]."
        )

    # Track ranges.
    cbh_min = min(cbh_min, pred["CBH_km"].min())
    cbh_max = max(cbh_max, pred["CBH_km"].max())

    prob_min = min(prob_min, probabilities.min())
    prob_max = max(prob_max, probabilities.max())

    conditional_min = min(
        conditional_min, conditional.min()
    )
    conditional_max = max(
        conditional_max, conditional.max()
    )

    final_min = min(final_min, final_rain.min())
    final_max = max(final_max, final_rain.max())

    anomaly_count += int(
        pred["bt_anomaly_flag"].sum()
    )

    if chunk_number % 3 == 0:
        print(
            f"Validated chunk {chunk_number} | "
            f"Rows: {prediction_rows:,}",
            flush=True
        )

if source_rows != prediction_rows:
    raise RuntimeError(
        f"Total row mismatch: source={source_rows}, "
        f"predictions={prediction_rows}"
    )

print("\n========== VALIDATION RESULTS ==========")
print("Source rows:", f"{source_rows:,}")
print("Prediction rows:", f"{prediction_rows:,}")
print("Chunks checked:", chunk_number)

print("\nMissing prediction values:")
for col, count in missing_counts.items():
    print(col, count)

print("\nNon-finite prediction values:")
for col, count in nonfinite_counts.items():
    print(col, count)

print("\nCloud class counts:")
for class_id, count in class_counts.items():
    print(class_id, count)

print("\nInvalid cloud class IDs:", invalid_class_count)
print("Rain detections:", f"{rain_count:,}")
print(
    "Rain detection percentage:",
    round(rain_count / prediction_rows * 100, 4)
)
print("Rainfall gate violations:", gate_violations)
print("Extreme BT flags:", anomaly_count)

print("\nPrediction ranges:")
print("CBH km:", cbh_min, "to", cbh_max)
print("Rain probability:", prob_min, "to", prob_max)
print(
    "Conditional rainfall mm/hr:",
    conditional_min, "to", conditional_max
)
print("Final rainfall mm/hr:", final_min, "to", final_max)

print("\n========== FINAL CHECKS ==========")

checks = {
    "Row counts match": source_rows == prediction_rows,
    "No missing predictions": all(v == 0 for v in missing_counts.values()),
    "No non-finite predictions": all(v == 0 for v in nonfinite_counts.values()),
    "Cloud classes valid": invalid_class_count == 0,
    "Rainfall gate correct": gate_violations == 0,
    "Exactly two BT anomalies flagged": anomaly_count == 2
}

for name, passed in checks.items():
    print(name, "PASS" if passed else "FAIL")

print("\nValidation finished.")

