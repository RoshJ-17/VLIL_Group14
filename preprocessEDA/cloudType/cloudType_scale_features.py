"""
Scale cloud-type features using StandardScaler.

Important:
- Fit scaler using training data only.
- Transform validation and test using the training scaler.
- Do not scale timestamps or target labels.
- Original split files remain unchanged.

Input:
eda_results/cloudType/preprocessing/splits/

Output:
eda_results/cloudType/preprocessing/scaled_splits/
"""

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/home/m_prathilothamai/aviation_weather_project"
)

INPUT_DIR = (
    PROJECT_DIR
    / "eda_results"
    / "cloudType"
    / "preprocessing"
    / "splits"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "eda_results"
    / "cloudType"
    / "preprocessing"
    / "scaled_splits"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
]

TARGET = "cloud_type_label"
TIMESTAMP = "timestamp"

CHUNK_SIZE = 500_000

SPLITS = {
    "train": INPUT_DIR / "train.csv",
    "validation": INPUT_DIR / "validation.csv",
    "test": INPUT_DIR / "test.csv",
}

EXPECTED_COLUMNS = [
    TIMESTAMP,
    *FEATURES,
    TARGET,
]

# ============================================================
# 2. INITIAL VALIDATION
# ============================================================

print("=" * 75)
print("CLOUD-TYPE FEATURE SCALING")
print("=" * 75)

for split_name, file_path in SPLITS.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"{split_name} split not found: {file_path}"
        )

    header = pd.read_csv(
        file_path,
        nrows=0,
    ).columns.tolist()

    missing = [
        column
        for column in EXPECTED_COLUMNS
        if column not in header
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {split_name}: {missing}"
        )

print("\nAll three input split files found.")
print("Required columns verified.")


# ============================================================
# 3. FIT SCALER USING TRAINING DATA ONLY
# ============================================================

print("\n" + "=" * 75)
print("STEP 1: FIT SCALER ON TRAINING DATA")
print("=" * 75)

scaler = StandardScaler()

train_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        SPLITS["train"],
        usecols=FEATURES,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    X = chunk[FEATURES].astype("float64")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            f"Non-finite training features in chunk {chunk_number}"
        )

    scaler.partial_fit(X)

    train_rows += len(X)

    if chunk_number % 10 == 0:
        print(
            f"Training scaler: {train_rows:,} rows processed."
        )

print(f"\nTraining rows used to fit scaler: {train_rows:,}")

print("\nTraining feature means:")
for feature, value in zip(FEATURES, scaler.mean_):
    print(f"{feature:10s}: {value:.8f}")

print("\nTraining feature standard deviations:")
for feature, value in zip(FEATURES, scaler.scale_):
    print(f"{feature:10s}: {value:.8f}")


# ============================================================
# 4. SAVE FITTED SCALER
# ============================================================

print("\n" + "=" * 75)
print("STEP 2: SAVE FITTED SCALER")
print("=" * 75)

scaler_path = OUTPUT_DIR / "standard_scaler.joblib"

joblib.dump(
    scaler,
    scaler_path,
)

scaler_metadata = {
    "scaler": "StandardScaler",
    "fitted_on": "training split only",
    "training_rows_used": int(train_rows),
    "features": FEATURES,
    "mean": {
        feature: float(value)
        for feature, value in zip(FEATURES, scaler.mean_)
    },
    "scale": {
        feature: float(value)
        for feature, value in zip(FEATURES, scaler.scale_)
    },
    "variance": {
        feature: float(value)
        for feature, value in zip(FEATURES, scaler.var_)
    },
}

with open(
    OUTPUT_DIR / "scaler_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        scaler_metadata,
        file,
        indent=4,
    )

print(f"Scaler saved: {scaler_path}")


# ============================================================
# 5. TRANSFORM TRAIN, VALIDATION AND TEST
# ============================================================

print("\n" + "=" * 75)
print("STEP 3: TRANSFORM ALL SPLITS")
print("=" * 75)

split_summary = {}

for split_name, input_path in SPLITS.items():

    print(f"\nScaling {split_name.upper()} split...")

    output_path = OUTPUT_DIR / f"{split_name}.csv"

    # Remove previous output to prevent accidental appending.
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    feature_min = {
        feature: float("inf")
        for feature in FEATURES
    }
    feature_max = {
        feature: float("-inf")
        for feature in FEATURES
    }

    feature_sum = {
        feature: 0.0
        for feature in FEATURES
    }
    feature_sum_sq = {
        feature: 0.0
        for feature in FEATURES
    }

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_path,
            usecols=EXPECTED_COLUMNS,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        # Preserve timestamp and target without transformation.
        X = chunk[FEATURES].astype("float64")

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(
                f"Non-finite values in {split_name}, "
                f"chunk {chunk_number}"
            )

        # Apply training-fitted scaler.
        X_scaled = scaler.transform(X)

        if not np.isfinite(X_scaled).all():
            raise ValueError(
                f"Non-finite scaled values in {split_name}, "
                f"chunk {chunk_number}"
            )

        scaled_features = pd.DataFrame(
            X_scaled,
            columns=FEATURES,
            index=chunk.index,
        )

        output_chunk = pd.concat(
            [
                chunk[[TIMESTAMP]].reset_index(drop=True),
                scaled_features.reset_index(drop=True),
                chunk[[TARGET]].reset_index(drop=True),
            ],
            axis=1,
        )

        output_chunk.to_csv(
            output_path,
            mode="a",
            header=not output_path.exists(),
            index=False,
        )

        total_rows += len(output_chunk)

        for feature in FEATURES:

            values = output_chunk[feature].to_numpy()

            feature_min[feature] = min(
                feature_min[feature],
                float(values.min()),
            )

            feature_max[feature] = max(
                feature_max[feature],
                float(values.max()),
            )

            feature_sum[feature] += float(values.sum())

            feature_sum_sq[feature] += float(
                np.square(values).sum()
            )

        if chunk_number % 10 == 0:
            print(
                f"{split_name}: {total_rows:,} rows transformed."
            )

    # Calculate final transformed feature statistics.
    transformed_summary = {}

    for feature in FEATURES:

        mean = feature_sum[feature] / total_rows

        variance = (
            feature_sum_sq[feature] / total_rows
            - mean ** 2
        )

        transformed_summary[feature] = {
            "mean": mean,
            "standard_deviation": float(
                np.sqrt(max(variance, 0.0))
            ),
            "minimum": feature_min[feature],
            "maximum": feature_max[feature],
        }

    split_summary[split_name] = {
        "input_rows": total_rows,
        "output_file": str(output_path),
        "features": transformed_summary,
    }

    print(
        f"{split_name.upper()} completed: "
        f"{total_rows:,} rows"
    )


# ============================================================
# 6. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("STEP 4: FINAL SCALING VALIDATION")
print("=" * 75)

expected_rows = {
    "train": 37_097_457,
    "validation": 7_964_026,
    "test": 7_914_388,
}

validation_results = {}

for split_name in SPLITS:

    output_path = OUTPUT_DIR / f"{split_name}.csv"

    output_rows = 0
    finite_values = True
    columns_correct = True

    for chunk in pd.read_csv(
        output_path,
        chunksize=CHUNK_SIZE,
    ):

        output_rows += len(chunk)

        if chunk.columns.tolist() != EXPECTED_COLUMNS:
            columns_correct = False

        values = chunk[FEATURES].to_numpy(dtype="float64")

        if not np.isfinite(values).all():
            finite_values = False

    validation_results[split_name] = {
        "expected_rows": expected_rows[split_name],
        "actual_rows": output_rows,
        "row_count_matches": (
            output_rows == expected_rows[split_name]
        ),
        "all_scaled_features_finite": finite_values,
        "columns_correct": columns_correct,
    }

    print(f"\n{split_name.upper()}")

    for key, value in validation_results[split_name].items():
        print(f"{key}: {value}")


all_checks_passed = all(
    result["row_count_matches"]
    and result["all_scaled_features_finite"]
    and result["columns_correct"]
    for result in validation_results.values()
)

final_metadata = {
    "scaler": "StandardScaler",
    "fitted_on_training_only": True,
    "features_scaled": FEATURES,
    "timestamp_unchanged": True,
    "target_unchanged": True,
    "split_summary": split_summary,
    "validation": validation_results,
    "all_checks_passed": all_checks_passed,
}

with open(
    OUTPUT_DIR / "scaling_report.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_metadata,
        file,
        indent=4,
    )

print("\n" + "=" * 75)
print("SCALING COMPLETED")
print("=" * 75)

print(f"\nOutput directory:\n{OUTPUT_DIR}")

print("\nFinal validation:", all_checks_passed)

if not all_checks_passed:
    raise RuntimeError(
        "One or more scaling validation checks failed."
    )

print("\nAll scaling checks passed.")

