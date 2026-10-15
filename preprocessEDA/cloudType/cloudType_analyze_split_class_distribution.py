"""
Analyze target-class representation across chronological splits.

Input:
eda_results/cloudType/preprocessing/splits/

Outputs:
eda_results/cloudType/preprocessing/class_analysis/

The original split files are not modified.
"""

from pathlib import Path
import json

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/home/m_prathilothamai/aviation_weather_project"
)

SPLIT_DIR = (
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
    / "class_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}

EXPECTED_CLASSES = [
    "clear",
    "liquid_low",
    "liquid_middle",
    "liquid_high",
    "ice_low",
    "ice_middle",
    "ice_high",
]

CHUNK_SIZE = 500_000

# ============================================================
# ANALYSIS
# ============================================================

print("=" * 75)
print("CHRONOLOGICAL SPLIT CLASS DISTRIBUTION ANALYSIS")
print("=" * 75)

all_results = {}
all_unknown_labels = {}
all_timestamp_ranges = {}

for split_name, file_path in SPLITS.items():

    print(f"\nAnalyzing {split_name.upper()} split...")

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    class_counts = {
        label: 0 for label in EXPECTED_CLASSES
    }

    unknown_labels = {}
    total_rows = 0

    min_timestamp = None
    max_timestamp = None

    for chunk in pd.read_csv(
        file_path,
        usecols=["timestamp", "cloud_type_label"],
        dtype={
            "timestamp": "string",
            "cloud_type_label": "string",
        },
        chunksize=CHUNK_SIZE,
    ):

        total_rows += len(chunk)

        counts = chunk["cloud_type_label"].value_counts(
            dropna=False
        )

        for label, count in counts.items():

            if label in class_counts:
                class_counts[label] += int(count)
            else:
                key = (
                    "<MISSING>"
                    if pd.isna(label)
                    else str(label)
                )

                unknown_labels[key] = (
                    unknown_labels.get(key, 0) + int(count)
                )

        timestamps = chunk["timestamp"].dropna()

        if len(timestamps) > 0:

            chunk_min = timestamps.min()
            chunk_max = timestamps.max()

            if min_timestamp is None or chunk_min < min_timestamp:
                min_timestamp = chunk_min

            if max_timestamp is None or chunk_max > max_timestamp:
                max_timestamp = chunk_max

    percentages = {
        label: (
            class_counts[label] / total_rows * 100
            if total_rows > 0 else 0
        )
        for label in EXPECTED_CLASSES
    }

    all_results[split_name] = {
        "total_rows": total_rows,
        "class_counts": class_counts,
        "class_percentages": percentages,
    }

    all_unknown_labels[split_name] = unknown_labels

    all_timestamp_ranges[split_name] = {
        "first_timestamp": min_timestamp,
        "last_timestamp": max_timestamp,
    }

    print(f"Rows: {total_rows:,}")
    print(f"First timestamp: {min_timestamp}")
    print(f"Last timestamp: {max_timestamp}")

    print("\nClass distribution:")

    for label in EXPECTED_CLASSES:
        print(
            f"{label:16s} "
            f"{class_counts[label]:12,} "
            f"({percentages[label]:8.4f}%)"
        )

    print("\nUnknown labels:", unknown_labels)


# ============================================================
# COMBINED CLASS DISTRIBUTION TABLE
# ============================================================

rows = []

for label in EXPECTED_CLASSES:

    row = {"class_label": label}

    for split_name in SPLITS:

        count = all_results[split_name]["class_counts"][label]
        percentage = all_results[split_name]["class_percentages"][label]

        row[f"{split_name}_count"] = count
        row[f"{split_name}_percentage"] = percentage

    rows.append(row)

distribution_df = pd.DataFrame(rows)

distribution_df.to_csv(
    OUTPUT_DIR / "split_class_distribution.csv",
    index=False,
)


# ============================================================
# CLASS PRESENCE CHECK
# ============================================================

presence_rows = []

for label in EXPECTED_CLASSES:

    row = {"class_label": label}

    for split_name in SPLITS:

        count = all_results[split_name]["class_counts"][label]

        row[split_name] = count > 0

    row["present_in_all_splits"] = all(
        row[split_name]
        for split_name in SPLITS
    )

    presence_rows.append(row)

presence_df = pd.DataFrame(presence_rows)

presence_df.to_csv(
    OUTPUT_DIR / "class_presence_check.csv",
    index=False,
)


# ============================================================
# CLASS PROPORTION DRIFT
# ============================================================

drift_rows = []

for label in EXPECTED_CLASSES:

    train_pct = all_results["train"]["class_percentages"][label]

    for split_name in ["validation", "test"]:

        split_pct = all_results[split_name]["class_percentages"][label]

        drift_rows.append({
            "class_label": label,
            "comparison": f"{split_name}_versus_train",
            "train_percentage": train_pct,
            "comparison_percentage": split_pct,
            "difference_percentage_points": split_pct - train_pct,
            "relative_change_percentage": (
                (split_pct - train_pct) / train_pct * 100
                if train_pct > 0 else None
            ),
        })

drift_df = pd.DataFrame(drift_rows)

drift_df.to_csv(
    OUTPUT_DIR / "class_proportion_drift.csv",
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

all_splits_nonempty = all(
    all_results[name]["total_rows"] > 0
    for name in SPLITS
)

all_expected_classes_present = all(
    all_results[name]["class_counts"][label] > 0
    for name in SPLITS
    for label in EXPECTED_CLASSES
)

no_unknown_labels = all(
    len(all_unknown_labels[name]) == 0
    for name in SPLITS
)

total_split_rows = sum(
    all_results[name]["total_rows"]
    for name in SPLITS
)

summary = {
    "split_row_counts": {
        name: all_results[name]["total_rows"]
        for name in SPLITS
    },
    "total_split_rows": total_split_rows,
    "class_counts": {
        name: all_results[name]["class_counts"]
        for name in SPLITS
    },
    "class_percentages": {
        name: all_results[name]["class_percentages"]
        for name in SPLITS
    },
    "unknown_labels": all_unknown_labels,
    "timestamp_ranges": all_timestamp_ranges,
    "all_splits_nonempty": all_splits_nonempty,
    "all_seven_classes_present_in_every_split": (
        all_expected_classes_present
    ),
    "no_unknown_or_missing_labels": no_unknown_labels,
}

with open(
    OUTPUT_DIR / "class_analysis_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4, default=str)


# ============================================================
# PRINT REPORT
# ============================================================

print("\n" + "=" * 75)
print("COMBINED CLASS DISTRIBUTION")
print("=" * 75)

print(distribution_df.to_string(index=False))

print("\n" + "=" * 75)
print("CLASS PRESENCE")
print("=" * 75)

print(presence_df.to_string(index=False))

print("\n" + "=" * 75)
print("CLASS PROPORTION DRIFT")
print("=" * 75)

print(drift_df.round(4).to_string(index=False))

print("\n" + "=" * 75)
print("FINAL CHECKS")
print("=" * 75)

print("All splits nonempty:", all_splits_nonempty)

print(
    "All seven classes present in every split:",
    all_expected_classes_present,
)

print("No unknown or missing labels:", no_unknown_labels)

print("Total rows across splits:", f"{total_split_rows:,}")

print("\nOutputs saved under:")
print(OUTPUT_DIR)

print("\nAnalysis completed.")

