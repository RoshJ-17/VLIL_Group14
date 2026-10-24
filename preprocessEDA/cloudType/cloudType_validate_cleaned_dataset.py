
#!/usr/bin/env python3

"""
Validate the cleaned cloud-type dataset.

Checks:
1. Total row and column counts.
2. Missing values.
3. Numeric feature statistics.
4. Remaining suspicious BT records.
5. BTD consistency.
6. Cloud-class distribution.
7. Comparison with the original dataset.

All outputs are saved under:
eda_results/cloudType/cleaned_validation/

Neither input dataset is modified.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ORIGINAL_FILE = Path("cloudtype_final.csv")

CLEANED_FILE = (
    Path("eda_results")
    / "cloudType"
    / "cleaning"
    / "cloudtype_final_cleaned.csv"
)

OUTPUT_DIR = (
    Path("eda_results")
    / "cloudType"
    / "cleaned_validation"
)

CHUNK_SIZE = 100_000

CLASSES = [
    "clear",
    "liquid_low",
    "liquid_middle",
    "liquid_high",
    "ice_low",
    "ice_middle",
    "ice_high",
]

NUMERIC_COLUMNS = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
]

REQUIRED_COLUMNS = [
    "timestamp",
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
    "cloud_type_label",
]

EXPECTED_ORIGINAL_ROWS = 52_975_877
EXPECTED_CLEANED_ROWS = 52_975_871
EXPECTED_REMOVED_ROWS = 6


# --------------------------------------------------
# Dataset validation function
# --------------------------------------------------

def validate_dataset(file_path):

    print(f"\nValidating: {file_path}")

    total_rows = 0

    class_counts = {cls: 0 for cls in CLASSES}

    missing_counts = {}
    numeric_stats = {}

    for col in NUMERIC_COLUMNS:
        numeric_stats[col] = {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": np.inf,
            "max": -np.inf,
        }

    btd_consistency = {
        "rows_compared": 0,
        "mean_absolute_difference_sum": 0.0,
        "max_absolute_difference": 0.0,
        "rows_difference_gt_0_05": 0,
    }

    suspicious_pair_count = 0

    threshold_counts = {
        "BT_11_gt_350": 0,
        "BT_12_gt_350": 0,
        "BT_11_gt_400": 0,
        "BT_12_gt_400": 0,
        "BT_11_lt_180": 0,
        "BT_12_lt_180": 0,
        "abs_BTD_gt_5": 0,
        "abs_BTD_gt_10": 0,
    }

    reader = pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE
    )

    first_chunk = True

    for chunk in reader:

        total_rows += len(chunk)

        missing_columns = (
            set(REQUIRED_COLUMNS) - set(chunk.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns: {sorted(missing_columns)}"
            )

        # Missing-value counts
        if first_chunk:
            missing_counts = {
                col: 0 for col in chunk.columns
            }

        for col in chunk.columns:
            missing_counts[col] += int(
                chunk[col].isna().sum()
            )

        first_chunk = False

        # Cloud class counts
        counts = chunk["cloud_type_label"].value_counts()

        for cls in CLASSES:
            class_counts[cls] += int(
                counts.get(cls, 0)
            )

        # Numeric validation
        for col in NUMERIC_COLUMNS:

            values = pd.to_numeric(
                chunk[col],
                errors="coerce"
            ).to_numpy(dtype=float)

            valid = values[np.isfinite(values)]

            if len(valid) == 0:
                continue

            stats = numeric_stats[col]

            stats["count"] += len(valid)
            stats["sum"] += float(valid.sum())
            stats["sum_sq"] += float(
                np.square(valid).sum()
            )
            stats["min"] = min(
                stats["min"], float(valid.min())
            )
            stats["max"] = max(
                stats["max"], float(valid.max())
            )

        bt11 = pd.to_numeric(
            chunk["BT_11"], errors="coerce"
        )

        bt12 = pd.to_numeric(
            chunk["BT_12"], errors="coerce"
        )

        btd = pd.to_numeric(
            chunk["BTD"], errors="coerce"
        )

        valid = (
            bt11.notna()
            & bt12.notna()
            & btd.notna()
        )

        # BTD consistency
        difference = (
            btd[valid] - (bt11[valid] - bt12[valid])
        ).abs()

        if len(difference):

            btd_consistency["rows_compared"] += len(
                difference
            )

            btd_consistency[
                "mean_absolute_difference_sum"
            ] += float(difference.sum())

            btd_consistency[
                "max_absolute_difference"
            ] = max(
                btd_consistency["max_absolute_difference"],
                float(difference.max())
            )

            btd_consistency[
                "rows_difference_gt_0_05"
            ] += int((difference > 0.05).sum())

        # Threshold checks
        threshold_counts["BT_11_gt_350"] += int(
            (bt11 > 350).sum()
        )

        threshold_counts["BT_12_gt_350"] += int(
            (bt12 > 350).sum()
        )

        threshold_counts["BT_11_gt_400"] += int(
            (bt11 > 400).sum()
        )

        threshold_counts["BT_12_gt_400"] += int(
            (bt12 > 400).sum()
        )

        threshold_counts["BT_11_lt_180"] += int(
            (bt11 < 180).sum()
        )

        threshold_counts["BT_12_lt_180"] += int(
            (bt12 < 180).sum()
        )

        threshold_counts["abs_BTD_gt_5"] += int(
            (btd.abs() > 5).sum()
        )

        threshold_counts["abs_BTD_gt_10"] += int(
            (btd.abs() > 10).sum()
        )

        # Check the six previously identified values
        suspicious_mask = (
            (bt11.round(2) == 486.80)
            & (bt12.round(2) == 495.75)
            & (btd.round(2) == -8.95)
        )

        suspicious_pair_count += int(
            suspicious_mask.sum()
        )

        if total_rows % 1_000_000 < CHUNK_SIZE:
            print(f"Processed {total_rows:,} rows...")

    # --------------------------------------------------
    # Calculate numeric summaries
    # --------------------------------------------------

    numeric_summary = []

    for col, stats in numeric_stats.items():

        count = stats["count"]

        if count == 0:
            continue

        mean = stats["sum"] / count

        variance = max(
            0.0,
            stats["sum_sq"] / count - mean ** 2
        )

        numeric_summary.append({
            "feature": col,
            "count": count,
            "mean": mean,
            "std": np.sqrt(variance),
            "min": stats["min"],
            "max": stats["max"],
        })

    numeric_summary = pd.DataFrame(
        numeric_summary
    )

    class_summary = pd.DataFrame([
        {
            "cloud_type_label": cls,
            "count": class_counts[cls],
            "percentage": (
                100 * class_counts[cls] / total_rows
                if total_rows else 0
            ),
        }
        for cls in CLASSES
    ])

    btd_consistency["mean_absolute_difference"] = (
        btd_consistency["mean_absolute_difference_sum"]
        / btd_consistency["rows_compared"]
        if btd_consistency["rows_compared"] else 0
    )

    btd_consistency.pop(
        "mean_absolute_difference_sum"
    )

    return {
        "rows": total_rows,
        "missing": missing_counts,
        "numeric": numeric_summary,
        "classes": class_summary,
        "btd": btd_consistency,
        "suspicious_pair_count": suspicious_pair_count,
        "thresholds": threshold_counts,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if not ORIGINAL_FILE.exists():
        raise FileNotFoundError(ORIGINAL_FILE)

    if not CLEANED_FILE.exists():
        raise FileNotFoundError(CLEANED_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    original = validate_dataset(ORIGINAL_FILE)
    cleaned = validate_dataset(CLEANED_FILE)

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    original["numeric"].to_csv(
        OUTPUT_DIR / "original_numeric_summary.csv",
        index=False
    )

    cleaned["numeric"].to_csv(
        OUTPUT_DIR / "cleaned_numeric_summary.csv",
        index=False
    )

    original["classes"].to_csv(
        OUTPUT_DIR / "original_class_distribution.csv",
        index=False
    )

    cleaned["classes"].to_csv(
        OUTPUT_DIR / "cleaned_class_distribution.csv",
        index=False
    )

    pd.DataFrame([
        {
            "dataset": "original",
            "rows": original["rows"],
            "missing_values": sum(
                original["missing"].values()
            ),
            "suspicious_pair_count":
                original["suspicious_pair_count"],
        },
        {
            "dataset": "cleaned",
            "rows": cleaned["rows"],
            "missing_values": sum(
                cleaned["missing"].values()
            ),
            "suspicious_pair_count":
                cleaned["suspicious_pair_count"],
        }
    ]).to_csv(
        OUTPUT_DIR / "dataset_comparison.csv",
        index=False
    )

    pd.DataFrame([
        {
            "feature": col,
            "original_min": original["numeric"].set_index(
                "feature"
            ).loc[col, "min"],
            "cleaned_min": cleaned["numeric"].set_index(
                "feature"
            ).loc[col, "min"],
            "original_max": original["numeric"].set_index(
                "feature"
            ).loc[col, "max"],
            "cleaned_max": cleaned["numeric"].set_index(
                "feature"
            ).loc[col, "max"],
        }
        for col in NUMERIC_COLUMNS
    ]).to_csv(
        OUTPUT_DIR / "extrema_comparison.csv",
        index=False
    )

    pd.DataFrame([
        {
            "dataset": "original",
            **original["btd"],
        },
        {
            "dataset": "cleaned",
            **cleaned["btd"],
        },
    ]).to_csv(
        OUTPUT_DIR / "btd_consistency_comparison.csv",
        index=False
    )

    pd.DataFrame([
        {
            "dataset": dataset_name,
            **result["thresholds"],
        }
        for dataset_name, result in [
            ("original", original),
            ("cleaned", cleaned),
        ]
    ]).to_csv(
        OUTPUT_DIR / "threshold_comparison.csv",
        index=False
    )

    # --------------------------------------------------
    # Validation checks
    # --------------------------------------------------

    checks = {
        "original_row_count_correct":
            original["rows"] == EXPECTED_ORIGINAL_ROWS,

        "cleaned_row_count_correct":
            cleaned["rows"] == EXPECTED_CLEANED_ROWS,

        "exactly_six_rows_removed":
            original["rows"] - cleaned["rows"]
            == EXPECTED_REMOVED_ROWS,

        "suspicious_pair_absent_from_cleaned":
            cleaned["suspicious_pair_count"] == 0,

        "no_missing_values_in_cleaned":
            sum(cleaned["missing"].values()) == 0,

        "all_cleaned_rows_btd_consistent":
            cleaned["btd"]["rows_difference_gt_0_05"] == 0,
    }

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    report = [
        "CLEANED DATASET VALIDATION REPORT",
        "=================================",
        "",
        f"Original rows: {original['rows']:,}",
        f"Cleaned rows: {cleaned['rows']:,}",
        f"Rows removed: {original['rows'] - cleaned['rows']:,}",
        "",
        "VALIDATION CHECKS",
        "-----------------",
    ]

    for name, result in checks.items():
        report.append(
            f"{name}: {'PASS' if result else 'FAIL'}"
        )

    report.extend([
        "",
        "CLEANED MISSING VALUES",
        "---------------------",
    ])

    for col, count in cleaned["missing"].items():
        report.append(f"{col}: {count:,}")

    report.extend([
        "",
        "CLEANED NUMERIC SUMMARY",
        "-----------------------",
        cleaned["numeric"].to_string(index=False),
        "",
        "CLEANED CLASS DISTRIBUTION",
        "--------------------------",
        cleaned["classes"].to_string(index=False),
        "",
        "CLEANED BTD CONSISTENCY",
        "-----------------------",
        str(cleaned["btd"]),
        "",
        "CLEANED THRESHOLD COUNTS",
        "------------------------",
        str(cleaned["thresholds"]),
        "",
        "Original dataset was not modified.",
    ])

    report.append("")

    report.append(
        "OVERALL VALIDATION: "
        + (
            "PASS"
            if all(checks.values())
            else "FAIL - INVESTIGATE FAILED CHECKS"
        )
    )

    report_path = (
        OUTPUT_DIR / "CLEANED_DATASET_VALIDATION_REPORT.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n".join(report))

    print(
        f"\nAll results saved under: {OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()

