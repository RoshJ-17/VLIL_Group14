"""
Cloud-type preprocessing and chronological train/validation/test splitting.

Processing order:
1. Load the cleaned CSV.
2. Check exact duplicate records and duplicate pixel keys.
3. Check timestamp format and chronological coverage.
4. Examine feature distributions and potential remaining outliers.
5. Check feature correlations.
6. Verify all seven target classes.
7. Prepare chronological train/validation/test splits.
8. Save preprocessing summaries and split metadata.

Original cleaned CSV is never modified.

All outputs are saved under:
eda_results/cloudType/preprocessing/
"""

from pathlib import Path
from datetime import datetime
import json
import re
import warnings

import numpy as np
import pandas as pd

# DuckDB is used for disk-based duplicate analysis on the large CSV.
try:
    import duckdb
except ImportError:
    raise SystemExit(
        "DuckDB is required for scalable duplicate checking.\n"
        "Install it in your aviation environment using:\n"
        "pip install duckdb"
    )


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path("/home/m_prathilothamai/aviation_weather_project")

INPUT_FILE = (
    PROJECT_DIR
    / "eda_results"
    / "cloudType"
    / "cleaning"
    / "cloudtype_final_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "eda_results"
    / "cloudType"
    / "preprocessing"
)

SPLIT_DIR = OUTPUT_DIR / "splits"
TEMP_DIR = OUTPUT_DIR / "temporary"

CHUNK_SIZE = 500_000

# Sampling is used for feature distributions, quantiles and correlations.
# These analyses do not require loading all 53 million rows into memory.
SAMPLE_PER_CHUNK = 10_000
RANDOM_SEED = 42

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

EXPECTED_CLASSES = [
    "clear",
    "liquid_low",
    "liquid_middle",
    "liquid_high",
    "ice_low",
    "ice_middle",
    "ice_high",
]

FEATURES = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
]

TARGET = "cloud_type_label"
TIMESTAMP = "timestamp"

REQUIRED_COLUMNS = [
    TIMESTAMP,
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
    TARGET,
]


# ============================================================
# INITIALIZATION
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

print("=" * 75)
print("CLOUD-TYPE DATA PREPROCESSING")
print("=" * 75)

print(f"\nInput:  {INPUT_FILE}")
print(f"Output: {OUTPUT_DIR}")

# Read only the CSV header initially.
header = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()

missing_columns = [
    col for col in REQUIRED_COLUMNS if col not in header
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if len(header) != len(REQUIRED_COLUMNS):
    print(
        "\nWARNING: The input has additional columns. "
        "Only the expected seven columns will be used."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_chunks():
    """Read the CSV in manageable chunks."""
    return pd.read_csv(
        INPUT_FILE,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE,
        dtype={
            TIMESTAMP: "string",
            TARGET: "string",
        },
    )


def parse_timestamp(value):
    """
    Expected format: YYYYDDD_HHMM

    Example:
    2022182_0340

    YYYY = year
    DDD  = day of year
    HHMM = time
    """
    if pd.isna(value):
        return pd.NaT

    value = str(value)

    if not re.fullmatch(r"\d{7}_\d{4}", value):
        return pd.NaT

    try:
        return datetime.strptime(value, "%Y%j_%H%M")
    except ValueError:
        return pd.NaT


def write_json(data, path):
    """Save a JSON report."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


# ============================================================
# STEP 1. LOAD THE CLEANED CSV
# ============================================================

print("\n" + "=" * 75)
print("STEP 1: LOAD CLEANED CSV")
print("=" * 75)

row_count = 0
timestamp_counts = {}
class_counts = {label: 0 for label in EXPECTED_CLASSES}
unknown_class_counts = {}

malformed_timestamp_count = 0
valid_timestamp_count = 0

timestamp_min = None
timestamp_max = None

timestamp_order_violations = 0
previous_timestamp = None

feature_sample_parts = []

numeric_nonfinite_counts = {
    feature: 0 for feature in FEATURES
}

numeric_missing_counts = {
    feature: 0 for feature in FEATURES
}

print("\nScanning dataset in chunks...")

for chunk_number, chunk in enumerate(load_chunks(), start=1):

    row_count += len(chunk)

    # Convert features to numeric values.
    for feature in FEATURES:
        chunk[feature] = pd.to_numeric(
            chunk[feature],
            errors="coerce",
        )

        numeric_missing_counts[feature] += int(
            chunk[feature].isna().sum()
        )

        numeric_nonfinite_counts[feature] += int(
            (~np.isfinite(chunk[feature])).sum()
            - chunk[feature].isna().sum()
        )

    # Count target classes.
    counts = chunk[TARGET].value_counts(dropna=False)

    for label, count in counts.items():
        if label in class_counts:
            class_counts[label] += int(count)
        else:
            key = "<MISSING>" if pd.isna(label) else str(label)
            unknown_class_counts[key] = (
                unknown_class_counts.get(key, 0) + int(count)
            )

    # Timestamp validation and coverage.
    chunk_timestamps = chunk[TIMESTAMP].dropna().unique()

    for value in chunk_timestamps:
        if value not in timestamp_counts:
            parsed = parse_timestamp(value)

            if pd.isna(parsed):
                malformed_timestamp_count += 1
            else:
                valid_timestamp_count += 1
                timestamp_counts[value] = parsed

                if timestamp_min is None or parsed < timestamp_min:
                    timestamp_min = parsed

                if timestamp_max is None or parsed > timestamp_max:
                    timestamp_max = parsed

    missing_timestamp_rows = int(chunk[TIMESTAMP].isna().sum())
    malformed_timestamp_count += missing_timestamp_rows

    # Check input ordering across chunks and within chunks.
    parsed_series = pd.to_datetime(
        chunk[TIMESTAMP].map(parse_timestamp),
        errors="coerce",
    )

    valid_sequence = parsed_series.dropna()

    if len(valid_sequence) > 0:

        if previous_timestamp is not None:
            if valid_sequence.iloc[0] < previous_timestamp:
                timestamp_order_violations += 1

        timestamp_order_violations += int(
            (valid_sequence.diff().dropna() < pd.Timedelta(0)).sum()
        )

        previous_timestamp = valid_sequence.iloc[-1]

    # Keep a sample for feature distribution and correlation analysis.
    valid_numeric = chunk[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(valid_numeric) > 0:
        sample = valid_numeric.sample(
            n=min(SAMPLE_PER_CHUNK, len(valid_numeric)),
            random_state=RANDOM_SEED + chunk_number,
        )
        feature_sample_parts.append(sample)

    if chunk_number % 10 == 0:
        print(
            f"Processed {row_count:,} rows "
            f"across {chunk_number} chunks..."
        )


print(f"\nTotal rows: {row_count:,}")
print(f"Unique valid timestamps: {len(timestamp_counts):,}")
print(f"Malformed timestamp values/rows: {malformed_timestamp_count:,}")

if row_count == 0:
    raise ValueError("The input dataset is empty.")

# Save the feature sample in memory for later analysis.
if feature_sample_parts:
    feature_sample = pd.concat(
        feature_sample_parts,
        ignore_index=True,
    )
else:
    raise ValueError("No valid numeric feature rows were found.")

# Save timestamp coverage.
sorted_timestamps = sorted(timestamp_counts.values())

timestamp_coverage = {
    "total_rows": row_count,
    "unique_valid_timestamps": len(sorted_timestamps),
    "malformed_timestamp_count": malformed_timestamp_count,
    "minimum_timestamp": (
        str(timestamp_min) if timestamp_min is not None else None
    ),
    "maximum_timestamp": (
        str(timestamp_max) if timestamp_max is not None else None
    ),
    "input_order_violations": timestamp_order_violations,
}

write_json(
    timestamp_coverage,
    OUTPUT_DIR / "timestamp_coverage.json",
)

print("\nTimestamp coverage:")
print(json.dumps(timestamp_coverage, indent=4))


# ============================================================
# STEP 2. CHECK DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 75)
print("STEP 2: CHECK DUPLICATE RECORDS")
print("=" * 75)

print(
    "\nRunning disk-based duplicate analysis with DuckDB. "
    "This can take some time for a 53-million-row CSV."
)

con = duckdb.connect(database=":memory:")

con.execute(
    f"SET temp_directory = '{TEMP_DIR.as_posix()}'"
)

con.execute("SET preserve_insertion_order = false")

csv_path_sql = INPUT_FILE.as_posix().replace("'", "''")

con.execute(f"""
    CREATE VIEW cloud_data AS
    SELECT *
    FROM read_csv_auto(
        '{csv_path_sql}',
        header = true,
        sample_size = 100000,
        ignore_errors = false
    )
""")

# Exact duplicate records across all seven columns.
exact_duplicate_groups = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT *
        FROM cloud_data
        GROUP BY ALL
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]

exact_duplicate_excess_rows = con.execute("""
    SELECT COALESCE(SUM(record_count - 1), 0)
    FROM (
        SELECT COUNT(*) AS record_count
        FROM cloud_data
        GROUP BY ALL
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]

# Duplicate pixel keys:
# Same timestamp, latitude and longitude.
# These may indicate repeated pixel observations, but are not
# automatically considered erroneous records.
pixel_duplicate_groups = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT timestamp, lat, lon
        FROM cloud_data
        GROUP BY timestamp, lat, lon
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]

pixel_duplicate_excess_rows = con.execute("""
    SELECT COALESCE(SUM(record_count - 1), 0)
    FROM (
        SELECT COUNT(*) AS record_count
        FROM cloud_data
        GROUP BY timestamp, lat, lon
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]

duplicate_summary = {
    "exact_duplicate_groups": int(exact_duplicate_groups),
    "exact_duplicate_excess_rows": int(exact_duplicate_excess_rows),
    "duplicate_pixel_key_groups": int(pixel_duplicate_groups),
    "duplicate_pixel_key_excess_rows": int(pixel_duplicate_excess_rows),
    "pixel_key_definition": [
        "timestamp",
        "lat",
        "lon",
    ],
    "note": (
        "Repeated timestamp-lat-lon keys are reported separately. "
        "They are not automatically removed."
    ),
}

write_json(
    duplicate_summary,
    OUTPUT_DIR / "duplicate_summary.json",
)

print("\nDuplicate summary:")
print(json.dumps(duplicate_summary, indent=4))

con.close()


# ============================================================
# STEP 3. TIMESTAMP FORMAT AND CHRONOLOGICAL COVERAGE
# ============================================================

print("\n" + "=" * 75)
print("STEP 3: TIMESTAMP FORMAT AND CHRONOLOGICAL COVERAGE")
print("=" * 75)

timestamp_frequency = pd.Series(
    list(timestamp_counts.values())
).value_counts().sort_index()

# Count rows per acquisition timestamp in a second pass.
timestamp_row_counts = {}

for chunk in load_chunks():
    counts = chunk[TIMESTAMP].value_counts()

    for timestamp_value, count in counts.items():
        if timestamp_value in timestamp_counts:
            timestamp_row_counts[timestamp_value] = (
                timestamp_row_counts.get(timestamp_value, 0)
                + int(count)
            )

timestamp_rows_df = pd.DataFrame([
    {
        "timestamp": timestamp,
        "datetime": parsed,
        "row_count": timestamp_row_counts.get(timestamp, 0),
    }
    for timestamp, parsed in timestamp_counts.items()
])

timestamp_rows_df = timestamp_rows_df.sort_values("datetime")

timestamp_rows_df.to_csv(
    OUTPUT_DIR / "rows_per_timestamp.csv",
    index=False,
)

# Date-level coverage.
timestamp_rows_df["date"] = (
    timestamp_rows_df["datetime"].dt.strftime("%Y-%m-%d")
)

daily_coverage = (
    timestamp_rows_df.groupby("date")["row_count"]
    .sum()
    .reset_index()
)

daily_coverage.to_csv(
    OUTPUT_DIR / "daily_timestamp_coverage.csv",
    index=False,
)

# Gaps between acquisition timestamps.
timestamp_rows_df["gap_minutes"] = (
    timestamp_rows_df["datetime"].diff().dt.total_seconds() / 60
)

timestamp_rows_df.to_csv(
    OUTPUT_DIR / "timestamp_gaps.csv",
    index=False,
)

print("\nTimestamp format expected: YYYYDDD_HHMM")
print(f"Valid unique timestamps: {len(timestamp_rows_df):,}")
print(f"Malformed timestamp rows/values: {malformed_timestamp_count:,}")

if len(timestamp_rows_df) > 0:
    print(
        "Chronological range:",
        timestamp_rows_df["datetime"].min(),
        "to",
        timestamp_rows_df["datetime"].max(),
    )

    print("\nLargest acquisition gaps:")
    print(
        timestamp_rows_df.nlargest(10, "gap_minutes")[
            ["timestamp", "datetime", "gap_minutes"]
        ].to_string(index=False)
    )


# ============================================================
# STEP 4. FEATURE DISTRIBUTIONS AND POTENTIAL OUTLIERS
# ============================================================

print("\n" + "=" * 75)
print("STEP 4: FEATURE DISTRIBUTIONS AND POTENTIAL OUTLIERS")
print("=" * 75)

distribution_summary = feature_sample.describe(
    percentiles=[
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    ]
).T

distribution_summary.to_csv(
    OUTPUT_DIR / "feature_distribution_summary.csv"
)

# IQR-based potential outlier screening using the sample.
outlier_summary = []

for feature in FEATURES:

    values = feature_sample[feature].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (values < lower_bound)
        | (values > upper_bound)
    )

    outlier_summary.append({
        "feature": feature,
        "sample_count": len(values),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_iqr_bound": lower_bound,
        "upper_iqr_bound": upper_bound,
        "potential_outlier_count_in_sample": int(outlier_mask.sum()),
        "potential_outlier_percentage_in_sample": (
            float(outlier_mask.mean() * 100)
        ),
        "sample_minimum": values.min(),
        "sample_maximum": values.max(),
    })

outlier_df = pd.DataFrame(outlier_summary)

outlier_df.to_csv(
    OUTPUT_DIR / "potential_outlier_summary.csv",
    index=False,
)

print("\nFeature distribution summary:")
print(distribution_summary)

print("\nPotential outlier summary:")
print(outlier_df.to_string(index=False))

print(
    "\nNOTE: IQR outliers are statistical flags, not automatic errors. "
    "No observations are removed by this step."
)


# ============================================================
# STEP 5. FEATURE CORRELATIONS
# ============================================================

print("\n" + "=" * 75)
print("STEP 5: FEATURE CORRELATIONS")
print("=" * 75)

correlation_matrix = feature_sample[FEATURES].corr(
    method="pearson"
)

correlation_matrix.to_csv(
    OUTPUT_DIR / "feature_correlation_matrix.csv"
)

print("\nPearson correlation matrix:")
print(correlation_matrix.round(5))

# Long-form correlation table.
correlation_pairs = []

for i, feature_a in enumerate(FEATURES):
    for feature_b in FEATURES[i + 1:]:
        value = correlation_matrix.loc[feature_a, feature_b]

        correlation_pairs.append({
            "feature_1": feature_a,
            "feature_2": feature_b,
            "pearson_correlation": value,
            "absolute_correlation": abs(value),
        })

correlation_pairs_df = pd.DataFrame(correlation_pairs)

correlation_pairs_df = correlation_pairs_df.sort_values(
    "absolute_correlation",
    ascending=False,
)

correlation_pairs_df.to_csv(
    OUTPUT_DIR / "feature_correlation_pairs.csv",
    index=False,
)

print("\nStrongest feature correlations:")
print(correlation_pairs_df.to_string(index=False))

print(
    "\nNOTE: Correlations are calculated on the sampled feature rows. "
    "They describe linear association, not causation."
)


# ============================================================
# STEP 6. VERIFY ALL SEVEN TARGET CLASSES
# ============================================================

print("\n" + "=" * 75)
print("STEP 6: TARGET CLASS VALIDATION")
print("=" * 75)

class_summary = pd.DataFrame([
    {
        "class_label": label,
        "count": class_counts[label],
        "percentage": (
            class_counts[label] / row_count * 100
        ),
        "valid": class_counts[label] > 0,
    }
    for label in EXPECTED_CLASSES
])

class_summary.to_csv(
    OUTPUT_DIR / "target_class_summary.csv",
    index=False,
)

all_classes_present = all(
    class_counts[label] > 0
    for label in EXPECTED_CLASSES
)

no_unknown_labels = len(unknown_class_counts) == 0

print("\nTarget class counts:")
print(class_summary.to_string(index=False))

print("\nUnexpected or missing labels:")
print(unknown_class_counts)

print(f"\nAll seven classes present: {all_classes_present}")
print(f"No unexpected/missing labels: {no_unknown_labels}")

if not all_classes_present:
    raise ValueError(
        "At least one expected target class is absent. "
        "Chronological splitting stopped."
    )

if not no_unknown_labels:
    raise ValueError(
        "Unexpected or missing target labels were found. "
        "Review target_class_summary and correct the labels before splitting."
    )


# ============================================================
# STEP 7. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 75)
print("STEP 7: CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 75)

if malformed_timestamp_count > 0:
    raise ValueError(
        "Malformed or missing timestamps were found. "
        "Resolve them before chronological splitting."
    )

if len(timestamp_rows_df) < 3:
    raise ValueError(
        "Not enough unique timestamps for three chronological splits."
    )

# Use whole acquisition timestamps as indivisible groups.
# Boundaries are selected by cumulative row count.
timestamp_rows_df = timestamp_rows_df.sort_values(
    "datetime"
).reset_index(drop=True)

total_rows = int(timestamp_rows_df["row_count"].sum())

train_target = total_rows * TRAIN_FRACTION
validation_target = total_rows * (
    TRAIN_FRACTION + VALIDATION_FRACTION
)

cumulative_rows = timestamp_rows_df["row_count"].cumsum()

train_end_index = int(
    np.searchsorted(
        cumulative_rows.to_numpy(),
        train_target,
        side="left",
    )
)

validation_end_index = int(
    np.searchsorted(
        cumulative_rows.to_numpy(),
        validation_target,
        side="left",
    )
)

train_end_index = max(0, min(train_end_index, len(timestamp_rows_df) - 3))
validation_end_index = max(
    train_end_index + 1,
    min(validation_end_index, len(timestamp_rows_df) - 2),
)

train_end_datetime = timestamp_rows_df.loc[
    train_end_index, "datetime"
]

validation_end_datetime = timestamp_rows_df.loc[
    validation_end_index, "datetime"
]

print("\nChronological boundaries:")
print(f"Train ends:      {train_end_datetime}")
print(f"Validation ends: {validation_end_datetime}")
print(
    "Test begins:     ",
    timestamp_rows_df.loc[
        validation_end_index + 1, "datetime"
    ],
)

# Determine each timestamp's split.
def assign_split(parsed_timestamp):
    if parsed_timestamp <= train_end_datetime:
        return "train"

    if parsed_timestamp <= validation_end_datetime:
        return "validation"

    return "test"


timestamp_to_split = {
    timestamp: assign_split(parsed)
    for timestamp, parsed in timestamp_counts.items()
}

# Prepare output paths.
split_paths = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}

# Remove old split outputs so repeated runs do not append to old files.
for path in split_paths.values():
    if path.exists():
        path.unlink()

split_row_counts = {
    "train": 0,
    "validation": 0,
    "test": 0,
}

split_class_counts = {
    split: {label: 0 for label in EXPECTED_CLASSES}
    for split in ["train", "validation", "test"]
}

split_first_timestamp = {
    "train": None,
    "validation": None,
    "test": None,
}

split_last_timestamp = {
    "train": None,
    "validation": None,
    "test": None,
}

print("\nWriting chronological split CSV files...")

for chunk_number, chunk in enumerate(load_chunks(), start=1):

    parsed = chunk[TIMESTAMP].map(parse_timestamp)

    chunk["_split"] = parsed.map(
        lambda value: (
            assign_split(value)
            if not pd.isna(value)
            else None
        )
    )

    for split_name in ["train", "validation", "test"]:

        subset = chunk[chunk["_split"] == split_name].copy()

        if subset.empty:
            continue

        split_row_counts[split_name] += len(subset)

        # Class counts for each split.
        counts = subset[TARGET].value_counts()

        for label, count in counts.items():
            split_class_counts[split_name][label] += int(count)

        # Chronological boundaries represented in each split.
        valid_times = parsed.loc[subset.index].dropna()

        if len(valid_times) > 0:
            current_min = valid_times.min()
            current_max = valid_times.max()

            if (
                split_first_timestamp[split_name] is None
                or current_min < split_first_timestamp[split_name]
            ):
                split_first_timestamp[split_name] = current_min

            if (
                split_last_timestamp[split_name] is None
                or current_max > split_last_timestamp[split_name]
            ):
                split_last_timestamp[split_name] = current_max

        subset = subset.drop(columns=["_split"])

        # Preserve original seven-column structure.
        subset = subset[REQUIRED_COLUMNS]

        subset.to_csv(
            split_paths[split_name],
            mode="a",
            header=not split_paths[split_name].exists(),
            index=False,
        )

    if chunk_number % 10 == 0:
        print(
            f"Split processing: {chunk_number} chunks completed."
        )


# ============================================================
# STEP 8. SAVE PREPROCESSING SUMMARIES AND SPLIT METADATA
# ============================================================

print("\n" + "=" * 75)
print("STEP 8: SAVE PREPROCESSING SUMMARIES AND SPLIT METADATA")
print("=" * 75)

split_metadata = {
    "input_file": str(INPUT_FILE),
    "output_directory": str(OUTPUT_DIR),
    "total_input_rows": row_count,
    "split_method": "Chronological split by unique acquisition timestamp",
    "split_fractions_requested": {
        "train": TRAIN_FRACTION,
        "validation": VALIDATION_FRACTION,
        "test": TEST_FRACTION,
    },
    "split_boundaries": {
        "train_last_timestamp": str(train_end_datetime),
        "validation_last_timestamp": str(validation_end_datetime),
    },
    "splits": {},
}

for split_name in ["train", "validation", "test"]:

    split_count = split_row_counts[split_name]

    split_metadata["splits"][split_name] = {
        "file": str(split_paths[split_name]),
        "rows": split_count,
        "percentage_of_total": (
            split_count / row_count * 100
        ),
        "first_timestamp": str(
            split_first_timestamp[split_name]
        ),
        "last_timestamp": str(
            split_last_timestamp[split_name]
        ),
        "class_counts": split_class_counts[split_name],
    }

# Save split class counts in tabular form.
split_class_table = []

for split_name in ["train", "validation", "test"]:

    split_total = split_row_counts[split_name]

    for label in EXPECTED_CLASSES:

        count = split_class_counts[split_name][label]

        split_class_table.append({
            "split": split_name,
            "class_label": label,
            "count": count,
            "percentage_within_split": (
                count / split_total * 100
                if split_total > 0 else 0
            ),
        })

pd.DataFrame(split_class_table).to_csv(
    OUTPUT_DIR / "split_class_distribution.csv",
    index=False,
)

write_json(
    split_metadata,
    OUTPUT_DIR / "split_metadata.json",
)

# Verify row totals.
split_total_rows = sum(split_row_counts.values())

split_verification = {
    "input_rows": row_count,
    "train_rows": split_row_counts["train"],
    "validation_rows": split_row_counts["validation"],
    "test_rows": split_row_counts["test"],
    "total_split_rows": split_total_rows,
    "row_count_matches": split_total_rows == row_count,
    "all_splits_nonempty": all(
        split_row_counts[name] > 0
        for name in ["train", "validation", "test"]
    ),
    "chronological_order": (
        split_last_timestamp["train"]
        <= split_first_timestamp["validation"]
        and
        split_last_timestamp["validation"]
        <= split_first_timestamp["test"]
    ),
}

write_json(
    split_verification,
    OUTPUT_DIR / "split_verification.json",
)

# Final report.
report_lines = [
    "CLOUD-TYPE PREPROCESSING REPORT",
    "=" * 60,
    "",
    f"Input dataset: {INPUT_FILE}",
    f"Total rows: {row_count:,}",
    "",
    "DUPLICATE CHECK",
    "-" * 60,
    f"Exact duplicate groups: {exact_duplicate_groups:,}",
    f"Exact duplicate excess rows: {exact_duplicate_excess_rows:,}",
    f"Duplicate pixel-key groups: {pixel_duplicate_groups:,}",
    f"Duplicate pixel-key excess rows: {pixel_duplicate_excess_rows:,}",
    "",
    "TIMESTAMP CHECK",
    "-" * 60,
    f"Valid unique timestamps: {len(timestamp_rows_df):,}",
    f"Malformed timestamp rows/values: {malformed_timestamp_count:,}",
    f"Minimum timestamp: {timestamp_min}",
    f"Maximum timestamp: {timestamp_max}",
    f"Input-order violations: {timestamp_order_violations:,}",
    "",
    "TARGET VALIDATION",
    "-" * 60,
    f"All seven classes present: {all_classes_present}",
    f"No unknown/missing labels: {no_unknown_labels}",
    "",
    "CHRONOLOGICAL SPLITS",
    "-" * 60,
]

for split_name in ["train", "validation", "test"]:
    report_lines.extend([
        f"{split_name.upper()} rows: {split_row_counts[split_name]:,}",
        (
            f"{split_name.upper()} percentage: "
            f"{split_row_counts[split_name] / row_count * 100:.3f}%"
        ),
        (
            f"{split_name.upper()} date range: "
            f"{split_first_timestamp[split_name]} "
            f"to {split_last_timestamp[split_name]}"
        ),
        "",
    ])

report_lines.extend([
    "FINAL VERIFICATION",
    "-" * 60,
    f"Split row count matches input: {split_verification['row_count_matches']}",
    f"All splits nonempty: {split_verification['all_splits_nonempty']}",
    f"Chronological ordering verified: {split_verification['chronological_order']}",
])

with open(
    OUTPUT_DIR / "preprocessing_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write("\n".join(report_lines))

print("\n" + "\n".join(report_lines))

# Temporary directory is retained only if DuckDB created temporary
# files there. It can be inspected or cleared after successful execution.

print("\n" + "=" * 75)
print("PREPROCESSING COMPLETED")
print("=" * 75)

print(f"\nAll outputs saved under:\n{OUTPUT_DIR}")

print("\nImportant output files:")
for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file() and TEMP_DIR not in path.parents:
        print(path.relative_to(OUTPUT_DIR))

if not all(split_verification.values()):
    raise RuntimeError(
        "One or more final split verification checks failed. "
        "Inspect split_verification.json."
    )

print("\nAll final split verification checks passed.")
