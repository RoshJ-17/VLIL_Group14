import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT_FILE = Path("modis_cloudtype_labeled_full.csv")
OUTPUT_FILE = Path("cloudtype_final.csv")

CHUNK_SIZE = 250_000

# Model input features
FEATURES = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
]

TIMESTAMP_COL = "timestamp"
TARGET_COL = "cloud_type_label"

OUTPUT_COLUMNS = [TIMESTAMP_COL] + FEATURES + [TARGET_COL]

NUMERIC_COLUMNS = FEATURES

# Keep clear, liquid-cloud, and ice-cloud classes only.
# Remove undetermined_* and unknown.
EXPECTED_CLASSES = {
    "clear",
    "liquid_low",
    "liquid_middle",
    "liquid_high",
    "ice_low",
    "ice_middle",
    "ice_high",
}

# --------------------------------------------------
# Main processing
# --------------------------------------------------
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE.resolve()}"
        )

    required_columns = OUTPUT_COLUMNS

    # Remove an old output to prevent appending duplicate rows.
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print(f"Removed previous output: {OUTPUT_FILE}")

    total_input_rows = 0
    excluded_target_rows = 0
    dropped_invalid_rows = 0
    total_output_rows = 0

    class_counts = {}
    first_write = True

    print(f"Input:  {INPUT_FILE.resolve()}")
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print("\nStarting processing...\n")

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=required_columns,
        chunksize=CHUNK_SIZE,
    ):
        total_input_rows += len(chunk)

        # Keep only the seven intended classes.
        valid_target = chunk[TARGET_COL].isin(EXPECTED_CLASSES)
        excluded_target_rows += int((~valid_target).sum())

        chunk = chunk.loc[valid_target].copy()

        if chunk.empty:
            print(f"Read {total_input_rows:,} input rows")
            continue

        # Convert model features to numeric values.
        for col in NUMERIC_COLUMNS:
            chunk[col] = pd.to_numeric(
                chunk[col], errors="coerce"
            )

        # Remove rows missing timestamp, features, or target.
        before_cleaning = len(chunk)

        chunk = chunk.dropna(subset=OUTPUT_COLUMNS)

        # Remove rows containing infinite numeric feature values.
        finite_mask = np.isfinite(
            chunk[NUMERIC_COLUMNS].to_numpy(dtype=float)
        ).all(axis=1)

        chunk = chunk.loc[finite_mask].copy()

        dropped_invalid_rows += before_cleaning - len(chunk)

        if chunk.empty:
            print(f"Read {total_input_rows:,} input rows")
            continue

        # Defensive check: no excluded target class can enter output.
        unexpected = set(chunk[TARGET_COL].unique()) - EXPECTED_CLASSES
        if unexpected:
            raise ValueError(
                f"Unexpected target classes found: {unexpected}"
            )

        # Accumulate target counts.
        counts = chunk[TARGET_COL].value_counts()
        for label, count in counts.items():
            class_counts[label] = (
                class_counts.get(label, 0) + int(count)
            )

        # Write each chunk; never load the full dataset into RAM.
        chunk[OUTPUT_COLUMNS].to_csv(
            OUTPUT_FILE,
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False
        total_output_rows += len(chunk)

        print(
            f"Read {total_input_rows:,} rows | "
            f"Saved {total_output_rows:,} rows"
        )

    if total_output_rows == 0:
        raise RuntimeError(
            "No rows were saved. Check the input labels and columns."
        )

    # --------------------------------------------------
    # Final summary and checks
    # --------------------------------------------------
    print("\n========== PROCESSING COMPLETE ==========")
    print(f"Input rows read:             {total_input_rows:,}")
    print(f"Excluded target rows:        {excluded_target_rows:,}")
    print(f"Dropped invalid feature rows:{dropped_invalid_rows:>10,}")
    print(f"Final output rows:            {total_output_rows:,}")
    print(f"Output file: {OUTPUT_FILE.resolve()}")

    print("\nFinal target-class counts:")
    counts_series = pd.Series(class_counts).reindex(
        sorted(EXPECTED_CLASSES), fill_value=0
    )
    print(counts_series.to_string())

    print("\nFinal target-class percentages:")
    percentages = counts_series / total_output_rows * 100
    print(percentages.round(3).astype(str).add("%").to_string())

    # All seven classes should be present.
    missing_classes = EXPECTED_CLASSES - set(class_counts)
    if missing_classes:
        raise ValueError(
            f"Expected classes absent from output: {missing_classes}"
        )

    # Verify that the row accounting balances.
    accounted_rows = (
        excluded_target_rows
        + dropped_invalid_rows
        + total_output_rows
    )

    if accounted_rows != total_input_rows:
        raise ValueError(
            "Row accounting mismatch: "
            f"{accounted_rows:,} accounted vs "
            f"{total_input_rows:,} read"
        )

    print("\nValidation passed.")
    print("Only the seven intended classes are included.")


if __name__ == "__main__":
    main()
