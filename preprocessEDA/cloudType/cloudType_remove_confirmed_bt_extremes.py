#!/usr/bin/env python3

"""
Remove exactly six identified suspicious brightness-temperature records.

Original CSV is never modified.

Output:
eda_results/cloudType/cleaning/
    cloudtype_final_cleaned.csv
    removed_bt_extreme_records.csv
    cleaning_report.txt
"""

from pathlib import Path
import pandas as pd

INPUT_FILE = Path("cloudtype_final.csv")

OUTPUT_DIR = (
    Path("eda_results")
    / "cloudType"
    / "cleaning"
)

OUTPUT_FILE = OUTPUT_DIR / "cloudtype_final_cleaned.csv"
REMOVED_FILE = OUTPUT_DIR / "removed_bt_extreme_records.csv"
REPORT_FILE = OUTPUT_DIR / "cleaning_report.txt"

CHUNK_SIZE = 100_000

TARGET_BT11 = 486.80
TARGET_BT12 = 495.75
TARGET_BTD = -8.95

EXPECTED_REMOVALS = 6


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. "
            "Rename or move it before running again to avoid overwriting."
        )

    if REMOVED_FILE.exists():
        raise FileExistsError(
            f"{REMOVED_FILE} already exists. "
            "Rename or move it before running again."
        )

    total_rows = 0
    removed_rows = 0

    removed_records = []

    first_chunk = True

    reader = pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        total_rows += len(chunk)

        mask = (
            (chunk["BT_11"].round(2) == TARGET_BT11)
            & (chunk["BT_12"].round(2) == TARGET_BT12)
            & (chunk["BTD"].round(2) == TARGET_BTD)
        )

        removed = chunk.loc[mask].copy()
        retained = chunk.loc[~mask].copy()

        removed_rows += len(removed)

        if not removed.empty:
            removed_records.append(removed)

        retained.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        if total_rows % 1_000_000 < CHUNK_SIZE:
            print(f"Processed {total_rows:,} rows...")

    # Verify the exact number of removed records.
    if removed_rows != EXPECTED_REMOVALS:

        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

        raise RuntimeError(
            f"Expected to remove {EXPECTED_REMOVALS} records, "
            f"but found {removed_rows}. "
            "Cleaned output was deleted. Original data is untouched."
        )

    removed_df = pd.concat(
        removed_records,
        ignore_index=True
    )

    removed_df.to_csv(
        REMOVED_FILE,
        index=False
    )

    retained_rows = total_rows - removed_rows

    report = f"""
CONFIRMED BRIGHTNESS TEMPERATURE RECORD REMOVAL
===============================================

Original dataset: {INPUT_FILE}

Total original rows: {total_rows:,}

Removal condition:
BT_11 = {TARGET_BT11}
BT_12 = {TARGET_BT12}
BTD   = {TARGET_BTD}

Expected records removed: {EXPECTED_REMOVALS}
Actual records removed:   {removed_rows}

Retained observations: {retained_rows:,}

Original CSV modified: NO

Cleaned dataset:
{OUTPUT_FILE}

Removed records log:
{REMOVED_FILE}

Only the six specified observations were removed.
No other observations were filtered.
"""

    REPORT_FILE.write_text(
        report,
        encoding="utf-8"
    )

    print(report)


if __name__ == "__main__":
    main()

