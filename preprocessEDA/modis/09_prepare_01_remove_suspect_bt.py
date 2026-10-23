
import os
import pandas as pd

INPUT_CSV = "modisPixel_with_GPM.csv"
OUTPUT_CSV = "modisPixelGPM_training_clean.csv"
CHUNK_SIZE = 500_000

# Exact suspicious values identified during EDA.
BT11_SUSPECT = 486.8
BT12_SUSPECT = 495.75
BTD_SUSPECT = -8.95

rows_read = 0
rows_removed = 0
rows_written = 0
first_chunk = True

print("Preparing cleaned training dataset...", flush=True)
print(f"Input:  {INPUT_CSV}")
print(f"Output: {OUTPUT_CSV}")

# Refuse to overwrite the original input.
if os.path.abspath(INPUT_CSV) == os.path.abspath(OUTPUT_CSV):
    raise ValueError("Output must not overwrite the original dataset.")

# Write each chunk directly to the output file to limit memory usage.
for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE):
    rows_read += len(chunk)

    bt11 = pd.to_numeric(chunk["BT_11"], errors="coerce")
    bt12 = pd.to_numeric(chunk["BT_12"], errors="coerce")
    btd = pd.to_numeric(chunk["BTD"], errors="coerce")

    suspect = (
        bt11.eq(BT11_SUSPECT)
        & bt12.eq(BT12_SUSPECT)
        & btd.eq(BTD_SUSPECT)
    )

    removed = int(suspect.sum())
    rows_removed += removed

    cleaned = chunk.loc[~suspect]
    rows_written += len(cleaned)

    cleaned.to_csv(
        OUTPUT_CSV,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
    )
    first_chunk = False

    if rows_read % 5_000_000 < CHUNK_SIZE:
        print(
            f"Read: {rows_read:,} | Removed: {rows_removed:,} "
            f"| Written: {rows_written:,}",
            flush=True,
        )

print("\n===== CLEANING SUMMARY =====")
print(f"Rows read:    {rows_read:,}")
print(f"Rows removed: {rows_removed:,}")
print(f"Rows written: {rows_written:,}")
print(f"Output file:  {OUTPUT_CSV}")

if rows_removed != 17:
    print(
        "\nWARNING: Expected to remove 17 rows, but removed "
        f"{rows_removed}. Verify the matching rule before using the output."
    )
else:
    print("\nRemoval count matches the 17 records identified during EDA.")

print("Original dataset was not modified.")

