import os
import csv
import pandas as pd

FILE = (
    "/home/m_prathilothamai/aviation_weather_project/"
    "finalIntegratedDataset.csv"
)

OUT_DIR = (
    "/home/m_prathilothamai/aviation_weather_project/"
    "missing_value_audit"
)

CHUNK_SIZE = 100_000

os.makedirs(OUT_DIR, exist_ok=True)

column_counts = None
total_rows = 0
rows_with_missing = 0
fully_empty_rows = 0

counts_file = os.path.join(OUT_DIR, "missing_by_column.csv")
details_file = os.path.join(OUT_DIR, "missing_rows_details.csv")
empty_file = os.path.join(OUT_DIR, "fully_empty_rows.csv")

print("Starting missing-value audit...")
print("Input:", FILE)
print("Output:", OUT_DIR)

with open(details_file, "w", newline="", encoding="utf-8") as details, \
     open(empty_file, "w", newline="", encoding="utf-8") as empty:

    detail_writer = csv.writer(details)
    empty_writer = csv.writer(empty)

    detail_writer.writerow([
        "data_row_number",
        "missing_value_count",
        "missing_columns"
    ])
    empty_writer.writerow(["data_row_number"])

    for chunk in pd.read_csv(
        FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        # Treat whitespace-only strings as missing too.
        for col in chunk.select_dtypes(include=["object", "string"]).columns:
            chunk[col] = chunk[col].replace(r"^\s*$", pd.NA, regex=True)

        n = len(chunk)
        start_row = total_rows + 1
        total_rows += n

        missing = chunk.isna()
        row_missing_counts = missing.sum(axis=1)
        missing_mask = row_missing_counts > 0
        empty_mask = row_missing_counts == len(chunk.columns)

        if column_counts is None:
            column_counts = pd.Series(
                0, index=chunk.columns, dtype="int64"
            )

        column_counts += missing.sum().astype("int64")

        rows_with_missing += int(missing_mask.sum())
        fully_empty_rows += int(empty_mask.sum())

        for offset in range(n):
            if missing_mask.iloc[offset]:
                missing_cols = missing.columns[
                    missing.iloc[offset].to_numpy()
                ].tolist()

                detail_writer.writerow([
                    start_row + offset,
                    int(row_missing_counts.iloc[offset]),
                    ";".join(missing_cols)
                ])

            if empty_mask.iloc[offset]:
                empty_writer.writerow([start_row + offset])

        print(
            f"Scanned {total_rows:,} rows | "
            f"Rows with missing values: {rows_with_missing:,}",
            flush=True
        )

column_counts = column_counts.rename("missing_count").to_frame()
column_counts["missing_percentage"] = (
    column_counts["missing_count"] / total_rows * 100
).round(4)

column_counts.index.name = "column"
column_counts.reset_index().to_csv(counts_file, index=False)

print("\n========== AUDIT COMPLETE ==========")
print(f"Total rows: {total_rows:,}")
print(f"Total columns: {len(column_counts)}")
print(f"Rows with missing values: {rows_with_missing:,}")
print(f"Completely empty rows: {fully_empty_rows:,}")
print(f"Rows without missing values: {total_rows - rows_with_missing:,}")
print(f"Total missing cells: {int(column_counts['missing_count'].sum()):,}")
print("\nReports saved in:", OUT_DIR)
print("1. missing_by_column.csv")
print("2. missing_rows_details.csv")
print("3. fully_empty_rows.csv")
