
import os
import sqlite3
import pandas as pd

INPUT = "modisPixelGPM_training_clean.csv"
OUTDIR = "eda_results"
DB_PATH = os.path.join(OUTDIR, "duplicate_check.sqlite")
CHUNK_SIZE = 500_000

os.makedirs(OUTDIR, exist_ok=True)

# Start with a fresh temporary database.
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE pixel_keys (
        timestamp TEXT NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        occurrences INTEGER NOT NULL,
        PRIMARY KEY (timestamp, lat, lon)
    )
""")

rows_scanned = 0
extra_occurrences = 0

try:
    for chunk in pd.read_csv(
        INPUT,
        usecols=["timestamp", "lat", "lon"],
        chunksize=CHUNK_SIZE,
    ):
        rows_scanned += len(chunk)

        chunk = chunk.dropna(subset=["timestamp", "lat", "lon"])
        keys = chunk.drop_duplicates(["timestamp", "lat", "lon"])

        # Count occurrences within each chunk first.
        local_counts = (
            chunk.groupby(["timestamp", "lat", "lon"], sort=False)
            .size()
            .reset_index(name="n")
        )

        for row in local_counts.itertuples(index=False, name=None):
            timestamp, lat, lon, n = row
            cur = conn.execute(
                """
                INSERT INTO pixel_keys
                    (timestamp, lat, lon, occurrences)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(timestamp, lat, lon)
                DO UPDATE SET occurrences = occurrences + excluded.occurrences
                """,
                (str(timestamp), float(lat), float(lon), int(n)),
            )
            extra_occurrences += max(int(n) - 1, 0)

        conn.commit()

        if rows_scanned % 5_000_000 < CHUNK_SIZE:
            print(f"Rows scanned: {rows_scanned:,}", flush=True)

    result = conn.execute("""
        SELECT
            COUNT(*) AS unique_keys,
            COALESCE(SUM(occurrences), 0) AS total_keyed_rows,
            COALESCE(SUM(CASE WHEN occurrences > 1 THEN 1 ELSE 0 END), 0)
                AS duplicated_keys,
            COALESCE(SUM(CASE WHEN occurrences > 1
                              THEN occurrences - 1 ELSE 0 END), 0)
                AS extra_duplicate_rows,
            COALESCE(MAX(occurrences), 0) AS max_occurrences
        FROM pixel_keys
    """).fetchone()

    report = {
        "rows_scanned": rows_scanned,
        "unique_timestamp_lat_lon_keys": result[0],
        "total_keyed_rows": result[1],
        "keys_appearing_multiple_times": result[2],
        "extra_duplicate_rows": result[3],
        "maximum_occurrences_of_one_key": result[4],
        "duplicate_key_definition": "exact timestamp + lat + lon",
    }

    out = os.path.join(OUTDIR, "global_duplicate_report.txt")
    with open(out, "w") as f:
        for key, value in report.items():
            f.write(f"{key}: {value}\n")

    print("\n===== GLOBAL DUPLICATE REPORT =====")
    for key, value in report.items():
        print(f"{key}: {value}")

    print(f"\nSaved report: {out}")

finally:
    conn.close()

