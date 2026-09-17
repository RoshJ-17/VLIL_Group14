
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODIS_FILE = Path("modisPixel.csv")
OUTPUT_FILE = Path("era5_required_cells.csv")
DB_FILE = Path("era5_required_cells_temp.sqlite")

CHUNK_SIZE = 250_000
MODIS_COLUMNS = ["timestamp", "lat", "lon"]

# ERA5 single-levels regular grid resolution
ERA5_RESOLUTION = 0.25

if not MODIS_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {MODIS_FILE}")

# --------------------------------------------------
# Create a disk-backed deduplication database
# This avoids holding all 64 million rows in RAM.
# --------------------------------------------------
conn = sqlite3.connect(DB_FILE)

conn.execute("""
    CREATE TABLE IF NOT EXISTS required_cells (
        valid_time TEXT NOT NULL,
        era5_lat REAL NOT NULL,
        era5_lon REAL NOT NULL,
        PRIMARY KEY (valid_time, era5_lat, era5_lon)
    )
""")
conn.commit()

rows_scanned = 0
rows_with_coordinates = 0
chunks_processed = 0

print("Building required ERA5 time-grid-cell combinations...")
print("Input:", MODIS_FILE)

try:
    for chunk in pd.read_csv(
        MODIS_FILE,
        usecols=MODIS_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):
        chunks_processed += 1
        rows_scanned += len(chunk)

        # Parse MODIS timestamps and round to nearest ERA5 hour.
        timestamps = pd.to_datetime(
    chunk["timestamp"],
    format="%Y%j_%H%M",
    errors="coerce"
).dt.round("h")

        lat = pd.to_numeric(chunk["lat"], errors="coerce")
        lon = pd.to_numeric(chunk["lon"], errors="coerce")

        valid = (
            timestamps.notna()
            & lat.notna()
            & lon.notna()
            & np.isfinite(lat)
            & np.isfinite(lon)
            & lat.between(6, 37)
            & lon.between(68, 97)
        )

        if not valid.any():
            continue

        rows_with_coordinates += int(valid.sum())

        # Map coordinates to the nearest ERA5 0.25-degree grid.
        # ERA5 latitude and longitude are expressed in degrees.
        era5_lat = (
            np.round(lat.loc[valid].to_numpy() / ERA5_RESOLUTION)
            * ERA5_RESOLUTION
        )
        era5_lon = (
            np.round(lon.loc[valid].to_numpy() / ERA5_RESOLUTION)
            * ERA5_RESOLUTION
        )

        valid_times = timestamps.loc[valid].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        ).to_numpy()

        current = pd.DataFrame({
            "valid_time": valid_times,
            "era5_lat": np.round(era5_lat, 2),
            "era5_lon": np.round(era5_lon, 2),
        }).drop_duplicates()

        # Insert only combinations not already present.
        conn.executemany(
            """
            INSERT OR IGNORE INTO required_cells
            (valid_time, era5_lat, era5_lon)
            VALUES (?, ?, ?)
            """,
            current.itertuples(index=False, name=None),
        )
        conn.commit()

        if chunks_processed % 20 == 0:
            unique_count = conn.execute(
                "SELECT COUNT(*) FROM required_cells"
            ).fetchone()[0]

            print(
                f"Chunks: {chunks_processed} | "
                f"Rows scanned: {rows_scanned:,} | "
                f"Unique time-cell combinations: {unique_count:,}"
            )

    # Export the final deduplicated table.
    result = pd.read_sql_query(
        """
        SELECT valid_time, era5_lat, era5_lon
        FROM required_cells
        ORDER BY valid_time, era5_lat, era5_lon
        """,
        conn,
    )

    result.to_csv(OUTPUT_FILE, index=False)

    print("\n========== SUMMARY ==========")
    print(f"MODIS rows scanned: {rows_scanned:,}")
    print(f"Rows with valid coordinates/time: {rows_with_coordinates:,}")
    print(f"Unique ERA5 time-cell combinations: {len(result):,}")
    print(f"Unique ERA5 timestamps: {result['valid_time'].nunique():,}")
    print(
        "Unique ERA5 grid cells:",
        result[["era5_lat", "era5_lon"]].drop_duplicates().shape[0],
    )
    print(f"Output saved to: {OUTPUT_FILE.resolve()}")

finally:
    conn.close()

