
from pathlib import Path
import pandas as pd
import json
import os

CSV_PATH = Path("modisPixel_with_GPM.csv")
OUT_DIR = Path("eda_results")
OUT_DIR.mkdir(exist_ok=True)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH.resolve()}")

print("=" * 60)
print("MODIS + GPM DATASET: INITIAL EDA")
print("=" * 60)

print(f"\nFile: {CSV_PATH.resolve()}")
print(f"File size: {CSV_PATH.stat().st_size / (1024**3):.2f} GB")

# Read only the header first
columns = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
print(f"\nNumber of columns: {len(columns)}")
print("\nColumn names:")
for col in columns:
    print(f"  - {col}")

print("\nScanning dataset in chunks...")

rows = 0
timestamps = set()
min_lat = float("inf")
max_lat = float("-inf")
min_lon = float("inf")
max_lon = float("-inf")
min_gpm = float("inf")
max_gpm = float("-inf")
valid_gpm = 0
missing_gpm = 0
invalid_coordinates = 0
chunk_count = 0

for chunk in pd.read_csv(
    CSV_PATH,
    chunksize=500_000,
    low_memory=False
):
    chunk_count += 1
    rows += len(chunk)

    if "timestamp" in chunk.columns:
        timestamps.update(
            chunk["timestamp"].dropna().astype(str).unique()
        )

    for col, current_min, current_max in [
        ("lat", "min_lat", "max_lat"),
        ("lon", "min_lon", "max_lon"),
    ]:
        if col in chunk.columns:
            values = pd.to_numeric(chunk[col], errors="coerce")
            values = values.dropna()
            if len(values):
                if col == "lat":
                    min_lat = min(min_lat, values.min())
                    max_lat = max(max_lat, values.max())
                else:
                    min_lon = min(min_lon, values.min())
                    max_lon = max(max_lon, values.max())

    if "GPM_precipitation_rate" in chunk.columns:
        gpm = pd.to_numeric(
            chunk["GPM_precipitation_rate"],
            errors="coerce"
        )
        valid = gpm.notna() & (gpm >= 0)
        valid_gpm += int(valid.sum())
        missing_gpm += int((~valid).sum())

        if valid.any():
            min_gpm = min(min_gpm, gpm[valid].min())
            max_gpm = max(max_gpm, gpm[valid].max())

    if {"lat", "lon"}.issubset(chunk.columns):
        lat = pd.to_numeric(chunk["lat"], errors="coerce")
        lon = pd.to_numeric(chunk["lon"], errors="coerce")
        invalid_coordinates += int(
            (~lat.between(-90, 90) | ~lon.between(-180, 180)).sum()
        )

    if chunk_count % 10 == 0:
        print(f"Chunks: {chunk_count}; rows scanned: {rows:,}")

summary = {
    "file": str(CSV_PATH.resolve()),
    "file_size_GB": round(CSV_PATH.stat().st_size / (1024**3), 3),
    "rows": rows,
    "columns": columns,
    "number_of_columns": len(columns),
    "unique_timestamps": len(timestamps),
    "first_timestamp": min(timestamps) if timestamps else None,
    "last_timestamp": max(timestamps) if timestamps else None,
    "latitude_min": min_lat if min_lat != float("inf") else None,
    "latitude_max": max_lat if max_lat != float("-inf") else None,
    "longitude_min": min_lon if min_lon != float("inf") else None,
    "longitude_max": max_lon if max_lon != float("-inf") else None,
    "valid_nonnegative_GPM_values": valid_gpm,
    "missing_or_invalid_GPM_values": missing_gpm,
    "GPM_min": min_gpm if min_gpm != float("inf") else None,
    "GPM_max": max_gpm if max_gpm != float("-inf") else None,
    "invalid_coordinate_rows": invalid_coordinates,
}

with open(OUT_DIR / "dataset_overview.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total rows: {rows:,}")
print(f"Total columns: {len(columns)}")
print(f"Unique timestamps: {len(timestamps):,}")
print(f"Latitude range: {summary['latitude_min']} to {summary['latitude_max']}")
print(f"Longitude range: {summary['longitude_min']} to {summary['longitude_max']}")
print(f"Valid nonnegative GPM values: {valid_gpm:,}")
print(f"Missing/invalid GPM values: {missing_gpm:,}")
print(f"GPM range: {summary['GPM_min']} to {summary['GPM_max']} mm/hr")
print(f"Invalid coordinate rows: {invalid_coordinates:,}")
print(f"\nReport saved to: {OUT_DIR / 'dataset_overview.json'}")
print("Initial EDA completed.")

