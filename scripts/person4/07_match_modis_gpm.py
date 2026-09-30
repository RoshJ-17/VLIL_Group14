from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr

# --------------------------------------------------
# Paths and settings
# --------------------------------------------------
PROJECT_DIR = Path.cwd()
MODIS_FILE = PROJECT_DIR / "modisPixel.csv"
GPM_DIR = PROJECT_DIR / "data" / "GPM" / "processed"
OUTPUT_FILE = PROJECT_DIR / "modisPixel_with_GPM.csv"

CHUNK_SIZE = 300_000
MAX_FALLBACK_MINUTES = 30

# --------------------------------------------------
# Find available GPM files and their interval times
# --------------------------------------------------
pattern = re.compile(r"^GPM_(\d{8})_(\d{4})\.nc$")

gpm_files = {}
for path in GPM_DIR.glob("GPM_*.nc"):
    match = pattern.match(path.name)
    if not match:
        continue

    date_part, time_part = match.groups()
    interval = pd.to_datetime(
        date_part + time_part, format="%Y%m%d%H%M", errors="coerce"
    )
    if not pd.isna(interval):
        gpm_files[interval] = path

if not gpm_files:
    raise RuntimeError(f"No usable GPM_YYYYMMDD_HHMM.nc files found in {GPM_DIR}")

available_times = pd.DatetimeIndex(sorted(gpm_files))
print(f"Available GPM intervals: {len(available_times)}")

if not MODIS_FILE.exists():
    raise FileNotFoundError(MODIS_FILE)

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists. Rename or move it before rerunning "
        "to avoid overwriting previous results."
    )

# --------------------------------------------------
# Read the GPM grid coordinates once
# --------------------------------------------------
sample_path = gpm_files[available_times[0]]
with xr.open_dataset(sample_path) as ds:
    required = {"lat", "lon", "GPM_precipitation_rate"}
    if not required.issubset(set(ds.variables) | set(ds.coords)):
        raise RuntimeError(f"Unexpected GPM dataset layout: {list(ds.variables)}")

    grid_lats = np.asarray(ds["lat"].values, dtype=float)
    grid_lons = np.asarray(ds["lon"].values, dtype=float)

if not (np.all(np.diff(grid_lats) > 0) and np.all(np.diff(grid_lons) > 0)):
    raise RuntimeError("Expected ascending GPM latitude and longitude coordinates.")

def nearest_indices(grid, values):
    """Indices of nearest grid centers for a sorted 1-D coordinate array."""
    values = np.asarray(values, dtype=float)
    right = np.searchsorted(grid, values, side="left")
    right = np.clip(right, 0, len(grid) - 1)
    left = np.clip(right - 1, 0, len(grid) - 1)

    use_left = np.abs(values - grid[left]) <= np.abs(values - grid[right])
    return np.where(use_left, left, right)

# --------------------------------------------------
# Choose exact half-hour interval, or nearest fallback
# --------------------------------------------------
def select_gpm_time(modis_time):
    # The required interval is the start of the MODIS pixel's
    # half-hour period: :00 for minutes 00-29, :30 for minutes 30-59.
    target = modis_time.floor("30min")

    if target in gpm_files:
        return target, "exact"

    distances = np.abs(available_times.asi8 - target.value)
    nearest_pos = int(np.argmin(distances))
    nearest_time = available_times[nearest_pos]
    difference = abs((nearest_time - target).total_seconds()) / 60.0

    if difference <= MAX_FALLBACK_MINUTES:
        return nearest_time, "nearest_time_fallback"

    return None, "no_time_within_30min"

# --------------------------------------------------
# Cache a small number of recently read GPM grids
# --------------------------------------------------
from collections import OrderedDict

GRID_CACHE_LIMIT = 4
grid_cache = OrderedDict()

def get_gpm_grid(gpm_time):
    if gpm_time in grid_cache:
        grid_cache.move_to_end(gpm_time)
        return grid_cache[gpm_time]

    path = gpm_files[gpm_time]
    with xr.open_dataset(path) as ds:
        values = np.asarray(ds["GPM_precipitation_rate"].values, dtype=np.float32)
        values = np.squeeze(values)

        if values.shape != (len(grid_lats), len(grid_lons)):
            raise RuntimeError(
                f"Unexpected grid shape in {path.name}: {values.shape}"
            )

        # Treat non-finite values and negative precipitation as missing.
        values = values.copy()
        values[~np.isfinite(values)] = np.nan
        values[values < 0] = np.nan

    grid_cache[gpm_time] = values
    while len(grid_cache) > GRID_CACHE_LIMIT:
        grid_cache.popitem(last=False)

    return values

# --------------------------------------------------
# Chunked MODIS -> GPM matching
# --------------------------------------------------
required_modis_columns = {"timestamp", "lat", "lon"}
total_rows = 0
valid_precip_rows = 0
fallback_rows = 0
missing_time_rows = 0
invalid_coordinate_rows = 0
missing_precip_rows = 0
chunk_number = 0

reader = pd.read_csv(
    MODIS_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
)

for chunk in reader:
    chunk = chunk.reset_index(drop=True)
    chunk_number += 1

    missing_columns = required_modis_columns - set(chunk.columns)
    if missing_columns:
        raise RuntimeError(f"MODIS CSV is missing columns: {sorted(missing_columns)}")

    n = len(chunk)
    total_rows += n

    # Preserve all original columns and add matching results.
    chunk["GPM_precipitation_rate"] = np.nan
    chunk["GPM_time_used"] = ""
    chunk["GPM_time_diff_minutes"] = np.nan
    chunk["GPM_match_status"] = ""

    modis_times = pd.to_datetime(
        chunk["timestamp"].astype("string"),
        format="%Y%j_%H%M",
        errors="coerce"
    )

    latitudes = pd.to_numeric(chunk["lat"], errors="coerce").to_numpy(dtype=float)
    longitudes = pd.to_numeric(chunk["lon"], errors="coerce").to_numpy(dtype=float)

    valid_time = modis_times.notna().to_numpy()
    valid_coords = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & (latitudes >= grid_lats.min() - 0.05)
        & (latitudes <= grid_lats.max() + 0.05)
        & (longitudes >= grid_lons.min() - 0.05)
        & (longitudes <= grid_lons.max() + 0.05)
    )

    chunk.loc[~valid_time, "GPM_match_status"] = "invalid_timestamp"
    chunk.loc[valid_time & ~valid_coords, "GPM_match_status"] = "invalid_coordinates"

    eligible = valid_time & valid_coords
    eligible_positions = np.flatnonzero(eligible)

    # Process each distinct MODIS timestamp in this chunk once.
    if len(eligible_positions):
        time_values = modis_times.iloc[eligible_positions]
        grouped_positions = pd.Series(
            eligible_positions, index=time_values.to_numpy()
        )

        for modis_time, positions in grouped_positions.groupby(level=0, sort=False):
            positions = positions.to_numpy(dtype=int)
            gpm_time, status = select_gpm_time(pd.Timestamp(modis_time))

            if gpm_time is None:
                chunk.loc[positions, "GPM_match_status"] = status
                missing_time_rows += len(positions)
                continue

            chunk.loc[positions, "GPM_time_used"] = gpm_time.strftime("%Y-%m-%d %H:%M:%S")
            difference_minutes = (
                gpm_time - pd.Timestamp(modis_time).floor("30min")
            ).total_seconds() / 60.0
            chunk.loc[positions, "GPM_time_diff_minutes"] = difference_minutes
            chunk.loc[positions, "GPM_match_status"] = status

            grid = get_gpm_grid(gpm_time)

            lat_idx = nearest_indices(grid_lats, latitudes[positions])
            lon_idx = nearest_indices(grid_lons, longitudes[positions])
            precipitation = grid[lat_idx, lon_idx]

            chunk.loc[positions, "GPM_precipitation_rate"] = precipitation

            if status == "nearest_time_fallback":
                fallback_rows += len(positions)

            finite = np.isfinite(precipitation)
            valid_precip_rows += int(finite.sum())
            missing_precip_rows += int((~finite).sum())

    # Write the header only for the first chunk.
    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if chunk_number == 1 else "a",
        header=(chunk_number == 1),
        index=False
    )

    invalid_coordinate_rows += int((valid_time & ~valid_coords).sum())

    print(
        f"Chunk {chunk_number}: processed {total_rows:,} rows; "
        f"valid precipitation values so far {valid_precip_rows:,}"
    )

print("\nMatching complete.")
print(f"Output: {OUTPUT_FILE}")
print(f"Total MODIS rows processed: {total_rows:,}")
print(f"Rows with valid precipitation: {valid_precip_rows:,}")
print(f"Rows matched using fallback time: {fallback_rows:,}")
print(f"Rows without an acceptable GPM time: {missing_time_rows:,}")
print(f"Rows with invalid coordinates: {invalid_coordinate_rows:,}")
print(f"Rows with missing precipitation at the selected grid cell: {missing_precip_rows:,}")
print("Original MODIS file was not modified.")
