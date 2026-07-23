import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
import re

# ============================================================
# CONFIG
# ============================================================

MODIS_FILE = "modis_pixel.csv"
GPM_DIR = Path("data/GPM")
OUTPUT_FILE = "modis_gpm_precip.csv"


# ============================================================
# 1. LOAD MODIS DATA
# ============================================================

print("=" * 70)
print("GPM PRECIPITATION-RATE DATASET CREATION")
print("=" * 70)

print("\nLoading MODIS data...")

df = pd.read_csv(MODIS_FILE)

print(f"Loaded {len(df):,} MODIS rows")
print(f"Unique timestamps: {df['timestamp'].nunique()}")


# ============================================================
# 2. PARSE MODIS TIMESTAMP
# ============================================================

def parse_modis_timestamp(ts):
    return datetime.strptime(str(ts), "%Y%j_%H%M")


print("\nParsing MODIS timestamps...")

df["datetime"] = df["timestamp"].apply(parse_modis_timestamp)


# ============================================================
# 3. BUILD INDEX OF GPM FILES
# ============================================================

print("\nIndexing GPM files...")

gpm_files = {}

for path in GPM_DIR.glob("*.nc4"):

    name = path.name

    # Example:
    # 3B-HHR.MS.MRG.3IMERG.20221119-S033000-E035959.0210.V07B.HDF5.SUB.nc4

    match = re.search(
        r"3IMERG\.(\d{8})-S(\d{6})-E(\d{6})\.",
        name
    )

    if not match:
        continue

    date_str = match.group(1)
    start_time = match.group(2)

    # Example:
    # 20221119 + 0330
    # -> 20221119_0330

    key = date_str + "_" + start_time[:4]

    gpm_files[key] = path


print(f"Indexed {len(gpm_files)} GPM files")


# ============================================================
# 4. DETERMINE GPM 30-MINUTE INTERVAL
# ============================================================

def get_gpm_key(dt):

    # MODIS observation before :30
    # belongs to :00 GPM interval
    #
    # MODIS observation at/after :30
    # belongs to :30 GPM interval

    minute = 0 if dt.minute < 30 else 30

    start = dt.replace(
        minute=minute,
        second=0,
        microsecond=0
    )

    return start.strftime("%Y%m%d_%H%M")


print("\nDetermining GPM intervals...")

df["gpm_key"] = df["datetime"].apply(get_gpm_key)


# ============================================================
# 5. INITIALIZE OUTPUT ARRAYS
# ============================================================

gpm_precip_rate = np.full(len(df), np.nan)
gpm_error = np.full(len(df), np.nan)
gpm_quality = np.full(len(df), np.nan)

missing_gpm_file_rows = 0
outside_gpm_area = 0

gpm_files_used = set()


# ============================================================
# 6. PROCESS EACH GPM TIME INTERVAL
# ============================================================

print("\nMatching MODIS pixels to GPM...")

# Group MODIS observations by their corresponding GPM interval.
# Each GPM file is opened only once.

for group_number, (gpm_key, group) in enumerate(
    df.groupby("gpm_key"),
    start=1
):

    path = gpm_files.get(gpm_key)

    if path is None:

        print(
            f"WARNING: No GPM file found for {gpm_key} "
            f"({len(group):,} MODIS rows)"
        )

        missing_gpm_file_rows += len(group)

        continue

    gpm_files_used.add(gpm_key)

    print(
        f"[{group_number}] {gpm_key} "
        f"→ {path.name} "
        f"→ {len(group):,} MODIS rows"
    )

    # --------------------------------------------------------
    # OPEN GPM FILE
    # --------------------------------------------------------

    ds = Dataset(path, "r")

    lat_array = np.asarray(
        ds.variables["lat"][:]
    )

    lon_array = np.asarray(
        ds.variables["lon"][:]
    )

    precipitation = np.asarray(
        ds.variables["precipitation"][0, :, :]
    )

    random_error = np.asarray(
        ds.variables["randomError"][0, :, :]
    )

    quality = np.asarray(
        ds.variables["precipitationQualityIndex"][0, :, :]
    )

    ds.close()


    # --------------------------------------------------------
    # GPM GRID-CELL BOUNDARIES
    # --------------------------------------------------------

    lat_resolution = np.median(
        np.diff(lat_array)
    )

    lon_resolution = np.median(
        np.diff(lon_array)
    )

    lat_min = (
        lat_array.min()
        - lat_resolution / 2
    )

    lat_max = (
        lat_array.max()
        + lat_resolution / 2
    )

    lon_min = (
        lon_array.min()
        - lon_resolution / 2
    )

    lon_max = (
        lon_array.max()
        + lon_resolution / 2
    )


    # --------------------------------------------------------
    # EXTRACT MODIS COORDINATES FOR THIS GROUP
    # --------------------------------------------------------

    group_indices = group.index.to_numpy()

    modis_lat = df.loc[
        group_indices,
        "lat"
    ].to_numpy()

    modis_lon = df.loc[
        group_indices,
        "lon"
    ].to_numpy()


    # --------------------------------------------------------
    # CHECK GPM SPATIAL COVERAGE
    # --------------------------------------------------------

    inside = (
        (modis_lat >= lat_min) &
        (modis_lat <= lat_max) &
        (modis_lon >= lon_min) &
        (modis_lon <= lon_max)
    )

    outside_count = np.count_nonzero(~inside)

    outside_gpm_area += outside_count


    if outside_count == len(group):
        continue


    # Only process pixels inside GPM coverage

    valid_indices = group_indices[inside]

    valid_lat = modis_lat[inside]
    valid_lon = modis_lon[inside]


    # --------------------------------------------------------
    # FIND NEAREST GPM GRID CELLS
    # --------------------------------------------------------

    # Vectorized nearest-neighbour lookup.
    #
    # This is much faster than running argmin() separately
    # for every MODIS row.

    lat_indices = np.searchsorted(
        lat_array,
        valid_lat
    )

    lat_indices = np.clip(
        lat_indices,
        1,
        len(lat_array) - 1
    )

    left_lat = lat_array[
        lat_indices - 1
    ]

    right_lat = lat_array[
        lat_indices
    ]

    choose_right_lat = (
        np.abs(right_lat - valid_lat)
        <
        np.abs(left_lat - valid_lat)
    )

    lat_indices = (
        lat_indices
        - 1
        + choose_right_lat.astype(int)
    )


    lon_indices = np.searchsorted(
        lon_array,
        valid_lon
    )

    lon_indices = np.clip(
        lon_indices,
        1,
        len(lon_array) - 1
    )

    left_lon = lon_array[
        lon_indices - 1
    ]

    right_lon = lon_array[
        lon_indices
    ]

    choose_right_lon = (
        np.abs(right_lon - valid_lon)
        <
        np.abs(left_lon - valid_lon)
    )

    lon_indices = (
        lon_indices
        - 1
        + choose_right_lon.astype(int)
    )


    # --------------------------------------------------------
    # EXTRACT GPM VALUES
    # --------------------------------------------------------

    gpm_precip_rate[valid_indices] = (
        precipitation[
            lon_indices,
            lat_indices
        ]
    )

    gpm_error[valid_indices] = (
        random_error[
            lon_indices,
            lat_indices
        ]
    )

    gpm_quality[valid_indices] = (
        quality[
            lon_indices,
            lat_indices
        ]
    )


# ============================================================
# 7. ADD GPM VARIABLES
# ============================================================

print("\nAdding GPM variables...")

df["GPM_randomError"] = gpm_error

df["GPM_precipitationQualityIndex"] = gpm_quality

# TARGET
#
# IMERG precipitation variable represents precipitation rate.
# Units: mm/hr

df["GPM_precipitation_rate"] = gpm_precip_rate


# ============================================================
# 8. REMOVE TEMPORARY COLUMNS
# ============================================================

df.drop(
    columns=[
        "datetime",
        "gpm_key"
    ],
    inplace=True
)


# ============================================================
# 9. SAVE FINAL DATASET
# ============================================================

print("\nSaving final dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 10. VALIDATION
# ============================================================

print()
print("=" * 70)
print("GPM PRECIPITATION-RATE DATASET COMPLETE")
print("=" * 70)

print(f"Output file: {OUTPUT_FILE}")

print(f"Rows: {len(df):,}")

print(
    f"MODIS timestamps: "
    f"{df['timestamp'].nunique():,}"
)

print(
    f"GPM files used: "
    f"{len(gpm_files_used):,}"
)

print(
    f"Missing GPM file rows: "
    f"{missing_gpm_file_rows:,}"
)

print(
    f"Outside GPM coverage: "
    f"{outside_gpm_area:,}"
)


# ============================================================
# 11. TARGET VALIDATION
# ============================================================

print()
print("GPM precipitation rate statistics (mm/hr):")

print(
    df["GPM_precipitation_rate"].describe()
)


print()
print(
    "Missing precipitation rate:",
    df["GPM_precipitation_rate"].isna().sum()
)

print(
    "Missing randomError:",
    df["GPM_randomError"].isna().sum()
)

print(
    "Missing quality index:",
    df["GPM_precipitationQualityIndex"].isna().sum()
)


# ============================================================
# 12. ZERO / NONZERO PRECIPITATION
# ============================================================

valid_precip = df[
    "GPM_precipitation_rate"
].dropna()

print()
print(
    "Zero precipitation-rate rows:",
    (valid_precip == 0).sum()
)

print(
    "Nonzero precipitation-rate rows:",
    (valid_precip > 0).sum()
)


# ============================================================
# 13. SHOW FINAL COLUMNS
# ============================================================

print()
print("Final columns:")

for i, column in enumerate(
    df.columns,
    start=1
):
    print(f"{i:2d}. {column}")


# ============================================================
# 14. FIRST 5 ROWS
# ============================================================

print()
print("First 5 rows:")

print(
    df.head().to_string()
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
