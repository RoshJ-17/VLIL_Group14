
from pathlib import Path
from collections import defaultdict
import time
import json

import pandas as pd
import cdsapi


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = Path("era5_required_cells.csv")

# Use a separate directory so existing downloads are untouched.
OUTPUT_DIR = Path("data/ERA5/cbh_targeted")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "download_log.csv"
FAILED_FILE = OUTPUT_DIR / "failed_downloads.json"

DATASET = "reanalysis-era5-single-levels"

VARIABLE = "cloud_base_height"

# Requested geographic area: India bounding box.
LAT_MIN, LAT_MAX = 6.0, 37.0
LON_MIN, LON_MAX = 68.0, 97.0

# Never request more than 2 degrees in either dimension.
TILE_SIZE = 2.0

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15

# Start with True to test a small subset first.
# After verifying the downloaded files, change to False.
TEST_MODE = True
TEST_MAX_REQUESTS = 2


# ============================================================
# 2. VALIDATE INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Required inventory not found: {INPUT_FILE}"
    )

required_columns = ["valid_time", "era5_lat", "era5_lon"]

cells = pd.read_csv(INPUT_FILE)

missing_columns = set(required_columns) - set(cells.columns)
if missing_columns:
    raise ValueError(
        f"Missing columns in {INPUT_FILE}: {sorted(missing_columns)}"
    )

cells["valid_time"] = pd.to_datetime(
    cells["valid_time"], errors="coerce"
)
cells["era5_lat"] = pd.to_numeric(
    cells["era5_lat"], errors="coerce"
)
cells["era5_lon"] = pd.to_numeric(
    cells["era5_lon"], errors="coerce"
)

cells = cells.dropna(subset=required_columns).copy()

# ERA5 requests use UTC hours.
if (
    (cells["valid_time"].dt.minute != 0).any()
    or (cells["valid_time"].dt.second != 0).any()
):
    raise ValueError(
        "Inventory contains non-hourly timestamps. "
        "Review the timestamp rounding before downloading."
    )

# Retain only the requested Indian region.
cells = cells[
    cells["era5_lat"].between(LAT_MIN, LAT_MAX)
    & cells["era5_lon"].between(LON_MIN, LON_MAX)
].copy()

cells = cells.drop_duplicates(
    ["valid_time", "era5_lat", "era5_lon"]
)

if cells.empty:
    raise ValueError("No valid required grid cells found.")

print("Required time-cell combinations:", f"{len(cells):,}")
print("Required timestamps:", cells["valid_time"].nunique())
print(
    "Required grid cells:",
    cells[["era5_lat", "era5_lon"]].drop_duplicates().shape[0],
)


# ============================================================
# 3. ASSIGN CELLS TO 2-DEGREE TILES
# ============================================================

def tile_bounds(lat, lon):
    """
    Assign a required grid point to a tile no larger than
    2 degrees by 2 degrees, aligned to the India bounding box.
    """
    lat_index = min(
        int((lat - LAT_MIN) // TILE_SIZE),
        int((LAT_MAX - LAT_MIN - 1e-9) // TILE_SIZE),
    )
    lon_index = min(
        int((lon - LON_MIN) // TILE_SIZE),
        int((LON_MAX - LON_MIN - 1e-9) // TILE_SIZE),
    )

    south = LAT_MIN + lat_index * TILE_SIZE
    west = LON_MIN + lon_index * TILE_SIZE

    north = min(south + TILE_SIZE, LAT_MAX)
    east = min(west + TILE_SIZE, LON_MAX)

    return (
        round(north, 2),
        round(west, 2),
        round(south, 2),
        round(east, 2),
    )


bounds = cells.apply(
    lambda row: tile_bounds(
        row["era5_lat"], row["era5_lon"]
    ),
    axis=1,
)

cells[["north", "west", "south", "east"]] = pd.DataFrame(
    bounds.tolist(), index=cells.index
)

# Each tile/time combination needs CBH only if at least one
# required grid point falls inside that tile at that time.
cells["date"] = cells["valid_time"].dt.strftime("%Y-%m-%d")
cells["year"] = cells["valid_time"].dt.strftime("%Y")
cells["month"] = cells["valid_time"].dt.strftime("%m")
cells["day"] = cells["valid_time"].dt.strftime("%d")
cells["hour"] = cells["valid_time"].dt.strftime("%H:%M")

tile_columns = ["north", "west", "south", "east"]

# Build unique requested hours per tile and date.
tile_day_hours = (
    cells.groupby(tile_columns + ["date"], as_index=False)
    .agg(
        year=("year", "first"),
        month=("month", "first"),
        day=("day", "first"),
        hours=("hour", lambda x: sorted(set(x))),
    )
)

# Group dates within the same tile and month when their
# exact required hour lists match. This avoids requesting
# every hour on every selected day.
jobs_grouped = defaultdict(list)

for row in tile_day_hours.itertuples(index=False):
    hour_key = tuple(row.hours)

    key = (
        row.north,
        row.west,
        row.south,
        row.east,
        row.year,
        row.month,
        hour_key,
    )

    jobs_grouped[key].append(row.day)

jobs = []

for key, days in jobs_grouped.items():
    north, west, south, east, year, month, hours = key

    days = sorted(set(days))

    tile_name = (
        f"N{north:05.1f}_W{west:05.1f}_"
        f"S{south:05.1f}_E{east:05.1f}"
    )

    day_tag = "-".join(days)
    hour_tag = "-".join(h[:2] for h in hours)

    output_file = OUTPUT_DIR / (
        f"cbh_{year}_{month}_{tile_name}"
        f"_D{day_tag}_H{hour_tag}.grib"
    )

    jobs.append({
        "year": year,
        "month": month,
        "days": days,
        "hours": list(hours),
        "area": [north, west, south, east],
        "tile_name": tile_name,
        "output_file": output_file,
    })

jobs.sort(
    key=lambda job: (
        job["year"],
        job["month"],
        job["tile_name"],
        job["days"],
    )
)

print("Unique tile-day groups:", len(tile_day_hours))
print("Grouped CDS requests:", len(jobs))

if TEST_MODE:
    jobs = jobs[:TEST_MAX_REQUESTS]
    print(f"\nTEST MODE: running only {len(jobs)} requests.")


# ============================================================
# 4. RESUME LOG
# ============================================================

if LOG_FILE.exists():
    old_log = pd.read_csv(LOG_FILE)
    completed = set(
        old_log.loc[
            old_log["status"] == "success", "output_file"
        ].astype(str)
    )
else:
    old_log = pd.DataFrame(
        columns=["output_file", "status", "message"]
    )
    completed = set()

client = cdsapi.Client()
new_log = []
failed_jobs = []


# ============================================================
# 5. DOWNLOAD REQUIRED DATA
# ============================================================

for index, job in enumerate(jobs, start=1):
    output_file = job["output_file"]
    output_string = str(output_file)

    # Skip previously validated successes recorded in the log.
    if (
        output_string in completed
        and output_file.exists()
        and output_file.stat().st_size > 0
    ):
        print(f"[{index}/{len(jobs)}] Already completed: {output_file.name}")
        continue

    # Do not trust non-empty files automatically; an interrupted
    # download may leave an incomplete file. Redownload unlogged
    # files to a temporary path and replace only on success.
    temporary_file = output_file.with_suffix(".part.grib")

    request = {
        "product_type": ["reanalysis"],
        "variable": [VARIABLE],
        "year": [job["year"]],
        "month": [job["month"]],
        "day": job["days"],
        "time": job["hours"],
        "area": job["area"],
        "data_format": "grib",
        "download_format": "unarchived",
    }

    print(
        f"\n[{index}/{len(jobs)}] "
        f"{job['year']}-{job['month']} "
        f"days={','.join(job['days'])} "
        f"hours={','.join(job['hours'])}"
    )
    print("Area N/W/S/E:", request["area"])

    success = False
    error_message = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if temporary_file.exists():
                temporary_file.unlink()

            client.retrieve(
                DATASET,
                request,
                str(temporary_file),
            )

            if (
                not temporary_file.exists()
                or temporary_file.stat().st_size == 0
            ):
                raise RuntimeError(
                    "Download returned no data or an empty file."
                )

            # Basic GRIB marker check. Full validation will be
            # performed later with a GRIB reader.
            with temporary_file.open("rb") as handle:
                start = handle.read(4)

            if start != b"GRIB":
                raise RuntimeError(
                    "Output does not begin with a GRIB marker."
                )

            temporary_file.replace(output_file)
            success = True
            break

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed: "
                f"{error_message}"
            )

            if temporary_file.exists():
                temporary_file.unlink()

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

    status = "success" if success else "failed"

    new_log.append({
        "output_file": output_string,
        "status": status,
        "message": error_message,
    })

    if success:
        print(
            "Saved:",
            output_file,
            f"({output_file.stat().st_size:,} bytes)",
        )
    else:
        failed_jobs.append({
            "output_file": output_string,
            "request": request,
            "error": error_message,
        })
        print("FAILED:", output_file.name)

    # Persist progress after every request.
    combined_log = pd.concat(
        [old_log, pd.DataFrame(new_log)],
        ignore_index=True,
    )
    combined_log = combined_log.drop_duplicates(
        subset=["output_file"], keep="last"
    )
    combined_log.to_csv(LOG_FILE, index=False)

    FAILED_FILE.write_text(
        json.dumps(failed_jobs, indent=2)
    )

print("\n========== DOWNLOAD SUMMARY ==========")
print("Requests attempted:", len(jobs))
print("New successful requests:", sum(
    row["status"] == "success" for row in new_log
))
print("Failed requests:", sum(
    row["status"] == "failed" for row in new_log
))
print("Download directory:", OUTPUT_DIR.resolve())
print("Log:", LOG_FILE.resolve())
print("Failures:", FAILED_FILE.resolve())

if TEST_MODE:
    print(
        "\nTEST MODE is enabled. If the test files are valid, "
        "set TEST_MODE = False to run the remaining requests."
    )

