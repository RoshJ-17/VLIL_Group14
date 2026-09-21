from pathlib import Path
from collections import defaultdict
import hashlib
import json
import time

import pandas as pd
import cdsapi
from netCDF4 import Dataset


# ============================================================
# CONFIGURATION: CHANGE ONLY MACHINE_ID ON EACH COMPUTER
# ============================================================

MACHINE_ID = 5          
NUMBER_OF_MACHINES = 6

INPUT_FILE = Path("era5_required_cells.csv")

# Separate output folder for each person
OUTPUT_DIR = Path(f"data/ERA5/cbh_targeted/person_{MACHINE_ID}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "download_log.csv"
FAILED_FILE = OUTPUT_DIR / "failed_downloads.json"

DATASET = "reanalysis-era5-single-levels"
VARIABLE = "cloud_base_height"

LAT_MIN, LAT_MAX = 6.0, 37.0
LON_MIN, LON_MAX = 68.0, 97.0
TILE_SIZE = 2.0

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15



# ============================================================
# NETCDF FILE CHECK
# ============================================================

def valid_netcdf_file(path):
    """Return True if the file opens as NetCDF and contains data variables."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Dataset(path, "r") as ds:
            return len(ds.variables) > 0
    except Exception:
        return False


# ============================================================
# LOAD REQUIRED CELLS
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing inventory: {INPUT_FILE}")

cells = pd.read_csv(INPUT_FILE)

required = {"valid_time", "era5_lat", "era5_lon"}
if not required.issubset(cells.columns):
    raise ValueError(f"Input must contain columns: {sorted(required)}")

cells["valid_time"] = pd.to_datetime(
    cells["valid_time"], errors="coerce"
)
cells["era5_lat"] = pd.to_numeric(cells["era5_lat"], errors="coerce")
cells["era5_lon"] = pd.to_numeric(cells["era5_lon"], errors="coerce")

cells = cells.dropna(subset=list(required)).copy()
cells = cells.drop_duplicates(
    ["valid_time", "era5_lat", "era5_lon"]
)

if cells.empty:
    raise ValueError("No valid required cells found.")

if (
    (cells["valid_time"].dt.minute != 0).any()
    or (cells["valid_time"].dt.second != 0).any()
):
    raise ValueError("All requested times must be hourly.")

print("Unique required combinations:", f"{len(cells):,}")


# ============================================================
# ASSIGN EACH GRID CELL TO A 2 x 2 DEGREE TILE
# ============================================================

def get_tile(lat, lon):
    lat_index = int((lat - LAT_MIN) // TILE_SIZE)
    lon_index = int((lon - LON_MIN) // TILE_SIZE)

    south = LAT_MIN + lat_index * TILE_SIZE
    west = LON_MIN + lon_index * TILE_SIZE

    north = min(south + TILE_SIZE, LAT_MAX)
    east = min(west + TILE_SIZE, LON_MAX)

    return tuple(round(v, 2) for v in (north, west, south, east))


tile_values = [
    get_tile(lat, lon)
    for lat, lon in zip(cells["era5_lat"], cells["era5_lon"])
]

cells[["north", "west", "south", "east"]] = pd.DataFrame(
    tile_values, index=cells.index
)

cells["date"] = cells["valid_time"].dt.strftime("%Y-%m-%d")
cells["year"] = cells["valid_time"].dt.strftime("%Y")
cells["month"] = cells["valid_time"].dt.strftime("%m")
cells["day"] = cells["valid_time"].dt.strftime("%d")
cells["hour"] = cells["valid_time"].dt.strftime("%H:%M")

tile_cols = ["north", "west", "south", "east"]

# First collect required hours for each tile and date.
tile_days = (
    cells.groupby(tile_cols + ["date"], as_index=False)
    .agg(
        year=("year", "first"),
        month=("month", "first"),
        day=("day", "first"),
        hours=("hour", lambda s: sorted(set(s))),
    )
)

# Group dates only when their required hour lists match.
grouped = defaultdict(list)

for row in tile_days.itertuples(index=False):
    key = (
        row.north, row.west, row.south, row.east,
        row.year, row.month, tuple(row.hours)
    )
    grouped[key].append(row.day)

jobs = []

for key, days in grouped.items():
    north, west, south, east, year, month, hours = key
    days = sorted(set(days))

    tile_name = (
        f"N{north:05.1f}_W{west:05.1f}_"
        f"S{south:05.1f}_E{east:05.1f}"
    )

    # A deterministic job identity shared by all three machines.
    identity = json.dumps(
        [year, month, tile_name, days, list(hours)],
        separators=(",", ":"),
    )
    job_id = hashlib.sha256(identity.encode()).hexdigest()[:16]

    out = OUTPUT_DIR / (
        f"cbh_{year}_{month}_{tile_name}_"
        f"{job_id}.nc"
    )

    jobs.append({
        "id": job_id,
        "identity": identity,
        "year": year,
        "month": month,
        "days": days,
        "hours": list(hours),
        "tile_name": tile_name,
        "area": [north, west, south, east],
        "output": out,
    })

jobs.sort(key=lambda j: j["identity"])

# Stable partition: each job belongs to exactly one machine.
def assigned_machine(job):
    digest = hashlib.sha256(job["identity"].encode()).hexdigest()
    return int(digest[:8], 16) % NUMBER_OF_MACHINES + 1

all_jobs = jobs
jobs = [j for j in all_jobs if assigned_machine(j) == MACHINE_ID]

print("All grouped requests:", len(all_jobs))
print(f"Requests assigned to machine {MACHINE_ID}:", len(jobs))


# ============================================================
# READ THIS MACHINE'S SUCCESS LOG
# ============================================================

if LOG_FILE.exists():
    previous_log = pd.read_csv(LOG_FILE)
else:
    previous_log = pd.DataFrame(
        columns=["job_id", "output_file", "status", "message"]
    )

client = cdsapi.Client()
new_log = []
failures = []


# ============================================================
# DOWNLOAD ASSIGNED REQUESTS
# ============================================================

jobs_to_run = jobs
print(f"Starting full download: {len(jobs_to_run)} requests assigned to this machine.")

for index, job in enumerate(jobs_to_run, start=1):
    output = job["output"]

    # Only skip an already-downloaded NetCDF file. Existing GRIB files
    # are deliberately ignored and will not prevent NetCDF downloads.
    if output.exists() and valid_netcdf_file(output):
        print(f"[{index}/{len(jobs_to_run)}] Already completed: {output.name}")
        continue

    request = {
        "product_type": ["reanalysis"],
        "variable": [VARIABLE],
        "year": [job["year"]],
        "month": [job["month"]],
        "day": job["days"],
        "time": job["hours"],
        "area": job["area"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    temporary = output.with_suffix(".part.nc")
    error_message = ""
    success = False

    print(
        f"\n[{index}/{len(jobs_to_run)}] Downloading "
        f"{job['year']}-{job['month']} {job['tile_name']}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if temporary.exists():
                temporary.unlink()

            client.retrieve(DATASET, request, str(temporary))

            if not valid_netcdf_file(temporary):
                raise RuntimeError(
                    "Downloaded file is not a readable NetCDF file or contains no variables."
                )

            temporary.replace(output)
            success = True
            break

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"Attempt {attempt} failed: {error_message}")

            if temporary.exists():
                temporary.unlink()

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

    status = "success" if success else "failed"

    new_log.append({
        "job_id": job["id"],
        "output_file": str(output),
        "status": status,
        "message": error_message,
    })

    if not success:
        failures.append({
            "job_id": job["id"],
            "request": request,
            "error": error_message,
        })

    combined = pd.concat(
        [previous_log, pd.DataFrame(new_log)],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(
        subset=["job_id"], keep="last"
    )
    combined.to_csv(LOG_FILE, index=False)

    FAILED_FILE.write_text(json.dumps(failures, indent=2))

print("\n========== SUMMARY ==========")
print("Machine:", MACHINE_ID)
print("Assigned requests:", len(jobs))
print("Requests processed this run:", len(jobs_to_run))
print("New successes:", sum(r["status"] == "success" for r in new_log))
print("Failures:", sum(r["status"] == "failed" for r in new_log))
print("Output:", OUTPUT_DIR.resolve())
print("Log:", LOG_FILE.resolve())


