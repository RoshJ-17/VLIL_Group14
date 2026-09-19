#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import cdsapi
import math
import time

# =====================================================
# CONFIGURATION
# =====================================================

TIMES_FILE = "era5_required_times.csv"
OUTPUT_DIR = Path("data/ERA5/cbh_tiles")

DATASET = "reanalysis-era5-single-levels"

# India bounding box
NORTH = 37.0
SOUTH = 6.0
WEST = 68.0
EAST = 97.0

# Maximum geographic tile size
TILE_SIZE = 2.0

# Retry failed downloads
MAX_RETRIES = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# CREATE 2 x 2 DEGREE TILES
# =====================================================

def create_tiles():
    tiles = []

    south = SOUTH

    while south < NORTH:
        north = min(south + TILE_SIZE, NORTH)

        west = WEST

        while west < EAST:
            east = min(west + TILE_SIZE, EAST)

            tiles.append({
                "north": north,
                "west": west,
                "south": south,
                "east": east,
            })

            west = east

        south = north

    return tiles


# =====================================================
# MAIN
# =====================================================

def main():

    required = pd.read_csv(
        TIMES_FILE,
        parse_dates=["valid_time"]
    )

    required["valid_time"] = pd.to_datetime(
        required["valid_time"]
    )

    required["year"] = required["valid_time"].dt.strftime("%Y")
    required["month"] = required["valid_time"].dt.strftime("%m")
    required["day"] = required["valid_time"].dt.strftime("%d")
    required["hour"] = required["valid_time"].dt.strftime("%H:%M")

    tiles = create_tiles()

    monthly_groups = list(
        required.groupby(["year", "month"], sort=True)
    )

    total_jobs = len(tiles) * len(monthly_groups)

    print("=" * 65)
    print("ERA5 CLOUD BASE HEIGHT DOWNLOAD")
    print("=" * 65)
    print("Required timestamps:", required["valid_time"].nunique())
    print("Required dates:", required["valid_time"].dt.date.nunique())
    print("Geographic tiles:", len(tiles))
    print("Monthly groups:", len(monthly_groups))
    print("Maximum download jobs:", total_jobs)
    print("Output directory:", OUTPUT_DIR)
    print("=" * 65)

    client = cdsapi.Client()

    completed = 0
    skipped = 0
    failed = []

    for tile_number, tile in enumerate(tiles, start=1):

        tile_name = (
            f"N{tile['north']:05.1f}_"
            f"W{tile['west']:05.1f}_"
            f"S{tile['south']:05.1f}_"
            f"E{tile['east']:05.1f}"
        )

        for (year, month), group in monthly_groups:

            output_file = OUTPUT_DIR / (
                f"cbh_{year}_{month}_{tile_name}.grib"
            )

            if output_file.exists() and output_file.stat().st_size > 0:
                print("SKIP existing:", output_file.name)
                skipped += 1
                continue

            days = sorted(group["day"].unique().tolist())
            hours = sorted(group["hour"].unique().tolist())

            request = {
                "product_type": ["reanalysis"],
                "variable": ["cloud_base_height"],
                "year": [year],
                "month": [month],
                "day": days,
                "time": hours,
                "area": [
                    tile["north"],
                    tile["west"],
                    tile["south"],
                    tile["east"],
                ],
                "data_format": "grib",
                "download_format": "unarchived",
            }

            print(
                f"\nTile {tile_number}/{len(tiles)} | "
                f"{year}-{month} | "
                f"Area N/W/S/E: "
                f"{tile['north']}/{tile['west']}/"
                f"{tile['south']}/{tile['east']}"
            )

            success = False

            for attempt in range(1, MAX_RETRIES + 1):

                try:
                    client.retrieve(
                        DATASET,
                        request,
                        str(output_file),
                    )

                    if (
                        not output_file.exists()
                        or output_file.stat().st_size == 0
                    ):
                        raise RuntimeError(
                            "Output file is missing or empty."
                        )

                    print(
                        "Downloaded:",
                        output_file.name,
                        f"({output_file.stat().st_size / 1024**2:.2f} MB)"
                    )

                    completed += 1
                    success = True
                    break

                except Exception as exc:
                    print(
                        f"Attempt {attempt}/{MAX_RETRIES} failed:",
                        exc
                    )

                    # Remove an incomplete file before retrying.
                    if output_file.exists():
                        output_file.unlink()

                    if attempt < MAX_RETRIES:
                        time.sleep(5 * attempt)

            if not success:
                failed.append(output_file.name)
                print("FAILED:", output_file.name)

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)
    print("Successful downloads:", completed)
    print("Existing files skipped:", skipped)
    print("Failed downloads:", len(failed))

    if failed:
        with open(
            OUTPUT_DIR / "failed_downloads.txt",
            "w"
        ) as file:
            file.write("\n".join(failed))

        print("Failed job list:", OUTPUT_DIR / "failed_downloads.txt")

    print("Output directory:", OUTPUT_DIR)
    print("=" * 65)


if __name__ == "__main__":
    main()
