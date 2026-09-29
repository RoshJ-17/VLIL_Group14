
from pathlib import Path
from datetime import datetime, timedelta
import re
import subprocess
import time

import h5py
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path.home() / "aviation_weather_project"
TIMES_FILE = ROOT / "gpm_required_times.csv"
RAW_DIR = ROOT / "data/GPM/raw"
OUT_DIR = ROOT / "data/GPM/processed"

LAT_MIN, LAT_MAX = 6.0, 37.0
LON_MIN, LON_MAX = 68.0, 97.0

BASE_URL = (
    "https://gpm1.gesdisc.eosdis.nasa.gov/data/"
    "GPM_L3/GPM_3IMERGHH.07"
)

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def granule_name(t):
    """Build the V07B filename for a half-hour interval start."""
    t = pd.Timestamp(t)
    end = t + pd.Timedelta(minutes=29, seconds=59)
    minute_index = t.hour * 60 + t.minute

    return (
        f"3B-HHR.MS.MRG.3IMERG."
        f"{t:%Y%m%d}-S{t:%H%M%S}"
        f"-E{end:%H%M%S}.{minute_index:04d}.V07B.HDF5"
    )

def download(url, destination):
    """Download safely; keep partial files separate."""
    partial = destination.with_suffix(destination.suffix + ".part")

    result = subprocess.run([
        "curl", "-fL", "-n",
        "-b", str(Path.home() / ".urs_cookies"),
        "-c", str(Path.home() / ".urs_cookies"),
        "--retry", "4",
        "--retry-delay", "3",
        "-o", str(partial),
        url,
    ])

    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {url}")

    if not h5py.is_hdf5(partial):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not valid HDF5: {url}")

    partial.replace(destination)

def extract_region(raw_file, output_file, t):
    """Extract precipitation only for the India bounding box."""
    with h5py.File(raw_file, "r") as f:
        lat = f["Grid/lat"][:]
        lon = f["Grid/lon"][:]

        lat_idx = np.flatnonzero(
            (lat >= LAT_MIN) & (lat <= LAT_MAX)
        )
        lon_idx = np.flatnonzero(
            (lon >= LON_MIN) & (lon <= LON_MAX)
        )

        # Array dimensions are time, longitude, latitude.
        ds = f["Grid/precipitation"]
        values = ds[0, lon_idx[0]:lon_idx[-1] + 1,
                    lat_idx[0]:lat_idx[-1] + 1].T.copy()

        # Convert the product's fill value to NaN.
        fill = ds.attrs.get("_FillValue", -9999.9)
        values = values.astype("float32")
        values[np.isclose(values, fill)] = np.nan
        values[values < 0] = np.nan

        region_lat = lat[lat_idx]
        region_lon = lon[lon_idx]

    data = xr.Dataset(
        {
            "GPM_precipitation_rate": (
                ("lat", "lon"), values,
                {"units": "mm/hr",
                 "long_name": "IMERG precipitation rate"}
            )
        },
        coords={
            "lat": region_lat,
            "lon": region_lon,
            "time": pd.Timestamp(t).to_datetime64(),
        },
        attrs={
            "source": "NASA GPM IMERG V07B",
            "region": "India bounding box",
            "time_interval_start_utc": str(pd.Timestamp(t)),
        }
    )

    temp = output_file.with_suffix(".tmp.nc")
    data.to_netcdf(temp)
    temp.replace(output_file)
    data.close()

def main():
    required = pd.read_csv(TIMES_FILE)
    time_col = "gpm_start" if "gpm_start" in required.columns else None

    if time_col is None:
        raise ValueError(
            f"Expected a 'gpm_start' column in {TIMES_FILE}"
        )

    times = pd.to_datetime(required[time_col]).drop_duplicates().sort_values()
    failures = []

    print(f"Required intervals: {len(times)}")
    print(f"Raw directory: {RAW_DIR}")
    print(f"Processed directory: {OUT_DIR}")

    for i, t in enumerate(times, start=1):
        t = pd.Timestamp(t)
        filename = granule_name(t)
        raw_file = RAW_DIR / filename
        output_file = OUT_DIR / f"GPM_{t:%Y%m%d_%H%M}.nc"

        if output_file.exists():
            print(f"[{i}/{len(times)}] Already processed: {output_file.name}")
            continue

        # NASA archive uses year/day-of-year folders.
        day_of_year = t.strftime("%j")
        url = f"{BASE_URL}/{t:%Y}/{day_of_year}/{filename}"

        try:
            if not raw_file.exists():
                print(f"[{i}/{len(times)}] Downloading {filename}")
                download(url, raw_file)

            print(f"    Extracting India precipitation")
            extract_region(raw_file, output_file, t)

        except Exception as exc:
            failures.append((str(t), str(exc)))
            print(f"    FAILED: {exc}")

        # Be polite to the data server.
        time.sleep(0.2)

    print("\nFinished.")
    print(f"Processed files: {len(list(OUT_DIR.glob('GPM_*.nc')))}")
    print(f"Failures: {len(failures)}")

    if failures:
        pd.DataFrame(
            failures, columns=["gpm_start", "error"]
        ).to_csv(ROOT / "gpm_download_failures.csv", index=False)
        print("Failures saved to gpm_download_failures.csv")
        print("Fix any failures and rerun this script to resume.")

if __name__ == "__main__":
    main()

