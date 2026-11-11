#!/usr/bin/env python3

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/home/m_prathilothamai/aviation_weather_project")
CSV = ROOT / "modisPixelGPM_training_clean.csv"
GPM_DIR = ROOT / "data" / "GPM"

TARGET = "GPM_precipitation_rate"
TIME = "timestamp"

print("\n========== GPM TARGET AUDIT ==========\n")

# 1. Locate source files
files = sorted(
    glob.glob(str(GPM_DIR / "*.nc4"))
    + glob.glob(str(GPM_DIR / "*.nc"))
    + glob.glob(str(GPM_DIR / "*.h5"))
    + glob.glob(str(GPM_DIR / "*.HDF5"))
)

print("GPM source directory:", GPM_DIR)
print("GPM files found:", len(files))

if not files:
    print("WARNING: No GPM NetCDF/HDF5 files found.")
else:
    # 2. Inspect first source file
    sample = files[0]
    print("\nSample source file:", sample)

    ds = None

    for group in [None, "Grid"]:
        try:
            if group is None:
                ds = xr.open_dataset(sample)
            else:
                ds = xr.open_dataset(sample, group=group)
            break
        except Exception:
            continue

    if ds is None:
        print("Could not open source file with xarray.")
    else:
        print("\nVariables:", list(ds.data_vars))
        print("Coordinates:", list(ds.coords))

        candidates = [
            "precipitation",
            "precipitationCal",
            "precipitationUncal",
        ]

        for name in candidates:
            if name in ds:
                var = ds[name]
                print("\nSelected variable:", name)
                print("Attributes:", var.attrs)
                print("Dimensions:", var.dims)

                values = np.asarray(var.values, dtype=np.float64)
                values = values[np.isfinite(values)]

                fill = var.attrs.get(
                    "_FillValue",
                    var.encoding.get("_FillValue")
                )

                if fill is not None:
                    values = values[values != float(fill)]

                values = values[values >= 0]

                if values.size:
                    print("Valid value count:", values.size)
                    print("Minimum:", float(np.min(values)))
                    print("Maximum:", float(np.max(values)))
                    print("Mean:", float(np.mean(values)))
                    print("Median:", float(np.median(values)))
                    print("95th percentile:", float(np.percentile(values, 95)))
                else:
                    print("No valid non-negative values found.")
                break
        else:
            print("WARNING: No recognised precipitation variable found.")

        print("\nGlobal metadata:")
        for key in [
            "FileHeader",
            "AlgorithmVersion",
            "ProductVersion",
            "StartGranuleDateTime",
            "StopGranuleDateTime",
            "TimeInterval",
        ]:
            if key in ds.attrs:
                print(key, ":", ds.attrs[key])

        ds.close()

# 3. Inspect cleaned training data
print("\n========== TRAINING CSV AUDIT ==========\n")

if not CSV.exists():
    print("Training CSV not found:", CSV)
    raise SystemExit(1)

print("Training CSV:", CSV)

header = pd.read_csv(CSV, nrows=0).columns.tolist()
print("Columns:", header)

required = [TARGET, TIME]

missing = [c for c in required if c not in header]

if missing:
    print("Missing required columns:", missing)
    raise SystemExit(1)

count = 0
valid = 0
positive = 0
zero = 0
negative = 0
invalid = 0
minimum = np.inf
maximum = -np.inf
sum_values = 0.0
sum_sq = 0.0
sample_values = []

timestamp_min = None
timestamp_max = None

for chunk in pd.read_csv(
    CSV,
    usecols=required,
    chunksize=500_000
):
    count += len(chunk)

    y = pd.to_numeric(chunk[TARGET], errors="coerce")
    finite = y.notna() & np.isfinite(y)

    valid += int(finite.sum())
    positive += int((finite & (y > 0)).sum())
    zero += int((finite & (y == 0)).sum())
    negative += int((finite & (y < 0)).sum())
    invalid += int((~finite).sum())

    vals = y[finite].to_numpy(dtype=np.float64)

    if len(vals):
        minimum = min(minimum, float(vals.min()))
        maximum = max(maximum, float(vals.max()))
        sum_values += float(vals.sum())
        sum_sq += float(np.square(vals).sum())

        if len(sample_values) < 200_000:
            sample_values.extend(
                vals[:200_000 - len(sample_values)].tolist()
            )

    ts = chunk[TIME].dropna().astype(str)

    if len(ts):
        lo, hi = ts.min(), ts.max()
        timestamp_min = lo if timestamp_min is None else min(timestamp_min, lo)
        timestamp_max = hi if timestamp_max is None else max(timestamp_max, hi)

print("\nRows:", count)
print("Valid target rows:", valid)
print("Positive target rows:", positive)
print("Zero target rows:", zero)
print("Negative target rows:", negative)
print("Invalid target rows:", invalid)

if valid:
    mean = sum_values / valid
    variance = max(sum_sq / valid - mean ** 2, 0)

    print("Minimum rate:", minimum, "mm/hr")
    print("Maximum rate:", maximum, "mm/hr")
    print("Mean rate:", mean, "mm/hr")
    print("Standard deviation:", np.sqrt(variance), "mm/hr")

    sample = np.asarray(sample_values)
    if len(sample):
        print("Sample median:", np.median(sample))
        print("Sample 90th percentile:", np.percentile(sample, 90))
        print("Sample 95th percentile:", np.percentile(sample, 95))
        print("Sample 99th percentile:", np.percentile(sample, 99))

print("Earliest timestamp:", timestamp_min)
print("Latest timestamp:", timestamp_max)

print("\n========== AUDIT COMPLETE ==========")
print("No source or training files were modified.")
