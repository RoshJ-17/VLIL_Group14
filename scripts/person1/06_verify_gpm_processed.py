
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path.home() / "aviation_weather_project"
GPM_DIR = ROOT / "data/GPM/processed"
REPORT = ROOT / "gpm_verification_report.csv"

files = sorted(GPM_DIR.glob("GPM_*.nc"))
print(f"Found {len(files)} processed files")

results = []

for i, path in enumerate(files, 1):
    row = {
        "file": path.name,
        "ok": False,
        "time": None,
        "lat_count": 0,
        "lon_count": 0,
        "valid_cells": 0,
        "missing_cells": 0,
        "min_precip": np.nan,
        "max_precip": np.nan,
        "error": "",
    }

    try:
        with xr.open_dataset(path) as ds:
            variable = "GPM_precipitation_rate"

            if variable not in ds:
                raise ValueError(
                    f"Missing variable {variable}; found {list(ds.data_vars)}"
                )

            da = ds[variable]

            if "lat" not in ds.coords or "lon" not in ds.coords:
                raise ValueError("Missing lat/lon coordinates")

            if da.dims != ("lat", "lon"):
                raise ValueError(f"Unexpected dimensions: {da.dims}")

            lat = ds["lat"].values
            lon = ds["lon"].values

            if not (lat.min() >= 6 and lat.max() <= 37):
                raise ValueError(
                    f"Latitude outside expected bounds: {lat.min()} to {lat.max()}"
                )

            if not (lon.min() >= 68 and lon.max() <= 97):
                raise ValueError(
                    f"Longitude outside expected bounds: {lon.min()} to {lon.max()}"
                )

            values = da.values
            valid = np.isfinite(values)

            row.update({
                "ok": True,
                "time": str(ds.attrs.get("time_interval_start_utc", "")),
                "lat_count": len(lat),
                "lon_count": len(lon),
                "valid_cells": int(valid.sum()),
                "missing_cells": int((~valid).sum()),
                "min_precip": float(np.nanmin(values)) if valid.any() else np.nan,
                "max_precip": float(np.nanmax(values)) if valid.any() else np.nan,
            })

    except Exception as exc:
        row["error"] = str(exc)

    results.append(row)

    if i % 50 == 0 or i == len(files):
        print(f"Checked {i}/{len(files)}")

report = pd.DataFrame(results)
report.to_csv(REPORT, index=False)

print("\n--- Verification summary ---")
print("Files found:", len(files))
print("Files passing checks:", int(report["ok"].sum()))
print("Files failing checks:", int((~report["ok"]).sum()))
print("Files with zero valid cells:", int((report["valid_cells"] == 0).sum()))
print("Report saved to:", REPORT)

bad = report.loc[~report["ok"], ["file", "error"]]
if not bad.empty:
    print("\nFiles requiring attention:")
    print(bad.to_string(index=False))

