#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xarray as xr

MODIS_FILE = "modis_pixel.csv"
ERA5_FILE = "data/ERA5/ERA5_CBH_India_2022-11-19_to_2022-11-24.nc"
OUTPUT_FILE = "modis_era5_cbh.csv"
CHUNKSIZE = 250000


def nearest_hour(ts):
    floor = ts.dt.floor("h")
    add_hour = (ts - floor) >= pd.Timedelta(minutes=30)
    return floor + pd.to_timedelta(add_hour.astype(int), unit="h")


def main():
    ds = xr.open_dataset(ERA5_FILE, engine="netcdf4")

    cbh = ds["cbh"].load().values
    era5_times = pd.to_datetime(ds["valid_time"].values)
    lat_vals = ds["latitude"].values
    lon_vals = ds["longitude"].values

    time_index = pd.Index(era5_times)

    lat0 = float(lat_vals[0])
    lat_step = float(lat_vals[1] - lat_vals[0])

    lon0 = float(lon_vals[0])
    lon_step = float(lon_vals[1] - lon_vals[0])

    print("ERA5 shape:", cbh.shape)
    print("ERA5 time range:", era5_times.min(), "to", era5_times.max())
    print("ERA5 latitude range:", float(lat_vals.min()), "to", float(lat_vals.max()))
    print("ERA5 longitude range:", float(lon_vals.min()), "to", float(lon_vals.max()))

    first_chunk = True
    total_rows = 0
    total_missing_cbh = 0

    for i, chunk in enumerate(pd.read_csv(MODIS_FILE, chunksize=CHUNKSIZE), start=1):
        modis_time = pd.to_datetime(chunk["timestamp"], format="%Y%j_%H%M")
        era5_time = nearest_hour(modis_time)

        time_idx = time_index.get_indexer(pd.DatetimeIndex(era5_time))

        lat_idx = np.rint((chunk["lat"].to_numpy() - lat0) / lat_step).astype(int)
        lon_idx = np.rint((chunk["lon"].to_numpy() - lon0) / lon_step).astype(int)

        lat_idx = np.clip(lat_idx, 0, len(lat_vals) - 1)
        lon_idx = np.clip(lon_idx, 0, len(lon_vals) - 1)

        era5_cbh = np.full(len(chunk), np.nan, dtype=np.float32)
        era5_lat = np.full(len(chunk), np.nan, dtype=np.float32)
        era5_lon = np.full(len(chunk), np.nan, dtype=np.float32)

        valid = time_idx >= 0

        era5_cbh[valid] = cbh[time_idx[valid], lat_idx[valid], lon_idx[valid]]
        era5_lat[valid] = lat_vals[lat_idx[valid]]
        era5_lon[valid] = lon_vals[lon_idx[valid]]

        chunk["era5_valid_time"] = era5_time.dt.strftime("%Y-%m-%d %H:%M:%S")
        chunk["era5_lat"] = era5_lat
        chunk["era5_lon"] = era5_lon
        chunk["ERA5_CBH"] = era5_cbh

        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False
        total_rows += len(chunk)
        total_missing_cbh += int(np.isnan(era5_cbh).sum())

        print(
            f"[{i}] rows={len(chunk):,} total_rows={total_rows:,} "
            f"missing_ERA5_CBH={total_missing_cbh:,}"
        )

    print("\n========================================")
    print("ERA5 matching finished")
    print("Output:", OUTPUT_FILE)
    print("Total rows:", total_rows)
    print("Missing ERA5_CBH:", total_missing_cbh)
    print("========================================")


if __name__ == "__main__":
    main()
