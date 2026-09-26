
import pandas as pd
from pathlib import Path

MODIS_FILE = "modisPixel.csv"
OUTPUT_FILE = "gpm_required_times.csv"

# Read only the timestamp column.
timestamps = pd.read_csv(
    MODIS_FILE,
    usecols=["timestamp"]
)["timestamp"].dropna().astype(str).unique()

modis_times = pd.to_datetime(
    timestamps,
    format="%Y%j_%H%M",
    errors="raise"
)

# Match the logic in 04_match_gpm_precip.py:
# before :30 -> interval starts at :00
# at/after :30 -> interval starts at :30
interval_minutes = (modis_times.minute >= 30) * 30

gpm_times = (
    modis_times.normalize()
    + pd.to_timedelta(modis_times.hour, unit="h")
    + pd.to_timedelta(interval_minutes, unit="m")
)

required = pd.DataFrame({
    "gpm_start": sorted(pd.DatetimeIndex(gpm_times).unique())
})

required["date"] = required["gpm_start"].dt.strftime("%Y%m%d")
required["time"] = required["gpm_start"].dt.strftime("%H%M")

required.to_csv(OUTPUT_FILE, index=False)

print("Unique MODIS timestamps:", len(modis_times))
print("Required GPM intervals:", len(required))
print("First interval:", required["gpm_start"].min())
print("Last interval:", required["gpm_start"].max())
print("Saved:", OUTPUT_FILE)
