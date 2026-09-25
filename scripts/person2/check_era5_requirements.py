import pandas as pd

MODIS_FILE = "modisPixel.csv"
CHUNKSIZE = 250_000

required_times = set()

for chunk in pd.read_csv(
    MODIS_FILE,
    usecols=["timestamp"],
    chunksize=CHUNKSIZE
):
    modis_time = pd.to_datetime(
        chunk["timestamp"],
        format="%Y%j_%H%M",
        errors="coerce"
    )

    # Match the nearest ERA5 hour, as in the existing script
    era5_time = modis_time.dt.round("h")

    required_times.update(
        era5_time.dropna().unique()
    )

times = pd.DatetimeIndex(sorted(required_times))

requirements = pd.DataFrame({
    "valid_time": times,
    "date": times.strftime("%Y-%m-%d"),
    "hour": times.strftime("%H:%M")
})

requirements.to_csv(
    "era5_required_times.csv",
    index=False
)

print("Unique ERA5 timestamps required:", len(times))
print("First required time:", times.min())
print("Last required time:", times.max())
print("Distinct ERA5 dates:", requirements["date"].nunique())

print("\nRequired hours by date:")
for date, group in requirements.groupby("date"):
    print(date, ", ".join(group["hour"].tolist()))

print("\nSaved: era5_required_times.csv")
