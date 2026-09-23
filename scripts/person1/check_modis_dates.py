import pandas as pd

MODIS_FILE = "modisPixel.csv"
CHUNKSIZE = 250_000

date_counts = {}
timestamps = set()
min_lat = float("inf")
max_lat = float("-inf")
min_lon = float("inf")
max_lon = float("-inf")

for chunk in pd.read_csv(
    MODIS_FILE,
    usecols=["timestamp", "lat", "lon"],
    chunksize=CHUNKSIZE
):
    ts = pd.to_datetime(
        chunk["timestamp"],
        format="%Y%j_%H%M",
        errors="coerce"
    )

    if ts.isna().any():
        print("WARNING: Invalid timestamps:", int(ts.isna().sum()))

    valid = ts.notna()
    dates = ts[valid].dt.strftime("%Y-%m-%d")

    for date, count in dates.value_counts().items():
        date_counts[date] = date_counts.get(date, 0) + int(count)

    timestamps.update(ts[valid].unique())

    min_lat = min(min_lat, chunk["lat"].min())
    max_lat = max(max_lat, chunk["lat"].max())
    min_lon = min(min_lon, chunk["lon"].min())
    max_lon = max(max_lon, chunk["lon"].max())

print("\n========== MODIS DATE SUMMARY ==========")
print("Total rows:", sum(date_counts.values()))
print("Unique acquisition timestamps:", len(timestamps))
print("Unique calendar dates:", len(date_counts))
print("First timestamp:", min(timestamps) if timestamps else "None")
print("Last timestamp:", max(timestamps) if timestamps else "None")

print("\nGeographic bounds:")
print("Latitude:", min_lat, "to", max_lat)
print("Longitude:", min_lon, "to", max_lon)

print("\nRows per date:")
for date in sorted(date_counts):
    print(f"{date}: {date_counts[date]:,} rows")

inventory = pd.DataFrame(
    [
        {"date": date, "modis_rows": date_counts[date]}
        for date in sorted(date_counts)
    ]
)

inventory.to_csv("modis_date_inventory.csv", index=False)
print("\nSaved date inventory: modis_date_inventory.csv")
