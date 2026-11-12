import pandas as pd
from collections import Counter

path = "data/processed/full_year_2022_2023/decision_support_chunked/decision_grid_025deg.csv"

status_counts = Counter()
severity_counts = Counter()

missing_status = 0
missing_severity = 0
missing_both = 0
total_groups = 0

small_groups = 0
small_group_status = Counter()
small_group_pixels = 0

assessable_groups = 0
large_groups = 0

sample = []

for chunk in pd.read_csv(path, chunksize=100000):

    total_groups += len(chunk)

    status = chunk["grid_data_status"]
    severity = chunk["maximum_pixel_advisory_level"]

    missing_status += int(status.isna().sum())
    missing_severity += int(severity.isna().sum())
    missing_both += int((status.isna() & severity.isna()).sum())

    status_counts.update(status.value_counts(dropna=False).to_dict())
    severity_counts.update(severity.value_counts(dropna=False).to_dict())

    total = chunk["total_input_pixels"]
    assessable = chunk["assessable_pixels"]

    small = total < 10
    small_groups += int(small.sum())
    small_group_pixels += int(total[small].sum())

    small_group_status.update(
        status[small].fillna("MISSING").value_counts().to_dict()
    )

    assessable_groups += int((status == "Assessable").sum())
    large_groups += int((total >= 10).sum())

    if len(sample) < 10:
        sample.extend(
            chunk.loc[
                small | status.isna() | severity.isna(),
                [
                    "decision_timestamp",
                    "grid_lat_center",
                    "grid_lon_center",
                    "total_input_pixels",
                    "assessable_pixels",
                    "grid_data_status",
                    "maximum_pixel_advisory_level"
                ]
            ].head(10 - len(sample)).to_dict("records")
        )

print("\n========== GRID STATUS DIAGNOSTIC ==========")
print("Total groups:", total_groups)
print("Missing grid_data_status:", missing_status)
print("Missing maximum severity:", missing_severity)
print("Missing both:", missing_both)

print("\nStatus distribution:")
for k, v in status_counts.items():
    print(repr(k), f"{v:,}")

print("\nMaximum severity distribution:")
for k, v in severity_counts.items():
    print(repr(k), f"{v:,}")

print("\nGroups with fewer than 10 pixels:", small_groups)
print("Pixels in those groups:", small_group_pixels)
print("Status of groups with fewer than 10 pixels:")
for k, v in small_group_status.items():
    print(repr(k), f"{v:,}")

print("\nGroups labelled Assessable:", assessable_groups)
print("Groups with at least 10 pixels:", large_groups)

print("\nSample suspicious rows:")
for row in sample:
    print(row)
