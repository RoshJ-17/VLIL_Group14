
import pandas as pd
import re
from collections import Counter

FILE = "finalIntegratedDataset.csv"
CHUNK_SIZE = 200_000

monthly_counts = Counter()
invalid_count = 0
total_count = 0
invalid_samples = []

def parse_timestamps(series):
    """Parse several possible timestamp formats."""
    values = series.astype("string").str.strip()
    result = pd.Series(
        pd.NaT, index=values.index, dtype="datetime64[ns]"
    )

    formats = [
        (r"^\d{4}\d{3}_\d{4}$", "%Y%j_%H%M"),
        (r"^\d{4}\d{3}_\d{6}$", "%Y%j_%H%M%S"),
        (r"^\d{8}_\d{4}$", "%Y%m%d_%H%M"),
        (r"^\d{8}\d{4}$", "%Y%m%d%H%M"),
        (r"^\d{8}$", "%Y%m%d"),
        (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
         "%Y-%m-%d %H:%M:%S"),
        (r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
         "%Y-%m-%dT%H:%M:%S"),
    ]

    for pattern, fmt in formats:
        mask = values.str.match(pattern, na=False) & result.isna()

        if mask.any():
            result.loc[mask] = pd.to_datetime(
                values.loc[mask],
                format=fmt,
                errors="coerce"
            )

    # Try general parsing for any remaining formats
    remaining = result.isna() & values.notna() & values.ne("")
    if remaining.any():
        result.loc[remaining] = pd.to_datetime(
            values.loc[remaining],
            format="mixed",
            errors="coerce"
        )

    return result


print("Scanning CSV in chunks...")

for chunk_no, chunk in enumerate(
    pd.read_csv(
        FILE,
        usecols=["timestamp"],
        dtype={"timestamp": "string"},
        chunksize=CHUNK_SIZE
    ),
    start=1
):
    raw = chunk["timestamp"]
    dates = parse_timestamps(raw)

    total_count += len(raw)
    invalid_mask = dates.isna()

    invalid_count += int(invalid_mask.sum())

    # Keep a few examples of unrecognised values
    if len(invalid_samples) < 20:
        invalid_samples.extend(
            raw[invalid_mask].dropna().head(
                20 - len(invalid_samples)
            ).tolist()
        )

    valid_dates = dates.dropna()

    counts = valid_dates.dt.to_period("M").value_counts()

    for month, count in counts.items():
        monthly_counts[str(month)] += int(count)

    if chunk_no % 5 == 0:
        print(
            f"Chunks: {chunk_no} | "
            f"Rows scanned: {total_count:,} | "
            f"Invalid: {invalid_count:,}"
        )


print("\n========== MONTHLY RECORD COUNTS ==========")

for month in sorted(monthly_counts):
    print(f"{month}: {monthly_counts[month]:,}")

print("\n========== SUMMARY ==========")
print("Total rows:", f"{total_count:,}")
print("Valid timestamps:", f"{total_count - invalid_count:,}")
print("Invalid timestamps:", f"{invalid_count:,}")

print("\nUnrecognised timestamp examples:")
for value in invalid_samples:
    print(repr(value))

