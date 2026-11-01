
import pandas as pd
import numpy as np
from collections import Counter

FILE = "finalIntegratedDataset.csv"

CHUNK_SIZE = 250_000
SAMPLE_PER_CHUNK = 1_000
SEED = 42

original_counts = Counter()
sampled_counts = Counter()

original_rows = 0
valid_target_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        FILE,
        usecols=["timestamp", "CBH_km"],
        chunksize=CHUNK_SIZE
    ),
    start=1
):
    original_rows += len(chunk)

    chunk["CBH_km"] = pd.to_numeric(
        chunk["CBH_km"], errors="coerce"
    )

    chunk = chunk.dropna(subset=["CBH_km"]).copy()

    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"],
        format="%Y%j_%H%M",
        errors="coerce"
    )

    chunk = chunk.dropna(subset=["timestamp"])
    chunk = chunk[chunk["CBH_km"] >= 0]

    valid_target_rows += len(chunk)

    for timestamp, count in chunk["timestamp"].value_counts().items():
        original_counts[timestamp] += count

    if len(chunk) > SAMPLE_PER_CHUNK:
        chunk = chunk.sample(
            n=SAMPLE_PER_CHUNK,
            random_state=SEED + chunk_number
        )

    for timestamp, count in chunk["timestamp"].value_counts().items():
        sampled_counts[timestamp] += count

print("Original CSV rows:", original_rows)
print("Valid target rows:", valid_target_rows)
print("Original unique timestamps:", len(original_counts))
print("Sampled unique timestamps:", len(sampled_counts))

original_ts = set(original_counts)
sampled_ts = set(sampled_counts)

print("Timestamps missing after sampling:", len(original_ts - sampled_ts))
print("Timestamps retained:", len(sampled_ts))

original = pd.Series(original_counts, name="original_rows")
sampled = pd.Series(sampled_counts, name="sampled_rows")

comparison = pd.concat(
    [original, sampled],
    axis=1
).fillna(0)

comparison["retention_rate"] = (
    comparison["sampled_rows"] /
    comparison["original_rows"]
)

comparison.index.name = "timestamp"

comparison.to_csv("cbh_sampling_timestamp_audit.csv")

print("\nOriginal monthly distribution:")
print(original.groupby(original.index.to_period("M")).sum())

print("\nSampled monthly distribution:")
print(sampled.groupby(sampled.index.to_period("M")).sum())

print("\nRetention rate:")
print(comparison["retention_rate"].describe())

print("\nLowest retained timestamps:")
print(comparison.sort_values("retention_rate").head(10))

