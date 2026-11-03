import os
import json
import shutil
import numpy as np
import pandas as pd

CSV = "modisPixelGPM_training_clean.csv"
OUT = "tabm_full_data"

TIME_COL = "timestamp"
TARGET_COL = "GPM_precipitation_rate"
TIME_FORMAT = "%Y%j_%H%M"

FEATURES = [
    "lat", "lon", "BT_11", "BT_12", "BTD",
    "Cloud_Mask_Binary", "Cloud_Mask_Raw",
    "CTT", "CTH", "Cloud_Phase"
]

CHUNK_SIZE = 500_000
BATCH_SIZE = 200_000
RAIN_THRESHOLD = 0.1

if os.path.exists(OUT):
    raise SystemExit(
        f"Output directory {OUT} already exists. "
        "Rename or inspect it before running again."
    )

os.makedirs(OUT)

def parse_times(series):
    return pd.to_datetime(
        series.astype("string"),
        format=TIME_FORMAT,
        errors="coerce"
    )

# -------------------------------------------------------
# PASS 1: Count valid rows per timestamp and find splits
# -------------------------------------------------------
print("Pass 1: Counting rows by timestamp...")

timestamp_counts = {}

for chunk in pd.read_csv(
    CSV,
    usecols=[TIME_COL, TARGET_COL],
    chunksize=CHUNK_SIZE
):
    times = chunk[TIME_COL].astype("string")
    targets = pd.to_numeric(chunk[TARGET_COL], errors="coerce")
    valid = targets.notna() & parse_times(times).notna()

    counts = times[valid].value_counts()
    for timestamp, count in counts.items():
        timestamp_counts[timestamp] = (
            timestamp_counts.get(timestamp, 0) + int(count)
        )

timestamps = sorted(timestamp_counts)
total = sum(timestamp_counts.values())

if total == 0:
    raise RuntimeError("No valid rows found.")

train_target = int(total * 0.70)
val_target = int(total * 0.85)

cumulative = 0
train_end = None
val_end = None

for timestamp in timestamps:
    cumulative += timestamp_counts[timestamp]

    if train_end is None and cumulative >= train_target:
        train_end = timestamp

    if val_end is None and cumulative >= val_target:
        val_end = timestamp

if train_end is None or val_end is None or train_end >= val_end:
    raise RuntimeError("Could not determine valid chronological boundaries.")

split_counts = {"train": 0, "val": 0, "test": 0}

for timestamp, count in timestamp_counts.items():
    if timestamp <= train_end:
        split_counts["train"] += count
    elif timestamp <= val_end:
        split_counts["val"] += count
    else:
        split_counts["test"] += count

print("Total valid rows:", f"{total:,}")
print("Train boundary:", train_end)
print("Validation boundary:", val_end)
print("Split sizes:", split_counts)

# -------------------------------------------------------
# Allocate raw memory-mapped arrays
# -------------------------------------------------------
raw = {}
labels = {}
positions = {name: 0 for name in split_counts}

for name, count in split_counts.items():
    raw[name] = np.lib.format.open_memmap(
        f"{OUT}/{name}_raw.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, len(FEATURES))
    )
    labels[name] = np.lib.format.open_memmap(
        f"{OUT}/{name}_y.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(count,)
    )

# -------------------------------------------------------
# PASS 2: Write rows to their chronological split
# -------------------------------------------------------
print("\nPass 2: Writing disk-backed arrays...")

usecols = [TIME_COL, TARGET_COL] + FEATURES

for chunk_number, chunk in enumerate(
    pd.read_csv(CSV, usecols=usecols, chunksize=CHUNK_SIZE),
    start=1
):
    times = chunk[TIME_COL].astype("string")
    targets = pd.to_numeric(chunk[TARGET_COL], errors="coerce")
    parsed = parse_times(times)

    valid = targets.notna() & parsed.notna()
    if not valid.any():
        continue

    chunk = chunk.loc[valid]
    times = times.loc[valid]
    targets = targets.loc[valid]

    X = chunk[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)

    y = (targets.to_numpy() >= RAIN_THRESHOLD).astype(np.uint8)

    train_mask = (times <= train_end).to_numpy(dtype=bool)
    val_mask = (
        (times > train_end) & (times <= val_end)
    ).to_numpy(dtype=bool)
    test_mask = (times > val_end).to_numpy(dtype=bool)

    if chunk_number == 1:
        print("\n--- First chunk split diagnostic ---")
        print("times dtype:", times.dtype)
        print("train_end:", repr(train_end))
        print("first timestamps:", times.head().tolist())
        print("train mask count:", int(train_mask.sum()))
        print("validation mask count:", int(val_mask.sum()))
        print("test mask count:", int(test_mask.sum()))
        print("mask total:", int(
            train_mask.sum() + val_mask.sum() + test_mask.sum()
        ))
        print("valid chunk rows:", len(times))

    for name, mask in [
        ("train", train_mask),
        ("val", val_mask),
        ("test", test_mask)
    ]:
        n = int(mask.sum())
        if n == 0:
            continue

        start = positions[name]
        end = start + n

        raw[name][start:end] = X[mask]
        labels[name][start:end] = y[mask]
        positions[name] = end

    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * CHUNK_SIZE:,} rows")

for name in split_counts:
    raw[name].flush()
    labels[name].flush()
    if positions[name] != split_counts[name]:
        raise RuntimeError(
            f"{name} count mismatch: {positions[name]} "
            f"vs {split_counts[name]}"
        )

# -------------------------------------------------------
# TRAIN-ONLY IMPUTATION AND STANDARDIZATION
# -------------------------------------------------------
print("\nComputing training-only preprocessing statistics...")

medians = np.zeros(len(FEATURES), dtype=np.float64)

for j, feature in enumerate(FEATURES):
    column = raw["train"][:, j]
    medians[j] = np.nanmedian(column)

    if not np.isfinite(medians[j]):
        raise RuntimeError(f"Feature {feature} has no valid training values.")

    print(f"{feature}: median={medians[j]:.6f}")

# Compute means and variances after median imputation.
sums = np.zeros(len(FEATURES), dtype=np.float64)
sums_sq = np.zeros(len(FEATURES), dtype=np.float64)
n_train = split_counts["train"]

for start in range(0, n_train, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_train)
    X = np.asarray(raw["train"][start:end], dtype=np.float32).copy()

    for j in range(len(FEATURES)):
        col = X[:, j]
        col[~np.isfinite(col)] = medians[j]

    sums += X.sum(axis=0, dtype=np.float64)
    sums_sq += np.square(X, dtype=np.float64).sum(axis=0)

means = sums / n_train
variances = np.maximum(sums_sq / n_train - means**2, 0)
stds = np.sqrt(variances)
stds[stds < 1e-8] = 1.0

# -------------------------------------------------------
# Normalize and save final X arrays
# -------------------------------------------------------
print("\nWriting normalized arrays...")

for name, count in split_counts.items():
    X_out = np.lib.format.open_memmap(
        f"{OUT}/{name}_X.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, len(FEATURES))
    )

    for start in range(0, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        X = np.asarray(raw[name][start:end], dtype=np.float32).copy()

        for j in range(len(FEATURES)):
            col = X[:, j]
            col[~np.isfinite(col)] = medians[j]

        X -= means.astype(np.float32)
        X /= stds.astype(np.float32)
        np.clip(X, -10, 10, out=X)

        X_out[start:end] = X

    X_out.flush()
    del X_out

    print(f"{name}: {count:,} rows normalized")

metadata = {
    "features": FEATURES,
    "threshold_mm_per_hr": RAIN_THRESHOLD,
    "time_format": TIME_FORMAT,
    "train_end_timestamp": train_end,
    "validation_end_timestamp": val_end,
    "split_counts": split_counts,
    "total_valid_rows": total,
    "medians": medians.tolist(),
    "means": means.tolist(),
    "stds": stds.tolist(),
    "chronological_split": True
}

with open(f"{OUT}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Release raw arrays and remove their temporary files.
del raw, labels
for name in split_counts:
    os.remove(f"{OUT}/{name}_raw.npy")

print("\nPreparation complete.")
print("Output directory:", OUT)
print("Files:", os.listdir(OUT))
print("Split sizes:", split_counts)
print("Metadata saved.")
