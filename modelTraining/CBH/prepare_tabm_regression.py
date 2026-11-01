#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------
# Configuration
# ----------------------------
CSV_PATH = Path("modisPixelGPM_training_clean.csv")
OUT_DIR = Path("tabm_regression_data")

TARGET_COL = "GPM_precipitation_rate"
TIME_COL = "timestamp"

FEATURES = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
    "Cloud_Mask_Binary",
    "Cloud_Mask_Raw",
    "CTT",
    "CTH",
    "Cloud_Phase",
]

CHUNK_ROWS = 500_000
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42


def parse_timestamp(series):
    """Parse timestamps formatted as YYYYJJJ_HHMM."""
    return pd.to_datetime(
        series.astype(str),
        format="%Y%j_%H%M",
        errors="coerce",
    )


def pass1_find_split_boundaries():
    """
    Count valid positive-rain rows per timestamp and find chronological
    split boundaries. Rows sharing a timestamp remain in the same split.
    """
    print("Pass 1: counting positive-rain rows by timestamp...", flush=True)

    counts = {}

    for chunk in pd.read_csv(
        CSV_PATH,
        usecols=[TIME_COL, TARGET_COL],
        chunksize=CHUNK_ROWS,
    ):
        target = pd.to_numeric(chunk[TARGET_COL], errors="coerce")
        timestamp = parse_timestamp(chunk[TIME_COL])

        valid = (
            target.notna()
            & np.isfinite(target)
            & (target > 0)
            & timestamp.notna()
        )

        if not valid.any():
            continue

        times = timestamp.loc[valid].dt.strftime("%Y%j_%H%M")
        grouped = times.value_counts()

        for time_key, count in grouped.items():
            counts[time_key] = counts.get(time_key, 0) + int(count)

    if not counts:
        raise RuntimeError("No valid positive-rain rows found.")

    ordered = sorted(counts.items(), key=lambda item: item[0])
    total = sum(count for _, count in ordered)

    train_target = int(total * TRAIN_FRAC)
    train_end = 0
    train_boundary = None

    for time_key, count in ordered:
        train_end += count
        if train_end >= train_target:
            train_boundary = time_key
            break

    val_target = int(total * (TRAIN_FRAC + VAL_FRAC))
    cumulative = 0
    val_boundary = None

    for time_key, count in ordered:
        cumulative += count
        if cumulative >= val_target:
            val_boundary = time_key
            break

    if train_boundary is None or val_boundary is None:
        raise RuntimeError("Could not determine split boundaries.")

    print(f"Positive-rain rows: {total:,}", flush=True)
    print(f"Train boundary: {train_boundary}", flush=True)
    print(f"Validation boundary: {val_boundary}", flush=True)

    return total, train_boundary, val_boundary


def allocate_memmap(path, shape, dtype):
    """Create a .npy file backed by a memory map."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def pass2_write_positive_rows(total, train_boundary, val_boundary):
    """
    Read the CSV again, keep only valid positive-rain rows, assign splits
    by timestamp, and write feature/target arrays.
    """
    print("Pass 2: writing positive-rain feature and target arrays...", flush=True)

    # First count rows per split so the output arrays can be allocated exactly.
    split_counts = {"train": 0, "validation": 0, "test": 0}

    for chunk in pd.read_csv(
        CSV_PATH,
        usecols=[TIME_COL, TARGET_COL],
        chunksize=CHUNK_ROWS,
    ):
        target = pd.to_numeric(chunk[TARGET_COL], errors="coerce")
        timestamp = parse_timestamp(chunk[TIME_COL])

        valid = (
            target.notna()
            & np.isfinite(target)
            & (target > 0)
            & timestamp.notna()
        )

        if not valid.any():
            continue

        time_keys = timestamp.loc[valid].dt.strftime("%Y%j_%H%M")
        train_mask = time_keys <= train_boundary
        val_mask = (time_keys > train_boundary) & (time_keys <= val_boundary)
        test_mask = time_keys > val_boundary

        split_counts["train"] += int(train_mask.sum())
        split_counts["validation"] += int(val_mask.sum())
        split_counts["test"] += int(test_mask.sum())

    if sum(split_counts.values()) != total:
        raise RuntimeError(
            f"Split count mismatch: {split_counts}, total={total}"
        )

    print(f"Split counts: {split_counts}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for split_name, n_rows in split_counts.items():
        arrays[f"{split_name}_X"] = allocate_memmap(
            OUT_DIR / f"{split_name}_X.npy",
            (n_rows, len(FEATURES)),
            np.float32,
        )
        arrays[f"{split_name}_y"] = allocate_memmap(
            OUT_DIR / f"{split_name}_y.npy",
            (n_rows,),
            np.float32,
        )

    offsets = {"train": 0, "validation": 0, "test": 0}

    for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_ROWS):
        # Keep only columns needed for features, target, and chronology.
        required = FEATURES + [TIME_COL, TARGET_COL]
        chunk = chunk[required].copy()

        target = pd.to_numeric(chunk[TARGET_COL], errors="coerce")
        timestamp = parse_timestamp(chunk[TIME_COL])

        valid = (
            target.notna()
            & np.isfinite(target)
            & (target > 0)
            & timestamp.notna()
        )

        if not valid.any():
            continue

        valid_chunk = chunk.loc[valid].copy()
        valid_target = target.loc[valid].to_numpy(dtype=np.float32)
        time_keys = timestamp.loc[valid].dt.strftime("%Y%j_%H%M")

        split_masks = {
            "train": (time_keys <= train_boundary).to_numpy(),
            "validation": (
                (time_keys > train_boundary)
                & (time_keys <= val_boundary)
            ).to_numpy(),
            "test": (time_keys > val_boundary).to_numpy(),
        }

        feature_frame = valid_chunk[FEATURES].apply(
            pd.to_numeric,
            errors="coerce",
        )
        feature_values = feature_frame.to_numpy(dtype=np.float32)

        for split_name, mask in split_masks.items():
            n = int(mask.sum())
            if n == 0:
                continue

            start = offsets[split_name]
            end = start + n

            arrays[f"{split_name}_X"][start:end] = feature_values[mask]
            arrays[f"{split_name}_y"][start:end] = valid_target[mask]

            offsets[split_name] = end

    for array in arrays.values():
        array.flush()

    # Fit feature imputation and scaling using training rows only.
    print("Fitting training-only imputation and scaling...", flush=True)

    train_X = arrays["train_X"]
    n_features = train_X.shape[1]

    medians = np.zeros(n_features, dtype=np.float64)

    # Chunked median approximation would be complex; this exact approach
    # reads one feature column at a time from the memory-mapped training set.
    for j in range(n_features):
        col = np.asarray(train_X[:, j])
        finite = col[np.isfinite(col)]
        medians[j] = np.median(finite) if finite.size else 0.0

    # Replace non-finite feature values with training medians, then compute
    # training means and standard deviations.
    train_sum = np.zeros(n_features, dtype=np.float64)
    train_sumsq = np.zeros(n_features, dtype=np.float64)
    train_count = np.zeros(n_features, dtype=np.int64)

    for start in range(0, len(train_X), CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, len(train_X))
        block = np.asarray(train_X[start:end], dtype=np.float32).copy()

        for j in range(n_features):
            bad = ~np.isfinite(block[:, j])
            block[bad, j] = medians[j]

        train_sum += block.sum(axis=0, dtype=np.float64)
        train_sumsq += np.square(block, dtype=np.float64).sum(axis=0)
        train_count += len(block)

    means = train_sum / train_count
    variances = np.maximum(train_sumsq / train_count - means**2, 0.0)
    stds = np.sqrt(variances)
    stds[stds < 1e-8] = 1.0

    # Normalize all splits in chunks; save standardized features in place.
    for split_name in ("train", "validation", "test"):
        X = arrays[f"{split_name}_X"]

        for start in range(0, len(X), CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, len(X))
            block = np.asarray(X[start:end], dtype=np.float32).copy()

            for j in range(n_features):
                bad = ~np.isfinite(block[:, j])
                block[bad, j] = medians[j]

            block = (block - means.astype(np.float32)) / stds.astype(np.float32)
            np.clip(block, -10.0, 10.0, out=block)
            X[start:end] = block

        X.flush()

    metadata = {
        "task": "positive_rain_rate_regression",
        "target_col": TARGET_COL,
        "target_transform": "log1p applied in training",
        "positive_rain_filter": f"{TARGET_COL} > 0",
        "time_col": TIME_COL,
        "features": FEATURES,
        "split_counts": split_counts,
        "train_boundary": train_boundary,
        "validation_boundary": val_boundary,
        "medians": medians.tolist(),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "feature_clip": [-10.0, 10.0],
        "seed": RANDOM_SEED,
    }

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Prepared arrays saved in: {OUT_DIR}", flush=True)
    print(f"Metadata: {OUT_DIR / 'metadata.json'}", flush=True)


def main():
    total, train_boundary, val_boundary = pass1_find_split_boundaries()
    pass2_write_positive_rows(total, train_boundary, val_boundary)


if __name__ == "__main__":
    main()

