# =============================================================================
# 12_create_mixed_day_precip_split.py
#
# RANDOM / MIXED-DAY SPATIAL SPLIT
#
# Every day is represented in:
#   - Train
#   - Validation
#   - Test
#
# Each spatial pixel (lat, lon) is assigned to ONE split only.
# All temporal observations for that pixel stay in the same split.
#
# timestamp is retained for temporal ordering but is NOT a model feature.
#
# Cloud_Optical_Thickness is completely excluded.
# =============================================================================

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "modis_gpm_precip_positive_modis.csv"

TRAIN_FILE = "precip_positive_modis_mixed_train.csv"
VAL_FILE   = "precip_positive_modis_mixed_val.csv"
TEST_FILE  = "precip_positive_modis_mixed_test.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

TARGET = "GPM_precipitation_rate"

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

OUTPUT_COLUMNS = [
    "timestamp"
] + FEATURES + [
    TARGET
]

# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("CREATE RANDOM / MIXED-DAY SPATIAL PRECIPITATION SPLIT")
print("=" * 90)

print(f"\nInput : {INPUT_FILE}")
print(f"Train : {TRAIN_FILE}")
print(f"Val   : {VAL_FILE}")
print(f"Test  : {TEST_FILE}")

print("\nSplit strategy:")
print("  Spatial pixels are randomly assigned to Train/Val/Test.")
print("  Every timestamp/day therefore appears in all three splits.")
print("  All observations belonging to a pixel remain in the same split.")

print(f"\nRatios:")
print(f"  Train: {TRAIN_RATIO:.0%}")
print(f"  Val  : {VAL_RATIO:.0%}")
print(f"  Test : {TEST_RATIO:.0%}")

print(f"\nRandom seed: {RANDOM_SEED}")

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "=" * 90)
print("LOADING DATASET")
print("=" * 90)

df = pd.read_csv(INPUT_FILE)

print(f"\nRows loaded   : {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")

# =============================================================================
# COLUMN CHECK
# =============================================================================

print("\nChecking columns...")

required_columns = ["timestamp"] + FEATURES + [TARGET]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns found.")

print("\nInput features:")
for i, col in enumerate(FEATURES, 1):
    print(f"{i:2d}. {col}")

print("\nExcluded:")
print(" - timestamp")
print(" - Cloud_Optical_Thickness")
print(" - GPM_randomError")
print(" - GPM_precipitationQualityIndex")

# =============================================================================
# TIMESTAMP
# =============================================================================

print("\n" + "=" * 90)
print("TIMESTAMP PROCESSING")
print("=" * 90)

raw_timestamp = df["timestamp"].astype(str).str.strip()

# IMPORTANT:
# Dataset timestamps are like:
# 2022323_0345
#
# Format:
# %Y%j_%H%M
#
# YYYY = year
# JJJJ = Julian day
# HHMM = time

df["timestamp"] = pd.to_datetime(
    raw_timestamp,
    format="%Y%j_%H%M",
    errors="coerce"
)

invalid_ts = df["timestamp"].isna().sum()

print(f"\nInvalid timestamps: {invalid_ts:,}")

if invalid_ts > 0:
    bad = raw_timestamp[df["timestamp"].isna()].head(10).tolist()

    print("\nExample invalid timestamps:")
    for x in bad:
        print(repr(x))

    raise ValueError(
        "Invalid timestamps found. Check timestamp format."
    )

print(
    f"Time range: "
    f"{df['timestamp'].min()} -> {df['timestamp'].max()}"
)

# =============================================================================
# TARGET CHECK
# =============================================================================

print("\n" + "=" * 90)
print("TARGET CHECK")
print("=" * 90)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

invalid_target = df[TARGET].isna().sum()

print(f"\nInvalid target values: {invalid_target:,}")

if invalid_target > 0:
    raise ValueError(
        "Invalid precipitation values found."
    )

zero_count = (df[TARGET] == 0).sum()
negative_count = (df[TARGET] < 0).sum()

print(f"Zero precipitation    : {zero_count:,}")
print(f"Negative precipitation: {negative_count:,}")
print(f"Positive precipitation: {(df[TARGET] > 0).sum():,}")

if zero_count > 0 or negative_count > 0:
    raise ValueError(
        "Input dataset must contain positive precipitation only."
    )

# =============================================================================
# FEATURE TYPES
# =============================================================================

print("\n" + "=" * 90)
print("FEATURE VALIDATION")
print("=" * 90)

for col in FEATURES:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print("\nMissing values:")

for col in FEATURES:
    n = df[col].isna().sum()

    print(
        f"{col:28s}: {n:8,d}"
        f" ({100*n/len(df):7.3f}%)"
    )

# =============================================================================
# CREATE PIXEL ID
# =============================================================================

print("\n" + "=" * 90)
print("CREATING SPATIAL GROUPS")
print("=" * 90)

# Each unique lat/lon pair is treated as one spatial pixel.
#
# This is critical:
#
# pixel A -> Train
# pixel B -> Validation
# pixel C -> Test
#
# All timestamps belonging to pixel A remain in Train, etc.

df["_pixel_id"] = (
    df["lat"].astype(str)
    + "_"
    + df["lon"].astype(str)
)

unique_pixels = df["_pixel_id"].unique()

n_pixels = len(unique_pixels)

print(f"\nUnique spatial pixels: {n_pixels:,}")

# =============================================================================
# RANDOM PIXEL SPLIT
# =============================================================================

print("\n" + "=" * 90)
print("RANDOM PIXEL ASSIGNMENT")
print("=" * 90)

rng = np.random.default_rng(RANDOM_SEED)

shuffled_pixels = unique_pixels.copy()

rng.shuffle(shuffled_pixels)

n_train = int(
    round(n_pixels * TRAIN_RATIO)
)

n_val = int(
    round(n_pixels * VAL_RATIO)
)

n_test = (
    n_pixels
    - n_train
    - n_val
)

train_pixels = set(
    shuffled_pixels[:n_train]
)

val_pixels = set(
    shuffled_pixels[
        n_train:n_train + n_val
    ]
)

test_pixels = set(
    shuffled_pixels[
        n_train + n_val:
    ]
)

print(f"\nTrain pixels: {len(train_pixels):,}")
print(f"Val pixels  : {len(val_pixels):,}")
print(f"Test pixels : {len(test_pixels):,}")

# =============================================================================
# PIXEL OVERLAP CHECK
# =============================================================================

assert train_pixels.isdisjoint(val_pixels)
assert train_pixels.isdisjoint(test_pixels)
assert val_pixels.isdisjoint(test_pixels)

print("\nPixel overlap check: PASSED")

# =============================================================================
# CREATE SPLITS
# =============================================================================

print("\nCreating datasets...")

train_mask = df["_pixel_id"].isin(train_pixels)
val_mask = df["_pixel_id"].isin(val_pixels)
test_mask = df["_pixel_id"].isin(test_pixels)

train_df = df.loc[train_mask].copy()
val_df = df.loc[val_mask].copy()
test_df = df.loc[test_mask].copy()

# =============================================================================
# SORT TEMPORALLY
# =============================================================================

# Important for TFT sequence generation.

train_df = train_df.sort_values(
    ["_pixel_id", "timestamp"]
).reset_index(drop=True)

val_df = val_df.sort_values(
    ["_pixel_id", "timestamp"]
).reset_index(drop=True)

test_df = test_df.sort_values(
    ["_pixel_id", "timestamp"]
).reset_index(drop=True)

# =============================================================================
# REMOVE INTERNAL PIXEL ID
# =============================================================================

train_df = train_df[OUTPUT_COLUMNS]
val_df = val_df[OUTPUT_COLUMNS]
test_df = test_df[OUTPUT_COLUMNS]

# =============================================================================
# VERIFY ROW CONSERVATION
# =============================================================================

print("\n" + "=" * 90)
print("ROW CONSERVATION")
print("=" * 90)

total_rows = (
    len(train_df)
    + len(val_df)
    + len(test_df)
)

print(f"\nOriginal rows: {len(df):,}")
print(f"Train rows   : {len(train_df):,}")
print(f"Val rows     : {len(val_df):,}")
print(f"Test rows    : {len(test_df):,}")
print(f"Split total  : {total_rows:,}")

assert total_rows == len(df)

print("\nRow conservation: PASSED")

# =============================================================================
# DAY COVERAGE
# =============================================================================

print("\n" + "=" * 90)
print("DAY COVERAGE")
print("=" * 90)

all_days = set(
    df["timestamp"].dt.date.unique()
)

train_days = set(
    train_df["timestamp"].dt.date.unique()
)

val_days = set(
    val_df["timestamp"].dt.date.unique()
)

test_days = set(
    test_df["timestamp"].dt.date.unique()
)

print(f"\nTotal days: {len(all_days)}")

print(
    f"Train days: {len(train_days)}"
)

print(
    f"Validation days: {len(val_days)}"
)

print(
    f"Test days: {len(test_days)}"
)

print(
    "\nDays in ALL three splits:",
    len(
        train_days
        & val_days
        & test_days
    )
)

print("\nDay coverage:")

for day in sorted(all_days):

    tr = (
        train_df["timestamp"].dt.date == day
    ).sum()

    va = (
        val_df["timestamp"].dt.date == day
    ).sum()

    te = (
        test_df["timestamp"].dt.date == day
    ).sum()

    print(
        f"{day} | "
        f"Train: {tr:7,d} | "
        f"Val: {va:7,d} | "
        f"Test: {te:7,d}"
    )

# =============================================================================
# TIMESTAMP COVERAGE
# =============================================================================

print("\n" + "=" * 90)
print("TIMESTAMP COVERAGE")
print("=" * 90)

all_timestamps = set(
    df["timestamp"].unique()
)

train_timestamps = set(
    train_df["timestamp"].unique()
)

val_timestamps = set(
    val_df["timestamp"].unique()
)

test_timestamps = set(
    test_df["timestamp"].unique()
)

print(
    f"\nTotal timestamps : {len(all_timestamps)}"
)

print(
    f"Train timestamps : {len(train_timestamps)}"
)

print(
    f"Val timestamps   : {len(val_timestamps)}"
)

print(
    f"Test timestamps  : {len(test_timestamps)}"
)

print(
    "\nTimestamps represented in ALL splits:",
    len(
        train_timestamps
        & val_timestamps
        & test_timestamps
    )
)

# =============================================================================
# TARGET DISTRIBUTION
# =============================================================================

print("\n" + "=" * 90)
print("TARGET DISTRIBUTION")
print("=" * 90)

for name, split_df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df),
]:

    print(f"\n{name}")

    print(
        f"Rows       : {len(split_df):,}"
    )

    print(
        f"Mean       : {split_df[TARGET].mean():.4f}"
    )

    print(
        f"Median     : {split_df[TARGET].median():.4f}"
    )

    print(
        f"Maximum    : {split_df[TARGET].max():.4f}"
    )

    print(
        f">= 5 mm/h  : "
        f"{(split_df[TARGET] >= 5).sum():,}"
    )

    print(
        f">= 10 mm/h : "
        f"{(split_df[TARGET] >= 10).sum():,}"
    )

    print(
        f">= 20 mm/h : "
        f"{(split_df[TARGET] >= 20).sum():,}"
    )

# =============================================================================
# OUTPUT COLUMN CHECK
# =============================================================================

print("\n" + "=" * 90)
print("OUTPUT COLUMN CHECK")
print("=" * 90)

print("\nColumns:")

for i, col in enumerate(OUTPUT_COLUMNS, 1):
    print(f"{i:2d}. {col}")

assert "timestamp" in OUTPUT_COLUMNS
assert TARGET in OUTPUT_COLUMNS
assert "Cloud_Optical_Thickness" not in OUTPUT_COLUMNS

for feature in FEATURES:
    assert feature in OUTPUT_COLUMNS

print("\nCloud_Optical_Thickness: EXCLUDED")
print("timestamp as model input: NO")

# =============================================================================
# SAVE
# =============================================================================

print("\n" + "=" * 90)
print("SAVING DATASETS")
print("=" * 90)

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

val_df.to_csv(
    VAL_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

print(f"\nSaved:")
print(f"  {TRAIN_FILE}")
print(f"  {VAL_FILE}")
print(f"  {TEST_FILE}")

# =============================================================================
# FINAL VERIFICATION
# =============================================================================

print("\n" + "=" * 90)
print("FINAL VERIFICATION")
print("=" * 90)

# Check that every pixel occurs in only one split.

train_pixels_final = set(
    (
        train_df["lat"].astype(str)
        + "_"
        + train_df["lon"].astype(str)
    ).unique()
)

val_pixels_final = set(
    (
        val_df["lat"].astype(str)
        + "_"
        + val_df["lon"].astype(str)
    ).unique()
)

test_pixels_final = set(
    (
        test_df["lat"].astype(str)
        + "_"
        + test_df["lon"].astype(str)
    ).unique()
)

assert train_pixels_final.isdisjoint(
    val_pixels_final
)

assert train_pixels_final.isdisjoint(
    test_pixels_final
)

assert val_pixels_final.isdisjoint(
    test_pixels_final
)

print("\nSpatial split integrity: PASSED")

print(
    "\nEvery day represented in train:",
    all_days.issubset(train_days)
)

print(
    "Every day represented in validation:",
    all_days.issubset(val_days)
)

print(
    "Every day represented in test:",
    all_days.issubset(test_days)
)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
