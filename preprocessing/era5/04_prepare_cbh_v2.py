#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "modis_era5_cbh.csv"

TRAIN_FILE = "cbh_train_v2.csv"
VAL_FILE = "cbh_val_v2.csv"
TEST_FILE = "cbh_test_v2.csv"

SCALER_FILE = "cbh_scaler_v2.pkl"


# ============================================================
# FEATURES
# ============================================================

CONTINUOUS_FEATURES = [
    "BT_11",
    "BT_12",
    "BTD",
    "lat",
    "lon",
    "CTT",
    "CTH",
    "hour",
    "day_of_year"
]

CATEGORICAL_FEATURES = [
    "Cloud_Mask_Raw",
    "Cloud_Phase"
]

TARGET = "ERA5_CBH"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("CBH V2 TRAINING DATASET PREPARATION")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))


# ============================================================
# FILTER DATA
# ============================================================

print("\n" + "=" * 80)
print("FILTERING TRAINING SAMPLES")
print("=" * 80)

rows_before = len(df)

# ------------------------------------------------------------
# Keep valid ERA5 CBH
# ------------------------------------------------------------

df = df[df[TARGET].notna()].copy()

print("\nAfter valid CBH filter:", len(df))

# ------------------------------------------------------------
# Keep cloudy pixels
# ------------------------------------------------------------

df = df[df["Cloud_Mask_Binary"] == 1].copy()

print("After cloudy-pixel filter:", len(df))

# ------------------------------------------------------------
# Keep complete MODIS cloud features
# ------------------------------------------------------------

required_cols = [
    "CTT",
    "CTH",
    "Cloud_Phase"
]

df = df.dropna(subset=required_cols).copy()

print("After complete cloud-feature filter:", len(df))

rows_after = len(df)

print("\nRows before filtering:", rows_before)
print("Rows after filtering :", rows_after)
print("Rows removed         :", rows_before - rows_after)


# ============================================================
# PARSE TIMESTAMP
# ============================================================

print("\n" + "=" * 80)
print("PARSING TIME FEATURES")
print("=" * 80)

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    format="%Y%j_%H%M"
)

df["hour"] = (
    df["datetime"].dt.hour
    + df["datetime"].dt.minute / 60.0
)

df["day_of_year"] = df["datetime"].dt.dayofyear


# ============================================================
# CLOUD PHASE ENCODING
# ============================================================

print("\n" + "=" * 80)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 80)

# Cloud_Phase values:
#
# 0 = clear
# 1 = liquid
# 2 = ice
# 6 = undetermined
#
# Convert to compact category indices.

PHASE_MAPPING = {
    0.0: 0,
    1.0: 1,
    2.0: 2,
    6.0: 3
}

df["Cloud_Phase"] = (
    df["Cloud_Phase"]
    .map(PHASE_MAPPING)
    .astype(int)
)

# Cloud_Mask_Raw values should be 0 or 1
# Convert to integer categorical indices.

df["Cloud_Mask_Raw"] = (
    df["Cloud_Mask_Raw"]
    .astype(int)
)


print("\nCloud Phase distribution:")

print(
    df["Cloud_Phase"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nCloud Mask Raw distribution:")

print(
    df["Cloud_Mask_Raw"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# TARGET TRANSFORMATION
# ============================================================

print("\n" + "=" * 80)
print("TARGET TRANSFORMATION")
print("=" * 80)

print("\nApplying log1p transformation to CBH...")

df["target_log_cbh"] = np.log1p(df[TARGET])


# ============================================================
# TEMPORAL SPLIT
# ============================================================

print("\n" + "=" * 80)
print("CREATING TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)


df["date"] = df["datetime"].dt.date


dates = sorted(df["date"].unique())

print("\nAvailable dates:")

for d in dates:
    print(d)


# ------------------------------------------------------------
# Split by DATE
#
# Train: Nov 19–22
# Validation: Nov 23
# Test: Nov 24
# ------------------------------------------------------------

train_dates = dates[:4]
val_dates = dates[4:5]
test_dates = dates[5:]


train_df = df[df["date"].isin(train_dates)].copy()

val_df = df[df["date"].isin(val_dates)].copy()

test_df = df[df["date"].isin(test_dates)].copy()


print("\nSplit sizes:")

print("Train     :", len(train_df))
print("Validation:", len(val_df))
print("Test      :", len(test_df))


print("\nTrain dates:")
print(sorted(train_df["date"].unique()))

print("\nValidation dates:")
print(sorted(val_df["date"].unique()))

print("\nTest dates:")
print(sorted(test_df["date"].unique()))


# ============================================================
# SCALE CONTINUOUS FEATURES
# ============================================================

print("\n" + "=" * 80)
print("SCALING CONTINUOUS FEATURES")
print("=" * 80)


scaler = StandardScaler()


print("\nFitting scaler using TRAINING DATA ONLY...")

train_df[CONTINUOUS_FEATURES] = scaler.fit_transform(
    train_df[CONTINUOUS_FEATURES]
)


print("Transforming validation data...")

val_df[CONTINUOUS_FEATURES] = scaler.transform(
    val_df[CONTINUOUS_FEATURES]
)


print("Transforming test data...")

test_df[CONTINUOUS_FEATURES] = scaler.transform(
    test_df[CONTINUOUS_FEATURES]
)


print("\nSaving scaler...")

joblib.dump(
    {
        "scaler": scaler,
        "continuous_features": CONTINUOUS_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES
    },
    SCALER_FILE
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

output_columns = (

    CONTINUOUS_FEATURES
    +

    CATEGORICAL_FEATURES
    +

    [
        "timestamp",
        "datetime",
        TARGET,
        "target_log_cbh"
    ]
)


train_df = train_df[output_columns]

val_df = val_df[output_columns]

test_df = test_df[output_columns]


# ============================================================
# SAVE DATASETS
# ============================================================

print("\n" + "=" * 80)
print("SAVING DATASETS")
print("=" * 80)


print("\nSaving training dataset...")

train_df.to_csv(
    TRAIN_FILE,
    index=False
)


print("Saving validation dataset...")

val_df.to_csv(
    VAL_FILE,
    index=False
)


print("Saving test dataset...")

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("CBH V2 DATASET PREPARATION COMPLETE")
print("=" * 80)


print("\nOutput files:")

print("1.", TRAIN_FILE)
print("2.", VAL_FILE)
print("3.", TEST_FILE)
print("4.", SCALER_FILE)


print("\nContinuous features:")

for i, feature in enumerate(CONTINUOUS_FEATURES, 1):
    print(f"{i}. {feature}")


print("\nCategorical features:")

for i, feature in enumerate(CATEGORICAL_FEATURES, 1):
    print(f"{i}. {feature}")


print("\nTarget:")

print(TARGET)


print("\nTransformed target:")

print("target_log_cbh = log1p(ERA5_CBH)")


print("\n" + "=" * 80)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("TARGET DISTRIBUTION BY SPLIT")


for name, data in [

    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)

]:

    print("\n" + name)

    print("-" * 60)

    print(

        data[TARGET]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
        .to_string()

    )


print("\n" + "=" * 80)
print("CBH V2 DATASET CREATION COMPLETE")
print("=" * 80)
