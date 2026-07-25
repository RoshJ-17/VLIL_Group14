import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "modis_gpm_precip.csv"

OUTPUT_FILE = "modis_gpm_precip_positive_modis.csv"

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
    "Cloud_Optical_Thickness",
]

TARGET = "GPM_precipitation_rate"

TIMESTAMP = "timestamp"


# ============================================================
# START
# ============================================================

print("=" * 90)
print("CREATE POSITIVE-PRECIPITATION MODIS DATASET")
print("=" * 90)

print()
print("Input :", INPUT_FILE)
print("Output:", OUTPUT_FILE)

# ============================================================
# LOAD
# ============================================================

print()
print("Loading dataset...")

usecols = [
    TIMESTAMP
] + FEATURES + [
    TARGET
]

df = pd.read_csv(
    INPUT_FILE,
    usecols=usecols
)

print("Rows loaded:", len(df))
print("Columns loaded:", len(df.columns))


# ============================================================
# PARSE TARGET
# ============================================================

print()
print("Parsing precipitation target...")

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

invalid_target = df[TARGET].isna().sum()

print(
    "Invalid precipitation values:",
    invalid_target
)

df = df.dropna(
    subset=[TARGET]
).reset_index(drop=True)


# ============================================================
# KEEP POSITIVE PRECIPITATION ONLY
# ============================================================

print()
print("Filtering positive precipitation...")

positive_df = df[
    df[TARGET] > 0
].copy()

positive_df = positive_df.reset_index(
    drop=True
)

print()
print("Original rows :", len(df))
print("Positive rows :", len(positive_df))

print(
    "Positive percentage:",
    f"{100 * len(positive_df) / len(df):.4f}%"
)


# ============================================================
# TIMESTAMP CHECK
# ============================================================

print()
print("Checking timestamps...")

positive_df[TIMESTAMP] = pd.to_datetime(
    positive_df[TIMESTAMP],
    format="%Y%j_%H%M",
    errors="coerce"
)

invalid_timestamp = positive_df[TIMESTAMP].isna().sum()

print(
    "Invalid timestamps:",
    invalid_timestamp
)

positive_df = positive_df.dropna(
    subset=[TIMESTAMP]
).copy()


# ============================================================
# SORT
# ============================================================

positive_df = positive_df.sort_values(
    TIMESTAMP
).reset_index(drop=True)


# ============================================================
# CONVERT TIMESTAMP BACK TO ORIGINAL FORMAT
# ============================================================

positive_df[TIMESTAMP] = positive_df[
    TIMESTAMP
].dt.strftime(
    "%Y%j_%H%M"
)


# ============================================================
# VERIFY COLUMNS
# ============================================================

expected_columns = [
    TIMESTAMP
] + FEATURES + [
    TARGET
]

positive_df = positive_df[
    expected_columns
]


print()
print("Final columns:")

for i, col in enumerate(
    positive_df.columns,
    1
):
    print(f"{i:2d}. {col}")


# ============================================================
# VERIFY NO GPM-DERIVED PREDICTORS
# ============================================================

print()
print("GPM-derived predictors excluded:")

print(" - GPM_randomError")
print(" - GPM_precipitationQualityIndex")


# ============================================================
# TARGET STATISTICS
# ============================================================

print()
print("=" * 90)
print("POSITIVE PRECIPITATION TARGET STATISTICS")
print("=" * 90)

print(
    positive_df[TARGET].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    )
)


# ============================================================
# MISSING VALUES
# ============================================================

print()
print("=" * 90)
print("MISSING VALUES")
print("=" * 90)

for col in positive_df.columns:

    missing = positive_df[col].isna().sum()

    if missing > 0:

        print(
            f"{col}: {missing}"
        )


# ============================================================
# SAVE
# ============================================================

print()
print("Saving dataset...")

positive_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 90)
print("DATASET CREATED")
print("=" * 90)

print()
print("File:")
print(OUTPUT_FILE)

print()
print("Rows:")
print(len(positive_df))

print()
print("Columns:")
print(len(positive_df.columns))

print()
print("Target minimum:")
print(positive_df[TARGET].min())

print()
print("Target maximum:")
print(positive_df[TARGET].max())

print()
print("Zero precipitation rows:")
print(
    (positive_df[TARGET] == 0).sum()
)

print()
print("=" * 90)
