import pandas as pd
import numpy as np

INPUT_FILE = "modis_gpm_precip.csv"
OUTPUT_FILE = "modis_gpm_precip_detection.csv"

print("=" * 80)
print("GPM PRECIPITATION DETECTION LABEL CREATION")
print("=" * 80)

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")

# ------------------------------------------------------------------
# 2. CHECK PRECIPITATION RATE
# ------------------------------------------------------------------

rate_col = "GPM_precipitation_rate"

print("\nChecking precipitation-rate target...")

print("Missing rate:", df[rate_col].isna().sum())
print("Negative rate:", (df[rate_col] < 0).sum())
print("Zero rate:", (df[rate_col] == 0).sum())
print("Positive rate:", (df[rate_col] > 0).sum())

# ------------------------------------------------------------------
# 3. CREATE BINARY PRECIPITATION TARGET
# ------------------------------------------------------------------
#
# 0 = No precipitation
# 1 = Precipitation
#
# Missing GPM rate remains NaN rather than being incorrectly
# classified as "no precipitation".
# ------------------------------------------------------------------

print("\nCreating precipitation detection target...")

df["precipitation_flag"] = np.nan

valid_rate = df[rate_col].notna()

df.loc[
    valid_rate & (df[rate_col] == 0),
    "precipitation_flag"
] = 0

df.loc[
    valid_rate & (df[rate_col] > 0),
    "precipitation_flag"
] = 1

# ------------------------------------------------------------------
# 4. LABEL DISTRIBUTION
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("PRECIPITATION DETECTION LABEL DISTRIBUTION")
print("=" * 80)

counts = df["precipitation_flag"].value_counts(dropna=False).sort_index()

print(counts.to_string())

print("\nMeaning:")
print("0 = No precipitation")
print("1 = Precipitation")
print("NaN = Missing GPM precipitation rate")

# ------------------------------------------------------------------
# 5. PERCENTAGES
# ------------------------------------------------------------------

valid_labels = df["precipitation_flag"].dropna()

print("\nPercentages among valid GPM observations:")

print(
    (valid_labels.value_counts(normalize=True) * 100)
    .sort_index()
    .round(3)
    .to_string()
)

# ------------------------------------------------------------------
# 6. CROSS-CHECK AGAINST RATE
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("QUALITY CHECK")
print("=" * 80)

print("\nRate = 0 but precipitation_flag != 0:")

print(
    (
        df.loc[
            valid_rate & (df[rate_col] == 0),
            "precipitation_flag"
        ] != 0
    ).sum()
)

print("\nRate > 0 but precipitation_flag != 1:")

print(
    (
        df.loc[
            valid_rate & (df[rate_col] > 0),
            "precipitation_flag"
        ] != 1
    ).sum()
)

print("\nMissing rate rows:")
missing = df[df[rate_col].isna()]

print(f"Rows: {len(missing)}")

if len(missing) > 0:
    print(
        missing[
            [
                "timestamp",
                "lat",
                "lon",
                rate_col,
                "GPM_randomError",
                "GPM_precipitationQualityIndex"
            ]
        ].to_string(index=False)
    )

# ------------------------------------------------------------------
# 7. CLASS IMBALANCE
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("CLASS IMBALANCE")
print("=" * 80)

n_no_precip = (df["precipitation_flag"] == 0).sum()
n_precip = (df["precipitation_flag"] == 1).sum()

if n_precip > 0:
    imbalance_ratio = n_no_precip / n_precip
else:
    imbalance_ratio = np.inf

print(f"No precipitation : {n_no_precip:,}")
print(f"Precipitation     : {n_precip:,}")
print(f"Imbalance ratio   : {imbalance_ratio:.2f}:1")

# ------------------------------------------------------------------
# 8. FINAL DATASET CHECK
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL DATASET")
print("=" * 80)

print("Shape:", df.shape)

print("\nColumns:")

for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# ------------------------------------------------------------------
# 9. SAVE
# ------------------------------------------------------------------

print("\nSaving...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nOutput file: {OUTPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\n" + "=" * 80)
print("PRECIPITATION DETECTION DATASET COMPLETE")
print("=" * 80)
