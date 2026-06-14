import pandas as pd
import os

INPUT = "modis_gpm_cloudtype.csv"
OUTPUT = "cloudtype_final.csv"

FEATURES = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
    "Cloud_Mask_Binary",
    "Cloud_Mask_Raw"
]

TARGET = "cloud_type_label"

# Timestamp is intentionally retained for temporal splitting.
COLUMNS = ["timestamp"] + FEATURES + [TARGET]

print("=" * 80)
print("CREATING FINAL CLOUD-TYPE DATASET")
print("=" * 80)

print("\nLoading MODIS + GPM cloud-type dataset...")

df = pd.read_csv(
    INPUT,
    usecols=COLUMNS
)

print("Loaded rows:", f"{len(df):,}")
print("Loaded columns:", len(df.columns))

# ------------------------------------------------------------------
# Check missing values
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("MISSING VALUE CHECK")
print("=" * 80)

print(df.isna().sum().to_string())

# Remove rows with missing values in actual ML features or target.
# Timestamp must also be valid.
before = len(df)

df = df.dropna(
    subset=COLUMNS
).copy()

removed = before - len(df)

print("\nRows removed:", f"{removed:,}")
print("Final rows:", f"{len(df):,}")

# ------------------------------------------------------------------
# Remove unknown target
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("TARGET CHECK")
print("=" * 80)

unknown = (df[TARGET] == "unknown").sum()

print("Unknown cloud-type rows:", f"{unknown:,}")

if unknown > 0:
    df = df[df[TARGET] != "unknown"].copy()

print("Rows after removing unknown:", f"{len(df):,}")

# ------------------------------------------------------------------
# Expected classes
# ------------------------------------------------------------------

EXPECTED_CLASSES = {
    "clear",
    "liquid_low",
    "liquid_middle",
    "liquid_high",
    "ice_low",
    "ice_middle",
    "ice_high",
    "undetermined_low",
    "undetermined_middle",
    "undetermined_high"
}

actual_classes = set(df[TARGET].unique())

print("\nExpected classes:")
for cls in sorted(EXPECTED_CLASSES):
    print(" ", cls)

print("\nActual classes:")
for cls in sorted(actual_classes):
    print(" ", cls)

print("\nUnexpected classes:", actual_classes - EXPECTED_CLASSES)
print("Missing classes:", EXPECTED_CLASSES - actual_classes)

assert actual_classes <= EXPECTED_CLASSES
assert len(EXPECTED_CLASSES - actual_classes) == 0

# ------------------------------------------------------------------
# Timestamp validation
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("TIMESTAMP CHECK")
print("=" * 80)

print("Unique timestamps:", df["timestamp"].nunique())

print("\nUnique observation days:")

df["date_group"] = df["timestamp"].astype(str).str[:7]

unique_days = sorted(df["date_group"].unique())

for day in unique_days:
    print(" ", day)

print("\nNumber of observation days:", len(unique_days))

# date_group is only a temporary validation column.
df.drop(columns=["date_group"], inplace=True)

# ------------------------------------------------------------------
# Target distribution
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL TARGET DISTRIBUTION")
print("=" * 80)

counts = df[TARGET].value_counts()

for cls in sorted(EXPECTED_CLASSES):
    count = counts.get(cls, 0)

    print(
        f"{cls:<25} "
        f"{count:>10,} "
        f"({count / len(df) * 100:6.2f}%)"
    )

# ------------------------------------------------------------------
# Feature validation
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL ML FEATURES")
print("=" * 80)

for i, feature in enumerate(FEATURES, 1):
    print(f"{i}. {feature}")

print("\nTarget:", TARGET)

print("\nTimestamp: RETAINED FOR TEMPORAL SPLITTING ONLY")

# Explicitly verify that forbidden variables are absent.
FORBIDDEN = [
    "CTH",
    "CTT",
    "Cloud_Phase",
    "cloud_phase_label",
    "cloud_height_label",
    "Cloud_Optical_Thickness",
    "GPM_randomError",
    "GPM_precipitationQualityIndex",
    "GPM_precipitation_rate"
]

for col in FORBIDDEN:
    assert col not in df.columns, (
        f"{col} should not be present in the final dataset."
    )

print("\nExcluded variables verified.")

# ------------------------------------------------------------------
# Final column order
# ------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "timestamp",
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
    "Cloud_Mask_Binary",
    "Cloud_Mask_Raw",
    "cloud_type_label"
]

assert list(df.columns) == EXPECTED_COLUMNS

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("SAVING FINAL DATASET")
print("=" * 80)

df.to_csv(
    OUTPUT,
    index=False
)

size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)

print("\nOutput file:", OUTPUT)
print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))
print(f"File size: {size_mb:.2f} MB")

print("\nFinal columns:")

for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n" + "=" * 80)
print("FINAL CLOUD-TYPE DATASET CREATED")
print("=" * 80)

print("""
Important:
- timestamp is retained for temporal train/validation/test splitting.
- timestamp will NOT be used as an ML feature.
- Cloud_Optical_Thickness is excluded.
- CTH is excluded.
- CTT is excluded.
- Cloud_Phase is excluded.
- Cloud phase/height labels are excluded as input features.
- GPM variables are excluded.
- cloud_type_label is the target.
""")
