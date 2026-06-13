import pandas as pd
import numpy as np

INPUT_FILE = "modis_gpm_precip.csv"
OUTPUT_FILE = "modis_gpm_cloudtype.csv"

print("=" * 80)
print("MODIS CLOUD-TYPE LABEL DATASET CREATION")
print("=" * 80)

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------

print("\nLoading MODIS + GPM merged data...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded rows: {len(df):,}")

# ------------------------------------------------------------------
# 2. CREATE CLOUD PHASE LABEL
# ------------------------------------------------------------------
#
# MODIS Cloud_Phase:
#   0 = clear
#   1 = liquid
#   2 = ice
#   6 = undetermined
#
# We keep "undetermined" because it is an actual MODIS phase
# category, not missing data.
# ------------------------------------------------------------------

print("\nCreating cloud phase labels...")

phase_map = {
    0.0: "clear",
    1.0: "liquid",
    2.0: "ice",
    6.0: "undetermined"
}

df["cloud_phase_label"] = df["Cloud_Phase"].map(phase_map)

# ------------------------------------------------------------------
# 3. CLEAN CTH
# ------------------------------------------------------------------
#
# CTH = 0 is treated as invalid/unknown rather than a genuine
# low cloud.
#
# This prevents artificial "low" labels from zero-valued CTH.
# ------------------------------------------------------------------

print("\nCleaning CTH...")

invalid_cth = df["CTH"].isna() | (df["CTH"] <= 0)

print("CTH missing:", df["CTH"].isna().sum())
print("CTH <= 0:", (df["CTH"] <= 0).sum())

# ------------------------------------------------------------------
# 4. CREATE CLOUD HEIGHT LABEL
# ------------------------------------------------------------------
#
# Valid CTH:
#
#   < 3 km       = low
#   3-6 km       = middle
#   > 6 km       = high
#
# We use metres because CTH is stored in metres.
#
# Clear pixels do not have a meaningful cloud height, therefore
# they receive "unknown".
# ------------------------------------------------------------------

print("\nCreating cloud height labels...")

df["cloud_height_label"] = "unknown"

valid_cloud = (
    (df["cloud_phase_label"] != "clear") &
    (~invalid_cth)
)

cth = df["CTH"]

df.loc[
    valid_cloud & (cth < 3000),
    "cloud_height_label"
] = "low"

df.loc[
    valid_cloud & (cth >= 3000) & (cth <= 6000),
    "cloud_height_label"
] = "middle"

df.loc[
    valid_cloud & (cth > 6000),
    "cloud_height_label"
] = "high"

# ------------------------------------------------------------------
# 5. CREATE COMBINED CLOUD TYPE
# ------------------------------------------------------------------

print("\nCreating combined cloud-type labels...")

df["cloud_type_label"] = "unknown"

# Clear sky
df.loc[
    df["cloud_phase_label"] == "clear",
    "cloud_type_label"
] = "clear"

# Cloud phase + height
cloud_types = [
    "liquid",
    "ice",
    "undetermined"
]

heights = [
    "low",
    "middle",
    "high"
]

for phase in cloud_types:
    for height in heights:

        mask = (
            (df["cloud_phase_label"] == phase) &
            (df["cloud_height_label"] == height)
        )

        df.loc[
            mask,
            "cloud_type_label"
        ] = f"{phase}_{height}"

# ------------------------------------------------------------------
# 6. QUALITY CHECKS
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("LABEL DISTRIBUTIONS")
print("=" * 80)

print("\nCloud Phase:")
print(
    df["cloud_phase_label"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nCloud Height:")
print(
    df["cloud_height_label"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nCombined Cloud Type:")
print(
    df["cloud_type_label"]
    .value_counts(dropna=False)
    .to_string()
)

# ------------------------------------------------------------------
# 7. PERCENTAGES
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("LABEL PERCENTAGES")
print("=" * 80)

print("\nCloud Phase (%):")
print(
    (df["cloud_phase_label"].value_counts(normalize=True) * 100)
    .round(3)
    .to_string()
)

print("\nCloud Height (%):")
print(
    (df["cloud_height_label"].value_counts(normalize=True) * 100)
    .round(3)
    .to_string()
)

print("\nCombined Cloud Type (%):")
print(
    (df["cloud_type_label"].value_counts(normalize=True) * 100)
    .round(3)
    .to_string()
)

# ------------------------------------------------------------------
# 8. IMPORTANT QC
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("QUALITY CHECKS")
print("=" * 80)

print("\nMissing Cloud Phase:")
print(df["cloud_phase_label"].isna().sum())

print("\nCTH missing:")
print(df["CTH"].isna().sum())

print("\nCTH <= 0:")
print((df["CTH"] <= 0).sum())

print("\nUnknown combined cloud type:")
print(
    (df["cloud_type_label"] == "unknown").sum()
)

print("\nUnexpected phase values:")
print(
    df.loc[
        df["Cloud_Phase"].notna() &
        df["cloud_phase_label"].isna(),
        "Cloud_Phase"
    ].value_counts()
)

# ------------------------------------------------------------------
# 9. FINAL 10-CLASS TARGET CHECK
# ------------------------------------------------------------------

final_classes = [
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
]

print("\n" + "=" * 80)
print("FINAL 10-CLASS TARGET")
print("=" * 80)

for cls in final_classes:
    count = (df["cloud_type_label"] == cls).sum()
    print(f"{cls:<25} {count:>10,}")

unknown_count = (
    df["cloud_type_label"] == "unknown"
).sum()

print(f"\nUnknown/excluded later: {unknown_count:,}")

# ------------------------------------------------------------------
# 10. SAVE
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("SAVING DATASET")
print("=" * 80)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nOutput file: {OUTPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nFinal columns:")

for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n" + "=" * 80)
print("CLOUD-TYPE DATASET CREATION COMPLETE")
print("=" * 80)
