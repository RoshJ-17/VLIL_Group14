import pandas as pd
import os

INPUT = "cloudtype_final.csv"

TRAIN_OUTPUT = "cloudtype_train.csv"
VAL_OUTPUT   = "cloudtype_val.csv"
TEST_OUTPUT  = "cloudtype_test.csv"

TARGET = "cloud_type_label"

FEATURES = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
    "Cloud_Mask_Binary",
    "Cloud_Mask_Raw"
]

print("=" * 80)
print("CLOUD-TYPE TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)

# ------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(INPUT)

print("Total rows:", f"{len(df):,}")
print("Columns:", len(df.columns))

# ------------------------------------------------------------------
# 2. Extract observation day
# ------------------------------------------------------------------

df["date_group"] = df["timestamp"].astype(str).str[:7]

unique_days = sorted(df["date_group"].unique())

print("\nObservation days:")

for day in unique_days:
    print(" ", day)

print("\nNumber of observation days:", len(unique_days))

if len(unique_days) != 6:
    raise ValueError(
        f"Expected exactly 6 observation days, found {len(unique_days)}"
    )

# ------------------------------------------------------------------
# 3. Temporal split
# ------------------------------------------------------------------

train_days = unique_days[:4]
val_days   = unique_days[4:5]
test_days  = unique_days[5:6]

print("\n" + "=" * 80)
print("TEMPORAL SPLIT")
print("=" * 80)

print("\nTRAIN:")
for day in train_days:
    print(" ", day)

print("\nVALIDATION:")
for day in val_days:
    print(" ", day)

print("\nTEST:")
for day in test_days:
    print(" ", day)

# ------------------------------------------------------------------
# 4. Create datasets
# ------------------------------------------------------------------

train_df = df[df["date_group"].isin(train_days)].copy()
val_df   = df[df["date_group"].isin(val_days)].copy()
test_df  = df[df["date_group"].isin(test_days)].copy()

# Remove helper column
train_df.drop(columns=["date_group"], inplace=True)
val_df.drop(columns=["date_group"], inplace=True)
test_df.drop(columns=["date_group"], inplace=True)

# ------------------------------------------------------------------
# 5. Row counts
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("ROW COUNTS")
print("=" * 80)

print(f"\nTotal:      {len(df):>10,}")
print(f"Train:      {len(train_df):>10,}")
print(f"Validation: {len(val_df):>10,}")
print(f"Test:       {len(test_df):>10,}")

print(f"\nTrain:      {len(train_df)/len(df)*100:.2f}%")
print(f"Validation: {len(val_df)/len(df)*100:.2f}%")
print(f"Test:       {len(test_df)/len(df)*100:.2f}%")

assert len(train_df) + len(val_df) + len(test_df) == len(df)

# ------------------------------------------------------------------
# 6. Timestamp leakage check
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("TIMESTAMP LEAKAGE CHECK")
print("=" * 80)

train_ts = set(train_df["timestamp"])
val_ts   = set(val_df["timestamp"])
test_ts  = set(test_df["timestamp"])

print("\nTrain ∩ Validation:",
      len(train_ts & val_ts))

print("Train ∩ Test:",
      len(train_ts & test_ts))

print("Validation ∩ Test:",
      len(val_ts & test_ts))

assert len(train_ts & val_ts) == 0
assert len(train_ts & test_ts) == 0
assert len(val_ts & test_ts) == 0

print("\nNo timestamp overlap detected.")

# ------------------------------------------------------------------
# 7. Target distribution
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

classes = sorted(df[TARGET].unique())

for name, subset in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print("\n" + name)
    print("-" * 70)

    counts = subset[TARGET].value_counts()

    for cls in classes:

        count = counts.get(cls, 0)
        pct = count / len(subset) * 100

        print(
            f"{cls:<25}"
            f"{count:>10,} "
            f"({pct:6.2f}%)"
        )

# ------------------------------------------------------------------
# 8. Class presence
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("CLASS PRESENCE CHECK")
print("=" * 80)

for name, subset in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    present = set(subset[TARGET].unique())
    missing = set(classes) - present

    print(f"\n{name}:")

    if missing:
        print("Missing classes:")
        for cls in sorted(missing):
            print(" ", cls)
    else:
        print("All 10 classes present.")

# ------------------------------------------------------------------
# 9. Feature missing values
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURE MISSING-VALUE CHECK")
print("=" * 80)

for name, subset in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print(f"\n{name}:")
    print(subset[FEATURES].isna().sum().to_string())

# ------------------------------------------------------------------
# 10. Verify timestamp is NOT an ML feature
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURE VALIDATION")
print("=" * 80)

print("\nML features:")

for i, feature in enumerate(FEATURES, 1):
    print(f"{i}. {feature}")

print("\nTarget:", TARGET)

print("\nTimestamp:")
print("Retained in CSV for temporal tracking only.")
print("It will NOT be supplied to the DL model.")

# ------------------------------------------------------------------
# 11. Save
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("SAVING DATASETS")
print("=" * 80)

train_df.to_csv(TRAIN_OUTPUT, index=False)
val_df.to_csv(VAL_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)

for file in [
    TRAIN_OUTPUT,
    VAL_OUTPUT,
    TEST_OUTPUT
]:

    size_mb = os.path.getsize(file) / (1024 * 1024)

    print(
        f"{file:<25}"
        f"{len(pd.read_csv(file)):>12,} rows "
        f"{size_mb:>10.2f} MB"
    )

# ------------------------------------------------------------------
# 12. Final
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("CLOUD-TYPE TEMPORAL SPLIT COMPLETE")
print("=" * 80)

print("\nCreated:")
print(" ", TRAIN_OUTPUT)
print(" ", VAL_OUTPUT)
print(" ", TEST_OUTPUT)

print("\nNo model training was performed.")

