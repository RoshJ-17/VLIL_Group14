import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_FILE = "cbh_train_v2.csv"
VAL_FILE   = "cbh_val_v2.csv"
TEST_FILE  = "cbh_test_v2.csv"

OUTPUT_TRAIN = "cbh_mixed_stratified_train.csv"
OUTPUT_VAL   = "cbh_mixed_stratified_val.csv"
OUTPUT_TEST  = "cbh_mixed_stratified_test.csv"

TARGET_COL = "ERA5_CBH"

RANDOM_STATE = 42

TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def print_separator():
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_separator()
    print("CBH MIXED-DAY STRATIFIED SPLIT")
    print("STRATIFICATION: DATE + CBH RANGE")
    print_separator()

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print("\nOriginal dataset sizes:")
    print(f"Train      : {len(train_df):,}")
    print(f"Validation : {len(val_df):,}")
    print(f"Test       : {len(test_df):,}")

    # =========================================================================
    # COMBINE DATA
    # =========================================================================

    print_separator()
    print("COMBINING DATASETS")
    print_separator()

    full_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    print(f"\nCombined dataset size: {len(full_df):,}")

    # =========================================================================
    # VALIDATE TARGET
    # =========================================================================

    if TARGET_COL not in full_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found."
        )

    missing_target = full_df[TARGET_COL].isna().sum()

    print(f"\nMissing {TARGET_COL}: {missing_target:,}")

    if missing_target > 0:
        print("Dropping rows with missing target...")
        full_df = full_df.dropna(subset=[TARGET_COL])

    # =========================================================================
    # PARSE DATETIME
    # =========================================================================

    print_separator()
    print("PARSING DATES")
    print_separator()

    if "datetime" not in full_df.columns:
        raise ValueError(
            "Column 'datetime' not found."
        )

    full_df["datetime"] = pd.to_datetime(
        full_df["datetime"],
        errors="coerce"
    )

    missing_datetime = full_df["datetime"].isna().sum()

    print(f"\nMissing/invalid datetime values: {missing_datetime:,}")

    if missing_datetime > 0:
        print("Dropping rows with invalid datetime...")
        full_df = full_df.dropna(subset=["datetime"])

    full_df["date"] = full_df["datetime"].dt.strftime("%Y-%m-%d")

    # =========================================================================
    # CREATE CBH RANGES
    # =========================================================================

    print_separator()
    print("CREATING CBH RANGES")
    print_separator()

    cbh_bins = [
        -np.inf,
        250,
        500,
        1000,
        2000,
        5000,
        10000,
        np.inf
    ]

    cbh_labels = [
        "0-250m",
        "250-500m",
        "500m-1km",
        "1-2km",
        "2-5km",
        "5-10km",
        ">10km"
    ]

    full_df["cbh_range"] = pd.cut(
        full_df[TARGET_COL],
        bins=cbh_bins,
        labels=cbh_labels,
        include_lowest=True
    )

    print("\nCBH range distribution:")

    print(
        full_df["cbh_range"]
        .value_counts()
        .sort_index()
    )

    # =========================================================================
    # CREATE STRATIFICATION LABEL
    # =========================================================================

    print_separator()
    print("CREATING STRATIFICATION LABEL")
    print_separator()

    full_df["stratify_label"] = (
        full_df["date"].astype(str)
        + "_"
        + full_df["cbh_range"].astype(str)
    )

    print("\nNumber of unique stratification groups:")

    print(full_df["stratify_label"].nunique())

    print("\nStratification groups:")

    group_counts = (
        full_df["stratify_label"]
        .value_counts()
        .sort_index()
    )

    print(group_counts)

    # =========================================================================
    # CHECK MINIMUM GROUP SIZE
    # =========================================================================

    min_group_size = group_counts.min()

    print(f"\nSmallest stratification group: {min_group_size}")

    if min_group_size < 3:

        print(
            "\nWARNING: Some stratification groups contain "
            "fewer than 3 samples."
        )

        print(
            "Those groups may cause train/validation/test "
            "stratification to fail."
        )

    # =========================================================================
    # FIRST SPLIT
    #
    # 80% TRAIN
    # 20% TEMPORARY
    # =========================================================================

    print_separator()
    print("CREATING TRAIN / TEMPORARY SPLIT")
    print_separator()

    train_split, temp_split = train_test_split(

        full_df,

        test_size=(VAL_SIZE + TEST_SIZE),

        random_state=RANDOM_STATE,

        shuffle=True,

        stratify=full_df["stratify_label"]
    )

    print("\nSplit sizes:")

    print(f"Train     : {len(train_split):,}")
    print(f"Temporary : {len(temp_split):,}")

    # =========================================================================
    # SECOND SPLIT
    #
    # TEMPORARY:
    #
    # 50% VALIDATION
    # 50% TEST
    #
    # RESULT:
    #
    # 80% TRAIN
    # 10% VALIDATION
    # 10% TEST
    # =========================================================================

    print_separator()
    print("CREATING VALIDATION / TEST SPLIT")
    print_separator()

    val_split, test_split = train_test_split(

        temp_split,

        test_size=0.50,

        random_state=RANDOM_STATE,

        shuffle=True,

        stratify=temp_split["stratify_label"]
    )

    print("\nFinal split sizes:")

    print(f"Train      : {len(train_split):,}")
    print(f"Validation : {len(val_split):,}")
    print(f"Test       : {len(test_split):,}")

    # =========================================================================
    # REMOVE TEMPORARY COLUMNS
    # =========================================================================

    print_separator()
    print("REMOVING TEMPORARY COLUMNS")
    print_separator()

    temporary_columns = [

        "date",

        "cbh_range",

        "stratify_label"

    ]

    train_split = train_split.drop(
        columns=temporary_columns
    )

    val_split = val_split.drop(
        columns=temporary_columns
    )

    test_split = test_split.drop(
        columns=temporary_columns
    )

    # =========================================================================
    # SAVE FILES
    # =========================================================================

    print_separator()
    print("SAVING DATASETS")
    print_separator()

    train_split.to_csv(
        OUTPUT_TRAIN,
        index=False
    )

    val_split.to_csv(
        OUTPUT_VAL,
        index=False
    )

    test_split.to_csv(
        OUTPUT_TEST,
        index=False
    )

    print()

    print(f"Saved: {OUTPUT_TRAIN}")

    print(f"Saved: {OUTPUT_VAL}")

    print(f"Saved: {OUTPUT_TEST}")

    # =========================================================================
    # FINAL DIAGNOSTICS
    # =========================================================================

    print_separator()
    print("FINAL SPLIT DIAGNOSTICS")
    print_separator()

    splits = {

        "TRAIN": train_split,

        "VALIDATION": val_split,

        "TEST": test_split

    }

    for name, df in splits.items():

        print()

        print("-" * 80)

        print(name)

        print("-" * 80)

        df_datetime = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

        dates = df_datetime.dt.strftime(
            "%Y-%m-%d"
        )

        print("\nDate distribution:")

        print(
            dates
            .value_counts()
            .sort_index()
        )

        print("\nTarget statistics:")

        print(
            df[TARGET_COL]
            .describe()
        )

        print("\nCBH range distribution:")

        temp_range = pd.cut(

            df[TARGET_COL],

            bins=cbh_bins,

            labels=cbh_labels,

            include_lowest=True

        )

        print(

            temp_range

            .value_counts()

            .sort_index()

        )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print_separator()
    print("MIXED-DAY STRATIFIED SPLIT COMPLETE")
    print_separator()

    print()

    print("Output files:")

    print(f"1. {OUTPUT_TRAIN}")

    print(f"2. {OUTPUT_VAL}")

    print(f"3. {OUTPUT_TEST}")

    print()

    print("Split configuration:")

    print(f"Train      : {TRAIN_SIZE * 100:.0f}%")

    print(f"Validation : {VAL_SIZE * 100:.0f}%")

    print(f"Test       : {TEST_SIZE * 100:.0f}%")

    print()

    print("Stratification:")

    print("DATE + CBH RANGE")

    print_separator()


if __name__ == "__main__":
    main()
