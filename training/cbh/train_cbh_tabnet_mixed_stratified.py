#!/usr/bin/env python3

import os
import json
import random
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from pytorch_tabnet.tab_model import TabNetRegressor


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_FILE = "cbh_mixed_stratified_train.csv"
VAL_FILE   = "cbh_mixed_stratified_val.csv"
TEST_FILE  = "cbh_mixed_stratified_test.csv"

MODEL_NAME = "cbh_tabnet_mixed_stratified"

MODEL_SAVE_PATH = f"{MODEL_NAME}_model"
METRICS_FILE = f"{MODEL_NAME}_metrics.json"
PREDICTIONS_FILE = f"{MODEL_NAME}_test_predictions.csv"
RANGE_FILE = f"{MODEL_NAME}_range_evaluation.csv"
FEATURE_FILE = f"{MODEL_NAME}_feature_importance.csv"


# =============================================================================
# RANDOM SEED
# =============================================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =============================================================================
# FEATURES
# =============================================================================

NUMERIC_FEATURES = [
    "BT_11",
    "BT_12",
    "BTD",
    "lat",
    "lon",
    "CTT",
    "CTH"
]

CATEGORICAL_FEATURES = [
    "Cloud_Mask_Raw",
    "Cloud_Phase"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "ERA5_CBH"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_separator():
    print("=" * 80)


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    bias = np.mean(y_pred - y_true)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Bias": float(bias)
    }


def print_metrics(metrics):

    print(f"MAE  : {metrics['MAE']:.2f} meters")
    print(f"RMSE : {metrics['RMSE']:.2f} meters")
    print(f"R²   : {metrics['R2']:.4f}")
    print(f"Bias : {metrics['Bias']:.2f} meters")


def create_cbh_ranges(cbh):

    conditions = [
        cbh < 250,
        (cbh >= 250) & (cbh < 500),
        (cbh >= 500) & (cbh < 1000),
        (cbh >= 1000) & (cbh < 2000),
        (cbh >= 2000) & (cbh < 5000),
        (cbh >= 5000) & (cbh < 10000),
        cbh >= 10000
    ]

    labels = [
        "0-250m",
        "250-500m",
        "500m-1km",
        "1-2km",
        "2-5km",
        "5-10km",
        ">10km"
    ]

    return np.select(
        conditions,
        labels,
        default="Unknown"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_separator()
    print("CBH TABNET TRAINING")
    print("SPLIT TYPE: MIXED-DAY STRATIFIED")
    print_separator()

    print()

    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():

        print(f"CUDA available: YES")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        DEVICE_NAME = "cuda"

    else:

        print("CUDA available: NO")
        print("Using CPU")

        DEVICE_NAME = "cpu"

    print_separator()


    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print("LOADING DATASETS")
    print_separator()

    print()
    print("Loading training data...")

    train_df = pd.read_csv(TRAIN_FILE)

    print("Loading validation data...")

    val_df = pd.read_csv(VAL_FILE)

    print("Loading test data...")

    test_df = pd.read_csv(TEST_FILE)

    print()

    print("Dataset sizes:")

    print(f"Train      : {len(train_df):,}")
    print(f"Validation : {len(val_df):,}")
    print(f"Test       : {len(test_df):,}")

    print_separator()


    # =========================================================================
    # DATA VALIDATION
    # =========================================================================

    print("DATA VALIDATION")
    print_separator()

    datasets = {
        "TRAIN": train_df,
        "VALIDATION": val_df,
        "TEST": test_df
    }

    required_columns = FEATURES + [TARGET]

    for name, df in datasets.items():

        print()
        print("-" * 80)
        print(name)
        print("-" * 80)

        print()
        print(f"Rows: {len(df):,}")

        missing_columns = [

            col for col in required_columns

            if col not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing columns in {name}: "
                f"{missing_columns}"
            )

        print()
        print("Missing values:")

        print(
            df[required_columns]
            .isnull()
            .sum()
        )

    print_separator()


    # =========================================================================
    # PREPARE NUMERIC FEATURES
    # =========================================================================

    print("PREPARING FEATURES")
    print_separator()

    print()
    print("Converting numeric features...")

    for df in [train_df, val_df, test_df]:

        for col in NUMERIC_FEATURES:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df[TARGET] = pd.to_numeric(
            df[TARGET],
            errors="coerce"
        )


    print()
    print("Missing values after numeric conversion:")

    print(
        train_df[
            NUMERIC_FEATURES + [TARGET]
        ]
        .isnull()
        .sum()
    )


    # =========================================================================
    # HANDLE CATEGORICAL FEATURES
    # =========================================================================

    print()
    print("Encoding categorical features...")

    print()

    print("Cloud_Mask_Raw values:")

    print(
        sorted(
            train_df["Cloud_Mask_Raw"]
            .dropna()
            .unique()
        )
    )

    print()

    print("Cloud_Phase values:")

    print(
        sorted(
            train_df["Cloud_Phase"]
            .dropna()
            .unique()
        )
    )


    # =========================================================================
    # CONSISTENT CATEGORY ENCODING
    # =========================================================================

    for categorical_column in CATEGORICAL_FEATURES:

        all_values = pd.concat([
            train_df[categorical_column],
            val_df[categorical_column],
            test_df[categorical_column]
        ])

        unique_values = sorted(
            all_values
            .dropna()
            .unique()
        )

        mapping = {

            value: index

            for index, value

            in enumerate(unique_values)
        }

        print()

        print(
            f"{categorical_column} mapping:"
        )

        print(mapping)

        train_df[categorical_column] = (
            train_df[categorical_column]
            .map(mapping)
        )

        val_df[categorical_column] = (
            val_df[categorical_column]
            .map(mapping)
        )

        test_df[categorical_column] = (
            test_df[categorical_column]
            .map(mapping)
        )


    # =========================================================================
    # CHECK MISSING VALUES
    # =========================================================================

    print()
    print("Final missing value check:")

    for name, df in datasets.items():

        missing = df[
            FEATURES + [TARGET]
        ].isnull().sum().sum()

        print(
            f"{name.capitalize()} missing values: "
            f"{missing}"
        )

        if missing > 0:

            raise ValueError(
                f"Missing values found in {name}"
            )


    # =========================================================================
    # SCALE NUMERIC FEATURES
    # =========================================================================

    print()
    print("Scaling numeric features...")

    scaler = StandardScaler()

    train_numeric = scaler.fit_transform(
        train_df[NUMERIC_FEATURES]
    )

    val_numeric = scaler.transform(
        val_df[NUMERIC_FEATURES]
    )

    test_numeric = scaler.transform(
        test_df[NUMERIC_FEATURES]
    )


    # =========================================================================
    # CATEGORICAL MATRICES
    # =========================================================================

    train_categorical = (
        train_df[CATEGORICAL_FEATURES]
        .values
        .astype(np.float32)
    )

    val_categorical = (
        val_df[CATEGORICAL_FEATURES]
        .values
        .astype(np.float32)
    )

    test_categorical = (
        test_df[CATEGORICAL_FEATURES]
        .values
        .astype(np.float32)
    )


    # =========================================================================
    # COMBINE FEATURES
    # =========================================================================

    X_train = np.column_stack([
        train_numeric,
        train_categorical
    ]).astype(np.float32)

    X_val = np.column_stack([
        val_numeric,
        val_categorical
    ]).astype(np.float32)

    X_test = np.column_stack([
        test_numeric,
        test_categorical
    ]).astype(np.float32)


    y_train = (
        train_df[TARGET]
        .values
        .astype(np.float32)
        .reshape(-1, 1)
    )

    y_val = (
        val_df[TARGET]
        .values
        .astype(np.float32)
        .reshape(-1, 1)
    )

    y_test = (
        test_df[TARGET]
        .values
        .astype(np.float32)
        .reshape(-1, 1)
    )


    print()

    print("Feature matrix shapes:")

    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")

    print()

    print("Final features:")

    for index, feature in enumerate(FEATURES):

        print(f"{index:2d}: {feature}")

    print_separator()


    # =========================================================================
    # BASELINE
    # =========================================================================

    print("BASELINE COMPARISON")
    print_separator()

    train_mean = float(
        np.mean(y_train)
    )

    baseline_predictions = np.full(
        len(y_test),
        train_mean
    )

    baseline_metrics = calculate_metrics(

        y_test.flatten(),

        baseline_predictions

    )

    print()

    print(
        f"Training mean: "
        f"{train_mean:.2f} m"
    )

    print()

    print_metrics(
        baseline_metrics
    )

    print_separator()


    # =========================================================================
    # TABNET MODEL
    # =========================================================================

    print("INITIALIZING TABNET")
    print_separator()

    print()

    print("Model configuration:")

    print("Target          : RAW ERA5_CBH")

    print("n_d             : 32")
    print("n_a             : 32")
    print("n_steps         : 5")

    print("gamma           : 1.5")

    print("n_independent   : 2")
    print("n_shared        : 2")

    print("lambda_sparse   : 1e-4")

    print("Optimizer       : Adam")

    print("Learning rate   : 0.02")

    print("Batch size      : 8192")

    print("Virtual batch   : 1024")

    print("Max epochs      : 200")

    print("Patience        : 20")

    print(f"Device          : {DEVICE_NAME}")

    print_separator()


    model = TabNetRegressor(

        n_d=32,

        n_a=32,

        n_steps=5,

        gamma=1.5,

        n_independent=2,

        n_shared=2,

        lambda_sparse=1e-4,

        optimizer_fn=torch.optim.Adam,

        optimizer_params=dict(
            lr=0.02
        ),

        mask_type="entmax",

        seed=RANDOM_SEED,

        verbose=10,

        device_name=DEVICE_NAME

    )


    # =========================================================================
    # TRAINING
    # =========================================================================

    print("STARTING TABNET TRAINING")
    print_separator()

    model.fit(

        X_train=X_train,

        y_train=y_train,

        eval_set=[

            (X_train, y_train),

            (X_val, y_val)

        ],

        eval_name=[

            "train",

            "validation"

        ],

        eval_metric=[

            "rmse"

        ],

        max_epochs=200,

        patience=20,

        batch_size=8192,

        virtual_batch_size=1024,

        num_workers=0,

        drop_last=False

    )

    print_separator()

    print("TRAINING COMPLETE")

    print_separator()


    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    print("SAVING MODEL")
    print_separator()

    model.save_model(
        MODEL_SAVE_PATH
    )

    print()

    print(
        f"Model saved: "
        f"{MODEL_SAVE_PATH}.zip"
    )

    print_separator()


    # =========================================================================
    # PREDICTIONS
    # =========================================================================

    print("RUNNING PREDICTIONS")
    print_separator()

    train_predictions = (

        model.predict(X_train)

        .flatten()

    )

    val_predictions = (

        model.predict(X_val)

        .flatten()

    )

    test_predictions = (

        model.predict(X_test)

        .flatten()

    )


    # =========================================================================
    # METRICS
    # =========================================================================

    train_metrics = calculate_metrics(

        y_train.flatten(),

        train_predictions

    )

    val_metrics = calculate_metrics(

        y_val.flatten(),

        val_predictions

    )

    test_metrics = calculate_metrics(

        y_test.flatten(),

        test_predictions

    )


    # =========================================================================
    # TEST PERFORMANCE
    # =========================================================================

    print("TEST PERFORMANCE")
    print_separator()

    print_metrics(
        test_metrics
    )

    print_separator()


    # =========================================================================
    # TRAIN PERFORMANCE
    # =========================================================================

    print("TRAIN PERFORMANCE")
    print_separator()

    print_metrics(
        train_metrics
    )

    print_separator()


    # =========================================================================
    # VALIDATION PERFORMANCE
    # =========================================================================

    print("VALIDATION PERFORMANCE")
    print_separator()

    print_metrics(
        val_metrics
    )

    print_separator()


    # =========================================================================
    # SAVE TEST PREDICTIONS
    # =========================================================================

    print("SAVING TEST PREDICTIONS")
    print_separator()

    prediction_df = test_df.copy()

    prediction_df["actual_CBH"] = (

        y_test.flatten()

    )

    prediction_df["predicted_CBH"] = (

        test_predictions

    )

    prediction_df["error"] = (

        prediction_df["predicted_CBH"]

        -

        prediction_df["actual_CBH"]

    )

    prediction_df["absolute_error"] = (

        np.abs(
            prediction_df["error"]
        )

    )


    prediction_df.to_csv(

        PREDICTIONS_FILE,

        index=False

    )

    print()

    print(
        f"Saved: "
        f"{PREDICTIONS_FILE}"
    )

    print_separator()


    # =========================================================================
    # ERROR ANALYSIS BY CBH RANGE
    # =========================================================================

    print("ERROR ANALYSIS BY CBH RANGE")
    print_separator()

    prediction_df["cbh_range"] = (

        create_cbh_ranges(

            prediction_df["actual_CBH"]

        )

    )


    range_results = []


    range_order = [

        "0-250m",

        "250-500m",

        "500m-1km",

        "1-2km",

        "2-5km",

        "5-10km",

        ">10km"

    ]


    for cbh_range in range_order:

        subset = prediction_df[

            prediction_df["cbh_range"]

            == cbh_range

        ]


        if len(subset) == 0:

            continue


        metrics = calculate_metrics(

            subset["actual_CBH"],

            subset["predicted_CBH"]

        )


        range_results.append({

            "cbh_range":

                cbh_range,

            "count":

                len(subset),

            "actual_mean":

                subset["actual_CBH"].mean(),

            "predicted_mean":

                subset["predicted_CBH"].mean(),

            "MAE":

                metrics["MAE"],

            "RMSE":

                metrics["RMSE"],

            "Bias":

                metrics["Bias"]

        })


    range_df = pd.DataFrame(

        range_results

    )


    print()

    print(

        range_df.to_string(

            index=False

        )

    )


    range_df.to_csv(

        RANGE_FILE,

        index=False

    )


    print()

    print(

        f"Range evaluation saved: "

        f"{RANGE_FILE}"

    )

    print_separator()


    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    print("TABNET FEATURE IMPORTANCE")
    print_separator()

    feature_importance = pd.DataFrame({

        "feature":

            FEATURES,

        "importance":

            model.feature_importances_

    })


    feature_importance = (

        feature_importance

        .sort_values(

            "importance",

            ascending=False

        )

        .reset_index(

            drop=True

        )

    )


    print()

    print(

        feature_importance.to_string(

            index=False

        )

    )


    feature_importance.to_csv(

        FEATURE_FILE,

        index=False

    )


    print()

    print(

        f"Feature importance saved: "

        f"{FEATURE_FILE}"

    )

    print_separator()


    # =========================================================================
    # SAVE METRICS
    # =========================================================================

    print("SAVING METRICS")
    print_separator()


    all_metrics = {

        "split_type":

            "mixed_stratified",

        "features":

            FEATURES,

        "target":

            TARGET,

        "baseline":

            baseline_metrics,

        "train":

            train_metrics,

        "validation":

            val_metrics,

        "test":

            test_metrics,

        "tabnet_parameters": {

            "n_d": 32,

            "n_a": 32,

            "n_steps": 5,

            "gamma": 1.5,

            "n_independent": 2,

            "n_shared": 2,

            "lambda_sparse": 1e-4,

            "learning_rate": 0.02,

            "batch_size": 8192,

            "virtual_batch_size": 1024,

            "max_epochs": 200,

            "patience": 20,

            "mask_type": "entmax",

            "seed": RANDOM_SEED

        }

    }


    with open(

        METRICS_FILE,

        "w"

    ) as f:

        json.dump(

            all_metrics,

            f,

            indent=4

        )


    print()

    print(

        f"Metrics saved: "

        f"{METRICS_FILE}"

    )

    print_separator()


    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print("CBH TABNET TRAINING COMPLETE")

    print_separator()

    print()

    print(

        "Split type: "

        "mixed_stratified"

    )

    print()

    print("Model:")

    print(

        f"{MODEL_SAVE_PATH}.zip"

    )

    print()

    print("Test predictions:")

    print(PREDICTIONS_FILE)

    print()

    print("Metrics:")

    print(METRICS_FILE)

    print()

    print("Range evaluation:")

    print(RANGE_FILE)

    print()

    print("Feature importance:")

    print(FEATURE_FILE)

    print()

    print("Final Test Performance:")

    print()

    print_metrics(

        test_metrics

    )

    print_separator()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()
