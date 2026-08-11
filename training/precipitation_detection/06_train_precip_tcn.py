#!/usr/bin/env python3

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "modis_gpm_precip_detection.csv"

MODEL_FILE = "precipitation_tcn.pt"

SEQUENCE_LENGTH = 12       # 12 consecutive observations
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 1e-3

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Features
# IMPORTANT:
# Do NOT include GPM_precipitation_rate.
# Do NOT include precipitation_flag.
# Do NOT include GPM-derived variables.
# ------------------------------------------------------------

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

TARGET = "precipitation_flag"

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=" * 80)
print("PRECIPITATION DETECTION - TCN")
print("=" * 80)

print("\nDevice:", DEVICE)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

usecols = ["timestamp"] + FEATURES + [TARGET]

df = pd.read_csv(
    CSV_FILE,
    usecols=usecols
)

print("Rows loaded:", len(df))

# ============================================================
# BASIC CLEANING
# ============================================================

print("\nMissing values before cleaning:")

print(df.isna().sum().sort_values(ascending=False).to_string())

# Remove rows where target is missing
df = df.dropna(subset=[TARGET]).copy()

# ============================================================
# PARSE TIMESTAMP
# ============================================================

print("\nParsing timestamps...")

# Dataset timestamp format:
# 2022323_0345
# YYYY + Julian day + HHMM

def parse_timestamp(x):

    x = str(x)

    year = int(x[:4])
    julian_day = int(x[4:7])
    hour = int(x[8:10])
    minute = int(x[10:12])

    return pd.Timestamp(year=year, month=1, day=1) + \
           pd.Timedelta(days=julian_day - 1,
                         hours=hour,
                         minutes=minute)


df["datetime"] = df["timestamp"].apply(parse_timestamp)

df = df.sort_values("datetime").reset_index(drop=True)

print(
    "Time range:",
    df["datetime"].min(),
    "to",
    df["datetime"].max()
)

# ============================================================
# HANDLE FEATURE MISSING VALUES
# ============================================================

print("\nHandling missing predictor values...")

# Cloud variables contain legitimate missing values.
# We use median imputation based ONLY on the training period later.
# For now, retain NaNs.

# ============================================================
# TEMPORAL SPLIT
# ============================================================

print("\nCreating chronological train/validation/test split...")

unique_times = np.sort(df["datetime"].unique())

n_times = len(unique_times)

train_end = int(n_times * TRAIN_RATIO)
val_end = int(n_times * (TRAIN_RATIO + VAL_RATIO))

train_times = unique_times[:train_end]
val_times = unique_times[train_end:val_end]
test_times = unique_times[val_end:]

train_df = df[df["datetime"].isin(train_times)].copy()
val_df = df[df["datetime"].isin(val_times)].copy()
test_df = df[df["datetime"].isin(test_times)].copy()

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nSplit time ranges:")

print(
    "Train:",
    train_df["datetime"].min(),
    "->",
    train_df["datetime"].max()
)

print(
    "Validation:",
    val_df["datetime"].min(),
    "->",
    val_df["datetime"].max()
)

print(
    "Test:",
    test_df["datetime"].min(),
    "->",
    test_df["datetime"].max()
)

# ============================================================
# IMPUTATION
# ============================================================

print("\nCalculating training-set medians...")

train_medians = train_df[FEATURES].median()

train_df[FEATURES] = train_df[FEATURES].fillna(train_medians)
val_df[FEATURES] = val_df[FEATURES].fillna(train_medians)
test_df[FEATURES] = test_df[FEATURES].fillna(train_medians)

# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing features...")

scaler = StandardScaler()

train_df[FEATURES] = scaler.fit_transform(
    train_df[FEATURES]
)

val_df[FEATURES] = scaler.transform(
    val_df[FEATURES]
)

test_df[FEATURES] = scaler.transform(
    test_df[FEATURES]
)

# ============================================================
# DATASET CREATION
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, df, features, target, sequence_length):

        self.X = df[features].values.astype(np.float32)
        self.y = df[target].values.astype(np.float32)

        self.sequence_length = sequence_length

    def __len__(self):

        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):

        x = self.X[
            idx:idx + self.sequence_length
        ]

        # Target corresponds to the final timestep
        y = self.y[
            idx + self.sequence_length - 1
        ]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


train_dataset = SequenceDataset(
    train_df,
    FEATURES,
    TARGET,
    SEQUENCE_LENGTH
)

val_dataset = SequenceDataset(
    val_df,
    FEATURES,
    TARGET,
    SEQUENCE_LENGTH
)

test_dataset = SequenceDataset(
    test_df,
    FEATURES,
    TARGET,
    SEQUENCE_LENGTH
)

print("\nSequence datasets:")
print("Train sequences:", len(train_dataset))
print("Validation sequences:", len(val_dataset))
print("Test sequences:", len(test_dataset))

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# ============================================================
# TCN BLOCK
# ============================================================

class Chomp1d(nn.Module):

    def __init__(self, chomp_size):

        super().__init__()

        self.chomp_size = chomp_size

    def forward(self, x):

        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout
    ):

        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.chomp1 = Chomp1d(padding)

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.chomp2 = Chomp1d(padding)

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )
            if in_channels != out_channels
            else None
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = (
            x
            if self.downsample is None
            else self.downsample(x)
        )

        return self.relu(out + residual)


# ============================================================
# TCN MODEL
# ============================================================

class TCN(nn.Module):

    def __init__(
        self,
        input_size,
        channels=(64, 64, 128),
        kernel_size=3,
        dropout=0.2
    ):

        super().__init__()

        layers = []

        for i, out_channels in enumerate(channels):

            in_channels = (
                input_size
                if i == 0
                else channels[i - 1]
            )

            dilation = 2 ** i

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        self.tcn = nn.Sequential(*layers)

        self.classifier = nn.Linear(
            channels[-1],
            1
        )

    def forward(self, x):

        # Input:
        # batch × sequence × features

        # Conv1d requires:
        # batch × features × sequence

        x = x.transpose(1, 2)

        x = self.tcn(x)

        # Last timestep
        x = x[:, :, -1]

        x = self.classifier(x)

        return x.squeeze(1)


model = TCN(
    input_size=len(FEATURES)
).to(DEVICE)

print("\nModel:")
print(model)

# ============================================================
# CLASS WEIGHT
# ============================================================

train_target = train_df[TARGET].values

n_negative = np.sum(train_target == 0)
n_positive = np.sum(train_target == 1)

pos_weight = n_negative / n_positive

print("\nClass distribution:")
print("Negative:", n_negative)
print("Positive:", n_positive)

print("Positive class weight:", pos_weight)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=DEVICE
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_targets = []
    all_probs = []

    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)

            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            n_samples += len(y)

            probs = torch.sigmoid(logits)

            all_targets.extend(
                y.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    targets = np.array(all_targets)
    probs = np.array(all_probs)

    predictions = (probs >= 0.5).astype(int)

    metrics = {

        "loss":
            total_loss / n_samples,

        "accuracy":
            accuracy_score(targets, predictions),

        "precision":
            precision_score(
                targets,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                targets,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                targets,
                predictions,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                targets,
                probs
            ),

        "pr_auc":
            average_precision_score(
                targets,
                probs
            )
    }

    return metrics, targets, probs


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_val_f1 = -np.inf

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    n_samples = 0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item() * len(y)
        n_samples += len(y)

    train_loss = running_loss / n_samples

    val_metrics, _, _ = evaluate(
        model,
        val_loader
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f} | "
        f"Val Precision: {val_metrics['precision']:.4f} | "
        f"Val PR-AUC: {val_metrics['pr_auc']:.4f}"
    )

    # Save best model based on F1
    if val_metrics["f1"] > best_val_f1:

        best_val_f1 = val_metrics["f1"]

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "features": FEATURES,
                "sequence_length": SEQUENCE_LENGTH,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "train_medians": train_medians.to_dict()
            },
            MODEL_FILE
        )

        print("  -> Best model saved.")

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

test_metrics, y_test, test_probs = evaluate(
    model,
    test_loader
)

for name, value in test_metrics.items():

    print(
        f"{name:12s}: {value:.6f}"
    )

test_predictions = (
    test_probs >= 0.5
).astype(int)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    test_predictions
)

print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "No Precipitation",
            "Precipitation"
        ],
        digits=4,
        zero_division=0
    )
)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

print("\nSaving test predictions...")

results = pd.DataFrame(
    {
        "actual": y_test.astype(int),
        "probability": test_probs,
        "prediction": test_predictions
    }
)

results.to_csv(
    "precipitation_tcn_test_predictions.csv",
    index=False
)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print("\nModel saved:")
print(MODEL_FILE)

print("\nTest predictions saved:")
print("precipitation_tcn_test_predictions.csv")
