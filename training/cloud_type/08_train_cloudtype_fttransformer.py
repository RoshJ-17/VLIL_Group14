import os
import json
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tab_transformer_pytorch import FTTransformer


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

TRAIN_FILE = "cloudtype_train.csv"
VAL_FILE = "cloudtype_val.csv"
TEST_FILE = "cloudtype_test.csv"

TARGET = "cloud_type_label"
TIMESTAMP = "timestamp"

BATCH_SIZE = 4096
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

PATIENCE = 5

MODEL_DIM = 64
DEPTH = 4
HEADS = 8
DIM_HEAD = 16

NUM_CLASSES = 10

BASE_FEATURES = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
    "Cloud_Mask_Binary",
    "Cloud_Mask_Raw"
]

EXTENDED_CONTINUOUS = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
    "Cloud_Mask_Binary",
    "Cloud_Mask_Raw",
    "CTH",
    "CTT"
]

CLOUD_PHASE = "Cloud_Phase"

CLASS_NAMES = [
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


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ================================================================
# DEVICE
# ================================================================

# Use GPU when CUDA is available.
# Otherwise fall back to CPU.
#
# This allows the pipeline to be tested/run now on CPU.
# Once the NVIDIA/CUDA environment is fixed, the same script
# will automatically use GPU.

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print("=" * 80)
print("CLOUD-TYPE FT-TRANSFORMER TRAINING")
print("=" * 80)

print("\nDevice:", DEVICE)

if CUDA_AVAILABLE:

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "WARNING: CUDA is NOT available."
    )

    print(
        "Training will run on CPU."
    )

print(
    "PyTorch:",
    torch.__version__
)

print(
    "CUDA:",
    torch.version.cuda
)


# ================================================================
# ARGUMENTS
# ================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["base", "extended"],
    required=True,
    help="base = 7 MODIS features, extended = MODIS + CTH + CTT + Cloud Phase"
)

args = parser.parse_args()

MODE = args.mode


if MODE == "base":

    CONTINUOUS_FEATURES = BASE_FEATURES
    CATEGORICAL_FEATURES = []

    OUTPUT_DIR = "cloudtype_fttransformer_base"

else:

    CONTINUOUS_FEATURES = EXTENDED_CONTINUOUS
    CATEGORICAL_FEATURES = [CLOUD_PHASE]

    OUTPUT_DIR = "cloudtype_fttransformer_extended"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)

print(
    "\nMode:",
    MODE
)

print(
    "\nContinuous features:"
)

for i, feature in enumerate(
    CONTINUOUS_FEATURES,
    1
):

    print(
        f"{i}. {feature}"
    )


print(
    "\nCategorical features:"
)

if CATEGORICAL_FEATURES:

    for feature in CATEGORICAL_FEATURES:

        print(
            "1.",
            feature
        )

else:

    print(
        "None"
    )


print(
    "\nTarget:",
    TARGET
)

print(
    "\nTimestamp:"
)

print(
    "Retained in CSV only."
)

print(
    "NOT used as an ML feature."
)


# ================================================================
# LOAD DATA
# ================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)


print(
    "\nLoading training dataset..."
)

train_df = pd.read_csv(
    TRAIN_FILE
)

print(
    "Training rows:",
    len(train_df)
)


print(
    "\nLoading validation dataset..."
)

val_df = pd.read_csv(
    VAL_FILE
)

print(
    "Validation rows:",
    len(val_df)
)


print(
    "\nLoading test dataset..."
)

test_df = pd.read_csv(
    TEST_FILE
)

print(
    "Test rows:",
    len(test_df)
)


# ================================================================
# FEATURE VALIDATION
# ================================================================

required_columns = (
    CONTINUOUS_FEATURES
    + CATEGORICAL_FEATURES
    + [TARGET]
)


for name, df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} dataset is missing columns: {missing}"
        )


# ================================================================
# TIMESTAMP CHECK
# ================================================================

print("\n" + "=" * 80)
print("TIMESTAMP HANDLING")
print("=" * 80)


if TIMESTAMP in train_df.columns:

    print(
        "\nTimestamp detected."
    )

    print(
        "It will NOT be supplied to FT-Transformer."
    )

else:

    print(
        "\nTimestamp column not present."
    )

    print(
        "This is okay because it is not an ML feature."
    )


# ================================================================
# MISSING VALUES
# ================================================================

print("\n" + "=" * 80)
print("MISSING VALUE CHECK")
print("=" * 80)


for name, df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print(
        "\n" + name
    )

    check_columns = (
        CONTINUOUS_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET]
    )

    print(
        df[check_columns]
        .isna()
        .sum()
        .to_string()
    )


# ================================================================
# CONTINUOUS FEATURE IMPUTATION
# ================================================================

print("\n" + "=" * 80)
print("CONTINUOUS FEATURE PREPROCESSING")
print("=" * 80)


# Median values are learned ONLY from training data.

train_medians = {}


for feature in CONTINUOUS_FEATURES:

    train_medians[feature] = (
        train_df[feature].median()
    )

    train_df[feature] = (
        train_df[feature]
        .fillna(
            train_medians[feature]
        )
    )

    val_df[feature] = (
        val_df[feature]
        .fillna(
            train_medians[feature]
        )
    )

    test_df[feature] = (
        test_df[feature]
        .fillna(
            train_medians[feature]
        )
    )


print(
    "\nTraining medians used for missing values."
)


# ================================================================
# STANDARDIZATION
# ================================================================

print("\n" + "=" * 80)
print("FEATURE STANDARDIZATION")
print("=" * 80)


scaler = StandardScaler()


train_df[CONTINUOUS_FEATURES] = (
    scaler.fit_transform(
        train_df[CONTINUOUS_FEATURES]
    )
)


val_df[CONTINUOUS_FEATURES] = (
    scaler.transform(
        val_df[CONTINUOUS_FEATURES]
    )
)


test_df[CONTINUOUS_FEATURES] = (
    scaler.transform(
        test_df[CONTINUOUS_FEATURES]
    )
)


print(
    "\nScaler fitted on TRAIN only."
)


# ================================================================
# CLOUD PHASE ENCODING
# ================================================================

phase_mapping = None


if MODE == "extended":

    print("\n" + "=" * 80)
    print("CLOUD PHASE ENCODING")
    print("=" * 80)

    # Treat Cloud_Phase as categorical.
    # Mapping is learned from TRAIN only.

    train_phases = (
        train_df[CLOUD_PHASE]
        .fillna("UNKNOWN")
        .astype(str)
    )

    unique_phases = sorted(
        train_phases.unique()
    )

    phase_mapping = {
        phase: idx
        for idx, phase in enumerate(
            unique_phases
        )
    }

    print(
        "\nCloud Phase categories:"
    )

    for phase, idx in phase_mapping.items():

        print(
            f"{idx}: {phase}"
        )


    def encode_phase(series):

        series = (
            series
            .fillna("UNKNOWN")
            .astype(str)
        )

        # Explicit UNK index for validation/test
        # categories not seen during training.

        unknown_index = len(
            phase_mapping
        )

        return (
            series
            .map(
                lambda x:
                    phase_mapping.get(
                        x,
                        unknown_index
                    )
            )
            .astype(np.int64)
        )


    train_df[CLOUD_PHASE] = (
        encode_phase(
            train_df[CLOUD_PHASE]
        )
    )


    val_df[CLOUD_PHASE] = (
        encode_phase(
            val_df[CLOUD_PHASE]
        )
    )


    test_df[CLOUD_PHASE] = (
        encode_phase(
            test_df[CLOUD_PHASE]
        )
    )


    num_phase_categories = (
        len(phase_mapping) + 1
    )


else:

    num_phase_categories = 0


# ================================================================
# TARGET ENCODING
# ================================================================

print("\n" + "=" * 80)
print("TARGET ENCODING")
print("=" * 80)


class_to_index = {
    name: i
    for i, name in enumerate(
        CLASS_NAMES
    )
}


index_to_class = {
    i: name
    for name, i in class_to_index.items()
}


for name, df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    unknown_targets = (
        set(df[TARGET].unique())
        - set(CLASS_NAMES)
    )

    if unknown_targets:

        raise ValueError(
            f"{name} contains unexpected classes: "
            f"{unknown_targets}"
        )


train_y = (
    train_df[TARGET]
    .map(class_to_index)
    .values
)


val_y = (
    val_df[TARGET]
    .map(class_to_index)
    .values
)


test_y = (
    test_df[TARGET]
    .map(class_to_index)
    .values
)


print(
    "\nClasses:"
)


for i, name in enumerate(
    CLASS_NAMES
):

    print(
        f"{i}: {name}"
    )


# ================================================================
# CLASS IMBALANCE
# ================================================================

print("\n" + "=" * 80)
print("CLASS IMBALANCE HANDLING")
print("=" * 80)


train_counts = np.bincount(
    train_y,
    minlength=NUM_CLASSES
)


print(
    "\nTraining class counts:"
)


for i, count in enumerate(
    train_counts
):

    print(
        f"{CLASS_NAMES[i]:<25}"
        f"{count:>10,}"
    )


# ---------------------------------------------------------------
# Balanced class weights
# ---------------------------------------------------------------

total_samples = (
    train_counts.sum()
)


raw_weights = (
    total_samples
    /
    (
        NUM_CLASSES
        *
        train_counts
    )
)


# Avoid excessive domination by extremely
# rare classes.

MAX_WEIGHT = 10.0


class_weights = np.minimum(
    raw_weights,
    MAX_WEIGHT
)


# Normalize mean weight to 1.

class_weights = (
    class_weights
    /
    class_weights.mean()
)


print(
    "\nClass weights:"
)


for i, weight in enumerate(
    class_weights
):

    print(
        f"{CLASS_NAMES[i]:<25}"
        f"{weight:>10.4f}"
    )


class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)


# ================================================================
# DATASET
# ================================================================

class CloudTypeDataset(Dataset):

    def __init__(
        self,
        df,
        labels,
        continuous_features,
        categorical_features
    ):

        self.continuous = torch.tensor(
            df[continuous_features].values,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        if categorical_features:

            self.categorical = torch.tensor(
                df[categorical_features].values,
                dtype=torch.long
            )

        else:

            self.categorical = None


    def __len__(self):

        return len(
            self.labels
        )


    def __getitem__(self, idx):

        continuous = (
            self.continuous[idx]
        )

        label = (
            self.labels[idx]
        )


        if self.categorical is not None:

            categorical = (
                self.categorical[idx]
            )

            return (
                categorical,
                continuous,
                label
            )

        else:

            return (
                torch.empty(
                    0,
                    dtype=torch.long
                ),
                continuous,
                label
            )


train_dataset = CloudTypeDataset(
    train_df,
    train_y,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES
)


val_dataset = CloudTypeDataset(
    val_df,
    val_y,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES
)


test_dataset = CloudTypeDataset(
    test_df,
    test_y,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES
)


# ================================================================
# DATALOADERS
# ================================================================

print("\n" + "=" * 80)
print("BUILDING DATALOADERS")
print("=" * 80)


# Pinned memory is useful when transferring tensors
# to CUDA, but unnecessary during CPU training.

PIN_MEMORY = CUDA_AVAILABLE


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=PIN_MEMORY
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=PIN_MEMORY
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=PIN_MEMORY
)


print(
    "\nBatch size:",
    BATCH_SIZE
)


print(
    "Training batches:",
    len(train_loader)
)


print(
    "Validation batches:",
    len(val_loader)
)


print(
    "Test batches:",
    len(test_loader)
)


print(
    "Pin memory:",
    PIN_MEMORY
)


# ================================================================
# BUILD FT-TRANSFORMER
# ================================================================

print("\n" + "=" * 80)
print("BUILDING FT-TRANSFORMER")
print("=" * 80)


if MODE == "base":

    categories = ()

else:

    categories = (
        num_phase_categories,
    )


model = FTTransformer(
    categories=categories,
    num_continuous=len(
        CONTINUOUS_FEATURES
    ),
    dim=MODEL_DIM,
    depth=DEPTH,
    heads=HEADS,
    dim_head=DIM_HEAD,
    dim_out=NUM_CLASSES,
    attn_dropout=0.1,
    ff_dropout=0.1
)


model = model.to(
    DEVICE
)


print(
    "\nModel created successfully."
)


print(
    "Continuous features:",
    len(CONTINUOUS_FEATURES)
)


print(
    "Categorical features:",
    len(CATEGORICAL_FEATURES)
)


print(
    "Output classes:",
    NUM_CLASSES
)


# ================================================================
# LOSS / OPTIMIZER
# ================================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ================================================================
# TRAINING FUNCTION
# ================================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_samples = 0


    for categorical, continuous, labels in train_loader:

        categorical = categorical.to(
            DEVICE,
            non_blocking=CUDA_AVAILABLE
        )

        continuous = continuous.to(
            DEVICE,
            non_blocking=CUDA_AVAILABLE
        )

        labels = labels.to(
            DEVICE,
            non_blocking=CUDA_AVAILABLE
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            categorical,
            continuous
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        batch_size = labels.size(0)


        total_loss += (
            loss.item()
            *
            batch_size
        )


        total_samples += (
            batch_size
        )


    return (
        total_loss
        /
        total_samples
    )


# ================================================================
# EVALUATION FUNCTION
# ================================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_labels = []


    with torch.no_grad():

        for categorical, continuous, labels in loader:

            categorical = categorical.to(
                DEVICE,
                non_blocking=CUDA_AVAILABLE
            )

            continuous = continuous.to(
                DEVICE,
                non_blocking=CUDA_AVAILABLE
            )

            labels = labels.to(
                DEVICE,
                non_blocking=CUDA_AVAILABLE
            )


            logits = model(
                categorical,
                continuous
            )


            loss = criterion(
                logits,
                labels
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            batch_size = labels.size(0)


            total_loss += (
                loss.item()
                *
                batch_size
            )


            total_samples += (
                batch_size
            )


            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )


            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )


    avg_loss = (
        total_loss
        /
        total_samples
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            all_predictions
        )
    )


    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    return (
        avg_loss,
        accuracy,
        balanced_accuracy,
        macro_f1,
        weighted_f1,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ================================================================
# TRAINING LOOP
# ================================================================

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)


best_macro_f1 = -1.0
epochs_without_improvement = 0

history = []


best_model_path = os.path.join(
    OUTPUT_DIR,
    "best_model.pt"
)


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss = train_one_epoch()


    (
        val_loss,
        val_accuracy,
        val_balanced_accuracy,
        val_macro_f1,
        val_weighted_f1,
        _,
        _
    ) = evaluate(
        val_loader
    )


    scheduler.step(
        val_macro_f1
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )


    print(
        f"Train Loss:          "
        f"{train_loss:.5f}"
    )


    print(
        f"Val Loss:            "
        f"{val_loss:.5f}"
    )


    print(
        f"Val Accuracy:        "
        f"{val_accuracy:.5f}"
    )


    print(
        f"Val Balanced Acc:    "
        f"{val_balanced_accuracy:.5f}"
    )


    print(
        f"Val Macro-F1:        "
        f"{val_macro_f1:.5f}"
    )


    print(
        f"Val Weighted-F1:     "
        f"{val_weighted_f1:.5f}"
    )


    print(
        f"Learning Rate:       "
        f"{current_lr:.7f}"
    )


    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "val_balanced_accuracy":
            val_balanced_accuracy,

        "val_macro_f1":
            val_macro_f1,

        "val_weighted_f1":
            val_weighted_f1,

        "learning_rate":
            current_lr

    })


    if val_macro_f1 > best_macro_f1:

        best_macro_f1 = (
            val_macro_f1
        )

        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "continuous_features":
                    CONTINUOUS_FEATURES,

                "categorical_features":
                    CATEGORICAL_FEATURES,

                "class_weights":
                    class_weights.tolist(),

                "mode":
                    MODE
            },
            best_model_path
        )


        print(
            "\n*** BEST MODEL SAVED ***"
        )


    else:

        epochs_without_improvement += 1


        print(
            f"\nNo improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping."
        )

        break


# ================================================================
# SAVE TRAINING HISTORY
# ================================================================

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)


pd.DataFrame(
    history
).to_csv(
    history_path,
    index=False
)


# ================================================================
# LOAD BEST MODEL
# ================================================================

print("\n" + "=" * 80)
print("LOADING BEST MODEL")
print("=" * 80)


checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    "\nBest validation Macro-F1:",
    f"{best_macro_f1:.5f}"
)


# ================================================================
# FINAL TEST EVALUATION
# ================================================================

print("\n" + "=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)


(
    test_loss,
    test_accuracy,
    test_balanced_accuracy,
    test_macro_f1,
    test_weighted_f1,
    test_labels,
    test_predictions
) = evaluate(
    test_loader
)


print(
    f"\nTest Loss:             "
    f"{test_loss:.5f}"
)


print(
    f"Test Accuracy:         "
    f"{test_accuracy:.5f}"
)


print(
    f"Test Balanced Acc:     "
    f"{test_balanced_accuracy:.5f}"
)


print(
    f"Test Macro-F1:         "
    f"{test_macro_f1:.5f}"
)


print(
    f"Test Weighted-F1:      "
    f"{test_weighted_f1:.5f}"
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)


report = classification_report(
    test_labels,
    test_predictions,
    labels=list(
        range(NUM_CLASSES)
    ),
    target_names=CLASS_NAMES,
    digits=5,
    zero_division=0
)


print(
    "\n"
)

print(
    report
)


report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)


with open(
    report_path,
    "w"
) as f:

    f.write(
        report
    )

    f.write(
        "\n\n"
    )

    f.write(
        f"Test Loss: "
        f"{test_loss:.5f}\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy:.5f}\n"
    )

    f.write(
        f"Test Balanced Accuracy: "
        f"{test_balanced_accuracy:.5f}\n"
    )

    f.write(
        f"Test Macro-F1: "
        f"{test_macro_f1:.5f}\n"
    )

    f.write(
        f"Test Weighted-F1: "
        f"{test_weighted_f1:.5f}\n"
    )


# ================================================================
# CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=list(
        range(NUM_CLASSES)
    )
)


cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)


pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
).to_csv(
    cm_path
)


# ================================================================
# SAVE CONFIGURATION
# ================================================================

config = {

    "mode":
        MODE,

    "train_file":
        TRAIN_FILE,

    "validation_file":
        VAL_FILE,

    "test_file":
        TEST_FILE,

    "continuous_features":
        CONTINUOUS_FEATURES,

    "categorical_features":
        CATEGORICAL_FEATURES,

    "target":
        TARGET,

    "timestamp_used_as_feature":
        False,

    "num_classes":
        NUM_CLASSES,

    "class_names":
        CLASS_NAMES,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "model_dim":
        MODEL_DIM,

    "depth":
        DEPTH,

    "heads":
        HEADS,

    "dim_head":
        DIM_HEAD,

    "best_validation_macro_f1":
        best_macro_f1,

    "test_accuracy":
        test_accuracy,

    "test_balanced_accuracy":
        test_balanced_accuracy,

    "test_macro_f1":
        test_macro_f1,

    "test_weighted_f1":
        test_weighted_f1,

    "device":
        str(DEVICE),

    "cuda_available":
        CUDA_AVAILABLE

}


config_path = os.path.join(
    OUTPUT_DIR,
    "config.json"
)


with open(
    config_path,
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 80)
print("CLOUD-TYPE FT-TRANSFORMER TRAINING COMPLETE")
print("=" * 80)


print(
    "\nModel:",
    MODE
)


print(
    "\nDevice:",
    DEVICE
)


print(
    "\nBest model:"
)

print(
    " ",
    best_model_path
)


print(
    "\nTraining history:"
)

print(
    " ",
    history_path
)


print(
    "\nClassification report:"
)

print(
    " ",
    report_path
)


print(
    "\nConfusion matrix:"
)

print(
    " ",
    cm_path
)


print(
    "\nTest Macro-F1:"
)

print(
    f" {test_macro_f1:.5f}"
)


print(
    "\nTest Balanced Accuracy:"
)

print(
    f" {test_balanced_accuracy:.5f}"
)


print(
    "\nTimestamp was NOT used as an ML feature."
)


print(
    "\n" + "=" * 80
)
