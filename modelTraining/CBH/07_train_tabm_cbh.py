
import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tabm import TabM
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import joblib


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = Path("finalIntegratedDataset.csv")
OUTPUT_DIR = Path("tabm_cbh_output")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "CBH_km"

# Read the CSV in chunks to control RAM consumption.
CHUNK_SIZE = 250_000

# Maximum rows sampled per chunk.
# With approximately 64 million rows, this gives roughly
# 1.28 million candidate rows before filtering.
SAMPLE_PER_CHUNK = 1_000

BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3
SEED = 42

NUM_FEATURES = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
    "Cloud_Mask_Binary",
    "CTT",
    "CTH",
    "Cloud_Optical_Thickness",
]

CAT_FEATURES = [
    "Cloud_Mask_Raw",
    "Cloud_Phase",
]

REQUIRED_COLUMNS = [
    "timestamp",
    *NUM_FEATURES,
    *CAT_FEATURES,
    TARGET,
]


# ============================================================
# 2. REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

print("Device:", DEVICE)


# ============================================================
# 3. LOAD AND SAMPLE DATA
# ============================================================

print("\nLoading dataset in chunks...")

if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)

sampled_chunks = []

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_FILE,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE
    ),
    start=1
):
    # Remove rows without a CBH target.
    chunk[TARGET] = pd.to_numeric(chunk[TARGET], errors="coerce")
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.dropna(subset=[TARGET]).copy()

    if chunk.empty:
        continue

    # Parse MODIS timestamp.
    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"],
        format="%Y%j_%H%M",
        errors="coerce"
    )

    chunk = chunk.dropna(subset=["timestamp"])

    # Convert features to numeric.
    for col in NUM_FEATURES + CAT_FEATURES:
        chunk[col] = pd.to_numeric(
            chunk[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    # Sample rows from this chunk.
    if len(chunk) > SAMPLE_PER_CHUNK:
        chunk = chunk.sample(
            n=SAMPLE_PER_CHUNK,
            random_state=SEED + chunk_number
        )

    sampled_chunks.append(chunk)

    print(
        f"Chunk {chunk_number}: sampled {len(chunk):,} rows"
    )

if not sampled_chunks:
    raise ValueError("No valid CBH observations were found.")

df = pd.concat(sampled_chunks, ignore_index=True)

del sampled_chunks

print("\nSampled dataset shape:", df.shape)
print("Valid CBH rows:", len(df))
print("Unique timestamps:", df["timestamp"].nunique())

print("\nCBH statistics:")
print(df[TARGET].describe())


# ============================================================
# 4. BASIC TARGET VALIDATION
# ============================================================

# CBH should be non-negative.
df = df[df[TARGET] >= 0].copy()

# Do not silently remove high CBH values without scientific
# justification. Inspect extreme values before deciding.

print("\nRows after target validation:", len(df))


# ============================================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

unique_times = np.sort(df["timestamp"].unique())

if len(unique_times) < 3:
    raise ValueError("Not enough unique timestamps for time split.")

n_times = len(unique_times)

train_cut = int(n_times * 0.70)
val_cut = int(n_times * 0.85)

train_times = unique_times[:train_cut]
val_times = unique_times[train_cut:val_cut]
test_times = unique_times[val_cut:]

train_df = df[df["timestamp"].isin(train_times)].copy()
val_df = df[df["timestamp"].isin(val_times)].copy()
test_df = df[df["timestamp"].isin(test_times)].copy()

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTime ranges:")
print("Train:", train_times[0], "to", train_times[-1])
print("Validation:", val_times[0], "to", val_times[-1])
print("Test:", test_times[0], "to", test_times[-1])

if min(len(train_df), len(val_df), len(test_df)) == 0:
    raise ValueError("One of the chronological splits is empty.")


# ============================================================
# 6. NUMERICAL FEATURE PREPROCESSING
# ============================================================

# Fit preprocessing ONLY on training data.
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_num = imputer.fit_transform(
    train_df[NUM_FEATURES]
)

X_val_num = imputer.transform(
    val_df[NUM_FEATURES]
)

X_test_num = imputer.transform(
    test_df[NUM_FEATURES]
)

X_train_num = scaler.fit_transform(X_train_num)
X_val_num = scaler.transform(X_val_num)
X_test_num = scaler.transform(X_test_num)


# ============================================================
# 7. CATEGORICAL FEATURE ENCODING
# ============================================================

category_maps = {}
cat_cardinalities = []

def encode_categorical(train_data, val_data, test_data, column):

    train_values = train_data[column].fillna(-1).astype(str)
    val_values = val_data[column].fillna(-1).astype(str)
    test_values = test_data[column].fillna(-1).astype(str)

    categories = sorted(train_values.unique())

    mapping = {
        category: index + 1
        for index, category in enumerate(categories)
    }

    category_maps[column] = mapping

    train_encoded = train_values.map(mapping).fillna(0).astype("int64")
    val_encoded = val_values.map(mapping).fillna(0).astype("int64")
    test_encoded = test_values.map(mapping).fillna(0).astype("int64")

    return (
        train_encoded.to_numpy(),
        val_encoded.to_numpy(),
        test_encoded.to_numpy(),
        len(categories) + 1
    )


train_cat_columns = []
val_cat_columns = []
test_cat_columns = []

for col in CAT_FEATURES:

    tr, va, te, cardinality = encode_categorical(
        train_df, val_df, test_df, col
    )

    train_cat_columns.append(tr)
    val_cat_columns.append(va)
    test_cat_columns.append(te)

    cat_cardinalities.append(cardinality)

X_train_cat = np.column_stack(train_cat_columns)
X_val_cat = np.column_stack(val_cat_columns)
X_test_cat = np.column_stack(test_cat_columns)

print("\nCategorical cardinalities:", cat_cardinalities)


# ============================================================
# 8. PREPARE TARGET
# ============================================================

y_train = train_df[TARGET].to_numpy(dtype=np.float32)
y_val = val_df[TARGET].to_numpy(dtype=np.float32)
y_test = test_df[TARGET].to_numpy(dtype=np.float32)

# Standardize target for stable neural-network training.
y_mean = float(y_train.mean())
y_std = float(y_train.std())

if y_std == 0:
    raise ValueError("CBH target has zero variance.")

y_train_scaled = (y_train - y_mean) / y_std
y_val_scaled = (y_val - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std


# ============================================================
# 9. CONVERT TO PYTORCH TENSORS
# ============================================================

def make_tensor_dataset(x_num, x_cat, y):

    return TensorDataset(
        torch.tensor(x_num, dtype=torch.float32),
        torch.tensor(x_cat, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32)
    )


train_dataset = make_tensor_dataset(
    X_train_num, X_train_cat, y_train_scaled
)

val_dataset = make_tensor_dataset(
    X_val_num, X_val_cat, y_val_scaled
)

test_dataset = make_tensor_dataset(
    X_test_num, X_test_cat, y_test_scaled
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(DEVICE.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 10. BUILD TABM REGRESSION MODEL
# ============================================================

model = TabM.make(
    n_num_features=len(NUM_FEATURES),
    cat_cardinalities=cat_cardinalities,
    d_out=1,
    k=4
).to(DEVICE)

print("\nTabM architecture:")
print(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0003
)

criterion = nn.MSELoss()


# ============================================================
# 11. TRAINING AND VALIDATION
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

best_model_path = OUTPUT_DIR / "best_tabm_cbh.pth"

for epoch in range(1, EPOCHS + 1):

    # ---------------- TRAIN ----------------

    model.train()
    train_loss = 0.0

    for x_num, x_cat, y in train_loader:

        x_num = x_num.to(DEVICE)
        x_cat = x_cat.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(x_num, x_cat).squeeze(-1)

        # Train every ensemble member independently.
        loss = criterion(
            predictions,
            y.unsqueeze(1).expand_as(predictions)
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(y)

    train_loss /= len(train_dataset)

    # ---------------- VALIDATION ----------------

    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for x_num, x_cat, y in val_loader:

            x_num = x_num.to(DEVICE)
            x_cat = x_cat.to(DEVICE)
            y = y.to(DEVICE)

            predictions = model(x_num, x_cat).squeeze(-1)

            loss = criterion(
                predictions,
                y.unsqueeze(1).expand_as(predictions)
            )

            val_loss += loss.item() * len(y)

    val_loss /= len(val_dataset)

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )

    # ---------------- EARLY STOPPING ----------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# 12. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(best_model_path, map_location=DEVICE)
)

model.eval()


# ============================================================
# 13. TEST PREDICTIONS
# ============================================================

predictions_list = []

with torch.no_grad():

    for x_num, x_cat, y in test_loader:

        x_num = x_num.to(DEVICE)
        x_cat = x_cat.to(DEVICE)

        # Average predictions from all TabM ensemble members.
        prediction = model(x_num, x_cat).mean(dim=1).squeeze(-1)

        predictions_list.append(prediction.cpu().numpy())


y_pred_scaled = np.concatenate(predictions_list)

# Convert predictions back to kilometres.
y_pred = y_pred_scaled * y_std + y_mean


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== FINAL TEST RESULTS ==========")
print("Best epoch:", best_epoch)
print(f"MAE:  {mae:.6f} km")
print(f"RMSE: {rmse:.6f} km")
print(f"R2:   {r2:.6f}")


# ============================================================
# 15. SAVE MODEL AND PREPROCESSING OBJECTS
# ============================================================

joblib.dump(
    {
        "imputer": imputer,
        "scaler": scaler,
        "category_maps": category_maps,
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "target_mean": y_mean,
        "target_std": y_std,
    },
    OUTPUT_DIR / "cbh_preprocessing.pkl"
)

config = {
    "target": TARGET,
    "num_features": NUM_FEATURES,
    "cat_features": CAT_FEATURES,
    "cat_cardinalities": cat_cardinalities,
    "best_epoch": best_epoch,
    "test_mae_km": float(mae),
    "test_rmse_km": float(rmse),
    "test_r2": float(r2),
    "device": str(DEVICE),
    "seed": SEED
}

with open(OUTPUT_DIR / "model_config.json", "w") as f:
    json.dump(config, f, indent=4)


# Save actual and predicted values.
results = pd.DataFrame({
    "Actual_CBH_km": y_test,
    "Predicted_CBH_km": y_pred,
    "Error_km": y_test - y_pred
})

results.to_csv(
    OUTPUT_DIR / "cbh_test_predictions.csv",
    index=False
)

print("\nSaved model and results in:", OUTPUT_DIR.resolve())
