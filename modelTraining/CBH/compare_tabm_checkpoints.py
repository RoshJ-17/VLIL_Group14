import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import numpy as np
import pandas as pd
import torch
from tabm import TabM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

CSV = "modisPixelGPM_training_clean.csv"
CHUNK_SIZE = 500_000
SAMPLE_SIZE = 200_000

FEATURES = [
    "lat", "lon", "BT_11", "BT_12", "BTD",
    "Cloud_Mask_Binary", "Cloud_Mask_Raw",
    "CTT", "CTH", "Cloud_Phase",
]
TARGET = "GPM_precipitation_rate"
USECOLS = ["timestamp"] + FEATURES + [TARGET]

RUNS = [
    ("baseline_w1", "tabm_cpu_run"),
    ("weighted_w3", "tabm_cpu_weighted_run"),
    ("weighted_w2", "tabm_cpu_weighted2_run"),
]

torch.set_num_threads(min(4, os.cpu_count() or 1))
device = torch.device("cpu")

# Reconstruct the deterministic sample used by the training scripts.
eligible_count = 0
for chunk in pd.read_csv(CSV, usecols=USECOLS, chunksize=CHUNK_SIZE):
    target = chunk[TARGET]
    eligible_count += int((target.notna() & np.isfinite(target)).sum())

if eligible_count == 0:
    raise RuntimeError("No valid target rows found.")

step = max(1, eligible_count // SAMPLE_SIZE)
sample_parts = []
eligible_seen = 0

for chunk in pd.read_csv(CSV, usecols=USECOLS, chunksize=CHUNK_SIZE):
    valid = chunk[TARGET].notna() & np.isfinite(chunk[TARGET])
    chunk = chunk.loc[valid]
    if chunk.empty:
        continue

    positions = np.arange(len(chunk), dtype=np.int64) + eligible_seen
    chosen = chunk.loc[(positions % step) == 0]
    if not chosen.empty:
        sample_parts.append(chosen)

    eligible_seen += len(chunk)

df = pd.concat(sample_parts, ignore_index=True)
if len(df) > SAMPLE_SIZE:
    df = df.iloc[:SAMPLE_SIZE].copy()

df = df.dropna(subset=["timestamp"])
df["datetime"] = pd.to_datetime(
    df["timestamp"].astype(str),
    format="%Y%j_%H%M",
    errors="coerce",
)
df = df.dropna(subset=["datetime"])
df["target"] = (df[TARGET] >= 0.1).astype("int64")
df = df.sort_values("datetime").reset_index(drop=True)

unique_times = np.sort(df["datetime"].unique())
train_time_end = unique_times[int(len(unique_times) * 0.70)]
val_time_end = unique_times[int(len(unique_times) * 0.85)]

train = df[df["datetime"] < train_time_end].copy()
val = df[
    (df["datetime"] >= train_time_end)
    & (df["datetime"] < val_time_end)
].copy()
test = df[df["datetime"] >= val_time_end].copy()

# Recreate preprocessing using training data only.
medians = train[FEATURES].median().fillna(0.0)

def prepare(part):
    x = part[FEATURES].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(medians).fillna(0.0)
    return x.to_numpy(dtype=np.float32), part["target"].to_numpy(dtype=np.int64)

X_train, _ = prepare(train)
X_val, _ = prepare(val)
X_test, y_test = prepare(test)

means = X_train.mean(axis=0)
stds = X_train.std(axis=0)
stds[stds < 1e-6] = 1.0

def normalize(x):
    return np.clip((x - means) / stds, -10, 10).astype(np.float32)

X_test = normalize(X_test)

print("Reconstructed split:")
print(f"  train={len(train):,}, validation={len(val):,}, test={len(test):,}")
print(f"  test dates={test['datetime'].min()} to {test['datetime'].max()}")
print(f"  test rain fraction={y_test.mean():.6f}")

results = []

for run_name, out_dir in RUNS:
    metrics_path = os.path.join(out_dir, "metrics.json")
    checkpoint_path = os.path.join(out_dir, "tabm_cpu.pt")

    with open(metrics_path) as f:
        saved_metrics = json.load(f)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Refuse to evaluate if checkpoint preprocessing does not match.
    if ckpt["features"] != FEATURES:
        raise RuntimeError(f"{run_name}: feature order mismatch.")

    if not np.allclose(
        np.asarray(ckpt["means"], dtype=float), means, rtol=1e-5, atol=1e-6
    ):
        raise RuntimeError(f"{run_name}: training means mismatch.")

    if not np.allclose(
        np.asarray(ckpt["stds"], dtype=float), stds, rtol=1e-5, atol=1e-6
    ):
        raise RuntimeError(f"{run_name}: training standard deviations mismatch.")

    for feature in FEATURES:
        if not np.isclose(
            float(ckpt["medians"][feature]),
            float(medians[feature]),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(f"{run_name}: median mismatch for {feature}.")

    # Verify that the reconstructed test split matches saved metadata.
    if len(test) != saved_metrics["test_rows"]:
        raise RuntimeError(f"{run_name}: test row count mismatch.")

    if str(test["datetime"].min()) != saved_metrics["test_start"]:
        raise RuntimeError(f"{run_name}: test start timestamp mismatch.")

    if str(test["datetime"].max()) != saved_metrics["test_end"]:
        raise RuntimeError(f"{run_name}: test end timestamp mismatch.")

    model = TabM.make(
        n_num_features=len(FEATURES),
        cat_cardinalities=None,
        d_out=1,
        k=4,
        d_block=64,
        n_blocks=2,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])

    model.eval()
    chunks = []

    with torch.no_grad():
        for start_idx in range(0, len(X_test), 2048):
            xb = torch.from_numpy(
                X_test[start_idx:start_idx + 2048]
            ).to(device)

            logits = model(xb).mean(dim=1).reshape(-1)
            chunks.append(logits.cpu())

    test_logits = torch.cat(chunks).numpy()

    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(test_logits, -30, 30))
    )

    threshold = float(ckpt["probability_threshold"])
    predictions = (probabilities >= threshold).astype(np.int64)
    row = {
        "model": run_name,
        "threshold": threshold,
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    results.append(row)
    print("\n" + json.dumps(row, indent=2))

output_path = "tabm_checkpoint_comparison.json"
with open(output_path, "w") as f:
    json.dump(
        {
            "test_start": str(test["datetime"].min()),
            "test_end": str(test["datetime"].max()),
            "test_rows": int(len(test)),
            "models": results,
        },
        f,
        indent=2,
    )

print(f"\nSaved comparison to {output_path}")
