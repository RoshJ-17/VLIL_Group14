import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabm import TabM

DATA_DIR = Path("tabm_regression_data")
OUT_DIR = Path("tabm_regression_weighted_run")
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 2048
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_LEVELS = [1.0, 1.5, 2.0, 3.0, 4.0]

torch.set_num_threads(4)
torch.manual_seed(42)
np.random.seed(42)

def load_array(name):
    return np.load(DATA_DIR / f"{name}.npy", mmap_mode="r")

X_train = load_array("train_X")
y_train = np.asarray(load_array("train_y"), dtype=np.float32)
X_val = load_array("validation_X")
y_val = np.asarray(load_array("validation_y"), dtype=np.float32)
X_test = load_array("test_X")
y_test = np.asarray(load_array("test_y"), dtype=np.float32)

# Calculate cutoffs using training targets only.
q = np.quantile(y_train, [0.50, 0.75, 0.90, 0.95, 0.99]).astype(np.float32)
print("Training-target quantile cutoffs:", q.tolist(), flush=True)

def get_weights(y):
    # Bins: <=50th, 50-75th, 75-90th, 90-95th, 95-99th, >99th percentile.
    bins = np.digitize(y, q, right=True)
    weights = np.asarray(WEIGHT_LEVELS, dtype=np.float32)[
        np.minimum(bins, len(WEIGHT_LEVELS) - 1)
    ]
    # Normalize to mean 1 so the overall loss scale stays comparable.
    return weights / weights.mean()

train_weights = get_weights(y_train)

model = TabM.make(
    n_num_features=X_train.shape[1],
    d_out=1,
    k=4,
    d_block=64,
    n_blocks=2,
    dropout=0.1,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def predict(X):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), 8192):
            xb = torch.from_numpy(
                np.array(X[start:start + 8192], dtype=np.float32)
            )
            output = model(xb)
            if output.ndim == 3:
                output = output.mean(dim=1)
            chunks.append(output.squeeze(-1).numpy())
    pred_log = np.concatenate(chunks)
    return np.maximum(np.expm1(pred_log), 0.0)

def metrics(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
    }

best_val_rmse = float("inf")
best_epoch = None
history = []
n_train = len(y_train)

for epoch in range(1, EPOCHS + 1):
    model.train()
    order = np.random.permutation(n_train)
    losses = []

    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]

        xb = torch.from_numpy(np.array(X_train[idx], dtype=np.float32))
        yb = torch.from_numpy(y_train[idx])
        wb = torch.from_numpy(train_weights[idx])

        optimizer.zero_grad()
        output = model(xb)
        if output.ndim == 3:
            output = output.mean(dim=1)
        pred_log = output.squeeze(-1)

        target_log = torch.log1p(yb)
        per_row_loss = (pred_log - target_log) ** 2
        loss = (per_row_loss * wb).mean()

        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    val_pred = predict(X_val)
    val_metrics = metrics(y_val, val_pred)
    record = {
        "epoch": epoch,
        "train_weighted_mse": float(np.mean(losses)),
        "validation": val_metrics,
    }
    history.append(record)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"weighted train loss={record['train_weighted_mse']:.6f} | "
        f"val MAE={val_metrics['MAE']:.6f} | "
        f"val RMSE={val_metrics['RMSE']:.6f} | "
        f"val R2={val_metrics['R2']:.6f}",
        flush=True,
    )

    if val_metrics["RMSE"] < best_val_rmse:
        best_val_rmse = val_metrics["RMSE"]
        best_epoch = epoch
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "quantile_cutoffs": q.tolist(),
                "weight_levels": WEIGHT_LEVELS,
                "validation_metrics": val_metrics,
            },
            OUT_DIR / "best_checkpoint.pt",
        )

    torch.save(
        {"model": model.state_dict(), "epoch": epoch},
        OUT_DIR / "last_checkpoint.pt",
    )

checkpoint = torch.load(OUT_DIR / "best_checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])
test_metrics = metrics(y_test, predict(X_test))

results = {
    "best_epoch": best_epoch,
    "best_validation_rmse": best_val_rmse,
    "test_metrics": test_metrics,
    "training_quantile_cutoffs": q.tolist(),
    "weight_levels": WEIGHT_LEVELS,
    "history": history,
}
(OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
print("\nFinal test metrics:")
print(json.dumps(results, indent=2), flush=True)
