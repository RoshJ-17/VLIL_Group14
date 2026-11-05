import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabm import TabM

DATA_DIR = Path("tabm_regression_data")
OUT_DIR = Path("tabm_regression_huber_run")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 2048
EPOCHS = 10
LEARNING_RATE = 1e-3
HUBER_DELTA = 1.0

torch.set_num_threads(4)
torch.manual_seed(42)
np.random.seed(42)

def load_array(name):
    return np.load(DATA_DIR / f"{name}.npy", mmap_mode="r")

X_train = load_array("train_X")
y_train = load_array("train_y")
X_val = load_array("validation_X")
y_val = np.asarray(load_array("validation_y"), dtype=np.float32)
X_test = load_array("test_X")
y_test = np.asarray(load_array("test_y"), dtype=np.float32)

model = TabM.make(
    n_num_features=X_train.shape[1],
    d_out=1,
    k=4,
    d_block=64,
    n_blocks=2,
    dropout=0.1,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.HuberLoss(delta=HUBER_DELTA)

def predict(X):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), 8192):
            xb = torch.from_numpy(
                np.array(X[start:start + 8192], dtype=np.float32)
            ).to(DEVICE)

            output = model(xb)
            if output.ndim == 3:
                output = output.mean(dim=1)

            predictions.append(output.squeeze(-1).cpu().numpy())

    return np.concatenate(predictions)

def calculate_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

best_val_rmse = float("inf")
best_epoch = None
history = []

n_train = len(y_train)

for epoch in range(1, EPOCHS + 1):
    model.train()
    permutation = np.random.permutation(n_train)
    batch_losses = []

    for start in range(0, n_train, BATCH_SIZE):
        indices = permutation[start:start + BATCH_SIZE]

        xb = torch.from_numpy(
            np.array(X_train[indices], dtype=np.float32)
        ).to(DEVICE)
        yb = torch.from_numpy(
            np.array(y_train[indices], dtype=np.float32)
        ).to(DEVICE)

        optimizer.zero_grad()
        output = model(xb)

        # Average the TabM ensemble members to obtain one rate per row.
        if output.ndim == 3:
            output = output.mean(dim=1)
        prediction = output.squeeze(-1)

        loss = loss_fn(prediction, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(float(loss.item()))

    val_pred = predict(X_val)
    val_metrics = calculate_metrics(y_val, val_pred)
    train_loss = float(np.mean(batch_losses))

    row = {
        "epoch": epoch,
        "train_huber_loss": train_loss,
        "validation": val_metrics,
    }
    history.append(row)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train Huber={train_loss:.6f} | "
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
                "validation_metrics": val_metrics,
                "config": {
                    "loss": "HuberLoss",
                    "delta": HUBER_DELTA,
                    "target_transform": "none",
                    "k": 4,
                    "d_block": 64,
                    "n_blocks": 2,
                    "dropout": 0.1,
                },
            },
            OUT_DIR / "best_checkpoint.pt",
        )

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
        },
        OUT_DIR / "last_checkpoint.pt",
    )

# Evaluate the selected checkpoint once on the held-out test set.
checkpoint = torch.load(
    OUT_DIR / "best_checkpoint.pt",
    map_location=DEVICE,
)
model.load_state_dict(checkpoint["model"])

test_pred = predict(X_test)
test_metrics = calculate_metrics(y_test, test_pred)

results = {
    "best_epoch": best_epoch,
    "best_validation_rmse": best_val_rmse,
    "test_metrics": test_metrics,
    "history": history,
}

(OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
print("\nFinal test metrics:")
print(json.dumps(results, indent=2), flush=True)

