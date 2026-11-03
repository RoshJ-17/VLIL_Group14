#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabm import TabM


def load_arrays(data_dir):
    data_dir = Path(data_dir)
    print("Loading disk-backed arrays...", flush=True)

    arrays = {}
    for split in ("train", "validation", "test"):
        arrays[f"{split}_X"] = np.load(
            data_dir / f"{split}_X.npy", mmap_mode="r"
        )
        arrays[f"{split}_y"] = np.load(
            data_dir / f"{split}_y.npy", mmap_mode="r"
        )

    for split in ("train", "validation", "test"):
        print(
            f"{split.capitalize()} rows: "
            f"{len(arrays[f'{split}_y']):,}",
            flush=True,
        )

    return arrays


def predict_log(model, X, batch_size, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            xb = torch.from_numpy(
                np.asarray(X[start:end], dtype=np.float32).copy()
            ).to(device)

            output = model(xb)
            # TabM output is [batch, ensemble members, output dimension].
            pred_log = output.mean(dim=1).squeeze(-1)
            predictions.append(pred_log.cpu().numpy())

    return np.concatenate(predictions)


def evaluate(model, X, y, batch_size, device):
    pred_log = predict_log(model, X, batch_size, device)

    # Targets were stored in original units.
    y_true = np.asarray(y, dtype=np.float32)
    y_pred = np.expm1(pred_log)
    y_pred = np.maximum(y_pred, 0.0)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="tabm_regression_data")
    parser.add_argument("--out", default="tabm_regression_run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    print(f"Device: {device}", flush=True)

    arrays = load_arrays(args.data_dir)
    X_train, y_train = arrays["train_X"], arrays["train_y"]
    X_val, y_val = arrays["validation_X"], arrays["validation_y"]
    X_test, y_test = arrays["test_X"], arrays["test_y"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TabM.make(
        n_num_features=X_train.shape[1],
        cat_cardinalities=None,
        d_out=1,
        k=4,
        d_block=64,
        n_blocks=2,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        order = np.random.permutation(len(y_train))
        total_loss = 0.0
        seen = 0

        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]

            xb = torch.from_numpy(
                np.asarray(X_train[idx], dtype=np.float32).copy()
            ).to(device)

            # Train in log1p space; all retained targets are positive.
            yb = torch.from_numpy(
                np.log1p(np.asarray(y_train[idx], dtype=np.float32))
            ).to(device)

            optimizer.zero_grad()
            output = model(xb).squeeze(-1)  # [batch, k]
            target = yb.unsqueeze(1).expand_as(output)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            seen += len(idx)

        train_loss = total_loss / seen
        val_metrics = evaluate(
            model, X_val, y_val, args.batch_size, device
        )

        record = {
            "epoch": epoch,
            "train_log_mse": train_loss,
            "validation_mae": val_metrics["mae"],
            "validation_rmse": val_metrics["rmse"],
            "validation_r2": val_metrics["r2"],
            "epoch_seconds": time.time() - start_time,
        }
        history.append(record)

        print(json.dumps(record), flush=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_rmse": best_val_rmse,
            "history": history,
            "args": vars(args),
        }

        torch.save(checkpoint, out_dir / "last_checkpoint.pt")

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            checkpoint["best_validation_rmse"] = best_val_rmse
            torch.save(checkpoint, out_dir / "best_checkpoint.pt")
            print(
                f"Saved best checkpoint at epoch {epoch} "
                f"(validation RMSE={best_val_rmse:.6f})",
                flush=True,
            )

    # Restore the best validation checkpoint before test evaluation.
    best = torch.load(out_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])

    test_metrics = evaluate(
        model, X_test, y_test, args.batch_size, device
    )

    results = {
        "task": "positive_rain_rate_regression",
        "target": "GPM_precipitation_rate",
        "target_transform": "log1p for training; expm1 for predictions",
        "best_epoch": best_epoch,
        "best_validation_rmse": best_val_rmse,
        "test_metrics": test_metrics,
        "history": history,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nFINAL TEST METRICS", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print(f"Training complete: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
