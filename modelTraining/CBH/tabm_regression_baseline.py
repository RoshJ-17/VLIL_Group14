import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("tabm_regression_data")
OUT_DIR = Path("tabm_regression_baseline")
OUT_DIR.mkdir(exist_ok=True)

def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

# Load target arrays only; no model training needed.
y_train = np.load(DATA_DIR / "train_y.npy", mmap_mode="r").astype(np.float64)
y_val = np.load(DATA_DIR / "validation_y.npy", mmap_mode="r").astype(np.float64)
y_test = np.load(DATA_DIR / "test_y.npy", mmap_mode="r").astype(np.float64)

train_mean = float(np.mean(y_train))
train_median = float(np.median(y_train))

results = {
    "training_target_summary": {
        "count": int(len(y_train)),
        "mean": train_mean,
        "median": train_median,
        "min": float(np.min(y_train)),
        "max": float(np.max(y_train)),
        "quantiles_50_75_90_95_99": [
            float(x) for x in np.quantile(y_train, [0.50, 0.75, 0.90, 0.95, 0.99])
        ],
    },
    "validation_baselines": {
        "predict_training_mean": metrics(y_val, np.full(len(y_val), train_mean)),
        "predict_training_median": metrics(y_val, np.full(len(y_val), train_median)),
    },
    "test_baselines": {
        "predict_training_mean": metrics(y_test, np.full(len(y_test), train_mean)),
        "predict_training_median": metrics(y_test, np.full(len(y_test), train_median)),
    },
}

out_path = OUT_DIR / "baseline_metrics.json"
out_path.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print(f"\nSaved: {out_path}")
