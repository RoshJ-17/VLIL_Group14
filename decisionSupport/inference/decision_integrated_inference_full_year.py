#!/usr/bin/env python3
"""
Full-year integrated inference for Aviation Weather Intelligence.

Uses the same checkpoints and fitted preprocessing artifacts as
decision_integrated_inference.py, but processes finalIntegratedDataset.csv.

Features:
- Streams the source CSV in chunks.
- Runs all four selected TabM models.
- Runs rainfall regression only for rows classified as rain.
- Writes one output CSV.
- Maintains a checkpoint manifest and can resume after interruption.
- Preserves all original source columns and appends prediction columns.

Run:
    python decision_integrated_inference_full_year.py

Optional:
    python decision_integrated_inference_full_year.py \
        --input /path/to/finalIntegratedDataset.csv \
        --chunk-size 8192 \
        --checkpoint-every 10 \
        --overwrite
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tabm import TabM


ROOT = Path("/home/m_prathilothamai/aviation_weather_project")

OUTPUT_DIR = ROOT / "data/processed/full_year_2022_2023"
OUTPUT_FILE = OUTPUT_DIR / "full_year_2022_2023_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "full_year_2022_2023_prediction_summary.json"
MANIFEST_FILE = OUTPUT_DIR / "full_year_inference_manifest.json"

CLOUD_CKPT = (
    ROOT / "eda_results/cloudType/training/sampling_10epoch_comparison/"
    "tabm_sqrt_weighted_best_checkpoint.pt"
)
CLOUD_SCALER = (
    ROOT / "eda_results/cloudType/preprocessing/scaled_splits/"
    "standard_scaler.joblib"
)
CBH_CKPT = ROOT / "tabm_cbh_output/best_tabm_cbh.pth"
CBH_PREPROCESS = ROOT / "tabm_cbh_output/cbh_preprocessing.pkl"
PRECIP_CKPT = ROOT / "tabm_full_w1_run/best_checkpoint.pt"
PRECIP_META = ROOT / "tabm_full_data/metadata.json"
RAIN_CKPT = ROOT / "tabm_regression_weighted_run/best_checkpoint.pt"
RAIN_META = ROOT / "tabm_regression_data/metadata.json"

CLOUD_FEATURES = ["lat", "lon", "BT_11", "BT_12", "BTD"]
PRECIP_FEATURES_EXPECTED = [
    "lat", "lon", "BT_11", "BT_12", "BTD",
    "Cloud_Mask_Binary", "Cloud_Mask_Raw",
    "CTT", "CTH", "Cloud_Phase"
]
CLOUD_LABELS = {
    0: "clear",
    1: "liquid_low",
    2: "liquid_middle",
    3: "liquid_high",
    4: "ice_low",
    5: "ice_middle",
    6: "ice_high",
}
PREDICTION_COLUMNS = [
    "cloud_type_id",
    "cloud_type",
    "CBH_km",
    "rain_probability",
    "rain_detected",
    "rainfall_intensity_conditional_mm_hr",
    "final_rainfall_mm_hr",
    "bt_anomaly_flag",
]
NUMERIC_PREDICTION_COLUMNS = [
    "cloud_type_id",
    "CBH_km",
    "rain_probability",
    "rain_detected",
    "rainfall_intensity_conditional_mm_hr",
    "final_rainfall_mm_hr",
    "bt_anomaly_flag",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run resumable full-year aviation weather inference."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to finalIntegratedDataset.csv. If omitted, search project root.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="CSV rows per inference/write chunk (default: 8192).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Persist resume manifest every N chunks (default: 10).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard prior full-year output and start from the beginning.",
    )
    return parser.parse_args()


def locate_input(explicit_path):
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return path

    candidates = sorted(
        p for p in ROOT.rglob("finalIntegratedDataset.csv")
        if p.is_file() and OUTPUT_DIR not in p.parents
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "Could not find finalIntegratedDataset.csv under project root. "
            "Pass --input /full/path/finalIntegratedDataset.csv."
        )
    raise RuntimeError(
        "Multiple finalIntegratedDataset.csv files found. "
        "Pass --input explicitly:\n" + "\n".join(map(str, candidates))
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint, keys):
    if isinstance(checkpoint, dict):
        for key in keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                candidate = checkpoint[key]
                if candidate and all(
                    torch.is_tensor(value) for value in candidate.values()
                ):
                    return candidate
        if checkpoint and all(
            torch.is_tensor(value) for value in checkpoint.values()
        ):
            return checkpoint
    raise RuntimeError("Could not extract model state_dict from checkpoint.")


def verify_files(paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Required model/preprocessing files are missing:\n"
            + "\n".join(missing)
        )


def tensor(array, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


def prepare_numeric(df, features, medians):
    values = df[features].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    medians = np.asarray(medians, dtype=np.float64)

    if len(medians) != len(features):
        raise ValueError(f"Median count mismatch for {features}")

    invalid = ~np.isfinite(values)
    if invalid.any():
        values[invalid] = np.broadcast_to(
            medians, values.shape
        )[invalid]

    if not np.isfinite(values).all():
        raise ValueError("Non-finite values remain after median imputation.")
    return values


def standardize(values, means, stds):
    values = np.asarray(values, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)

    if np.any(stds == 0):
        raise ValueError("Zero standard deviation found.")

    result = np.clip((values - means) / stds, -10, 10)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite values found after standardization.")
    return result.astype(np.float32)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def input_signature(path):
    stat = path.stat()
    return {
        "input_path": str(path.resolve()),
        "input_size_bytes": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
    }


def initial_manifest(signature):
    return {
        "status": "running",
        **signature,
        "chunk_size": None,
        "chunks_completed": 0,
        "rows_processed": 0,
        "output_bytes_committed": 0,
        "cloud_class_counts": {str(i): 0 for i in range(7)},
        "rain_detected_rows": 0,
        "bt_anomaly_rows": 0,
        "probability_threshold": None,
    }


def load_or_initialize_manifest(signature, chunk_size, overwrite):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for path in [OUTPUT_FILE, SUMMARY_FILE, MANIFEST_FILE]:
            if path.exists():
                path.unlink()

    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)

        for key in ["input_path", "input_size_bytes", "input_mtime_ns"]:
            if manifest.get(key) != signature.get(key):
                raise RuntimeError(
                    f"Input {key} differs from saved run manifest. "
                    "Use --overwrite only if you intend to restart."
                )

        if manifest.get("chunk_size") not in (None, chunk_size):
            raise RuntimeError(
                "Chunk size differs from the saved run. Resume using "
                f"--chunk-size {manifest['chunk_size']} or use --overwrite."
            )

        if manifest.get("status") == "completed":
            raise RuntimeError(
                "This full-year run is already marked completed. "
                "Use --overwrite to intentionally rerun it."
            )

        manifest["chunk_size"] = chunk_size
        committed = int(manifest["output_bytes_committed"])

        if OUTPUT_FILE.exists():
            with open(OUTPUT_FILE, "r+b") as f:
                f.truncate(committed)
                f.flush()
                os.fsync(f.fileno())
        elif committed != 0:
            raise RuntimeError(
                "Manifest records committed output bytes, but output CSV "
                "does not exist. Use --overwrite to restart."
            )

        print(
            "Resuming from chunk",
            manifest["chunks_completed"] + 1,
            "| rows already committed:",
            f"{manifest['rows_processed']:,}",
        )
        return manifest

    if OUTPUT_FILE.exists():
        raise RuntimeError(
            f"{OUTPUT_FILE} exists without a manifest. "
            "Move/delete it or run with --overwrite."
        )

    manifest = initial_manifest(signature)
    manifest["chunk_size"] = chunk_size
    atomic_write_json(MANIFEST_FILE, manifest)
    return manifest


def load_models(device):
    verify_files([
        CLOUD_CKPT, CLOUD_SCALER, CBH_CKPT, CBH_PREPROCESS,
        PRECIP_CKPT, PRECIP_META, RAIN_CKPT, RAIN_META
    ])

    cloud_scaler = joblib.load(CLOUD_SCALER)
    cbh_prep = joblib.load(CBH_PREPROCESS)
    with open(PRECIP_META) as f:
        precip_meta = json.load(f)
    with open(RAIN_META) as f:
        rain_meta = json.load(f)

    precip_checkpoint = load_checkpoint(PRECIP_CKPT, device)
    rain_checkpoint = load_checkpoint(RAIN_CKPT, device)
    cloud_checkpoint = load_checkpoint(CLOUD_CKPT, device)

    cloud_features = list(cloud_scaler.feature_names_in_)
    if cloud_features != CLOUD_FEATURES:
        raise RuntimeError(f"Cloud feature mismatch: {cloud_features}")

    cbh_num_features = list(cbh_prep["num_features"])
    cbh_cat_features = list(cbh_prep["cat_features"])
    if len(cbh_num_features) != 9 or len(cbh_cat_features) != 2:
        raise RuntimeError("Unexpected CBH feature configuration.")

    precip_features = precip_meta["features"]
    rain_features = rain_meta["features"]

    if precip_features != PRECIP_FEATURES_EXPECTED:
        raise RuntimeError(f"Precipitation feature mismatch: {precip_features}")
    if rain_features != PRECIP_FEATURES_EXPECTED:
        raise RuntimeError(f"Rainfall feature mismatch: {rain_features}")

    for key in ["features", "medians", "means", "stds"]:
        if key not in precip_checkpoint:
            raise RuntimeError(f"Precipitation checkpoint missing {key}")

    if precip_checkpoint["features"] != precip_features:
        raise RuntimeError("Precipitation checkpoint feature order mismatch.")

    for key in ["medians", "means", "stds"]:
        if not np.array_equal(
            np.asarray(precip_meta[key]),
            np.asarray(precip_checkpoint[key])
        ):
            raise RuntimeError(f"Precipitation {key} mismatch.")

    if len(precip_features) != 10 or len(rain_features) != 10:
        raise RuntimeError("Unexpected precipitation/rainfall feature count.")

    cloud_model = TabM.make(
        n_num_features=5, d_out=7, k=4,
        d_block=64, n_blocks=2, dropout=0.1
    ).to(device)
    cloud_model.load_state_dict(
        extract_state_dict(
            cloud_checkpoint, ["state_dict", "model_state_dict", "model"]
        ),
        strict=True
    )
    cloud_model.eval()

    cat_cardinalities = [
        max(mapping.values()) + 1
        for mapping in cbh_prep["category_maps"].values()
    ]
    cbh_model = TabM.make(
        n_num_features=len(cbh_num_features),
        cat_cardinalities=cat_cardinalities,
        d_out=1, k=4
    ).to(device)
    cbh_model.load_state_dict(
        load_checkpoint(CBH_CKPT, device), strict=True
    )
    cbh_model.eval()

    precip_model = TabM.make(
        n_num_features=10, cat_cardinalities=None, d_out=1,
        k=4, d_block=64, n_blocks=2, dropout=0.1
    ).to(device)
    precip_model.load_state_dict(
        extract_state_dict(precip_checkpoint, ["state_dict"]),
        strict=True
    )
    precip_model.eval()
    probability_threshold = float(
        precip_checkpoint["probability_threshold"]
    )

    rain_model = TabM.make(
        n_num_features=10, d_out=1, k=4,
        d_block=64, n_blocks=2, dropout=0.1
    ).to(device)
    rain_model.load_state_dict(
        extract_state_dict(rain_checkpoint, ["model"]),
        strict=True
    )
    rain_model.eval()

    artifacts = {
        "cloud_scaler": cloud_scaler,
        "cbh_prep": cbh_prep,
        "precip_meta": precip_meta,
        "rain_meta": rain_meta,
        "precip_checkpoint": precip_checkpoint,
        "rain_checkpoint": rain_checkpoint,
        "cbh_num_features": cbh_num_features,
        "cbh_cat_features": cbh_cat_features,
        "precip_features": precip_features,
        "rain_features": rain_features,
        "probability_threshold": probability_threshold,
    }
    models = {
        "cloud": cloud_model,
        "cbh": cbh_model,
        "precip": precip_model,
        "rain": rain_model,
    }
    return models, artifacts


def predict_chunk(chunk, models, artifacts, device):
    cloud_scaler = artifacts["cloud_scaler"]
    cbh_prep = artifacts["cbh_prep"]
    cbh_num_features = artifacts["cbh_num_features"]
    cbh_cat_features = artifacts["cbh_cat_features"]
    precip_features = artifacts["precip_features"]
    rain_features = artifacts["rain_features"]
    precip_checkpoint = artifacts["precip_checkpoint"]
    rain_meta = artifacts["rain_meta"]
    threshold = artifacts["probability_threshold"]

    # Cloud type: all five inputs were verified complete in the source audit.
    cloud_df = chunk[CLOUD_FEATURES].apply(pd.to_numeric, errors="coerce")
    if cloud_df.isna().any().any():
        raise ValueError("Missing/non-numeric cloud-type input found.")
    x_cloud = cloud_scaler.transform(cloud_df)

    # CBH: preserve the fitted imputer, scaler and category mappings.
    cbh_num_df = chunk[cbh_num_features].apply(
        pd.to_numeric, errors="coerce"
    )
    x_cbh_num = cbh_prep["imputer"].transform(cbh_num_df)
    x_cbh_num_df = pd.DataFrame(
        x_cbh_num, columns=cbh_num_features, index=chunk.index
    )
    x_cbh_num = cbh_prep["scaler"].transform(x_cbh_num_df)

    x_cbh_cat = np.zeros(
        (len(chunk), len(cbh_cat_features)), dtype=np.int64
    )
    for j, col in enumerate(cbh_cat_features):
        mapping = cbh_prep["category_maps"][col]
        values = chunk[col].fillna(-1).astype(str)
        x_cbh_cat[:, j] = (
            values.map(mapping).fillna(0).astype(np.int64).to_numpy()
        )

    # Precipitation detection: saved training medians and scaling statistics.
    x_precip = prepare_numeric(
        chunk, precip_features, precip_checkpoint["medians"]
    )
    x_precip = standardize(
        x_precip, precip_checkpoint["means"], precip_checkpoint["stds"]
    )

    with torch.inference_mode():
        cloud_output = models["cloud"](tensor(x_cloud, device))
        cloud_logits = cloud_output.mean(dim=1)
        cloud_type_id = cloud_logits.argmax(dim=1).cpu().numpy()

        cbh_output = models["cbh"](
            tensor(x_cbh_num, device),
            tensor(x_cbh_cat, device, dtype=torch.long)
        )
        cbh_scaled = (
            cbh_output.mean(dim=1).squeeze(-1).cpu().numpy()
        )
        cbh_km = (
            cbh_scaled * cbh_prep["target_std"]
            + cbh_prep["target_mean"]
        )

        precip_output = models["precip"](tensor(x_precip, device))
        precip_logits = precip_output.mean(dim=1).reshape(-1)
        rain_probability = torch.sigmoid(
            torch.clamp(precip_logits, -30, 30)
        ).cpu().numpy()
        rain_detected = (rain_probability >= threshold).astype(np.int32)

        # Conditional regressor: run only for rows detected as rainy.
        conditional_rainfall = np.full(len(chunk), np.nan, dtype=np.float32)
        detected_indices = np.flatnonzero(rain_detected == 1)

        if len(detected_indices):
            detected_rows = chunk.iloc[detected_indices]
            x_rain = prepare_numeric(
                detected_rows, rain_features, rain_meta["medians"]
            )
            x_rain = standardize(
                x_rain, rain_meta["means"], rain_meta["stds"]
            )
            rain_output = models["rain"](tensor(x_rain, device))
            rain_log = rain_output.mean(dim=1).reshape(-1)
            rain_values = torch.expm1(rain_log).clamp(min=0).cpu().numpy()
            conditional_rainfall[detected_indices] = rain_values

    final_rainfall = np.where(
        rain_detected == 1, conditional_rainfall, 0.0
    ).astype(np.float32)

    return {
        "cloud_type_id": cloud_type_id.astype(np.int64),
        "cloud_type": [CLOUD_LABELS[int(x)] for x in cloud_type_id],
        "CBH_km": np.asarray(cbh_km, dtype=np.float32),
        "rain_probability": rain_probability.astype(np.float32),
        "rain_detected": rain_detected,
        "rainfall_intensity_conditional_mm_hr": conditional_rainfall,
        "final_rainfall_mm_hr": final_rainfall,
    }


def validate_chunk(chunk, chunk_number):
    # Conditional intensity is intentionally NaN when rain is not detected.
    always_finite = [
        "cloud_type_id",
        "CBH_km",
        "rain_probability",
        "rain_detected",
        "final_rainfall_mm_hr",
        "bt_anomaly_flag",
    ]
    for col in always_finite:
        values = chunk[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite prediction in {col}, chunk {chunk_number}."
            )

    if not chunk["cloud_type_id"].between(0, 6).all():
        raise RuntimeError(f"Invalid cloud class in chunk {chunk_number}.")

    no_rain = chunk["rain_detected"] == 0
    if not (chunk.loc[no_rain, "final_rainfall_mm_hr"] == 0).all():
        raise RuntimeError(
            f"Rainfall gate consistency failed in chunk {chunk_number}."
        )
    if chunk.loc[
        no_rain, "rainfall_intensity_conditional_mm_hr"
    ].notna().any():
        raise RuntimeError(
            f"Conditional rainfall should be NaN for non-rain rows "
            f"in chunk {chunk_number}."
        )

    detected = chunk["rain_detected"] == 1
    detected_intensity = chunk.loc[
        detected, "rainfall_intensity_conditional_mm_hr"
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(detected_intensity).all():
        raise RuntimeError(
            f"Detected rain has missing/non-finite conditional intensity "
            f"in chunk {chunk_number}."
        )


def checkpoint_manifest(manifest):
    if OUTPUT_FILE.exists():
        manifest["output_bytes_committed"] = OUTPUT_FILE.stat().st_size
    atomic_write_json(MANIFEST_FILE, manifest)


def main():
    args = parse_args()
    if args.chunk_size <= 0 or args.checkpoint_every <= 0:
        raise ValueError("Chunk size and checkpoint interval must be positive.")

    input_file = locate_input(args.input)
    signature = input_signature(input_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("FULL-YEAR AVIATION WEATHER INFERENCE")
    print("=" * 72)
    print("Input:", input_file)
    print("Output:", OUTPUT_FILE)
    print("Device:", device)
    print("Chunk size:", args.chunk_size)

    manifest = load_or_initialize_manifest(
        signature, args.chunk_size, args.overwrite
    )

    models, artifacts = load_models(device)
    threshold = artifacts["probability_threshold"]
    manifest["probability_threshold"] = threshold
    atomic_write_json(MANIFEST_FILE, manifest)

    required_columns = sorted(set(
        ["timestamp"]
        + CLOUD_FEATURES
        + artifacts["cbh_num_features"]
        + artifacts["cbh_cat_features"]
        + artifacts["precip_features"]
        + artifacts["rain_features"]
    ))

    # Check header before beginning or resuming.
    header = pd.read_csv(input_file, nrows=0)
    missing_columns = [c for c in required_columns if c not in header.columns]
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")

    start_chunk = int(manifest["chunks_completed"])
    total_rows = int(manifest["rows_processed"])
    cloud_class_counts = {
        str(k): int(v)
        for k, v in manifest["cloud_class_counts"].items()
    }
    rain_detected_count = int(manifest["rain_detected_rows"])
    anomaly_count = int(manifest["bt_anomaly_rows"])

    for chunk_index, chunk in enumerate(
        pd.read_csv(input_file, chunksize=args.chunk_size)
    ):
        if chunk_index < start_chunk:
            continue

        chunk_number = chunk_index + 1

        bt_anomaly = (
            (pd.to_numeric(chunk["BT_11"], errors="coerce") > 330)
            | (pd.to_numeric(chunk["BT_12"], errors="coerce") > 330)
        ).fillna(False)
        chunk["bt_anomaly_flag"] = bt_anomaly.astype(np.int8)

        predictions = predict_chunk(chunk, models, artifacts, device)
        for col, values in predictions.items():
            chunk[col] = values

        validate_chunk(chunk, chunk_number)

        unique_classes, counts = np.unique(
            predictions["cloud_type_id"], return_counts=True
        )
        for class_id, count in zip(unique_classes, counts):
            cloud_class_counts[str(int(class_id))] += int(count)

        rain_detected_count += int(predictions["rain_detected"].sum())
        anomaly_count += int(bt_anomaly.sum())

        # Append a chunk to the single output CSV.
        write_header = total_rows == 0
        with open(OUTPUT_FILE, "a", newline="") as output_handle:
            chunk.to_csv(
                output_handle,
                index=False,
                header=write_header
            )
            output_handle.flush()
            os.fsync(output_handle.fileno())

        total_rows += len(chunk)
        completed_chunks = chunk_number

        manifest.update({
            "chunks_completed": completed_chunks,
            "rows_processed": total_rows,
            "cloud_class_counts": cloud_class_counts,
            "rain_detected_rows": rain_detected_count,
            "bt_anomaly_rows": anomaly_count,
            "probability_threshold": threshold,
        })

        if completed_chunks % args.checkpoint_every == 0:
            checkpoint_manifest(manifest)
            print(
                f"Checkpoint: chunk {completed_chunks:,} | "
                f"rows {total_rows:,} | "
                f"rain detections {rain_detected_count:,}",
                flush=True
            )

    # Finalize manifest and summary.
    checkpoint_manifest(manifest)
    manifest["status"] = "completed"
    manifest["completed_rows"] = total_rows
    manifest["completed_chunks"] = manifest["chunks_completed"]
    atomic_write_json(MANIFEST_FILE, manifest)

    summary = {
        "project": "Aviation Weather Intelligence",
        "period_intended": "2022-07-01 to 2023-06-30",
        "input_file": str(input_file),
        "output_file": str(OUTPUT_FILE),
        "total_rows_processed": total_rows,
        "chunk_size": args.chunk_size,
        "number_of_chunks": manifest["chunks_completed"],
        "device": str(device),
        "models": {
            "cloud_type_checkpoint": str(CLOUD_CKPT),
            "cbh_checkpoint": str(CBH_CKPT),
            "precipitation_detection_checkpoint": str(PRECIP_CKPT),
            "rainfall_intensity_checkpoint": str(RAIN_CKPT),
        },
        "precipitation_probability_threshold": threshold,
        "cloud_type_class_counts": cloud_class_counts,
        "rain_detected_rows": rain_detected_count,
        "rain_detection_percentage": (
            rain_detected_count / total_rows * 100 if total_rows else 0
        ),
        "extreme_bt_rows_flagged": anomaly_count,
        "rainfall_regression_policy": (
            "Conditional rainfall model is run only for rows with "
            "rain_probability >= the saved model threshold. Conditional "
            "intensity is NaN for rows not classified as rain; final "
            "rainfall is zero for those rows."
        ),
        "resume_manifest": str(MANIFEST_FILE),
        "coverage_note": (
            "The supplied integrated dataset was previously observed to "
            "end at timestamp 2023176_0540 (2023-06-25 05:40), so it does "
            "not cover the complete intended period through 2023-06-30."
        ),
        "evaluation_note": (
            "Predictions are model outputs, not independent validation. "
            "Evaluate each model only against its available reference "
            "target and respect the original train/validation/test split."
        ),
    }
    atomic_write_json(SUMMARY_FILE, summary)

    print("\n" + "=" * 72)
    print("FULL-YEAR INFERENCE COMPLETED")
    print("=" * 72)
    print("Total rows processed:", f"{total_rows:,}")
    print("Chunks:", manifest["chunks_completed"])
    print("Cloud class counts:", cloud_class_counts)
    print("Rain detections:", f"{rain_detected_count:,}")
    print("Extreme BT rows flagged:", anomaly_count)
    print("Output CSV:", OUTPUT_FILE)
    print("Summary JSON:", SUMMARY_FILE)
    print("Resume manifest:", MANIFEST_FILE)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nINFERENCE FAILED:", repr(exc), file=sys.stderr)
        raise

