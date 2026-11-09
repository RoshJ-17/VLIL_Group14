import argparse

import json

import pickle

from pathlib import Path

import joblib

import numpy as np

import pandas as pd

import torch

from tabm import TabM

# ============================================================

# 1. CONFIGURATION

# ============================================================

ROOT = Path("/home/m_prathilothamai/aviation_weather_project")

def parse_args():

    parser = argparse.ArgumentParser(

        description="Run four-model aviation weather inference."

    )

    parser.add_argument("--input", required=True, help="Prepared feature CSV")

    parser.add_argument("--output-dir", required=True, help="Batch output directory")

    parser.add_argument("--batch-id", required=True, help="Unique batch identifier")

    parser.add_argument("--batch-size", type=int, default=8192)

    return parser.parse_args()

ARGS = parse_args()

INPUT_FILE = Path(ARGS.input).resolve()

OUTPUT_DIR = Path(ARGS.output_dir).resolve()

BATCH_ID = ARGS.batch_id

if not BATCH_ID.replace("-", "").replace("_", "").isalnum():

    raise ValueError("Batch ID may contain only letters, numbers, hyphens and underscores.")

if ARGS.batch_size < 1:

    raise ValueError("Batch size must be positive.")

if not INPUT_FILE.is_file():

    raise FileNotFoundError(f"Input feature CSV not found: {INPUT_FILE}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / f"{BATCH_ID}_predictions.csv"

SUMMARY_FILE = OUTPUT_DIR / f"{BATCH_ID}_prediction_summary.json"

BATCH_SIZE = ARGS.batch_size

DEVICE = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)

print("INTEGRATED AVIATION WEATHER INFERENCE")

print("=" * 65)

print("Device:", DEVICE)

print("Input:", INPUT_FILE)

print("Output:", OUTPUT_FILE)

print("Batch size:", BATCH_SIZE)

# ============================================================

# 2. MODEL AND PREPROCESSING PATHS

# ============================================================

CLOUD_CKPT = (

    ROOT /

    "eda_results/cloudType/training/sampling_10epoch_comparison/"

    "tabm_sqrt_weighted_best_checkpoint.pt"

)

CLOUD_SCALER = (

    ROOT /

    "eda_results/cloudType/preprocessing/scaled_splits/"

    "standard_scaler.joblib"

)

CBH_CKPT = ROOT / "tabm_cbh_output/best_tabm_cbh.pth"

CBH_PREPROCESS = ROOT / "tabm_cbh_output/cbh_preprocessing.pkl"

PRECIP_CKPT = ROOT / "tabm_full_w1_run/best_checkpoint.pt"

PRECIP_META = ROOT / "tabm_full_data/metadata.json"

RAIN_CKPT = (

    ROOT / "tabm_regression_weighted_run/best_checkpoint.pt"

)

RAIN_META = ROOT / "tabm_regression_data/metadata.json"

# ============================================================

# 3. FEATURES AND LABELS

# ============================================================

CLOUD_FEATURES = [

    "lat", "lon", "BT_11", "BT_12", "BTD"

]

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

    6: "ice_high"

}

# ============================================================

# 4. GENERAL HELPERS

# ============================================================

def load_checkpoint(path):

    try:

        return torch.load(

            path,

            map_location=DEVICE,

            weights_only=False

        )

    except TypeError:

        return torch.load(path, map_location=DEVICE)

def extract_state_dict(checkpoint, keys):

    if isinstance(checkpoint, dict):

        for key in keys:

            if key in checkpoint and isinstance(checkpoint[key], dict):

                candidate = checkpoint[key]

                if candidate and all(

                    torch.is_tensor(value)

                    for value in candidate.values()

                ):

                    return candidate

        if checkpoint and all(

            torch.is_tensor(value)

            for value in checkpoint.values()

        ):

            return checkpoint

    raise RuntimeError(

        "Could not extract model state_dict from checkpoint."

    )

def tensor(array, dtype=torch.float32):

    return torch.as_tensor(

        np.asarray(array),

        dtype=dtype,

        device=DEVICE

    )

def prepare_numeric(df, features, medians):

    values = df[features].apply(

        pd.to_numeric,

        errors="coerce"

    ).to_numpy(dtype=np.float64)

    medians = np.asarray(medians, dtype=np.float64)

    if len(medians) != len(features):

        raise ValueError(

            f"Median count mismatch for features: {features}"

        )

    invalid = ~np.isfinite(values)

    if invalid.any():

        values[invalid] = np.broadcast_to(

            medians, values.shape

        )[invalid]

    if not np.isfinite(values).all():

        raise ValueError(

            "Non-finite values remain after median imputation."

        )

    return values

def standardize(values, means, stds):

    values = np.asarray(values, dtype=np.float64)

    means = np.asarray(means, dtype=np.float64)

    stds = np.asarray(stds, dtype=np.float64)

    if np.any(stds == 0):

        raise ValueError("Zero standard deviation found.")

    result = (values - means) / stds

    # Same clipping policy used during model preparation.

    result = np.clip(result, -10, 10)

    if not np.isfinite(result).all():

        raise ValueError(

            "Non-finite values found after standardization."

        )

    return result.astype(np.float32)

def verify_files(paths):

    missing = [str(path) for path in paths if not path.exists()]

    if missing:

        raise FileNotFoundError(

            "Required files are missing:\n" + "\n".join(missing)

        )

# ============================================================

# 5. LOAD PREPROCESSING ARTIFACTS

# ============================================================

print("\nLoading preprocessing artifacts...")

verify_files([

    INPUT_FILE,

    CLOUD_CKPT,

    CLOUD_SCALER,

    CBH_CKPT,

    CBH_PREPROCESS,

    PRECIP_CKPT,

    PRECIP_META,

    RAIN_CKPT,

    RAIN_META

])

cloud_scaler = joblib.load(CLOUD_SCALER)

cbh_prep = joblib.load(CBH_PREPROCESS)

with open(PRECIP_META, "r") as f:

    precip_meta = json.load(f)

with open(RAIN_META, "r") as f:

    rain_meta = json.load(f)

precip_checkpoint = load_checkpoint(PRECIP_CKPT)

rain_checkpoint = load_checkpoint(RAIN_CKPT)

print("Cloud type scaler loaded.")

print("CBH preprocessing loaded.")

print("Precipitation metadata loaded.")

print("Rainfall metadata loaded.")

# ============================================================

# 6. VERIFY FEATURE ORDER

# ============================================================

cloud_features = list(cloud_scaler.feature_names_in_)

if cloud_features != CLOUD_FEATURES:

    raise RuntimeError(

        f"Cloud feature mismatch: {cloud_features}"

    )

cbh_num_features = cbh_prep["num_features"]

cbh_cat_features = cbh_prep["cat_features"]

if len(cbh_num_features) != 9 or len(cbh_cat_features) != 2:

    raise RuntimeError("Unexpected CBH feature configuration.")

precip_features = precip_meta["features"]

rain_features = rain_meta["features"]

if precip_features != PRECIP_FEATURES_EXPECTED:

    raise RuntimeError(

        f"Precipitation feature mismatch: {precip_features}"

    )

if rain_features != PRECIP_FEATURES_EXPECTED:

    raise RuntimeError(

        f"Rainfall feature mismatch: {rain_features}"

    )

for key in ["features", "medians", "means", "stds"]:

    if key not in precip_checkpoint:

        raise RuntimeError(

            f"Precipitation checkpoint missing {key}"

        )

if precip_checkpoint["features"] != precip_features:

    raise RuntimeError(

        "Precipitation checkpoint feature order mismatch."

    )

for key in ["medians", "means", "stds"]:

    if not np.array_equal(

        np.asarray(precip_meta[key]),

        np.asarray(precip_checkpoint[key])

    ):

        raise RuntimeError(

            f"Precipitation {key} mismatch."

        )

if len(cbh_num_features) != 9:

    raise RuntimeError("Unexpected CBH numerical feature count.")

if len(precip_features) != 10 or len(rain_features) != 10:

    raise RuntimeError("Unexpected precipitation/rainfall feature count.")

print("All feature orders verified.")

# ============================================================

# 7. BUILD AND LOAD CLOUD TYPE MODEL

# ============================================================

print("\nLoading cloud type model...")

cloud_checkpoint = load_checkpoint(CLOUD_CKPT)

cloud_model = TabM.make(

    n_num_features=5,

    d_out=7,

    k=4,

    d_block=64,

    n_blocks=2,

    dropout=0.1

).to(DEVICE)

cloud_model.load_state_dict(

    extract_state_dict(

        cloud_checkpoint,

        ["state_dict", "model_state_dict", "model"]

    ),

    strict=True

)

cloud_model.eval()

print("Cloud type model loaded.")

# ============================================================

# 8. BUILD AND LOAD CBH MODEL

# ============================================================

print("\nLoading CBH model...")

cat_cardinalities = [

    max(mapping.values()) + 1

    for mapping in cbh_prep["category_maps"].values()

]

cbh_model = TabM.make(

    n_num_features=len(cbh_num_features),

    cat_cardinalities=cat_cardinalities,

    d_out=1,

    k=4

).to(DEVICE)

cbh_model.load_state_dict(

    load_checkpoint(CBH_CKPT),

    strict=True

)

cbh_model.eval()

print("CBH model loaded.")

# ============================================================

# 9. BUILD AND LOAD PRECIPITATION DETECTION MODEL

# ============================================================

print("\nLoading precipitation detection model...")

precip_model = TabM.make(

    n_num_features=10,

    cat_cardinalities=None,

    d_out=1,

    k=4,

    d_block=64,

    n_blocks=2,

    dropout=0.1

).to(DEVICE)

precip_model.load_state_dict(

    extract_state_dict(

        precip_checkpoint,

        ["state_dict"]

    ),

    strict=True

)

precip_model.eval()

PROBABILITY_THRESHOLD = float(

    precip_checkpoint["probability_threshold"]

)

print(

    "Precipitation model loaded. Threshold:",

    PROBABILITY_THRESHOLD

)

# ============================================================

# 10. BUILD AND LOAD RAINFALL INTENSITY MODEL

# ============================================================

print("\nLoading rainfall intensity model...")

rain_model = TabM.make(

    n_num_features=10,

    d_out=1,

    k=4,

    d_block=64,

    n_blocks=2,

    dropout=0.1

).to(DEVICE)

rain_model.load_state_dict(

    extract_state_dict(

        rain_checkpoint,

        ["model"]

    ),

    strict=True

)

rain_model.eval()

print("Rainfall intensity model loaded.")

# ============================================================

# 11. PREDICT ONE CHUNK

# ============================================================

def predict_chunk(chunk):

    # --------------------------------------------------------

    # A. CLOUD TYPE

    # --------------------------------------------------------

    X_cloud_df = chunk[CLOUD_FEATURES].apply(

        pd.to_numeric,

        errors="coerce"

    )

    if X_cloud_df.isna().any().any():

        raise ValueError(

            "Missing cloud type model input found."

        )

    X_cloud = cloud_scaler.transform(X_cloud_df)

    # --------------------------------------------------------

    # B. CBH

    # --------------------------------------------------------

    X_cbh_num_df = chunk[cbh_num_features].apply(

        pd.to_numeric,

        errors="coerce"

    )

    X_cbh_num = cbh_prep["imputer"].transform(

        X_cbh_num_df

    )

    # Preserve fitted feature names for scaler.

    X_cbh_num_df = pd.DataFrame(

        X_cbh_num,

        columns=cbh_num_features,

        index=chunk.index

    )

    X_cbh_num = cbh_prep["scaler"].transform(

        X_cbh_num_df

    )

    X_cbh_cat = np.zeros(

        (len(chunk), len(cbh_cat_features)),

        dtype=np.int64

    )

    for j, col in enumerate(cbh_cat_features):

        mapping = cbh_prep["category_maps"][col]

        values = chunk[col].fillna(-1).astype(str)

        X_cbh_cat[:, j] = (

            values.map(mapping)

            .fillna(0)

            .astype(np.int64)

            .to_numpy()

        )

    # --------------------------------------------------------

    # C. PRECIPITATION DETECTION

    # --------------------------------------------------------

    X_precip = prepare_numeric(

        chunk,

        precip_features,

        precip_checkpoint["medians"]

    )

    X_precip = standardize(

        X_precip,

        precip_checkpoint["means"],

        precip_checkpoint["stds"]

    )

    # --------------------------------------------------------

    # E. MODEL INFERENCE

    # --------------------------------------------------------

    with torch.inference_mode():

        # Cloud type

        cloud_output = cloud_model(

            tensor(X_cloud)

        )

        cloud_logits = cloud_output.mean(dim=1)

        cloud_type_id = (

            cloud_logits.argmax(dim=1)

            .cpu()

            .numpy()

        )

        # CBH

        cbh_output = cbh_model(

            tensor(X_cbh_num),

            tensor(X_cbh_cat, dtype=torch.long)

        )

        cbh_scaled = (

            cbh_output.mean(dim=1)

            .squeeze(-1)

            .cpu()

            .numpy()

        )

        cbh_km = (

            cbh_scaled * cbh_prep["target_std"]

            + cbh_prep["target_mean"]

        )

        # Precipitation detection

        precip_output = precip_model(

            tensor(X_precip)

        )

        precip_logits = (

            precip_output.mean(dim=1)

            .reshape(-1)

        )

        rain_probability = torch.sigmoid(

            torch.clamp(precip_logits, -30, 30)

        ).cpu().numpy()

        rain_detected = (

            rain_probability >= PROBABILITY_THRESHOLD

        ).astype(np.int32)

        # Rainfall intensity: infer only where precipitation is detected.
        # All other rows retain the default rainfall intensity of 0 mm/hr.
        conditional_rainfall = np.zeros(len(chunk), dtype=np.float32)
        rain_indices = np.flatnonzero(rain_detected == 1)

        if rain_indices.size > 0:
            rain_chunk = chunk.iloc[rain_indices]

            X_rain = prepare_numeric(
                rain_chunk,
                rain_features,
                rain_meta["medians"]
            )

            X_rain = standardize(
                X_rain,
                rain_meta["means"],
                rain_meta["stds"]
            )

            rain_output = rain_model(tensor(X_rain))
            rain_log = rain_output.mean(dim=1).reshape(-1)
            rain_values = torch.expm1(rain_log).clamp(min=0).cpu().numpy()
            conditional_rainfall[rain_indices] = rain_values

    # The conditional array is already zero for rows without detected rain.
    final_rainfall = conditional_rainfall.copy()

    return {

        "cloud_type_id": cloud_type_id,

        "cloud_type": [

            CLOUD_LABELS[int(x)]

            for x in cloud_type_id

        ],

        "CBH_km": cbh_km,

        "rain_probability": rain_probability,

        "rain_detected": rain_detected,

        "rainfall_intensity_conditional_mm_hr": conditional_rainfall,

        "final_rainfall_mm_hr": final_rainfall

    }

# ============================================================

# 12. CHUNKED FULL-DATASET INFERENCE

# ============================================================

print("\n" + "=" * 65)

print("STARTING BATCH INFERENCE")

print("=" * 65)

total_rows = 0

chunk_number = 0

first_chunk = True

cloud_class_counts = {

    str(i): 0 for i in range(7)

}

rain_detected_count = 0

rain_intensity_inference_rows = 0

anomaly_count = 0

prediction_columns = [

    "cloud_type_id",

    "cloud_type",

    "CBH_km",

    "rain_probability",

    "rain_detected",

    "rainfall_intensity_conditional_mm_hr",

    "final_rainfall_mm_hr"

]

numeric_prediction_columns = [

    "CBH_km",

    "rain_probability",

    "rain_detected",

    "rainfall_intensity_conditional_mm_hr",

    "final_rainfall_mm_hr"

]

with torch.inference_mode():

    for chunk in pd.read_csv(

        INPUT_FILE,

        chunksize=BATCH_SIZE

    ):

        chunk_number += 1

        required_columns = (

            ["timestamp"] +

            CLOUD_FEATURES +

            cbh_num_features +

            cbh_cat_features +

            precip_features +

            rain_features

        )

        missing_columns = [

            col for col in required_columns

            if col not in chunk.columns

        ]

        if missing_columns:

            raise ValueError(

                f"Missing input columns: {missing_columns}"

            )

        # Identify extreme BT observations without deleting them.

        bt_anomaly = (

            (chunk["BT_11"] > 330) |

            (chunk["BT_12"] > 330)

        )

        chunk["bt_anomaly_flag"] = (

            bt_anomaly.astype(np.int8)

        )

        anomaly_count += int(bt_anomaly.sum())

        # Generate predictions.

        predictions = predict_chunk(chunk)

        for col in prediction_columns:

            chunk[col] = predictions[col]

        # Count classes and rain detections.

        unique_classes, counts = np.unique(

            predictions["cloud_type_id"],

            return_counts=True

        )

        for class_id, count in zip(unique_classes, counts):

            cloud_class_counts[str(int(class_id))] += int(count)

        rain_detected_count += int(

            predictions["rain_detected"].sum()

        )

        rain_intensity_inference_rows += int(

            predictions["rain_detected"].sum()

        )

        # Validate each chunk.

        for col in numeric_prediction_columns:

            if not np.isfinite(

                chunk[col].to_numpy(dtype=np.float64)

            ).all():

                raise ValueError(

                    f"Non-finite predictions found in {col}, "

                    f"chunk {chunk_number}"

                )

        # Verify the rainfall gate.

        no_rain = chunk["rain_detected"] == 0

        if not (

            chunk.loc[no_rain, "final_rainfall_mm_hr"] == 0

        ).all():

            raise RuntimeError(

                "Rainfall gate consistency check failed."

            )

        # Save this chunk.

        chunk.to_csv(

            OUTPUT_FILE,

            mode="w" if first_chunk else "a",

            header=first_chunk,

            index=False

        )

        first_chunk = False

        total_rows += len(chunk)

        print(

            f"Chunk {chunk_number}: "

            f"{len(chunk):,} rows processed | "

            f"Total: {total_rows:,} | "

            f"Rain detections: {rain_detected_count:,}",

            flush=True

        )

# ============================================================

# 13. FINAL SUMMARY

# ============================================================

summary = {

    "project": "Aviation Weather Intelligence",

    "input_file": str(INPUT_FILE),

    "output_file": str(OUTPUT_FILE),

    "total_rows_processed": total_rows,

    "batch_size": BATCH_SIZE,

    "number_of_chunks": chunk_number,

    "device": str(DEVICE),

    "models": {

        "cloud_type": "TabM",

        "CBH": "TabM",

        "precipitation_detection": "TabM",

        "rainfall_intensity": "TabM"

    },

    "precipitation_probability_threshold": PROBABILITY_THRESHOLD,

    "cloud_type_class_counts": cloud_class_counts,

    "rain_detected_rows": rain_detected_count,

    "rain_intensity_inference_rows": rain_intensity_inference_rows,

    "rain_intensity_skipped_rows": total_rows - rain_intensity_inference_rows,

    "rain_detection_percentage": (

        rain_detected_count / total_rows * 100

        if total_rows else 0

    ),

    "extreme_bt_rows_flagged": anomaly_count,

    "rainfall_gate": (

        "Final rainfall is zero when rain_detected is zero."

    ),

    "evaluation_warning": (

        "Prediction outputs are inference results. "

        "Evaluation status depends on the input acquisition dates "

        "and the training periods of the respective models."

    )

}

with open(SUMMARY_FILE, "w") as f:

    json.dump(summary, f, indent=4)

print("\n" + "=" * 65)

print("FULL INFERENCE COMPLETED")

print("=" * 65)

print("Total rows processed:", f"{total_rows:,}")

print("Chunks:", chunk_number)

print("Cloud class counts:", cloud_class_counts)

print("Rain detections:", f"{rain_detected_count:,}")

print("Rainfall intensity inference rows:", f"{rain_intensity_inference_rows:,}")

print("Rainfall intensity skipped rows:", f"{total_rows - rain_intensity_inference_rows:,}")

print("Extreme BT rows flagged:", anomaly_count)

print("Output CSV:", OUTPUT_FILE)

print("Summary JSON:", SUMMARY_FILE)

print("\nNo source data was modified.")

print("Inference pipeline finished.")
