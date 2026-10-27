"""
Prepare scaled cloud-type data for TabM using PyTorch.

- Uses memory-mapped NumPy arrays to avoid loading the full dataset into RAM.
- Encodes seven cloud classes as integers 0-6.
- Calculates class weights using training data only.
- Creates PyTorch Dataset and DataLoader objects.
- Does not train the model.

All outputs:
eda_results/cloudType/preprocessing/tabm_data/
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, Sampler


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/home/m_prathilothamai/aviation_weather_project"
)

INPUT_DIR = (
    PROJECT_DIR
    / "eda_results"
    / "cloudType"
    / "preprocessing"
    / "scaled_splits"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "eda_results"
    / "cloudType"
    / "preprocessing"
    / "tabm_data"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
]

TARGET = "cloud_type_label"

EXPECTED_CLASSES = [
    "clear",
    "liquid_low",
    "liquid_middle",
    "liquid_high",
    "ice_low",
    "ice_middle",
    "ice_high",
]

# Fixed class-to-integer mapping.
CLASS_TO_INDEX = {
    label: index
    for index, label in enumerate(EXPECTED_CLASSES)
}

INDEX_TO_CLASS = {
    index: label
    for label, index in CLASS_TO_INDEX.items()
}

SPLITS = {
    "train": INPUT_DIR / "train.csv",
    "validation": INPUT_DIR / "validation.csv",
    "test": INPUT_DIR / "test.csv",
}

CHUNK_SIZE = 500_000

BATCH_SIZE = 2048

# Keep workers at zero initially for reliable memory-mapped access.
NUM_WORKERS = 0

# Number of samples in each block shuffled by the sampler.
SHUFFLE_BLOCK_SIZE = 100_000

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 2. CHECK ENVIRONMENT
# ============================================================

print("=" * 75)
print("CLOUD-TYPE TABM DATA PREPARATION")
print("=" * 75)

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU not detected. Data preparation will use CPU.")

for split_name, path in SPLITS.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Scaled {split_name} file not found: {path}"
        )

print("\nAll scaled split files found.")


# ============================================================
# 3. FIRST PASS: COUNT ROWS AND VALIDATE LABELS
# ============================================================

print("\n" + "=" * 75)
print("STEP 1: COUNT ROWS AND VALIDATE TARGET LABELS")
print("=" * 75)

split_row_counts = {}
split_class_counts = {}

for split_name, path in SPLITS.items():

    print(f"\nScanning {split_name.upper()}...")

    total_rows = 0

    counts = {
        label: 0
        for label in EXPECTED_CLASSES
    }

    for chunk in pd.read_csv(
        path,
        usecols=[TARGET],
        dtype={TARGET: "string"},
        chunksize=CHUNK_SIZE,
    ):

        total_rows += len(chunk)

        values = chunk[TARGET].value_counts(
            dropna=False
        )

        for label, count in values.items():

            if pd.isna(label) or label not in CLASS_TO_INDEX:
                raise ValueError(
                    f"Unexpected or missing target label "
                    f"in {split_name}: {label}"
                )

            counts[label] += int(count)

    split_row_counts[split_name] = total_rows
    split_class_counts[split_name] = counts

    print(f"Rows: {total_rows:,}")

    for label in EXPECTED_CLASSES:
        print(f"{label:16s}: {counts[label]:,}")


# ============================================================
# 4. CREATE MEMORY-MAPPED NUMPY ARRAYS
# ============================================================

print("\n" + "=" * 75)
print("STEP 2: CONVERT CSV FILES TO NUMPY ARRAYS")
print("=" * 75)

array_paths = {}

for split_name, csv_path in SPLITS.items():

    print(f"\nConverting {split_name.upper()}...")

    total_rows = split_row_counts[split_name]

    X_path = OUTPUT_DIR / f"X_{split_name}.npy"
    y_path = OUTPUT_DIR / f"y_{split_name}.npy"

    # Create disk-backed arrays.
    X_array = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, len(FEATURES)),
    )

    y_array = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.int64,
        shape=(total_rows,),
    )

    position = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            csv_path,
            usecols=FEATURES + [TARGET],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        X_chunk = chunk[FEATURES].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(X_chunk).all():
            raise ValueError(
                f"Non-finite feature values in "
                f"{split_name}, chunk {chunk_number}"
            )

        y_chunk = (
            chunk[TARGET]
            .map(CLASS_TO_INDEX)
            .to_numpy(dtype=np.int64)
        )

        if np.any(y_chunk < 0) or np.any(
            y_chunk >= len(EXPECTED_CLASSES)
        ):
            raise ValueError(
                f"Invalid encoded labels in {split_name}"
            )

        end_position = position + len(chunk)

        X_array[position:end_position] = X_chunk
        y_array[position:end_position] = y_chunk

        position = end_position

        if chunk_number % 10 == 0:
            print(
                f"{split_name}: {position:,} rows converted."
            )

    if position != total_rows:
        raise RuntimeError(
            f"Row count mismatch in {split_name}: "
            f"expected {total_rows}, got {position}"
        )

    X_array.flush()
    y_array.flush()

    del X_array
    del y_array

    array_paths[split_name] = {
        "features": str(X_path),
        "labels": str(y_path),
        "rows": total_rows,
    }

    print(f"{split_name.upper()} conversion completed.")
    print(f"Feature array: {X_path}")
    print(f"Label array:   {y_path}")


# ============================================================
# 5. CALCULATE TRAINING CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 75)
print("STEP 3: CALCULATE TRAINING CLASS WEIGHTS")
print("=" * 75)

train_counts = np.array(
    [
        split_class_counts["train"][label]
        for label in EXPECTED_CLASSES
    ],
    dtype=np.float64,
)

if np.any(train_counts == 0):
    raise ValueError(
        "At least one class is absent from training."
    )

number_of_classes = len(EXPECTED_CLASSES)
number_of_train_rows = split_row_counts["train"]

# Balanced class-weight formula:
# weight = total_samples / (number_of_classes * class_count)

class_weights = (
    number_of_train_rows
    / (number_of_classes * train_counts)
)

class_weight_dict = {
    label: float(weight)
    for label, weight in zip(
        EXPECTED_CLASSES,
        class_weights,
    )
}

print("\nTraining class weights:")

for label, weight in class_weight_dict.items():
    print(f"{label:16s}: {weight:.6f}")

weights_path = OUTPUT_DIR / "class_weights.json"

with open(weights_path, "w", encoding="utf-8") as file:
    json.dump(class_weight_dict, file, indent=4)

# Save weights in PyTorch format.
weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
)

torch.save(
    weights_tensor,
    OUTPUT_DIR / "class_weights.pt",
)


# ============================================================
# 6. SAVE CLASS MAPPING
# ============================================================

print("\n" + "=" * 75)
print("STEP 4: SAVE CLASS LABEL MAPPING")
print("=" * 75)

class_mapping = {
    "class_to_index": CLASS_TO_INDEX,
    "index_to_class": {
        str(index): label
        for index, label in INDEX_TO_CLASS.items()
    },
}

with open(
    OUTPUT_DIR / "class_mapping.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(class_mapping, file, indent=4)

print(json.dumps(class_mapping, indent=4))


# ============================================================
# 7. PYTORCH DATASET
# ============================================================

print("\n" + "=" * 75)
print("STEP 5: DEFINE PYTORCH DATASET AND DATALOADER")
print("=" * 75)


class CloudTypeDataset(Dataset):
    """
    Memory-mapped dataset for cloud-type classification.
    """

    def __init__(self, X_path, y_path):

        self.X_path = str(X_path)
        self.y_path = str(y_path)

        self.X = np.load(
            self.X_path,
            mmap_mode="r",
        )

        self.y = np.load(
            self.y_path,
            mmap_mode="r",
        )

        if len(self.X) != len(self.y):
            raise ValueError(
                "Feature and label array lengths differ."
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        # Copy individual samples before converting to tensors.
        x = torch.from_numpy(
            np.array(self.X[index], dtype=np.float32, copy=True)
        )

        y = torch.tensor(
            int(self.y[index]),
            dtype=torch.long,
        )

        return x, y


# ============================================================
# 8. MEMORY-EFFICIENT BLOCK SHUFFLE SAMPLER
# ============================================================

class BlockShuffleSampler(Sampler):
    """
    Shuffle blocks and then shuffle indices within each block.

    This avoids creating a full random permutation of 37 million
    indices in memory.
    """

    def __init__(
        self,
        dataset,
        block_size=100_000,
        seed=42,
    ):
        self.dataset = dataset
        self.block_size = block_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):

        rng = np.random.default_rng(
            self.seed + self.epoch
        )

        self.epoch += 1

        total_rows = len(self.dataset)

        blocks = [
            (start, min(start + self.block_size, total_rows))
            for start in range(0, total_rows, self.block_size)
        ]

        rng.shuffle(blocks)

        for start, end in blocks:

            indices = np.arange(start, end)

            rng.shuffle(indices)

            yield from indices.tolist()


# ============================================================
# 9. CREATE DATASETS AND DATALOADERS
# ============================================================

datasets = {}
dataloaders = {}

for split_name in ["train", "validation", "test"]:

    paths = array_paths[split_name]

    dataset = CloudTypeDataset(
        paths["features"],
        paths["labels"],
    )

    datasets[split_name] = dataset

    if split_name == "train":

        sampler = BlockShuffleSampler(
            dataset,
            block_size=SHUFFLE_BLOCK_SIZE,
            seed=RANDOM_SEED,
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    else:

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    dataloaders[split_name] = loader

    print(
        f"{split_name.upper()} dataset: "
        f"{len(dataset):,} samples"
    )

    print(
        f"{split_name.upper()} batches: "
        f"{len(loader):,}"
    )


# ============================================================
# 10. VERIFY PYTORCH BATCHES
# ============================================================

print("\n" + "=" * 75)
print("STEP 6: VERIFY PYTORCH BATCHES")
print("=" * 75)

batch_results = {}

for split_name, loader in dataloaders.items():

    print(f"\nChecking {split_name.upper()} batch...")

    X_batch, y_batch = next(iter(loader))

    print("Feature batch shape:", tuple(X_batch.shape))
    print("Label batch shape:", tuple(y_batch.shape))

    print("Feature dtype:", X_batch.dtype)
    print("Label dtype:", y_batch.dtype)

    print(
        "Feature values finite:",
        bool(torch.isfinite(X_batch).all()),
    )

    print(
        "Labels in valid range:",
        bool(
            (y_batch >= 0).all()
            and (y_batch < number_of_classes).all()
        ),
    )

    print(
        "Unique labels in batch:",
        torch.unique(y_batch).tolist(),
    )

    if X_batch.shape[1] != len(FEATURES):
        raise RuntimeError("Incorrect feature batch shape.")

    if not torch.isfinite(X_batch).all():
        raise RuntimeError("Non-finite values in batch.")

    if not (
        (y_batch >= 0).all()
        and (y_batch < number_of_classes).all()
    ):
        raise RuntimeError("Invalid target labels.")

    batch_results[split_name] = {
        "feature_shape": list(X_batch.shape),
        "label_shape": list(y_batch.shape),
        "feature_dtype": str(X_batch.dtype),
        "label_dtype": str(y_batch.dtype),
        "features_finite": True,
        "labels_valid": True,
    }

    del X_batch
    del y_batch


# ============================================================
# 11. SAVE PREPARATION METADATA
# ============================================================

metadata = {
    "task": "Seven-class cloud-type classification",
    "features": FEATURES,
    "target": TARGET,
    "class_mapping": CLASS_TO_INDEX,
    "split_rows": split_row_counts,
    "split_class_counts": split_class_counts,
    "class_weights": class_weight_dict,
    "class_weight_method": (
        "Balanced inverse-frequency weighting "
        "using training data only"
    ),
    "feature_array_paths": array_paths,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "shuffle_method": "BlockShuffleSampler for training",
    "shuffle_block_size": SHUFFLE_BLOCK_SIZE,
    "batch_validation": batch_results,
    "device_available": (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    "model_training_started": False,
}

with open(
    OUTPUT_DIR / "tabm_data_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, indent=4)


# ============================================================
# 12. FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("TABM DATA PREPARATION COMPLETED")
print("=" * 75)

print("\nPrepared datasets:")

for split_name, dataset in datasets.items():
    print(f"{split_name:12s}: {len(dataset):,} rows")

print("\nFeature count:", len(FEATURES))
print("Number of classes:", number_of_classes)
print("Batch size:", BATCH_SIZE)

print("\nSaved outputs:")
for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        print(path.name)

print("\nNo model training was performed.")
print("\nAll batch validation checks passed.")

