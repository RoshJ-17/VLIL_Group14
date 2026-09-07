# Enhancing Aviation and Flight Safety Through Near-Real-Time Weather Intelligence

A satellite-driven machine learning pipeline that processes multi-source remote sensing data over India to produce four weather intelligence outputs useful for aviation safety.

**Data coverage:** 01 July 2022 – 30 June 2023  
**Region:** India (6°–38°N, 68°–98°E)

---

## Project Overview

This project fuses four satellite/reanalysis data sources to train four separate ML models:

| Task | Model | Output |
|---|---|---|
| Cloud Base Height (CBH) | TabNet | Regression — CBH in metres |
| Cloud Type Classification | FT-Transformer | 10-class classification |
| Precipitation Detection | TCN | Binary — rain / no rain |
| Rainfall Intensity | Residual MLP + SMOTE | Regression — mm/h |

---

## Data Sources

### NASA MODIS (Terra satellite)
- **Products:** `MOD06_L2` (cloud properties) and `MOD021KM` (1 km radiance)
- **Source:** NASA LAADS DAAC
- **Extracted fields:** Brightness temperatures (BT_11, BT_12), brightness temperature difference (BTD), Cloud Top Temperature (CTT), Cloud Top Height (CTH), Cloud Mask, Cloud Phase, Cloud Optical Thickness
- **Geographic filter:** India bounding box — lat 5–35°N, lon 65–100°E
- **Overpass filter:** India-relevant UTC hours — 03–08 and 15–20

### CALIPSO (CAL_LID_L2_VFM)
- **Used for:** Cloud type ground truth labels
- **Extraction:** Feature_Classification_Flags 16-bit field decoded as:
  - `feature_type = flags & 7` → 2 = cloud
  - `cloud_subtype = (flags >> 9) & 7` → cloud class 0–7
- **10 output classes:** clear, liquid_low/middle/high, ice_low/middle/high, undetermined_low/middle/high

### ERA5 (ECMWF Reanalysis v5)
- **Variable:** `cbh` — Cloud Base Height (metres)
- **Resolution:** Hourly, 0.25° grid
- **Used for:** Ground-truth regression target for CBH prediction
- **Matching:** MODIS pixels matched to ERA5 by nearest-hour rounding and nearest grid-cell index lookup

### GPM (Global Precipitation Measurement)
- **Used for:** Precipitation detection binary labels and rainfall intensity regression target (`GPM_precipitation_rate` in mm/h)
- **Matching:** Spatiotemporally joined to MODIS pixels

---

## Repository Structure

```
VLIL_Group14/
│
├── dataset/                        # MODIS HDF downloader scripts
│   ├── download.py                 # Bearer token authentication
│   ├── noToken.py                  # .netrc authentication
│   └── optimized.py                # Optimized: geo-check on MOD06 only
│
├── extraction/
│   └── cloud_type.py               # CALIPSO VFM bit-field extraction example
│
├── preprocessing/
│   ├── modis/
│   │   ├── 01_modis_extraction.py          # Extract pixel-level fields from HDF
│   │   ├── 02_combine_modis_chunks.py      # Merge chunks into modis_pixel.csv
│   │   └── README.md
│   ├── calipso/
│   │   ├── 05_create_cloud_labels.py       # Decode VFM flags → cloud labels
│   │   ├── 06_create_cloudtype_final.py    # Align CALIPSO labels to MODIS pixels
│   │   ├── 07_split_cloudtype_dataset.py   # Train/val/test split
│   │   └── README.md (see preprocessing/calipso/)
│   ├── era5/
│   │   ├── 03_match_era5_cbh.py            # Join MODIS pixels with ERA5 CBH
│   │   ├── 04_prepare_cbh_v2.py            # Feature engineering + temporal split
│   │   ├── create_cbh_mixed_day_stratified_split.py   # Stratified resplit
│   │   └── README.md
│   └── gpm/
│       ├── 04_match_gpm_precip.py          # Join MODIS with GPM precipitation
│       ├── 06_create_precip_labels.py      # Threshold GPM → binary rain label
│       ├── 08_create_precip_positive_modis_dataset.py  # Filter to rain-positive pixels
│       ├── 12_create_mixed_day_precip_split.py         # Mixed-day stratified split
│       └── README.md
│
├── training/
│   ├── cbh/
│   │   └── train_cbh_tabnet_mixed_stratified.py   # TabNet CBH training
│   ├── cloud_type/
│   │   └── 08_train_cloudtype_fttransformer.py    # FT-Transformer cloud type training
│   ├── precipitation_detection/
│   │   └── 06_train_precip_tcn.py                 # TCN precipitation detection training
│   └── rainfall_intensity/
│       └── 14_train_precip_rate_residual_mlp_v2_smote.py  # Residual MLP + SMOTE training
│
├── data/
│   ├── raw/
│   │   ├── modis/          # Downloaded MOD06_L2 and MOD021KM HDF files
│   │   ├── calipso/        # CALIPSO VFM HDF files
│   │   ├── era5/           # ERA5 NetCDF files
│   │   └── gpm/            # GPM precipitation files
│   └── processed/
│       ├── pixel_level/
│       │   └── modis_pixel.csv                    # All MODIS pixel features (~2.6M rows)
│       ├── cbh/
│       │   ├── cbh_mixed_stratified_train.csv.zip (~686k rows, zipped)
│       │   ├── cbh_mixed_stratified_val.csv       (~85k rows)
│       │   └── cbh_mixed_stratified_test.csv      (~85k rows)
│       ├── cloud_type/
│       │   ├── cloudtype_train.csv
│       │   ├── cloudtype_val.csv
│       │   └── cloudtype_test.csv
│       ├── precipitation_detection/
│       │   ├── modis_gpm_precip.csv
│       │   └── modis_gpm_precip_detection.csv
│       └── rainfall_intensity/
│           ├── precip_positive_modis_mixed_train.csv (~99.8k rows)
│           ├── precip_positive_modis_mixed_val.csv   (~21.4k rows)
│           └── precip_positive_modis_mixed_test.csv  (~21.4k rows)
│
├── models/
│   ├── cbh/
│   │   └── cbh_tabnet_mixed_stratified_model.zip
│   ├── cloud_type/
│   │   └── best_model.pt
│   ├── precipitation_detection/
│   │   └── precipitation_tcn.pt
│   └── rainfall_intensity/
│       └── precip_rate_residual_mlp_v2_smote.pt
│
├── evaluation/
│   ├── cbh/
│   │   ├── cbh_tabnet_mixed_stratified_metrics.json
│   │   └── cbh_tabnet_mixed_stratified_feature_importance.csv
│   ├── cloud_type/
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.csv
│   │   └── config.json
│   ├── precipitation_detection/
│   │   ├── precipitation_tcn_threshold_results.csv
│   │   ├── precipitation_tcn_fine_threshold_results.csv
│   │   └── precipitation_tcn_test_predictions.csv
│   └── rainfall_intensity/
│       ├── precip_rate_residual_mlp_v2_smote_intensity_evaluation.csv
│       └── precip_rate_residual_mlp_v2_smote_test_predictions.csv
│
├── docs/
│   ├── architecture/
│   ├── methodology/
│   └── results/
│
├── requirements.txt
└── README.md
```

---

## Pipeline

### Step 1 — Download MODIS Data (`dataset/`)

Three downloader variants are provided. All share the same logic:

1. Scrape the [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) HTML listing for `.hdf` URLs per day
2. Match `MOD06_L2` and `MOD021KM` files by their embedded timestamp (year + DOY + HHMM)
3. Filter to India overpass hours (03–08 UTC and 15–20 UTC)
4. Download to `.tmp`, verify India geographic coverage, rename or delete

**Which script to use:**

| Script | Auth method | Notes |
|---|---|---|
| `download.py` | Bearer token in header | Set `TOKEN` variable |
| `noToken.py` | `~/.netrc` (NASA Earthdata) | Most convenient |
| `optimized.py` | `~/.netrc` | Skips MOD021KM download if MOD06 fails India geo-check — saves bandwidth |

For NASA Earthdata credentials, register at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) and add to `~/.netrc`:
```
machine urs.earthdata.nasa.gov login <username> password <password>
```

### Step 2 — MODIS Extraction (`preprocessing/modis/`)

- `01_modis_extraction.py` — reads each HDF pair, extracts per-pixel fields, saves chunked CSVs
- `02_combine_modis_chunks.py` — merges all chunks into `data/processed/pixel_level/modis_pixel.csv`

Output columns: `timestamp, lat, lon, BT_11, BT_12, BTD, CTT, CTH, Cloud_Mask_Raw, Cloud_Mask_Binary, Cloud_Phase, Cloud_Optical_Thickness`

### Step 3 — CALIPSO Extraction (`extraction/`, `preprocessing/calipso/`)

The core bit-field decoding logic is demonstrated in `extraction/cloud_type.py`:
```python
feature_type  = flags & 7          # bits 0-2 → 2 = cloud
cloud_subtype = (flags >> 9) & 7   # bits 9-11 → 0-7 cloud type
```

The pipeline scripts (`05`, `06`, `07`) process all CALIPSO VFM files, align them to MODIS pixels spatiotemporally, and produce the cloud type labelled dataset.

### Step 4 — ERA5 Matching (`preprocessing/era5/`)

**`03_match_era5_cbh.py`**
- Loads ERA5 NetCDF (`cbh` variable)
- Processes `modis_pixel.csv` in 250,000-row chunks
- Rounds MODIS timestamp to nearest ERA5 hour
- Computes grid indices: `round((value - origin) / step)`, clipped to bounds
- Direct array lookup: `cbh[time_idx, lat_idx, lon_idx]`
- Outputs `modis_era5_cbh.csv`

**`04_prepare_cbh_v2.py`**
- Filters: valid ERA5_CBH, cloudy pixels only (`Cloud_Mask_Binary == 1`), non-null CTT/CTH/Cloud_Phase
- Derives time features: `hour` (float), `day_of_year`
- Encodes Cloud Phase: `{0→0 (clear), 1→1 (liquid), 2→2 (ice), 6→3 (undetermined)}`
- Target transformation: `target_log_cbh = log1p(ERA5_CBH)`
- Temporal split by date: Train = Nov 19–22, Val = Nov 23, Test = Nov 24, 2022
- Fits `StandardScaler` on train only; transforms val/test

**`create_cbh_mixed_day_stratified_split.py`**
- Recombines all 3 splits
- Stratifies by compound label: `date + CBH range bin` (7 bins: 0–250m, 250–500m, 500m–1km, 1–2km, 2–5km, 5–10km, >10km)
- Two-stage split: 80% train / 10% val / 10% test
- Outputs `cbh_mixed_stratified_{train,val,test}.csv`

### Step 5 — GPM Matching (`preprocessing/gpm/`)

- `04_match_gpm_precip.py` — spatiotemporal join of MODIS pixels with GPM precipitation rates
- `06_create_precip_labels.py` — thresholds GPM rate → binary `precipitation_flag`
- `08_create_precip_positive_modis_dataset.py` — filters to precipitation-positive pixels only (for intensity model)
- `12_create_mixed_day_precip_split.py` — mixed-day stratified split

### Step 6 — Model Training (`training/`)

#### Cloud Base Height — TabNet (`train_cbh_tabnet_mixed_stratified.py`)

- **Features (9):** BT_11, BT_12, BTD, lat, lon, CTT, CTH, Cloud_Mask_Raw, Cloud_Phase
- **Target:** `ERA5_CBH` (metres, raw)
- **Architecture:** TabNet with attention masking (`entmax`)
  - `n_d=32, n_a=32, n_steps=5, gamma=1.5`
  - `n_independent=2, n_shared=2, lambda_sparse=1e-4`
- **Training:** Adam lr=0.02, batch=8192, virtual_batch=1024, max_epochs=200, patience=20
- **Outputs:** model `.zip`, test predictions, metrics JSON, feature importance CSV, per-range error analysis

Run:
```bash
python training/cbh/train_cbh_tabnet_mixed_stratified.py
```

#### Cloud Type — FT-Transformer (`08_train_cloudtype_fttransformer.py`)

- **Two modes:**
  - `base` — 7 MODIS features (lat, lon, BT_11, BT_12, BTD, Cloud_Mask_Binary, Cloud_Mask_Raw)
  - `extended` — base + CTH + CTT + Cloud_Phase (categorical)
- **Target:** 10-class cloud type label
- **Architecture:** FT-Transformer (`tab-transformer-pytorch`)
  - `dim=64, depth=4, heads=8, dim_head=16, attn_dropout=0.1, ff_dropout=0.1`
- **Training:** AdamW lr=1e-4, ReduceLROnPlateau, batch=4096, max_epochs=30, patience=5
- **Class imbalance:** Balanced class weights (capped at 10×), applied to CrossEntropyLoss
- **Best model:** saved by validation macro-F1
- **Outputs:** `best_model.pt`, `classification_report.txt`, `confusion_matrix.csv`, `config.json`, `training_history.csv`

Run:
```bash
python training/cloud_type/08_train_cloudtype_fttransformer.py --mode extended
```

#### Precipitation Detection — TCN (`06_train_precip_tcn.py`)

- **Features (11):** lat, lon, BT_11, BT_12, BTD, Cloud_Mask_Binary, Cloud_Mask_Raw, CTT, CTH, Cloud_Phase, Cloud_Optical_Thickness
- **Target:** `precipitation_flag` (binary 0/1)
- **Architecture:** Temporal Convolutional Network
  - 3 dilated residual blocks, channels `(64, 64, 128)`, kernel_size=3, dilation `2^i`
  - Causal convolutions via `Chomp1d`, residual connections
  - Sequence length: 12 consecutive observations
- **Training:** AdamW lr=1e-3, BCEWithLogitsLoss with positive class weight, max_epochs=30
- **Class imbalance:** `pos_weight = n_negative / n_positive` passed to BCE loss
- **Best model:** saved by validation F1
- **Outputs:** `precipitation_tcn.pt`, test predictions CSV

Run:
```bash
python training/precipitation_detection/06_train_precip_tcn.py
```

#### Rainfall Intensity — Residual MLP + SMOTE (`14_train_precip_rate_residual_mlp_v2_smote.py`)

- **Features (10):** lat, lon, BT_11, BT_12, BTD, Cloud_Mask_Binary, Cloud_Mask_Raw, CTT, CTH, Cloud_Phase
- **Target:** `GPM_precipitation_rate` (mm/h, positive-only pixels)
- **Architecture:** Residual MLP with dual-head output
  - Input layer → 5 residual blocks (Linear + BatchNorm + ReLU + Dropout, dim=256)
  - **Regression head:** predicts `log1p(precipitation_rate)`
  - **Classification head:** predicts intensity bin (5 classes: <1, 1–5, 5–10, 10–20, >20 mm/h)
- **Loss:** Intensity-weighted Huber loss (log1p space) + 0.2 × auxiliary cross-entropy classification loss
- **Class imbalance:** Custom SMOTE oversampling applied only to training data
  - Synthetic samples generated by feature-space interpolation between k-nearest neighbours
  - Precipitation rate also interpolated in log1p space for consistency
  - Target synthetic counts: {<1: 67274, 1–5: 26781, 5–10: 8000, 10–20: 4000, >20: 1600}
- **Training:** AdamW lr=5e-4, ReduceLROnPlateau, batch=2048, max_epochs=100, patience=15
- **Outputs:** `precip_rate_residual_mlp_v2_smote.pt`, test predictions CSV, intensity evaluation CSV

Run:
```bash
python training/rainfall_intensity/14_train_precip_rate_residual_mlp_v2_smote.py
```

---

## Features Summary

| Feature | Description | Used in |
|---|---|---|
| `BT_11` | 11 µm brightness temperature | All models |
| `BT_12` | 12 µm brightness temperature | All models |
| `BTD` | BT_11 – BT_12 split-window difference | All models |
| `CTT` | Cloud Top Temperature | CBH, Cloud Type, Precip |
| `CTH` | Cloud Top Height | CBH, Cloud Type, Precip |
| `Cloud_Mask_Binary` | 0 = clear, 1 = cloudy | All models |
| `Cloud_Mask_Raw` | Raw mask confidence value | CBH, Cloud Type |
| `Cloud_Phase` | 1=liquid, 2=ice, 3=undetermined | All models |
| `Cloud_Optical_Thickness` | COT | Precipitation Detection |
| `lat`, `lon` | Pixel geolocation | All models |
| `hour`, `day_of_year` | Time-of-day and seasonal features | CBH |

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the core stack. Additional packages needed:

```bash
pip install pyhdf           # HDF4 reading (MODIS + CALIPSO)
pip install beautifulsoup4  # HTML scraping for LAADS DAAC downloader
pip install xarray netCDF4  # ERA5 NetCDF reading
pip install joblib          # Scaler serialization
pip install pytorch-tabnet  # TabNet model
pip install tab-transformer-pytorch  # FT-Transformer model
pip install imbalanced-learn         # SMOTE (sklearn-compatible)
pip install git-lfs                  # Required to pull large data files
```

For HDF4 support on macOS, install the HDF4 C library first:
```bash
brew install hdf4
pip install pyhdf
```

### GPU / CPU

All training scripts auto-detect CUDA. On CPU they run correctly but will be significantly slower, especially for cloud type (FT-Transformer) and CBH (TabNet) which have large datasets (~600k+ rows).

### NASA Earthdata Access

Data download requires a free NASA Earthdata account. Configure credentials in `~/.netrc`:
```
machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
```

---

## Running the Full Pipeline

Run scripts in numbered order:

```bash
# 1. Download MODIS HDF files
python dataset/optimized.py

# 2. Extract pixel-level features
python preprocessing/modis/01_modis_extraction.py
python preprocessing/modis/02_combine_modis_chunks.py

# 3. Match with ERA5 and prepare CBH dataset
python preprocessing/era5/03_match_era5_cbh.py
python preprocessing/era5/04_prepare_cbh_v2.py
python preprocessing/era5/create_cbh_mixed_day_stratified_split.py

# 4. CALIPSO cloud type labels
python preprocessing/calipso/05_create_cloud_labels.py
python preprocessing/calipso/06_create_cloudtype_final.py
python preprocessing/calipso/07_split_cloudtype_dataset.py

# 5. GPM precipitation matching and labelling
python preprocessing/gpm/04_match_gpm_precip.py
python preprocessing/gpm/06_create_precip_labels.py
python preprocessing/gpm/08_create_precip_positive_modis_dataset.py
python preprocessing/gpm/12_create_mixed_day_precip_split.py

# 6. Train models
python training/cbh/train_cbh_tabnet_mixed_stratified.py
python training/cloud_type/08_train_cloudtype_fttransformer.py --mode extended
python training/precipitation_detection/06_train_precip_tcn.py
python training/rainfall_intensity/14_train_precip_rate_residual_mlp_v2_smote.py
```

---

## Notes on Large Files

`data/processed/pixel_level/modis_pixel.csv` is ~181 MB and exceeds GitHub's 100 MB file size limit. It is tracked via **Git LFS**. To pull it after cloning:

```bash
git lfs install
git lfs pull
```

The CBH training split (`cbh_mixed_stratified_train.csv.zip`) is stored as a zip archive (~51 MB compressed). Unzip before training:
```bash
unzip data/processed/cbh/cbh_mixed_stratified_train.csv.zip -d data/processed/cbh/
```

---

## Evaluation Outputs

| Task | Metrics |
|---|---|
| CBH | MAE, RMSE, R², Bias — overall and per CBH range bin |
| Cloud Type | Precision, Recall, F1 per class; Balanced Accuracy; Confusion Matrix |
| Precipitation Detection | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC; threshold sweep results |
| Rainfall Intensity | MAE, RMSE, R², Correlation, Bias — overall and per intensity bin (<1, 1–5, 5–10, 10–20, >20 mm/h) |

All evaluation outputs are saved under `evaluation/`.
