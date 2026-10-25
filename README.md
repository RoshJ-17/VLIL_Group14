# A Multisource Satellite-Based Deep Learning Framework for Aviation Weather Monitoring and Decision Support

## Overview

This project develops a multisource satellite-based deep learning framework for aviation weather monitoring over India. It integrates satellite observations, meteorological reanalysis data, precipitation products, and cloud-type reference labels to estimate weather conditions that may affect aviation operations.

The framework covers four weather-estimation tasks: cloud-type classification, cloud-base-height estimation, precipitation detection, and rainfall-intensity estimation. The resulting outputs provide a foundation for weather monitoring and aviation weather decision support.

**Study region:** India  
**Study period:** July 1, 2022, to June 30, 2023

## Features Summary

### 1. Multisource Weather Data Integration
- Integrates MODIS satellite observations, ERA5 reanalysis data, GPM IMERG precipitation products, and CALIPSO-derived cloud-type labels.
- Organizes raw and processed datasets for downstream model development.
- Supports the preparation of task-specific training, validation, and testing datasets.

### 2. MODIS Data Processing
- Extracts relevant information from MODIS MOD021KM and MOD06_L2 products.
- Supports combining extracted MODIS chunks.
- Provides MODIS-derived features for the integrated weather-estimation workflow.

### 3. Cloud-Type Classification
- Trains a deep learning model for cloud-type classification using the prepared cloud-type dataset.
- Uses CALIPSO-derived cloud-type labels as reference targets.
- Includes model artifacts and classification evaluation outputs.

### 4. Cloud-Base-Height Estimation
- Uses ERA5 cloud-base-height data as the reference for model preparation and evaluation.
- Includes scripts for matching ERA5 data and preparing cloud-base-height datasets.
- Provides a trained model and evaluation outputs for cloud-base-height estimation.

### 5. Precipitation Detection
- Prepares precipitation-detection datasets using GPM-related processing workflows.
- Includes a training workflow for precipitation detection.
- Stores the trained model and evaluation results.

### 6. Rainfall-Intensity Estimation
- Prepares rainfall-intensity datasets for regression.
- Includes a training workflow for rainfall-rate prediction.
- Stores the trained model and evaluation outputs.

### 7. Model Evaluation and Results
- Organizes evaluation artifacts by weather-estimation task.
- Includes metrics, classification reports, confusion matrices, feature-importance results, and other task-specific output files where available.
- Separates trained model artifacts from evaluation outputs.

### 8. Aviation Weather Decision Support
- Establishes a foundation for combining cloud, cloud-base-height, precipitation, and rainfall-intensity estimates.
- Can support future integration of regional cloud coverage, temporal weather trends, hazard persistence, and rule-based risk indicators.
- Provides a basis for adding model-explanation methods such as SHAP.

> This repository is intended for research and decision-support development. It is not a certified aviation weather forecasting or operational air-traffic-control system.

## Repository Structure

```text
VLIL_Group14/
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── modis/
│   │       └── MOD021KM / MOD06_L2 ZIP and chunk files
│   │
│   └── processed/
│       ├── cbh/
│       │   └── train / val / test CSVs
│       ├── cloud_type/
│       │   └── train / val / test CSVs
│       ├── pixel_level/
│       │   └── modis_pixel.csv
│       ├── precipitation_detection/
│       │   └── processed CSV files
│       └── rainfall_intensity/
│           └── train / val / test CSVs
│
├── dataset/
│   ├── download.py
│   ├── noToken.py
│   └── optimized.py
│
├── preprocessing/
│   ├── calipso/
│   │   ├── 05_create_cloud_labels.py
│   │   ├── 06_create_cloudtype_final.py
│   │   └── 07_split_cloudtype_dataset.py
│   │
│   ├── era5/
│   │   ├── 03_match_era5_cbh.py
│   │   ├── 04_prepare_cbh_v2.py
│   │   └── create_cbh_mixed_day_stratified_split.py
│   │
│   └── modis/
│       ├── 01_modis_extraction.py
│       └── 02_combine_modis_chunks.py
│
├── scripts/
│   ├── person1/
│   │   ├── 04_download_era5_cbh.py
│   │   ├── 06_download_era5_cbh_team_netcdf_person1.py
│   │   ├── 06_verify_gpm_processed.py
│   │   └── check_modis_dates.py
│   │
│   ├── person2/
│   │   ├── 05_build_era5_required_cells.py
│   │   ├── 05_download_era5_cbh_targeted.py
│   │   ├── 05_download_extract_gpm.py
│   │   ├── 06_download_era5_cbh_team_netcdf_person5.py
│   │   ├── check_era5_requirements.py
│   │   ├── era5_required_cells_temp.sqlite
│   │   └── nano.save
│   │
│   ├── person3/
│   │   ├── pairs.txt
│   │   ├── 06_download_era5_cbh_team_netcdf_person2.py
│   │   ├── 06_download_era5_cbh_team_netcdf_person4.py
│   │   ├── 06_download_era5_cbh_team_netcdf_person6.py
│   │   ├── check_values1709.py
│   │   ├── match1509.py
│   │   └── run_modis.pbs
│   │
│   └── person4/
│       ├── 05_create_cloud_labels2.py
│       ├── 06_create_cloudtype_final2.py
│       ├── 06_download_era5_cbh_team_netcdf_person3.py
│       ├── 07_match_modis_gpm.py
│       ├── era5_download.log
│       └── make_gpm_required_times.py
│
├── training/
│   ├── cbh/
│   │   └── train_cbh_tabnet_mixed_stratified.py
│   ├── cloud_type/
│   │   └── 08_train_cloudtype_fttransformer.py
│   ├── precipitation_detection/
│   │   └── 06_train_precip_tcn.py
│   └── rainfall_intensity/
│       └── 14_train_precip_rate_residual_mlp_v2_smote.py
│
├── evaluation/
│   ├── cbh/
│   │   ├── feature_importance.csv
│   │   └── metrics.json
│   ├── cloud_type/
│   │   ├── classification_report.txt
│   │   ├── config.json
│   │   └── confusion_matrix.csv
│   ├── precipitation_detection/
│   │   └── evaluation result CSV files
│   └── rainfall_intensity/
│       └── evaluation result CSV files
│
└── models/
    ├── cbh/
    │   └── cbh_tabnet_mixed_stratified_model.zip
    ├── cloud_type/
    │   └── best_model.pt
    ├── precipitation_detection/
    │   └── precipitation_tcn.pt
    └── rainfall_intensity/
        └── precip_rate_residual_mlp_v2_smote.pt
```

## Data Sources

| Data source | Role in the project |
|---|---|
| MODIS MOD021KM | Level-1B radiance observations |
| MODIS MOD06_L2 | Cloud-related satellite products |
| ERA5 | Meteorological reanalysis, including cloud-base-height reference data |
| GPM IMERG | Precipitation information for detection and rainfall-intensity estimation |
| CALIPSO-derived labels | Reference labels for cloud-type classification |

The source datasets differ in format, resolution, and temporal coverage. Preprocessing scripts are used to extract, match, and prepare data for the downstream tasks.

## Models and Training Workflows

| Task | Training script | Model artifact |
|---|---|---|
| Cloud-base-height estimation | `training/cbh/train_cbh_tabnet_mixed_stratified.py` | `models/cbh/cbh_tabnet_mixed_stratified_model.zip` |
| Cloud-type classification | `training/cloud_type/08_train_cloudtype_fttransformer.py` | `models/cloud_type/best_model.pt` |
| Precipitation detection | `training/precipitation_detection/06_train_precip_tcn.py` | `models/precipitation_detection/precipitation_tcn.pt` |
| Rainfall-intensity estimation | `training/rainfall_intensity/14_train_precip_rate_residual_mlp_v2_smote.py` | `models/rainfall_intensity/precip_rate_residual_mlp_v2_smote.pt` |

The script and artifact names above describe the repository layout. The actual architecture, training options, input features, and target definitions should be verified in the corresponding source files.

## Installation and Setup

### Prerequisites

- Python version compatible with `requirements.txt`
- Git
- `pip`
- Git LFS, if required for tracked large files
- macOS or Linux
- Optional GPU support for accelerated training

### 1. Clone the Repository

```bash
git clone https://github.com/RoshJ-17/VLIL_Group14.git
cd VLIL_Group14
```

### 2. Create and Activate a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv aviation
source aviation/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

Additional packages may be required depending on the scripts you intend to run:

```bash
pip install pyhdf beautifulsoup4 xarray netCDF4 joblib
pip install pytorch-tabnet tab-transformer-pytorch imbalanced-learn
```

Check `requirements.txt` before installing these separately to avoid unnecessary duplicate installations.

### 4. Git LFS

If the repository uses Git LFS to track large files, install and initialize it.

**macOS:**

```bash
brew install git-lfs
git lfs install
```

**Linux:** Install Git LFS using the package manager or installation instructions for your distribution, then run:

```bash
git lfs install
```

### 5. HDF4 Support on macOS

The `pyhdf` package requires the HDF4 library. On macOS, install HDF4 first:

```bash
brew install hdf4
pip install pyhdf
```

If installation fails, check that your compiler can locate the HDF4 headers and libraries. The exact configuration may vary by Mac architecture and Python version.

### 6. CPU and GPU Support

The scripts may run on a CPU, although training can take longer for large datasets.

For NVIDIA GPU acceleration, install a PyTorch version compatible with your CUDA environment using the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Check CUDA availability:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

On supported Apple Silicon Macs, check whether the PyTorch MPS backend is available:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

GPU support depends on the hardware, installed libraries, and device handling implemented in each training script.

## NASA Earthdata Authentication

Some satellite-data downloads may require a NASA Earthdata account.

1. Register at [NASA Earthdata Login](https://urs.earthdata.nasa.gov/).
2. Configure authentication according to the requirements of the download script.
3. If the script uses a `.netrc` file, create or edit `~/.netrc` in your home directory.

Example:

```text
machine urs.earthdata.nasa.gov
login YOUR_EARTHDATA_USERNAME
password YOUR_EARTHDATA_PASSWORD
```

Set restrictive permissions:

```bash
chmod 600 ~/.netrc
```

Replace the placeholders with your own credentials. Never commit `.netrc`, passwords, or access tokens to GitHub. Follow the current NASA Earthdata instructions if a particular service requires a different authentication method.

## Dataset Preparation

The repository separates raw data from processed data.

- `data/raw/` is intended for source datasets organized by data source.
- `data/processed/` contains task-specific prepared datasets.
- `dataset/` contains general dataset download utilities.
- `preprocessing/` contains source-specific extraction, matching, label creation, and dataset preparation scripts.
- `scripts/` contains supporting data-acquisition, verification, matching, and team workflow utilities.

The general preparation workflow is:

1. Acquire the required source data.
2. Extract relevant observations and variables.
3. Match observations across the required spatial and temporal dimensions.
4. Generate labels and prepare task-specific features.
5. Split datasets into training, validation, and testing subsets where applicable.
6. Save the processed data in the expected locations.
7. Verify the resulting files before training.

Refer to the README files within the relevant preprocessing directories for additional instructions.

## Running Training Scripts

Run a training script from the repository root after installing its dependencies and preparing its expected input data.

For example, the cloud-base-height training script is located at:

```bash
python training/cbh/train_cbh_tabnet_mixed_stratified.py
```

Other training scripts are located at:

```text
training/cloud_type/08_train_cloudtype_fttransformer.py
training/precipitation_detection/06_train_precip_tcn.py
training/rainfall_intensity/14_train_precip_rate_residual_mlp_v2_smote.py
```

Before running a script, inspect its configuration and confirm that the input paths, output paths, required arguments, and device settings match your environment. The command above assumes the script can be launched directly from the repository root without additional command-line arguments.

## Evaluation and Results

Evaluation artifacts are organized by task in `evaluation/`.

Available artifact types include:

- Cloud-base-height metrics and feature-importance results.
- Cloud-type classification reports, configuration information, and confusion matrices.
- Precipitation-detection evaluation CSV files.
- Rainfall-intensity evaluation CSV files.

The metrics and outputs generated by each evaluation workflow should be interpreted according to that task's target definition and test-data split.

## Aviation Weather Decision Support

The project provides a foundation for combining weather-estimation outputs into a regional picture of atmospheric conditions. Possible extensions include tracking changes over time, calculating regional cloud coverage, assessing the persistence of potentially hazardous conditions, and applying transparent rule-based risk indicators.

Model-explanation techniques such as SHAP may be incorporated to help interpret model predictions. Any decision-support outputs require appropriate validation before they can be considered for operational aviation use.

## HPC and Remote Linux Execution

Large datasets and computationally intensive experiments may be run on an institutional high-performance computing (HPC) system.

A typical workflow is:

1. Connect to the authorized remote server using SSH.
2. Activate the project environment.
3. Verify the data paths and installed dependencies.
4. Run the required preprocessing or training scripts.
5. Save logs, checkpoints, and evaluation outputs.
6. Confirm that the jobs completed and the output files are valid.

Use the institution's job scheduler where required. For long-running jobs, check whether an existing process is active before starting another copy.

## Reproducibility

For reproducible experiments:

- Record Python and package versions.
- Document source datasets and preprocessing settings.
- Keep training, validation, and testing splits consistent.
- Record random seeds where supported.
- Save model configurations, checkpoints, and evaluation metrics.
- Document the hardware and compute device used.
- Keep credentials and restricted data out of the repository.

Large datasets may be better hosted in an appropriate data repository rather than directly in GitHub. Follow the applicable data-use and licensing terms.

## Limitations

- Results depend on the quality and representativeness of the source and reference data.
- Differences in spatial and temporal resolution can affect data alignment.
- Missing observations, class imbalance, and preprocessing errors may affect model performance.
- Research predictions should not be treated as certified aviation forecasts.
- Operational use would require further validation, uncertainty assessment, and domain-specific review.

## Acknowledgments

This project uses data and scientific resources associated with NASA MODIS, ERA5, GPM IMERG, and CALIPSO-related products. Please consult and cite the relevant dataset documentation and research publications when using these resources.

## License

See the repository's `LICENSE` file for the applicable license. Ensure that the use and redistribution of datasets and third-party software also comply with their respective terms.

## Contact

For questions, bug reports, or collaboration, please open an issue in the [GitHub repository](https://github.com/RoshJ-17/VLIL_Group14).
