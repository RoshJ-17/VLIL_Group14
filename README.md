# A Multisource Satellite-Based Deep Learning Framework for Aviation Weather Monitoring and Decision Support

## Overview

This project develops a multisource satellite-based deep learning framework for aviation weather monitoring over India. It combines satellite observations, meteorological reanalysis data, precipitation products, and cloud-type reference labels to estimate weather conditions that may affect aviation operations.

The framework focuses on four key weather-estimation tasks: cloud-type classification, cloud-base-height estimation, precipitation detection, and rainfall-rate estimation. These outputs are intended to support weather monitoring and provide useful information for aviation weather assessment.

**Study region:** India  
**Study period:** July 1, 2022, to June 30, 2023

## Features Summary

### 1. Multisource Weather Data Integration
- Integrates data from multiple meteorological and satellite sources.
- Uses MODIS observations, ERA5 reanalysis data, GPM IMERG precipitation products, and CALIPSO-derived cloud-type labels.
- Supports the development of a unified dataset for multiple weather-estimation tasks.

### 2. Cloud Detection and Cloud Coverage Estimation
- Supports cloud identification using available satellite-derived features.
- Calculates cloud coverage from the available cloud-related data and classifications.
- Provides regional cloud information that can complement the individual weather-estimation outputs.

### 3. Cloud-Type Classification
- Uses the **TabM** deep learning model to classify cloud types.
- Uses CALIPSO-derived labels as reference targets for model development and evaluation.
- Helps characterize cloud conditions relevant to weather monitoring.

### 4. Cloud-Base-Height Estimation
- Uses **TabM** to estimate cloud-base height.
- Uses ERA5 cloud-base-height data as the reference for model training and evaluation.
- Produces continuous height estimates for analyzing cloud conditions.

### 5. Precipitation Detection
- Uses **Oya-U-Net** for precipitation detection.
- Formulates precipitation detection as a binary classification task using labels derived from GPM IMERG data.
- Identifies whether precipitation is present in the evaluated samples.

### 6. Rainfall-Rate Estimation
- Uses **Oya-QPE** to estimate rainfall rate.
- Uses GPM IMERG precipitation data as the reference source.
- Treats rainfall-rate estimation as a regression task to quantify precipitation intensity.

### 7. Aviation Weather Decision Support
- Brings together cloud, cloud-base-height, precipitation, and rainfall-rate estimates.
- Provides a foundation for assessing weather conditions that may be relevant to aviation.
- Supports the planned integration of regional cloud coverage, temporal weather trends, hazard persistence, and rule-based risk assessment.
- Can be extended with model-explanation methods such as SHAP to help interpret predictions.

> The framework is intended for research and decision-support development. It is not a certified aviation weather forecasting or operational air-traffic-control system.

## Data Sources

The project uses the following data sources:

| Data source | Purpose |
|---|---|
| MODIS MOD06_L2 | Cloud-related observations and products |
| MODIS MOD021KM | Moderate Resolution Imaging Spectroradiometer Level-1B radiance observations |
| ERA5 | Meteorological reanalysis data, including cloud-base height |
| GPM IMERG | Precipitation information used for precipitation detection and rainfall-rate estimation |
| CALIPSO-derived labels | Reference labels for cloud-type classification |

The availability, spatial and temporal alignment, and quality of the source data affect the final integrated dataset and model performance.

## Models and Estimation Tasks

| Task | Model | Reference data | Task type |
|---|---|---|---|
| Cloud-type classification | TabM | CALIPSO-derived cloud-type labels | Classification |
| Cloud-base-height estimation | TabM | ERA5 cloud-base height | Regression |
| Precipitation detection | Oya-U-Net | GPM IMERG-derived precipitation labels | Binary classification |
| Rainfall-rate estimation | Oya-QPE | GPM IMERG rainfall rate | Regression |

## Installation and Setup

### Prerequisites

- Python 3.10 or a version compatible with the project's dependencies
- Git
- `pip`
- Git Large File Storage (Git LFS), if the repository uses it
- A compatible operating system: macOS or Linux
- Optional GPU support for accelerated training

Clone the repository and move into its directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Replace the placeholders with your actual GitHub repository URL and repository directory name.

### 1. Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv aviation
source aviation/bin/activate
```

Upgrade the package installer:

```bash
python -m pip install --upgrade pip
```

### 2. Install the Requirements

If the repository contains a `requirements.txt` file, install its dependencies:

```bash
pip install -r requirements.txt
```

### 3. Install Additional Packages

Depending on which scripts you run, the project may require the following packages:

```bash
pip install pyhdf beautifulsoup4 xarray netCDF4 joblib
pip install pytorch-tabnet tab-transformer-pytorch imbalanced-learn
```

Install Git LFS if it is used to track files in the repository.

**macOS:**

```bash
brew install git-lfs
git lfs install
```

**Linux:** Install Git LFS using the package manager or installation instructions for your distribution, then run:

```bash
git lfs install
```

### 4. HDF4 Support for `pyhdf` on macOS

The `pyhdf` package depends on the HDF4 library. On macOS, install HDF4 first:

```bash
brew install hdf4
```

Then try installing the Python package:

```bash
pip install pyhdf
```

If installation fails, check that the HDF4 headers and libraries are discoverable by your compiler and Python environment. The exact setup may vary by Mac architecture and Python version.

### 5. CPU and GPU Support

The models can be run on a CPU, although training may take longer for large datasets.

For NVIDIA GPU acceleration on Linux, install a PyTorch version compatible with your CUDA environment. Follow the official PyTorch installation selector to choose the appropriate command.

Verify the PyTorch installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

On supported Apple Silicon Macs, PyTorch may also use the Metal Performance Shaders (MPS) backend. Check availability with:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

GPU availability depends on the hardware, installed libraries, and whether the training scripts support the selected device.

## NASA Earthdata Authentication

Some satellite datasets may require a NASA Earthdata account and authentication.

1. Create an account through the [NASA Earthdata Login](https://urs.earthdata.nasa.gov/) website.
2. Configure your credentials using the authentication method supported by the data-download scripts.
3. If the scripts use a `.netrc` file, create it in your home directory.

Example `.netrc` entry:

```text
machine urs.earthdata.nasa.gov
login YOUR_EARTHDATA_USERNAME
password YOUR_EARTHDATA_PASSWORD
```

Set restrictive file permissions:

```bash
chmod 600 ~/.netrc
```

Replace the placeholders with your own credentials. **Never commit `.netrc`, passwords, access tokens, or other secrets to GitHub.** Follow NASA Earthdata's current authentication guidance if a data service requires a different method.

## Dataset Preparation

The project uses multiple data sources that may differ in spatial resolution, temporal frequency, coordinate conventions, and file format.

The preparation workflow may include:

1. Acquiring the required satellite and meteorological data.
2. Extracting relevant variables and features from the source files.
3. Processing HDF and NetCDF data as required.
4. Aligning observations across the required spatial and temporal dimensions.
5. Preparing cloud-type, cloud-base-height, precipitation, and rainfall-rate targets.
6. Handling missing or invalid observations.
7. Saving the integrated dataset in the format expected by the model scripts.

The exact commands and file paths depend on the data-processing scripts available in the repository. Use the paths and options defined in those scripts rather than assuming that every stage can be run with one command.

## Running the Models

Run the individual scripts for the required task from the repository's working directory.

The exact script names and command-line arguments should match the files committed to the repository. Before starting a long training run, check each script for its input paths, configuration parameters, output directory, and device settings.

The four model workflows are:

- **TabM:** Cloud-type classification
- **TabM:** Cloud-base-height estimation
- **Oya-U-Net:** Precipitation detection
- **Oya-QPE:** Rainfall-rate estimation

Use the corresponding configuration and preprocessing steps for each task. Do not assume that the same input columns or target format are shared across all models.

## Evaluation Metrics

The project evaluates its models using metrics suited to each task.

| Task | Metrics |
|---|---|
| Cloud-type classification | Accuracy, macro precision, macro recall, macro F1-score, weighted F1-score |
| Cloud-base-height estimation | Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), coefficient of determination (R²) |
| Precipitation detection | Precision, recall / Probability of Detection (POD), F1-score |
| Rainfall-rate estimation | MAE, RMSE, R² |

For classification tasks, macro-averaged metrics help assess performance across classes, while weighted metrics account for class frequencies. For regression tasks, MAE and RMSE measure prediction error, and R² summarizes the variance explained by the model.

## Aviation Weather Decision Support

The project aims to combine the model outputs into a broader view of regional weather conditions. Potential decision-support components include regional cloud coverage, changes in weather estimates over time, the duration of potentially hazardous conditions, and rule-based risk indicators.

Model-explanation techniques such as SHAP may help identify which input features influence predictions. Any risk indicators or reports produced by such a system should be treated as research outputs and validated before operational use.

## Working on an HPC or Remote Linux Server

For large datasets and computationally intensive training, the workflow can be run on a remote Linux server or high-performance computing (HPC) environment.

A typical workflow is:

1. Connect to the authorized server using SSH.
2. Activate the project's Python environment.
3. Confirm that the required data files and dependencies are available.
4. Run preprocessing or training scripts from the appropriate working directory.
5. Save logs, model checkpoints, and evaluation results.
6. Verify that the output files were generated successfully.

For long-running experiments, use the job scheduler provided by your institution if required. Avoid starting duplicate training or data-transfer jobs without checking whether an existing job is already running.

## Reproducibility

For reproducible experiments:

- Record the Python and package versions used.
- Document the source datasets and preprocessing settings.
- Keep training, validation, and test splits consistent.
- Record random seeds where supported.
- Save model configurations, checkpoints, and evaluation metrics.
- Document hardware and device settings.
- Avoid publishing credentials or restricted data.

Large datasets may not be appropriate for direct storage in GitHub. Use an approved data repository or other suitable hosting service and provide access instructions where permitted.

## Limitations

- Model performance depends on the quality and representativeness of the reference data.
- Satellite and reanalysis products may differ in spatial and temporal resolution.
- Results can be affected by missing observations, class imbalance, and alignment errors.
- Research-based weather estimates should not be interpreted as certified aviation forecasts.
- Operational use would require further validation, uncertainty assessment, and domain-specific review.

## Acknowledgments

This project uses data and scientific resources from NASA MODIS, ERA5, GPM IMERG, and CALIPSO-related products. Please cite the relevant dataset documentation and research publications when using these data.

## License

Add the license that applies to this repository and its contents. Make sure that the selected license is compatible with the licenses and terms of use for the datasets and third-party code.

## Contact

For questions, bug reports, or collaboration, please open an issue in the GitHub repository or contact the project maintainers.
