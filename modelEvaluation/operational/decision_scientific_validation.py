
import os
import re
import json
import math
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.cwd()

OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "scientific_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing model and analysis locations.
SEARCH_DIRS = [
    PROJECT_ROOT / "eda_results",
    PROJECT_ROOT / "tabm_cbh_output",
    PROJECT_ROOT / "tabm_full_w1_run",
    PROJECT_ROOT / "tabm_full_data",
    PROJECT_ROOT / "tabm_regression_weighted_run",
    PROJECT_ROOT / "tabm_regression_data",
    PROJECT_ROOT / "data" / "processed" / "full_year_2022_2023",
    PROJECT_ROOT / "analysis",
]

KNOWN_FILES = {
    "cloud_type_evaluation": (
        PROJECT_ROOT
        / "eda_results"
        / "cloudType"
        / "training"
        / "final_test_evaluation"
        / "tabm_final_test_evaluation_report.json"
    ),
    "cbh_preprocessing": (
        PROJECT_ROOT
        / "tabm_cbh_output"
        / "cbh_preprocessing.pkl"
    ),
    "cbh_model_config": (
        PROJECT_ROOT
        / "tabm_cbh_output"
        / "model_config.json"
    ),
    "precipitation_metadata": (
        PROJECT_ROOT
        / "tabm_full_data"
        / "metadata.json"
    ),
    "rainfall_intensity_metadata": (
        PROJECT_ROOT
        / "tabm_regression_data"
        / "metadata.json"
    ),
    "decision_run_metadata": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "full_year_2022_2023"
        / "decision_support_chunked"
        / "decision_run_metadata.json"
    ),
}

KNOWN_SUMMARY_FILES = {
    "monthly_decision_summary": (
        PROJECT_ROOT / "analysis" / "decision_monthly_summary.csv"
    ),
    "seasonal_decision_summary": (
        PROJECT_ROOT / "analysis" / "decision_seasonal_summary.csv"
    ),
    "geographical_decision_summary": (
        PROJECT_ROOT / "analysis" / "decision_geographical_summary.csv"
    ),
    "seasonal_geographical_summary": (
        PROJECT_ROOT
        / "analysis"
        / "seasonal_geographical"
        / "decision_seasonal_geographical_summary.csv"
    ),
    "overall_rainfall_intensity": (
        PROJECT_ROOT
        / "data" / "processed" / "full_year_2022_2023"
        / "analysis" / "rainfall_intensity"
        / "decision_overall_rainfall_intensity.csv"
    ),
    "monthly_rainfall_intensity": (
        PROJECT_ROOT
        / "data" / "processed" / "full_year_2022_2023"
        / "analysis" / "rainfall_intensity"
        / "decision_monthly_rainfall_intensity.csv"
    ),
    "seasonal_rainfall_intensity": (
        PROJECT_ROOT
        / "data" / "processed" / "full_year_2022_2023"
        / "analysis" / "rainfall_intensity"
        / "decision_seasonal_rainfall_intensity.csv"
    ),
    "regional_seasonal_case_studies": (
        PROJECT_ROOT
        / "data" / "processed" / "full_year_2022_2023"
        / "analysis" / "regional_case_studies"
        / "decision_regional_seasonal_case_studies.csv"
    ),
    "regional_overall_case_studies": (
        PROJECT_ROOT
        / "data" / "processed" / "full_year_2022_2023"
        / "analysis" / "regional_case_studies"
        / "decision_regional_overall_case_studies.csv"
    ),
    "regional_cloud_type_distribution": (
        PROJECT_ROOT
        / "data" / "processed" / "full_year_2022_2023"
        / "analysis" / "regional_case_studies"
        / "decision_regional_cloud_type_distribution.csv"
    ),
    "regional_grid_statistics": (
        PROJECT_ROOT
        / "data" / "processed" / "full_year_2022_2023"
        / "analysis" / "regional_grid_analysis"
        / "decision_regional_grid_seasonal_statistics.csv"
    ),
    "regional_grid_threshold_coverage": (
        PROJECT_ROOT
        / "data" / "processed" / "full_year_2022_2023"
        / "analysis" / "regional_grid_analysis"
        / "sensitivity"
        / "decision_grid_threshold_coverage.csv"
    ),
    "regional_grid_rankings": (
        PROJECT_ROOT
        / "data" / "processed" / "full_year_2022_2023"
        / "analysis" / "regional_grid_analysis"
        / "sensitivity"
        / "decision_grid_rankings_all_thresholds.csv"
    ),
    "regional_grid_stability": (
        PROJECT_ROOT
        / "data" / "processed" / "full_year_2022_2023"
        / "analysis" / "regional_grid_analysis"
        / "sensitivity"
        / "decision_grid_top5_stability.csv"
    ),
}

# Metrics to extract from JSON evaluation reports.
METRIC_PATTERN = re.compile(
    r"(accuracy|precision|recall|f1|f1_score|"
    r"roc_auc|pr_auc|auc|mae|rmse|r2|"
    r"mape|mse|loss|specificity|sensitivity|"
    r"balanced_accuracy|explained_variance)",
    re.IGNORECASE,
)

# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def safe_relative(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def json_load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {
            "_read_error": str(exc)
        }


def flatten_json(obj, prefix=""):
    """Flatten nested JSON values into key/value pairs."""
    output = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_json(value, new_key))

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            new_key = f"{prefix}[{i}]"
            output.update(flatten_json(value, new_key))

    else:
        output[prefix] = obj

    return output


def model_from_path(path):
    value = str(path).lower()

    if "cloudtype" in value or "cloud_type" in value:
        return "Cloud_Type"

    if "cbh" in value or "cloud_base" in value:
        return "CBH"

    if "regression" in value or "rainfall_intensity" in value:
        return "Rainfall_Intensity"

    if "precipitation" in value or "full_w1" in value:
        return "Precipitation_Detection"

    if "decision" in value:
        return "Decision_Support"

    return "Unclassified"


def numeric_value(value):
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)

    return None


# ============================================================
# 3. FIND AND INVENTORY EXISTING JSON REPORTS
# ============================================================

print("\n[1/7] Discovering existing JSON reports...")

json_inventory = []
json_paths = set()

for root in SEARCH_DIRS:

    if not root.exists():
        continue

    for path in root.rglob("*.json"):

        if not path.is_file():
            continue

        # Avoid duplicate processing.
        resolved = path.resolve()

        if resolved in json_paths:
            continue

        json_paths.add(resolved)

        try:
            size_mb = path.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = np.nan

        json_inventory.append({
            "file": safe_relative(path),
            "size_mb": size_mb,
            "model": model_from_path(path),
        })

json_inventory_df = pd.DataFrame(json_inventory)

json_inventory_file = (
    OUTPUT_DIR / "decision_json_artifact_inventory.csv"
)

json_inventory_df.to_csv(
    json_inventory_file,
    index=False,
)

# ============================================================
# 4. EXTRACT MODEL EVALUATION METRICS
# ============================================================

print("\n[2/7] Extracting model evaluation metrics...")

metric_rows = []
processed_json = set()

for path in sorted(json_paths):

    # Prioritize JSON files whose names suggest evaluation.
    filename = path.name.lower()

    likely_evaluation = any(
        word in filename
        for word in [
            "evaluation",
            "eval",
            "report",
            "metric",
            "result",
            "metadata",
            "summary",
        ]
    )

    if not likely_evaluation:
        continue

    if path in processed_json:
        continue

    processed_json.add(path)

    content = json_load(path)

    if not isinstance(content, (dict, list)):
        continue

    flattened = flatten_json(content)

    for key, value in flattened.items():

        if not METRIC_PATTERN.search(key):
            continue

        number = numeric_value(value)

        if number is None:
            continue

        metric_rows.append({
            "model": model_from_path(path),
            "metric_key": key,
            "metric_value": number,
            "source_file": safe_relative(path),
        })

metrics_df = pd.DataFrame(metric_rows)

if not metrics_df.empty:
    metrics_df = metrics_df.sort_values(
        ["model", "source_file", "metric_key"]
    )

metrics_file = (
    OUTPUT_DIR / "decision_model_metrics_summary.csv"
)

metrics_df.to_csv(metrics_file, index=False)

# ============================================================
# 5. INVENTORY KNOWN FILES AND EXISTING EVIDENCE
# ============================================================

print("\n[3/7] Checking known model and analysis artifacts...")

artifact_rows = []

for name, path in KNOWN_FILES.items():

    artifact_rows.append({
        "artifact": name,
        "path": safe_relative(path),
        "exists": path.exists(),
        "size_mb": (
            path.stat().st_size / (1024 * 1024)
            if path.exists() and path.is_file()
            else np.nan
        ),
    })

for name, path in KNOWN_SUMMARY_FILES.items():

    artifact_rows.append({
        "artifact": name,
        "path": safe_relative(path),
        "exists": path.exists(),
        "size_mb": (
            path.stat().st_size / (1024 * 1024)
            if path.exists() and path.is_file()
            else np.nan
        ),
    })

artifact_inventory = pd.DataFrame(artifact_rows)

artifact_inventory_file = (
    OUTPUT_DIR / "decision_validation_artifact_inventory.csv"
)

artifact_inventory.to_csv(
    artifact_inventory_file,
    index=False,
)

# ============================================================
# 6. SUMMARIZE EXISTING ANALYSIS TABLES
# ============================================================

print("\n[4/7] Reviewing existing analysis summaries...")

summary_rows = []

for name, path in KNOWN_SUMMARY_FILES.items():

    if not path.exists():
        summary_rows.append({
            "analysis": name,
            "file": safe_relative(path),
            "status": "Missing",
            "rows": np.nan,
            "columns": np.nan,
            "notes": "Expected summary file not found.",
        })
        continue

    try:
        table = pd.read_csv(path)

        summary_rows.append({
            "analysis": name,
            "file": safe_relative(path),
            "status": "Available",
            "rows": len(table),
            "columns": len(table.columns),
            "notes": ", ".join(table.columns.astype(str)),
        })

    except Exception as exc:
        summary_rows.append({
            "analysis": name,
            "file": safe_relative(path),
            "status": "Read_Error",
            "rows": np.nan,
            "columns": np.nan,
            "notes": str(exc),
        })

summary_inventory = pd.DataFrame(summary_rows)

summary_inventory_file = (
    OUTPUT_DIR / "decision_existing_analysis_inventory.csv"
)

summary_inventory.to_csv(
    summary_inventory_file,
    index=False,
)

# ============================================================
# 7. REFERENCE VALIDATION STATUS
# ============================================================

print("\n[5/7] Preparing reference-validation assessment...")

reference_rows = [
    {
        "model": "Cloud_Type",
        "reference_or_target": (
            "Seven-class cloud type labels used in model development"
        ),
        "available_evidence": (
            "Existing held-out test evaluation report, if present"
        ),
        "validation_type": "Internal held-out evaluation",
        "independent_validation_status": (
            "Not established by available artifact inventory"
        ),
        "additional_requirement": (
            "Independent cloud-type reference labels with documented "
            "class definitions and spatial/temporal matching."
        ),
    },
    {
        "model": "CBH",
        "reference_or_target": (
            "ERA5-derived cloud_base_height target"
        ),
        "available_evidence": (
            "Existing CBH test metrics and preprocessing artifacts"
        ),
        "validation_type": "Evaluation against ERA5-derived target",
        "independent_validation_status": (
            "Not independent of ERA5-derived training target"
        ),
        "additional_requirement": (
            "Independent cloud-base observations or suitable "
            "ground-based reference measurements."
        ),
    },
    {
        "model": "Precipitation_Detection",
        "reference_or_target": (
            "GPM IMERG precipitation labels; rain label threshold "
            "0.1 mm/hr"
        ),
        "available_evidence": (
            "Existing held-out test metrics and GPM-matched dataset"
        ),
        "validation_type": "Internal held-out evaluation",
        "independent_validation_status": (
            "Not established for full-year regional predictions"
        ),
        "additional_requirement": (
            "A temporally and spatially independent precipitation "
            "reference or strictly held-out GPM comparison."
        ),
    },
    {
        "model": "Rainfall_Intensity",
        "reference_or_target": (
            "GPM IMERG precipitation rate for positive-rain samples"
        ),
        "available_evidence": (
            "Existing held-out regression metrics and regional summaries"
        ),
        "validation_type": "Internal held-out regression evaluation",
        "independent_validation_status": (
            "Not established for full-year regional predictions"
        ),
        "additional_requirement": (
            "Independent rainfall-rate reference data and "
            "error analysis across intensity ranges."
        ),
    },
]

reference_df = pd.DataFrame(reference_rows)

reference_file = (
    OUTPUT_DIR / "decision_reference_validation.csv"
)

reference_df.to_csv(reference_file, index=False)

# ============================================================
# 8. METEOROLOGICAL PLAUSIBILITY CHECKLIST
# ============================================================

print("\n[6/7] Preparing meteorological plausibility assessment...")

plausibility_rows = [
    {
        "topic": "Seasonal rainfall pattern",
        "existing_evidence": (
            "Monthly and seasonal rainfall detection and intensity summaries"
        ),
        "assessment": (
            "Descriptive seasonal comparison is available. "
            "It does not independently verify rainfall accuracy."
        ),
        "status": "Descriptive analysis available",
    },
    {
        "topic": "Geographical rainfall pattern",
        "existing_evidence": (
            "Annual and seasonal geographical summaries and heatmaps"
        ),
        "assessment": (
            "Candidate geographical patterns are identified. "
            "They require independent meteorological comparison."
        ),
        "status": "Descriptive analysis available",
    },
    {
        "topic": "Cloud-type distribution",
        "existing_evidence": (
            "Seven-class model predictions and regional cloud distributions"
        ),
        "assessment": (
            "Broad class patterns can be described. The model does not "
            "identify specific aviation hazard genera such as CB or TCU."
        ),
        "status": "Descriptive analysis available",
    },
    {
        "topic": "Cloud-base-height pattern",
        "existing_evidence": (
            "ERA5-derived CBH predictions and low-CBH frequency summaries"
        ),
        "assessment": (
            "Predicted CBH patterns can be examined. CBH is not equivalent "
            "to observed aviation ceiling."
        ),
        "status": "Descriptive analysis available",
    },
    {
        "topic": "Combined advisory pattern",
        "existing_evidence": (
            "Pixel, grid and timestamp advisory summaries"
        ),
        "assessment": (
            "Rule consistency was checked. Operational validity is not "
            "established by internal consistency."
        ),
        "status": "Technical consistency checked",
    },
    {
        "topic": "Independent meteorological plausibility",
        "existing_evidence": (
            "No independent reference dataset is automatically "
            "verified by this script."
        ),
        "assessment": (
            "Requires a separately sourced and documented reference."
        ),
        "status": "Further validation required",
    },
]

plausibility_df = pd.DataFrame(plausibility_rows)

plausibility_file = (
    OUTPUT_DIR / "decision_meteorological_plausibility.csv"
)

plausibility_df.to_csv(
    plausibility_file,
    index=False,
)

# ============================================================
# 9. DECISION-SUPPORT VALIDATION STATUS
# ============================================================

decision_rows = [
    {
        "check": "Pixel-level rule consistency",
        "evidence": (
            "Previously completed validation reported zero mismatches "
            "for the checked decision rules."
        ),
        "status": "Previously checked",
        "interpretation": (
            "Internal consistency only; not meteorological validation."
        ),
    },
    {
        "check": "Grid and timestamp aggregate reconciliation",
        "evidence": (
            "Previously completed aggregation validation reported "
            "reconciled totals."
        ),
        "status": "Previously checked",
        "interpretation": (
            "Confirms arithmetic consistency of saved aggregates."
        ),
    },
    {
        "check": "Advisory severity rules",
        "evidence": (
            "Project-defined Low, Moderate, High and Insufficient_Data "
            "rules."
        ),
        "status": "Rule-based prototype",
        "interpretation": (
            "Thresholds are project-defined, not official aviation "
            "weather categories."
        ),
    },
    {
        "check": "Operational suitability",
        "evidence": (
            "Important aviation hazards are not represented in the "
            "current four-model system."
        ),
        "status": "Not established",
        "interpretation": (
            "Must not be used as a certified flight authorization, "
            "cancellation or mission-go/no-go system."
        ),
    },
]

decision_df = pd.DataFrame(decision_rows)

decision_file = (
    OUTPUT_DIR / "decision_system_validation.csv"
)

decision_df.to_csv(
    decision_file,
    index=False,
)

# ============================================================
# 10. RELIABILITY AND LIMITATIONS
# ============================================================

reliability_rows = [
    {
        "area": "Cloud type",
        "finding": (
            "Seven broad predicted classes; not a full WMO cloud-genus "
            "or aviation-hazard classification."
        ),
        "impact": "Specific CB/TCU hazards cannot be established.",
    },
    {
        "area": "CBH",
        "finding": (
            "ERA5-derived cloud-base target; CBH is not observed "
            "aviation ceiling."
        ),
        "impact": (
            "Low-CBH flags cannot replace observed ceiling or "
            "visibility measurements."
        ),
    },
    {
        "area": "Precipitation detection",
        "finding": (
            "Test precision 0.5221, recall 0.6155, F1 0.5650 and "
            "PR-AUC 0.5749."
        ),
        "impact": (
            "False positives and missed rain remain material."
        ),
    },
    {
        "area": "Rainfall intensity",
        "finding": (
            "Test MAE 1.1250 mm/hr, RMSE 2.5168 mm/hr and R2 0.2373."
        ),
        "impact": (
            "Limited explanatory performance; intensity estimates "
            "require caution."
        ),
    },
    {
        "area": "GPM spatial dependence",
        "finding": (
            "Multiple MODIS pixels can map to the same GPM grid cell."
        ),
        "impact": (
            "Pixel counts are not independent rainfall observations."
        ),
    },
    {
        "area": "Temporal coverage",
        "finding": (
            "Project data coverage is 1 July 2022 to 25 June 2023."
        ),
        "impact": (
            "Does not establish long-term climatology or year-to-year "
            "generalization."
        ),
    },
    {
        "area": "Spatial resolution",
        "finding": (
            "Regional grid summaries use 0.25-degree cells."
        ),
        "impact": (
            "Small-scale local weather variation may not be represented."
        ),
    },
    {
        "area": "Missing aviation hazards",
        "finding": (
            "Visibility/RVR, wind, windshear, turbulence, icing, hail, "
            "freezing precipitation and observed ceiling are not "
            "fully represented."
        ),
        "impact": (
            "The system cannot provide complete operational flight "
            "safety assessment."
        ),
    },
    {
        "area": "Advisory thresholds",
        "finding": (
            "Low, Moderate and High categories use project-defined "
            "rules and thresholds."
        ),
        "impact": (
            "They are not official aviation weather categories."
        ),
    },
    {
        "area": "Independent validation",
        "finding": (
            "This script inventories evidence but does not create "
            "independent ground truth."
        ),
        "impact": (
            "External validity remains to be established."
        ),
    },
]

reliability_df = pd.DataFrame(reliability_rows)

reliability_file = (
    OUTPUT_DIR / "decision_reliability_assessment.csv"
)

reliability_df.to_csv(
    reliability_file,
    index=False,
)

# ============================================================
# 11. GENERATE CONSOLIDATED MARKDOWN REPORT
# ============================================================

print("\n[7/7] Writing consolidated scientific validation report...")


def markdown_table(df, max_rows=30):
    """Generate Markdown tables without requiring tabulate."""

    if df.empty:
        return "_No records available._"

    display_df = df.head(max_rows).copy()

    def format_cell(value):
        if pd.isna(value):
            return ""

        value = str(value)
        value = value.replace("|", "\\|")
        value = value.replace("\n", " ")

        return value

    headers = [format_cell(c) for c in display_df.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in display_df.itertuples(index=False, name=None):
        formatted = [format_cell(value) for value in row]
        lines.append("| " + " | ".join(formatted) + " |")

    if len(df) > max_rows:
        lines.append(
            f"\n_Showing {max_rows} of {len(df)} rows._"
        )

    return "\n".join(lines)

report_path = (
    OUTPUT_DIR / "decision_scientific_validation_report.md"
)

limitations_path = (
    OUTPUT_DIR / "decision_project_limitations.md"
)

existing_count = int(
    artifact_inventory["exists"].sum()
)

missing_count = int(
    (~artifact_inventory["exists"]).sum()
)

report = f"""# Scientific Validation and Limitations Report

Generated: {datetime.now(timezone.utc).isoformat()}

## 1. Scope

This report consolidates existing model evaluation artifacts,
completed analyses, reference-data status, meteorological
plausibility considerations, decision-support checks and limitations.

This script does not retrain models or rerun full-year inference.

It does not claim independent scientific validation where
independent reference evidence has not been established.

## 2. Model evaluation metrics found in existing JSON artifacts

{markdown_table(metrics_df, 100)}

Metric extraction is automatic. Check each source file and metric key
before using the values in a paper. Configuration values and
training metrics must not be confused with held-out test metrics.

Known evaluation results from the project records:

- Cloud-type model: use the existing final test evaluation report.
- CBH model: MAE 1.332 km, RMSE 1.803 km, R2 0.8083.
- Precipitation detection: accuracy 0.8691, precision 0.5221,
  recall 0.6155, F1 0.5650, PR-AUC 0.5749.
- Rainfall intensity: MAE 1.1250 mm/hr, RMSE 2.5168, R2 0.2373.

## 3. Existing artifact inventory

Found: {existing_count}

Missing: {missing_count}

{markdown_table(artifact_inventory, 100)}

## 4. Existing completed analysis inventory

{markdown_table(summary_inventory, 100)}

## 5. Reference-data validation status

{markdown_table(reference_df, 20)}

A held-out test split is useful internal evaluation, but it is not
automatically independent meteorological validation. In particular,
evaluation against ERA5-derived CBH targets does not independently
verify observed cloud base. GPM comparisons must also be separated
from the data used to construct training labels.

## 6. Meteorological plausibility

{markdown_table(plausibility_df, 20)}

The seasonal and geographical analyses describe model-generated
patterns. Agreement with broad meteorological expectations is
supporting context, not proof that individual predictions are correct.

## 7. Decision-support validation

{markdown_table(decision_df, 20)}

Internal rule and aggregation checks establish computational
consistency only. They do not establish operational validity.

## 8. Reliability assessment

{markdown_table(reliability_df, 30)}

## 9. Scientific interpretation

The project provides a satellite-based prototype for weather
intelligence and advisory support. The four models estimate cloud
type, cloud base height, precipitation occurrence and conditional
rainfall intensity.

The results should be interpreted as model-generated estimates.
The project does not establish a certified operational aviation
weather service.

The advisory categories are project-defined screening levels.
They must not be interpreted as official aviation minima or
automatic flight or mission authorization.

## 10. Work still required

1. Verify the exact test-evaluation report and split provenance
   for each model.
2. Document the provenance and independence of each reference dataset.
3. Perform independent comparisons where suitable reference data
   can be obtained.
4. Review prediction errors by season, location and target range
   where held-out predictions are available.
5. Document the limitations listed in the separate limitations file.

## 11. Output files

- decision_model_metrics_summary.csv
- decision_json_artifact_inventory.csv
- decision_validation_artifact_inventory.csv
- decision_existing_analysis_inventory.csv
- decision_reference_validation.csv
- decision_meteorological_plausibility.csv
- decision_system_validation.csv
- decision_reliability_assessment.csv
- decision_project_limitations.md
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)


limitations_text = """# Project Limitations

## 1. Dataset and temporal coverage
The integrated project dataset covers 1 July 2022 to 25 June 2023.
It does not establish long-term climatology or year-to-year
generalization. Seasonal sample counts are unequal, and June 2023
is a partial period.

## 2. Spatial resolution and dependence
Regional summaries use 0.25-degree grid cells. Small-scale weather
variation may not be represented. Multiple MODIS pixels may map to
the same GPM grid cell, so pixel counts are not independent rainfall
observations.

## 3. Cloud-type classification
The model predicts seven broad classes. It does not identify all
WMO cloud genera or reliably establish aviation hazard clouds such
as cumulonimbus or towering cumulus.

## 4. Cloud-base height
The CBH target is derived from ERA5 cloud-base-height data.
It is not an observed aviation ceiling. CBH estimates cannot replace
reported cloud ceiling, visibility or RVR.

## 5. Precipitation detection
The detector has test precision 0.5221, recall 0.6155, F1 0.5650
and PR-AUC 0.5749. False detections and missed precipitation remain
possible.

## 6. Rainfall intensity
The intensity model has test MAE 1.1250 mm/hr, RMSE 2.5168 mm/hr
and R2 0.2373. Its explanatory performance is limited. Its output
is conditional on the precipitation detector identifying rain.

## 7. Reference-data limitations
GPM IMERG is used as the precipitation target/reference in model
development. ERA5-derived cloud base is used as the CBH target.
Comparisons against these same sources do not constitute fully
independent validation. Independent reference observations are
needed to establish external validity.

## 8. Missing aviation hazards
The current system does not fully assess visibility/RVR, surface
wind, crosswind, windshear, turbulence, icing, hail, freezing
precipitation or observed aviation ceiling.

## 9. Advisory-rule limitations
Low, Moderate, High and Insufficient_Data are project-defined
categories. They are not official aviation weather categories.
The thresholds are screening rules and are not certified
operational decision thresholds.

## 10. Operational limitations
The prototype must not be used as a standalone flight authorization,
cancellation or military mission go/no-go system. Official
meteorological reports, visibility, ceiling, operational procedures
and mission-specific limits remain necessary.

## 11. Validation limitations
Internal rule consistency and aggregate reconciliation do not
prove meteorological accuracy. Independent validation remains
necessary before operational claims can be made.
"""

with open(limitations_path, "w", encoding="utf-8") as f:
    f.write(limitations_text)

# ============================================================
# 12. FINAL TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("SCIENTIFIC VALIDATION AUDIT COMPLETE")
print("=" * 65)

print(f"\nProject root: {PROJECT_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")

print("\nModel metric rows extracted:", len(metrics_df))
print("JSON artifacts inventoried:", len(json_inventory_df))
print("Known artifacts found:", existing_count)
print("Known artifacts missing:", missing_count)

print("\nREFERENCE VALIDATION STATUS:")
print(
    reference_df[
        ["model", "independent_validation_status"]
    ].to_string(index=False)
)

print("\nIMPORTANT:")
print(
    "This is an evidence inventory and validation assessment. "
    "It does not create independent ground truth or prove "
    "meteorological accuracy."
)

print("\nMAIN REPORT:")
print(report_path)

print("\nLIMITATIONS DOCUMENT:")
print(limitations_path)

print("\nAll output files saved successfully.")

