
import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/full_year_2022_2023/"
    "decision_support_chunked/decision_pixel_level.csv"
)

OUTPUT_DIR = (
    "data/processed/full_year_2022_2023/"
    "analysis/regional_case_studies"
)

CHUNK_SIZE = 200_000
LOW_CBH_THRESHOLD_KM = 1.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEASON_ORDER = [
    "SW_Monsoon_2022",
    "Post_Monsoon_2022",
    "Winter_2023",
    "Pre_Monsoon_2023",
    "June_2023_Partial",
]

REGIONS = {
    "Southwestern": {
        "lat_min": 9.0, "lat_max": 12.0,
        "lon_min": 74.0, "lon_max": 76.0,
    },
    "Western": {
        "lat_min": 22.0, "lat_max": 26.0,
        "lon_min": 72.0, "lon_max": 74.0,
    },
    "Eastern": {
        "lat_min": 18.0, "lat_max": 21.0,
        "lon_min": 89.0, "lon_max": 93.0,
    },
}

USECOLS = [
    "timestamp", "lat", "lon",
    "cloud_type_id", "cloud_type",
    "cloud_condition",
    "CBH_km", "cbh_status", "low_cbh_flag",
    "rain_probability", "rain_detected",
    "rainfall_intensity_conditional_mm_hr",
    "final_rainfall_mm_hr",
    "advisory_level",
]

CLOUD_CLASSES = [
    "clear",
    "liquid_low",
    "liquid_middle",
    "liquid_high",
    "ice_low",
    "ice_middle",
    "ice_high",
]

ADVISORY_LEVELS = ["Low", "Moderate", "High"]

INTENSITY_BANDS = [
    "Negligible_or_Inconsistent",
    "Light",
    "Moderate",
    "Heavy",
]

# ============================================================
# SEASON ASSIGNMENT
# ============================================================

def assign_season(month):
    return np.select(
        [
            month.isin([7, 8, 9]),
            month.isin([10, 11, 12]),
            month.isin([1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6]),
        ],
        SEASON_ORDER,
        default="Unknown",
    )


# ============================================================
# ACCUMULATORS
# ============================================================

partial_results = []
total_rows = 0
region_rows = {name: 0 for name in REGIONS}

# ============================================================
# READ AND PROCESS FULL-YEAR PIXEL DATA
# ============================================================

print("Starting regional case-study analysis...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    total_rows += len(chunk)

    # Parse timestamp.
    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"],
        format="%Y%j_%H%M",
        errors="coerce",
    )

    chunk = chunk.dropna(
        subset=["timestamp", "lat", "lon"]
    ).copy()

    # Assign season.
    chunk["season"] = assign_season(
        chunk["timestamp"].dt.month
    )

    chunk = chunk[
        chunk["season"] != "Unknown"
    ].copy()

    # Convert numerical columns safely.
    numeric_columns = [
        "cloud_type_id",
        "CBH_km",
        "rain_probability",
        "rainfall_intensity_conditional_mm_hr",
        "final_rainfall_mm_hr",
    ]

    for col in numeric_columns:
        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce",
        )

    # Normalize boolean columns.
    for col in ["rain_detected", "low_cbh_flag"]:
        chunk[col] = (
            chunk[col]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )

    # --------------------------------------------------------
    # REGIONAL SELECTION
    # --------------------------------------------------------

    regional_chunks = []

    for region_name, bounds in REGIONS.items():

        mask = (
            chunk["lat"].between(
                bounds["lat_min"],
                bounds["lat_max"],
                inclusive="left",
            )
            &
            chunk["lon"].between(
                bounds["lon_min"],
                bounds["lon_max"],
                inclusive="left",
            )
        )

        region_data = chunk.loc[mask].copy()

        if region_data.empty:
            continue

        region_data["region"] = region_name
        regional_chunks.append(region_data)

        region_rows[region_name] += len(region_data)

    if not regional_chunks:
        continue

    data = pd.concat(
        regional_chunks,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # CLOUD-TYPE INDICATORS
    # --------------------------------------------------------

    for cloud_class in CLOUD_CLASSES:
        data[f"cloud_{cloud_class}"] = (
            data["cloud_type"]
            .astype(str)
            .eq(cloud_class)
            .astype(int)
        )

    data["cloud_type_unknown"] = (
        ~data["cloud_type"].astype(str).isin(CLOUD_CLASSES)
    ).astype(int)

    data["cloudy_pixels"] = (
        data["cloud_type_id"].notna()
        & data["cloud_type_id"].ne(0)
    ).astype(int)

    # --------------------------------------------------------
    # CBH VALIDITY
    # --------------------------------------------------------

    cbh_valid = (
        data["cbh_status"].astype(str).eq("Valid")
        & data["cloudy_pixels"].eq(1)
        & data["CBH_km"].notna()
        & np.isfinite(data["CBH_km"])
        & data["CBH_km"].ge(0)
    )

    data["cbh_valid_count"] = cbh_valid.astype(int)

    data["cbh_sum"] = (
        data["CBH_km"].where(cbh_valid, 0)
    )

    data["cbh_min"] = data["CBH_km"].where(cbh_valid)

    data["cbh_max"] = data["CBH_km"].where(cbh_valid)

    data["low_cbh_count"] = (
        cbh_valid
        & data["CBH_km"].lt(LOW_CBH_THRESHOLD_KM)
    ).astype(int)

    data["cbh_unavailable_cloudy"] = (
        data["cloudy_pixels"].eq(1)
        & ~cbh_valid
    ).astype(int)

    # --------------------------------------------------------
    # PRECIPITATION DETECTION
    # --------------------------------------------------------

    data["rain_detected_count"] = (
        data["rain_detected"].astype(int)
    )

    data["rain_probability_valid"] = (
        data["rain_probability"].notna()
    ).astype(int)

    # --------------------------------------------------------
    # RAINFALL INTENSITY
    # --------------------------------------------------------

    intensity = data[
        "rainfall_intensity_conditional_mm_hr"
    ]

    valid_intensity = (
        data["rain_detected"]
        & intensity.notna()
        & np.isfinite(intensity)
        & intensity.ge(0.1)
    )

    data["intensity_valid_count"] = (
        valid_intensity.astype(int)
    )

    data["intensity_sum"] = (
        intensity.where(valid_intensity, 0)
    )

    data["intensity_max"] = intensity.where(
        valid_intensity
    )

    data["intensity_negligible"] = (
        data["rain_detected"]
        & intensity.notna()
        & intensity.lt(0.1)
    ).astype(int)

    data["intensity_light"] = (
        valid_intensity & intensity.lt(2)
    ).astype(int)

    data["intensity_moderate"] = (
        valid_intensity
        & intensity.ge(2)
        & intensity.lt(4)
    ).astype(int)

    data["intensity_heavy"] = (
        valid_intensity & intensity.ge(4)
    ).astype(int)

    # --------------------------------------------------------
    # ADVISORY INDICATORS
    # --------------------------------------------------------

    for level in ADVISORY_LEVELS:
        data[f"advisory_{level.lower()}"] = (
            data["advisory_level"]
            .astype(str)
            .eq(level)
            .astype(int)
        )

    data["advisory_insufficient"] = (
        data["advisory_level"]
        .astype(str)
        .eq("Insufficient_Data")
        .astype(int)
    )

    # --------------------------------------------------------
    # GROUP BY REGION AND SEASON
    # --------------------------------------------------------

    aggregation = {
        "total_pixels": ("timestamp", "size"),
        "rain_detected_pixels": ("rain_detected_count", "sum"),
        "rain_probability_available_pixels": (
            "rain_probability_valid", "sum"
        ),
        "cloudy_pixels": ("cloudy_pixels", "sum"),
        "cbh_valid_pixels": ("cbh_valid_count", "sum"),
        "cbh_sum": ("cbh_sum", "sum"),
        "cbh_min": ("cbh_min", "min"),
        "cbh_max": ("cbh_max", "max"),
        "low_cbh_pixels": ("low_cbh_count", "sum"),
        "cbh_unavailable_cloudy_pixels": (
            "cbh_unavailable_cloudy", "sum"
        ),
        "valid_intensity_pixels": (
            "intensity_valid_count", "sum"
        ),
        "intensity_sum": ("intensity_sum", "sum"),
        "maximum_intensity": ("intensity_max", "max"),
        "negligible_intensity_pixels": (
            "intensity_negligible", "sum"
        ),
        "light_rain_pixels": ("intensity_light", "sum"),
        "moderate_rain_pixels": ("intensity_moderate", "sum"),
        "heavy_rain_pixels": ("intensity_heavy", "sum"),
        "low_advisory_pixels": ("advisory_low", "sum"),
        "moderate_advisory_pixels": (
            "advisory_moderate", "sum"
        ),
        "high_advisory_pixels": ("advisory_high", "sum"),
        "insufficient_advisory_pixels": (
            "advisory_insufficient", "sum"
        ),
    }

    for cloud_class in CLOUD_CLASSES:
        aggregation[f"cloud_{cloud_class}_pixels"] = (
            f"cloud_{cloud_class}", "sum"
        )

    aggregation["unknown_cloud_type_pixels"] = (
        "cloud_type_unknown", "sum"
    )

    grouped = (
        data.groupby(
            ["region", "season"],
            observed=True,
        )
        .agg(**aggregation)
        .reset_index()
    )

    partial_results.append(grouped)

    if chunk_number % 20 == 0:
        print(
            f"Processed {chunk_number} chunks | "
            f"{total_rows:,} source rows"
        )


# ============================================================
# COMBINE PARTIAL RESULTS
# ============================================================

if not partial_results:
    raise RuntimeError(
        "No pixels matched the three regional coordinate windows."
    )

print("\nCombining regional and seasonal statistics...")

combined = pd.concat(
    partial_results,
    ignore_index=True,
)

sum_columns = [
    col for col in combined.columns
    if col not in [
        "region",
        "season",
        "cbh_min",
        "cbh_max",
        "maximum_intensity",
    ]
]

aggregation = {
    col: "sum" for col in sum_columns
}

aggregation.update({
    "cbh_min": "min",
    "cbh_max": "max",
    "maximum_intensity": "max",
})

summary = (
    combined.groupby(
        ["region", "season"],
        as_index=False,
    )
    .agg(aggregation)
)

# ============================================================
# CALCULATE FINAL METRICS
# ============================================================

def safe_pct(numerator, denominator):
    return (
        numerator / denominator.replace(0, np.nan)
    ) * 100


summary["rain_detection_pct"] = safe_pct(
    summary["rain_detected_pixels"],
    summary["total_pixels"],
)

summary["cloudy_pixel_pct"] = safe_pct(
    summary["cloudy_pixels"],
    summary["total_pixels"],
)

summary["cbh_availability_pct_among_cloudy"] = safe_pct(
    summary["cbh_valid_pixels"],
    summary["cloudy_pixels"],
)

summary["low_cbh_pct_among_valid_cloudy"] = safe_pct(
    summary["low_cbh_pixels"],
    summary["cbh_valid_pixels"],
)

summary["mean_cbh_km"] = (
    summary["cbh_sum"]
    / summary["cbh_valid_pixels"].replace(0, np.nan)
)

summary["intensity_availability_pct_among_detected"] = safe_pct(
    summary["valid_intensity_pixels"],
    summary["rain_detected_pixels"],
)

summary["mean_rainfall_intensity_mm_hr"] = (
    summary["intensity_sum"]
    / summary["valid_intensity_pixels"].replace(0, np.nan)
)

summary["light_rain_pct_among_valid_intensity"] = safe_pct(
    summary["light_rain_pixels"],
    summary["valid_intensity_pixels"],
)

summary["moderate_rain_pct_among_valid_intensity"] = safe_pct(
    summary["moderate_rain_pixels"],
    summary["valid_intensity_pixels"],
)

summary["heavy_rain_pct_among_valid_intensity"] = safe_pct(
    summary["heavy_rain_pixels"],
    summary["valid_intensity_pixels"],
)

summary["low_advisory_pct"] = safe_pct(
    summary["low_advisory_pixels"],
    summary["total_pixels"],
)

summary["moderate_advisory_pct"] = safe_pct(
    summary["moderate_advisory_pixels"],
    summary["total_pixels"],
)

summary["high_advisory_pct"] = safe_pct(
    summary["high_advisory_pixels"],
    summary["total_pixels"],
)

for cloud_class in CLOUD_CLASSES:
    summary[f"cloud_{cloud_class}_pct"] = safe_pct(
        summary[f"cloud_{cloud_class}_pixels"],
        summary["total_pixels"],
    )

# ============================================================
# ORDER AND SAVE SEASONAL SUMMARY
# ============================================================

summary["season"] = pd.Categorical(
    summary["season"],
    categories=SEASON_ORDER,
    ordered=True,
)

summary = summary.sort_values(
    ["region", "season"]
).reset_index(drop=True)

output_file = os.path.join(
    OUTPUT_DIR,
    "decision_regional_seasonal_case_studies.csv",
)

summary.to_csv(output_file, index=False)

# ============================================================
# REGIONAL OVERALL SUMMARY
# ============================================================

overall = (
    summary.groupby("region", observed=True)
    .agg({
        "total_pixels": "sum",
        "rain_detected_pixels": "sum",
        "cloudy_pixels": "sum",
        "cbh_valid_pixels": "sum",
        "low_cbh_pixels": "sum",
        "valid_intensity_pixels": "sum",
        "light_rain_pixels": "sum",
        "moderate_rain_pixels": "sum",
        "heavy_rain_pixels": "sum",
        "low_advisory_pixels": "sum",
        "moderate_advisory_pixels": "sum",
        "high_advisory_pixels": "sum",
        "intensity_sum": "sum",
        "cbh_sum": "sum",
    })
    .reset_index()
)

overall["rain_detection_pct"] = safe_pct(
    overall["rain_detected_pixels"],
    overall["total_pixels"],
)

overall["mean_cbh_km"] = (
    overall["cbh_sum"]
    / overall["cbh_valid_pixels"].replace(0, np.nan)
)

overall["low_cbh_pct_among_valid_cloudy"] = safe_pct(
    overall["low_cbh_pixels"],
    overall["cbh_valid_pixels"],
)

overall["mean_rainfall_intensity_mm_hr"] = (
    overall["intensity_sum"]
    / overall["valid_intensity_pixels"].replace(0, np.nan)
)

overall["heavy_rain_pct_among_valid_intensity"] = safe_pct(
    overall["heavy_rain_pixels"],
    overall["valid_intensity_pixels"],
)

overall["high_advisory_pct"] = safe_pct(
    overall["high_advisory_pixels"],
    overall["total_pixels"],
)

overall_file = os.path.join(
    OUTPUT_DIR,
    "decision_regional_overall_case_studies.csv",
)

overall.to_csv(overall_file, index=False)

# ============================================================
# CLOUD TYPE DISTRIBUTION TABLE
# ============================================================

cloud_columns = [
    "region",
    "season",
    "total_pixels",
    "cloudy_pixels",
]

cloud_columns += [
    f"cloud_{name}_pixels" for name in CLOUD_CLASSES
]

cloud_table = summary[cloud_columns].copy()

cloud_file = os.path.join(
    OUTPUT_DIR,
    "decision_regional_cloud_type_distribution.csv",
)

cloud_table.to_csv(cloud_file, index=False)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n========== REGIONAL CASE STUDY ANALYSIS COMPLETE ==========")

print(f"Source rows processed: {total_rows:,}")

print("\nREGIONAL ROW COUNTS:")
for name, count in region_rows.items():
    print(f"{name}: {count:,}")

print("\nREGIONAL SEASONAL COMPARISON:")
print(
    summary[
        [
            "region",
            "season",
            "total_pixels",
            "cloudy_pixel_pct",
            "mean_cbh_km",
            "low_cbh_pct_among_valid_cloudy",
            "rain_detection_pct",
            "mean_rainfall_intensity_mm_hr",
            "heavy_rain_pct_among_valid_intensity",
            "high_advisory_pct",
        ]
    ].to_string(index=False)
)

print("\nREGIONAL OVERALL COMPARISON:")
print(overall.to_string(index=False))

print("\nSaved files:")
print(output_file)
print(overall_file)
print(cloud_file)

print(f"\nOutput directory: {OUTPUT_DIR}")

