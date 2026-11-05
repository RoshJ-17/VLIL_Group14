
import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/full_year_2022_2023/"
    "decision_support_chunked/decision_grid_025deg.csv"
)

OUTPUT_DIR = (
    "data/processed/full_year_2022_2023/"
    "analysis/regional_grid_analysis"
)

CHUNK_SIZE = 200_000
MIN_TOTAL_PIXELS = 10
MIN_METRIC_PIXELS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

REGIONS = {
    "Southwestern": (9, 12, 74, 76),
    "Western": (22, 26, 72, 74),
    "Eastern": (18, 21, 89, 93),
}

SEASONS = [
    "SW_Monsoon_2022",
    "Post_Monsoon_2022",
    "Winter_2023",
    "Pre_Monsoon_2023",
    "June_2023_Partial",
]

USECOLS = [
    "decision_timestamp",
    "grid_lat_center",
    "grid_lon_center",
    "total_input_pixels",
    "assessable_pixels",
    "low_pixels",
    "moderate_pixels",
    "high_pixels",
    "cloudy_pixels",
    "cbh_available_pixels",
    "low_cbh_pixels",
    "rain_probability_available_pixels",
    "rain_detected_pixels",
    "rain_intensity_available_when_detected_pixels",
    "heavy_rain_pixels",
    "mean_detected_rain_rate_mm_hr",
    "max_detected_rain_rate_mm_hr",
    "cbh_unavailable_cloudy_pixels",
    "rain_intensity_unavailable_detected_pixels",
    "no_rain_detected_pixels",
]

SUM_COLUMNS = [
    "total_input_pixels",
    "assessable_pixels",
    "low_pixels",
    "moderate_pixels",
    "high_pixels",
    "cloudy_pixels",
    "cbh_available_pixels",
    "low_cbh_pixels",
    "rain_probability_available_pixels",
    "rain_detected_pixels",
    "rain_intensity_available_when_detected_pixels",
    "heavy_rain_pixels",
    "cbh_unavailable_cloudy_pixels",
    "rain_intensity_unavailable_detected_pixels",
    "no_rain_detected_pixels",
]

GROUP_KEYS = [
    "region",
    "season",
    "grid_lat_center",
    "grid_lon_center",
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
            month.eq(6),
        ],
        SEASONS,
        default="Unknown",
    )


# ============================================================
# CHUNKED PROCESSING
# ============================================================

partial_results = []
source_rows = 0
matched_rows = {name: 0 for name in REGIONS}

print("Starting regional grid-cell analysis...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    source_rows += len(chunk)

    # Convert timestamps.
    chunk["decision_timestamp"] = pd.to_datetime(
        chunk["decision_timestamp"],
        errors="coerce",
    )

    chunk = chunk.dropna(
        subset=[
            "decision_timestamp",
            "grid_lat_center",
            "grid_lon_center",
        ]
    ).copy()

    chunk["season"] = assign_season(
        chunk["decision_timestamp"].dt.month
    )

    chunk = chunk[
        chunk["season"].isin(SEASONS)
    ].copy()

    if chunk.empty:
        continue

    # Convert count and rate columns.
    numeric_columns = [
        col for col in USECOLS
        if col not in [
            "decision_timestamp",
            "grid_lat_center",
            "grid_lon_center",
        ]
    ]

    for col in numeric_columns:
        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce",
        )

    # Create region subsets.
    regional_parts = []

    for region_name, bounds in REGIONS.items():

        lat_min, lat_max, lon_min, lon_max = bounds

        mask = (
            chunk["grid_lat_center"].ge(lat_min)
            & chunk["grid_lat_center"].lt(lat_max)
            & chunk["grid_lon_center"].ge(lon_min)
            & chunk["grid_lon_center"].lt(lon_max)
        )

        part = chunk.loc[mask].copy()

        if part.empty:
            continue

        part["region"] = region_name
        regional_parts.append(part)

        matched_rows[region_name] += len(part)

    if not regional_parts:
        continue

    data = pd.concat(
        regional_parts,
        ignore_index=True,
    )

    # Reconstruct weighted rainfall-rate sum.
    # The mean rate is based on intensity-available detections.
    data["rain_rate_weighted_sum"] = (
        data["mean_detected_rain_rate_mm_hr"]
        * data["rain_intensity_available_when_detected_pixels"]
    )

    # Aggregate each chunk by region, season and grid cell.
    agg_dict = {
        col: "sum" for col in SUM_COLUMNS
    }

    agg_dict.update({
        "rain_rate_weighted_sum": "sum",
        "max_detected_rain_rate_mm_hr": "max",
    })

    partial = (
        data.groupby(
            GROUP_KEYS,
            observed=True,
            as_index=False,
        )
        .agg(agg_dict)
    )

    partial_results.append(partial)

    if chunk_number % 10 == 0:
        print(
            f"Chunks processed: {chunk_number} | "
            f"Source rows: {source_rows:,}"
        )

if not partial_results:
    raise RuntimeError("No records matched the regional windows.")

print("\nCombining partial grid-cell aggregates...")

# ============================================================
# COMBINE CHUNK AGGREGATES
# ============================================================

combined = pd.concat(
    partial_results,
    ignore_index=True,
)

agg_dict = {
    col: "sum" for col in SUM_COLUMNS
}

agg_dict.update({
    "rain_rate_weighted_sum": "sum",
    "max_detected_rain_rate_mm_hr": "max",
})

grid = (
    combined.groupby(
        GROUP_KEYS,
        observed=True,
        as_index=False,
    )
    .agg(agg_dict)
)

# ============================================================
# RECOMPUTE METRICS FROM SUMMED COUNTS
# ============================================================

def percentage(numerator, denominator):
    result = (
        numerator
        / denominator.replace(0, np.nan)
        * 100
    )
    return result


grid["assessable_coverage_pct"] = percentage(
    grid["assessable_pixels"],
    grid["total_input_pixels"],
)

grid["rain_detection_pct"] = percentage(
    grid["rain_detected_pixels"],
    grid["assessable_pixels"],
)

grid["high_advisory_pct"] = percentage(
    grid["high_pixels"],
    grid["assessable_pixels"],
)

grid["moderate_advisory_pct"] = percentage(
    grid["moderate_pixels"],
    grid["assessable_pixels"],
)

grid["low_advisory_pct"] = percentage(
    grid["low_pixels"],
    grid["assessable_pixels"],
)

grid["low_cbh_pct_among_cbh_available"] = percentage(
    grid["low_cbh_pixels"],
    grid["cbh_available_pixels"],
)

grid["cbh_availability_pct_among_cloudy"] = percentage(
    grid["cbh_available_pixels"],
    grid["cloudy_pixels"],
)

grid["heavy_rain_pct_among_intensity_available"] = percentage(
    grid["heavy_rain_pixels"],
    grid["rain_intensity_available_when_detected_pixels"],
)

grid["rain_intensity_availability_pct"] = percentage(
    grid["rain_intensity_available_when_detected_pixels"],
    grid["rain_detected_pixels"],
)

grid["mean_detected_rain_rate_mm_hr"] = (
    grid["rain_rate_weighted_sum"]
    / grid[
        "rain_intensity_available_when_detected_pixels"
    ].replace(0, np.nan)
)

# ============================================================
# SAMPLE SUFFICIENCY FLAGS
# ============================================================

grid["sufficient_total_samples"] = (
    grid["total_input_pixels"] >= MIN_TOTAL_PIXELS
)

grid["sufficient_advisory_samples"] = (
    grid["assessable_pixels"] >= MIN_METRIC_PIXELS
)

grid["sufficient_rain_samples"] = (
    grid["rain_detected_pixels"] >= MIN_METRIC_PIXELS
)

grid["sufficient_cbh_samples"] = (
    grid["cbh_available_pixels"] >= MIN_METRIC_PIXELS
)

grid["sufficient_intensity_samples"] = (
    grid[
        "rain_intensity_available_when_detected_pixels"
    ] >= MIN_METRIC_PIXELS
)

grid["eligible_for_ranking"] = (
    grid["sufficient_total_samples"]
)

# ============================================================
# SAVE COMPLETE GRID-CELL TABLE
# ============================================================

grid["season"] = pd.Categorical(
    grid["season"],
    categories=SEASONS,
    ordered=True,
)

grid = grid.sort_values(
    ["region", "season", "grid_lat_center", "grid_lon_center"]
).reset_index(drop=True)

grid_file = os.path.join(
    OUTPUT_DIR,
    "decision_regional_grid_seasonal_statistics.csv",
)

grid.to_csv(grid_file, index=False)

# ============================================================
# REGIONAL AND SEASONAL COVERAGE OVERVIEW
# ============================================================

overview_rows = []

for (region, season), group in grid.groupby(
    ["region", "season"],
    observed=True,
):

    eligible = group[
        group["sufficient_total_samples"]
    ]

    overview_rows.append({
        "region": region,
        "season": season,
        "total_grid_cells": len(group),
        "grid_cells_with_30plus_pixels": len(eligible),
        "grid_cells_with_30plus_assessable": int(
            (group["sufficient_advisory_samples"]).sum()
        ),
        "grid_cells_with_30plus_valid_cbh": int(
            (group["sufficient_cbh_samples"]).sum()
        ),
        "grid_cells_with_30plus_rain_detections": int(
            (group["sufficient_rain_samples"]).sum()
        ),
        "grid_cells_with_30plus_valid_intensity": int(
            (group["sufficient_intensity_samples"]).sum()
        ),
        "total_input_pixels": int(group["total_input_pixels"].sum()),
        "assessable_pixels": int(group["assessable_pixels"].sum()),
        "rain_detected_pixels": int(
            group["rain_detected_pixels"].sum()
        ),
        "high_advisory_pixels": int(group["high_pixels"].sum()),
    })

overview = pd.DataFrame(overview_rows)

overview_file = os.path.join(
    OUTPUT_DIR,
    "decision_regional_grid_coverage_overview.csv",
)

overview.to_csv(overview_file, index=False)

# ============================================================
# TOP FIVE GRID CELLS BY METRIC
# ============================================================

ranking_definitions = [
    {
        "metric": "Heavy rain share",
        "value_column": "heavy_rain_pct_among_intensity_available",
        "denominator": "rain_intensity_available_when_detected_pixels",
        "minimum": MIN_METRIC_PIXELS,
        "sample_column": "rain_intensity_available_when_detected_pixels",
    },
    {
        "metric": "High advisory frequency",
        "value_column": "high_advisory_pct",
        "denominator": "assessable_pixels",
        "minimum": MIN_METRIC_PIXELS,
        "sample_column": "assessable_pixels",
    },
    {
        "metric": "Low CBH frequency",
        "value_column": "low_cbh_pct_among_cbh_available",
        "denominator": "cbh_available_pixels",
        "minimum": MIN_METRIC_PIXELS,
        "sample_column": "cbh_available_pixels",
    },
]

ranking_tables = []

for definition in ranking_definitions:

    metric_name = definition["metric"]
    value_column = definition["value_column"]
    denominator = definition["denominator"]
    minimum = definition["minimum"]
    sample_column = definition["sample_column"]

    eligible = grid[
        grid["sufficient_total_samples"]
        & (grid[denominator] >= minimum)
        & grid[value_column].notna()
    ].copy()

    eligible = eligible.sort_values(
        value_column,
        ascending=False,
    )

    eligible["rank"] = (
        eligible.groupby(
            ["region", "season"],
            observed=True,
        )
        .cumcount() + 1
    )

    top = eligible[
        eligible["rank"] <= 5
    ].copy()

    top["ranked_metric"] = metric_name
    top["metric_value_pct"] = top[value_column]
    top["metric_sample_count"] = top[sample_column]

    ranking_tables.append(
        top[
            [
                "ranked_metric",
                "region",
                "season",
                "rank",
                "grid_lat_center",
                "grid_lon_center",
                "metric_value_pct",
                "metric_sample_count",
                "total_input_pixels",
                "assessable_pixels",
                "rain_detected_pixels",
                "high_pixels",
                "cbh_available_pixels",
                "low_cbh_pixels",
                "rain_intensity_available_when_detected_pixels",
                "heavy_rain_pixels",
            ]
        ]
    )

rankings = pd.concat(
    ranking_tables,
    ignore_index=True,
)

rankings_file = os.path.join(
    OUTPUT_DIR,
    "decision_regional_grid_top5_rankings.csv",
)

rankings.to_csv(rankings_file, index=False)

# ============================================================
# TOP FIVE SUMMARY BY REGION AND SEASON
# ============================================================

print("\n========== GRID ANALYSIS COMPLETE ==========")

print(f"Source rows processed: {source_rows:,}")

print("\nREGIONAL GRID ROW COUNTS:")
for region, count in matched_rows.items():
    print(f"{region}: {count:,}")

print("\nGRID COVERAGE OVERVIEW:")
print(overview.to_string(index=False))

for metric_name in [
    "Heavy rain share",
    "High advisory frequency",
    "Low CBH frequency",
]:
    print(f"\nTOP FIVE: {metric_name}")

    selected = rankings[
        rankings["ranked_metric"] == metric_name
    ]

    for (region, season), group in selected.groupby(
        ["region", "season"],
        observed=True,
    ):
        print(f"\n{region} | {season}")
        print(
            group[
                [
                    "rank",
                    "grid_lat_center",
                    "grid_lon_center",
                    "metric_value_pct",
                    "metric_sample_count",
                    "total_input_pixels",
                ]
            ].to_string(index=False)
        )

print("\nSAVED FILES:")
print(grid_file)
print(overview_file)
print(rankings_file)
print(f"\nOutput directory: {OUTPUT_DIR}")
