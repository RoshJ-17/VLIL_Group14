
import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/full_year_2022_2023/analysis/"
    "regional_grid_analysis/"
    "decision_regional_grid_seasonal_statistics.csv"
)

OUTPUT_DIR = (
    "data/processed/full_year_2022_2023/analysis/"
    "regional_grid_analysis/sensitivity"
)

THRESHOLDS = [5, 10, 30]
PRIMARY_THRESHOLD = 10
TOP_N = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

REGIONS = ["Southwestern", "Western", "Eastern"]

SEASONS = [
    "SW_Monsoon_2022",
    "Post_Monsoon_2022",
    "Winter_2023",
    "Pre_Monsoon_2023",
    "June_2023_Partial",
]

# Each metric has its own scientifically relevant denominator.
METRICS = {
    "Heavy_rain_share": {
        "numerator": "heavy_rain_pixels",
        "denominator": "rain_intensity_available_when_detected_pixels",
    },
    "High_advisory_frequency": {
        "numerator": "high_pixels",
        "denominator": "assessable_pixels",
    },
    "Low_CBH_frequency": {
        "numerator": "low_cbh_pixels",
        "denominator": "cbh_available_pixels",
    },
}

# ============================================================
# LOAD EXISTING GRID-CELL STATISTICS
# ============================================================

print("Loading existing regional grid-cell statistics...")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "region",
    "season",
    "grid_lat_center",
    "grid_lon_center",
    "total_input_pixels",
]

for metric in METRICS.values():
    required_columns.extend([
        metric["numerator"],
        metric["denominator"],
    ])

missing = sorted(set(required_columns) - set(df.columns))

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.drop_duplicates(
    subset=[
        "region",
        "season",
        "grid_lat_center",
        "grid_lon_center",
    ]
).copy()

for col in required_columns:
    if col not in ["region", "season"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[
    df["region"].isin(REGIONS)
    & df["season"].isin(SEASONS)
].copy()

# ============================================================
# CREATE RANKINGS AND COVERAGE FOR ALL THRESHOLDS
# ============================================================

all_rankings = []
coverage_rows = []

print("\nEvaluating thresholds:", THRESHOLDS)

for threshold in THRESHOLDS:

    print(f"\nProcessing threshold: {threshold}")

    for metric_name, config in METRICS.items():

        numerator = config["numerator"]
        denominator = config["denominator"]

        for region in REGIONS:

            for season in SEASONS:

                subset = df[
                    (df["region"] == region)
                    & (df["season"] == season)
                ].copy()

                # Require sufficient total input samples.
                total_eligible = subset[
                    subset["total_input_pixels"] >= threshold
                ].copy()

                # Require sufficient metric-specific samples.
                metric_eligible = total_eligible[
                    total_eligible[denominator] >= threshold
                ].copy()

                # Calculate percentage from counts, not averaged percentages.
                metric_eligible["metric_value_pct"] = (
                    metric_eligible[numerator]
                    / metric_eligible[denominator]
                    * 100
                )

                # Do not rank zero-event cells as hotspots.
                positive_events = metric_eligible[
                    metric_eligible[numerator] > 0
                ].copy()

                positive_events = positive_events.sort_values(
                    ["metric_value_pct", numerator],
                    ascending=[False, False],
                )

                coverage_rows.append({
                    "threshold": threshold,
                    "metric": metric_name,
                    "region": region,
                    "season": season,
                    "total_grid_cells": len(subset),
                    "cells_meeting_total_sample_threshold": len(
                        total_eligible
                    ),
                    "cells_meeting_metric_denominator": len(
                        metric_eligible
                    ),
                    "cells_with_positive_events": len(
                        positive_events
                    ),
                    "cells_ranked": min(
                        TOP_N, len(positive_events)
                    ),
                    "total_input_pixels": int(
                        subset["total_input_pixels"].sum()
                    ),
                })

                if positive_events.empty:
                    continue

                top = positive_events.head(TOP_N).copy()

                top["threshold"] = threshold
                top["metric"] = metric_name
                top["rank"] = range(1, len(top) + 1)

                top["metric_sample_count"] = top[denominator]
                top["event_count"] = top[numerator]

                all_rankings.append(
                    top[[
                        "threshold",
                        "metric",
                        "region",
                        "season",
                        "rank",
                        "grid_lat_center",
                        "grid_lon_center",
                        "metric_value_pct",
                        "metric_sample_count",
                        "event_count",
                        "total_input_pixels",
                    ]]
                )

# ============================================================
# SAVE ALL RANKINGS
# ============================================================

if all_rankings:
    rankings = pd.concat(
        all_rankings,
        ignore_index=True,
    )
else:
    rankings = pd.DataFrame()

rankings_file = os.path.join(
    OUTPUT_DIR,
    "decision_grid_rankings_all_thresholds.csv",
)

rankings.to_csv(rankings_file, index=False)

# ============================================================
# SAVE COVERAGE
# ============================================================

coverage = pd.DataFrame(coverage_rows)

coverage_file = os.path.join(
    OUTPUT_DIR,
    "decision_grid_threshold_coverage.csv",
)

coverage.to_csv(coverage_file, index=False)

# ============================================================
# PRIMARY RESULTS: THRESHOLD 10
# ============================================================

primary = rankings[
    rankings["threshold"] == PRIMARY_THRESHOLD
].copy()

primary_file = os.path.join(
    OUTPUT_DIR,
    "decision_grid_primary_10.csv",
)

primary.to_csv(primary_file, index=False)

# ============================================================
# TOP-FIVE STABILITY COMPARISON
# ============================================================

stability_rows = []

if not rankings.empty:

    grouping = ["metric", "region", "season"]

    for keys, group in rankings.groupby(
        grouping,
        dropna=False,
    ):

        metric_name, region, season = keys

        cell_sets = {}

        for threshold in THRESHOLDS:

            selected = group[
                group["threshold"] == threshold
            ]

            cell_sets[threshold] = set(
                zip(
                    selected["grid_lat_center"],
                    selected["grid_lon_center"],
                )
            )

        for comparison_threshold in [5, 30]:

            set_10 = cell_sets[PRIMARY_THRESHOLD]
            set_other = cell_sets[comparison_threshold]

            intersection = set_10.intersection(set_other)
            union = set_10.union(set_other)

            jaccard = (
                len(intersection) / len(union) * 100
                if union else np.nan
            )

            stability_rows.append({
                "metric": metric_name,
                "region": region,
                "season": season,
                "primary_threshold": PRIMARY_THRESHOLD,
                "comparison_threshold": comparison_threshold,
                "primary_top5_count": len(set_10),
                "comparison_top5_count": len(set_other),
                "shared_grid_cells": len(intersection),
                "top5_jaccard_similarity_pct": jaccard,
            })

stability = pd.DataFrame(stability_rows)

stability_file = os.path.join(
    OUTPUT_DIR,
    "decision_grid_top5_stability.csv",
)

stability.to_csv(stability_file, index=False)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n========== SENSITIVITY ANALYSIS COMPLETE ==========")

print("\nGRID-CELL COVERAGE BY THRESHOLD:")

print(
    coverage[
        [
            "threshold",
            "metric",
            "region",
            "season",
            "cells_meeting_metric_denominator",
            "cells_with_positive_events",
            "cells_ranked",
        ]
    ].to_string(index=False)
)

print("\nTOP-FIVE STABILITY:")

if not stability.empty:
    print(stability.to_string(index=False))

print("\nPRIMARY TOP-FIVE RESULTS: THRESHOLD 10")

if not primary.empty:
    print(
        primary[
            [
                "metric",
                "region",
                "season",
                "rank",
                "grid_lat_center",
                "grid_lon_center",
                "metric_value_pct",
                "metric_sample_count",
                "event_count",
            ]
        ].to_string(index=False)
    )

print("\nSAVED FILES:")
print(rankings_file)
print(coverage_file)
print(primary_file)
print(stability_file)

print(f"\nOutput directory: {OUTPUT_DIR}")

