
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/full_year_2022_2023/"
    "decision_support_chunked/decision_grid_025deg.csv"
)

OUTPUT_DIR = (
    "data/processed/full_year_2022_2023/"
    "analysis/seasonal_geographical"
)

CHUNK_SIZE = 200_000

# Seasonal maps use a minimum of 10 assessable observations
# per grid location. This is a provisional reporting filter.
MIN_SEASONAL_OBSERVATIONS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEASONS = {
    "SW_Monsoon_2022": [7, 8, 9],
    "Post_Monsoon_2022": [10, 11, 12],
    "Winter_2023": [1, 2],
    "Pre_Monsoon_2023": [3, 4, 5],
    "June_2023_Partial": [6],
}

SEASON_ORDER = list(SEASONS.keys())

USECOLS = [
    "decision_timestamp",
    "grid_lat_center",
    "grid_lon_center",
    "total_input_pixels",
    "assessable_pixels",
    "low_pixels",
    "moderate_pixels",
    "high_pixels",
    "rain_detected_pixels",
    "mean_detected_rain_rate_mm_hr",
    "max_detected_rain_rate_mm_hr",
    "grid_data_status",
    "maximum_pixel_advisory_level",
]

# ============================================================
# ACCUMULATE SEASONAL GRID STATISTICS
# ============================================================

print("Starting seasonal geographical analysis...")

partial_results = []

total_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(INPUT_FILE, usecols=USECOLS, chunksize=CHUNK_SIZE),
    start=1
):

    total_rows += len(chunk)

    chunk["decision_timestamp"] = pd.to_datetime(
        chunk["decision_timestamp"],
        errors="coerce"
    )

    chunk = chunk.dropna(
        subset=[
            "decision_timestamp",
            "grid_lat_center",
            "grid_lon_center",
        ]
    )

    month = chunk["decision_timestamp"].dt.month

    chunk["season"] = np.select(
        [
            month.isin([7, 8, 9]),
            month.isin([10, 11, 12]),
            month.isin([1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6]),
        ],
        SEASON_ORDER,
        default="Unknown"
    )

    chunk = chunk[chunk["season"] != "Unknown"].copy()

    eligible = chunk["grid_data_status"].eq("Assessable")

    # Only assessable grid/time observations contribute
    # to advisory and rainfall percentages.
    for col in [
        "assessable_pixels",
        "low_pixels",
        "moderate_pixels",
        "high_pixels",
        "rain_detected_pixels",
    ]:
        chunk[f"valid_{col}"] = (
            pd.to_numeric(chunk[col], errors="coerce")
            .where(eligible, 0)
            .fillna(0)
        )

    chunk["valid_high_severity"] = (
        eligible
        & chunk["maximum_pixel_advisory_level"].eq("High")
    ).astype(int)

    rain_rate = pd.to_numeric(
        chunk["mean_detected_rain_rate_mm_hr"],
        errors="coerce"
    )

    chunk["rain_rate_weighted_sum"] = (
        rain_rate.fillna(0)
        * chunk["valid_rain_detected_pixels"]
    )

    chunk["valid_max_rain_rate"] = (
        pd.to_numeric(
            chunk["max_detected_rain_rate_mm_hr"],
            errors="coerce"
        ).where(eligible)
    )

    # Aggregate each chunk by season and grid location.
    grouped = (
        chunk.groupby(
            ["season", "grid_lat_center", "grid_lon_center"],
            observed=True
        )
        .agg(
            total_grid_time_observations=(
                "decision_timestamp", "size"
            ),
            assessable_observations=(
                "valid_assessable_pixels",
                lambda x: int((x > 0).sum())
            ),
            assessable_pixels=("valid_assessable_pixels", "sum"),
            low_pixels=("valid_low_pixels", "sum"),
            moderate_pixels=("valid_moderate_pixels", "sum"),
            high_pixels=("valid_high_pixels", "sum"),
            high_severity_observations=(
                "valid_high_severity", "sum"
            ),
            rain_detected_pixels=(
                "valid_rain_detected_pixels", "sum"
            ),
            rain_rate_weighted_sum=(
                "rain_rate_weighted_sum", "sum"
            ),
            maximum_detected_rain_rate=(
                "valid_max_rain_rate", "max"
            ),
        )
        .reset_index()
    )

    partial_results.append(grouped)

    if chunk_number % 5 == 0:
        print(
            f"Processed {chunk_number} chunks | "
            f"{total_rows:,} rows"
        )

print("\nCombining seasonal grid statistics...")

# There are only five seasons and approximately 14,000
# grid locations, so the partial aggregates remain manageable.
combined = pd.concat(partial_results, ignore_index=True)

seasonal = (
    combined.groupby(
        ["season", "grid_lat_center", "grid_lon_center"],
        as_index=False
    )
    .agg({
        "total_grid_time_observations": "sum",
        "assessable_observations": "sum",
        "assessable_pixels": "sum",
        "low_pixels": "sum",
        "moderate_pixels": "sum",
        "high_pixels": "sum",
        "high_severity_observations": "sum",
        "rain_detected_pixels": "sum",
        "rain_rate_weighted_sum": "sum",
        "maximum_detected_rain_rate": "max",
    })
)

# ============================================================
# CALCULATE SEASONAL METRICS
# ============================================================

seasonal["insufficient_observations"] = (
    seasonal["total_grid_time_observations"]
    - seasonal["assessable_observations"]
)

seasonal["assessable_observation_pct"] = (
    seasonal["assessable_observations"]
    / seasonal["total_grid_time_observations"].replace(0, np.nan)
) * 100

denominator = seasonal["assessable_pixels"].replace(0, np.nan)

seasonal["high_pct"] = (
    seasonal["high_pixels"] / denominator
) * 100

seasonal["low_pct"] = (
    seasonal["low_pixels"] / denominator
) * 100

seasonal["moderate_pct"] = (
    seasonal["moderate_pixels"] / denominator
) * 100

seasonal["high_severity_observation_pct"] = (
    seasonal["high_severity_observations"]
    / seasonal["assessable_observations"].replace(0, np.nan)
) * 100

seasonal["rain_detection_pct"] = (
    seasonal["rain_detected_pixels"] / denominator
) * 100

seasonal["mean_detected_rain_rate_mm_hr"] = (
    seasonal["rain_rate_weighted_sum"]
    / seasonal["rain_detected_pixels"].replace(0, np.nan)
)

# Apply minimum seasonal observation filter.
seasonal["meets_observation_threshold"] = (
    seasonal["assessable_observations"]
    >= MIN_SEASONAL_OBSERVATIONS
)

# ============================================================
# SAVE DETAILED SEASONAL GEOGRAPHICAL CSV
# ============================================================

detail_file = os.path.join(
    OUTPUT_DIR,
    "decision_seasonal_geographical_summary.csv"
)

seasonal.to_csv(detail_file, index=False)

print(f"Saved: {detail_file}")

# ============================================================
# SEASONAL OVERVIEW
# ============================================================

overview_rows = []

for season in SEASON_ORDER:

    all_locations = seasonal[
        seasonal["season"] == season
    ]

    valid = all_locations[
        all_locations["meets_observation_threshold"]
    ].copy()

    total_pixels = valid["assessable_pixels"].sum()

    row = {
        "season": season,
        "grid_locations_observed": len(all_locations),
        "locations_meeting_threshold": len(valid),
        "locations_excluded": len(all_locations) - len(valid),
        "total_grid_time_observations":
            all_locations["total_grid_time_observations"].sum(),
        "assessable_grid_time_observations":
            all_locations["assessable_observations"].sum(),
        "assessable_pixels": int(total_pixels),
        "high_pixels": int(valid["high_pixels"].sum()),
        "high_advisory_pct": (
            valid["high_pixels"].sum() / total_pixels * 100
            if total_pixels > 0 else np.nan
        ),
        "high_severity_observations": int(
            valid["high_severity_observations"].sum()
        ),
        "rain_detected_pixels": int(
            valid["rain_detected_pixels"].sum()
        ),
        "rain_detection_pct": (
            valid["rain_detected_pixels"].sum()
            / total_pixels * 100
            if total_pixels > 0 else np.nan
        ),
    }

    overview_rows.append(row)

overview = pd.DataFrame(overview_rows)

overview_file = os.path.join(
    OUTPUT_DIR,
    "decision_seasonal_geographical_overview.csv"
)

overview.to_csv(overview_file, index=False)

# ============================================================
# GENERATE 15 SEASONAL HEATMAPS
# Use common colour scales across seasons for each metric.
# ============================================================

PLOT_METRICS = [
    (
        "high_pct",
        "High Advisory Pixels (%)",
        "high_advisory"
    ),
    (
        "high_severity_observation_pct",
        "High-Severity Observations (%)",
        "high_severity_frequency"
    ),
    (
        "rain_detection_pct",
        "Rain-Detected Pixels (%)",
        "rain_detection"
    ),
]

# Determine one common maximum for each metric.
# This makes colour comparisons between seasons meaningful.
global_max = {}

for column, _, _ in PLOT_METRICS:
    values = seasonal.loc[
        seasonal["meets_observation_threshold"], column
    ].dropna()

    global_max[column] = (
        float(values.max()) if not values.empty else 1.0
    )

    if global_max[column] <= 0:
        global_max[column] = 1.0

for season in SEASON_ORDER:

    season_data = seasonal[
        (seasonal["season"] == season)
        & seasonal["meets_observation_threshold"]
    ].copy()

    print(
        f"\n{season}: "
        f"{len(season_data)} eligible grid locations"
    )

    for column, label, filename_prefix in PLOT_METRICS:

        fig, ax = plt.subplots(figsize=(12, 8))

        if season_data.empty:
            ax.text(
                0.5, 0.5,
                "No locations meet the observation threshold",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
            ax.set_axis_off()

        else:
            scatter = ax.scatter(
                season_data["grid_lon_center"],
                season_data["grid_lat_center"],
                c=season_data[column],
                cmap="YlOrRd",
                vmin=0,
                vmax=global_max[column],
                marker="s",
                s=13,
                linewidths=0
            )

            colorbar = plt.colorbar(scatter, ax=ax)
            colorbar.set_label(label)

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim(68, 97)
            ax.set_ylim(6, 37)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)

        ax.set_title(
            f"{label} — {season.replace('_', ' ')}",
            fontsize=14
        )

        plt.tight_layout()

        output_file = os.path.join(
            OUTPUT_DIR,
            f"decision_{filename_prefix}_{season}.png"
        )

        plt.savefig(
            output_file,
            dpi=250,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_file}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== SEASONAL GEOGRAPHICAL ANALYSIS COMPLETE ==========")

print(f"Input rows processed: {total_rows:,}")
print(f"Seasonal grid records: {len(seasonal):,}")
print(f"Minimum observations per location: {MIN_SEASONAL_OBSERVATIONS}")

print("\nSEASONAL OVERVIEW:")
print(overview.to_string(index=False))

print(f"\nOutput directory: {OUTPUT_DIR}")
print("Generated 15 seasonal heatmaps.")
print("Generated 2 seasonal CSV files.")
