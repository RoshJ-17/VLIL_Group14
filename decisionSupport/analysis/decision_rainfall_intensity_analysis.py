
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/full_year_2022_2023/"
    "full_year_2022_2023_predictions.csv"
)

OUTPUT_DIR = (
    "data/processed/full_year_2022_2023/"
    "analysis/rainfall_intensity"
)

CHUNK_SIZE = 200_000
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
    "timestamp",
    "lat",
    "lon",
    "rain_detected",
    "rain_probability",
    "rainfall_intensity_conditional_mm_hr",
    "final_rainfall_mm_hr",
]

# ============================================================
# HELPERS
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
        default="Unknown"
    )


def intensity_band(values):
    return np.select(
        [
            values.isna(),
            values < 0.1,
            values < 2,
            values < 4,
            values >= 4,
        ],
        [
            "Missing",
            "Negligible_or_Inconsistent",
            "Light",
            "Moderate",
            "Heavy",
        ],
        default="Missing"
    )


# ============================================================
# ACCUMULATORS
# ============================================================

overall_counts = {
    "total_pixels": 0,
    "rain_detected_pixels": 0,
    "non_rain_pixels": 0,
    "detected_missing_intensity": 0,
    "detected_below_0_1": 0,
    "detected_valid_intensity": 0,
    "light": 0,
    "moderate": 0,
    "heavy": 0,
    "nonrain_nonzero_final": 0,
    "detected_final_mismatch": 0,
}

monthly_parts = []
seasonal_parts = []
geographical_parts = []
seasonal_geographical_parts = []

total_rows = 0

# ============================================================
# PROCESS FULL-YEAR CSV IN CHUNKS
# ============================================================

print("Starting rainfall intensity analysis...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    total_rows += len(chunk)

    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"],
        format="%Y%j_%H%M",
        errors="coerce"
    )

    chunk = chunk.dropna(
        subset=["timestamp", "lat", "lon"]
    ).copy()

    chunk["rain_detected"] = (
        pd.to_numeric(
            chunk["rain_detected"],
            errors="coerce"
        ).fillna(0).astype(int).eq(1)
    )

    chunk["intensity"] = pd.to_numeric(
        chunk["rainfall_intensity_conditional_mm_hr"],
        errors="coerce"
    )

    chunk["final_rainfall"] = pd.to_numeric(
        chunk["final_rainfall_mm_hr"],
        errors="coerce"
    )

    chunk["month"] = chunk["timestamp"].dt.to_period("M").astype(str)

    chunk["season"] = assign_season(
        chunk["timestamp"].dt.month
    )

    detected = chunk["rain_detected"]
    intensity = chunk["intensity"]

    missing_intensity = detected & intensity.isna()
    below_threshold = detected & intensity.notna() & (intensity < 0.1)
    valid_intensity = detected & intensity.notna() & (intensity >= 0.1)

    light = valid_intensity & (intensity < 2)
    moderate = valid_intensity & (intensity >= 2) & (intensity < 4)
    heavy = valid_intensity & (intensity >= 4)

    # Overall counters
    overall_counts["total_pixels"] += len(chunk)
    overall_counts["rain_detected_pixels"] += int(detected.sum())
    overall_counts["non_rain_pixels"] += int((~detected).sum())
    overall_counts["detected_missing_intensity"] += int(missing_intensity.sum())
    overall_counts["detected_below_0_1"] += int(below_threshold.sum())
    overall_counts["detected_valid_intensity"] += int(valid_intensity.sum())
    overall_counts["light"] += int(light.sum())
    overall_counts["moderate"] += int(moderate.sum())
    overall_counts["heavy"] += int(heavy.sum())

    overall_counts["nonrain_nonzero_final"] += int(
        ((~detected) & chunk["final_rainfall"].fillna(0).ne(0)).sum()
    )

    overall_counts["detected_final_mismatch"] += int(
        (
            detected
            & (
                chunk["final_rainfall"].isna()
                | ~np.isclose(
                    chunk["final_rainfall"].fillna(-999),
                    intensity.fillna(-999),
                    atol=1e-5
                )
            )
        ).sum()
    )

    # Create intensity category for detected pixels.
    chunk["intensity_band"] = intensity_band(
        intensity.where(detected)
    )

    # Spatial grid centres: same 0.25-degree grid convention
    # used in the existing decision-support analysis.
    chunk["grid_lat_center"] = (
        np.floor(chunk["lat"] * 4) / 4 + 0.125
    )

    chunk["grid_lon_center"] = (
        np.floor(chunk["lon"] * 4) / 4 + 0.125
    )

    # Intensity metrics use detected pixels with valid
    # intensity >= 0.1 mm/hr.
    chunk["valid_intensity"] = intensity.where(valid_intensity)

    chunk["light_count"] = light.astype(int)
    chunk["moderate_count"] = moderate.astype(int)
    chunk["heavy_count"] = heavy.astype(int)

    chunk["detected_count"] = detected.astype(int)
    chunk["valid_intensity_count"] = valid_intensity.astype(int)
    chunk["inconsistent_count"] = (
        missing_intensity | below_threshold
    ).astype(int)

    chunk["intensity_sum"] = (
        chunk["valid_intensity"].fillna(0)
    )

    chunk["intensity_max"] = chunk["valid_intensity"]

    # Monthly aggregation
    monthly = chunk.groupby("month").agg(
        total_pixels=("rain_detected", "size"),
        rain_detected_pixels=("detected_count", "sum"),
        valid_intensity_pixels=("valid_intensity_count", "sum"),
        inconsistent_pixels=("inconsistent_count", "sum"),
        light_pixels=("light_count", "sum"),
        moderate_pixels=("moderate_count", "sum"),
        heavy_pixels=("heavy_count", "sum"),
        intensity_sum=("intensity_sum", "sum"),
        maximum_intensity=("intensity_max", "max"),
    ).reset_index()

    monthly_parts.append(monthly)

    # Seasonal aggregation
    seasonal = chunk.groupby("season").agg(
        total_pixels=("rain_detected", "size"),
        rain_detected_pixels=("detected_count", "sum"),
        valid_intensity_pixels=("valid_intensity_count", "sum"),
        inconsistent_pixels=("inconsistent_count", "sum"),
        light_pixels=("light_count", "sum"),
        moderate_pixels=("moderate_count", "sum"),
        heavy_pixels=("heavy_count", "sum"),
        intensity_sum=("intensity_sum", "sum"),
        maximum_intensity=("intensity_max", "max"),
    ).reset_index()

    seasonal_parts.append(seasonal)

    # Annual geographical aggregation
    geo = chunk.groupby(
        ["grid_lat_center", "grid_lon_center"]
    ).agg(
        total_pixels=("rain_detected", "size"),
        rain_detected_pixels=("detected_count", "sum"),
        valid_intensity_pixels=("valid_intensity_count", "sum"),
        inconsistent_pixels=("inconsistent_count", "sum"),
        light_pixels=("light_count", "sum"),
        moderate_pixels=("moderate_count", "sum"),
        heavy_pixels=("heavy_count", "sum"),
        intensity_sum=("intensity_sum", "sum"),
        maximum_intensity=("intensity_max", "max"),
    ).reset_index()

    geographical_parts.append(geo)

    # Seasonal geographical aggregation
    season_geo = chunk.groupby(
        ["season", "grid_lat_center", "grid_lon_center"]
    ).agg(
        total_pixels=("rain_detected", "size"),
        rain_detected_pixels=("detected_count", "sum"),
        valid_intensity_pixels=("valid_intensity_count", "sum"),
        inconsistent_pixels=("inconsistent_count", "sum"),
        light_pixels=("light_count", "sum"),
        moderate_pixels=("moderate_count", "sum"),
        heavy_pixels=("heavy_count", "sum"),
        intensity_sum=("intensity_sum", "sum"),
        maximum_intensity=("intensity_max", "max"),
    ).reset_index()

    seasonal_geographical_parts.append(season_geo)

    if chunk_number % 10 == 0:
        print(
            f"Processed {chunk_number} chunks | "
            f"{total_rows:,} rows"
        )


# ============================================================
# COMBINE PARTIAL AGGREGATIONS
# ============================================================

def combine_parts(parts, keys):
    combined = pd.concat(parts, ignore_index=True)

    return (
        combined.groupby(keys, as_index=False)
        .agg({
            "total_pixels": "sum",
            "rain_detected_pixels": "sum",
            "valid_intensity_pixels": "sum",
            "inconsistent_pixels": "sum",
            "light_pixels": "sum",
            "moderate_pixels": "sum",
            "heavy_pixels": "sum",
            "intensity_sum": "sum",
            "maximum_intensity": "max",
        })
    )


monthly = combine_parts(monthly_parts, ["month"])
seasonal = combine_parts(seasonal_parts, ["season"])

geographical = combine_parts(
    geographical_parts,
    ["grid_lat_center", "grid_lon_center"]
)

seasonal_geographical = combine_parts(
    seasonal_geographical_parts,
    ["season", "grid_lat_center", "grid_lon_center"]
)


# ============================================================
# DERIVE SUMMARY METRICS
# ============================================================

def add_metrics(df):

    detected = df["rain_detected_pixels"].replace(0, np.nan)
    valid = df["valid_intensity_pixels"].replace(0, np.nan)

    df["rain_detection_pct"] = (
        df["rain_detected_pixels"]
        / df["total_pixels"].replace(0, np.nan)
    ) * 100

    df["valid_intensity_pct_among_detected"] = (
        df["valid_intensity_pixels"] / detected
    ) * 100

    df["inconsistent_pct_among_detected"] = (
        df["inconsistent_pixels"] / detected
    ) * 100

    df["light_pct_among_valid_intensity"] = (
        df["light_pixels"] / valid
    ) * 100

    df["moderate_pct_among_valid_intensity"] = (
        df["moderate_pixels"] / valid
    ) * 100

    df["heavy_pct_among_valid_intensity"] = (
        df["heavy_pixels"] / valid
    ) * 100

    df["mean_valid_intensity_mm_hr"] = (
        df["intensity_sum"] / valid
    )

    return df


monthly = add_metrics(monthly)
seasonal = add_metrics(seasonal)
geographical = add_metrics(geographical)
seasonal_geographical = add_metrics(seasonal_geographical)

# ============================================================
# SAVE CSV FILES
# ============================================================

monthly.to_csv(
    os.path.join(OUTPUT_DIR, "decision_monthly_rainfall_intensity.csv"),
    index=False
)

seasonal.to_csv(
    os.path.join(OUTPUT_DIR, "decision_seasonal_rainfall_intensity.csv"),
    index=False
)

geographical.to_csv(
    os.path.join(OUTPUT_DIR, "decision_geographical_rainfall_intensity.csv"),
    index=False
)

seasonal_geographical.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "decision_seasonal_geographical_rainfall_intensity.csv"
    ),
    index=False
)

# Overall intensity distribution
overall = pd.DataFrame([
    ["Total pixels", overall_counts["total_pixels"]],
    ["Rain detected pixels", overall_counts["rain_detected_pixels"]],
    ["Non-rain pixels", overall_counts["non_rain_pixels"]],
    ["Detected but missing intensity", overall_counts["detected_missing_intensity"]],
    ["Detected but intensity below 0.1", overall_counts["detected_below_0_1"]],
    ["Valid intensity predictions", overall_counts["detected_valid_intensity"]],
    ["Light rainfall pixels", overall_counts["light"]],
    ["Moderate rainfall pixels", overall_counts["moderate"]],
    ["Heavy rainfall pixels", overall_counts["heavy"]],
    ["Non-rain pixels with nonzero final rainfall", overall_counts["nonrain_nonzero_final"]],
    ["Detected-rain final/intensity mismatches", overall_counts["detected_final_mismatch"]],
], columns=["metric", "count"])

overall.to_csv(
    os.path.join(OUTPUT_DIR, "decision_overall_rainfall_intensity.csv"),
    index=False
)

# ============================================================
# GENERATE ANNUAL AND SEASONAL HEATMAPS
# ============================================================

valid_values = geographical[
    "mean_valid_intensity_mm_hr"
].dropna()

vmax = (
    float(valid_values.quantile(0.95))
    if not valid_values.empty else 1.0
)

vmax = max(vmax, 0.1)

def make_heatmap(data, title, filename):

    fig, ax = plt.subplots(figsize=(12, 8))

    values = data["mean_valid_intensity_mm_hr"]

    plot = ax.scatter(
        data["grid_lon_center"],
        data["grid_lat_center"],
        c=values,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        marker="s",
        s=13,
        linewidths=0
    )

    cb = plt.colorbar(plot, ax=ax)
    cb.set_label("Mean predicted rainfall intensity (mm/hr)")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(68, 97)
    ax.set_ylim(6, 37)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=250,
        bbox_inches="tight"
    )
    plt.close()


make_heatmap(
    geographical,
    "Annual Mean Predicted Rainfall Intensity",
    "decision_annual_mean_rainfall_intensity.png"
)

for season in SEASON_ORDER:

    data = seasonal_geographical[
        seasonal_geographical["season"] == season
    ].copy()

    # Retain locations with at least five detected-rain
    # observations having valid intensity predictions.
    data = data[
        data["valid_intensity_pixels"] >= MIN_SEASONAL_OBSERVATIONS
    ]

    make_heatmap(
        data,
        f"Mean Predicted Rainfall Intensity - {season}",
        f"decision_mean_rainfall_intensity_{season}.png"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n========== RAINFALL INTENSITY ANALYSIS COMPLETE ==========")

print(f"Total rows processed: {total_rows:,}")

print("\nOVERALL COUNTS:")
print(overall.to_string(index=False))

print("\nMONTHLY SUMMARY:")
print(monthly.to_string(index=False))

print("\nSEASONAL SUMMARY:")
print(seasonal.to_string(index=False))

print("\nTOP 20 LOCATIONS BY HEAVY RAIN PIXEL COUNT:")
print(
    geographical.sort_values(
        "heavy_pixels", ascending=False
    ).head(20)[
        [
            "grid_lat_center",
            "grid_lon_center",
            "rain_detected_pixels",
            "valid_intensity_pixels",
            "heavy_pixels",
            "heavy_pct_among_valid_intensity",
            "mean_valid_intensity_mm_hr",
            "maximum_intensity",
        ]
    ].to_string(index=False)
)

print(f"\nAll outputs saved in: {OUTPUT_DIR}")

