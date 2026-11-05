
import os
import pandas as pd
import numpy as np

INPUT = (
    "data/processed/full_year_2022_2023/"
    "analysis/decision_monthly_summary.csv"
)

OUTPUT_DIR = (
    "data/processed/full_year_2022_2023/analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)

# Define project reporting periods
season_map = {
    "2022-07": "Southwest Monsoon 2022",
    "2022-08": "Southwest Monsoon 2022",
    "2022-09": "Southwest Monsoon 2022",

    "2022-10": "Post-Monsoon 2022",
    "2022-11": "Post-Monsoon 2022",
    "2022-12": "Post-Monsoon 2022",

    "2023-01": "Winter 2023",
    "2023-02": "Winter 2023",

    "2023-03": "Pre-Monsoon 2023",
    "2023-04": "Pre-Monsoon 2023",
    "2023-05": "Pre-Monsoon 2023",

    "2023-06": "June 2023 (Partial)",
}

season_order = [
    "Southwest Monsoon 2022",
    "Post-Monsoon 2022",
    "Winter 2023",
    "Pre-Monsoon 2023",
    "June 2023 (Partial)",
]

df["season"] = df["month"].map(season_map)

if df["season"].isna().any():
    missing = df.loc[df["season"].isna(), "month"].tolist()
    raise ValueError(f"Unmapped months: {missing}")

# Count columns to aggregate
sum_cols = [
    "observation_timestamps",
    "total_input_pixels",
    "total_grid_cells",
    "assessable_grid_cells",
    "assessable_grid_cell_pixels",
    "assessable_grid_low_pixels",
    "assessable_grid_moderate_pixels",
    "assessable_grid_high_pixels",
    "assessable_grid_cloudy_pixels",
    "assessable_grid_cbh_available_pixels",
    "assessable_grid_rain_detected_pixels",
    "assessable_grid_rain_intensity_available_pixels",
    "assessable_grid_cbh_unavailable_cloudy_pixels",
]

for col in sum_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Aggregate counts, not percentages
seasonal = (
    df.groupby("season", sort=False)[sum_cols]
    .sum()
    .reindex(season_order)
    .reset_index()
)

# Calculate coverage and advisory percentages
total_groups = seasonal["total_grid_cells"].replace(0, np.nan)
assessable_groups = seasonal["assessable_grid_cells"].replace(
    0, np.nan
)
assessable_pixels = seasonal["assessable_grid_cell_pixels"].replace(
    0, np.nan
)
cloudy_pixels = seasonal["assessable_grid_cloudy_pixels"].replace(
    0, np.nan
)
detected_pixels = seasonal[
    "assessable_grid_rain_detected_pixels"
].replace(0, np.nan)

seasonal["assessable_grid_coverage_pct"] = (
    seasonal["assessable_grid_cells"] / total_groups * 100
)

seasonal["assessable_pixel_coverage_pct"] = (
    seasonal["assessable_grid_cell_pixels"]
    / seasonal["total_input_pixels"].replace(0, np.nan) * 100
)

seasonal["low_pct_pixel_weighted"] = (
    seasonal["assessable_grid_low_pixels"]
    / assessable_pixels * 100
)

seasonal["moderate_pct_pixel_weighted"] = (
    seasonal["assessable_grid_moderate_pixels"]
    / assessable_pixels * 100
)

seasonal["high_pct_pixel_weighted"] = (
    seasonal["assessable_grid_high_pixels"]
    / assessable_pixels * 100
)

seasonal["cbh_availability_pct"] = (
    seasonal["assessable_grid_cbh_available_pixels"]
    / cloudy_pixels * 100
)

seasonal["rain_detection_pct"] = (
    seasonal["assessable_grid_rain_detected_pixels"]
    / assessable_pixels * 100
)

seasonal["rain_intensity_availability_pct"] = (
    seasonal["assessable_grid_rain_intensity_available_pixels"]
    / detected_pixels * 100
)

# Calculate equal-weight grid-cell metrics.
# Monthly values were weighted by assessable grid cells.
# Reweight monthly values by the corresponding number of cells
# to obtain a season-level timestamp-weighted grid mean.
equal_grid_columns = {
    "equal_grid_cbh_availability_pct":
        "equal_grid_cbh_availability_pct",
    "equal_grid_rain_probability_availability_pct":
        "equal_grid_rain_probability_availability_pct",
    "equal_grid_rain_intensity_availability_pct":
        "equal_grid_rain_intensity_availability_pct",
    "equal_grid_low_pct":
        "equal_grid_low_pct",
    "equal_grid_moderate_pct":
        "equal_grid_moderate_pct",
    "equal_grid_high_pct":
        "equal_grid_high_pct",
}

weights = df["assessable_grid_cells"].fillna(0)

for output_col, source_col in equal_grid_columns.items():
    values = pd.to_numeric(df[source_col], errors="coerce")

    numerator = (
        values * weights
    ).groupby(df["season"]).sum(min_count=1)

    denominator = (
        weights.where(values.notna(), 0)
        .groupby(df["season"])
        .sum()
    )

    seasonal[output_col] = (
        numerator / denominator.replace(0, np.nan)
    ).reindex(seasonal["season"]).to_numpy()

# Save output
output_path = os.path.join(
    OUTPUT_DIR,
    "decision_seasonal_summary.csv"
)

seasonal.to_csv(output_path, index=False)

# Print results
display_cols = [
    "season",
    "observation_timestamps",
    "total_input_pixels",
    "total_grid_cells",
    "assessable_grid_cells",
    "assessable_grid_coverage_pct",
    "low_pct_pixel_weighted",
    "moderate_pct_pixel_weighted",
    "high_pct_pixel_weighted",
    "cbh_availability_pct",
    "rain_detection_pct",
    "rain_intensity_availability_pct",
]

print("\n========== SEASONAL WEATHER ANALYSIS ==========")

print("Total timestamps:", seasonal["observation_timestamps"].sum())
print("Total input pixels:", f"{seasonal['total_input_pixels'].sum():,}")
print("Total grid/time groups:", f"{seasonal['total_grid_cells'].sum():,}")

print("\nSeasonal summary:")
print(
    seasonal[display_cols]
    .round(2)
    .to_string(index=False)
)

print("\nSaved:", output_path)


