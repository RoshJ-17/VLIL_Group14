
import os
import pandas as pd
import numpy as np

INPUT = (
    "data/processed/full_year_2022_2023/"
    "decision_support_chunked/decision_timestamp_summary.csv"
)

OUTPUT_DIR = (
    "data/processed/full_year_2022_2023/analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)

df["decision_timestamp"] = pd.to_datetime(
    df["decision_timestamp"], errors="coerce"
)

df = df.dropna(subset=["decision_timestamp"])
df["month"] = df["decision_timestamp"].dt.to_period("M").astype(str)

# Columns that must be summed across timestamps
sum_cols = [
    "total_input_pixels",
    "assessable_pixels",
    "low_pixels",
    "moderate_pixels",
    "high_pixels",
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

monthly = df.groupby("month")[sum_cols].sum().reset_index()

timestamp_counts = (
    df.groupby("month")["decision_timestamp"]
    .nunique()
    .rename("observation_timestamps")
    .reset_index()
)

monthly = monthly.merge(timestamp_counts, on="month")

# Coverage percentages
monthly["assessable_grid_coverage_pct"] = (
    monthly["assessable_grid_cells"]
    / monthly["total_grid_cells"] * 100
)

monthly["assessable_pixel_coverage_pct"] = (
    monthly["assessable_grid_cell_pixels"]
    / monthly["total_input_pixels"] * 100
)

# Advisory percentages: only assessable grid cells
den = monthly["assessable_grid_cell_pixels"].replace(0, np.nan)

monthly["low_pct_pixel_weighted"] = (
    monthly["assessable_grid_low_pixels"] / den * 100
)

monthly["moderate_pct_pixel_weighted"] = (
    monthly["assessable_grid_moderate_pixels"] / den * 100
)

monthly["high_pct_pixel_weighted"] = (
    monthly["assessable_grid_high_pixels"] / den * 100
)

# Weather-parameter availability
cloudy = monthly["assessable_grid_cloudy_pixels"].replace(0, np.nan)
detected = monthly["assessable_grid_rain_detected_pixels"].replace(
    0, np.nan
)

monthly["cbh_availability_pct"] = (
    monthly["assessable_grid_cbh_available_pixels"]
    / cloudy * 100
)

monthly["rain_detection_pct"] = (
    monthly["assessable_grid_rain_detected_pixels"]
    / den * 100
)

monthly["rain_intensity_availability_pct"] = (
    monthly["assessable_grid_rain_intensity_available_pixels"]
    / detected * 100
)

# Equal-weight means across assessable grid cells.
# Weight each timestamp's grid mean by its number of assessable cells.
mean_cols = {
    "mean_grid_cbh_availability_pct":
        "equal_grid_cbh_availability_pct",
    "mean_grid_rain_probability_availability_pct":
        "equal_grid_rain_probability_availability_pct",
    "mean_grid_rain_intensity_availability_pct":
        "equal_grid_rain_intensity_availability_pct",
    "mean_grid_low_pct":
        "equal_grid_low_pct",
    "mean_grid_moderate_pct":
        "equal_grid_moderate_pct",
    "mean_grid_high_pct":
        "equal_grid_high_pct",
}

weights = df["assessable_grid_cells"].fillna(0)

for source, output in mean_cols.items():
    df[source] = pd.to_numeric(df[source], errors="coerce")

    numerator = (
        df[source] * weights
    ).groupby(df["month"]).sum(min_count=1)

    denominator = weights.groupby(df["month"]).sum()

    monthly[output] = (
        numerator / denominator.replace(0, np.nan)
    ).reindex(monthly["month"]).to_numpy()

# Save
output_path = os.path.join(
    OUTPUT_DIR, "decision_monthly_summary.csv"
)

monthly.to_csv(output_path, index=False)

print("\n========== MONTHLY WEATHER ANALYSIS ==========")
print("Months:", len(monthly))
print("Total timestamps:", monthly["observation_timestamps"].sum())
print("Total input pixels:", f"{monthly['total_input_pixels'].sum():,}")
print("Total grid/time groups:", f"{monthly['total_grid_cells'].sum():,}")

print("\nMonthly summary:")
display_cols = [
    "month",
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

print(
    monthly[display_cols].round(2).to_string(index=False)
)

print("\nSaved:", output_path)

