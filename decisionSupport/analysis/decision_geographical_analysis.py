
import os
import sqlite3
import pandas as pd
import numpy as np

INPUT = (
    "data/processed/full_year_2022_2023/"
    "decision_support_chunked/decision_grid_025deg.csv"
)

OUTPUT_DIR = (
    "data/processed/full_year_2022_2023/analysis"
)

DB_PATH = os.path.join(
    OUTPUT_DIR, "geographical_analysis_temp.sqlite"
)

OUTPUT = os.path.join(
    OUTPUT_DIR, "decision_geographical_summary.csv"
)

MIN_ASSESSABLE_OBSERVATIONS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)

conn.execute("""
CREATE TABLE geo (
    lat REAL,
    lon REAL,
    total_groups INTEGER,
    assessable_groups INTEGER,
    insufficient_groups INTEGER,
    total_input_pixels INTEGER,
    assessable_pixels INTEGER,
    low_pixels INTEGER,
    moderate_pixels INTEGER,
    high_pixels INTEGER,
    high_severity_groups INTEGER,
    rain_rate_weighted_sum REAL,
    rain_detected_pixels INTEGER,
    maximum_rain REAL
)
""")

print("Reading grid-level data in chunks...")

for i, chunk in enumerate(
    pd.read_csv(INPUT, chunksize=100000), start=1
):

    eligible = chunk["grid_data_status"].eq("Assessable")

    # Preserve all observations for coverage statistics.
    chunk["is_assessable"] = eligible.astype(int)
    chunk["is_insufficient"] = (
        chunk["grid_data_status"]
        .eq("Insufficient_Data").astype(int)
    )

    # Advisory metrics: eligible grid/time groups only.
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

    rain_pixels = chunk["valid_rain_detected_pixels"]

    chunk["rain_rate_weighted_sum"] = (
        rain_rate.fillna(0) * rain_pixels
    )

    chunk["rain_rate_weight"] = (
        rain_pixels.where(rain_rate.notna(), 0)
    )

    chunk["valid_maximum_rain"] = (
        pd.to_numeric(
            chunk["max_detected_rain_rate_mm_hr"],
            errors="coerce"
        ).where(eligible)
    )

    grouped = chunk.groupby(
        ["grid_lat_center", "grid_lon_center"],
        dropna=False
    ).agg(
        total_groups=("total_input_pixels", "size"),
        assessable_groups=("is_assessable", "sum"),
        insufficient_groups=("is_insufficient", "sum"),
        total_input_pixels=("total_input_pixels", "sum"),
        assessable_pixels=("valid_assessable_pixels", "sum"),
        low_pixels=("valid_low_pixels", "sum"),
        moderate_pixels=("valid_moderate_pixels", "sum"),
        high_pixels=("valid_high_pixels", "sum"),
        high_severity_groups=("valid_high_severity", "sum"),
        rain_rate_weighted_sum=("rain_rate_weighted_sum", "sum"),
        rain_detected_pixels=("rain_rate_weight", "sum"),
        maximum_rain=("valid_maximum_rain", "max")
    ).reset_index()

    grouped.rename(columns={
        "grid_lat_center": "lat",
        "grid_lon_center": "lon"
    }, inplace=True)

    grouped.to_sql(
        "geo",
        conn,
        if_exists="append",
        index=False
    )

    print(f"Processed chunk {i}")

print("\nCombining repeated locations...")

result = pd.read_sql_query("""
SELECT
    lat,
    lon,
    SUM(total_groups) AS total_grid_time_observations,
    SUM(assessable_groups) AS assessable_observations,
    SUM(insufficient_groups) AS insufficient_observations,
    SUM(total_input_pixels) AS total_input_pixels,
    SUM(assessable_pixels) AS assessable_pixels,
    SUM(low_pixels) AS low_pixels,
    SUM(moderate_pixels) AS moderate_pixels,
    SUM(high_pixels) AS high_pixels,
    SUM(high_severity_groups) AS high_severity_observations,
    SUM(rain_rate_weighted_sum) AS rain_rate_weighted_sum,
    SUM(rain_detected_pixels) AS rain_detected_pixels,
    MAX(maximum_rain) AS maximum_detected_rain_rate
FROM geo
GROUP BY lat, lon
""", conn)

# Data availability
result["assessable_observation_pct"] = (
    result["assessable_observations"]
    / result["total_grid_time_observations"] * 100
)

result["insufficient_observation_pct"] = (
    result["insufficient_observations"]
    / result["total_grid_time_observations"] * 100
)

# Advisory percentages use eligible pixels only.
den = result["assessable_pixels"].replace(0, np.nan)

result["low_pct"] = result["low_pixels"] / den * 100
result["moderate_pct"] = result["moderate_pixels"] / den * 100
result["high_pct"] = result["high_pixels"] / den * 100

# High-severity observation frequency.
obs_den = result["assessable_observations"].replace(
    0, np.nan
)

result["high_severity_observation_pct"] = (
    result["high_severity_observations"]
    / obs_den * 100
)

# Rainfall weighted by detected-rain pixel count.
rain_den = result["rain_detected_pixels"].replace(
    0, np.nan
)

result["mean_detected_rain_rate_mm_hr"] = (
    result["rain_rate_weighted_sum"] / rain_den
)

result.drop(
    columns=["rain_rate_weighted_sum"],
    inplace=True
)

result.to_csv(OUTPUT, index=False)

# Exclude locations with inadequate temporal sampling
rankable = result[
    result["assessable_observations"]
    >= MIN_ASSESSABLE_OBSERVATIONS
].copy()

print("\n========== CORRECTED GEOGRAPHICAL ANALYSIS ==========")
print("Unique grid locations:", len(result))
print("Locations meeting observation threshold:", len(rankable))

print("Total grid/time observations:",
      f"{result['total_grid_time_observations'].sum():,}")

print("Total assessable observations:",
      f"{result['assessable_observations'].sum():,}")

print("Total assessable pixels:",
      f"{result['assessable_pixels'].sum():,}")

print("Total High advisory pixels:",
      f"{result['high_pixels'].sum():,}")

print("\nTop 20 by High advisory percentage:")
print(
    rankable.sort_values(
        ["high_pct", "assessable_observations"],
        ascending=[False, False]
    ).head(20)[[
        "lat", "lon",
        "assessable_observations",
        "high_pixels",
        "high_pct",
        "high_severity_observation_pct"
    ]].round(2).to_string(index=False)
)

print("\nTop 20 by High advisory pixel count:")
print(
    rankable.sort_values(
        "high_pixels", ascending=False
    ).head(20)[[
        "lat", "lon",
        "assessable_observations",
        "high_pixels",
        "high_pct",
        "high_severity_observation_pct"
    ]].round(2).to_string(index=False)
)

print("\nSaved:", OUTPUT)

conn.close()
os.remove(DB_PATH)
