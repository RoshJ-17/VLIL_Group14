#!/usr/bin/env python3
"""Apply Version 1 project-defined decision rules to TabM prediction outputs.

Outputs:
  decision_pixel_level.csv
  decision_grid_025deg.csv
  decision_timestamp_summary.csv
  decision_run_metadata.json

Advisory levels and thresholds are provisional project rules, not official
aviation risk categories. Critical is deliberately not assigned.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "timestamp": ["timestamp", "datetime", "time"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "longitude"],
    "cloud": ["cloud_type_id", "cloud_type_pred", "predicted_cloud_type",
              "cloud_type_prediction", "cloud_type", "cloud_class",
              "predicted_class", "cloud_pred"],
    "cbh": ["CBH_km", "cbh_pred_km", "predicted_cbh_km",
            "cbh_prediction_km", "predicted_cbh", "cbh_pred", "CBH_pred", "cbh"],
    "rain_prob": ["rain_probability", "precipitation_probability",
                  "rain_prob", "precip_probability", "rain_detection_probability",
                  "precipitation_detection_probability", "rain_prob_pred"],
    "rain_rate": ["rainfall_intensity_conditional_mm_hr",
                  "rainfall_intensity", "predicted_rainfall_rate",
                  "rainfall_rate_pred", "rain_rate_pred",
                  "predicted_precipitation_rate", "GPM_precipitation_rate_pred",
                  "rainfall_pred", "rain_rate", "precipitation_rate_pred"],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/january_2023/january_2023_predictions.csv")
    p.add_argument("--output-dir", default="data/processed/decision_support")
    p.add_argument("--grid-size", type=float, default=0.25)
    p.add_argument("--min-pixels", type=int, default=10)
    p.add_argument("--rain-prob-threshold", type=float, default=0.33)
    p.add_argument("--low-cbh-km", type=float, default=1.5)
    p.add_argument("--light-rain-min", type=float, default=0.1)
    p.add_argument("--moderate-rain-min", type=float, default=2.0)
    p.add_argument("--heavy-rain-min", type=float, default=4.0)
    for key in ALIASES:
        p.add_argument("--" + key.replace("_", "-") + "-col")
    return p.parse_args()


def resolve_column(df, explicit, key):
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Column {explicit!r} not found. Available: {list(df.columns)}")
        return explicit
    lower = {str(c).lower(): c for c in df.columns}
    for alias in ALIASES[key]:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise ValueError(
        f"Cannot identify {key} column. Available: {list(df.columns)}. "
        f"Pass --{key.replace('_', '-')}-col COLUMN_NAME."
    )


def pct(numerator, denominator):
    return np.where(denominator > 0, 100.0 * numerator / denominator, np.nan)


def main():
    a = parse_args()
    source = Path(a.input)
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(source)
    if a.grid_size <= 0 or a.min_pixels < 1:
        raise ValueError("--grid-size must be positive and --min-pixels must be >= 1.")
    if not (0 <= a.rain_prob_threshold <= 1):
        raise ValueError("--rain-prob-threshold must be between 0 and 1.")
    if not (0 <= a.light_rain_min < a.moderate_rain_min < a.heavy_rain_min):
        raise ValueError("Rainfall thresholds must be increasing and non-negative.")

    df = pd.read_csv(source)
    print(f"Loaded {len(df):,} rows from {source}")
    cols = {k: resolve_column(df, getattr(a, k + "_col"), k) for k in ALIASES}

    lat = pd.to_numeric(df[cols["lat"]], errors="coerce")
    lon = pd.to_numeric(df[cols["lon"]], errors="coerce")
    cloud = pd.to_numeric(df[cols["cloud"]], errors="coerce")
    cbh = pd.to_numeric(df[cols["cbh"]], errors="coerce")
    rain_prob = pd.to_numeric(df[cols["rain_prob"]], errors="coerce")
    rain_rate = pd.to_numeric(df[cols["rain_rate"]], errors="coerce")

    raw_timestamp = df[cols["timestamp"]].astype(str)
    timestamp = pd.to_datetime(raw_timestamp, format="%Y%j_%H%M", errors="coerce")
    unresolved = timestamp.isna()
    if unresolved.any():
        timestamp.loc[unresolved] = pd.to_datetime(
            raw_timestamp.loc[unresolved], errors="coerce"
        )
    df["decision_timestamp"] = timestamp

    coordinates_valid = lat.between(-90, 90) & lon.between(-180, 180)
    cloud_valid = cloud.notna() & cloud.isin(range(7))
    rain_probability_valid = rain_prob.notna() & np.isfinite(rain_prob) & rain_prob.between(0, 1)
    cbh_valid = cbh.notna() & np.isfinite(cbh) & cbh.ge(0)
    rain_intensity_valid = rain_rate.notna() & np.isfinite(rain_rate) & rain_rate.ge(0)

    # Keep the decision assessability definition explicit and independent of CBH:
    # cloud class + rain probability + valid location/time are required.
    assessable = coordinates_valid & cloud_valid & rain_probability_valid & timestamp.notna()
    cloudy = cloud_valid & cloud.ne(0)
    clear = cloud_valid & cloud.eq(0)
    cbh_applicable = cloudy
    low_cbh = cbh_applicable & cbh_valid & cbh.lt(a.low_cbh_km)
    rain_detected = rain_probability_valid & rain_prob.ge(a.rain_prob_threshold)

    rain_category = np.select(
        [
            ~rain_detected,
            rain_detected & rain_intensity_valid & rain_rate.lt(a.light_rain_min),
            rain_detected & rain_intensity_valid & rain_rate.ge(a.light_rain_min) & rain_rate.lt(a.moderate_rain_min),
            rain_detected & rain_intensity_valid & rain_rate.ge(a.moderate_rain_min) & rain_rate.lt(a.heavy_rain_min),
            rain_detected & rain_intensity_valid & rain_rate.ge(a.heavy_rain_min),
        ],
        ["No_rain_detected", "Negligible_or_inconsistent", "Light", "Moderate", "Heavy"],
        default="Unavailable"
    )

    cbh_status = np.select(
        [~cloud_valid, clear, cloudy & cbh_valid],
        ["Unavailable", "Not_applicable", "Valid"],
        default="Unavailable"
    )
    rain_intensity_status = np.select(
        [~rain_detected, rain_detected & rain_intensity_valid],
        ["Not_used_detection_negative", "Valid"],
        default="Unavailable_or_inconsistent"
    )

    high = (rain_category == "Heavy") | (low_cbh & rain_detected)
    moderate = low_cbh | np.isin(rain_category, ["Light", "Moderate"])
    advisory = np.select(
        [~assessable, high, moderate],
        ["Insufficient_Data", "High", "Moderate"],
        default="Low"
    )

    df["coordinates_valid"] = coordinates_valid
    df["cloud_type_valid"] = cloud_valid
    df["rain_probability_valid"] = rain_probability_valid
    df["cloud_condition"] = np.select(
        [~cloud_valid, clear, cloud.isin([1, 4]) & cloud_valid],
        ["Unavailable", "Clear", "Low_level_cloud"],
        default="Cloud_present"
    )
    df["cbh_status"] = cbh_status
    df["cbh_available_for_cloudy_pixel"] = cbh_applicable & cbh_valid
    df["low_cbh_flag"] = low_cbh
    df["rain_detected"] = rain_detected
    df["rain_intensity_status"] = rain_intensity_status
    df["rain_category"] = rain_category
    df["advisory_level"] = advisory


    # Vectorized aviation and military recommendations.
    # Avoids calling a Python function separately for every row.

    low_flag = low_cbh.to_numpy(dtype=bool)
    rain_flag = rain_detected.to_numpy(dtype=bool)
    heavy_flag = (rain_category == "Heavy")

    # Encode hazard combinations.
    # Bit 1: low CBH
    # Bit 2: precipitation detected
    # Bit 4: heavy precipitation
    hazard_code = (
        low_flag.astype(np.int8)
        + 2 * rain_flag.astype(np.int8)
        + 4 * heavy_flag.astype(np.int8)
    )

    aviation_hazards = np.array([
        "",
        "low cloud-base",
        "precipitation",
        "low cloud-base, precipitation",
        "",
        "",
        "precipitation, heavy precipitation",
        "low cloud-base, precipitation, heavy precipitation"
    ], dtype=object)

    military_hazards = aviation_hazards.copy()

    hazard_text_aviation = aviation_hazards[hazard_code]
    hazard_text_military = military_hazards[hazard_code]

    insufficient = (advisory == "Insufficient_Data")
    no_hazard = (hazard_code == 0)

    aviation_recommendation = np.where(
        insufficient,
        "Insufficient model inputs. Consult authoritative meteorological information and applicable procedures.",
        np.where(
            no_hazard,
            "No low-CBH or precipitation trigger was identified. Continue using current official weather reports and applicable minima.",
            "Review current aerodrome weather reports, visibility/ceiling observations, and applicable procedures for: "
            + hazard_text_aviation + "."
        )
    )

    military_recommendation = np.where(
        insufficient,
        "Insufficient model inputs. Obtain independent meteorological information for mission planning.",
        np.where(
            no_hazard,
            "No low-CBH or precipitation trigger was identified. Continue mission-specific meteorological assessment.",
            "Review local cloud-base and precipitation conditions against mission-specific operating limits for: "
            + hazard_text_military + "."
        )
    )

    # Append missing-data warnings.
    cbh_unavailable = (cbh_status == "Unavailable")
    intensity_unavailable = (
        rain_intensity_status == "Unavailable_or_inconsistent"
    )

    aviation_recommendation = np.where(
        cbh_unavailable,
        aviation_recommendation + " CBH estimate unavailable.",
        aviation_recommendation
    )

    military_recommendation = np.where(
        cbh_unavailable,
        military_recommendation + " CBH estimate unavailable.",
        military_recommendation
    )

    aviation_recommendation = np.where(
        intensity_unavailable,
        aviation_recommendation + " Rain intensity estimate requires review.",
        aviation_recommendation
    )

    military_recommendation = np.where(
        intensity_unavailable,
        military_recommendation + " Rain intensity estimate requires review.",
        military_recommendation
    )

    df["aviation_recommendation"] = aviation_recommendation
    df["military_recommendation"] = military_recommendation

    del aviation_recommendation, military_recommendation
    del hazard_text_aviation, hazard_text_military
    del hazard_code, aviation_hazards, military_hazards

    # Grid centers for fixed 0.25-degree cells.
    size = a.grid_size
    grid_lat = np.floor((lat + 90.0) / size) * size - 90.0 + size / 2
    grid_lon = np.floor((lon + 180.0) / size) * size - 180.0 + size / 2
    df["grid_lat_center"] = grid_lat.where(coordinates_valid)
    df["grid_lon_center"] = grid_lon.where(coordinates_valid)

    # Retain all spatial/time groups, including groups below the minimum.
    spatial_rows = df[coordinates_valid & timestamp.notna()].copy()
    spatial_rows["is_assessable"] = assessable.loc[spatial_rows.index].astype(int)
    spatial_rows["is_low"] = (advisory == "Low")[spatial_rows.index].astype(int)
    spatial_rows["is_moderate"] = (advisory == "Moderate")[spatial_rows.index].astype(int)
    spatial_rows["is_high"] = (advisory == "High")[spatial_rows.index].astype(int)
    spatial_rows["is_low_cbh"] = low_cbh.loc[spatial_rows.index].astype(int)
    spatial_rows["cbh_applicable"] = cbh_applicable.loc[spatial_rows.index].astype(int)
    spatial_rows["cbh_available"] = (cbh_applicable & cbh_valid).loc[spatial_rows.index].astype(int)
    spatial_rows["rain_probability_available"] = rain_probability_valid.loc[spatial_rows.index].astype(int)
    spatial_rows["rain_detected_flag"] = rain_detected.loc[spatial_rows.index].astype(int)
    spatial_rows["rain_intensity_available_when_detected"] = (
        rain_detected & rain_intensity_valid
    ).loc[spatial_rows.index].astype(int)
    spatial_rows["heavy_rain_flag"] = (rain_category == "Heavy")[spatial_rows.index].astype(int)
    spatial_rows["rate_for_stats"] = rain_rate.loc[spatial_rows.index].where(
        (rain_detected & rain_intensity_valid).loc[spatial_rows.index]
    )

    group_keys = ["decision_timestamp", "grid_lat_center", "grid_lon_center"]
    grid = spatial_rows.groupby(group_keys, dropna=True).agg(
        total_input_pixels=("is_assessable", "size"),
        assessable_pixels=("is_assessable", "sum"),
        low_pixels=("is_low", "sum"),
        moderate_pixels=("is_moderate", "sum"),
        high_pixels=("is_high", "sum"),
        cloudy_pixels=("cbh_applicable", "sum"),
        cbh_available_pixels=("cbh_available", "sum"),
        low_cbh_pixels=("is_low_cbh", "sum"),
        rain_probability_available_pixels=("rain_probability_available", "sum"),
        rain_detected_pixels=("rain_detected_flag", "sum"),
        rain_intensity_available_when_detected_pixels=("rain_intensity_available_when_detected", "sum"),
        heavy_rain_pixels=("heavy_rain_flag", "sum"),
        mean_detected_rain_rate_mm_hr=("rate_for_stats", "mean"),
        max_detected_rain_rate_mm_hr=("rate_for_stats", "max"),
    ).reset_index()

    grid["assessable_coverage_pct"] = pct(grid["assessable_pixels"], grid["total_input_pixels"])
    grid["low_pct"] = pct(grid["low_pixels"], grid["assessable_pixels"])
    grid["moderate_pct"] = pct(grid["moderate_pixels"], grid["assessable_pixels"])
    grid["high_pct"] = pct(grid["high_pixels"], grid["assessable_pixels"])
    grid["cbh_unavailable_cloudy_pixels"] = (
        grid["cloudy_pixels"] - grid["cbh_available_pixels"]
    )
    grid["rain_intensity_unavailable_detected_pixels"] = (
        grid["rain_detected_pixels"]
        - grid["rain_intensity_available_when_detected_pixels"]
    )
    grid["no_rain_detected_pixels"] = (
        grid["assessable_pixels"] - grid["rain_detected_pixels"]
    )
    grid["cbh_availability_pct_among_cloudy"] = pct(
        grid["cbh_available_pixels"], grid["cloudy_pixels"]
    )
    grid["low_cbh_pct_among_cloudy"] = pct(grid["low_cbh_pixels"], grid["cloudy_pixels"])
    grid["rain_probability_availability_pct"] = pct(
        grid["rain_probability_available_pixels"], grid["total_input_pixels"]
    )
    grid["rain_detected_pct_among_assessable"] = pct(
        grid["rain_detected_pixels"], grid["assessable_pixels"]
    )
    grid["rain_intensity_availability_pct_among_detected"] = pct(
        grid["rain_intensity_available_when_detected_pixels"], grid["rain_detected_pixels"]
    )
    grid["heavy_rain_pct_among_detected"] = pct(
        grid["heavy_rain_pixels"], grid["rain_detected_pixels"]
    )
    grid["grid_data_status"] = np.where(
        grid["assessable_pixels"] >= a.min_pixels, "Assessable", "Insufficient_Data"
    )
    grid["maximum_pixel_advisory_level"] = np.select(
        [grid["high_pixels"] > 0, grid["moderate_pixels"] > 0],
        ["High", "Moderate"], default="Low"
    )

    # Timestamp summaries use all input rows for coverage denominators.
    ts_rows = df[df["decision_timestamp"].notna()].copy()
    ts_rows["is_assessable"] = assessable.loc[ts_rows.index].astype(int)
    ts_rows["is_low"] = (advisory == "Low")[ts_rows.index].astype(int)
    ts_rows["is_moderate"] = (advisory == "Moderate")[ts_rows.index].astype(int)
    ts_rows["is_high"] = (advisory == "High")[ts_rows.index].astype(int)
    ts_rows["cloudy"] = cbh_applicable.loc[ts_rows.index].astype(int)
    ts_rows["cbh_available"] = (cbh_applicable & cbh_valid).loc[ts_rows.index].astype(int)
    ts_rows["rain_prob_available"] = rain_probability_valid.loc[ts_rows.index].astype(int)
    ts_rows["rain_detected_flag"] = rain_detected.loc[ts_rows.index].astype(int)
    ts_rows["rain_intensity_available"] = (
        rain_detected & rain_intensity_valid
    ).loc[ts_rows.index].astype(int)
    ts_rows["heavy_rain_flag"] = (rain_category == "Heavy")[ts_rows.index].astype(int)
    ts_rows["rate_for_stats"] = rain_rate.loc[ts_rows.index].where(
        (rain_detected & rain_intensity_valid).loc[ts_rows.index]
    )

    time_summary = ts_rows.groupby("decision_timestamp").agg(
        total_input_pixels=("is_assessable", "size"),
        assessable_pixels=("is_assessable", "sum"),
        low_pixels=("is_low", "sum"),
        moderate_pixels=("is_moderate", "sum"),
        high_pixels=("is_high", "sum"),
        cloudy_pixels=("cloudy", "sum"),
        cbh_available_pixels=("cbh_available", "sum"),
        rain_probability_available_pixels=("rain_prob_available", "sum"),
        rain_detected_pixels=("rain_detected_flag", "sum"),
        rain_intensity_available_when_detected_pixels=("rain_intensity_available", "sum"),
        heavy_rain_pixels=("heavy_rain_flag", "sum"),
        mean_detected_rain_rate_mm_hr=("rate_for_stats", "mean"),
        max_detected_rain_rate_mm_hr=("rate_for_stats", "max"),
    ).reset_index()

    time_summary["assessable_coverage_pct"] = pct(
        time_summary["assessable_pixels"], time_summary["total_input_pixels"]
    )
    time_summary["low_pct"] = pct(time_summary["low_pixels"], time_summary["assessable_pixels"])
    time_summary["moderate_pct"] = pct(time_summary["moderate_pixels"], time_summary["assessable_pixels"])
    time_summary["high_pct"] = pct(time_summary["high_pixels"], time_summary["assessable_pixels"])
    time_summary["cbh_availability_pct_among_cloudy"] = pct(
        time_summary["cbh_available_pixels"], time_summary["cloudy_pixels"]
    )
    time_summary["rain_probability_availability_pct"] = pct(
        time_summary["rain_probability_available_pixels"], time_summary["total_input_pixels"]
    )
    time_summary["rain_detected_pct_among_assessable"] = pct(
        time_summary["rain_detected_pixels"], time_summary["assessable_pixels"]
    )
    time_summary["rain_intensity_availability_pct_among_detected"] = pct(
        time_summary["rain_intensity_available_when_detected_pixels"],
        time_summary["rain_detected_pixels"]
    )
    time_summary["heavy_rain_pct_among_detected"] = pct(
        time_summary["heavy_rain_pixels"], time_summary["rain_detected_pixels"]
    )
    # Explicit missingness counts; zero detections are not the same as unavailable outputs.
    time_summary["cloudy_pixels_without_valid_cbh"] = (
        time_summary["cloudy_pixels"] - time_summary["cbh_available_pixels"]
    )
    time_summary["detected_rain_pixels_without_valid_intensity"] = (
        time_summary["rain_detected_pixels"]
        - time_summary["rain_intensity_available_when_detected_pixels"]
    )
    time_summary["no_rain_detected_pixels"] = (
        time_summary["assessable_pixels"] - time_summary["rain_detected_pixels"]
    )

    # Regional distribution restricted to assessable grid cells only.
    # Pixel counts are summed across those cells, so the percentages are pixel-weighted.
    assessable_grid = grid[grid["grid_data_status"] == "Assessable"].copy()
    grid_region = assessable_grid.groupby("decision_timestamp").agg(
        assessable_grid_cell_pixels=("assessable_pixels", "sum"),
        assessable_grid_low_pixels=("low_pixels", "sum"),
        assessable_grid_moderate_pixels=("moderate_pixels", "sum"),
        assessable_grid_high_pixels=("high_pixels", "sum"),
        assessable_grid_cloudy_pixels=("cloudy_pixels", "sum"),
        assessable_grid_cbh_available_pixels=("cbh_available_pixels", "sum"),
        assessable_grid_rain_detected_pixels=("rain_detected_pixels", "sum"),
        assessable_grid_rain_intensity_available_pixels=(
            "rain_intensity_available_when_detected_pixels", "sum"
        ),
    )
    # Derive unavailable cloudy CBH pixels from summed denominators, not a placeholder.
    grid_region["assessable_grid_cbh_unavailable_cloudy_pixels"] = (
        grid_region["assessable_grid_cloudy_pixels"]
        - grid_region["assessable_grid_cbh_available_pixels"]
    )
    time_summary = time_summary.merge(
        grid_region, left_on="decision_timestamp", right_index=True, how="left"
    )
    for label in ["low", "moderate", "high"]:
        time_summary[f"assessable_grid_{label}_pct_pixel_weighted"] = pct(
            time_summary[f"assessable_grid_{label}_pixels"],
            time_summary["assessable_grid_cell_pixels"]
        )
    time_summary["assessable_grid_cbh_availability_pct_pixel_weighted"] = pct(
        time_summary["assessable_grid_cbh_available_pixels"],
        time_summary["assessable_grid_cloudy_pixels"]
    )
    time_summary["assessable_grid_cbh_unavailable_pct_pixel_weighted"] = pct(
        time_summary["assessable_grid_cbh_unavailable_cloudy_pixels"],
        time_summary["assessable_grid_cloudy_pixels"]
    )
    time_summary["assessable_grid_rain_intensity_availability_pct_pixel_weighted"] = pct(
        time_summary["assessable_grid_rain_intensity_available_pixels"],
        time_summary["assessable_grid_rain_detected_pixels"]
    )

    # Equal-weight mean of grid-cell percentages is reported separately from
    # pixel-weighted regional percentages. Only assessable grid cells are included.
    grid_means = assessable_grid.groupby("decision_timestamp").agg(
        mean_grid_cbh_availability_pct=("cbh_availability_pct_among_cloudy", "mean"),
        mean_grid_rain_probability_availability_pct=("rain_probability_availability_pct", "mean"),
        mean_grid_rain_intensity_availability_pct=("rain_intensity_availability_pct_among_detected", "mean"),
        mean_grid_low_pct=("low_pct", "mean"),
        mean_grid_moderate_pct=("moderate_pct", "mean"),
        mean_grid_high_pct=("high_pct", "mean"),
    )
    time_summary = time_summary.merge(
        grid_means, left_on="decision_timestamp", right_index=True, how="left"
    )

    assessable_grid_count = assessable_grid.groupby("decision_timestamp").size()
    total_grid_count = grid.groupby("decision_timestamp").size()
    time_summary["total_grid_cells"] = time_summary["decision_timestamp"].map(total_grid_count)
    time_summary["assessable_grid_cells"] = (
        time_summary["decision_timestamp"].map(assessable_grid_count).fillna(0).astype(int)
    )
    time_summary["assessable_grid_cell_pct"] = pct(
        time_summary["assessable_grid_cells"], time_summary["total_grid_cells"]
    )

    outputs = {
        "pixel": output / "decision_pixel_level.csv",
        "grid": output / "decision_grid_025deg.csv",
        "timestamp": output / "decision_timestamp_summary.csv",
        "metadata": output / "decision_run_metadata.json",
    }
    df.to_csv(outputs["pixel"], index=False)
    grid.to_csv(outputs["grid"], index=False)
    time_summary.to_csv(outputs["timestamp"], index=False)

    metadata = {
        "input": str(source),
        "rows": int(len(df)),
        "columns_used": cols,
        "grid_size_degrees": size,
        "minimum_assessable_pixels_per_grid_timestamp": a.min_pixels,
        "rain_probability_threshold": a.rain_prob_threshold,
        "low_cbh_reference_km": a.low_cbh_km,
        "rainfall_bands_mm_hr": {
            "light_from": a.light_rain_min,
            "moderate_from": a.moderate_rain_min,
            "heavy_from": a.heavy_rain_min
        },
        "advisory_levels": ["Low", "Moderate", "High", "Insufficient_Data"],
        "critical_assigned": False,
        "grid_policy": "All grid/time groups retained; groups below minimum are labelled Insufficient_Data. Regional grid-based advisory distributions exclude insufficient grid cells.",
        "regional_reporting": "Timestamp summary reports pixel-weighted metrics over assessable grid cells and equal-weight means across assessable grid cells separately.",
        "warning": "Project-defined provisional rules, not official aviation risk categories."
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2))

    print("\nDECISION SUPPORT COMPLETE")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    print("\nPixel-level advisory counts:")
    print(df["advisory_level"].value_counts(dropna=False).to_string())
    print(f"\nGrid/time groups retained: {len(grid):,}")
    print(f"Assessable grid/time groups (minimum {a.min_pixels}): "
          f"{(grid['grid_data_status'] == 'Assessable').sum():,}")
    print(f"Insufficient grid/time groups: "
          f"{(grid['grid_data_status'] == 'Insufficient_Data').sum():,}")
    print(f"Timestamp summaries: {len(time_summary):,}")


if __name__ == "__main__":
    main()
