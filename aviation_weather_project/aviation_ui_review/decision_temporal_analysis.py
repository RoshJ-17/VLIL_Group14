#!/usr/bin/env python3
"""Analyse time-dependent weather changes within fixed spatial grid cells.

This script consumes the four-model prediction CSV. It does not retrain models
or alter the existing instantaneous decision rules.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate grid-cell temporal weather trends and operational report."
    )
    p.add_argument("--input", required=True, help="Four-model prediction CSV")
    p.add_argument("--output-dir", required=True, help="Temporal output directory")
    p.add_argument("--grid-size", type=float, default=0.25)
    p.add_argument("--min-pixels", type=int, default=10)
    p.add_argument("--low-cbh-km", type=float, default=1.5)
    p.add_argument("--rain-prob-threshold", type=float, default=0.33)
    p.add_argument("--window-minutes", type=int, default=15)
    return p.parse_args()


def get_single_event_recommendation(event):
    if event == "COMBINED_DETERIORATION":
        return (
            "A simultaneous decrease in cloud-base height and increase in precipitation has been detected "
            "across successive observations. This combination may indicate deteriorating weather conditions "
            "with potential impacts on ceiling, visibility, approach and landing operations. Prioritise monitoring "
            "of the affected area, verify conditions using current METAR, TAF, weather radar and aerodrome observations, "
            "and assess operations against applicable minima."
        )
    elif event == "CBH_FALLING":
        return (
            "A decreasing cloud-base height has been detected between successive satellite observations. "
            "This may indicate deteriorating ceiling conditions. Monitor the trend closely and assess the "
            "potential impact on aircraft approach, departure and landing operations. Verify the condition "
            "using current METAR, TAF and aerodrome observations, and apply the relevant operating minima."
        )
    elif event == "CBH_RISING":
        return (
            "An increasing cloud-base height has been detected between successive satellite observations. "
            "This may indicate improving ceiling conditions. Continue monitoring subsequent observations to "
            "determine whether the improvement is sustained. Verify the conditions using current METAR, TAF "
            "and aerodrome observations before considering any operational changes."
        )
    elif event == "NEW_PRECIPITATION":
        return (
            "Precipitation has been detected in a grid cell where it was not identified in the preceding "
            "assessable observation. This may indicate the development or arrival of precipitation that could "
            "affect flight visibility, runway conditions and ground operations. Monitor the affected area and "
            "verify precipitation conditions using current weather radar, METAR and aerodrome observations."
        )
    elif event == "RAINFALL_INTENSIFYING":
        return (
            "An increase in detected rainfall intensity has been observed across successive satellite observations. "
            "This may indicate worsening precipitation conditions, potentially affecting visibility, runway surface "
            "conditions and aircraft ground operations. Monitor rainfall development and verify its intensity using "
            "current weather radar and aerodrome meteorological observations."
        )
    elif event == "IMPROVING_CONDITIONS":
        return (
            "Successive observations indicate improving cloud-base or precipitation conditions in the affected area. "
            "This may suggest gradual weather improvement. Continue monitoring subsequent observations to confirm "
            "that the improvement is sustained, and verify current conditions through official meteorological reports "
            "before considering operational adjustments."
        )
    elif event == "SUSTAINED_CBH_DECREASE":
        return (
            "A sustained decrease in cloud-base height across three or more successive observations has been detected. "
            "This indicates progressive ceiling deterioration. Closely monitor approach and departure corridors, "
            "verify trends against METAR/TAF observations, and review applicable aerodrome operating minima."
        )
    elif event == "SUSTAINED_CBH_INCREASE":
        return (
            "A sustained increase in cloud-base height across three or more successive observations has been detected. "
            "This indicates progressive ceiling lifting. Continue monitoring observations to verify stability using "
            "official aerodrome reports before adjusting operational posture."
        )
    elif event == "PRECIPITATION_CEASED":
        return (
            "Precipitation previously detected is no longer identified in current observations. This may indicate "
            "improving visibility and runway conditions. Continue monitoring to verify weather stability against "
            "current aerodrome meteorological observations."
        )
    elif event == "RAINFALL_WEAKENING":
        return (
            "A decrease in detected rainfall intensity has been observed across successive observations. This may "
            "suggest easing precipitation impact. Monitor the trend and confirm conditions using official weather "
            "radar and aerodrome reports."
        )
    return (
        "No defined temporal deterioration event was identified in the available observations. "
        "Continue using current official weather reports and applicable minima."
    )


def recommendation(event_types):
    if not event_types:
        return (
            "No defined temporal deterioration event was identified in the available observations. "
            "Continue using current official weather reports and applicable minima."
        )

    # Priority hierarchy for the overall grid timeline recommendation
    priority_order = [
        "COMBINED_DETERIORATION",
        "NEW_PRECIPITATION",
        "RAINFALL_INTENSIFYING",
        "CBH_FALLING",
        "SUSTAINED_CBH_DECREASE",
        "IMPROVING_CONDITIONS",
        "CBH_RISING",
        "SUSTAINED_CBH_INCREASE",
        "PRECIPITATION_CEASED",
        "RAINFALL_WEAKENING",
    ]

    for p in priority_order:
        if p in event_types:
            return get_single_event_recommendation(p)

    return get_single_event_recommendation(event_types[0])


def main():
    a = parse_args()
    if a.grid_size <= 0 or a.min_pixels < 1 or a.window_minutes < 1:
        raise ValueError("grid-size, min-pixels and window-minutes must be positive.")
    if not 0 <= a.rain_prob_threshold <= 1:
        raise ValueError("rain-prob-threshold must be between 0 and 1.")

    input_path = Path(a.input)
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = [
        "timestamp", "lat", "lon", "cloud_type_id", "CBH_km",
        "rain_probability", "rain_detected",
        "rainfall_intensity_conditional_mm_hr",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    raw_ts = df["timestamp"].astype(str)
    ts = pd.to_datetime(raw_ts, format="%Y%j_%H%M", errors="coerce")
    unresolved = ts.isna()
    ts.loc[unresolved] = pd.to_datetime(raw_ts.loc[unresolved], errors="coerce")
    df["decision_timestamp"] = ts

    for col in [
        "lat", "lon", "cloud_type_id", "CBH_km", "rain_probability",
        "rainfall_intensity_conditional_mm_hr",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_location = (
        df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
        & df["decision_timestamp"].notna()
    )
    cloud_valid = df["cloud_type_id"].isin(range(7))
    rain_prob_valid = (
        df["rain_probability"].notna()
        & np.isfinite(df["rain_probability"])
        & df["rain_probability"].between(0, 1)
    )
    assessable = valid_location & cloud_valid & rain_prob_valid
    cloudy = cloud_valid & df["cloud_type_id"].ne(0)
    cbh_valid = df["CBH_km"].notna() & np.isfinite(df["CBH_km"]) & df["CBH_km"].ge(0)
    cbh_usable = cloudy & cbh_valid
    rain_detected = assessable & df["rain_probability"].ge(a.rain_prob_threshold)
    rate = df["rainfall_intensity_conditional_mm_hr"]
    rate_valid = rate.notna() & np.isfinite(rate) & rate.ge(0)
    detected_rate = rate.where(rain_detected & rate_valid)

    size = a.grid_size
    df["grid_lat_center"] = np.floor((df["lat"] + 90.0) / size) * size - 90.0 + size / 2
    df["grid_lon_center"] = np.floor((df["lon"] + 180.0) / size) * size - 180.0 + size / 2
    df.loc[~valid_location, ["grid_lat_center", "grid_lon_center"]] = np.nan

    rows = df[valid_location].copy()
    rows["assessable_flag"] = assessable.loc[rows.index].astype(int)
    rows["cloudy_flag"] = cloudy.loc[rows.index].astype(int)
    rows["cbh_usable_flag"] = cbh_usable.loc[rows.index].astype(int)
    rows["low_cbh_flag"] = (
        cbh_usable & df["CBH_km"].lt(a.low_cbh_km)
    ).loc[rows.index].astype(int)
    rows["rain_detected_flag"] = rain_detected.loc[rows.index].astype(int)
    rows["rain_rate_for_stats"] = detected_rate.loc[rows.index]

    keys = ["decision_timestamp", "grid_lat_center", "grid_lon_center"]
    grid = rows.groupby(keys).agg(
        total_input_pixels=("assessable_flag", "size"),
        assessable_pixels=("assessable_flag", "sum"),
        cloudy_pixels=("cloudy_flag", "sum"),
        cbh_available_pixels=("cbh_usable_flag", "sum"),
        low_cbh_pixels=("low_cbh_flag", "sum"),
        mean_cbh_km=("CBH_km", lambda x: x[cbh_usable.loc[x.index]].mean()),
        rain_probability_mean=("rain_probability", "mean"),
        rain_detected_pixels=("rain_detected_flag", "sum"),
        rain_rate_mean_detected_mm_hr=("rain_rate_for_stats", "mean"),
        rain_rate_max_detected_mm_hr=("rain_rate_for_stats", "max"),
    ).reset_index()

    grid["cloud_coverage_pct"] = (
        100 * grid["cloudy_pixels"] / grid["total_input_pixels"].replace(0, np.nan)
    )
    grid["rain_detected_pct"] = (
        100 * grid["rain_detected_pixels"] / grid["assessable_pixels"].replace(0, np.nan)
    )
    grid["grid_data_status"] = np.where(
        grid["assessable_pixels"] >= a.min_pixels, "Assessable", "Insufficient_Data"
    )
    grid = grid.sort_values(
        ["grid_lat_center", "grid_lon_center", "decision_timestamp"]
    ).reset_index(drop=True)

    timeline_rows = []
    event_rows = []

    for (glat, glon), group in grid.groupby(["grid_lat_center", "grid_lon_center"]):
        group = group.sort_values("decision_timestamp").reset_index(drop=True)
        group["rain_transition"] = "No_transition"
        group["temporal_events"] = ""
        valid_group = group[group["grid_data_status"] == "Assessable"].copy()
        valid_group = valid_group.reset_index(drop=True)
        valid_group["cbh_delta_km"] = valid_group["mean_cbh_km"].diff()
        valid_group["rain_rate_delta_mm_hr"] = (
            valid_group["rain_rate_mean_detected_mm_hr"].diff()
        )
        # Do not report changes across gaps exceeding the analysis window.
        time_gaps = (
            valid_group["decision_timestamp"].diff().dt.total_seconds() / 60
        )
        invalid_gap = time_gaps > a.window_minutes

        valid_group.loc[
            invalid_gap,
            ["cbh_delta_km", "rain_rate_delta_mm_hr"]
        ] = np.nan        
        if valid_group.empty:
            continue

        for i in range(len(valid_group)):
            current = valid_group.iloc[i]
            event_types = []
            previous = valid_group.iloc[i - 1] if i > 0 else None

            if previous is not None:
                prev_time = previous["decision_timestamp"]
                curr_time = current["decision_timestamp"]
                gap_minutes = (curr_time - prev_time).total_seconds() / 60
                if gap_minutes <= a.window_minutes:
                    cbh_prev = previous["mean_cbh_km"]
                    cbh_now = current["mean_cbh_km"]
                    if pd.notna(cbh_prev) and pd.notna(cbh_now):
                        if cbh_now < cbh_prev:
                            event_types.append("CBH_FALLING")
                        elif cbh_now > cbh_prev:
                            event_types.append("CBH_RISING")

                    prev_rain = previous["rain_detected_pixels"] > 0
                    curr_rain = current["rain_detected_pixels"] > 0
                    if not prev_rain and curr_rain:
                        event_types.append("NEW_PRECIPITATION")
                    elif prev_rain and not curr_rain:
                        event_types.append("PRECIPITATION_CEASED")

                    prev_rate = previous["rain_rate_mean_detected_mm_hr"]
                    curr_rate = current["rain_rate_mean_detected_mm_hr"]
                    if prev_rain and curr_rain and pd.notna(prev_rate) and pd.notna(curr_rate):
                        if curr_rate > prev_rate:
                            event_types.append("RAINFALL_INTENSIFYING")
                        elif curr_rate < prev_rate:
                            event_types.append("RAINFALL_WEAKENING")

                    if (
                        "CBH_FALLING" in event_types
                        and ("NEW_PRECIPITATION" in event_types
                             or "RAINFALL_INTENSIFYING" in event_types)
                    ):
                        event_types.append("COMBINED_DETERIORATION")

                    if (
                        "CBH_RISING" in event_types
                        and ("PRECIPITATION_CEASED" in event_types
                             or "RAINFALL_WEAKENING" in event_types)
                    ):
                        event_types.append("IMPROVING_CONDITIONS")

            # Add a sustained-direction label only when at least three
            # consecutive assessable observations support the direction.
            if i >= 2:
                last3 = valid_group.iloc[i - 2:i + 1]

                times = last3["decision_timestamp"]
                gaps = times.diff().dt.total_seconds().div(60)
                cbhs = last3["mean_cbh_km"]

                if (
                    cbhs.notna().all()
                    and gaps.iloc[1:].le(a.window_minutes).all()
                ):
                    diffs = np.diff(cbhs.to_numpy(dtype=float))

                    if np.all(diffs < 0):
                        event_types.append("SUSTAINED_CBH_DECREASE")
                    elif np.all(diffs > 0):
                        event_types.append("SUSTAINED_CBH_INCREASE")

            event_types = list(dict.fromkeys(event_types))
            if previous is None:
                transition = "INITIAL_OBSERVATION"

            elif gap_minutes > a.window_minutes:
                transition = "NO_COMPARABLE_OBSERVATION"

            else:
                prev_rain = previous["rain_detected_pixels"] > 0
                curr_rain = current["rain_detected_pixels"] > 0

                if not prev_rain and not curr_rain:
                    transition = "NO_PRECIPITATION"

                elif not prev_rain and curr_rain:
                    transition = "NEW_PRECIPITATION"

                elif prev_rain and not curr_rain:
                    transition = "PRECIPITATION_CEASED"

                else:
                    transition = "PRECIPITATION_CONTINUES"
            aviation = recommendation(event_types)
            out_row = current.to_dict()
            out_row["rain_transition"] = transition
            out_row["temporal_events"] = ";".join(event_types) if event_types else "NONE"
            out_row["aviation_temporal_recommendation"] = aviation
            timeline_rows.append(out_row)

            for event in event_types:
                event_rows.append({
                    "timestamp": current["decision_timestamp"],
                    "grid_lat_center": glat,
                    "grid_lon_center": glon,
                    "event_type": event,
                    "previous_cbh_km": (
                        previous["mean_cbh_km"] if previous is not None else np.nan
                    ),
                    "current_cbh_km": current["mean_cbh_km"],
                    "cbh_change_km": (
                        current["mean_cbh_km"] - previous["mean_cbh_km"]
                        if previous is not None
                        and pd.notna(current["mean_cbh_km"])
                        and pd.notna(previous["mean_cbh_km"]) else np.nan
                    ),
                    "previous_rain_pixels": (
                        previous["rain_detected_pixels"] if previous is not None else np.nan
                    ),
                    "current_rain_pixels": current["rain_detected_pixels"],
                    "previous_mean_detected_rain_rate_mm_hr": (
                        previous["rain_rate_mean_detected_mm_hr"]
                        if previous is not None else np.nan
                    ),
                    "current_mean_detected_rain_rate_mm_hr":
                        current["rain_rate_mean_detected_mm_hr"],
                    "aviation_recommendation": get_single_event_recommendation(event)
                })

    timeline = pd.DataFrame(timeline_rows)
    events = pd.DataFrame(event_rows)

    grid_path = output_dir / "temporal_grid_analysis.csv"
    event_path = output_dir / "temporal_event_log.csv"
    timeline_path = output_dir / "temporal_timeline.csv"
    txt_path = output_dir / "temporal_operational_report.txt"
    json_path = output_dir / "temporal_operational_report.json"

    grid.to_csv(grid_path, index=False)
    events.to_csv(event_path, index=False)
    timeline.to_csv(timeline_path, index=False)

    event_counts = (
        events["event_type"].value_counts().to_dict() if not events.empty else {}
    )
    complete_cells = 0
    for _, g in grid.groupby(["grid_lat_center", "grid_lon_center"]):
        assessable_g = g[g["grid_data_status"] == "Assessable"]
        if len(assessable_g) >= 2:
            span = (
                assessable_g["decision_timestamp"].max()
                - assessable_g["decision_timestamp"].min()
            ).total_seconds() / 60
            if span >= a.window_minutes:
                complete_cells += 1

    report_lines = [
        "TEMPORAL AVIATION WEATHER OPERATIONAL PROFILE",
        "=" * 58,
        f"Input: {input_path}",
        f"Grid size: {a.grid_size} degrees",
        f"Minimum assessable pixels per grid/time: {a.min_pixels}",
        f"Low-CBH reference: below {a.low_cbh_km} km",
        f"Rain probability threshold: {a.rain_prob_threshold}",
        f"Requested trend window: {a.window_minutes} minutes",
        f"Grid/time records: {len(grid)}",
        f"Assessable grid/time records: {(grid['grid_data_status'] == 'Assessable').sum()}",
        f"Grid cells spanning at least {a.window_minutes} minutes: {complete_cells}",
        f"Temporal event records: {len(events)}",
        "",
        "EVENT COUNTS",
        "-" * 58,
    ]
    if event_counts:
        report_lines.extend([f"{k}: {v}" for k, v in sorted(event_counts.items())])
    else:
        report_lines.append("No temporal events identified.")
    report_lines.extend([
        "",
        "INTERPRETATION",
        "-" * 58,
        "Trends are calculated from successive assessable observations in the same grid cell.",
        "CBH means use cloudy pixels with valid non-negative CBH only.",
        "Precipitation transitions use whether any assessable pixel in the grid/time group",
        "crosses the configured rain-probability threshold.",
        "Rainfall-rate comparisons use detected-rain pixels only.",
        "A missing observation is not treated as no rain or as a stable condition.",
"These outputs are derived from actual MODIS satellite input files using trained models.",
"Model-derived predictions are not independently validated weather observations and must not be used alone for operational aviation decisions.",    ])
    txt_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "input_file": str(input_path.resolve()),
        "grid_size_degrees": a.grid_size,
        "minimum_assessable_pixels_per_grid_timestamp": a.min_pixels,
        "low_cbh_reference_km": a.low_cbh_km,
        "rain_probability_threshold": a.rain_prob_threshold,
        "window_minutes": a.window_minutes,
        "grid_time_records": int(len(grid)),
        "assessable_grid_time_records": int((grid["grid_data_status"] == "Assessable").sum()),
        "grid_cells_spanning_window": int(complete_cells),
        "temporal_event_records": int(len(events)),
        "event_counts": {k: int(v) for k, v in event_counts.items()},
        "outputs": {
            "grid_analysis": str(grid_path),
            "event_log": str(event_path),
            "timeline": str(timeline_path),
            "text_report": str(txt_path),
            "json_report": str(json_path),
        },
        "limitations": [
            "Trends are based only on observed timestamps; no interpolation is performed.",
            "CBH is aggregated from cloudy pixels with valid CBH only.",
            "Precipitation presence is defined as one or more detected pixels in an assessable grid/time group.",
"Actual MODIS satellite inputs were processed; model-derived predictions have not been independently validated against ground-truth weather observations.",        ],
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("TEMPORAL ANALYSIS COMPLETE")
    print("Grid/time records:", len(grid))
    print("Assessable grid/time records:", int((grid["grid_data_status"] == "Assessable").sum()))
    print("Temporal event records:", len(events))
    print("Event counts:", event_counts)
    print("Grid analysis:", grid_path)
    print("Event log:", event_path)
    print("Timeline:", timeline_path)
    print("Text report:", txt_path)
    print("JSON report:", json_path)


if __name__ == "__main__":
    main()
