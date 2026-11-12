#!/usr/bin/env python3
"""Chunked decision-support processing for large TabM prediction CSV files.

Writes pixel-level decisions incrementally and accumulates grid/time statistics
in SQLite. Partial aggregates are combined before final percentages are computed.
Project-defined advisory rules; not official aviation risk categories.
"""
import argparse
import json
import sqlite3
import sys
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
    "rain_prob": ["rain_probability", "precipitation_probability", "rain_prob",
                  "precip_probability", "rain_detection_probability",
                  "precipitation_detection_probability", "rain_prob_pred"],
    "rain_rate": ["rainfall_intensity_conditional_mm_hr", "rainfall_intensity",
                  "predicted_rainfall_rate", "rainfall_rate_pred", "rain_rate_pred",
                  "predicted_precipitation_rate", "GPM_precipitation_rate_pred",
                  "rainfall_pred", "rain_rate", "precipitation_rate_pred"],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/processed/january_2023/january_2023_predictions.csv")
    p.add_argument("--output-dir", default="data/processed/decision_support_chunked")
    p.add_argument("--grid-size", type=float, default=0.25)
    p.add_argument("--min-pixels", type=int, default=10)
    p.add_argument("--rain-prob-threshold", type=float, default=0.33)
    p.add_argument("--low-cbh-km", type=float, default=1.5)
    p.add_argument("--light-rain-min", type=float, default=0.1)
    p.add_argument("--moderate-rain-min", type=float, default=2.0)
    p.add_argument("--heavy-rain-min", type=float, default=4.0)
    p.add_argument("--chunksize", type=int, default=50000)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    for key in ALIASES:
        p.add_argument("--" + key.replace("_", "-") + "-col")
    return p.parse_args()


def resolve_columns(columns, args):
    lower = {str(c).lower(): c for c in columns}
    out = {}
    for key, aliases in ALIASES.items():
        explicit = getattr(args, key + "_col")
        if explicit:
            if explicit not in columns:
                raise ValueError(f"Column {explicit!r} not found. Available: {list(columns)}")
            out[key] = explicit
        else:
            found = next((lower[x.lower()] for x in aliases if x.lower() in lower), None)
            if found is None:
                raise ValueError(
                    f"Cannot identify {key} column. Pass --{key.replace('_', '-')}-col. "
                    f"Available: {list(columns)}"
                )
            out[key] = found
    return out


def pct(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    shape = np.broadcast_shapes(num.shape, den.shape)
    out = np.full(shape, np.nan, dtype=np.float64)
    np.divide(100.0 * num, den, out=out, where=den > 0)
    return out


GRID_METRICS = [
    "total_input_pixels", "assessable_pixels", "low_pixels", "moderate_pixels",
    "high_pixels", "cloudy_pixels", "cbh_available_pixels", "low_cbh_pixels",
    "rain_probability_available_pixels", "rain_detected_pixels",
    "rain_intensity_available_when_detected_pixels", "heavy_rain_pixels",
]
TIME_METRICS = [
    "total_input_pixels", "assessable_pixels", "low_pixels", "moderate_pixels",
    "high_pixels", "cloudy_pixels", "cbh_available_pixels",
    "rain_probability_available_pixels", "rain_detected_pixels",
    "rain_intensity_available_when_detected_pixels", "heavy_rain_pixels",
]


def init_db(conn):
    gcols = ", ".join(f'"{c}" INTEGER NOT NULL' for c in GRID_METRICS)
    tcols = ", ".join(f'"{c}" INTEGER NOT NULL' for c in TIME_METRICS)
    conn.execute(f"""CREATE TABLE grid_agg (
        decision_timestamp TEXT NOT NULL, grid_lat_center REAL NOT NULL,
        grid_lon_center REAL NOT NULL, {gcols},
        rate_sum REAL NOT NULL, rate_count INTEGER NOT NULL, max_rate REAL,
        PRIMARY KEY(decision_timestamp, grid_lat_center, grid_lon_center))""")
    conn.execute(f"""CREATE TABLE time_agg (
        decision_timestamp TEXT PRIMARY KEY, {tcols},
        rate_sum REAL NOT NULL, rate_count INTEGER NOT NULL, max_rate REAL)""")
    conn.commit()


def update_sqlite(conn, grouped, table, metrics, has_grid_key):
    if grouped.empty:
        return
    rows = []
    if has_grid_key:
        for (ts, glat, glon), r in grouped.iterrows():
            rows.append((
                pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                float(glat), float(glon),
                *[int(r[c]) for c in metrics],
                float(r["rate_sum"]), int(r["rate_count"]),
                None if pd.isna(r["max_rate"]) else float(r["max_rate"]),
            ))
        keys = ["decision_timestamp", "grid_lat_center", "grid_lon_center"]
        key_count = 3
    else:
        for ts, r in grouped.iterrows():
            rows.append((
                pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                *[int(r[c]) for c in metrics],
                float(r["rate_sum"]), int(r["rate_count"]),
                None if pd.isna(r["max_rate"]) else float(r["max_rate"]),
            ))
        keys = ["decision_timestamp"]
        key_count = 1

    placeholders = ",".join(["?"] * (key_count + len(metrics) + 3))
    updates = [f'"{c}" = {table}."{c}" + excluded."{c}"' for c in metrics]
    updates += [
        f'"rate_sum" = {table}."rate_sum" + excluded."rate_sum"',
        f'"rate_count" = {table}."rate_count" + excluded."rate_count"',
        '"max_rate" = CASE WHEN ' + table + '.max_rate IS NULL THEN excluded.max_rate '
        'WHEN excluded.max_rate IS NULL THEN ' + table + '.max_rate '
        'ELSE MAX(' + table + '.max_rate, excluded.max_rate) END',
    ]
    sql = (
        f"INSERT INTO {table} VALUES ({placeholders}) "
        f"ON CONFLICT({','.join(keys)}) DO UPDATE SET " + ", ".join(updates)
    )
    conn.executemany(sql, rows)


def process_chunk(df, cols, a):
    lat = pd.to_numeric(df[cols["lat"]], errors="coerce")
    lon = pd.to_numeric(df[cols["lon"]], errors="coerce")
    cloud = pd.to_numeric(df[cols["cloud"]], errors="coerce")
    cbh = pd.to_numeric(df[cols["cbh"]], errors="coerce")
    rain_prob = pd.to_numeric(df[cols["rain_prob"]], errors="coerce")
    rain_rate = pd.to_numeric(df[cols["rain_rate"]], errors="coerce")

    raw_ts = df[cols["timestamp"]].astype("string")
    ts = pd.to_datetime(raw_ts, format="%Y%j_%H%M", errors="coerce")
    bad = ts.isna()
    if bad.any():
        ts.loc[bad] = pd.to_datetime(raw_ts.loc[bad], errors="coerce")
    df["decision_timestamp"] = ts

    coord_ok = lat.between(-90, 90) & lon.between(-180, 180)
    cloud_ok = cloud.notna() & cloud.isin(range(7))
    prob_ok = rain_prob.notna() & np.isfinite(rain_prob) & rain_prob.between(0, 1)
    cbh_ok = cbh.notna() & np.isfinite(cbh) & cbh.ge(0)
    rate_ok = rain_rate.notna() & np.isfinite(rain_rate) & rain_rate.ge(0)

    assess = coord_ok & cloud_ok & prob_ok & ts.notna()
    cloudy = cloud_ok & cloud.ne(0)
    clear = cloud_ok & cloud.eq(0)
    cbh_app = cloudy
    low_cbh = cbh_app & cbh_ok & cbh.lt(a.low_cbh_km)
    rain = prob_ok & rain_prob.ge(a.rain_prob_threshold)

    category = np.select(
        [
            ~rain,
            rain & rate_ok & rain_rate.lt(a.light_rain_min),
            rain & rate_ok & rain_rate.ge(a.light_rain_min) & rain_rate.lt(a.moderate_rain_min),
            rain & rate_ok & rain_rate.ge(a.moderate_rain_min) & rain_rate.lt(a.heavy_rain_min),
            rain & rate_ok & rain_rate.ge(a.heavy_rain_min),
        ],
        ["No_rain_detected", "Negligible_or_inconsistent", "Light", "Moderate", "Heavy"],
        default="Unavailable",
    )
    cbh_status = np.select(
        [~cloud_ok, clear, cloudy & cbh_ok],
        ["Unavailable", "Not_applicable", "Valid"], default="Unavailable",
    )
    intensity_status = np.select(
        [~rain, rain & rate_ok],
        ["Not_used_detection_negative", "Valid"],
        default="Unavailable_or_inconsistent",
    )

    high = (category == "Heavy") | (low_cbh & rain)
    moderate = low_cbh | np.isin(category, ["Light", "Moderate"])
    advisory = np.select(
        [~assess, high, moderate],
        ["Insufficient_Data", "High", "Moderate"], default="Low",
    )

    df["coordinates_valid"] = coord_ok.to_numpy()
    df["cloud_type_valid"] = cloud_ok.to_numpy()
    df["rain_probability_valid"] = prob_ok.to_numpy()
    df["cloud_condition"] = np.select(
        [~cloud_ok, clear, cloud.isin([1, 4]) & cloud_ok],
        ["Unavailable", "Clear", "Low_level_cloud"], default="Cloud_present",
    )
    df["cbh_status"] = cbh_status
    df["cbh_available_for_cloudy_pixel"] = (cbh_app & cbh_ok).to_numpy()
    df["low_cbh_flag"] = low_cbh.to_numpy()
    df["rain_detected"] = rain.to_numpy()
    df["rain_intensity_status"] = intensity_status
    df["rain_category"] = category
    df["advisory_level"] = advisory

    # Vectorized recommendations (same hazard wording as the original rules).
    hazard_code = (
        low_cbh.to_numpy(dtype=np.int8)
        + 2 * rain.to_numpy(dtype=np.int8)
        + 4 * (category == "Heavy").astype(np.int8)
    )
    hazard_texts = np.array([
        "", "low cloud-base", "precipitation", "low cloud-base, precipitation",
        "", "", "precipitation, heavy precipitation",
        "low cloud-base, precipitation, heavy precipitation",
    ], dtype=object)
    hazard = hazard_texts[hazard_code]
    insufficient = advisory == "Insufficient_Data"
    no_hazard = hazard_code == 0

    aviation = np.where(
        insufficient,
        "Insufficient model inputs. Consult authoritative meteorological information and applicable procedures.",
        np.where(
            no_hazard,
            "No low-CBH or precipitation trigger was identified. Continue using current official weather reports and applicable minima.",
            "Review current aerodrome weather reports, visibility/ceiling observations, and applicable procedures for: "
            + hazard + ".",
        ),
    )
    military = np.where(
        insufficient,
        "Insufficient model inputs. Obtain independent meteorological information for mission planning.",
        np.where(
            no_hazard,
            "No low-CBH or precipitation trigger was identified. Continue mission-specific meteorological assessment.",
            "Review local cloud-base and precipitation conditions against mission-specific operating limits for: "
            + hazard + ".",
        ),
    )
    cbh_missing = cbh_status == "Unavailable"
    intensity_missing = intensity_status == "Unavailable_or_inconsistent"
    aviation = np.where(cbh_missing, aviation + " CBH estimate unavailable.", aviation)
    military = np.where(cbh_missing, military + " CBH estimate unavailable.", military)
    aviation = np.where(
        intensity_missing, aviation + " Rain intensity estimate requires review.", aviation
    )
    military = np.where(
        intensity_missing, military + " Rain intensity estimate requires review.", military
    )
    df["aviation_recommendation"] = aviation
    df["military_recommendation"] = military

    size = a.grid_size
    glat = np.floor((lat + 90.0) / size) * size - 90.0 + size / 2
    glon = np.floor((lon + 180.0) / size) * size - 180.0 + size / 2
    df["grid_lat_center"] = glat.where(coord_ok)
    df["grid_lon_center"] = glon.where(coord_ok)

    spatial = coord_ok & ts.notna()
    if spatial.any():
        s = pd.DataFrame({
            "decision_timestamp": ts.loc[spatial],
            "grid_lat_center": glat.loc[spatial],
            "grid_lon_center": glon.loc[spatial],
            "is_assessable": assess.loc[spatial].astype("int8"),
            "is_low": (advisory == "Low")[spatial].astype("int8"),
            "is_moderate": (advisory == "Moderate")[spatial].astype("int8"),
            "is_high": (advisory == "High")[spatial].astype("int8"),
            "is_low_cbh": low_cbh.loc[spatial].astype("int8"),
            "cloudy": cbh_app.loc[spatial].astype("int8"),
            "cbh_available": (cbh_app & cbh_ok).loc[spatial].astype("int8"),
            "rain_prob_available": prob_ok.loc[spatial].astype("int8"),
            "rain_detected": rain.loc[spatial].astype("int8"),
            "intensity_available": (rain & rate_ok).loc[spatial].astype("int8"),
            "heavy_rain": (category == "Heavy")[spatial].astype("int8"),
            "rate": rain_rate.loc[spatial].where((rain & rate_ok).loc[spatial]),
        })
        gg = s.groupby(
            ["decision_timestamp", "grid_lat_center", "grid_lon_center"],
            sort=False, dropna=True,
        ).agg(
            total_input_pixels=("is_assessable", "size"),
            assessable_pixels=("is_assessable", "sum"),
            low_pixels=("is_low", "sum"),
            moderate_pixels=("is_moderate", "sum"),
            high_pixels=("is_high", "sum"),
            cloudy_pixels=("cloudy", "sum"),
            cbh_available_pixels=("cbh_available", "sum"),
            low_cbh_pixels=("is_low_cbh", "sum"),
            rain_probability_available_pixels=("rain_prob_available", "sum"),
            rain_detected_pixels=("rain_detected", "sum"),
            rain_intensity_available_when_detected_pixels=("intensity_available", "sum"),
            heavy_rain_pixels=("heavy_rain", "sum"),
            rate_sum=("rate", "sum"),
            rate_count=("rate", "count"),
            max_rate=("rate", "max"),
        )
    else:
        gg = pd.DataFrame()

    timed = ts.notna()
    if timed.any():
        t = pd.DataFrame({
            "decision_timestamp": ts.loc[timed],
            "is_assessable": assess.loc[timed].astype("int8"),
            "is_low": (advisory == "Low")[timed].astype("int8"),
            "is_moderate": (advisory == "Moderate")[timed].astype("int8"),
            "is_high": (advisory == "High")[timed].astype("int8"),
            "cloudy": cbh_app.loc[timed].astype("int8"),
            "cbh_available": (cbh_app & cbh_ok).loc[timed].astype("int8"),
            "rain_prob_available": prob_ok.loc[timed].astype("int8"),
            "rain_detected": rain.loc[timed].astype("int8"),
            "intensity_available": (rain & rate_ok).loc[timed].astype("int8"),
            "heavy_rain": (category == "Heavy")[timed].astype("int8"),
            "rate": rain_rate.loc[timed].where((rain & rate_ok).loc[timed]),
        })
        tg = t.groupby("decision_timestamp", sort=False).agg(
            total_input_pixels=("is_assessable", "size"),
            assessable_pixels=("is_assessable", "sum"),
            low_pixels=("is_low", "sum"),
            moderate_pixels=("is_moderate", "sum"),
            high_pixels=("is_high", "sum"),
            cloudy_pixels=("cloudy", "sum"),
            cbh_available_pixels=("cbh_available", "sum"),
            rain_probability_available_pixels=("rain_prob_available", "sum"),
            rain_detected_pixels=("rain_detected", "sum"),
            rain_intensity_available_when_detected_pixels=("intensity_available", "sum"),
            heavy_rain_pixels=("heavy_rain", "sum"),
            rate_sum=("rate", "sum"),
            rate_count=("rate", "count"),
            max_rate=("rate", "max"),
        )
    else:
        tg = pd.DataFrame()
    return df, gg, tg


def upsert_grouped(conn, grouped, table, metrics, grid_keys=False):
    if grouped.empty:
        return
    rows = []
    if grid_keys:
        for (ts, lat, lon), r in grouped.iterrows():
            rows.append((
                pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"), float(lat), float(lon),
                *[int(r[c]) for c in metrics],
                float(r["rate_sum"]), int(r["rate_count"]),
                None if pd.isna(r["max_rate"]) else float(r["max_rate"]),
            ))
        keys = "decision_timestamp,grid_lat_center,grid_lon_center"
        prefix = 3
    else:
        for ts, r in grouped.iterrows():
            rows.append((
                pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                *[int(r[c]) for c in metrics],
                float(r["rate_sum"]), int(r["rate_count"]),
                None if pd.isna(r["max_rate"]) else float(r["max_rate"]),
            ))
        keys = "decision_timestamp"
        prefix = 1

    n = prefix + len(metrics) + 3
    placeholders = ",".join(["?"] * n)
    updates = [f'"{c}"={table}."{c}"+excluded."{c}"' for c in metrics]
    updates += [
        f'"rate_sum"={table}.rate_sum+excluded.rate_sum',
        f'"rate_count"={table}.rate_count+excluded.rate_count',
        f'"max_rate"=CASE WHEN {table}.max_rate IS NULL THEN excluded.max_rate '
        f'WHEN excluded.max_rate IS NULL THEN {table}.max_rate '
        f'ELSE MAX({table}.max_rate,excluded.max_rate) END',
    ]
    sql = (
        f"INSERT INTO {table} VALUES ({placeholders}) "
        f"ON CONFLICT({keys}) DO UPDATE SET " + ",".join(updates)
    )
    conn.executemany(sql, rows)


def finalize_grid(conn, min_pixels):
    grid = pd.read_sql_query(
        """SELECT * FROM grid_agg
           ORDER BY decision_timestamp,grid_lat_center,grid_lon_center""",
        conn, parse_dates=["decision_timestamp"],
    )
    grid["mean_detected_rain_rate_mm_hr"] = (
        grid["rate_sum"] / grid["rate_count"].replace(0, np.nan)
    )
    grid["max_detected_rain_rate_mm_hr"] = grid["max_rate"]
    grid["assessable_coverage_pct"] = pct(grid.assessable_pixels, grid.total_input_pixels)
    for level in ["low", "moderate", "high"]:
        grid[f"{level}_pct"] = pct(
            grid[f"{level}_pixels"], grid.assessable_pixels
        )
    grid["cbh_unavailable_cloudy_pixels"] = (
        grid.cloudy_pixels - grid.cbh_available_pixels
    )
    grid["rain_intensity_unavailable_detected_pixels"] = (
        grid.rain_detected_pixels - grid.rain_intensity_available_when_detected_pixels
    )
    grid["no_rain_detected_pixels"] = grid.assessable_pixels - grid.rain_detected_pixels
    grid["cbh_availability_pct_among_cloudy"] = pct(
        grid.cbh_available_pixels, grid.cloudy_pixels
    )
    grid["low_cbh_pct_among_cloudy"] = pct(grid.low_cbh_pixels, grid.cloudy_pixels)
    grid["rain_probability_availability_pct"] = pct(
        grid.rain_probability_available_pixels, grid.total_input_pixels
    )
    grid["rain_detected_pct_among_assessable"] = pct(
        grid.rain_detected_pixels, grid.assessable_pixels
    )
    grid["rain_intensity_availability_pct_among_detected"] = pct(
        grid.rain_intensity_available_when_detected_pixels,
        grid.rain_detected_pixels,
    )
    grid["heavy_rain_pct_among_detected"] = pct(
        grid.heavy_rain_pixels, grid.rain_detected_pixels
    )
    grid["grid_data_status"] = np.where(
        grid.assessable_pixels >= min_pixels, "Assessable", "Insufficient_Data"
    )
    grid["maximum_pixel_advisory_level"] = np.select(
        [grid.high_pixels > 0, grid.moderate_pixels > 0],
        ["High", "Moderate"], default="Low",
    )
    grid.drop(columns=["rate_sum", "rate_count", "max_rate"], inplace=True)
    return grid


def finalize_time(conn, grid):
    time = pd.read_sql_query(
        """SELECT * FROM time_agg ORDER BY decision_timestamp""",
        conn, parse_dates=["decision_timestamp"],
    )
    time["mean_detected_rain_rate_mm_hr"] = (
        time.rate_sum / time.rate_count.replace(0, np.nan)
    )
    time["max_detected_rain_rate_mm_hr"] = time.max_rate
    time["assessable_coverage_pct"] = pct(time.assessable_pixels, time.total_input_pixels)
    for level in ["low", "moderate", "high"]:
        time[f"{level}_pct"] = pct(time[f"{level}_pixels"], time.assessable_pixels)
    time["cbh_availability_pct_among_cloudy"] = pct(
        time.cbh_available_pixels, time.cloudy_pixels
    )
    time["rain_probability_availability_pct"] = pct(
        time.rain_probability_available_pixels, time.total_input_pixels
    )
    time["rain_detected_pct_among_assessable"] = pct(
        time.rain_detected_pixels, time.assessable_pixels
    )
    time["rain_intensity_availability_pct_among_detected"] = pct(
        time.rain_intensity_available_when_detected_pixels, time.rain_detected_pixels
    )
    time["heavy_rain_pct_among_detected"] = pct(
        time.heavy_rain_pixels, time.rain_detected_pixels
    )
    time["cloudy_pixels_without_valid_cbh"] = (
        time.cloudy_pixels - time.cbh_available_pixels
    )
    time["detected_rain_pixels_without_valid_intensity"] = (
        time.rain_detected_pixels - time.rain_intensity_available_when_detected_pixels
    )
    time["no_rain_detected_pixels"] = time.assessable_pixels - time.rain_detected_pixels

    assessable_grid = grid[grid.grid_data_status == "Assessable"].copy()
    if not assessable_grid.empty:
        region = assessable_grid.groupby("decision_timestamp").agg(
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
        region["assessable_grid_cbh_unavailable_cloudy_pixels"] = (
            region.assessable_grid_cloudy_pixels
            - region.assessable_grid_cbh_available_pixels
        )
        time = time.merge(region, left_on="decision_timestamp", right_index=True, how="left")
        for label in ["low", "moderate", "high"]:
            time[f"assessable_grid_{label}_pct_pixel_weighted"] = pct(
                time[f"assessable_grid_{label}_pixels"],
                time.assessable_grid_cell_pixels,
            )
        time["assessable_grid_cbh_availability_pct_pixel_weighted"] = pct(
            time.assessable_grid_cbh_available_pixels,
            time.assessable_grid_cloudy_pixels,
        )
        time["assessable_grid_cbh_unavailable_pct_pixel_weighted"] = pct(
            time.assessable_grid_cbh_unavailable_cloudy_pixels,
            time.assessable_grid_cloudy_pixels,
        )
        time["assessable_grid_rain_intensity_availability_pct_pixel_weighted"] = pct(
            time.assessable_grid_rain_intensity_available_pixels,
            time.assessable_grid_rain_detected_pixels,
        )
        means = assessable_grid.groupby("decision_timestamp").agg(
            mean_grid_cbh_availability_pct=("cbh_availability_pct_among_cloudy", "mean"),
            mean_grid_rain_probability_availability_pct=("rain_probability_availability_pct", "mean"),
            mean_grid_rain_intensity_availability_pct=("rain_intensity_availability_pct_among_detected", "mean"),
            mean_grid_low_pct=("low_pct", "mean"),
            mean_grid_moderate_pct=("moderate_pct", "mean"),
            mean_grid_high_pct=("high_pct", "mean"),
        )
        time = time.merge(means, left_on="decision_timestamp", right_index=True, how="left")
        assessable_counts = assessable_grid.groupby("decision_timestamp").size()
    else:
        assessable_counts = pd.Series(dtype="int64")

    total_counts = grid.groupby("decision_timestamp").size()
    time["total_grid_cells"] = time.decision_timestamp.map(total_counts).fillna(0).astype(int)
    time["assessable_grid_cells"] = (
        time.decision_timestamp.map(assessable_counts).fillna(0).astype(int)
    )
    time["assessable_grid_cell_pct"] = pct(
        time.assessable_grid_cells, time.total_grid_cells
    )
    time.drop(columns=["rate_sum", "rate_count", "max_rate"], inplace=True)
    return time


def main():
    a = parse_args()
    src = Path(a.input)
    out = Path(a.output_dir)
    work = Path(a.work_dir) if a.work_dir else out / ".work"
    if not src.is_file():
        raise FileNotFoundError(src)
    if a.grid_size <= 0 or a.min_pixels < 1 or a.chunksize < 1:
        raise ValueError("grid-size, min-pixels, and chunksize must be positive.")
    if not 0 <= a.rain_prob_threshold <= 1:
        raise ValueError("rain-prob-threshold must be between 0 and 1.")
    if not (0 <= a.light_rain_min < a.moderate_rain_min < a.heavy_rain_min):
        raise ValueError("Rain thresholds must be increasing and non-negative.")

    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    pixel_path = out / "decision_pixel_level.csv"
    grid_path = out / "decision_grid_025deg.csv"
    time_path = out / "decision_timestamp_summary.csv"
    meta_path = out / "decision_run_metadata.json"
    db_path = work / "decision_aggregation.sqlite"

    existing = [p for p in [pixel_path, grid_path, time_path, meta_path, db_path] if p.exists()]
    if existing and not a.overwrite:
        raise FileExistsError(
            "Existing output/work files found. Use a new output directory or --overwrite: "
            + ", ".join(map(str, existing))
        )
    if a.overwrite:
        for p in [pixel_path, grid_path, time_path, meta_path, db_path,
                  Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
            if p.exists():
                p.unlink()

    header = pd.read_csv(src, nrows=0)
    cols = resolve_columns(header.columns, a)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    init_db(conn)

    count = 0
    chunks = 0
    try:
        for chunk in pd.read_csv(src, chunksize=a.chunksize):
            chunks += 1
            processed, grid_part, time_part = process_chunk(chunk, cols, a)
            processed.to_csv(
                pixel_path, mode="w" if chunks == 1 else "a",
                header=(chunks == 1), index=False,
            )
            upsert_grouped(conn, grid_part, "grid_agg", GRID_METRICS, True)
            upsert_grouped(conn, time_part, "time_agg", TIME_METRICS, False)
            conn.commit()
            count += len(processed)
            print(f"Chunk {chunks}: {len(processed):,} rows; total {count:,}", flush=True)
            del processed, grid_part, time_part, chunk

        if count == 0:
            raise RuntimeError("No data rows were read.")

        print("Finalizing grid and timestamp summaries...", flush=True)
        grid = finalize_grid(conn, a.min_pixels)
        time = finalize_time(conn, grid)
        grid.to_csv(grid_path, index=False)
        time.to_csv(time_path, index=False)

        metadata = {
            "input": str(src),
            "rows": int(count),
            "chunks_processed": int(chunks),
            "chunksize": int(a.chunksize),
            "columns_used": cols,
            "grid_size_degrees": a.grid_size,
            "minimum_assessable_pixels_per_grid_timestamp": a.min_pixels,
            "rain_probability_threshold": a.rain_prob_threshold,
            "low_cbh_reference_km": a.low_cbh_km,
            "rainfall_bands_mm_hr": {
                "light_from": a.light_rain_min,
                "moderate_from": a.moderate_rain_min,
                "heavy_from": a.heavy_rain_min,
            },
            "advisory_levels": ["Low", "Moderate", "High", "Insufficient_Data"],
            "critical_assigned": False,
            "grid_policy": (
                "All grid/time groups retained; groups below minimum are labelled "
                "Insufficient_Data. Regional grid-based distributions exclude "
                "insufficient grid cells."
            ),
            "regional_reporting": (
                "Pixel-weighted metrics over assessable grid cells and equal-weight "
                "means across assessable grid cells are reported separately."
            ),
            "processing": (
                "Chunked CSV processing with SQLite incremental aggregation; "
                "partial group aggregates combined before percentages are calculated."
            ),
            "warning": "Project-defined provisional rules, not official aviation risk categories.",
        }
        meta_path.write_text(json.dumps(metadata, indent=2))
        print("\nCHUNKED DECISION SUPPORT COMPLETE", flush=True)
        print(f"Rows: {count:,}; chunks: {chunks:,}", flush=True)
        print(f"Grid/time groups: {len(grid):,}", flush=True)
        print(
            "Assessable groups: "
            f"{(grid.grid_data_status == 'Assessable').sum():,}",
            flush=True,
        )
        print(
            "Insufficient groups: "
            f"{(grid.grid_data_status == 'Insufficient_Data').sum():,}",
            flush=True,
        )
        print(f"Timestamp summaries: {len(time):,}", flush=True)
        print(f"Pixel output: {pixel_path} ({pixel_path.stat().st_size / 1024**3:.2f} GiB)")
        print(f"Grid output: {grid_path} ({grid_path.stat().st_size / 1024**3:.2f} GiB)")
        print(f"Timestamp output: {time_path} ({time_path.stat().st_size / 1024**3:.2f} GiB)")
    except Exception:
        conn.rollback()
        print(
            "FAILED. Partial pixel output and SQLite work database retained; "
            "do not treat partial outputs as complete.",
            file=sys.stderr,
        )
        raise
    finally:
        conn.close()

    # Remove SQLite temporary files only after successful final output creation.
    for p in [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
        if p.exists():
            p.unlink()
    try:
        work.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()

