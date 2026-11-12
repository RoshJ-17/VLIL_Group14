#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a unified aviation and military operational profile."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-id", default="unspecified")
    parser.add_argument(
     "--temporal-dir",
     default=None,
     help="Optional directory containing temporal analysis outputs."
    )
    return parser.parse_args()


def clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def records_from_dataframe(df):
    return [
        {key: clean_value(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def recommendation_counts(df, column):
    if column not in df.columns:
        return []

    counts = df[column].dropna().value_counts()

    return [
        {
            "recommendation": str(recommendation),
            "pixel_count": int(count)
        }
        for recommendation, count in counts.items()
    ]


def format_recommendations(title, recommendations):
    lines = [title]

    if not recommendations:
        lines.append("  No recommendation data available.")
        return lines

    for item in recommendations:
        lines.append(
            f"  [{item['pixel_count']:,} pixels] "
            f"{item['recommendation']}"
        )

    return lines


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_file = input_dir / "decision_pixel_level.csv"
    grid_file = input_dir / "decision_grid_025deg.csv"
    timestamp_file = input_dir / "decision_timestamp_summary.csv"

    for path in (pixel_file, grid_file, timestamp_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    print("Loading decision-support outputs...", flush=True)

    pixel = pd.read_csv(pixel_file)
    grid = pd.read_csv(grid_file)
    timestamp = pd.read_csv(timestamp_file)

    temporal_grid = None
    temporal_events = None
    temporal_timeline = None
    temporal_metadata = None

    if args.temporal_dir:
        temporal_dir = Path(args.temporal_dir)

        temporal_grid_file = temporal_dir / "temporal_grid_analysis.csv"
        temporal_event_file = temporal_dir / "temporal_event_log.csv"
        temporal_timeline_file = temporal_dir / "temporal_timeline.csv"
        temporal_json_file = temporal_dir / "temporal_operational_report.json"

        temporal_files = [
            temporal_grid_file,
            temporal_event_file,
            temporal_timeline_file,
            temporal_json_file
        ]

        for path in temporal_files:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required temporal output not found: {path}"
                )

        print("Loading temporal analysis outputs...", flush=True)

        temporal_grid = pd.read_csv(temporal_grid_file)

        try:
            temporal_events = pd.read_csv(temporal_event_file)
        except pd.errors.EmptyDataError:
            temporal_events = pd.DataFrame()

        temporal_timeline = pd.read_csv(temporal_timeline_file)

        with temporal_json_file.open(encoding="utf-8") as f:
            temporal_metadata = json.load(f)
    required_pixel_columns = {
        "timestamp",
        "advisory_level",
        "aviation_recommendation",
        "military_recommendation"
    }

    required_grid_columns = {
        "grid_data_status",
        "maximum_pixel_advisory_level"
    }

    required_timestamp_columns = {
        "decision_timestamp",
        "total_input_pixels",
        "assessable_pixels",
        "low_pixels",
        "moderate_pixels",
        "high_pixels"
    }

    for name, df, required in [
        ("pixel", pixel, required_pixel_columns),
        ("grid", grid, required_grid_columns),
        ("timestamp", timestamp, required_timestamp_columns)
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} file is missing required columns: {sorted(missing)}"
            )

    advisory_counts = {
        str(key): int(value)
        for key, value in pixel["advisory_level"]
        .value_counts(dropna=False).items()
    }

    grid_status_counts = {
        str(key): int(value)
        for key, value in grid["grid_data_status"]
        .value_counts(dropna=False).items()
    }

    aviation_recommendations = recommendation_counts(
        pixel, "aviation_recommendation"
    )

    military_recommendations = recommendation_counts(
        pixel, "military_recommendation"
    )

    profile = {
        "batch_id": args.batch_id,
        "profile_name": "Unified Operational Profile",
        "recommendation_perspectives": [
            "Aviation",
            "Military"
        ],
        "input_files": {
            "pixel": str(pixel_file),
            "grid": str(grid_file),
            "timestamp": str(timestamp_file)
        },
        "data_overview": {
            "total_pixels": int(len(pixel)),
            "grid_time_groups": int(len(grid)),
            "timestamps": int(len(timestamp)),
            "advisory_level_counts": advisory_counts,
            "grid_data_status_counts": grid_status_counts
        },
        "recommendations": {
            "aviation": aviation_recommendations,
            "military": military_recommendations
        },
        "timestamp_summaries": records_from_dataframe(timestamp),
        "grid_summaries": records_from_dataframe(grid)
    }
    if temporal_metadata is not None:
        profile["input_files"]["temporal"] = {
            "grid": str(temporal_grid_file),
            "events": str(temporal_event_file),
            "timeline": str(temporal_timeline_file),
            "metadata": str(temporal_json_file)
        }

        profile["temporal_analysis"] = {
            "metadata": temporal_metadata,
            "event_counts": temporal_metadata.get("event_counts", {}),
            "grid_summaries": records_from_dataframe(temporal_grid),
            "event_log": records_from_dataframe(temporal_events),
            "timeline": records_from_dataframe(temporal_timeline)
        }
    json_file = output_dir / "operational_profile_report.json"
    text_file = output_dir / "operational_profile_report.txt"

    json_file.write_text(
        json.dumps(profile, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    lines = [
        "UNIFIED OPERATIONAL PROFILE",
        "=" * 70,
        f"Batch ID: {args.batch_id}",
        "",
        "1. DATA OVERVIEW",
        "-" * 70,
        f"Total pixels processed: {len(pixel):,}",
        f"Grid/time groups: {len(grid):,}",
        f"Timestamps: {len(timestamp):,}",
        "",
        "2. ADVISORY LEVEL DISTRIBUTION",
        "-" * 70
    ]

    for level, count in advisory_counts.items():
        lines.append(f"{level}: {count:,} pixels")

    lines.extend([
        "",
        "3. GRID DATA STATUS",
        "-" * 70
    ])

    for status, count in grid_status_counts.items():
        lines.append(f"{status}: {count:,} grid/time groups")

    lines.extend([
        "",
        "4. RECOMMENDATIONS",
        "-" * 70,
        "Unified profile containing two distinct perspectives.",
        ""
    ])

    lines.extend(
        format_recommendations(
            "AVIATION-SPECIFIC RECOMMENDATIONS",
            aviation_recommendations
        )
    )

    lines.append("")

    lines.extend(
        format_recommendations(
            "MILITARY-SPECIFIC RECOMMENDATIONS",
            military_recommendations
        )
    )

    lines.extend([
        "",
        "5. TIMESTAMP-WISE WEATHER SUMMARY",
        "-" * 70
    ])

    for row in records_from_dataframe(timestamp):
        lines.extend([
            "",
            f"Timestamp: {row.get('decision_timestamp')}",
            f"  Total pixels: {row.get('total_input_pixels')}",
            f"  Assessable pixels: {row.get('assessable_pixels')}",
            f"  Low advisory pixels: {row.get('low_pixels')}",
            f"  Moderate advisory pixels: {row.get('moderate_pixels')}",
            f"  High advisory pixels: {row.get('high_pixels')}",
            f"  Assessable coverage: {row.get('assessable_coverage_pct')}%",
            f"  Rain detected pixels: {row.get('rain_detected_pixels')}",
            f"  Heavy rain pixels: {row.get('heavy_rain_pixels')}",
            f"  Mean detected rain rate: "
            f"{row.get('mean_detected_rain_rate_mm_hr')} mm/hr",
            f"  Maximum detected rain rate: "
            f"{row.get('max_detected_rain_rate_mm_hr')} mm/hr"
        ])

    lines.extend([
        "",
        "6. GRID-LEVEL SUMMARY",
        "-" * 70,
        f"Total grid/time groups: {len(grid):,}"
    ])

    if "assessable_pixels" in grid.columns:
        lines.append(
            "Total assessable pixels across grid/time groups: "
            f"{int(grid['assessable_pixels'].sum()):,}"
        )

    if "high_pixels" in grid.columns:
        lines.append(
            f"High advisory pixels across grid/time groups: "
            f"{int(grid['high_pixels'].sum()):,}"
        )


    if temporal_metadata is not None:
        lines.extend([
            "",
            "7. TEMPORAL ANALYSIS OVERVIEW",
            "-" * 70,
            f"Grid/time records: {temporal_metadata.get('grid_time_records')}",
            f"Assessable grid/time records: {temporal_metadata.get('assessable_grid_time_records')}",
            f"Grid cells spanning analysis window: {temporal_metadata.get('grid_cells_spanning_window')}",
            f"Trend window: {temporal_metadata.get('window_minutes')} minutes",
            f"Grid size: {temporal_metadata.get('grid_size_degrees')} degrees",
            f"Low-CBH reference: {temporal_metadata.get('low_cbh_reference_km')} km",
            f"Rain probability threshold: {temporal_metadata.get('rain_probability_threshold')}",
            "",
            "8. TEMPORAL EVENT SUMMARY",
            "-" * 70
        ])

        event_counts = temporal_metadata.get("event_counts", {})

        if event_counts:
            for event, count in event_counts.items():
                lines.append(f"{event}: {count}")
        else:
            lines.append("No temporal events detected.")

        lines.extend([
            "",
            "9. DETECTED TEMPORAL EVENTS",
            "-" * 70
        ])

        if temporal_events is not None and not temporal_events.empty:
            for row in records_from_dataframe(temporal_events):
                lines.extend([
                    "",
                    f"Event: {row.get('event_type')}",
                    f"Timestamp: {row.get('timestamp')}",
                    f"Grid location: {row.get('grid_lat_center')}, "
                    f"{row.get('grid_lon_center')}",
                    f"Previous CBH: {row.get('previous_cbh_km')} km",
                    f"Current CBH: {row.get('current_cbh_km')} km",
                    f"CBH change: {row.get('cbh_change_km')} km",
                    f"Previous rain pixels: {row.get('previous_rain_pixels')}",
                    f"Current rain pixels: {row.get('current_rain_pixels')}",
                    f"Previous mean rain rate: "
                    f"{row.get('previous_mean_detected_rain_rate_mm_hr')} mm/hr",
                    f"Current mean rain rate: "
                    f"{row.get('current_mean_detected_rain_rate_mm_hr')} mm/hr"
                ])
        else:
            lines.append("No temporal event records available.")

        lines.extend([
            "",
            "10. AVIATION TEMPORAL RECOMMENDATIONS",
            "-" * 70
        ])

        if temporal_events is not None and not temporal_events.empty:
            for row in records_from_dataframe(temporal_events):
                lines.extend([
                    "",
                    f"{row.get('timestamp')} | "
                    f"{row.get('event_type')} | "
                    f"Grid: {row.get('grid_lat_center')}, "
                    f"{row.get('grid_lon_center')}",
                    f"Recommendation: {row.get('aviation_recommendation')}"
                ])
        else:
            lines.append("No aviation temporal recommendations available.")

        lines.extend([
            "",
            "11. MILITARY TEMPORAL RECOMMENDATIONS",
            "-" * 70
        ])

        if temporal_events is not None and not temporal_events.empty:
            for row in records_from_dataframe(temporal_events):
                lines.extend([
                    "",
                    f"{row.get('timestamp')} | "
                    f"{row.get('event_type')} | "
                    f"Grid: {row.get('grid_lat_center')}, "
                    f"{row.get('grid_lon_center')}",
                    f"Recommendation: {row.get('military_recommendation')}"
                ])
        else:
            lines.append("No military temporal recommendations available.")

        lines.extend([
            "",
            "12. TEMPORAL ANALYSIS INTERPRETATION AND LIMITATIONS",
            "-" * 70
        ])

        for limitation in temporal_metadata.get("limitations", []):
            lines.append(f"- {limitation}")

        lines.extend([
            "",
            "Temporal trends are calculated from successive assessable "
            "observations in the same grid cell.",
            "CBH trends use cloudy pixels with valid non-negative CBH.",
            "Precipitation transitions use the configured rain-probability "
            "threshold.",
            "Rainfall-rate comparisons use detected-rain pixels only.",
            "A missing observation is not treated as no rain or as a "
            "stable condition."
        ])

    lines.extend([
        "",
        "END OF REPORT",
        ""
    ])

    text_file.write_text("\n".join(lines), encoding="utf-8")

    print("\nOPERATIONAL PROFILE REPORT GENERATED")
    print("Text report:", text_file)
    print("JSON report:", json_file)


if __name__ == "__main__":
    main()
