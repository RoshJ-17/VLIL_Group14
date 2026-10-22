#!/usr/bin/env python3
"""
Investigate extreme BTD observations in cloudtype_final.csv.

Analyses:
1. Extract the largest positive and negative BTD observations.
2. Count extreme observations by cloud class.
3. Summarize positive and negative BTD separately.
4. Examine spatial and temporal concentrations.
5. Generate plots for interpretation.

All outputs are saved under:
    eda_results/cloudType/extreme_btd_investigation/

No rows are deleted or modified.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASSES = [
    "clear",
    "liquid_low",
    "liquid_middle",
    "liquid_high",
    "ice_low",
    "ice_middle",
    "ice_high",
]

INPUT_FILE = Path("cloudtype_final.csv")

OUTPUT_DIR = (
    Path("eda_results")
    / "cloudType"
    / "extreme_btd_investigation"
)

CHUNK_SIZE = 100_000
TOP_N = 500

REQUIRED_COLUMNS = [
    "timestamp",
    "lat",
    "lon",
    "BT_11",
    "BT_12",
    "BTD",
    "cloud_type_label",
]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting extreme BTD investigation...")

    # Full-dataset counters
    class_counts = {
        cls: {
            "total": 0,
            "positive_gt_5": 0,
            "negative_lt_minus5": 0,
            "absolute_gt_5": 0,
            "absolute_gt_10": 0,
        }
        for cls in CLASSES
    }

    # Temporal and spatial aggregation
    time_counts = {}
    spatial_counts = {}

    # Keep only the most extreme records
    top_positive = pd.DataFrame()
    top_negative = pd.DataFrame()

    rows_scanned = 0

    reader = pd.read_csv(
        args.input,
        usecols=lambda c: c in REQUIRED_COLUMNS,
        chunksize=args.chunksize,
    )

    for chunk in reader:

        missing = set(REQUIRED_COLUMNS) - set(chunk.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        rows_scanned += len(chunk)

        for col in ["lat", "lon", "BT_11", "BT_12", "BTD"]:
            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="coerce"
            )

        chunk = chunk.dropna(
            subset=[
                "lat",
                "lon",
                "BT_11",
                "BT_12",
                "BTD",
                "cloud_type_label",
            ]
        )

        chunk["abs_BTD"] = chunk["BTD"].abs()

        # -------------------------------------------------
        # 1. Full-dataset class statistics
        # -------------------------------------------------

        for cls, group in chunk.groupby("cloud_type_label"):

            if cls not in class_counts:
                continue

            btd = group["BTD"]

            class_counts[cls]["total"] += len(group)

            class_counts[cls]["positive_gt_5"] += int(
                (btd > 5).sum()
            )

            class_counts[cls]["negative_lt_minus5"] += int(
                (btd < -5).sum()
            )

            class_counts[cls]["absolute_gt_5"] += int(
                (btd.abs() > 5).sum()
            )

            class_counts[cls]["absolute_gt_10"] += int(
                (btd.abs() > 10).sum()
            )

        # -------------------------------------------------
        # 2. Top positive and negative BTD observations
        # -------------------------------------------------

        positive = chunk.nlargest(TOP_N, "BTD")

        negative = chunk.nsmallest(TOP_N, "BTD")

        top_positive = (
            pd.concat(
                [top_positive, positive],
                ignore_index=True
            )
            .nlargest(TOP_N, "BTD")
        )

        top_negative = (
            pd.concat(
                [top_negative, negative],
                ignore_index=True
            )
            .nsmallest(TOP_N, "BTD")
        )

        # -------------------------------------------------
        # 3. Temporal concentration
        # -------------------------------------------------

        extreme = chunk[chunk["abs_BTD"] > 5].copy()

        if not extreme.empty:

            extreme["date"] = (
                extreme["timestamp"]
                .astype(str)
                .str.slice(0, 7)
            )

            counts = extreme.groupby("date").size()

            for date, count in counts.items():
                time_counts[date] = (
                    time_counts.get(date, 0) + int(count)
                )

            # -------------------------------------------------
            # 4. Spatial concentration
            # -------------------------------------------------

            # One-degree latitude/longitude bins
            extreme["lat_bin"] = np.floor(
                extreme["lat"]
            ).astype(int)

            extreme["lon_bin"] = np.floor(
                extreme["lon"]
            ).astype(int)

            grouped = (
                extreme.groupby(["lat_bin", "lon_bin"])
                .size()
            )

            for (lat_bin, lon_bin), count in grouped.items():

                key = (int(lat_bin), int(lon_bin))

                spatial_counts[key] = (
                    spatial_counts.get(key, 0) + int(count)
                )

        if rows_scanned % (args.chunksize * 10) == 0:
            print(f"Scanned {rows_scanned:,} rows...")

    # -----------------------------------------------------
    # 5. Save class summary
    # -----------------------------------------------------

    records = []

    for cls in CLASSES:

        data = class_counts[cls]
        total = data["total"]

        if total == 0:
            continue

        records.append({
            "cloud_type_label": cls,
            "total_observations": total,
            "positive_BTD_gt_5_count": data["positive_gt_5"],
            "negative_BTD_lt_minus5_count": data["negative_lt_minus5"],
            "absolute_BTD_gt_5_count": data["absolute_gt_5"],
            "absolute_BTD_gt_5_percent": (
                100 * data["absolute_gt_5"] / total
            ),
            "absolute_BTD_gt_10_count": data["absolute_gt_10"],
            "absolute_BTD_gt_10_percent": (
                100 * data["absolute_gt_10"] / total
            ),
        })

    class_summary = pd.DataFrame(records)

    class_summary.to_csv(
        OUTPUT_DIR / "extreme_btd_by_class.csv",
        index=False
    )

    # -----------------------------------------------------
    # 6. Save top extreme observations
    # -----------------------------------------------------

    top_positive.to_csv(
        OUTPUT_DIR / "top_500_positive_btd.csv",
        index=False
    )

    top_negative.to_csv(
        OUTPUT_DIR / "top_500_negative_btd.csv",
        index=False
    )

    # -----------------------------------------------------
    # 7. Save temporal summary
    # -----------------------------------------------------

    temporal = pd.DataFrame(
        [
            {"date": date, "extreme_observation_count": count}
            for date, count in sorted(time_counts.items())
        ]
    )

    temporal.to_csv(
        OUTPUT_DIR / "extreme_btd_by_date.csv",
        index=False
    )

    # -----------------------------------------------------
    # 8. Save spatial summary
    # -----------------------------------------------------

    spatial = pd.DataFrame(
        [
            {
                "lat_bin": lat,
                "lon_bin": lon,
                "extreme_observation_count": count,
            }
            for (lat, lon), count in spatial_counts.items()
        ]
    )

    if not spatial.empty:
        spatial = spatial.sort_values(
            "extreme_observation_count",
            ascending=False
        )

    spatial.to_csv(
        OUTPUT_DIR / "extreme_btd_by_spatial_grid.csv",
        index=False
    )

    # -----------------------------------------------------
    # 9. Plot extreme BTD counts by class
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        class_summary["cloud_type_label"],
        class_summary["positive_BTD_gt_5_count"],
        label="BTD > 5 K",
    )

    ax.bar(
        class_summary["cloud_type_label"],
        class_summary["negative_BTD_lt_minus5_count"],
        bottom=class_summary["positive_BTD_gt_5_count"],
        label="BTD < -5 K",
    )

    ax.set_title("Extreme BTD observations by cloud class")
    ax.set_xlabel("Cloud class")
    ax.set_ylabel("Number of observations")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "extreme_btd_counts_by_class.png",
        dpi=180
    )

    plt.close(fig)

    # -----------------------------------------------------
    # 10. Plot temporal concentration
    # -----------------------------------------------------

    if not temporal.empty:

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            temporal["date"],
            temporal["extreme_observation_count"],
            marker="o",
        )

        ax.set_title(
            "Temporal distribution of observations with |BTD| > 5 K"
        )

        ax.set_xlabel("Timestamp group")
        ax.set_ylabel("Observation count")
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR / "extreme_btd_temporal_distribution.png",
            dpi=180
        )

        plt.close(fig)

    # -----------------------------------------------------
    # 11. Plot spatial concentration
    # -----------------------------------------------------

    if not spatial.empty:

        fig, ax = plt.subplots(figsize=(10, 7))

        scatter = ax.scatter(
            spatial["lon_bin"] + 0.5,
            spatial["lat_bin"] + 0.5,
            c=spatial["extreme_observation_count"],
            s=25,
        )

        ax.set_title(
            "Spatial distribution of observations with |BTD| > 5 K"
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        fig.colorbar(
            scatter,
            ax=ax,
            label="Extreme observation count"
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR / "extreme_btd_spatial_distribution.png",
            dpi=180
        )

        plt.close(fig)

    # -----------------------------------------------------
    # 12. Write report
    # -----------------------------------------------------

    report = []

    report.append("EXTREME BTD INVESTIGATION")
    report.append("=========================")
    report.append(f"Rows scanned: {rows_scanned:,}")
    report.append("")
    report.append("EXTREME BTD COUNTS BY CLASS")
    report.append("---------------------------")
    report.append(class_summary.to_string(index=False))
    report.append("")

    report.append("TOP 10 POSITIVE BTD OBSERVATIONS")
    report.append("--------------------------------")
    report.append(
        top_positive.head(10).to_string(index=False)
    )
    report.append("")

    report.append("TOP 10 NEGATIVE BTD OBSERVATIONS")
    report.append("--------------------------------")
    report.append(
        top_negative.head(10).to_string(index=False)
    )
    report.append("")

    report.append("TOP 20 SPATIAL GRID CELLS")
    report.append("-------------------------")

    if not spatial.empty:
        report.append(
            spatial.head(20).to_string(index=False)
        )
    else:
        report.append("No extreme observations found.")

    report.append("")
    report.append(
        "Note: BTD thresholds are inspection flags, not "
        "automatic invalidity criteria."
    )
    report.append(
        "Spatial bins are 1-degree latitude/longitude cells."
    )
    report.append(
        "Temporal groups use the first seven characters "
        "of the timestamp."
    )
    report.append(
        "No observations were removed or modified."
    )

    (OUTPUT_DIR / "EXTREME_BTD_INVESTIGATION_REPORT.txt").write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\nInvestigation complete.")
    print(f"Rows scanned: {rows_scanned:,}")
    print(f"Results saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
