#!/usr/bin/env python3
"""
Investigate repeated brightness-temperature values.

All outputs are saved under:
eda_results/cloudType/repeated_bt_investigation/

The original dataset is never modified.
"""

from pathlib import Path
import argparse
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = (
    Path("eda_results")
    / "cloudType"
    / "repeated_bt_investigation"
)

CLASSES = [
    "clear",
    "liquid_low",
    "liquid_middle",
    "liquid_high",
    "ice_low",
    "ice_middle",
    "ice_high",
]

REQUIRED = [
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
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cloudtype_final.csv")
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count exact two-decimal BT values and BT pairs.
    bt11_counts = Counter()
    bt12_counts = Counter()
    pair_counts = Counter()

    # Track classes associated with frequent pairs.
    pair_class_counts = {}

    # Track occurrences of suspicious candidate values.
    candidate_values = {
        (486.80, 495.75),
    }

    candidate_records = []

    rows_scanned = 0

    reader = pd.read_csv(
        args.input,
        usecols=lambda c: c in REQUIRED,
        chunksize=args.chunksize
    )

    for chunk in reader:

        missing = set(REQUIRED) - set(chunk.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        rows_scanned += len(chunk)

        for col in ["lat", "lon", "BT_11", "BT_12", "BTD"]:
            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="coerce"
            )

        chunk = chunk.dropna(
            subset=["BT_11", "BT_12"]
        )

        # Round to 2 decimal places to match CSV precision.
        bt11 = chunk["BT_11"].round(2)
        bt12 = chunk["BT_12"].round(2)

        bt11_counts.update(bt11.tolist())
        bt12_counts.update(bt12.tolist())

        pairs = list(zip(bt11.tolist(), bt12.tolist()))
        pair_counts.update(pairs)

        # Class-wise pair frequencies.
        temp = pd.DataFrame({
            "BT_11": bt11,
            "BT_12": bt12,
            "cloud_type_label": chunk["cloud_type_label"].values,
        })

        grouped = (
            temp.groupby(
                ["BT_11", "BT_12", "cloud_type_label"]
            )
            .size()
        )

        for (v11, v12, cls), count in grouped.items():

            key = (float(v11), float(v12))

            if key not in pair_class_counts:
                pair_class_counts[key] = Counter()

            pair_class_counts[key][cls] += int(count)

        # Save all records matching the known suspicious pair.
        mask = (
            (bt11 == 486.80)
            & (bt12 == 495.75)
        )

        if mask.any():

            selected = chunk.loc[mask].copy()

            candidate_records.append(
                selected[REQUIRED]
            )

        if rows_scanned % (args.chunksize * 10) == 0:
            print(f"Scanned {rows_scanned:,} rows...")

    print(f"\nTotal rows scanned: {rows_scanned:,}")

    # -------------------------------------------------
    # 1. Most frequent individual brightness temperatures
    # -------------------------------------------------

    bt11_df = pd.DataFrame(
        bt11_counts.most_common(100),
        columns=["BT_11", "frequency"]
    )

    bt12_df = pd.DataFrame(
        bt12_counts.most_common(100),
        columns=["BT_12", "frequency"]
    )

    bt11_df.to_csv(
        OUTPUT_DIR / "top_100_BT11_values.csv",
        index=False
    )

    bt12_df.to_csv(
        OUTPUT_DIR / "top_100_BT12_values.csv",
        index=False
    )

    # -------------------------------------------------
    # 2. Most frequent brightness-temperature pairs
    # -------------------------------------------------

    pair_df = pd.DataFrame(
        [
            {
                "BT_11": pair[0],
                "BT_12": pair[1],
                "frequency": count,
            }
            for pair, count in pair_counts.most_common(200)
        ]
    )

    pair_df.to_csv(
        OUTPUT_DIR / "top_200_BT_pairs.csv",
        index=False
    )

    # -------------------------------------------------
    # 3. Class distribution for frequent pairs
    # -------------------------------------------------

    class_records = []

    for pair, count in pair_counts.most_common(200):

        classes = pair_class_counts.get(pair, Counter())

        for cls, class_count in classes.items():

            class_records.append({
                "BT_11": pair[0],
                "BT_12": pair[1],
                "pair_total_frequency": count,
                "cloud_type_label": cls,
                "class_frequency": class_count,
                "percentage_of_pair": (
                    100 * class_count / count
                ),
            })

    pd.DataFrame(class_records).to_csv(
        OUTPUT_DIR / "top_pairs_by_cloud_class.csv",
        index=False
    )

    # -------------------------------------------------
    # 4. Known candidate pair occurrences
    # -------------------------------------------------

    if candidate_records:

        candidate_df = pd.concat(
            candidate_records,
            ignore_index=True
        )

        candidate_df.to_csv(
            OUTPUT_DIR / "candidate_486_80_495_75_records.csv",
            index=False
        )

        candidate_summary = (
            candidate_df.groupby("cloud_type_label")
            .agg(
                count=("BTD", "size"),
                timestamp_count=("timestamp", "nunique"),
                latitude_min=("lat", "min"),
                latitude_max=("lat", "max"),
                longitude_min=("lon", "min"),
                longitude_max=("lon", "max"),
            )
            .reset_index()
        )

        candidate_summary.to_csv(
            OUTPUT_DIR / "candidate_pair_class_summary.csv",
            index=False
        )

    else:
        candidate_summary = pd.DataFrame()

    # -------------------------------------------------
    # 5. Plot most frequent BT values
    # -------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(
        bt11_df.head(20)["BT_11"].astype(str),
        bt11_df.head(20)["frequency"]
    )

    axes[0].set_title("Top 20 repeated BT_11 values")
    axes[0].set_xlabel("BT_11 (K)")
    axes[0].set_ylabel("Frequency")
    axes[0].tick_params(axis="x", rotation=75)

    axes[1].bar(
        bt12_df.head(20)["BT_12"].astype(str),
        bt12_df.head(20)["frequency"]
    )

    axes[1].set_title("Top 20 repeated BT_12 values")
    axes[1].set_xlabel("BT_12 (K)")
    axes[1].set_ylabel("Frequency")
    axes[1].tick_params(axis="x", rotation=75)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "repeated_bt_value_frequencies.png",
        dpi=180
    )

    plt.close(fig)

    # -------------------------------------------------
    # 6. Plot most frequent BT pairs
    # -------------------------------------------------

    if not pair_df.empty:

        fig, ax = plt.subplots(figsize=(11, 6))

        top_pairs = pair_df.head(20).copy()

        labels = [
            f"{r.BT_11:.2f}, {r.BT_12:.2f}"
            for r in top_pairs.itertuples()
        ]

        ax.bar(labels, top_pairs["frequency"])

        ax.set_title("Top 20 repeated BT_11 / BT_12 pairs")
        ax.set_xlabel("BT_11, BT_12 (K)")
        ax.set_ylabel("Frequency")
        ax.tick_params(axis="x", rotation=75)

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR / "repeated_bt_pair_frequencies.png",
            dpi=180
        )

        plt.close(fig)

    # -------------------------------------------------
    # 7. Report
    # -------------------------------------------------

    report = [
        "REPEATED BRIGHTNESS TEMPERATURE INVESTIGATION",
        "==============================================",
        f"Rows scanned: {rows_scanned:,}",
        "",
        "TOP 20 BT_11 VALUES",
        "-------------------",
        bt11_df.head(20).to_string(index=False),
        "",
        "TOP 20 BT_12 VALUES",
        "-------------------",
        bt12_df.head(20).to_string(index=False),
        "",
        "TOP 20 BT PAIRS",
        "---------------",
        pair_df.head(20).to_string(index=False),
        "",
        "KNOWN CANDIDATE PAIR: 486.80 / 495.75",
        "-------------------------------------",
    ]

    if not candidate_summary.empty:
        report.append(candidate_summary.to_string(index=False))
        report.append(
            f"Total matching records: {len(candidate_df):,}"
        )
    else:
        report.append("No matching records found.")

    report.extend([
        "",
        "IMPORTANT NOTES",
        "---------------",
        "Repeated values alone do not prove that a value is invalid.",
        "Quantization and rounding can produce repeated measurements.",
        "The candidate pair requires verification against source-product",
        "documentation and quality flags before any exclusion decision.",
        "No original observations were modified or deleted.",
    ])

    (OUTPUT_DIR / "REPEATED_BT_INVESTIGATION_REPORT.txt").write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\nInvestigation complete.")
    print(f"Rows scanned: {rows_scanned:,}")
    print(f"Results saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
