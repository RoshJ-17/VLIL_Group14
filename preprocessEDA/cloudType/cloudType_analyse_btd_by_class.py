#!/usr/bin/env python3
"""
BTD-by-cloud-class investigation for cloudtype_final.csv.

All generated results are written beneath:
    eda_results/cloudType/btd_class_analysis/

Run from the project directory:
    python cloudType_analyse_btd_by_class.py

Optional:
    python cloudType_analyse_btd_by_class.py --input path/to/cloudtype_final.csv
    python cloudType_analyse_btd_by_class.py --chunksize 100000
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = [
    "clear", "liquid_low", "liquid_middle", "liquid_high",
    "ice_low", "ice_middle", "ice_high",
]
REQUIRED = {"BT_11", "BT_12", "BTD", "cloud_type_label"}
DEFAULT_INPUT = Path("cloudtype_final.csv")
OUTPUT_DIR = Path("eda_results") / "cloudType" / "btd_class_analysis"
SAMPLE_PER_CLASS = 20000
TOP_N = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Cannot find {args.input}. Run this script from the project directory "
            "or pass --input with the CSV path."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    totals = {c: 0 for c in CLASSES}
    over5 = {c: 0 for c in CLASSES}
    over10 = {c: 0 for c in CLASSES}
    sums = {c: 0.0 for c in CLASSES}
    sumsq = {c: 0.0 for c in CLASSES}
    mins = {c: np.inf for c in CLASSES}
    maxs = {c: -np.inf for c in CLASSES}
    samples = {c: [] for c in CLASSES}
    top_pos = pd.DataFrame()
    top_neg = pd.DataFrame()
    rows_scanned = 0
    required_cols = ["timestamp", "lat", "lon", "BT_11", "BT_12", "BTD", "cloud_type_label"]

    reader = pd.read_csv(args.input, usecols=lambda x: x in required_cols,
                         chunksize=args.chunksize)
    for chunk in reader:
        missing = REQUIRED - set(chunk.columns)
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
        rows_scanned += len(chunk)
        for col in ["BT_11", "BT_12", "BTD"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk["abs_BTD"] = chunk["BTD"].abs()

        for cls, group in chunk.groupby("cloud_type_label", sort=False):
            if cls not in totals:
                continue
            vals = group["BTD"].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            totals[cls] += int(vals.size)
            over5[cls] += int((np.abs(vals) > 5).sum())
            over10[cls] += int((np.abs(vals) > 10).sum())
            sums[cls] += float(vals.sum())
            sumsq[cls] += float(np.square(vals).sum())
            mins[cls] = min(mins[cls], float(vals.min()))
            maxs[cls] = max(maxs[cls], float(vals.max()))

            # Deterministic bounded sample for quantiles and plots.
            remaining = SAMPLE_PER_CLASS - sum(len(x) for x in samples[cls])
            if remaining > 0:
                take = min(remaining, len(vals))
                # Evenly spaced positions avoid keeping only the beginning of the file.
                idx = np.linspace(0, len(vals) - 1, num=take, dtype=int)
                samples[cls].append(vals[idx])

        keep_cols = [c for c in required_cols if c in chunk.columns]
        candidates = chunk.loc[chunk["BTD"].notna(), keep_cols + ["abs_BTD"]]
        if not candidates.empty:
            pos = candidates.nlargest(TOP_N, "BTD")
            neg = candidates.nsmallest(TOP_N, "BTD")
            top_pos = pd.concat([top_pos, pos], ignore_index=True).nlargest(TOP_N, "BTD")
            top_neg = pd.concat([top_neg, neg], ignore_index=True).nsmallest(TOP_N, "BTD")

    records = []
    quantile_rows = []
    for cls in CLASSES:
        n = totals[cls]
        if n == 0:
            continue
        mean = sums[cls] / n
        variance = max(0.0, sumsq[cls] / n - mean * mean)
        vals = np.concatenate(samples[cls]) if samples[cls] else np.array([])
        q = np.quantile(vals, [0, .01, .05, .25, .5, .75, .95, .99, 1]) if vals.size else [np.nan]*9
        records.append({
            "cloud_type_label": cls, "count": n,
            "abs_BTD_gt_5_count": over5[cls],
            "abs_BTD_gt_5_percent_of_class": 100 * over5[cls] / n,
            "abs_BTD_gt_10_count": over10[cls],
            "abs_BTD_gt_10_percent_of_class": 100 * over10[cls] / n,
            "BTD_mean": mean, "BTD_std_population": np.sqrt(variance),
            "BTD_min": mins[cls], "BTD_max": maxs[cls],
        })
        quantile_rows.append({
            "cloud_type_label": cls, "sample_n": vals.size,
            **dict(zip(["q0", "q01", "q05", "q25", "median", "q75", "q95", "q99", "q100"], q))
        })

    summary = pd.DataFrame(records)
    quantiles = pd.DataFrame(quantile_rows)
    summary.to_csv(OUTPUT_DIR / "btd_summary_by_class.csv", index=False)
    quantiles.to_csv(OUTPUT_DIR / "btd_sample_quantiles_by_class.csv", index=False)
    top_pos.to_csv(OUTPUT_DIR / "top_100_positive_btd.csv", index=False)
    top_neg.to_csv(OUTPUT_DIR / "top_100_negative_btd.csv", index=False)

    # Sample-based boxplot, with outliers hidden so the central distributions remain legible.
    plot_data = [np.concatenate(samples[c]) for c in CLASSES if samples[c]]
    plot_labels = [c for c in CLASSES if samples[c]]
    fig, ax = plt.subplots(figsize=(12, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        ax.boxplot(plot_data, tick_labels=plot_labels, showfliers=False)
    ax.set_title("BTD distribution by cloud class (bounded sample; fliers hidden)")
    ax.set_ylabel("BTD (K)")
    ax.set_xlabel("Cloud class")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "btd_boxplot_by_class.png", dpi=180)
    plt.close(fig)

    # Sample-based overlaid histograms.
    fig, ax = plt.subplots(figsize=(12, 6))
    for cls in CLASSES:
        if samples[cls]:
            ax.hist(np.concatenate(samples[cls]), bins=100, density=True,
                    histtype="step", linewidth=1.4, label=cls)
    ax.set_title("Sampled BTD distributions by cloud class")
    ax.set_xlabel("BTD (K)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "btd_histograms_by_class.png", dpi=180)
    plt.close(fig)

    report = [
        "BTD BY CLOUD CLASS ANALYSIS",
        "===========================",
        f"Rows scanned: {rows_scanned:,}",
        f"Output directory: {OUTPUT_DIR}",
        "",
        "Exact full-file counts and threshold percentages; quantiles/plots use bounded per-class samples.",
        "",
        summary.to_string(index=False) if not summary.empty else "No recognized classes found.",
        "",
        "Sample quantiles:",
        quantiles.to_string(index=False) if not quantiles.empty else "No quantiles available.",
        "",
        "Interpretation note: |BTD| thresholds are inspection flags, not automatic invalidity criteria.",
        "No observations were removed or modified.",
    ]
    (OUTPUT_DIR / "BTD_CLASS_ANALYSIS_REPORT.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"Analysis complete: scanned {rows_scanned:,} rows.")
    print(f"Results saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
