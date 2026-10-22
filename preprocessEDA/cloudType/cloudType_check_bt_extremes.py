#!/usr/bin/env python3
"""Read-only, chunked diagnostic for extreme BT_11, BT_12 and BTD values."""
from pathlib import Path
import json, random
import pandas as pd
import numpy as np

INPUT = Path("cloudtype_final.csv")
OUT = Path("eda_results/cloudType/bt_quality_check")
CHUNK = 100_000
TOP_N = 100
COLS = ["timestamp", "lat", "lon", "BT_11", "BT_12", "BTD", "cloud_type_label"]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Cannot find {INPUT.resolve()}")
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {k: 0 for k in ["rows", "BT_11_gt_350", "BT_11_gt_400", "BT_12_gt_350",
             "BT_12_gt_400", "BT_11_lt_180", "BT_12_lt_180", "abs_BTD_gt_5", "abs_BTD_gt_10"]}
    extrema = {}
    tops = {k: pd.DataFrame() for k in ["BT_11_high", "BT_11_low", "BT_12_high", "BT_12_low", "BTD_abs"]}
    class_counts = {}
    print(f"Scanning {INPUT.resolve()} in chunks of {CHUNK:,}", flush=True)
    for i, df in enumerate(pd.read_csv(INPUT, usecols=COLS, chunksize=CHUNK, low_memory=False), 1):
        counts["rows"] += len(df)
        for c in ["BT_11", "BT_12", "BTD", "lat", "lon"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        tests = {
            "BT_11_gt_350": df.BT_11 > 350, "BT_11_gt_400": df.BT_11 > 400,
            "BT_12_gt_350": df.BT_12 > 350, "BT_12_gt_400": df.BT_12 > 400,
            "BT_11_lt_180": df.BT_11 < 180, "BT_12_lt_180": df.BT_12 < 180,
            "abs_BTD_gt_5": df.BTD.abs() > 5, "abs_BTD_gt_10": df.BTD.abs() > 10,
        }
        for k, m in tests.items(): counts[k] += int(m.sum())
        df["BTD_recomputed"] = df.BT_11 - df.BT_12
        df["BTD_difference"] = df.BTD - df.BTD_recomputed
        for c in ["BT_11", "BT_12", "BTD", "BTD_difference"]:
            s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s):
                extrema.setdefault(c, {"min": float("inf"), "max": float("-inf")})
                extrema[c]["min"] = min(extrema[c]["min"], float(s.min()))
                extrema[c]["max"] = max(extrema[c]["max"], float(s.max()))
        candidates = {
            "BT_11_high": df.nlargest(TOP_N, "BT_11"),
            "BT_11_low": df.nsmallest(TOP_N, "BT_11"),
            "BT_12_high": df.nlargest(TOP_N, "BT_12"),
            "BT_12_low": df.nsmallest(TOP_N, "BT_12"),
            "BTD_abs": df.assign(_abs_BTD=df.BTD.abs()).nlargest(TOP_N, "_abs_BTD"),
        }
        for key, cand in candidates.items():
            merged = pd.concat([tops[key], cand], ignore_index=True)
            if key == "BT_11_high": merged = merged.nlargest(TOP_N, "BT_11")
            elif key == "BT_11_low": merged = merged.nsmallest(TOP_N, "BT_11")
            elif key == "BT_12_high": merged = merged.nlargest(TOP_N, "BT_12")
            elif key == "BT_12_low": merged = merged.nsmallest(TOP_N, "BT_12")
            else: merged = merged.assign(_abs_BTD=merged.BTD.abs()).nlargest(TOP_N, "_abs_BTD")
            tops[key] = merged.drop_duplicates(subset=["timestamp","lat","lon","BT_11","BT_12","BTD"]).head(TOP_N)
        extreme_mask = (df.BT_11 > 350) | (df.BT_12 > 350) | (df.BT_11 < 180) | (df.BT_12 < 180) | (df.BTD.abs() > 5)
        vc = df.loc[extreme_mask, "cloud_type_label"].fillna("<MISSING>").astype(str).value_counts()
        for label, n in vc.items(): class_counts[label] = class_counts.get(label, 0) + int(n)
        if i % 10 == 0: print(f"Scanned {counts['rows']:,} rows", flush=True)

    for key, df in tops.items():
        df.drop(columns=["_abs_BTD"], errors="ignore").to_csv(OUT / f"top_{key.lower()}.csv", index=False)
    pd.DataFrame([{"check": k, "count": v} for k,v in counts.items()]).to_csv(OUT/"threshold_counts.csv", index=False)
    pd.DataFrame([{"feature": k, **v} for k,v in extrema.items()]).to_csv(OUT/"observed_extrema.csv", index=False)
    pd.DataFrame([{"class": k, "extreme_rows": v} for k,v in sorted(class_counts.items(), key=lambda x:-x[1])]).to_csv(OUT/"extreme_counts_by_class.csv", index=False)

    # Second pass: verify whether BTD approximately equals BT_11 - BT_12.
    n = mismatch = 0
    abs_sum = abs_max = 0.0
    for df in pd.read_csv(INPUT, usecols=["BT_11","BT_12","BTD"], chunksize=CHUNK, low_memory=False):
        for c in ["BT_11","BT_12","BTD"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        d = (df.BTD - (df.BT_11 - df.BT_12)).replace([np.inf,-np.inf], np.nan).dropna().abs()
        if len(d):
            n += len(d); abs_sum += float(d.sum()); abs_max = max(abs_max, float(d.max()))
            mismatch += int((d > .05).sum())
    consistency = {"rows_compared": n, "mean_absolute_difference": abs_sum/n if n else None,
                   "max_absolute_difference": abs_max, "rows_difference_gt_0_05": mismatch}
    report = {"input": str(INPUT.resolve()), "counts": counts, "extrema": extrema,
              "BTD_vs_BT11_minus_BT12": consistency, "extreme_counts_by_class": class_counts,
              "interpretation": "Thresholds are diagnostic flags, not proof of invalidity. No data were modified."}
    (OUT/"quality_check_summary.json").write_text(json.dumps(report, indent=2))
    with (OUT/"QUALITY_CHECK_REPORT.txt").open("w") as f:
        f.write("BRIGHTNESS TEMPERATURE QUALITY CHECK\n====================================\n")
        f.write(f"Rows scanned: {counts['rows']:,}\n\nThreshold counts:\n")
        for k,v in counts.items():
            if k != "rows": f.write(f"{k}: {v:,}\n")
        f.write("\nObserved extrema:\n")
        for k,v in extrema.items(): f.write(f"{k}: min={v['min']}, max={v['max']}\n")
        f.write("\nBTD consistency:\n"+json.dumps(consistency, indent=2)+"\n")
        f.write("\nThresholds are inspection flags only; no rows were removed.\n")
    print(f"Done. Results saved to {OUT.resolve()}", flush=True)

if __name__ == "__main__": main()
