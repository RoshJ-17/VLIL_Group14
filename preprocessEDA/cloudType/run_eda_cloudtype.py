#!/usr/bin/env python3
"""Chunked EDA for cloudtype_final.csv; saves tables and plots without loading all rows."""
from pathlib import Path
import json, random, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = Path("cloudtype_final.csv")
OUT = Path("eda_results/cloudType")
CHUNK = 100_000
SAMPLE_CAP = 200_000
TIME, TARGET = "timestamp", "cloud_type_label"
FEATURES = ["lat", "lon", "BT_11", "BT_12", "BTD"]
CLASSES = ["clear", "liquid_low", "liquid_middle", "liquid_high",
           "ice_low", "ice_middle", "ice_high"]
REQ = [TIME, *FEATURES, TARGET]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Cannot find {INPUT.resolve()}. Put cloudtype_final.csv beside this script.")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    total = 0
    missing = None
    class_counts = {}
    n = {c: 0 for c in FEATURES}
    sums = {c: 0.0 for c in FEATURES}
    sums2 = {c: 0.0 for c in FEATURES}
    mins = {c: float("inf") for c in FEATURES}
    maxs = {c: float("-inf") for c in FEATURES}
    date_class = {}
    sample, seen = [], 0
    ts_min = ts_max = None
    print(f"Input: {INPUT.resolve()}; chunk size: {CHUNK:,}", flush=True)

    for ci, chunk in enumerate(pd.read_csv(INPUT, chunksize=CHUNK, low_memory=False), 1):
        if missing is None:
            absent = [c for c in REQ if c not in chunk.columns]
            if absent:
                raise ValueError(f"Missing required columns {absent}. Found: {list(chunk.columns)}")
            missing = pd.Series(0, index=chunk.columns, dtype="int64")
        total += len(chunk)
        missing += chunk.isna().sum()
        vc = chunk[TARGET].fillna("<MISSING>").astype(str).value_counts()
        for k, v in vc.items(): class_counts[str(k)] = class_counts.get(str(k), 0) + int(v)

        vals = chunk[FEATURES].apply(pd.to_numeric, errors="coerce")
        for col in FEATURES:
            a = vals[col].to_numpy(dtype="float64")
            a = a[np.isfinite(a)]
            if len(a):
                n[col] += len(a); sums[col] += a.sum(); sums2[col] += np.square(a).sum()
                mins[col] = min(mins[col], float(a.min())); maxs[col] = max(maxs[col], float(a.max()))

        ts = chunk[TIME].dropna().astype(str)
        if len(ts):
            lo, hi = ts.min(), ts.max()
            ts_min = lo if ts_min is None else min(ts_min, lo)
            ts_max = hi if ts_max is None else max(ts_max, hi)

        days = chunk[TIME].astype("string").str.slice(0, 7)
        pairs = pd.DataFrame({"day": days, "class": chunk[TARGET].fillna("<MISSING>").astype(str)})
        for (day, label), count in pairs.groupby(["day", "class"], dropna=False).size().items():
            key = (str(day), str(label))
            date_class[key] = date_class.get(key, 0) + int(count)

        for row in chunk[FEATURES + [TARGET]].itertuples(index=False, name=None):
            seen += 1
            if len(sample) < SAMPLE_CAP: sample.append(row)
            else:
                j = rng.randrange(seen)
                if j < SAMPLE_CAP: sample[j] = row
        if ci % 10 == 0: print(f"Scanned {total:,} rows", flush=True)

    if not total: raise ValueError("CSV has no rows.")
    pd.DataFrame({"missing_count": missing, "missing_percent": missing / total * 100}).sort_values(
        "missing_count", ascending=False).to_csv(OUT / "missingness.csv", index_label="column")

    ct = pd.Series(class_counts, name="row_count").sort_values(ascending=False).to_frame()
    ct["percent"] = ct.row_count / total * 100
    ct.to_csv(OUT / "class_distribution.csv", index_label="class")

    summary_rows = []
    for c in FEATURES:
        mean = sums[c] / n[c] if n[c] else np.nan
        sd = math.sqrt(max(sums2[c] / n[c] - mean**2, 0)) if n[c] else np.nan
        summary_rows.append({"feature": c, "finite_count": n[c], "missing_or_nonfinite_count": total-n[c],
                             "mean": mean, "std_population": sd, "min": mins[c], "max": maxs[c]})
    pd.DataFrame(summary_rows).to_csv(OUT / "numeric_summary_all_rows.csv", index=False)

    sdf = pd.DataFrame(sample, columns=FEATURES + [TARGET])
    for c in FEATURES: sdf[c] = pd.to_numeric(sdf[c], errors="coerce")
    sdf.to_csv(OUT / "plot_sample.csv", index=False)
    sdf[FEATURES].quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).T.to_csv(OUT / "sample_quantiles.csv", index_label="feature")
    sdf[FEATURES].corr().to_csv(OUT / "sample_correlations.csv")

    # Exact class counts and percentages
    plot_counts = pd.Series(class_counts).reindex(CLASSES, fill_value=0)
    for mode, vals in [("counts", plot_counts), ("percent", plot_counts / total * 100)]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(vals.index, vals.values)
        ax.set_title(f"Cloud-class distribution ({mode})")
        ax.set_ylabel("Rows" if mode == "counts" else "Percent of rows")
        ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(OUT / f"class_distribution_{mode}.png", dpi=150); plt.close(fig)

    for c in FEATURES:
        v = sdf[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(v):
            fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(v, bins=60)
            ax.set_title(f"{c} distribution (sample)"); ax.set_xlabel(c); ax.set_ylabel("Sample frequency")
            fig.tight_layout(); fig.savefig(OUT / f"hist_{c}.png", dpi=150); plt.close(fig)
        tmp = sdf[[TARGET, c]].replace([np.inf, -np.inf], np.nan).dropna()
        labs = [x for x in CLASSES if x in set(tmp[TARGET].astype(str))]
        arr = [tmp.loc[tmp[TARGET].astype(str) == lab, c].to_numpy() for lab in labs]
        if arr:
            fig, ax = plt.subplots(figsize=(11, 5)); ax.boxplot(arr, labels=labs, showfliers=False)
            ax.set_title(f"{c} by class (sample; outliers hidden)"); ax.tick_params(axis="x", rotation=35)
            fig.tight_layout(); fig.savefig(OUT / f"boxplot_{c}_by_class.png", dpi=150); plt.close(fig)

    dc = pd.DataFrame([(d, lab, v) for (d, lab), v in date_class.items()],
                      columns=["date_key", "class", "row_count"]).sort_values(["date_key", "class"])
    dc.to_csv(OUT / "class_counts_by_date.csv", index=False)
    daily = dc.groupby("date_key", as_index=False).row_count.sum()
    daily.to_csv(OUT / "rows_by_date.csv", index=False)
    if len(daily):
        fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(daily.date_key, daily.row_count)
        ticks = np.linspace(0, len(daily)-1, min(12, len(daily)), dtype=int)
        ax.set_xticks(ticks); ax.set_xticklabels(daily.date_key.iloc[ticks], rotation=45, ha="right")
        ax.set_title("Rows by timestamp date key"); ax.set_ylabel("Rows"); fig.tight_layout()
        fig.savefig(OUT / "rows_by_date.png", dpi=150); plt.close(fig)

    result = {"input": str(INPUT.resolve()), "total_rows": total, "columns": list(missing.index),
              "timestamp_lexical_min": ts_min, "timestamp_lexical_max": ts_max,
              "class_counts": class_counts, "plot_sample_rows": len(sdf),
              "note": "Quantiles, correlations and plots use a reservoir sample; aggregate counts and numeric min/max/means use all rows."}
    (OUT / "eda_summary.json").write_text(json.dumps(result, indent=2))
    with (OUT / "EDA_REPORT.txt").open("w") as f:
        f.write("CLOUD TYPE DATASET — EDA SUMMARY\n" + "="*42 + "\n")
        f.write(f"Rows scanned: {total:,}\nColumns: {len(missing)}\nTimestamp lexical range: {ts_min} to {ts_max}\n\n")
        f.write("CLASS DISTRIBUTION\n" + ct.to_string() + "\n\n")
        f.write("MISSINGNESS (top 20)\n")
        f.write(pd.DataFrame({"count": missing, "percent": missing/total*100}).sort_values("count", ascending=False).head(20).to_string())
        f.write("\n\nALL-ROW NUMERIC SUMMARY\n" + pd.DataFrame(summary_rows).to_string(index=False))
        f.write("\n\nSAMPLE CORRELATIONS\n" + sdf[FEATURES].corr().to_string())
        f.write("\n\nQuantiles/correlations/plots are based on the saved bounded sample.\n")
    print(f"EDA complete: scanned {total:,} rows. Results: {OUT.resolve()}", flush=True)

if __name__ == "__main__":
    main()
