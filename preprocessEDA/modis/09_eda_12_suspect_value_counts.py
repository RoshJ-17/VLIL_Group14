
import pandas as pd
import numpy as np
import os

INPUT = "modisPixel_with_GPM.csv"
OUTDIR = "eda_results"
CHUNK_SIZE = 1_000_000

os.makedirs(OUTDIR, exist_ok=True)

cols = [
    "BT_11", "BT_12", "BTD", "CTT", "CTH",
    "GPM_precipitation_rate"
]

counts = {
    "rows": 0,
    "bt11_at_max": 0,
    "bt12_at_max": 0,
    "both_bt_at_max": 0,
    "bt_pair_and_btd_match": 0,
    "cth_zero": 0,
    "cth_zero_ctt_present": 0,
    "cth_zero_ctt_ge_310": 0,
    "cth_zero_rainy": 0,
    "cth_zero_gpm_sum": 0.0,
    "cth_zero_gpm_n": 0,
    "bt_pair_gpm_zero": 0,
    "bt_pair_gpm_positive": 0,
    "bt_pair_gpm_missing": 0,
}

for chunk in pd.read_csv(INPUT, usecols=cols, chunksize=CHUNK_SIZE):
    counts["rows"] += len(chunk)

    for col in cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    b11 = chunk["BT_11"]
    b12 = chunk["BT_12"]
    btd = chunk["BTD"]
    ctt = chunk["CTT"]
    cth = chunk["CTH"]
    gpm = chunk["GPM_precipitation_rate"]

    m11 = b11.eq(486.8)
    m12 = b12.eq(495.75)
    pair = m11 & m12
    cth0 = cth.eq(0)

    counts["bt11_at_max"] += int(m11.sum())
    counts["bt12_at_max"] += int(m12.sum())
    counts["both_bt_at_max"] += int(pair.sum())
    counts["bt_pair_and_btd_match"] += int(
        (pair & btd.eq(-8.95)).sum()
    )

    counts["cth_zero"] += int(cth0.sum())
    counts["cth_zero_ctt_present"] += int((cth0 & ctt.notna()).sum())
    counts["cth_zero_ctt_ge_310"] += int(
        (cth0 & ctt.ge(310)).sum()
    )

    valid_gpm_zero_cth = cth0 & gpm.notna() & np.isfinite(gpm) & (gpm >= 0)
    counts["cth_zero_rainy"] += int(
        (valid_gpm_zero_cth & (gpm >= 0.1)).sum()
    )
    counts["cth_zero_gpm_sum"] += float(
        gpm[valid_gpm_zero_cth].sum()
    )
    counts["cth_zero_gpm_n"] += int(valid_gpm_zero_cth.sum())

    counts["bt_pair_gpm_zero"] += int((pair & gpm.eq(0)).sum())
    counts["bt_pair_gpm_positive"] += int(
        (pair & gpm.gt(0)).sum()
    )
    counts["bt_pair_gpm_missing"] += int(
        (pair & gpm.isna()).sum()
    )

    if counts["rows"] % 5_000_000 < CHUNK_SIZE:
        print(f"Rows scanned: {counts['rows']:,}", flush=True)

print("\n===== SUSPECT VALUE COUNTS =====")
for k, v in counts.items():
    print(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}")

n = counts["cth_zero_gpm_n"]
print(
    "CTH-zero mean GPM:",
    counts["cth_zero_gpm_sum"] / n if n else np.nan
)

pd.Series(counts).to_json(
    os.path.join(OUTDIR, "suspect_value_counts.json"),
    indent=2
)
print(f"\nSaved: {OUTDIR}/suspect_value_counts.json")
