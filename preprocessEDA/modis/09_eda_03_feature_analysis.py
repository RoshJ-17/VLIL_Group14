
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path("modisPixel_with_GPM.csv")
OUT = Path("eda_results")
OUT.mkdir(exist_ok=True)

CHUNK_SIZE = 500_000
SAMPLE_PER_CHUNK = 2_000
SEED = 42

features = [
    "BT_11", "BT_12", "BTD",
    "Cloud_Mask_Binary", "Cloud_Mask_Raw",
    "CTT", "CTH", "Cloud_Phase",
    "Cloud_Optical_Thickness",
    "GPM_precipitation_rate"
]

samples = []
rng = np.random.default_rng(SEED)
rows_scanned = 0

print("Sampling records for feature analysis...")

for chunk in pd.read_csv(
    CSV,
    chunksize=CHUNK_SIZE,
    usecols=features,
    low_memory=False
):
    rows_scanned += len(chunk)
    sample_n = min(SAMPLE_PER_CHUNK, len(chunk))
    samples.append(
        chunk.sample(n=sample_n, random_state=SEED)
    )

    if rows_scanned % 5_000_000 < CHUNK_SIZE:
        print(f"Rows scanned: {rows_scanned:,}")

df = pd.concat(samples, ignore_index=True)
print(f"\nSample size: {len(df):,}")

# Convert columns safely to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Summary statistics
summary = df[features].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

summary["missing_percent"] = (
    df[features].isna().mean() * 100
)

summary.to_csv(OUT / "feature_summary.csv")

# Histograms
for col in features:
    values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        continue

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=60)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Sample count")
    plt.tight_layout()
    plt.savefig(OUT / f"hist_{col}.png", dpi=140)
    plt.close()

# Correlation matrix, sampled rows only
corr = df[features].corr(method="spearman")
corr.to_csv(OUT / "feature_spearman_correlation.csv")

plt.figure(figsize=(11, 9))
plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Spearman correlation")
plt.xticks(range(len(features)), features, rotation=70, ha="right")
plt.yticks(range(len(features)), features)
plt.title("Feature and GPM Spearman Correlations")
plt.tight_layout()
plt.savefig(OUT / "feature_correlation.png", dpi=160)
plt.close()

# Compare feature distributions between rain and no-rain samples
df["rain_label"] = (
    df["GPM_precipitation_rate"] >= 0.1
).astype("int8")

comparison = df.groupby("rain_label")[features].median().T
comparison.columns = ["no_rain_median", "rain_median"]
comparison.to_csv(OUT / "feature_medians_by_rain_class.csv")

print("\nSaved outputs:")
for name in [
    "feature_summary.csv",
    "feature_spearman_correlation.csv",
    "feature_correlation.png",
    "feature_medians_by_rain_class.csv",
]:
    print(" ", OUT / name)

print(f"\nHistograms saved as {OUT}/hist_*.png")
print("Feature analysis completed.")

