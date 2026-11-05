
import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/full_year_2022_2023/analysis/"
    "decision_geographical_summary.csv"
)

OUTPUT_DIR = (
    "data/processed/full_year_2022_2023/analysis/"
    "geographical_heatmaps"
)

MIN_OBSERVATIONS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading geographical summary...")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "lat",
    "lon",
    "assessable_observations",
    "assessable_pixels",
    "high_pct",
    "high_severity_observation_pct",
    "rain_detected_pixels",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# --------------------------------------------------
# FILTER LOCATIONS
# --------------------------------------------------

original_count = len(df)

df = df[
    (df["assessable_observations"] >= MIN_OBSERVATIONS)
    & (df["assessable_pixels"] > 0)
].copy()

print(f"Original locations: {original_count}")
print(f"Locations retained: {len(df)}")
print(f"Locations excluded: {original_count - len(df)}")

# --------------------------------------------------
# CALCULATE RAIN DETECTION PERCENTAGE
# --------------------------------------------------

df["rain_detection_pct"] = (
    df["rain_detected_pixels"]
    / df["assessable_pixels"]
) * 100

# --------------------------------------------------
# HEATMAP FUNCTION
# --------------------------------------------------

def generate_heatmap(data, column, title, colorbar_label, filename):

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        data["lon"],
        data["lat"],
        c=data[column],
        cmap="YlOrRd",
        marker="s",
        s=13,
        linewidths=0,
    )

    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label(colorbar_label, fontsize=11)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_xlim(data["lon"].min() - 0.25, data["lon"].max() + 0.25)
    ax.set_ylim(data["lat"].min() - 0.25, data["lat"].max() + 0.25)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# --------------------------------------------------
# 1. HIGH ADVISORY PIXEL PERCENTAGE
# --------------------------------------------------

generate_heatmap(
    df,
    "high_pct",
    "Geographical Distribution of High Advisory Pixels",
    "High advisory pixels (%)",
    "decision_high_advisory_heatmap.png",
)

# --------------------------------------------------
# 2. HIGH-SEVERITY OBSERVATION PERCENTAGE
# --------------------------------------------------

generate_heatmap(
    df,
    "high_severity_observation_pct",
    "Frequency of High-Severity Observations",
    "High-severity observations (%)",
    "decision_high_severity_frequency_heatmap.png",
)

# --------------------------------------------------
# 3. RAIN DETECTION PERCENTAGE
# --------------------------------------------------

generate_heatmap(
    df,
    "rain_detection_pct",
    "Geographical Distribution of Rain Detection",
    "Rain-detected pixels (%)",
    "decision_rain_detection_heatmap.png",
)

print("\nGEOGRAPHICAL HEATMAP GENERATION COMPLETE")
print(f"Output directory: {OUTPUT_DIR}")

