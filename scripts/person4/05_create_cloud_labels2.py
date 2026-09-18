import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT_FILE = Path("modisPixel.csv")
OUTPUT_FILE = Path("modis_cloudtype_labeled_full.csv")

CHUNK_SIZE = 250_000

# Pilot run first. Set to None only after checking
# the pilot output and deciding to process the full dataset.
MAX_ROWS = None

LOW_MAX_M = 2000
MIDDLE_MAX_M = 6000

PHASE_MAP = {
    0: "clear",
    1: "liquid",
    2: "ice",
    6: "undetermined",
}

VALID_CLOUD_PHASES = ["liquid", "ice", "undetermined"]


def create_labels(df):
    """Create cloud phase, height band, and combined target labels."""
    df = df.copy()

    phase = pd.to_numeric(df["Cloud_Phase"], errors="coerce")
    cth = pd.to_numeric(df["CTH"], errors="coerce")

    # NASA MODIS infrared cloud phase categories
    df["cloud_phase_label"] = phase.map(PHASE_MAP).fillna("unknown")

    # Height bands apply only to cloudy phases with positive CTH
    df["cloud_height_label"] = "unknown"

    is_clear = df["cloud_phase_label"].eq("clear")
    is_cloud = df["cloud_phase_label"].isin(VALID_CLOUD_PHASES)
    valid_height = cth.notna() & cth.gt(0)

    df.loc[is_clear, "cloud_height_label"] = "not_applicable"

    valid = is_cloud & valid_height

    df.loc[valid & (cth < LOW_MAX_M), "cloud_height_label"] = "low"

    df.loc[
        valid & (cth >= LOW_MAX_M) & (cth < MIDDLE_MAX_M),
        "cloud_height_label"
    ] = "middle"

    df.loc[
        valid & (cth >= MIDDLE_MAX_M),
        "cloud_height_label"
    ] = "high"

    # Combined target: 10 known classes plus "unknown"
    df["cloud_type_label"] = "unknown"
    df.loc[is_clear, "cloud_type_label"] = "clear"

    for phase_name in VALID_CLOUD_PHASES:
        for height_name in ["low", "middle", "high"]:
            mask = (
                df["cloud_phase_label"].eq(phase_name)
                & df["cloud_height_label"].eq(height_name)
            )
            df.loc[mask, "cloud_type_label"] = (
                f"{phase_name}_{height_name}"
            )

    return df


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE.resolve()}"
        )

    # Reset pilot output so previous runs cannot be appended accidentally.
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total_rows = 0
    class_counts = {}
    phase_counts = {}
    height_counts = {}

    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
        if MAX_ROWS is not None:
            remaining = MAX_ROWS - total_rows
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()

        if chunk.empty:
            break

        labeled = create_labels(chunk)

        # Accumulate counts across chunks
        for column, counter in [
            ("cloud_type_label", class_counts),
            ("cloud_phase_label", phase_counts),
            ("cloud_height_label", height_counts),
        ]:
            counts = labeled[column].value_counts(dropna=False)
            for label, count in counts.items():
                counter[label] = counter.get(label, 0) + int(count)

        # First chunk creates the output; later chunks append
        labeled.to_csv(
            OUTPUT_FILE,
            index=False,
            mode="a" if total_rows > 0 else "w",
            header=(total_rows == 0),
        )

        total_rows += len(labeled)
        print(f"Processed {total_rows:,} rows")

    print("\n=== Labeling complete ===")
    print(f"Rows processed: {total_rows:,}")
    print(f"Output: {OUTPUT_FILE.resolve()}")

    print("\n=== Combined cloud-type counts ===")
    print(pd.Series(class_counts).sort_index().to_string())

    print("\n=== Phase-label counts ===")
    print(pd.Series(phase_counts).sort_index().to_string())

    print("\n=== Height-label counts ===")
    print(pd.Series(height_counts).sort_index().to_string())


if __name__ == "__main__":
    main()

