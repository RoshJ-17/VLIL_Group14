from pathlib import Path
import json
import duckdb

PROJECT_DIR = Path(
    "/home/m_prathilothamai/aviation_weather_project"
)

INPUT_FILE = (
    PROJECT_DIR
    / "eda_results"
    / "cloudType"
    / "cleaning"
    / "cloudtype_final_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "eda_results"
    / "cloudType"
    / "preprocessing"
)

TEMP_DIR = OUTPUT_DIR / "temporary"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "final_duplicate_verification.json"

print("=" * 75)
print("FINAL DUPLICATE VERIFICATION")
print("=" * 75)

con = duckdb.connect(database=":memory:")

con.execute(
    f"SET temp_directory = '{TEMP_DIR.as_posix()}'"
)

con.execute("SET preserve_insertion_order = false")

csv_path = INPUT_FILE.as_posix().replace("'", "''")

# Read all columns as strings first, then explicitly convert
# numerical fields to DOUBLE for consistent comparison.
con.execute(f"""
    CREATE VIEW cloud_data AS
    SELECT
        CAST(timestamp AS VARCHAR) AS timestamp,
        TRY_CAST(lat AS DOUBLE) AS lat,
        TRY_CAST(lon AS DOUBLE) AS lon,
        TRY_CAST(BT_11 AS DOUBLE) AS BT_11,
        TRY_CAST(BT_12 AS DOUBLE) AS BT_12,
        TRY_CAST(BTD AS DOUBLE) AS BTD,
        CAST(cloud_type_label AS VARCHAR) AS cloud_type_label
    FROM read_csv(
        '{csv_path}',
        header = true,
        all_varchar = true,
        sample_size = -1
    )
""")

# Verify row count.
total_rows = con.execute("""
    SELECT COUNT(*) FROM cloud_data
""").fetchone()[0]

# Exact duplicate check across the seven explicit columns.
con.execute("""
    CREATE TEMP TABLE duplicate_groups AS
    SELECT
        timestamp,
        lat,
        lon,
        BT_11,
        BT_12,
        BTD,
        cloud_type_label,
        COUNT(*) AS occurrences
    FROM cloud_data
    GROUP BY
        timestamp,
        lat,
        lon,
        BT_11,
        BT_12,
        BTD,
        cloud_type_label
    HAVING COUNT(*) > 1
""")

duplicate_groups, excess_rows, rows_in_groups, largest_group = (
    con.execute("""
        SELECT
            COUNT(*),
            COALESCE(SUM(occurrences - 1), 0),
            COALESCE(SUM(occurrences), 0),
            COALESCE(MAX(occurrences), 0)
        FROM duplicate_groups
    """).fetchone()
)

# Check repeated pixel keys separately.
pixel_groups, pixel_excess = con.execute("""
    SELECT
        COUNT(*),
        COALESCE(SUM(occurrences - 1), 0)
    FROM (
        SELECT COUNT(*) AS occurrences
        FROM cloud_data
        GROUP BY timestamp, lat, lon
        HAVING COUNT(*) > 1
    )
""").fetchone()

summary = {
    "total_rows": int(total_rows),
    "exact_duplicate_groups": int(duplicate_groups),
    "exact_duplicate_excess_rows": int(excess_rows),
    "rows_in_exact_duplicate_groups": int(rows_in_groups),
    "largest_exact_duplicate_group": int(largest_group),
    "duplicate_pixel_key_groups": int(pixel_groups),
    "duplicate_pixel_key_excess_rows": int(pixel_excess),
    "duplicate_check_columns": [
        "timestamp",
        "lat",
        "lon",
        "BT_11",
        "BT_12",
        "BTD",
        "cloud_type_label",
    ],
    "duplicate_metrics_consistent": (
        (
            duplicate_groups == 0 and excess_rows == 0
        )
        or (
            duplicate_groups > 0 and excess_rows > 0
        )
    ),
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("\nResults:")
print(json.dumps(summary, indent=4))

print(f"\nSaved to: {OUTPUT_FILE}")

con.close()

print("\nVerification completed.")
