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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = OUTPUT_DIR / "temporary"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CORRECTED EXACT DUPLICATE CHECK")
print("=" * 70)

con = duckdb.connect(database=":memory:")

con.execute(
    f"SET temp_directory = '{TEMP_DIR.as_posix()}'"
)

csv_path = INPUT_FILE.as_posix().replace("'", "''")

con.execute(f"""
    CREATE VIEW cloud_data AS
    SELECT *
    FROM read_csv_auto(
        '{csv_path}',
        header = true,
        sample_size = 100000
    )
""")

# Group identical complete records and count their occurrences.
con.execute("""
    CREATE TEMP TABLE duplicate_groups AS
    SELECT COUNT(*) AS record_count
    FROM cloud_data
    GROUP BY ALL
    HAVING COUNT(*) > 1
""")

# Calculate both metrics from exactly the same grouped table.
result = con.execute("""
    SELECT
        COUNT(*) AS duplicate_groups,
        COALESCE(SUM(record_count - 1), 0)
            AS excess_duplicate_rows,
        COALESCE(SUM(record_count), 0)
            AS rows_in_duplicate_groups,
        COALESCE(MAX(record_count), 0)
            AS largest_duplicate_group
    FROM duplicate_groups
""").fetchone()

duplicate_groups = int(result[0])
excess_rows = int(result[1])
rows_in_groups = int(result[2])
largest_group = int(result[3])

summary = {
    "duplicate_groups": duplicate_groups,
    "excess_duplicate_rows": excess_rows,
    "rows_in_duplicate_groups": rows_in_groups,
    "largest_duplicate_group": largest_group,
    "interpretation": (
        "No exact duplicate records found."
        if duplicate_groups == 0
        else "Exact duplicate records were found."
    ),
}

report_path = OUTPUT_DIR / "corrected_duplicate_summary.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("\nCorrected duplicate results:")
print(json.dumps(summary, indent=4))

if duplicate_groups == 0 and excess_rows != 0:
    raise RuntimeError("Duplicate metrics are inconsistent.")

if duplicate_groups > 0 and excess_rows == 0:
    raise RuntimeError("Duplicate metrics are inconsistent.")

print(f"\nSaved report: {report_path}")

con.close()

print("\nDuplicate check completed.")
