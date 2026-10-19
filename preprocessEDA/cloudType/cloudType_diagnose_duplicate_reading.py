from pathlib import Path
import pandas as pd
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

print("=" * 75)
print("CSV DUPLICATE-READING DIAGNOSTICS")
print("=" * 75)

# ------------------------------------------------------------
# 1. Inspect raw CSV lines
# ------------------------------------------------------------

print("\n1. RAW CSV CONTENT")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i in range(6):
        print(f"Line {i + 1}: {f.readline().rstrip()}")

# ------------------------------------------------------------
# 2. Read using pandas
# ------------------------------------------------------------

print("\n2. PANDAS INTERPRETATION")

pandas_sample = pd.read_csv(
    INPUT_FILE,
    nrows=10,
)

print("\nPandas column names:")
print(pandas_sample.columns.tolist())

print("\nPandas data types:")
print(pandas_sample.dtypes)

print("\nPandas first 10 rows:")
print(pandas_sample.to_string(index=False))

print("\nPandas distinct values in sample:")
print(pandas_sample.nunique(dropna=False))

# ------------------------------------------------------------
# 3. Read using DuckDB
# ------------------------------------------------------------

print("\n3. DUCKDB INTERPRETATION")

con = duckdb.connect(database=":memory:")

csv_path = INPUT_FILE.as_posix().replace("'", "''")

# Explicitly disable automatic type inference for this diagnostic.
con.execute(f"""
    CREATE VIEW diagnostic_data AS
    SELECT *
    FROM read_csv(
        '{csv_path}',
        header = true,
        all_varchar = true,
        sample_size = -1
    )
""")

print("\nDuckDB column names:")
print(
    con.execute(
        "DESCRIBE diagnostic_data"
    ).fetchdf().to_string(index=False)
)

print("\nDuckDB first 10 rows:")
print(
    con.execute(
        "SELECT * FROM diagnostic_data LIMIT 10"
    ).fetchdf().to_string(index=False)
)

print("\nDuckDB distinct values in sample:")
print(
    con.execute("""
        SELECT
            COUNT(DISTINCT timestamp) AS unique_timestamps,
            COUNT(DISTINCT lat) AS unique_latitudes,
            COUNT(DISTINCT lon) AS unique_longitudes,
            COUNT(DISTINCT BT_11) AS unique_BT_11,
            COUNT(DISTINCT BT_12) AS unique_BT_12,
            COUNT(DISTINCT BTD) AS unique_BTD,
            COUNT(DISTINCT cloud_type_label) AS unique_classes
        FROM diagnostic_data
    """).fetchdf().to_string(index=False)
)

# ------------------------------------------------------------
# 4. Check a few distinct records
# ------------------------------------------------------------

print("\n4. DISTINCT RECORD CHECK")

print(
    con.execute("""
        SELECT DISTINCT *
        FROM diagnostic_data
        LIMIT 10
    """).fetchdf().to_string(index=False)
)

# ------------------------------------------------------------
# 5. Check whether DuckDB sees all rows
# ------------------------------------------------------------

print("\n5. ROW COUNT")

print(
    con.execute("""
        SELECT COUNT(*) AS total_rows
        FROM diagnostic_data
    """).fetchdf().to_string(index=False)
)

con.close()

print("\nDiagnostic completed.")
