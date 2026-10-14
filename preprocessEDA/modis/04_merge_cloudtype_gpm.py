import duckdb
import os

CLOUDTYPE_FILE = "cloudtype_final.csv"
MODIS_GPM_FILE = "modisPixel_with_GPM.csv"
OUTPUT_FILE = "modisCALIPSOGPM.csv"

if not os.path.exists(CLOUDTYPE_FILE):
    raise FileNotFoundError(CLOUDTYPE_FILE)

if not os.path.exists(MODIS_GPM_FILE):
    raise FileNotFoundError(MODIS_GPM_FILE)

con = duckdb.connect()

print("Starting merge...")
print("Main file:", MODIS_GPM_FILE)
print("Label file:", CLOUDTYPE_FILE)

# Keep only the label and join keys from the cloud-type file.
# DISTINCT removes identical duplicate mappings.
con.execute(f"""
    CREATE VIEW cloud_labels AS
    SELECT DISTINCT
        CAST(timestamp AS VARCHAR) AS timestamp,
        lat,
        lon,
        cloud_type_label
    FROM read_csv_auto('{CLOUDTYPE_FILE}')
""")

# Check whether a timestamp/location maps to conflicting labels.
conflicts = con.execute("""
    SELECT timestamp, lat, lon,
           COUNT(DISTINCT cloud_type_label) AS label_count
    FROM cloud_labels
    GROUP BY timestamp, lat, lon
    HAVING COUNT(DISTINCT cloud_type_label) > 1
    LIMIT 10
""").fetchall()

if conflicts:
    print("ERROR: Conflicting cloud labels found for the same key.")
    for row in conflicts:
        print(row)
    raise ValueError(
        "Resolve conflicting labels before merging; output not written."
    )

# Collapse duplicate identical keys so the join cannot multiply rows.
con.execute("""
    CREATE VIEW labels_unique AS
    SELECT
        timestamp,
        lat,
        lon,
        MAX(cloud_type_label) AS cloud_type_label
    FROM cloud_labels
    GROUP BY timestamp, lat, lon
""")

# Left join preserves every row in the MODIS + GPM dataset.
con.execute(f"""
    COPY (
        SELECT
            m.*,
            c.cloud_type_label
        FROM read_csv_auto(
            '{MODIS_GPM_FILE}',
            sample_size = 200000
        ) AS m
        LEFT JOIN labels_unique AS c
          ON CAST(m.timestamp AS VARCHAR) = c.timestamp
         AND m.lat = c.lat
         AND m.lon = c.lon
    )
    TO '{OUTPUT_FILE}'
    (HEADER, DELIMITER ',')
""")

print("Merge completed:", OUTPUT_FILE)

# Validate row counts and label coverage.
main_rows = con.execute(f"""
    SELECT COUNT(*)
    FROM read_csv_auto('{MODIS_GPM_FILE}')
""").fetchone()[0]

output_rows = con.execute(f"""
    SELECT COUNT(*)
    FROM read_csv_auto('{OUTPUT_FILE}')
""").fetchone()[0]

matched_rows = con.execute(f"""
    SELECT COUNT(*)
    FROM read_csv_auto('{OUTPUT_FILE}')
    WHERE cloud_type_label IS NOT NULL
""").fetchone()[0]

print(f"Main-file rows: {main_rows:,}")
print(f"Output rows:    {output_rows:,}")
print(f"Rows with cloud labels: {matched_rows:,}")
print(f"Rows without cloud labels: {output_rows - matched_rows:,}")

if main_rows != output_rows:
    raise RuntimeError(
        "Row counts differ! Check join keys and duplicate mappings."
    )

con.close()
