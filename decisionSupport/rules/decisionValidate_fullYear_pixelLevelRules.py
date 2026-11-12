import pandas as pd
import numpy as np

path = "data/processed/full_year_2022_2023/decision_support_chunked/decision_pixel_level.csv"

checks = {
    "rain_detection_threshold_mismatch": 0,
    "non_rain_with_nonzero_final_rainfall": 0,
    "non_rain_with_conditional_intensity": 0,
    "detected_rain_missing_intensity": 0,
    "final_vs_conditional_intensity_mismatch": 0,
    "cloud_type_mapping_mismatch": 0,
    "low_cbh_rule_mismatch": 0,
    "missing_advisory": 0,
}

total = 0
advisory_counts = {}
rain_intensity_detected = 0

mapping = {
    0: "clear",
    1: "liquid_low",
    2: "liquid_middle",
    3: "liquid_high",
    4: "ice_low",
    5: "ice_middle",
    6: "ice_high",
}

for chunk in pd.read_csv(path, chunksize=200000):
    total += len(chunk)

    prob = pd.to_numeric(chunk["rain_probability"], errors="coerce")
    detected = chunk["rain_detected"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    conditional = pd.to_numeric(
        chunk["rainfall_intensity_conditional_mm_hr"], errors="coerce"
    )
    final_rain = pd.to_numeric(
        chunk["final_rainfall_mm_hr"], errors="coerce"
    )

    # 1. Rain detection must agree with the 0.33 threshold
    expected_detected = prob >= 0.33
    checks["rain_detection_threshold_mismatch"] += int(
        (detected != expected_detected).sum()
    )

    # 2. No detected rain should result in zero final rainfall
    no_rain = detected == False
    checks["non_rain_with_nonzero_final_rainfall"] += int(
        (no_rain & (final_rain.abs() > 1e-8)).sum()
    )

    # 3. Conditional intensity should be absent when rain is not detected
    checks["non_rain_with_conditional_intensity"] += int(
        (no_rain & conditional.notna()).sum()
    )

    # 4. Count detected-rain rows without intensity
    rain = detected == True
    rain_intensity_detected += int(rain.sum())
    checks["detected_rain_missing_intensity"] += int(
        (rain & conditional.isna()).sum()
    )

    # 5. Final rainfall should equal conditional intensity when detected
    comparable = rain & conditional.notna() & final_rain.notna()
    checks["final_vs_conditional_intensity_mismatch"] += int(
        (
            comparable
            & ~np.isclose(
                final_rain.fillna(0),
                conditional.fillna(0),
                rtol=1e-5,
                atol=1e-6
            )
        ).sum()
    )

    # 6. Cloud type ID and name should agree
    cloud_id = pd.to_numeric(chunk["cloud_type_id"], errors="coerce")
    expected_name = cloud_id.map(mapping)
    actual_name = chunk["cloud_type"].astype("string")
    checks["cloud_type_mapping_mismatch"] += int(
        (expected_name != actual_name).sum()
    )

    # 7. Low CBH flag should agree with CBH threshold for available cloudy pixels
    cbh = pd.to_numeric(chunk["CBH_km"], errors="coerce")
    available = chunk["cbh_available_for_cloudy_pixel"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    low_flag = chunk["low_cbh_flag"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )

    valid_cbh = available == True
    expected_low = cbh < 1.5
    checks["low_cbh_rule_mismatch"] += int(
        (valid_cbh & (low_flag != expected_low)).sum()
    )

    checks["missing_advisory"] += int(chunk["advisory_level"].isna().sum())

    for name, count in chunk["advisory_level"].value_counts().items():
        advisory_counts[name] = advisory_counts.get(name, 0) + int(count)

print("\n========== DECISION RULE VALIDATION ==========")
print("Total rows:", total)
print("Detected-rain rows:", rain_intensity_detected)

for name, count in checks.items():
    print(f"{name}: {count:,}")

print("\nAdvisory counts:")
for name, count in sorted(advisory_counts.items()):
    print(f"{name}: {count:,}")
