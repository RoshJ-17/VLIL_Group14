#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the end-to-end aviation weather batch pipeline."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--mod06-dir", required=True)
    parser.add_argument("--mod021-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--temporal-window-minutes", type=int, default=15)
    return parser.parse_args()


def run_stage(name, command):
    print("\n" + "=" * 70, flush=True)
    print(f"STAGE: {name}", flush=True)
    print("=" * 70, flush=True)
    print("Command:", " ".join(map(str, command)), flush=True)
    t0 = time.time()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.time() - t0
    print(f"Completed: {name} [Time: {elapsed:.2f}s ({elapsed/60:.2f}m)]", flush=True)
    return elapsed


def main():
    args = parse_args()
    batch_id = args.batch_id

    if not batch_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid batch ID.")

    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")

    if args.temporal_window_minutes < 1:
        raise ValueError("Temporal window must be positive.")

    mod06_dir = Path(args.mod06_dir).resolve()
    mod021_dir = Path(args.mod021_dir).resolve()

    for directory in (mod06_dir, mod021_dir):
        if not directory.is_dir():
            raise NotADirectoryError(directory)

    batch_root = ROOT / "data/processed/pipeline_batches" / batch_id
    chunk_dir = batch_root / "chunks"
    feature_file = batch_root / f"{batch_id}_features.csv"

    # The corrected inference script writes predictions here.
    prediction_dir = batch_root / "predictions"
    prediction_file = prediction_dir / f"{batch_id}_predictions.csv"

    decision_dir = ROOT / "data/outputs/aviation_decision_support" / batch_id
    temporal_dir = decision_dir / "temporal"

    if batch_root.exists():
        shutil.rmtree(batch_root, ignore_errors=True)
    if decision_dir.exists():
        shutil.rmtree(decision_dir, ignore_errors=True)

    batch_root.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AVIATION WEATHER INTELLIGENCE - BATCH PIPELINE")
    print("=" * 70)
    print("Batch ID:", batch_id)
    print("MOD06 input:", mod06_dir)
    print("MOD021 input:", mod021_dir)
    print("Batch output:", batch_root)
    print("Decision output:", decision_dir)
    print("Temporal output:", temporal_dir)

    total_start_time = time.time()
    stage_timings = {}

    # Stage 1: Raw MODIS extraction
    stage_timings["MODIS extraction"] = run_stage(
        "MODIS extraction",
        [
            PYTHON,
            str(SCRIPT_DIR / "01_modis_extractionNew.py"),
            "--mod06-dir", str(mod06_dir),
            "--mod021-dir", str(mod021_dir),
            "--chunk-dir", str(chunk_dir),
        ],
    )

    # Stage 2: Combine extracted chunks
    stage_timings["Feature CSV combination"] = run_stage(
        "Feature CSV combination",
        [
            PYTHON,
            str(SCRIPT_DIR / "decision_combine_modis_chunks.py"),
            "--chunk-dir", str(chunk_dir),
            "--output", str(feature_file),
        ],
    )

    # Stage 3: Four-model inference.
    # decision_integrated_inference.py now runs rainfall-intensity inference
    # only for rows where precipitation is detected.
    stage_timings["Four-model inference"] = run_stage(
        "Four-model inference (conditional rainfall intensity)",
        [
            PYTHON,
            str(SCRIPT_DIR / "decision_integrated_inference.py"),
            "--input", str(feature_file),
            "--output-dir", str(prediction_dir),
            "--batch-id", batch_id,
            "--batch-size", str(args.batch_size),
        ],
    )

    if not prediction_file.is_file():
        raise FileNotFoundError(
            f"Inference did not create the expected predictions file: "
            f"{prediction_file}"
        )

    # Stage 4: Aviation decision-support rules
    stage_timings["Aviation decision support"] = run_stage(
        "Aviation decision support",
        [
            PYTHON,
            str(SCRIPT_DIR / "decision_apply_rules.py"),
            "--input", str(prediction_file),
            "--output-dir", str(decision_dir),
        ],
    )

    # Stage 5: Temporal trend and event analysis
    stage_timings["Temporal trend analysis"] = run_stage(
        "Temporal trend analysis",
        [
            PYTHON,
            str(SCRIPT_DIR / "decision_temporal_analysis.py"),
            "--input", str(prediction_file),
            "--output-dir", str(temporal_dir),
            "--window-minutes", str(args.temporal_window_minutes),
        ],
    )

    # Stage 6: Generate the unified operational profile, including temporal data
    stage_timings["Unified operational profile report"] = run_stage(
        "Unified operational profile report",
        [
            PYTHON,
            str(SCRIPT_DIR / "decision_generate_operational_profile.py"),
            "--input-dir", str(decision_dir),
            "--temporal-dir", str(temporal_dir),
            "--output-dir", str(decision_dir),
            "--batch-id", batch_id,
        ],
    )

    total_elapsed = time.time() - total_start_time

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION TIME SUMMARY")
    print("=" * 70)
    for stage_name, s_time in stage_timings.items():
        print(f"  • {stage_name:<38}: {s_time:6.2f}s ({s_time/60:4.2f}m)")
    print("-" * 70)
    print(f"  Total Pipeline Execution Time        : {total_elapsed:6.2f}s ({total_elapsed/60:4.2f}m)")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("BATCH PIPELINE COMPLETED")
    print("=" * 70)
    print("Features:", feature_file)
    print("Predictions:", prediction_file)
    print("Aviation decision outputs:", decision_dir)
    print("Temporal analysis outputs:", temporal_dir)
    print("Unified report:", decision_dir / "operational_profile_report.txt")
    print("Unified JSON:", decision_dir / "operational_profile_report.json")


if __name__ == "__main__":
    main()
