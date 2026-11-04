#!/usr/bin/env python3

import argparse
import glob
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Combine MODIS extraction chunks into a batch feature CSV."
    )
    parser.add_argument("--chunk-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    output_file = Path(args.output)

    if not chunk_dir.is_dir():
        raise NotADirectoryError(chunk_dir)

    chunk_files = sorted(chunk_dir.glob("modis_*.csv"))

    if not chunk_files:
        raise RuntimeError(f"No MODIS chunks found in {chunk_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MODIS BATCH CHUNK COMBINATION")
    print("=" * 65)
    print("Chunk directory:", chunk_dir)
    print("Chunks found:", len(chunk_files))
    print("Output:", output_file)

    expected_columns = None
    total_rows = 0

    # Write to a temporary file so an interrupted run does not leave
    # a partially written final output.
    temporary_file = output_file.with_name(output_file.name + ".tmp")

    try:
        for index, chunk_file in enumerate(chunk_files, start=1):
            print(f"[{index}/{len(chunk_files)}] {chunk_file.name}")

            df = pd.read_csv(chunk_file)

            if df.empty:
                print("  Skipping empty chunk.")
                continue

            if expected_columns is None:
                expected_columns = list(df.columns)
                df.to_csv(temporary_file, index=False)
            else:
                if list(df.columns) != expected_columns:
                    raise ValueError(
                        f"Column mismatch in {chunk_file.name}\n"
                        f"Expected: {expected_columns}\n"
                        f"Found: {list(df.columns)}"
                    )

                df.to_csv(
                    temporary_file,
                    mode="a",
                    header=False,
                    index=False
                )

            total_rows += len(df)
            print(f"  Rows: {len(df):,}")
            del df

        if expected_columns is None or total_rows == 0:
            raise RuntimeError("All MODIS chunks were empty.")

        temporary_file.replace(output_file)

    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()
        raise

    print("\nCOMBINATION COMPLETE")
    print("Chunks:", len(chunk_files))
    print("Total rows:", f"{total_rows:,}")
    print("Columns:", expected_columns)
    print("Output size:", f"{output_file.stat().st_size / (1024**2):.2f} MB")
    print("Output:", output_file)


if __name__ == "__main__":
    main()
