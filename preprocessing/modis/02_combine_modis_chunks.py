#!/usr/bin/env python3

import os
import glob
import pandas as pd


CHUNK_DIR = "modis_chunks"
OUTPUT_FILE = "modis_pixel.csv"


def main():

    chunk_files = sorted(
        glob.glob(
            os.path.join(
                CHUNK_DIR,
                "modis_*.csv"
            )
        )
    )

    if not chunk_files:

        raise RuntimeError(
            "No MODIS chunk files found."
        )

    print(
        "Chunk files found:",
        len(chunk_files)
    )

    # Read and combine incrementally
    first = True

    total_rows = 0

    for i, chunk_file in enumerate(
        chunk_files,
        start=1
    ):

        print(
            f"[{i}/{len(chunk_files)}] "
            f"Reading {chunk_file}"
        )

        df = pd.read_csv(
            chunk_file
        )

        total_rows += len(df)

        if first:

            df.to_csv(
                OUTPUT_FILE,
                index=False
            )

            first = False

        else:

            df.to_csv(
                OUTPUT_FILE,
                mode="a",
                header=False,
                index=False
            )

        del df

    print("\n========================================")
    print("COMBINATION COMPLETE")
    print("========================================")

    print(
        "Chunks:",
        len(chunk_files)
    )

    print(
        "Total rows:",
        f"{total_rows:,}"
    )

    print(
        "Output:",
        OUTPUT_FILE
    )

    print(
        "Output size:",
        os.path.getsize(
            OUTPUT_FILE
        ) / (1024**2),
        "MB"
    )


if __name__ == "__main__":
    main()

