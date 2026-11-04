#!/usr/bin/env python3

import os
import argparse
import gc
import numpy as np
import pandas as pd

from pyhdf.SD import SD, SDC


# =====================================================
# CONFIGURATION
# =====================================================

MOD06_DIR = "data/MOD06"
MOD21_DIR = "data/MOD021"

CHUNK_DIR = "modis_chunks"

# Kept only for compatibility with the existing workflow.
# This script does NOT create the combined dataset.
OUTPUT_FILE = "modis_pixel.csv"


def parse_extraction_args():
    parser = argparse.ArgumentParser(
        description="Extract matched MODIS HDF acquisitions."
    )
    parser.add_argument("--mod06-dir", default=MOD06_DIR)
    parser.add_argument("--mod021-dir", default=MOD21_DIR)
    parser.add_argument("--chunk-dir", default=CHUNK_DIR)
    return parser.parse_args()

# India bounding box
INDIA_LAT_MIN = 6.0
INDIA_LAT_MAX = 37.0
INDIA_LON_MIN = 68.0
INDIA_LON_MAX = 97.0



# =====================================================
# PHYSICAL CONSTANTS
# =====================================================

H = 6.62607015e-34
C = 2.99792458e8
K = 1.380649e-23

# MODIS thermal band central wavelengths
LAMBDA_31 = 11.03e-6
LAMBDA_32 = 12.02e-6


# =====================================================
# HELPERS
# =====================================================

def extract_time(filename):
    """
    Extract MODIS acquisition timestamp from filename.

    Example:
        MOD06_L2.A2022323.0345.061.xxxxx.hdf

    Returns:
        2022323_0345
    """

    parts = os.path.basename(filename).split(".")

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected MODIS filename: {filename}"
        )

    return parts[1][1:] + "_" + parts[2]


def radiance_to_bt(radiance, wavelength):
    """
    Convert radiance from:

        W/m^2/um/sr

    to brightness temperature in Kelvin using
    the inverse Planck equation.
    """

    radiance = radiance.astype(np.float64)

    # Convert per micrometer to per meter
    radiance_m = radiance * 1e6

    # Invalid/non-positive radiances
    radiance_m = np.where(
        radiance_m > 0,
        radiance_m,
        np.nan
    )

    term = (
        (2.0 * H * C**2)
        / (wavelength**5 * radiance_m)
    )

    bt = (
        H * C
        / (
            wavelength
            * K
            * np.log1p(term)
        )
    )

    return bt.astype(np.float32)


# =====================================================
# MODIS 1-KM -> 5-KM SAMPLING
# =====================================================

def get_5km_sampling_indices(
    target_shape,
    one_km_shape
):
    """
    Generate the 1-km pixel positions corresponding
    to the MOD06 5-km grid.

    MODIS 5-km sampling starts at the 3rd 1-km
    pixel and uses a step of 5.

    1-based MODIS positions:

        3, 8, 13, 18, ...

    Python zero-based indices:

        2, 7, 12, 17, ...

    Supports different along-track target sizes,
    including 406 and 408 rows.

    Parameters
    ----------
    target_shape : tuple
        Shape of the MOD06 5-km grid.

    one_km_shape : tuple
        Shape of the source 1-km array.

    Returns
    -------
    along_indices : np.ndarray
        Along-track sampling indices.

    across_indices : np.ndarray
        Across-track sampling indices.
    """

    target_rows, target_cols = target_shape
    source_rows, source_cols = one_km_shape

    # -------------------------------------------------
    # Along-track indices
    # -------------------------------------------------

    along_indices = (
        2 + 5 * np.arange(
            target_rows,
            dtype=np.int32
        )
    )

    # -------------------------------------------------
    # Across-track indices
    # -------------------------------------------------

    across_indices = (
        2 + 5 * np.arange(
            target_cols,
            dtype=np.int32
        )
    )

    # -------------------------------------------------
    # Validate number of indices
    # -------------------------------------------------

    if len(along_indices) != target_rows:

        raise ValueError(
            "Incorrect number of along-track "
            "sampling indices."
        )

    if len(across_indices) != target_cols:

        raise ValueError(
            "Incorrect number of across-track "
            "sampling indices."
        )

    # -------------------------------------------------
    # Bounds check
    # -------------------------------------------------

    if along_indices[-1] >= source_rows:

        raise ValueError(
            "Along-track sampling exceeds "
            "the 1-km source array.\n"
            f"Last index: {along_indices[-1]}\n"
            f"Source rows: {source_rows}\n"
            f"Target shape: {target_shape}\n"
            f"Source shape: {one_km_shape}"
        )

    if across_indices[-1] >= source_cols:

        raise ValueError(
            "Across-track sampling exceeds "
            "the 1-km source array.\n"
            f"Last index: {across_indices[-1]}\n"
            f"Source columns: {source_cols}\n"
            f"Target shape: {target_shape}\n"
            f"Source shape: {one_km_shape}"
        )

    return (
        along_indices,
        across_indices
    )


def sample_1km_to_5km(
    arr,
    along_indices,
    across_indices
):
    """
    Select MODIS 1-km pixels corresponding to
    the MOD06 5-km grid.

    No interpolation.
    No averaging.
    No image resizing.
    """

    if arr.ndim != 2:

        raise ValueError(
            f"Expected a 2-D array, got {arr.shape}"
        )

    return arr[
        np.ix_(
            along_indices,
            across_indices
        )
    ]


# =====================================================
# CLOUD MASK FUNCTIONS
# =====================================================

def extract_cloud_mask_first_byte(
    cloud_mask_data
):
    """
    Extract the first byte from MOD06 Cloud_Mask_5km.

    Actual MOD06 files may store this as:

        (406, 270, 2)

    where:

        [:, :, 0] = first byte
        [:, :, 1] = second byte

    The first byte contains the main cloud-mask
    information.
    """

    if cloud_mask_data.ndim == 3:

        if cloud_mask_data.shape[-1] < 1:

            raise ValueError(
                f"Unexpected Cloud_Mask_5km shape: "
                f"{cloud_mask_data.shape}"
            )

        # First byte
        byte0 = cloud_mask_data[:, :, 0]

    elif cloud_mask_data.ndim == 2:

        # Some representations may expose only
        # the first byte.
        byte0 = cloud_mask_data

    else:

        raise ValueError(
            f"Unexpected Cloud_Mask_5km dimensions: "
            f"{cloud_mask_data.shape}"
        )

    return byte0.astype(np.uint8)


def decode_cloud_mask_first_byte(
    byte0
):
    """
    Decode the first byte of MODIS Cloud_Mask_5km.

    Bit 0:
        0 = Not determined
        1 = Determined

    Bits 1-2:
        00 = Cloudy
        01 = Uncertain
        10 = Probably Clear
        11 = Confident Clear

    Binary cloud label used here:

        0 = Clear
        1 = Cloudy

    Therefore:

        flag 0 -> Cloudy
        flag 1 -> Cloudy / Uncertain
        flag 2 -> Clear
        flag 3 -> Clear

    Undetermined pixels -> NaN
    """

    byte0 = byte0.astype(np.uint8)

    # -------------------------------------------------
    # Bit 0
    # -------------------------------------------------

    determined = (
        (byte0 & 0b00000001) == 1
    )

    # -------------------------------------------------
    # Bits 1-2
    # -------------------------------------------------

    cloud_flag = (
        (byte0 >> 1) & 0b00000011
    )

    # -------------------------------------------------
    # Binary cloud label
    #
    # flag 0,1 -> cloudy
    # flag 2,3 -> clear
    #
    # cloudy = 1
    # clear  = 0
    # -------------------------------------------------

    cloud_mask_binary = np.where(
        determined,
        (cloud_flag <= 1).astype(np.float32),
        np.nan
    )

    # Preserve original 0-3 flag
    cloud_mask_raw = np.where(
        determined,
        cloud_flag.astype(np.float32),
        np.nan
    )

    return (
        cloud_mask_binary.astype(np.float32),
        cloud_mask_raw.astype(np.float32)
    )


# =====================================================
# FILE MATCHING
# =====================================================

def match_files(
    mod06_files,
    mod21_files
):
    """
    Match MOD06 and MOD021 files using
    acquisition timestamp.
    """

    mod06_dict = {}
    mod21_dict = {}

    for filename in mod06_files:

        if filename.endswith(".hdf"):

            timestamp = extract_time(
                filename
            )

            mod06_dict[timestamp] = os.path.join(
                MOD06_DIR,
                filename
            )

    for filename in mod21_files:

        if filename.endswith(".hdf"):

            timestamp = extract_time(
                filename
            )

            mod21_dict[timestamp] = os.path.join(
                MOD21_DIR,
                filename
            )

    common = sorted(
        set(mod06_dict.keys()) &
        set(mod21_dict.keys())
    )

    pairs = [
        (
            mod06_dict[timestamp],
            mod21_dict[timestamp]
        )
        for timestamp in common
    ]

    return pairs


# =====================================================
# PROCESS ONE MODIS PAIR
# =====================================================

def process_pair(
    mod06_file,
    mod21_file
):

    timestamp = extract_time(
        os.path.basename(mod06_file)
    )

    print("\n" + "=" * 60)
    print(f"Processing: {timestamp}")
    print("=" * 60)

    hdf06 = None
    hdf21 = None

    try:

        # =================================================
        # OPEN MOD06
        # =================================================

        hdf06 = SD(
            mod06_file,
            SDC.READ
        )

        # =================================================
        # 5-KM GEOLOCATION
        # =================================================

        lat_sds = hdf06.select(
            "Latitude"
        )

        lon_sds = hdf06.select(
            "Longitude"
        )

        lat = (
            lat_sds
            .get()
            .astype(np.float32)
        )

        lon = (
            lon_sds
            .get()
            .astype(np.float32)
        )

        # Invalid geolocation values
        lat[lat < -900] = np.nan
        lon[lon < -900] = np.nan

        target_shape = lat.shape

        print(
            f"MOD06 5-km shape: {target_shape}"
        )

        # =================================================
        # MOD06 CLOUD TOP TEMPERATURE
        # =================================================

        ctt_sds = hdf06.select(
            "Cloud_Top_Temperature"
        )

        ctt_raw = (
            ctt_sds
            .get()
            .astype(np.float32)
        )

        ctt_attrs = ctt_sds.attributes()

        ctt_fill = ctt_attrs["_FillValue"]
        ctt_scale = ctt_attrs["scale_factor"]
        ctt_offset = ctt_attrs["add_offset"]

        # Initialize everything as NaN
        ctt = np.full(
            ctt_raw.shape,
            np.nan,
            dtype=np.float32
        )

        # Valid raw pixels
        ctt_valid = (
            ctt_raw != ctt_fill
        )

        # IMPORTANT:
        # MODIS CTT packing uses:
        #
        #     physical = (raw - add_offset) * scale_factor
        #
        # Here:
        #     add_offset   = -15000
        #     scale_factor = 0.01
        #
        # Therefore:
        #     physical = (raw + 15000) * 0.01

        ctt[ctt_valid] = (
            (ctt_raw[ctt_valid] - ctt_offset)
            * ctt_scale
        )

        # Keep only physically meaningful temperature values
        ctt = np.where(
            (ctt >= 150.0) &
            (ctt <= 350.0),
            ctt,
            np.nan
        ).astype(np.float32)

        if np.isfinite(ctt).any():

            print(
                "CTT valid values:",
                int(np.isfinite(ctt).sum())
            )

            print(
                "CTT range:",
                float(np.nanmin(ctt)),
                "to",
                float(np.nanmax(ctt)),
                "K"
            )

        else:

            print(
                "WARNING: No valid CTT values found."
            )

        # =================================================
        # MOD06 CLOUD TOP HEIGHT
        # =================================================

        cth_sds = hdf06.select(
            "Cloud_Top_Height"
        )

        cth_raw = (
            cth_sds
            .get()
            .astype(np.float32)
        )

        cth_attrs = cth_sds.attributes()

        cth_fill = cth_attrs[
            "_FillValue"
        ]

        cth_scale = cth_attrs[
            "scale_factor"
        ]

        cth_offset = cth_attrs[
            "add_offset"
        ]

        cth = np.where(
            cth_raw == cth_fill,
            np.nan,
            cth_raw * cth_scale + cth_offset
        ).astype(np.float32)

        # =================================================
        # MOD06 CLOUD PHASE
        # =================================================

        phase_sds = hdf06.select(
            "Cloud_Phase_Infrared"
        )

        phase_raw = (
            phase_sds
            .get()
            .astype(np.float32)
        )

        phase_attrs = phase_sds.attributes()

        phase_fill = phase_attrs[
            "_FillValue"
        ]

        phase_scale = phase_attrs[
            "scale_factor"
        ]

        phase_offset = phase_attrs[
            "add_offset"
        ]

        cloud_phase = np.where(
            phase_raw == phase_fill,
            np.nan,
            phase_raw * phase_scale + phase_offset
        ).astype(np.float32)

        # Additional validity protection
        cloud_phase = np.where(
            (cloud_phase >= 0) &
            (cloud_phase <= 6),
            cloud_phase,
            np.nan
        ).astype(np.float32)

        # =================================================
        # MOD06 CLOUD MASK — 5-KM
        # =================================================

        cloud_mask_5km_sds = hdf06.select(
            "Cloud_Mask_5km"
        )

        cloud_mask_5km = (
            cloud_mask_5km_sds
            .get()
            .astype(np.uint8)
        )

        print(
            f"Cloud_Mask_5km raw shape: "
            f"{cloud_mask_5km.shape}"
        )

        # -------------------------------------------------
        # Extract first byte
        # -------------------------------------------------

        cloud_mask_byte0 = (
            extract_cloud_mask_first_byte(
                cloud_mask_5km
            )
        )

        print(
            f"Cloud_Mask_5km byte-0 shape: "
            f"{cloud_mask_byte0.shape}"
        )

        # -------------------------------------------------
        # Validate 5-km shape
        # -------------------------------------------------

        if cloud_mask_byte0.shape != target_shape:

            raise ValueError(
                "Cloud_Mask_5km first-byte shape "
                "does not match Latitude shape.\n"
                f"Cloud mask byte 0: "
                f"{cloud_mask_byte0.shape}\n"
                f"Latitude: {target_shape}"
            )

        # -------------------------------------------------
        # Decode first byte
        # -------------------------------------------------

        (
            cloud_mask_binary,
            cloud_mask_raw
        ) = decode_cloud_mask_first_byte(
            cloud_mask_byte0
        )

        # =================================================
        # MOD06 1-KM CLOUD OPTICAL THICKNESS
        # =================================================

        cot_sds = hdf06.select(
            "Cloud_Optical_Thickness"
        )

        cot_raw = (
            cot_sds
            .get()
            .astype(np.float32)
        )

        print(
            f"COT 1-km shape: {cot_raw.shape}"
        )

        cot_attrs = cot_sds.attributes()

        cot_fill = cot_attrs[
            "_FillValue"
        ]

        cot_scale = cot_attrs[
            "scale_factor"
        ]

        cot_offset = cot_attrs[
            "add_offset"
        ]

        # -------------------------------------------------
        # Convert raw COT to physical values
        # -------------------------------------------------

        cot_physical = np.full(
            cot_raw.shape,
            np.nan,
            dtype=np.float32
        )

        cot_valid = (
            cot_raw != cot_fill
        )

        cot_physical[cot_valid] = (
            cot_raw[cot_valid] * cot_scale
            + cot_offset
        )

        # =================================================
        # CLOSE MOD06
        # =================================================

        hdf06.end()
        hdf06 = None

        # =================================================
        # OPEN MOD021KM
        # =================================================

        hdf21 = SD(
            mod21_file,
            SDC.READ
        )

        emissive_sds = hdf21.select(
            "EV_1KM_Emissive"
        )

        emissive = (
            emissive_sds
            .get()
            .astype(np.float32)
        )

        print(
            f"EV_1KM_Emissive shape: "
            f"{emissive.shape}"
        )

        emissive_attrs = (
            emissive_sds.attributes()
        )

        radiance_scales = np.array(
            emissive_attrs[
                "radiance_scales"
            ],
            dtype=np.float32
        )

        radiance_offsets = np.array(
            emissive_attrs[
                "radiance_offsets"
            ],
            dtype=np.float32
        )

        fill_value = emissive_attrs[
            "_FillValue"
        ]

        if emissive.ndim != 3:

            raise ValueError(
                "Unexpected EV_1KM_Emissive shape: "
                f"{emissive.shape}"
            )

        # =================================================
        # MODIS THERMAL BAND INDICES
        # =================================================

        # EV_1KM_Emissive band order confirmed from
        # the actual MOD021KM metadata:
        #
        # Band 31 -> index 10
        # Band 32 -> index 11

        idx_31 = 10
        idx_32 = 11

        print(
            f"Band 31 index: {idx_31}"
        )

        print(
            f"Band 32 index: {idx_32}"
        )

        # =================================================
        # EXTRACT DN
        # =================================================

        if emissive.shape[0] == len(
            radiance_scales
        ):

            dn_31 = emissive[
                idx_31
            ].copy()

            dn_32 = emissive[
                idx_32
            ].copy()

        elif emissive.shape[-1] == len(
            radiance_scales
        ):

            dn_31 = emissive[
                :, :, idx_31
            ].copy()

            dn_32 = emissive[
                :, :, idx_32
            ].copy()

        else:

            raise ValueError(
                "Cannot determine band dimension for "
                f"EV_1KM_Emissive shape "
                f"{emissive.shape}"
            )

        # =================================================
        # VALIDATE 1-KM SHAPE
        # =================================================

        if dn_31.ndim != 2:

            raise ValueError(
                f"Band 31 is not 2D: "
                f"{dn_31.shape}"
            )

        if dn_32.ndim != 2:

            raise ValueError(
                f"Band 32 is not 2D: "
                f"{dn_32.shape}"
            )

        if dn_31.shape != dn_32.shape:

            raise ValueError(
                "Band 31 and Band 32 shapes differ.\n"
                f"Band 31: {dn_31.shape}\n"
                f"Band 32: {dn_32.shape}"
            )

        # =================================================
        # DN -> RADIANCE
        # =================================================

        dn_31 = np.where(
            dn_31 == fill_value,
            np.nan,
            dn_31
        )

        dn_32 = np.where(
            dn_32 == fill_value,
            np.nan,
            dn_32
        )

        rad_31 = (
            dn_31
            - radiance_offsets[idx_31]
        ) * radiance_scales[idx_31]

        rad_32 = (
            dn_32
            - radiance_offsets[idx_32]
        ) * radiance_scales[idx_32]

        # =================================================
        # RADIANCE -> BRIGHTNESS TEMPERATURE
        # =================================================

        bt_11_1km = radiance_to_bt(
            rad_31,
            LAMBDA_31
        )

        bt_12_1km = radiance_to_bt(
            rad_32,
            LAMBDA_32
        )

        print(
            f"BT11 1-km shape: "
            f"{bt_11_1km.shape}"
        )

        print(
            f"BT12 1-km shape: "
            f"{bt_12_1km.shape}"
        )

        # =================================================
        # GENERATE MODIS 5-KM SAMPLING INDICES
        # =================================================

        along_5km_idx, across_5km_idx = (
            get_5km_sampling_indices(
                target_shape,
                bt_11_1km.shape
            )
        )

        print(
            f"Number of along-track indices: "
            f"{len(along_5km_idx)}"
        )

        print(
            f"Number of across-track indices: "
            f"{len(across_5km_idx)}"
        )

        # =================================================
        # SAMPLE COT 1-KM -> 5-KM
        # =================================================

        cloud_optical_thickness = (
            sample_1km_to_5km(
                cot_physical,
                along_5km_idx,
                across_5km_idx
            )
            .astype(np.float32)
        )

        print(
            f"COT 5-km shape: "
            f"{cloud_optical_thickness.shape}"
        )

        # =================================================
        # SAMPLE BT11 / BT12 1-KM -> 5-KM
        # =================================================

        BT_11 = sample_1km_to_5km(
            bt_11_1km,
            along_5km_idx,
            across_5km_idx
        ).astype(np.float32)

        BT_12 = sample_1km_to_5km(
            bt_12_1km,
            along_5km_idx,
            across_5km_idx
        ).astype(np.float32)

        print(
            f"BT_11 5-km shape: "
            f"{BT_11.shape}"
        )

        print(
            f"BT_12 5-km shape: "
            f"{BT_12.shape}"
        )

        # =================================================
        # CLOSE MOD021
        # =================================================

        hdf21.end()
        hdf21 = None

        # =================================================
        # FINAL SHAPE CHECK
        # =================================================

        arrays = {

            "lat":
                lat,

            "lon":
                lon,

            "BT_11":
                BT_11,

            "BT_12":
                BT_12,

            "CTT":
                ctt,

            "CTH":
                cth,

            "Cloud_Phase":
                cloud_phase,

            "Cloud_Optical_Thickness":
                cloud_optical_thickness,

            "Cloud_Mask_Binary":
                cloud_mask_binary,

            "Cloud_Mask_Raw":
                cloud_mask_raw,
        }

        for name, arr in arrays.items():

            if arr.shape != target_shape:

                raise ValueError(
                    f"{name} shape {arr.shape} "
                    f"does not match MOD06 target "
                    f"{target_shape}"
                )

        # =================================================
        # DERIVED FEATURE
        # =================================================

        BTD = (
            BT_11 - BT_12
        ).astype(np.float32)

        print(
            f"BTD shape: {BTD.shape}"
        )

        # =================================================
        # INDIA BOUNDING-BOX FILTER
        # =================================================

        india_mask = (

            (lat >= INDIA_LAT_MIN) &
            (lat <= INDIA_LAT_MAX) &
            (lon >= INDIA_LON_MIN) &
            (lon <= INDIA_LON_MAX)

        )

        india_mask_flat = (
            india_mask.flatten()
        )

        n_india = int(
            india_mask_flat.sum()
        )

        print(
            f"India pixels: {n_india:,}"
        )

        if n_india == 0:

            print(
                "WARNING: No pixels found inside "
                "the India bounding box."
            )

        # =================================================
        # BUILD DATAFRAME
        # =================================================

        df = pd.DataFrame({

            "timestamp":
                np.full(
                    n_india,
                    timestamp
                ),

            "lat":
                np.round(
                    lat.flatten()[
                        india_mask_flat
                    ],
                    6
                ),

            "lon":
                np.round(
                    lon.flatten()[
                        india_mask_flat
                    ],
                    6
                ),

            "BT_11":
                np.round(
                    BT_11.flatten()[
                        india_mask_flat
                    ],
                    2
                ),

            "BT_12":
                np.round(
                    BT_12.flatten()[
                        india_mask_flat
                    ],
                    2
                ),

            "BTD":
                np.round(
                    BTD.flatten()[
                        india_mask_flat
                    ],
                    2
                ),

            "Cloud_Mask_Binary":
                cloud_mask_binary.flatten()[
                    india_mask_flat
                ],

            "Cloud_Mask_Raw":
                cloud_mask_raw.flatten()[
                    india_mask_flat
                ],

            "CTT":
                np.round(
                    ctt.flatten()[
                        india_mask_flat
                    ],
                    2
                ),

            "CTH":
                np.round(
                    cth.flatten()[
                        india_mask_flat
                    ],
                    2
                ),

            "Cloud_Phase":
                cloud_phase.flatten()[
                    india_mask_flat
                ],

            "Cloud_Optical_Thickness":
                np.round(
                    cloud_optical_thickness.flatten()[
                        india_mask_flat
                    ],
                    2
                ),
        })

        # =================================================
        # SAVE CHUNK
        # =================================================

        chunk_file = os.path.join(
            CHUNK_DIR,
            f"modis_{timestamp}.csv"
        )

        df.to_csv(
            chunk_file,
            index=False
        )

        print(
            f"Saved: {chunk_file}"
        )

        print(
            f"Rows: {len(df):,}"
        )

        print(
            "Columns:",
            ", ".join(df.columns)
        )


    except Exception as e:
        print(
            f"ERROR processing {timestamp}: {e}",
            flush=True
        )
        raise

    finally:
        if hdf06 is not None:
            hdf06.end()

        if hdf21 is not None:
            hdf21.end()

        gc.collect()

# =====================================================
# MAIN
# =====================================================

def main():
    global MOD06_DIR, MOD21_DIR, CHUNK_DIR

    args = parse_extraction_args()
    MOD06_DIR = args.mod06_dir
    MOD21_DIR = args.mod021_dir
    CHUNK_DIR = args.chunk_dir

    for directory in (MOD06_DIR, MOD21_DIR):
        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)

    os.makedirs(CHUNK_DIR, exist_ok=True)


    mod06_files = sorted([

        f
        for f in os.listdir(
            MOD06_DIR
        )

        if f.endswith(".hdf")

    ])

    mod21_files = sorted([

        f
        for f in os.listdir(
            MOD21_DIR
        )

        if f.endswith(".hdf")

    ])

    print("=" * 60)
    print("MODIS EXTRACTION")
    print("=" * 60)

    print(
        f"MOD06 files:  {len(mod06_files):,}"
    )

    print(
        f"MOD021 files: {len(mod21_files):,}"
    )

    # =================================================
    # MATCH PAIRS
    # =================================================

    pairs = match_files(
        mod06_files,
        mod21_files
    )

    print(
        f"Matched pairs: {len(pairs):,}"
    )

    if len(pairs) == 0:
        raise RuntimeError(
            "No matching MOD06/MOD021 pairs found."
        )

    # =================================================
    # PROCESS SEQUENTIALLY
    # =================================================

    # -------------------------------------------------
    # SINGLE-GRANULE TEST MODE
    #
    # Currently processes ONLY the first matched pair.
    #
    # After successful validation, change:
    #
    #     pairs[:1]
    #
    # to:
    #
    #     pairs
    # -------------------------------------------------
    test_pairs = pairs    

    for i, (
        mod06_file,
        mod21_file
    ) in enumerate(
        test_pairs,
        start=1
    ):

        timestamp = extract_time(
            os.path.basename(
                mod06_file
            )
        )

        print(
            f"\n[{i}/{len(test_pairs)}] "
            f"{timestamp}"
        )

        process_pair(
            mod06_file,
            mod21_file
        )

    # =================================================
    # FINISHED
    # =================================================

    print("\n" + "=" * 60)
    print("EXTRACTION FINISHED")
    print("=" * 60)

    print(
        f"Chunks directory: {CHUNK_DIR}"
    )

    print(
        "Next step: combine chunks into modis_pixel.csv"
    )

    print("=" * 60)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
