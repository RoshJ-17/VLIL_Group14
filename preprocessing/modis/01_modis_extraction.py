#!/usr/bin/env python3

import os
import gc
import numpy as np
import pandas as pd

from pyhdf.SD import SD, SDC
from skimage.transform import resize


# =====================================================
# CONFIGURATION
# =====================================================

MOD06_DIR = "data/MOD06"
MOD21_DIR = "data/MOD021"

CHUNK_DIR = "modis_chunks"
OUTPUT_FILE = "modis_pixel.csv"  # kept for compatibility with the existing workflow

INDIA_LAT_MIN = 6.0
INDIA_LAT_MAX = 37.0
INDIA_LON_MIN = 68.0
INDIA_LON_MAX = 97.0

os.makedirs(CHUNK_DIR, exist_ok=True)


# =====================================================
# PHYSICAL CONSTANTS
# =====================================================

H = 6.62607015e-34
C = 2.99792458e8
K = 1.380649e-23

LAMBDA_31 = 11.03e-6
LAMBDA_32 = 12.02e-6


# =====================================================
# HELPERS
# =====================================================

def extract_time(filename):
    """
    Example:
        MOD06_L2.A2022323.0345.061.xxxxx.hdf

    Returns:
        2022323_0345
    """
    parts = os.path.basename(filename).split(".")
    return parts[1][1:] + "_" + parts[2]


def radiance_to_bt(radiance, wavelength):
    """
    Convert radiance (W/m^2/um/sr) to brightness temperature (K)
    using inverse Planck.
    """
    radiance_m = radiance * 1e6
    radiance_m = np.where(radiance_m > 0, radiance_m, np.nan)

    term = (2.0 * H * C**2) / (wavelength**5 * radiance_m)
    bt = (H * C) / (wavelength * K * np.log1p(term))

    return bt.astype(np.float32)


def resize_array(arr, target_shape, order):
    if arr.shape == target_shape:
        return arr.astype(np.float32, copy=False)

    return resize(
        arr,
        target_shape,
        order=order,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.float32)


def match_files(mod06_files, mod21_files):
    mod06_dict = {}
    mod21_dict = {}

    for f in mod06_files:
        if f.endswith(".hdf"):
            mod06_dict[extract_time(f)] = os.path.join(MOD06_DIR, f)

    for f in mod21_files:
        if f.endswith(".hdf"):
            mod21_dict[extract_time(f)] = os.path.join(MOD21_DIR, f)

    common = sorted(set(mod06_dict.keys()) & set(mod21_dict.keys()))

    pairs = [
        (mod06_dict[t], mod21_dict[t])
        for t in common
    ]

    return pairs


# =====================================================
# PROCESS ONE PAIR
# =====================================================

def process_pair(mod06_file, mod21_file):
    timestamp = extract_time(os.path.basename(mod06_file))

    print("\n----------------------------------------")
    print("Processing:", timestamp)
    print("----------------------------------------")

    hdf06 = None
    hdf21 = None

    try:
        # =================================================
        # MOD06
        # =================================================
        hdf06 = SD(mod06_file, SDC.READ)

        lat = hdf06.select("Latitude").get().astype(np.float32)
        lon = hdf06.select("Longitude").get().astype(np.float32)

        lat[lat < -900] = np.nan
        lon[lon < -900] = np.nan

        ctt_raw = hdf06.select("Cloud_Top_Temperature").get().astype(np.float32)
        ctt = np.where(
            ctt_raw == -32768,
            np.nan,
            (ctt_raw - (-15000.0)) * 0.01
        ).astype(np.float32)

        cth_raw = hdf06.select("Cloud_Top_Height").get().astype(np.float32)
        cth = np.where(
            cth_raw == -32767,
            np.nan,
            cth_raw
        ).astype(np.float32)

        cloud_phase_raw = hdf06.select("Cloud_Phase_Infrared").get().astype(np.float32)
        cloud_phase = np.where(
            (cloud_phase_raw < 0) | (cloud_phase_raw > 6),
            np.nan,
            cloud_phase_raw
        ).astype(np.float32)

        cot_raw = hdf06.select("Cloud_Optical_Thickness").get().astype(np.float32)
        cloud_optical_thickness = np.where(
            cot_raw == -9999,
            np.nan,
            cot_raw * 0.01
        ).astype(np.float32)

        cloud_mask_data = hdf06.select("Cloud_Mask_1km").get()

        if cloud_mask_data.ndim == 3:
            if cloud_mask_data.shape[0] <= 10:
                cloud_mask_byte0 = cloud_mask_data[0, :, :]
            elif cloud_mask_data.shape[-1] <= 10:
                cloud_mask_byte0 = cloud_mask_data[:, :, 0]
            else:
                raise ValueError(f"Unexpected Cloud_Mask_1km shape: {cloud_mask_data.shape}")
        else:
            cloud_mask_byte0 = cloud_mask_data

        cloud_mask_byte0 = cloud_mask_byte0.astype(np.uint8)

        determined = (cloud_mask_byte0 & 0b1).astype(bool)
        cloud_flag = (cloud_mask_byte0 >> 1) & 0b11

        cloud_mask_binary = np.where(
            determined,
            (cloud_flag <= 1).astype(np.float32),
            np.nan
        ).astype(np.float32)

        cloud_mask_raw = np.where(
            determined,
            cloud_flag.astype(np.float32),
            np.nan
        ).astype(np.float32)

        hdf06.end()
        hdf06 = None

        # =================================================
        # MOD021KM
        # =================================================
        hdf21 = SD(mod21_file, SDC.READ)

        sds = hdf21.select("EV_1KM_Emissive")
        emissive = sds.get().astype(np.float32)
        attrs = sds.attributes()

        radiance_scales = np.array(attrs["radiance_scales"], dtype=np.float32)
        radiance_offsets = np.array(attrs["radiance_offsets"], dtype=np.float32)
        fill_value = attrs["_FillValue"]

        if emissive.ndim != 3:
            raise ValueError(f"Unexpected EV_1KM_Emissive shape: {emissive.shape}")

        if emissive.shape[0] >= 12:
            dn_31 = emissive[10].copy()
            dn_32 = emissive[11].copy()
        elif emissive.shape[-1] >= 12:
            dn_31 = emissive[:, :, 10].copy()
            dn_32 = emissive[:, :, 11].copy()
        else:
            raise ValueError(f"Cannot find MODIS bands 31/32 in shape: {emissive.shape}")

        dn_31[dn_31 == fill_value] = np.nan
        dn_32[dn_32 == fill_value] = np.nan

        rad_31 = (dn_31 - radiance_offsets[10]) * radiance_scales[10]
        rad_32 = (dn_32 - radiance_offsets[11]) * radiance_scales[11]

        BT_11 = radiance_to_bt(rad_31, LAMBDA_31)
        BT_12 = radiance_to_bt(rad_32, LAMBDA_32)

        hdf21.end()
        hdf21 = None

        # =================================================
        # RESIZE TO COMMON MOD06 SHAPE
        # =================================================
        target_shape = lat.shape

        BT_11 = resize_array(BT_11, target_shape, order=1)
        BT_12 = resize_array(BT_12, target_shape, order=1)

        ctt = resize_array(ctt, target_shape, order=1)
        cth = resize_array(cth, target_shape, order=1)
        cloud_optical_thickness = resize_array(cloud_optical_thickness, target_shape, order=1)

        cloud_phase = resize_array(cloud_phase, target_shape, order=0)
        cloud_mask_raw = resize_array(cloud_mask_raw, target_shape, order=0)
        cloud_mask_binary = resize_array(cloud_mask_binary, target_shape, order=0)

        arrays = {
            "lat": lat,
            "lon": lon,
            "BT_11": BT_11,
            "BT_12": BT_12,
            "CTT": ctt,
            "CTH": cth,
            "Cloud_Phase": cloud_phase,
            "Cloud_Optical_Thickness": cloud_optical_thickness,
            "Cloud_Mask_Binary": cloud_mask_binary,
            "Cloud_Mask_Raw": cloud_mask_raw,
        }

        for name, arr in arrays.items():
            if arr.shape != target_shape:
                raise ValueError(f"{name} shape {arr.shape} does not match target {target_shape}")

        # =================================================
        # DERIVED FIELD
        # =================================================
        BTD = BT_11 - BT_12

        # =================================================
        # INDIA BOUNDING-BOX FILTER
        # =================================================
        india_mask = (
            (lat >= INDIA_LAT_MIN) &
            (lat <= INDIA_LAT_MAX) &
            (lon >= INDIA_LON_MIN) &
            (lon <= INDIA_LON_MAX)
        )

        india_mask_flat = india_mask.flatten()
        n_india = int(india_mask_flat.sum())

        # =================================================
        # BUILD FILTERED DATAFRAME
        # =================================================
        df = pd.DataFrame({
            "timestamp": np.full(n_india, timestamp),
            "lat": np.round(lat.flatten()[india_mask_flat], 6),
            "lon": np.round(lon.flatten()[india_mask_flat], 6),
            "BT_11": np.round(BT_11.flatten()[india_mask_flat], 2),
            "BT_12": np.round(BT_12.flatten()[india_mask_flat], 2),
            "BTD": np.round(BTD.flatten()[india_mask_flat], 2),
            "Cloud_Mask_Binary": cloud_mask_binary.flatten()[india_mask_flat],
            "Cloud_Mask_Raw": cloud_mask_raw.flatten()[india_mask_flat],
            "CTT": np.round(ctt.flatten()[india_mask_flat], 2),
            "CTH": np.round(cth.flatten()[india_mask_flat], 2),
            "Cloud_Phase": cloud_phase.flatten()[india_mask_flat],
            "Cloud_Optical_Thickness": np.round(
                cloud_optical_thickness.flatten()[india_mask_flat], 2
            ),
        })

        # Always write one chunk per matched pair, even if it has 0 rows,
        # so the existing 81-file chunk workflow stays compatible.
        chunk_file = os.path.join(CHUNK_DIR, f"modis_{timestamp}.csv")
        df.to_csv(chunk_file, index=False)

        print(f"Saved: {chunk_file}")
        print(f"India-filtered pixels: {len(df):,}")

    except Exception as e:
        print(f"ERROR processing {timestamp}: {e}")

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
    mod06_files = sorted([
        f for f in os.listdir(MOD06_DIR)
        if f.endswith(".hdf")
    ])

    mod21_files = sorted([
        f for f in os.listdir(MOD21_DIR)
        if f.endswith(".hdf")
    ])

    print("MOD06 files:", len(mod06_files))
    print("MOD021 files:", len(mod21_files))

    pairs = match_files(mod06_files, mod21_files)

    print("Matched pairs:", len(pairs))

    for i, (mod06_file, mod21_file) in enumerate(pairs, start=1):
        print(f"\n[{i}/{len(pairs)}] {extract_time(os.path.basename(mod06_file))}")
        process_pair(mod06_file, mod21_file)

    print("\n========================================")
    print("Extraction finished")
    print("Chunks directory:", CHUNK_DIR)
    print("========================================")


if __name__ == "__main__":
    main()
