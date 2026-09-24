from pyhdf.SD import SD, SDC
import numpy as np


MOD06_FILE = "data/MOD06/MOD06_L2.A2022323.0345.061.2022337040351.hdf"
MOD21_FILE = "data/MOD021/MOD021KM.A2022323.0345.061.2022336211014.hdf"


def print_stats(name, raw, fill_value=None, scale=1.0, offset=0.0):
    raw = np.asarray(raw)

    if fill_value is not None:
        valid = raw[raw != fill_value]
    else:
        valid = raw.flatten()

    valid = valid[np.isfinite(valid)]

    print(f"\n{name}")
    print("-" * 50)
    print("Shape:", raw.shape)

    if fill_value is not None:
        print("Fill value:", fill_value)
        print("Fill count:", np.sum(raw == fill_value))

    if valid.size > 0:
        print("Raw min:", np.min(valid))
        print("Raw max:", np.max(valid))
        print("Raw mean:", np.mean(valid))

        physical = (valid - offset) * scale

        print("Physical min:", np.min(physical))
        print("Physical max:", np.max(physical))
        print("Physical mean:", np.mean(physical))
    else:
        print("No valid values!")


# ============================================================
# MOD06
# ============================================================

hdf06 = SD(MOD06_FILE, SDC.READ)

print("\n========================================")
print("MOD06 ACTUAL VALUE CHECK")
print("========================================")


# ------------------------------------------------------------
# CTT
# ------------------------------------------------------------

sds = hdf06.select("Cloud_Top_Temperature")
raw = sds.get()
attrs = sds.attributes()

print_stats(
    "Cloud Top Temperature (CTT)",
    raw,
    attrs["_FillValue"],
    attrs["scale_factor"],
    attrs["add_offset"]
)


# ------------------------------------------------------------
# CTH
# ------------------------------------------------------------

sds = hdf06.select("Cloud_Top_Height")
raw = sds.get()
attrs = sds.attributes()

print_stats(
    "Cloud Top Height (CTH)",
    raw,
    attrs["_FillValue"],
    attrs["scale_factor"],
    attrs["add_offset"]
)


# ------------------------------------------------------------
# CLOUD PHASE
# ------------------------------------------------------------

sds = hdf06.select("Cloud_Phase_Infrared")
raw = sds.get()
attrs = sds.attributes()

print_stats(
    "Cloud Phase",
    raw,
    attrs["_FillValue"],
    attrs["scale_factor"],
    attrs["add_offset"]
)

valid_phase = raw[raw != attrs["_FillValue"]]

print("\nCloud Phase unique values:")
values, counts = np.unique(valid_phase, return_counts=True)

for value, count in zip(values, counts):
    print(f"  {value}: {count:,}")


# ------------------------------------------------------------
# CLOUD OPTICAL THICKNESS
# ------------------------------------------------------------

sds = hdf06.select("Cloud_Optical_Thickness")
raw = sds.get()
attrs = sds.attributes()

print_stats(
    "Cloud Optical Thickness (COT)",
    raw,
    attrs["_FillValue"],
    attrs["scale_factor"],
    attrs["add_offset"]
)


# ------------------------------------------------------------
# CLOUD MASK
# ------------------------------------------------------------

sds = hdf06.select("Cloud_Mask_1km")
cloud_mask = sds.get()
attrs = sds.attributes()

print("\nCloud Mask")
print("-" * 50)
print("Shape:", cloud_mask.shape)
print("Fill value:", attrs["_FillValue"])

# Actual file is (2030, 1354, 2)
byte0 = cloud_mask[:, :, 0].astype(np.uint8)

print("\nByte 0 unique values:")
values, counts = np.unique(byte0, return_counts=True)

for value, count in zip(values, counts):
    print(f"  {value}: {count:,}")


# Decode bit 0 and bits 1-2

determined = byte0 & 0b1
cloud_flag = (byte0 >> 1) & 0b11

print("\nDetermined flag:")
values, counts = np.unique(determined, return_counts=True)

for value, count in zip(values, counts):
    print(f"  {value}: {count:,}")


print("\nCloud flag:")
values, counts = np.unique(cloud_flag, return_counts=True)

for value, count in zip(values, counts):
    print(f"  {value}: {count:,}")


hdf06.end()


# ============================================================
# MOD021KM
# ============================================================

hdf21 = SD(MOD21_FILE, SDC.READ)

print("\n========================================")
print("MOD021KM ACTUAL VALUE CHECK")
print("========================================")

sds = hdf21.select("EV_1KM_Emissive")

emissive = sds.get().astype(np.float32)
attrs = sds.attributes()

fill_value = attrs["_FillValue"]
scales = np.asarray(attrs["radiance_scales"])
offsets = np.asarray(attrs["radiance_offsets"])

print("\nEV_1KM_Emissive")
print("-" * 50)
print("Shape:", emissive.shape)
print("Fill value:", fill_value)

# ------------------------------------------------------------
# BAND 31
# ------------------------------------------------------------

dn31 = emissive[10]

valid31 = dn31[dn31 != fill_value]

rad31 = (
    valid31 - offsets[10]
) * scales[10]

print("\nBand 31")
print("-" * 50)
print("DN min:", np.min(valid31))
print("DN max:", np.max(valid31))
print("DN mean:", np.mean(valid31))
print("Radiance min:", np.min(rad31))
print("Radiance max:", np.max(rad31))
print("Radiance mean:", np.mean(rad31))


# ------------------------------------------------------------
# BAND 32
# ------------------------------------------------------------

dn32 = emissive[11]

valid32 = dn32[dn32 != fill_value]

rad32 = (
    valid32 - offsets[11]
) * scales[11]

print("\nBand 32")
print("-" * 50)
print("DN min:", np.min(valid32))
print("DN max:", np.max(valid32))
print("DN mean:", np.mean(valid32))
print("Radiance min:", np.min(rad32))
print("Radiance max:", np.max(rad32))
print("Radiance mean:", np.mean(rad32))


hdf21.end()

print("\n========================================")
print("CHECK COMPLETE")
print("========================================")

