from pyhdf.SD import SD, SDC

mod06_file = "data/MOD06/MOD06_L2.A2022323.0345.061.2022337040351.hdf"
mod21_file = "data/MOD021/MOD021KM.A2022323.0345.061.2022336211014.hdf"


# ============================================================
# MOD06
# ============================================================

hdf06 = SD(mod06_file, SDC.READ)

print("\n========== MOD06 ==========")

for name in [
    "Latitude",
    "Longitude",
    "Cloud_Top_Temperature",
    "Cloud_Top_Height",
    "Cloud_Phase_Infrared",
    "Cloud_Optical_Thickness",
    "Cloud_Mask_1km"
]:
    sds = hdf06.select(name)

    print(f"\n{name}")
    print("Shape:", sds.get().shape)

    attrs = sds.attributes()

    for key in [
        "_FillValue",
        "scale_factor",
        "add_offset",
        "valid_range",
        "units",
        "Cell_Along_Swath_Sampling",
        "Cell_Across_Swath_Sampling"
    ]:
        if key in attrs:
            print(f"{key}: {attrs[key]}")

hdf06.end()


# ============================================================
# MOD021KM
# ============================================================

hdf21 = SD(mod21_file, SDC.READ)

print("\n========== MOD021KM ==========")

sds = hdf21.select("EV_1KM_Emissive")

print("Shape:", sds.get().shape)

attrs = sds.attributes()

for key in [
    "_FillValue",
    "radiance_scales",
    "radiance_offsets",
    "radiance_units",
    "band_names"
]:
    if key in attrs:
        print(f"{key}: {attrs[key]}")

hdf21.end()
