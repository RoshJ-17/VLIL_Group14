from pyhdf.SD import SD, SDC
import pandas as pd
import numpy as np

hdf = SD(
    "/Users/abinayasivanesan/Desktop/projects/CLOUD/streaming_data/cloud_type_extraction/jun_13_2022/CAL_LID_L2_VFM-Standard-V4-51.2022-06-13T22-30-48ZN.hdf",
    SDC.READ
)

lat = hdf.select("Latitude").get()
lon = hdf.select("Longitude").get()

flags = hdf.select(
    "Feature_Classification_Flags"
).get()

feature_type = flags & 7
cloud_subtype = (flags >> 9) & 7

n_bins = flags.shape[1]
lat_rep = np.repeat(lat.flatten(), n_bins)
lon_rep = np.repeat(lon.flatten(), n_bins)

df = pd.DataFrame({
    "lat": lat_rep,
    "lon": lon_rep,
    "feature_type": feature_type.flatten(),
    "cloud_subtype": cloud_subtype.flatten()
})

# Keep only cloud features (feature_type == 2)
df = df[df["feature_type"] == 2].drop(columns=["feature_type"]).reset_index(drop=True)

df.to_csv(
    "calipso_labels.csv",
    index=False
)
