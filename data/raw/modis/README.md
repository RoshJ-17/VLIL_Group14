# MODIS Raw Data

This directory contains raw MODIS Level 1B calibrated radiances and Level 2 cloud product HDF files.

### Files:
- `MOD021KM.A2023111.0345.061.2023111131202.hdf` (Split into `.z01`, `.z02`, `.z03`, `.zip`)
- `MOD06_L2.A2023111.0345.061.2023134235838.hdf` (Split into `.z01`, `.z02`, `.zip`)

### How to extract:
To combine and unzip on macOS/Linux:
```bash
zip -s 0 MOD021KM.A2023111.0345.061.2023111131202.zip --out MOD021KM_full.zip && unzip MOD021KM_full.zip
zip -s 0 MOD06_L2.A2023111.0345.061.2023134235838.zip --out MOD06_L2_full.zip && unzip MOD06_L2_full.zip
```
