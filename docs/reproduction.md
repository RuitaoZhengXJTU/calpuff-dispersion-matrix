# Reproduction

## Required inputs

The canonical case is `configs/dc_md_va_20250623.yaml`. It points to four
input classes: a WGS84 partition GeoJSON, generator inventory CSV, nine-station
CSV, and selected HRRR GRIB2 input plus manifests. Use Git LFS before running
`verify-hrrr`.

The partition GeoJSON is a FeatureCollection of Polygon or MultiPolygon
features with unique `properties.region_id`. The generator CSV requires
`generator_id,facility_id,site_no,region_id,lon,lat,stack_height`; its order is
the B0 column order. Station CSV requires `station_index,station_id,name,lon,lat`.

## External software

Set `CALMET_EXE`, `CALPUFF_EXE`, `CALPOST_EXE`, `WGRIB2_EXE`,
`CALPUFF_SEED`, and `CALPOST_TEMPLATE` to existing local files. The exact
environment setup is shown in the root README. No binary is downloaded by
Python package installation.

## Run from raw files

```powershell
python -m pip install -e ".[dev]"
calpuff-matrix verify-hrrr --case configs\dc_md_va_20250623.yaml
calpuff-matrix prepare --case configs\dc_md_va_20250623.yaml
calpuff-matrix build-weather --case configs\dc_md_va_20250623.yaml
calpuff-matrix build-calmet --case configs\dc_md_va_20250623.yaml
calpuff-matrix run --case configs\dc_md_va_20250623.yaml --mode b0 --max-workers 1 --continue-on-error
calpuff-matrix run --case configs\dc_md_va_20250623.yaml --mode a --max-workers 1 --continue-on-error --resume
calpuff-matrix validate --output-root outputs\official_ab_20250623_18z_ppb
```

Run `build-calmet --dry-run` and a bounded `run --dry-run` first. A stopped
matrix run can be repeated with the same A command and `--resume`.
