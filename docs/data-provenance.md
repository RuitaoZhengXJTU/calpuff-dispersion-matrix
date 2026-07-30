# Data Provenance

The example uses NOAA HRRR CONUS surface-product fields from the `2025-06-23`
00Z spin-up (f04--f17) and 18Z main (f00--f24) cycles. `data/manifests/`
records NOAA source URLs, selected GRIB2 index byte ranges, expected local
subset sizes, and SHA-256 digests. `calpuff-matrix verify-hrrr` validates the
local LFS/downloaded files without a network request.

Only selected GRIB2 subsets are committed with LFS. Generated station data,
regional weather, CALMET files, CALPUFF/CALPOST run folders, and matrices are
local products and ignored by Git. The partition and sample generator/station
files are small versioned inputs; their stable identifiers determine matrix
row/column semantics.
