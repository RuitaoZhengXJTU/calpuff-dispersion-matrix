# Model Assumptions

CALMET receives formatted surface and upper-air observations produced by
sampling selected HRRR fields at nine configured locations and HRRR terrain at
the configured CALMET grid-cell centers. CALMET produces `CALMET.DAT`, which
is linked privately into each CALPUFF source experiment.

Each regional A experiment releases an equivalent one-hour source mass through
16 equal-weight 15 m volume sources in the source region. CALPOST produces
one-hour receptor concentrations; the nine receptors in each target region
are averaged. B0 instead uses the generator inventory, with each generator
represented by 16 equal-weight 15 m sources. The A operator uses only targets
within its configured geometric sparse radius; B0 uses all receptor batches.

The tracer is an inert NO2-equivalent gas: chemistry, dry deposition, wet
deposition, and decay are off. ppb is an ideal-gas conversion of CALPUFF's
g/m3 output and is invalid for particulate matter. The direct HRRR station
route is a reproducible research workflow, not a replacement for a fully
validated regulatory WRF-ARW/MMIF meteorological preprocessing chain.
