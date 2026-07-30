# Custom Case

Copy `configs/case_template.yaml` and make it the sole source of truth for a
new study. Relative paths are resolved from repository root.

1. Set `time.start_utc` and `time.hours`. Set `hrrr.main_window` to `hours + 1`
   forecasts so ppb conversion has endpoints 0 through H; set a previous
   `hrrr.spinup_window` for CALMET.
2. Replace `paths.subregions` and `paths.partition_dir` with a WGS84 GeoJSON
   partition with unique `region_id`s. `prepare` creates the stable region
   order and all source/receptor tables.
3. Replace `paths.generators` with the required UTF-8 inventory CSV. The
   row order becomes B0's generator order; all `region_id`s must exist in the
   partition.
4. Replace `paths.stations` with nine station locations spanning the selected
   CALMET domain. Replace every `paths.hrrr_*` and `paths.*manifest` location.
5. Replace the complete `calpuff_domain` block to match the CALMET grid. The
   direct HRRR-to-CALMET builder uses these grid fields when sampling terrain
   and writing `GEO.DAT`/`CALMET.INP`.
6. Run `fetch-hrrr`, `prepare`, `build-weather`, `build-calmet`, and `run`.

Changing time, domain, partition, or CALMET grid requires new meteorology and
new B0/A matrices. Changing only the generator CSV requires new B0 but not A.
The passive tracer, 15 m release height, chemistry/deposition, and ppb
assumptions are intentionally fixed by this repository's formal route.
