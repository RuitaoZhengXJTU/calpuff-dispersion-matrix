# Dispersion Matrix Simulation via CALPUFF

This repository generates concentration-response operators with CALMET,
CALPUFF, and CALPOST. It is intended to be run from a clean clone by a user
who either uses the included DC/Maryland/Virginia input bundle or replaces the
four input classes below with an independent study case.

For a horizon of `H` one-hour intervals, the output contract is:

```text
c_1 = B0 @ emitted_mass_lb
c_(h+1) = A[h] @ c_h, h = 1, ..., H-1
```

`B0` maps one-hour diesel-generator emissions to regional concentration at the
end of the first hour. `A[h]` maps a regional concentration state to the next
regional concentration state. Rows are target regions; columns are source
regions or generators. The repository's default physical settings are fixed:
one-hour passive tracer releases, 15 m volume-source height, no chemistry, no
dry/wet deposition, and no decay.

The default output state is a gaseous, passive NO2-equivalent volume mixing
ratio in **ppb**. CALPUFF/CALPOST first produces mass concentration in g/m3;
the workflow applies the ideal-gas conversion using local HRRR 2 m temperature
and surface pressure. This is appropriate for a gas with molecular weight
46.0055 g/mol, not for PM2.5. A particulate-matter study must set
`concentration.output_unit: g_m3` and retain mass-concentration units (or
convert g/m3 to ug/m3 downstream).

The repository does **not** include solved matrices, CALPUFF executables, or
generated CALMET/CALPUFF case files. Those are regenerated locally. The
official output directory is configured by the case YAML and is ignored by
Git.

The included GRIB2 input files are large. Clone with Git LFS enabled:

```powershell
git lfs install
git clone <repository-url>
Set-Location dc_va_md_pollution_transfer
git lfs pull
```

## 1. Input bundle

All paths below are relative to the repository root. The four input classes
are defined by `config/official_case_20250623_18z.yaml`; a different study
starts by copying `config/official_case_template.yaml`.

| Input class | Included DC/MD/VA path | Required structure | Replace for a new study |
|---|---|---|---|
| Time, units, and CALPUFF domain | `config/official_case_20250623_18z.yaml` | YAML with `time`, `paths`, `model`, `concentration`, `calpuff_domain`, and `meteorology` blocks | Edit `time.start_utc`, `time.hours`, `concentration`, all relevant `paths`, and every grid/projection field in `calpuff_domain` so they match the new CALMET grid. |
| Subregional partition | `population_partitions/area_capped_30sqmi_population_balanced/subregions.geojson` | WGS84 GeoJSON FeatureCollection; every Polygon/MultiPolygon has unique `properties.region_id`; `area_m2` is recommended | Replace `subregions.geojson` and set `paths.partition_dir`. The preparer creates the matrix order, sources, receptors, and receptor batches. |
| Generator inventory | `data/data_centers_example.csv` | UTF-8 CSV with `generator_id,facility_id,site_no,region_id,lon,lat,stack_height` | Replace `paths.generators`. `generator_id` must be unique; `region_id` must match a partition feature; `lon,lat` are WGS84. Set `stack_height` to `15` for the repository's fixed source assumption. |
| Meteorology and CALMET sampling input | `data/raw/hrrr_20250623_18z/`, `data/raw/hrrr_20250623_00z_spinup/`, and `data/inputs/dc_va_md_20250623_18z/surrogate_surface_stations.csv` | Selected NOAA HRRR GRIB2 messages plus `.idx` inventories; station CSV has `station_index,station_id,name,lon,lat` and exactly nine rows for the included HRRR-to-CALMET route | Download selected HRRR messages for the new cycle/window, provide a station CSV for the new domain, and create a CALMET file matching `calpuff_domain`. |

### Included case

The included YAML specifies the following reproducible example:

```text
Start:       2025-06-23T18:00:00Z (2025-06-23 14:00 EDT)
Horizon:     24 one-hour intervals
Area:        District of Columbia, Maryland, and Virginia
Partition:   5,042 population-aware regions, maximum area about 30 sq mi
Generators:  data/data_centers_example.csv
```

The supplied `region_area_population_summary.csv` and
`region_partition_index_30sqmi_population_balanced.csv` are companion
metadata for the included partition. They are not required to create a new
partition from GeoJSON, but their order and IDs must not be changed when
reproducing the included case.

### Create a custom case

```powershell
Copy-Item config\official_case_template.yaml config\official_case_my_study.yaml
```

Then set the paths and domain fields in the copied YAML, place the new
partition and generator CSV at those paths, and follow the same commands below
with `config\official_case_my_study.yaml`. A changed time, domain, or
partition requires a new raw meteorology set, a new `CALMET.DAT`, a new `B0`,
and all new `A[h]` matrices. Changing only the generator CSV requires a new
`B0` but not new `A[h]` matrices.

`CALMET.DAT`, `calpuff_domain`, and the generated source/receptor coordinates
must refer to the same projected grid. Reusing the included CALMET file for a
different time or domain is invalid.

The raw-HRRR commands in Section 3 reproduce the included DC/MD/VA
79-by-38 LCC CALMET construction. `build_hrrr_calmet_inputs.py` currently
encodes that grid and its nine-station design. For another domain, generate a
matching `CALMET.DAT` with WRF-ARW/MMIF/CALMET or modify that builder's grid,
terrain, station, and control settings before running CALPUFF.

### Concentration-unit configuration

The `concentration` block specifies the final matrix state unit. The supplied
case uses:

```yaml
concentration:
  output_unit: ppb
  tracer_species: NO2_equivalent_passive_tracer
  molecular_weight_g_mol: 46.0055
```

For every region i and state endpoint h, the workflow forms
`d[i,h] = R * T[i,h] * 1e9 / (P[i,h] * MW)`, in ppb per g/m3. It writes
`B0_ppb = D[1] @ B0_g_m3` and
`A_ppb[h] = D[h+1] @ A_g_m3[h] @ inverse(D[h])`. Therefore B0 has unit
`ppb per lb emitted during [t0,t1)`, while every `A[h]` maps ppb to ppb. The
weather table must include endpoint rows `0` through `H` inclusive. Do not set
`output_unit: ppb` for PM2.5 or another particulate species.

## 2. Software and environment variables

Create the Python environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
```

Install these external programs outside the repository. CALMET/CALPUFF/CALPOST
are available from the [official CALPUFF download page](https://www.calpuff.org/calpuff/download/download.htm).
Download wgrib2 from the [NOAA/CPC wgrib2 page](https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/).

| Environment variable | Point it to | Why it is required |
|---|---|---|
| `CALMET_EXE` | CALMET executable | Converts formatted surface, upper-air, and terrain inputs into `CALMET.DAT`. |
| `CALPUFF_EXE` | CALPUFF executable | Runs every one-hour source/receptor concentration experiment. |
| `CALPOST_EXE` | CALPOST executable compatible with the supplied control template | Extracts one-hour receptor concentrations from `CALPUFF.CON`. |
| `WGRIB2_EXE` | NOAA/CPC `wgrib2.exe` | Reads selected HRRR GRIB2 fields at station and subregion locations. |
| `CALPUFF_SEED` | Official CALPUFF `.INP` seed/control template | Supplies the input-group structure used to render each CALPUFF case. |
| `CALPOST_TEMPLATE` | Official CALPOST `.INP` template | Supplies the input-group structure used to render each CALPOST case. |

Example placeholders, to be replaced with local paths:

```powershell
$env:CALMET_EXE = '<path-to-calmet-executable>'
$env:CALPUFF_EXE = '<path-to-calpuff-executable>'
$env:CALPOST_EXE = '<path-to-calpost-executable>'
$env:WGRIB2_EXE = '<path-to-wgrib2-executable>'
$env:CALPUFF_SEED = '<path-to-official-calpuff-seed-input-file>'
$env:CALPOST_TEMPLATE = '<path-to-official-calpost-template-input-file>'

$required = 'CALMET_EXE','CALPUFF_EXE','CALPOST_EXE','WGRIB2_EXE','CALPUFF_SEED','CALPOST_TEMPLATE'
foreach ($name in $required) {
  if (-not (Test-Path -LiteralPath (Get-Item "Env:$name").Value)) { throw "Missing $name" }
}
& $env:WGRIB2_EXE -version
```

Use a CALPOST executable and template that have been tested together. The
included workflow was verified with CALMET 6.5.0, CALPUFF 7.2.1, and the
CALPOST 7.1.0 example template.

## 3. Reproduce from raw HRRR input

The commands below create every intermediate file locally. If the HRRR input
directories are absent after cloning, download the required selected messages
first. The downloader reads the public NOAA HRRR archive and writes only the
fields used by the included route.

```powershell
python fetch_hrrr_selected_messages.py `
  --date 20250623 --cycle 18 --start-hour 0 --hours 25 `
  --output-dir data\raw\hrrr_20250623_18z

python fetch_hrrr_selected_messages.py `
  --date 20250623 --cycle 0 --start-hour 4 --hours 14 `
  --output-dir data\raw\hrrr_20250623_00z_spinup
```

Prepare the case-specific matrix order, source points, receptor points, and
receptor batches. This command must be repeated after changing the partition.

```powershell
python prepare_official_sparse_calpuff.py `
  --case-config config\official_case_20250623_18z.yaml `
  --receptor-points-per-region 9 --max-discrete-receptors 10000 `
  --calpuff-seed $env:CALPUFF_SEED
```

Create the regional weather table used to convert the unit concentration state
to one-hour source mass for `A[h]` and to convert gaseous CALPOST g/m3 outputs
to ppb. Request 25 HRRR files for a 24-hour horizon, because ppb conversion
needs endpoints 0 through 24:

```powershell
python build_region_weather_from_hrrr.py `
  --grib-dir data\raw\hrrr_20250623_18z `
  --subregions population_partitions\area_capped_30sqmi_population_balanced\subregions.geojson `
  --date 20250623 --cycle 18 --start-hour 0 --hours 25 `
  --wgrib2 $env:WGRIB2_EXE `
  --output data\processed\hrrr_region_weather_20250623_18z\weather_by_region_hour.csv
```

Sample station meteorology from raw HRRR and write the CALMET formatted input
files. The 00Z files are a pre-window spin-up sequence; they are included in
the CALMET input but do not change the 24-hour matrix horizon.

```powershell
python extract_hrrr_station_data.py `
  --date 20250623 --cycle 18 --start-hour 0 --hours 24 `
  --grib-dir data\raw\hrrr_20250623_18z `
  --stations-csv data\inputs\dc_va_md_20250623_18z\surrogate_surface_stations.csv `
  --output-dir data\processed\hrrr_station_met_20250623_18z `
  --wgrib2 $env:WGRIB2_EXE

python extract_hrrr_station_data.py `
  --date 20250623 --cycle 0 --start-hour 4 --hours 14 `
  --grib-dir data\raw\hrrr_20250623_00z_spinup `
  --stations-csv data\inputs\dc_va_md_20250623_18z\surrogate_surface_stations.csv `
  --output-dir data\processed\hrrr_station_met_20250623_00z_spinup `
  --wgrib2 $env:WGRIB2_EXE

python build_hrrr_calmet_inputs.py `
  --surface-csv data\processed\hrrr_station_met_20250623_18z\hrrr_surface_station_hourly.csv `
  --upper-csv data\processed\hrrr_station_met_20250623_18z\hrrr_upper_air_hourly.csv `
  --spinup-surface-csv data\processed\hrrr_station_met_20250623_00z_spinup\hrrr_surface_station_hourly.csv `
  --spinup-upper-csv data\processed\hrrr_station_met_20250623_00z_spinup\hrrr_upper_air_hourly.csv `
  --stations-csv data\inputs\dc_va_md_20250623_18z\surrogate_surface_stations.csv `
  --terrain-grib data\raw\hrrr_20250623_18z\hrrr.t18z.wrfsfcf00.grib2.selected.grib2 `
  --wgrib2 $env:WGRIB2_EXE `
  --output-dir official_calpuff\case_20250623_18z_30sqmi\met\calmet_hrrr `
  --hours 24 --itest 2 --irtype 1 --terrain-batch-size 25

Push-Location official_calpuff\case_20250623_18z_30sqmi\met\calmet_hrrr
& $env:CALMET_EXE CALMET.INP *> CALMET_RUN.log
if ($LASTEXITCODE -ne 0) { throw "CALMET failed: $LASTEXITCODE" }
Pop-Location
```

Before running CALPUFF, confirm that `CALMET.DAT` and `CALMET.DAT.aux` exist,
that `CALMET.LST` covers the requested window, and that it contains no fatal,
error, or end-of-file marker. This repository's direct HRRR station-sampling
route is reproducible but remains a simplified CALMET construction; a WRF-ARW
and MMIF workflow may be substituted when a validated regulatory meteorology
path is required.

## 4. Generate and validate the matrices

Run a real, small CALPUFF/CALPOST smoke test first:

```powershell
python run_official_ab_matrices.py `
  --case-config config\official_case_20250623_18z.yaml `
  --mode all --output-root runs\official_ab_smoke `
  --b0-source-count 1 --a-source-count 3 --a-start-hour 1 --a-hours 1 `
  --max-workers 1

python validate_official_ab.py --output-root runs\official_ab_smoke --allow-partial
```

Generate the full package after the smoke test passes:

```powershell
python run_official_ab_matrices.py `
  --case-config config\official_case_20250623_18z.yaml `
  --mode b0 --max-workers 1 --continue-on-error

python run_official_ab_matrices.py `
  --case-config config\official_case_20250623_18z.yaml `
  --mode a --max-workers 1 --continue-on-error --resume

python validate_official_ab.py --output-root outputs\official_ab_20250623_18z_ppb
```

The expected final files are:

```text
outputs/official_ab_20250623_18z_ppb/
  matrix_contract.json
  B0/B0_ppb_per_lb.npz
  B0/generator_columns.csv
  A/hour_01.npz ... A/hour_23.npz
  validation_report.json
```

To convert an existing completed official passive-gas package that was stored
in g/m3, generate the required 25-row-per-region weather table first and run:

```powershell
python convert_official_ab_to_ppb.py `
  --input-root <existing-g-m3-package> `
  --output-root <new-ppb-package> `
  --region-index official_calpuff\case_20250623_18z_30sqmi\inputs\matrix_region_index.csv `
  --weather data\processed\hrrr_region_weather_20250623_18z\weather_by_region_hour.csv `
  --molecular-weight-g-mol 46.0055

python validate_official_ab.py --output-root <new-ppb-package>
```

This post-processing route is a unit transformation of the completed official
CALPUFF response, not an emulator and not a new atmospheric simulation.

On Windows, use `--max-workers 1`. CALPUFF 7.2.1 can contend for temporary
files when cases run concurrently. If a run stops, repeat the same command
with `--resume`; completed compact receptor responses are reused.

Run the unit tests with:

```powershell
$env:PYTHONPATH = 'src'
python -m pytest -q
```
