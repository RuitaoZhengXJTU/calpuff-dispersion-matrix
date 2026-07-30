# CALPUFF Dispersion Matrix

This repository produces formal CALMET/CALPUFF/CALPOST concentration-response
matrices for a partitioned study area. It contains no solved matrices,
executables, generated CALMET files, or generated CALPUFF cases. Those outputs
are regenerated locally from the committed inputs and a local CALPUFF install.

The default example is DC, Maryland, and Virginia from `2025-06-23T18:00Z`
for 24 one-hour intervals. The selected NOAA HRRR inputs are versioned with
Git LFS.

## Matrix contract

```text
c1       = B0 @ emitted_mass_lb
c(h + 1) = A[h] @ c(h), h = 1, ..., H - 1
```

Rows are target regions. `B0` has rows for target regions and columns for the
ordered generator inventory. Each `A[h]` maps regional concentration state at
hour `h` to the state at `h + 1`. The default state is ppb of a passive
NO2-equivalent gas. The source height is 15 m; chemistry, dry/wet deposition,
and decay are disabled. `A` is geometrically sparse using the configured
150 km candidate radius.

CALPUFF/CALPOST returns g/m3. For a gas only, the workflow uses HRRR 2 m
temperature and surface pressure to write ppb operators:

```text
B0_ppb   = D[1] @ B0_g_m3
A_ppb[h] = D[h + 1] @ A_g_m3[h] @ inverse(D[h])
```

where `D[h]` is the diagonal ideal-gas g/m3-to-ppb factor at endpoint `h`.
Do not use ppb for PM2.5; retain a particulate state in g/m3 or ug/m3.

## Install

```powershell
git lfs install
git clone https://github.com/RuitaoZhengXJTU/calpuff-dispersion-matrix.git
Set-Location calpuff-dispersion-matrix
git lfs pull

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install CALMET, CALPUFF, CALPOST, and NOAA/CPC wgrib2 outside this repository,
then set these executable paths in the current shell:

```powershell
$env:CALMET_EXE = '<path-to-calmet.exe>'
$env:CALPUFF_EXE = '<path-to-calpuff.exe>'
$env:CALPOST_EXE = '<path-to-calpost.exe>'
$env:WGRIB2_EXE = '<path-to-wgrib2.exe>'
$env:CALPUFF_SEED = '<path-to-verified-calpuff-seed.inp>'
$env:CALPOST_TEMPLATE = '<path-to-matching-calpost-template.inp>'
```

`CALPUFF_SEED` and `CALPOST_TEMPLATE` must be compatible with the installed
versions. The workflow will fail with contextual messages if an executable or
required input is absent.

## Five-minute smoke path

This verifies configuration and creates only a small set of controls; it does
not download HRRR or run an atmospheric simulation.

```powershell
calpuff-matrix verify-hrrr --case configs\dc_md_va_20250623.yaml
calpuff-matrix prepare --case configs\dc_md_va_20250623.yaml `
  --receptor-points-per-region 9 --max-discrete-receptors 10000
calpuff-matrix build-calmet --case configs\dc_md_va_20250623.yaml --dry-run
calpuff-matrix run --case configs\dc_md_va_20250623.yaml --mode all --dry-run `
  --output-root runs\smoke --b0-source-count 1 --a-source-count 1 --a-hours 1
```

For the complete route, run the following from a clean clone. The first two
commands are no-ops when the committed LFS files already verify.

```powershell
calpuff-matrix fetch-hrrr --case configs\dc_md_va_20250623.yaml
calpuff-matrix prepare --case configs\dc_md_va_20250623.yaml
calpuff-matrix build-weather --case configs\dc_md_va_20250623.yaml
calpuff-matrix build-calmet --case configs\dc_md_va_20250623.yaml
calpuff-matrix run --case configs\dc_md_va_20250623.yaml --mode b0 --max-workers 1 --continue-on-error
calpuff-matrix run --case configs\dc_md_va_20250623.yaml --mode a --max-workers 1 --continue-on-error --resume
calpuff-matrix validate --output-root outputs\official_ab_20250623_18z_ppb
```

## Inputs and outputs

`configs/dc_md_va_20250623.yaml` is the full case contract. It identifies the
partition GeoJSON, generator CSV, station CSV, HRRR windows/manifests, weather
table, CALMET location, and output location. Copy `configs/case_template.yaml`
for a new time, area, partition, or generator inventory.

Final local output is:

```text
outputs/official_ab_<case>_ppb/
  matrix_contract.json
  B0/B0_ppb_per_lb.npz
  B0/generator_columns.csv
  A/hour_01.npz ... A/hour_23.npz
  validation_report.json
```

See [reproduction.md](docs/reproduction.md), [custom-case.md](docs/custom-case.md),
[model-assumptions.md](docs/model-assumptions.md), and
[data-provenance.md](docs/data-provenance.md). Repository migration details
are in [migration.md](docs/migration.md).

## Development

```powershell
python -m pytest -q
ruff check .
calpuff-matrix --help
```

The test suite mocks or avoids external model calls. It never downloads HRRR
or requires CALMET/CALPUFF/CALPOST/wgrib2.
