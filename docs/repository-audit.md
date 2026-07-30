# Repository Audit

Audit date: 2026-07-30. This document records the root layout before the
`calpuff_matrix` package refactor and the deliberate disposition of each class
of file. It does not change the formal CALPUFF science or the ppb contract.

## Formal workflow moved to `src/calpuff_matrix`

| Previous root module | New module | Public command |
| --- | --- | --- |
| `official_case_config.py` | `config.py` | internal configuration loader |
| `official_case_builder.py` | `case_builder.py` | internal case rendering |
| `concentration_units.py` | `units.py` | internal unit helpers |
| `fetch_hrrr_selected_messages.py` | `hrrr.py` | `fetch-hrrr` |
| `extract_hrrr_station_data.py` | `station_data.py` | composed by `build-calmet` |
| `build_hrrr_calmet_inputs.py` | `calmet_inputs.py` | composed by `build-calmet` |
| `build_region_weather_from_hrrr.py` | `weather.py` | `build-weather` |
| `prepare_official_sparse_calpuff.py` | `preparation.py` | `prepare` |
| `run_official_ab_matrices.py` | `matrices.py` | `run` |
| `convert_official_ab_to_ppb.py` | `conversion.py` | `convert-units` |
| `validate_official_ab.py` | `validation.py` | `validate` |
| `parse_calpost_tseries.py` | `calpost.py` | called by `run` |

The same root names remain one-release wrappers. They issue a deprecation
notice and import only the packaged implementation.

## Archived workflows

`archive/legacy_emulator/` contains `harness.py`, `src/transfer_matrix/`,
pseudo-meteorology, fallback matrix builders, and partition comparison scripts.
Those programs can estimate a transfer operator but are not CALPUFF output.

`archive/legacy_sparse_workflow/` contains superseded candidate-manifest,
initial-response, surrogate CALMET, MMIF/smoke, sparse assembly, and readiness
workflows. Reusable runtime behavior was extracted to `runtime.py`; archived
programs are not test-collected or documented as the default route.

Reusable support programs remain active under `tools/partition/`,
`tools/visualization/`, and `tools/experimental/`.

## Tests and imports before migration

Active formal tests were `test_calpost_tseries_parser`, `test_concentration_units`,
`test_convert_official_ab_to_ppb`, `test_hrrr`, `test_official_ab_contract`,
`test_official_case_config`, `test_receptor_batches`, and
`test_region_weather_hrrr`. Fallback/old-pipeline tests are archived beside
their legacy implementation. Test imports are updated to package-qualified
modules; no test relies on root-level script imports.

## Data and LFS inventory

The 39 selected HRRR GRIB2 files remain in `data/raw/` and are tracked with
Git LFS: 25 main-window files (`2025-06-23 18Z`, f00--f24) and 14 spin-up
files (`2025-06-23 00Z`, f04--f17). Their inventories move to
`data/manifests/`, retaining NOAA source URLs, selected byte ranges, expected
subset size, and SHA-256 values. No solved A/B matrices, generated CALMET
files, generated cases, or external executables are tracked.

## Removed configuration duplication

`requirements.txt` and `pytest.ini` are replaced by `pyproject.toml` after
their dependencies and pytest collection settings are represented there.
Legacy configuration paths under `config/` remain for one release through
explicit YAML `extends` declarations to the canonical `configs/` files.
