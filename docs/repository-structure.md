# Repository Structure

| Location | Purpose |
| --- | --- |
| `src/calpuff_matrix/` | installable formal workflow and `calpuff-matrix` CLI |
| `configs/` | canonical complete case configurations |
| `config/` | one-release compatibility YAML paths using `extends` |
| `data/raw/` | selected HRRR GRIB2 LFS inputs |
| `data/manifests/` | verifiable HRRR inventories |
| `data/examples/` | versioned generator and station examples |
| `population_partitions/` | versioned study partition inputs |
| `tools/partition/` | non-default partition construction utilities |
| `tools/visualization/` | non-default maps and concentration visualizations |
| `tools/experimental/` | non-default diagnostics |
| `archive/` | legacy code retained for provenance, not default or test-collected |
| `tests/` | active package tests only |

`outputs/`, `runs/`, generated `data/processed/`, and generated
`official_calpuff/` files are deliberately not committed.
