# Migration

The formal root scripts were moved into the installable `calpuff_matrix`
package. Existing root names remain one-release wrappers and emit a warning.
Use `calpuff-matrix` for all new work.

The previous fallback/emulator implementation is now in
`archive/legacy_emulator/`; it is retained only to reproduce exploratory work
and must not be presented as CALPUFF output. Superseded sparse, MMIF,
initial-response, surrogate, smoke, and assembly programs are in
`archive/legacy_sparse_workflow/`. Both archive directories contain a README
describing their replacement commands.

`requirements.txt` and `pytest.ini` were replaced by `pyproject.toml`. Case
configuration moved to `configs/`; the old `config/` YAML files use explicit
`extends` so existing commands remain readable for one release.
