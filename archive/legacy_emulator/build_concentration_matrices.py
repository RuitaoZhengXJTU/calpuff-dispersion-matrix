from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags, load_npz, save_npz


ROOT = Path(__file__).resolve().parent
UG_PER_LB = 453_592_370.0
# NOx -> PPB conversion (NO2-equivalent MW, 25C, 1 atm)
MW_NO2 = 46.01
MOLAR_VOLUME_L_MOL = 24.45
PPB_PER_UGM3_NO2 = MOLAR_VOLUME_L_MOL / MW_NO2  # dimensionless, ~0.5314


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert the existing sparse mass-transfer matrices into concentration-transfer matrices."
    )
    parser.add_argument(
        "--partition-dir",
        default="population_partitions/area_capped_30sqmi_population_balanced",
    )
    parser.add_argument(
        "--mass-matrix-dir",
        default=None,
        help="Defaults to sparse_transfer_matrices_20250623_18z under --partition-dir.",
    )
    parser.add_argument(
        "--data-centers",
        default="data/data_centers_example.csv",
    )
    parser.add_argument("--pollutant", choices=["pm25", "nox"], default="pm25")
    parser.add_argument("--case-tag", default=None, help="Output tag; inferred from the sparse metadata when omitted.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to the legacy generic package path when omitted.",
    )
    args = parser.parse_args()

    partition_dir = _resolve(args.partition_dir)
    mass_dir = _resolve(args.mass_matrix_dir) if args.mass_matrix_dir else partition_dir / "sparse_transfer_matrices_20250623_18z"
    case_tag = args.case_tag or _infer_case_tag(mass_dir)
    data_centers_path = _resolve(args.data_centers)
    out_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else partition_dir / f"concentration_transfer_matrices_{case_tag}"
    )
    matrices_dir = out_dir / "matrices_sparse"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    metadata = np.load(mass_dir / f"transfer_sparse_metadata_{case_tag}.npz", allow_pickle=True)
    region_ids = metadata["region_ids"].astype(str)
    hours_utc = metadata["hours_utc"].astype(str)
    areas_m2 = metadata["area_m2"].astype(np.float64)
    weather = pd.read_csv(mass_dir / "weather_by_region_hour.csv")
    pbl_by_hour = _aligned_pbl(weather, region_ids, len(hours_utc))

    # A concentration state is defined as the region's mass uniformly mixed
    # through its hourly boundary-layer volume: V = horizontal area * PBL height.
    volumes_m3 = np.empty((len(hours_utc) + 1, len(region_ids)), dtype=np.float64)
    volumes_m3[:-1] = pbl_by_hour * areas_m2[None, :]
    # The weather archive provides 24 fields for 24 transition hours. Keep the
    # final mixing volume at the last available field rather than changing data.
    volumes_m3[-1] = volumes_m3[-2]

    hourly_stats: list[dict[str, object]] = []
    for hour in range(len(hours_utc)):
        mass_matrix = load_npz(mass_dir / "matrices_sparse" / f"hour_{hour:02d}.npz").tocsc()
        concentration_matrix = (
            diags(1.0 / volumes_m3[hour + 1])
            @ mass_matrix
            @ diags(volumes_m3[hour])
        ).astype(np.float32).tocsc()
        save_npz(matrices_dir / f"hour_{hour:02d}.npz", concentration_matrix, compressed=True)
        hourly_stats.append(
            {
                "hour_index": hour,
                "time_utc": hours_utc[hour],
                "shape": f"{concentration_matrix.shape[0]}x{concentration_matrix.shape[1]}",
                "nnz": int(concentration_matrix.nnz),
                "coefficient_min": float(concentration_matrix.data.min()),
                "coefficient_max": float(concentration_matrix.data.max()),
                "mass_equivalence_max_abs_error": _mass_equivalence_error(
                    mass_matrix, concentration_matrix, volumes_m3[hour], volumes_m3[hour + 1]
                ),
            }
        )

    np.savez_compressed(
        out_dir / f"transfer_concentration_metadata_{case_tag}.npz",
        region_ids=region_ids,
        hours_utc=hours_utc,
        effective_mixing_volume_m3=volumes_m3.astype(np.float32),
        pbl_height_m_by_transition=pbl_by_hour.astype(np.float32),
        matrix_convention=np.asarray([_matrix_convention_string(args.pollutant)]),
        concentration_unit=np.asarray([_concentration_unit(args.pollutant)]),
        pollutant=np.asarray([args.pollutant]),
        method=np.asarray(["mass_matrix_similarity_transform_with_area_times_pbl_mixing_volume"]),
    )
    _write_volume_table(out_dir, region_ids, hours_utc, areas_m2, pbl_by_hour, volumes_m3)

    generators = _read_generators(data_centers_path, region_ids, args.pollutant)
    initial_response = _initial_response_matrix(generators, region_ids, volumes_m3[0], args.pollutant)
    save_npz(out_dir / "initial_generator_to_concentration_response.npz", initial_response, compressed=True)
    _write_generator_columns(out_dir, generators, args.pollutant)
    if args.pollutant == "pm25":
        _write_budget_example(out_dir, generators, region_ids, initial_response)
    else:
        _write_nox_input_template(out_dir, generators)

    pd.DataFrame(hourly_stats).to_csv(out_dir / "matrix_hourly_summary.csv", index=False)
    _write_provenance(
        out_dir, partition_dir, mass_dir, region_ids, hours_utc, generators, hourly_stats, args.pollutant, case_tag
    )
    _write_readme(out_dir, args.pollutant)
    if args.pollutant == "nox":
        _write_nox_chemistry_note(out_dir)
    print(out_dir)
    return 0


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _infer_case_tag(mass_dir: Path) -> str:
    candidates = sorted(mass_dir.glob("transfer_sparse_metadata_*.npz"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot infer case tag in {mass_dir}; pass --case-tag explicitly. "
            f"Found {len(candidates)} metadata files."
        )
    prefix = "transfer_sparse_metadata_"
    name = candidates[0].stem
    if not name.startswith(prefix):
        raise RuntimeError(f"Unexpected sparse metadata filename: {candidates[0].name}")
    return name[len(prefix):]


def _aligned_pbl(weather: pd.DataFrame, region_ids: np.ndarray, hours: int) -> np.ndarray:
    required = {"region_id", "hour_index", "boundary_layer_height_m"}
    missing = required - set(weather.columns)
    if missing:
        raise RuntimeError(f"Weather file lacks required columns: {sorted(missing)}")
    pbl_rows: list[np.ndarray] = []
    for hour in range(hours):
        frame = weather[weather["hour_index"] == hour].copy()
        ordered = frame.set_index("region_id").reindex(region_ids)
        if len(frame) != len(region_ids) or ordered["boundary_layer_height_m"].isna().any():
            raise RuntimeError(f"Hour {hour} cannot be aligned to all matrix regions.")
        values = ordered["boundary_layer_height_m"].to_numpy(float)
        if (values <= 0).any() or not np.isfinite(values).all():
            raise RuntimeError(f"Hour {hour} contains non-positive or invalid boundary-layer heights.")
        pbl_rows.append(values)
    return np.vstack(pbl_rows)


def _read_generators(path: Path, region_ids: np.ndarray, pollutant: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Generator inventory not found: {path}")
    data = pd.read_csv(path)
    needed = {"generator_id", "region_id", "lon", "lat", "pm25_budget_lb_per_hour"}
    missing = needed - set(data.columns)
    if missing:
        raise RuntimeError(f"Generator inventory lacks required columns: {sorted(missing)}")
    data = data.copy()
    data["region_id"] = data["region_id"].astype(str)
    data["pm25_budget_lb_per_hour"] = pd.to_numeric(data["pm25_budget_lb_per_hour"], errors="coerce")
    known_regions = set(region_ids)
    data = data[data["region_id"].isin(known_regions)].dropna(subset=["pm25_budget_lb_per_hour"]).reset_index(drop=True)
    if data.empty:
        raise RuntimeError("No inventory generators map to the current partition.")
    data.insert(0, "generator_matrix_index", np.arange(len(data), dtype=np.int32))
    data["initial_response_unit"] = f"{_concentration_unit(pollutant)} per lb {pollutant.upper()} emitted at t0"
    return data


def _initial_response_matrix(generators: pd.DataFrame, region_ids: np.ndarray, initial_volumes_m3: np.ndarray, pollutant: str = "pm25") -> csc_matrix:
    region_index = {region_id: idx for idx, region_id in enumerate(region_ids)}
    rows = generators["region_id"].map(region_index).to_numpy(np.int32)
    cols = generators["generator_matrix_index"].to_numpy(np.int32)
    values = UG_PER_LB / initial_volumes_m3[rows]
    if pollutant == "nox":
        values *= PPB_PER_UGM3_NO2
    return csc_matrix((values.astype(np.float32), (rows, cols)), shape=(len(region_ids), len(generators)))


def _write_budget_example(out_dir: Path, generators: pd.DataFrame, region_ids: np.ndarray, response: csc_matrix) -> None:
    """Provide a reproducible one-hour-budget example without fixing an optimization scenario."""
    initial_mass_lb = generators["pm25_budget_lb_per_hour"].to_numpy(float)
    initial_concentration = response @ initial_mass_lb
    pd.DataFrame(
        {
            "region_id": region_ids,
            "initial_concentration_ug_m3_from_inventory_1h_budget": initial_concentration,
        }
    ).to_csv(out_dir / "example_initial_concentration_from_inventory_1h_budget.csv", index=False)


def _write_nox_input_template(out_dir: Path, generators: pd.DataFrame) -> None:
    pd.DataFrame(
        {
            "generator_matrix_index": generators["generator_matrix_index"],
            "generator_id": generators["generator_id"],
            "region_id": generators["region_id"],
            "nox_initial_emitted_mass_lb": 0.0,
        }
    ).to_csv(out_dir / "nox_initial_emission_input_template.csv", index=False)


def _write_generator_columns(out_dir: Path, generators: pd.DataFrame, pollutant: str) -> None:
    columns = [
        "generator_matrix_index",
        "generator_id",
        "facility_id",
        "site_no",
        "region_id",
        "lon",
        "lat",
        "stack_height",
        "initial_response_unit",
    ]
    if pollutant == "pm25":
        columns.insert(-1, "pm25_budget_lb_per_hour")
    generators[columns].to_csv(out_dir / "generator_initial_response_columns.csv", index=False)


def _mass_equivalence_error(
    mass_matrix: csc_matrix,
    concentration_matrix: csc_matrix,
    volume_now: np.ndarray,
    volume_next: np.ndarray,
) -> float:
    # Verify D(V_next) G D(V_now)^-1 reconstructs the original mass matrix.
    recovered = diags(volume_next) @ concentration_matrix @ diags(1.0 / volume_now)
    difference = (recovered - mass_matrix).tocoo()
    return float(np.max(np.abs(difference.data))) if difference.nnz else 0.0


def _write_volume_table(
    out_dir: Path,
    region_ids: np.ndarray,
    hours_utc: np.ndarray,
    areas_m2: np.ndarray,
    pbl_by_hour: np.ndarray,
    volumes_m3: np.ndarray,
) -> None:
    labels = list(hours_utc) + [f"{hours_utc[-1]}_end_volume_persistence"]
    rows = []
    for step, label in enumerate(labels):
        pbl = pbl_by_hour[min(step, len(hours_utc) - 1)]
        rows.append(
            pd.DataFrame(
                {
                    "state_step": step,
                    "time_label": label,
                    "region_id": region_ids,
                    "area_m2": areas_m2,
                    "boundary_layer_height_m": pbl,
                    "effective_mixing_volume_m3": volumes_m3[step],
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "effective_mixing_volume_by_region_step.csv", index=False)


def _write_provenance(
    out_dir: Path,
    partition_dir: Path,
    mass_dir: Path,
    region_ids: np.ndarray,
    hours_utc: np.ndarray,
    generators: pd.DataFrame,
    hourly_stats: list[dict[str, object]],
    pollutant: str,
    case_tag: str,
) -> None:
    source_provenance = json.loads((mass_dir / "provenance.json").read_text(encoding="utf-8"))
    summary = pd.DataFrame(hourly_stats)
    provenance = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "case_tag": case_tag,
        "region_count": int(len(region_ids)),
        "generator_count": int(len(generators)),
        "hours_utc": hours_utc.tolist(),
        "pollutant": pollutant,
        "state_unit": _concentration_unit(pollutant),
        "generator_input_unit": f"lb {pollutant.upper()} emitted at t0",
        "initial_response_unit": f"{_concentration_unit(pollutant)} per lb {pollutant.upper()} emitted at t0",
        "concentration_matrix_convention": _matrix_convention_string(pollutant),
        "mass_conversion": "G_h = D(1/V_{h+1}) @ T_mass_h @ D(V_h)",
        "effective_mixing_volume": "region area [m2] multiplied by boundary-layer height [m]",
        "final_volume_handling": "The final state volume uses hour-23 PBL persistence because the source archive contains 24 transition-hour weather fields.",
        "source_mass_matrix_dir": str(mass_dir.relative_to(ROOT)).replace("\\", "/"),
        "source_method": source_provenance["method"],
        "not_official_calpuff": True,
        "known_limitation": (
            "This is a unit-consistent concentration reformulation of the existing CALPUFF-style "
            "advection-diffusion emulator, not a new CALPUFF CONC.DAT simulation."
        ),
        "validation": {
            "max_mass_reconstruction_abs_error": float(summary["mass_equivalence_max_abs_error"].max()),
            "all_concentration_coefficients_nonnegative": bool((summary["coefficient_min"] >= 0).all()),
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def _write_readme(out_dir: Path, pollutant: str) -> None:
    species_label = "PM2.5" if pollutant == "pm25" else "NOx"
    additional = (
        "The inventory's PM2.5 budget is supplied as an optional one-hour example input."
        if pollutant == "pm25"
        else "NOx mass inputs must be supplied separately in `nox_initial_emission_input_template.csv`."
    )
    text = f"""# {species_label} Concentration Transfer Matrices, 2025-06-23 18Z

This package reformulates the existing 5,042-region sparse mass-transfer matrices
as regional-average concentration matrices without changing the weather,
partition, advection-diffusion, or source assumptions.

## State convention

```python
{_matrix_convention_string(pollutant)}
```

`G_h[j, i]` maps the average {species_label} concentration ({_concentration_unit(pollutant)}) in source region `i` at the start of
hour `h` to the average concentration in target region `j` at its end.

For each hour, the effective mixing volume is:

```text
V[h, region] = area_m2 * boundary_layer_height_m
G_h = D(1 / V[h+1]) @ T_mass_h @ D(V[h])
```

## Initial generator response

```python
{_c0_convention_string(pollutant)}
```

`B0` is `initial_generator_to_concentration_response.npz`, with shape
`(n_regions, n_generators)`. It represents an instantaneous, uniformly mixed
initial release in each generator's host region. Its nonzero values have units
`{_concentration_unit(pollutant)} per lb {species_label} emitted at t0`.

{additional}

## Important limitation

These are a unit-consistent transformation of the existing sparse CALPUFF-style
emulator matrices. They are not formal CALPUFF receptor concentration output.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _write_nox_chemistry_note(out_dir: Path) -> None:
    text = """# NOx Chemistry Scope

This package treats NOx as a passive, non-depositing mass tracer so that it
preserves the same weather and transport assumptions as the PM2.5 package.
Consequently, its transfer coefficients are numerically equal to the PM2.5
coefficients when both inputs are expressed as mass concentration.

It is not a NO2 regulatory concentration package. A formal NO2 run requires
CALPUFF NOx concentration output followed by CALNO2, using either ARM or OLM.
OLM additionally requires source-specific initial NO2/NOx ratios and hourly
ambient ozone. Those inputs are not present in this project.
"""
    (out_dir / "NOX_CHEMISTRY_SCOPE.md").write_text(text, encoding="utf-8")


def _concentration_unit(pollutant: str) -> str:
    if pollutant == "pm25":
        return "ug PM2.5/m3"
    elif pollutant == "nox":
        return "ppb NOx"
    raise ValueError(f"Unknown pollutant: {pollutant}")


def _matrix_convention_string(pollutant: str) -> str:
    if pollutant == "pm25":
        return "c_next_ug_pm25_m3 = G_h @ c_now_ug_pm25_m3"
    elif pollutant == "nox":
        return "c_next_ppb_nox = G_h @ c_now_ppb_nox"
    raise ValueError(f"Unknown pollutant: {pollutant}")


def _c0_convention_string(pollutant: str) -> str:
    if pollutant == "pm25":
        return "c0_ug_m3 = B0 @ emitted_mass_lb"
    elif pollutant == "nox":
        return "c0_ppb = B0 @ emitted_mass_lb"
    raise ValueError(f"Unknown pollutant: {pollutant}")


if __name__ == "__main__":
    raise SystemExit(main())
