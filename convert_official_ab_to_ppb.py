"""Convert an official CALPUFF g/m3 A/B package to gaseous ppb units.

This script does not run CALMET, CALPUFF, or CALPOST. It applies the ideal-gas
unit change to an already completed passive-gas matrix package using the
region/hour temperature and surface-pressure fields sampled from HRRR.

For every regional state endpoint h, define D[h] as the diagonal matrix of
``ppb_per_g_m3`` factors. The resulting operators are:

    B0_ppb = D[1] @ B0_g_m3
    A_ppb[h] = D[h+1] @ A_g_m3[h] @ inverse(D[h])

The relationship is exact for the linear, non-reactive CALPUFF tracer runs
represented by the input package. It is only meaningful for a gaseous species;
do not use it for PM2.5 mass concentration.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import diags, load_npz, save_npz

from concentration_units import DEFAULT_NO2_MOLECULAR_WEIGHT_G_MOL, ppb_factor_array


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="completed g/m3 package")
    parser.add_argument("--output-root", type=Path, required=True, help="new ppb package directory")
    parser.add_argument("--region-index", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--b0-input", type=Path, default=None)
    parser.add_argument("--molecular-weight-g-mol", type=float, default=DEFAULT_NO2_MOLECULAR_WEIGHT_G_MOL)
    parser.add_argument("--tracer-species", default="NO2_equivalent_passive_tracer")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.molecular_weight_g_mol <= 0:
        raise ValueError("molecular-weight-g-mol must be positive")
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if input_root == output_root:
        raise ValueError("output-root must differ from input-root")
    contract_path = input_root / "matrix_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output-root is not empty: {output_root}; pass --overwrite to replace matrix files")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    horizon_hours = int(contract.get("horizon_hours", 24))
    a_hours = [int(hour) for hour in contract.get("a_hour_indices", range(1, horizon_hours))]
    region_index = pd.read_csv(args.region_index).sort_values("matrix_index").reset_index(drop=True)
    region_ids = region_index["region_id"].astype(str).to_numpy()
    if len(region_ids) != int(contract["region_count"]):
        raise ValueError("region-index row count does not match input matrix contract")
    weather = pd.read_csv(args.weather)
    factors = ppb_factor_array(
        weather,
        region_ids,
        horizon_hours,
        float(args.molecular_weight_g_mol),
    )

    output_b0 = output_root / "B0"
    output_a = output_root / "A"
    output_b0.mkdir(parents=True, exist_ok=True)
    output_a.mkdir(parents=True, exist_ok=True)
    b0_input = args.b0_input or input_root / str(contract.get("b0_file", "B0/B0_g_m3_per_lb.npz"))
    if not b0_input.exists():
        raise FileNotFoundError(b0_input)
    b0 = load_npz(b0_input).tocsc()
    expected_b0_shape = (len(region_ids), int(contract["generator_count"]))
    if b0.shape != expected_b0_shape:
        raise ValueError(f"B0 shape {b0.shape} != {expected_b0_shape}")
    b0_ppb = (diags(factors[1]) @ b0).tocsc()
    save_npz(output_b0 / "B0_ppb_per_lb.npz", b0_ppb, compressed=True)

    for hour in a_hours:
        source_path = input_root / "A" / f"hour_{hour:02d}.npz"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        matrix = load_npz(source_path).tocsc()
        expected_a_shape = (len(region_ids), len(region_ids))
        if matrix.shape != expected_a_shape:
            raise ValueError(f"A[{hour}] shape {matrix.shape} != {expected_a_shape}")
        converted = (diags(factors[hour + 1]) @ matrix @ diags(1.0 / factors[hour])).tocsc()
        save_npz(output_a / source_path.name, converted, compressed=True)

    _copy_if_present(input_root / "B0" / "generator_columns.csv", output_b0 / "generator_columns.csv")
    _write_provenance(
        input_root / "B0" / "provenance.json",
        output_b0 / "provenance.json",
        {"matrix": "B0", "shape": list(b0_ppb.shape), "nnz": int(b0_ppb.nnz), "unit": _b0_unit(args)},
        args,
        source_path=b0_input,
    )
    _write_provenance(
        input_root / "A" / "provenance.json",
        output_a / "provenance.json",
        {"matrix": "A", "hours_converted": a_hours, "unit": _state_unit(args)},
        args,
        source_path=input_root / "A",
    )

    contract.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "concentration_unit": "ppb",
        "state_unit": _state_unit(args),
        "b0_unit": _b0_unit(args),
        "tracer_species": args.tracer_species,
        "molecular_weight_g_mol": float(args.molecular_weight_g_mol),
        "b0_file": "B0/B0_ppb_per_lb.npz",
        "gas_unit_conversion": "ppb = (g/m3) * R * temperature_k * 1e9 / (pressure_pa * molecular_weight_g_mol)",
        "unit_conversion": {
            "source_matrix_package": str(input_root),
            "weather": str(args.weather),
            "formula": "B0_ppb = D[1] @ B0_g_m3; A_ppb[h] = D[h+1] @ A_g_m3[h] @ inverse(D[h])",
            "D_definition": "diagonal(ppb_per_g_m3 by target/source region at state endpoint h)",
            "weather_endpoint_indices": list(range(horizon_hours + 1)),
        },
    })
    (output_root / "matrix_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "b0_nnz": int(b0_ppb.nnz), "a_hours": a_hours}))
    return 0


def _copy_if_present(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copy2(source, target)


def _write_provenance(
    source: Path,
    target: Path,
    updates: dict[str, object],
    args: argparse.Namespace,
    *,
    source_path: Path,
) -> None:
    payload = json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}
    payload.update(updates)
    payload.update({
        "concentration_unit": "ppb",
        "tracer_species": args.tracer_species,
        "molecular_weight_g_mol": float(args.molecular_weight_g_mol),
        "postprocess": "ideal-gas conversion of an existing official CALPUFF passive-gas output; CALPUFF was not rerun",
        "source_g_m3_path": str(source_path),
        "ppb_conversion": "ppb_per_g_m3 = R * temperature_k * 1e9 / (pressure_pa * molecular_weight_g_mol)",
        "uses_calpuff": True,
        "not_emulator": True,
    })
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _state_unit(args: argparse.Namespace) -> str:
    return f"ppb {args.tracer_species} volume mixing ratio"


def _b0_unit(args: argparse.Namespace) -> str:
    return f"ppb {args.tracer_species} per lb emitted during [t0,t1)"


if __name__ == "__main__":
    raise SystemExit(main())
