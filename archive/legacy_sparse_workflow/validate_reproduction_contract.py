from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parent
CASE_TAG = "20250623_18z"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the current matrix reproduction contract.")
    parser.add_argument("--case", default="config/case_20250623_18z.yaml")
    parser.add_argument("--partition-dir", default="population_partitions/area_capped_30sqmi_population_balanced")
    parser.add_argument("--strict-tools", action="store_true", help="Fail if configured executables are missing.")
    args = parser.parse_args()

    case_path = _resolve(args.case)
    partition_dir = _resolve(args.partition_dir)
    config = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    report: dict[str, object] = {"case": args.case, "partition_dir": args.partition_dir, "checks": []}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: object) -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)

    check(
        "case_time_window",
        config["time"]["start_utc"] == "2025-06-23T18:00:00Z" and int(config["time"]["hours"]) == 24,
        {"start_utc": config["time"]["start_utc"], "hours": config["time"]["hours"]},
    )

    validation = json.loads((partition_dir / "validation.json").read_text(encoding="utf-8"))
    region_count = int(validation["region_count"])
    check("partition_region_count", region_count == 5042, region_count)
    check("partition_area_cap", bool(validation["area_cap_ok"]) and float(validation["max_area_sqmi"]) <= 30.0, validation)

    mass_dir = partition_dir / f"sparse_transfer_matrices_{CASE_TAG}"
    mass_meta = np.load(mass_dir / f"transfer_sparse_metadata_{CASE_TAG}.npz", allow_pickle=True)
    region_ids = mass_meta["region_ids"].astype(str)
    check("mass_metadata_region_count", len(region_ids) == region_count, len(region_ids))
    mass_stats = _check_sparse_hourly(mass_dir / "matrices_sparse", len(region_ids), 24)
    check("mass_hourly_sparse_contract", len(mass_stats["failures"]) == 0, mass_stats)
    check("mass_no_zero_source_columns", mass_stats["zero_columns"] == 0, mass_stats["zero_columns"])

    species_root = partition_dir / f"species_concentration_optimization_{CASE_TAG}"
    species_stats = {}
    for species in ("pm25", "nox"):
        species_dir = species_root / species
        meta = np.load(species_dir / f"transfer_concentration_metadata_{CASE_TAG}.npz", allow_pickle=True)
        g_stats = _check_sparse_hourly(species_dir / "matrices_sparse", len(region_ids), 24)
        response = load_npz(species_dir / "initial_generator_to_concentration_response.npz")
        ok = len(g_stats["failures"]) == 0 and response.shape == (len(region_ids), 352) and response.data.size > 0
        species_stats[species] = {"hourly": g_stats, "initial_response_shape": list(response.shape)}
        check(f"{species}_concentration_contract", ok, species_stats[species])

    comparison = _compare_species(species_root / "pm25", species_root / "nox", 24)
    check("passive_pm25_nox_same_kernel", comparison["max_abs_difference"] == 0.0, comparison)

    tool_status = {}
    for tool, configured in config["calpuff"].get("executable_paths", {}).items():
        if configured:
            path = Path(configured)
            tool_status[tool] = {"path": str(path), "exists": path.exists()}
    check(
        "configured_external_tools",
        not args.strict_tools or all(item["exists"] for item in tool_status.values()),
        tool_status,
    )

    report["ok"] = not failures
    report["failures"] = failures
    report_path = partition_dir / f"reproduction_validation_{CASE_TAG}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(report_path)
    return 0 if not failures else 1


def _check_sparse_hourly(directory: Path, n: int, hours: int) -> dict[str, object]:
    failures: list[str] = []
    zero_columns = 0
    nnz = 0
    for hour in range(hours):
        path = directory / f"hour_{hour:02d}.npz"
        if not path.exists():
            failures.append(f"missing {path.name}")
            continue
        matrix = load_npz(path).tocsc()
        if matrix.shape != (n, n):
            failures.append(f"{path.name}: shape={matrix.shape}")
        if matrix.nnz and (not np.isfinite(matrix.data).all() or matrix.data.min() < 0):
            failures.append(f"{path.name}: invalid values")
        zero_columns += int(n - np.asarray(matrix.getnnz(axis=0)).astype(bool).sum())
        nnz += int(matrix.nnz)
    return {"failures": failures, "zero_columns": zero_columns, "total_nnz": nnz}


def _compare_species(pm_dir: Path, nox_dir: Path, hours: int) -> dict[str, object]:
    max_abs = 0.0
    for hour in range(hours):
        pm = load_npz(pm_dir / "matrices_sparse" / f"hour_{hour:02d}.npz")
        nox = load_npz(nox_dir / "matrices_sparse" / f"hour_{hour:02d}.npz")
        difference = (pm - nox).data
        if difference.size:
            max_abs = max(max_abs, float(np.max(np.abs(difference))))
    return {"max_abs_difference": max_abs}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
