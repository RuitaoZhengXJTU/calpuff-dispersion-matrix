from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the official CALPUFF source/receptor/batch manifests before any run."
    )
    parser.add_argument(
        "--case-root",
        default="official_calpuff/case_20250623_18z_30sqmi",
        help="Prepared official CALPUFF case directory.",
    )
    parser.add_argument(
        "--partition-dir",
        default="population_partitions/area_capped_30sqmi_population_balanced",
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--source-points-per-region", type=int, default=16)
    parser.add_argument("--receptor-points-per-region", type=int, default=9)
    parser.add_argument("--max-discrete-receptors", type=int, default=10000)
    parser.add_argument(
        "--output",
        default="official_calpuff/case_20250623_18z_30sqmi/outputs/manifest_validation.json",
    )
    args = parser.parse_args()

    case_root = _resolve(args.case_root)
    partition_dir = _resolve(args.partition_dir)
    inputs = case_root / "inputs"
    matrix_index = pd.read_csv(inputs / "matrix_region_index.csv", dtype={"region_id": str})
    sources = pd.read_csv(inputs / "sources_16_per_region.csv", dtype={"region_id": str})
    receptors = pd.read_csv(inputs / "receptors_9_per_region.csv", dtype={"region_id": str})
    regions = pd.read_csv(partition_dir / "region_area_population_summary.csv", dtype={"region_id": str})

    errors: list[str] = []
    checks: dict[str, object] = {}
    _check_region_index(matrix_index, regions, errors, checks)
    _check_sources(sources, matrix_index, args.source_points_per_region, errors, checks)
    _check_receptors(receptors, matrix_index, args.receptor_points_per_region, errors, checks)
    _check_batches(inputs, receptors, args.max_discrete_receptors, errors, checks)
    _check_candidate_manifest(
        inputs / "sparse_candidate_manifest",
        region_count=len(matrix_index),
        hours=args.hours,
        max_discrete_receptors=args.max_discrete_receptors,
        errors=errors,
        checks=checks,
    )

    report = {
        "case_root": str(case_root),
        "partition_dir": str(partition_dir),
        "parameters": {
            "hours": args.hours,
            "source_points_per_region": args.source_points_per_region,
            "receptor_points_per_region": args.receptor_points_per_region,
            "max_discrete_receptors": args.max_discrete_receptors,
        },
        "ok": not errors,
        "errors": errors,
        "checks": checks,
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(output)
    return 0 if not errors else 1


def _check_region_index(
    matrix_index: pd.DataFrame,
    regions: pd.DataFrame,
    errors: list[str],
    checks: dict[str, object],
) -> None:
    expected_ids = regions["region_id"].astype(str).tolist()
    actual_ids = matrix_index["region_id"].astype(str).tolist()
    checks["region_count"] = len(actual_ids)
    checks["matrix_indices_consecutive"] = matrix_index["matrix_index"].tolist() == list(range(len(matrix_index)))
    checks["region_ids_match_partition_order"] = actual_ids == expected_ids
    checks["region_ids_unique"] = matrix_index["region_id"].is_unique
    if not checks["matrix_indices_consecutive"]:
        errors.append("matrix_region_index.csv matrix_index is not consecutive from zero")
    if not checks["region_ids_match_partition_order"]:
        errors.append("matrix_region_index.csv region order differs from the authoritative partition index")
    if not checks["region_ids_unique"]:
        errors.append("matrix_region_index.csv contains duplicate region_id values")


def _check_sources(
    sources: pd.DataFrame,
    matrix_index: pd.DataFrame,
    points_per_region: int,
    errors: list[str],
    checks: dict[str, object],
) -> None:
    counts = sources.groupby("matrix_index").size()
    weights = sources.groupby("matrix_index")["release_fraction"].sum()
    checks["source_row_count"] = len(sources)
    checks["source_points_per_region"] = int(counts.min()) if len(counts) else 0
    checks["source_counts_all_expected"] = bool(
        len(counts) == len(matrix_index)
        and counts.reindex(range(len(matrix_index))).eq(points_per_region).all()
    )
    checks["source_weights_sum_to_one"] = bool(
        len(weights) == len(matrix_index)
        and np.allclose(weights.reindex(range(len(matrix_index))).to_numpy(), 1.0, rtol=0, atol=1e-12)
    )
    checks["source_ids_unique"] = bool(sources["source_id"].is_unique)
    if not checks["source_counts_all_expected"]:
        errors.append(f"each region must have exactly {points_per_region} source points")
    if not checks["source_weights_sum_to_one"]:
        errors.append("source release_fraction does not sum to one in every region")
    if not checks["source_ids_unique"]:
        errors.append("source_id values are not unique")


def _check_receptors(
    receptors: pd.DataFrame,
    matrix_index: pd.DataFrame,
    points_per_region: int,
    errors: list[str],
    checks: dict[str, object],
) -> None:
    counts = receptors.groupby("matrix_index").size()
    checks["receptor_row_count"] = len(receptors)
    checks["receptor_counts_all_expected"] = bool(
        len(counts) == len(matrix_index)
        and counts.reindex(range(len(matrix_index))).eq(points_per_region).all()
    )
    checks["receptor_ids_unique"] = bool(receptors["receptor_id"].is_unique)
    checks["receptor_count"] = len(receptors)
    if not checks["receptor_counts_all_expected"]:
        errors.append(f"each region must have exactly {points_per_region} receptor points")
    if not checks["receptor_ids_unique"]:
        errors.append("receptor_id values are not unique")


def _check_batches(
    inputs: Path,
    receptors: pd.DataFrame,
    max_discrete_receptors: int,
    errors: list[str],
    checks: dict[str, object],
) -> None:
    manifest_path = inputs / "receptor_batch_manifest.csv"
    batch_dir = inputs / "receptor_batches"
    manifest = pd.read_csv(manifest_path)
    expected_ids = set(receptors["receptor_id"].astype(str))
    seen_ids: list[str] = []
    batch_counts: list[int] = []
    split_regions: list[str] = []
    for row in manifest.itertuples(index=False):
        path = inputs / str(row.filename)
        if not path.exists():
            errors.append(f"missing receptor batch file: {path}")
            continue
        batch = pd.read_csv(path, dtype={"receptor_id": str, "region_id": str})
        batch_counts.append(len(batch))
        seen_ids.extend(batch["receptor_id"].tolist())
        if len(batch) > max_discrete_receptors:
            errors.append(f"batch {row.batch_id} exceeds MXREC={max_discrete_receptors}")
        region_counts = batch.groupby("region_id").size()
        expected_counts = receptors.groupby("region_id").size()
        for region_id, count in region_counts.items():
            if count != expected_counts.get(region_id, -1):
                split_regions.append(str(region_id))
    checks["batch_count"] = len(manifest)
    checks["batch_row_counts"] = batch_counts
    checks["batch_total_receptors"] = len(seen_ids)
    checks["batch_ids_unique"] = len(seen_ids) == len(set(seen_ids))
    checks["batch_receptors_cover_full_table"] = set(seen_ids) == expected_ids
    checks["regions_split_across_batches"] = sorted(set(split_regions))
    if not checks["batch_ids_unique"]:
        errors.append("receptor_id is duplicated across receptor batches")
    if not checks["batch_receptors_cover_full_table"]:
        errors.append("receptor batches do not cover exactly the full receptor table")
    if split_regions:
        errors.append("at least one target region is split across receptor batches")


def _check_candidate_manifest(
    directory: Path,
    region_count: int,
    hours: int,
    max_discrete_receptors: int,
    errors: list[str],
    checks: dict[str, object],
) -> None:
    npz_path = directory / "candidate_targets_by_hour_source.npz"
    provenance_path = directory / "provenance.json"
    if not npz_path.exists():
        checks["candidate_manifest_present"] = False
        return
    with np.load(npz_path, allow_pickle=False) as data:
        indptr = np.asarray(data["indptr"])
        target_indices = np.asarray(data["target_region_indices"])
    expected_cases = hours * region_count
    counts = np.diff(indptr)
    checks["candidate_manifest_present"] = True
    checks["candidate_case_count"] = len(counts)
    checks["candidate_target_pair_count"] = len(target_indices)
    checks["candidate_indices_in_range"] = bool(
        len(target_indices) == 0
        or (target_indices.min() >= 0 and target_indices.max() < region_count)
    )
    checks["candidate_max_receptor_count"] = int(counts.max()) if len(counts) else 0
    checks["candidate_within_mxrec"] = bool(
        len(counts) == 0 or counts.max() <= max_discrete_receptors
    )
    checks["candidate_provenance_present"] = provenance_path.exists()
    if len(indptr) != expected_cases + 1:
        errors.append(
            f"candidate manifest has {len(counts)} cases; expected {expected_cases}"
        )
    if not checks["candidate_indices_in_range"]:
        errors.append("candidate target region index is out of range")
    if not checks["candidate_within_mxrec"]:
        errors.append(f"candidate receptor count exceeds MXREC={max_discrete_receptors}")
    if not provenance_path.exists():
        errors.append("candidate manifest provenance.json is missing")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
