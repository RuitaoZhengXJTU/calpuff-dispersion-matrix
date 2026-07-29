from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parent
CASE_TAG = "20250623_18z"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package the current partition index and 24 hourly sparse transfer matrices into two files."
    )
    parser.add_argument(
        "--partition-dir",
        default="population_partitions/area_capped_30sqmi_population_balanced",
    )
    parser.add_argument(
        "--matrix-dir",
        default=(
            "population_partitions/area_capped_30sqmi_population_balanced/"
            "sparse_transfer_matrices_20250623_18z/matrices_sparse"
        ),
    )
    parser.add_argument(
        "--region-output",
        default=(
            "population_partitions/area_capped_30sqmi_population_balanced/"
            "region_partition_index_30sqmi_population_balanced.csv"
        ),
    )
    parser.add_argument(
        "--matrix-output",
        default=(
            "population_partitions/area_capped_30sqmi_population_balanced/"
            "transfer_matrices_24h_sparse_20250623_18z.npz"
        ),
    )
    args = parser.parse_args()

    partition_dir = _resolve(args.partition_dir)
    matrix_dir = _resolve(args.matrix_dir)
    region_output = _resolve(args.region_output)
    matrix_output = _resolve(args.matrix_output)

    region_index = _build_region_index(partition_dir)
    region_output.parent.mkdir(parents=True, exist_ok=True)
    region_index.to_csv(region_output, index=False)

    matrix_payload = _build_matrix_payload(matrix_dir, region_index)
    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(matrix_output, **matrix_payload)

    print(region_output)
    print(matrix_output)
    return 0


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def _build_region_index(partition_dir: Path) -> pd.DataFrame:
    regions = pd.read_csv(partition_dir / "region_area_population_summary.csv")
    regions = regions.sort_values("region_id").reset_index(drop=True)
    regions.insert(0, "matrix_index", np.arange(len(regions), dtype=np.int32))
    regions.insert(1, "case_tag", CASE_TAG)
    regions.insert(2, "partition_name", "area_capped_30sqmi_population_balanced")
    regions.insert(3, "partition_basis", "TIGER/2020 Census block population allocation; maximum area 30 square miles; recursive population-aware subdivision")
    regions.insert(4, "matrix_convention", "T[h,j,i] is source region i to target region j; x_next = T[h] @ x_now")
    regions.insert(5, "geometry_file", "subregions.geojson")
    regions.insert(6, "simplified_geometry_file", "subregions_simplified.geojson")
    return regions


def _build_matrix_payload(matrix_dir: Path, region_index: pd.DataFrame) -> dict[str, np.ndarray]:
    matrices = [load_npz(matrix_dir / f"hour_{hour:02d}.npz").tocsc() for hour in range(24)]
    n = len(region_index)
    for hour, matrix in enumerate(matrices):
        if matrix.shape != (n, n):
            raise RuntimeError(f"hour_{hour:02d}.npz shape {matrix.shape} does not match region count {n}")
        if matrix.data.size and matrix.data.min() < 0:
            raise RuntimeError(f"hour_{hour:02d}.npz contains negative transfer coefficients")

    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray(["sparse_csc_24h_v1"]),
        "case_tag": np.asarray([CASE_TAG]),
        "created_utc": np.asarray([datetime.utcnow().isoformat() + "Z"]),
        "matrix_convention": np.asarray(["T[h,j,i] is source region i to target region j; x_next = T[h] @ x_now"]),
        "shape": np.asarray([24, n, n], dtype=np.int64),
        "region_ids": region_index["region_id"].astype(str).to_numpy(),
        "matrix_indices": region_index["matrix_index"].to_numpy(np.int32),
        "hours_utc": np.asarray([f"2025-06-23T{18 + h:02d}:00:00+00:00" if h < 6 else f"2025-06-24T{h - 6:02d}:00:00+00:00" for h in range(24)]),
        "nnz_by_hour": np.asarray([matrix.nnz for matrix in matrices], dtype=np.int64),
    }
    for hour, matrix in enumerate(matrices):
        prefix = f"h{hour:02d}"
        payload[f"{prefix}_data"] = matrix.data.astype(np.float32, copy=False)
        payload[f"{prefix}_indices"] = matrix.indices.astype(np.int32, copy=False)
        payload[f"{prefix}_indptr"] = matrix.indptr.astype(np.int32, copy=False)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
