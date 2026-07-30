"""Build a documented receptor candidate manifest for sparse official runs.

The current emulator's positive support is used only as a computational
screening set. It is not treated as proof that omitted CALPUFF responses are
zero. The final workflow must validate this screening radius on expanded
receptor smoke tests before accepting official sparse matrices.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parent
DEFAULT_PARTITION = ROOT / "population_partitions" / "area_capped_30sqmi_population_balanced"
DEFAULT_MASS_DIR = DEFAULT_PARTITION / "sparse_transfer_matrices_20250623_18z" / "matrices_sparse"
DEFAULT_OUTPUT = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi" / "inputs" / "sparse_candidate_manifest"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-dir", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--mass-matrices", type=Path, default=DEFAULT_MASS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receptor-points-per-region", type=int, default=9)
    parser.add_argument("--max-discrete-receptors", type=int, default=10000)
    parser.add_argument("--min-coefficient", type=float, default=0.0)
    args = parser.parse_args()
    if args.receptor_points_per_region <= 0 or args.max_discrete_receptors <= 0:
        raise ValueError("receptor-points-per-region and max-discrete-receptors must be positive")

    region_index = pd.read_csv(args.partition_dir / "region_partition_index_30sqmi_population_balanced.csv")
    region_index = region_index.sort_values("matrix_index").reset_index(drop=True)
    region_ids = region_index["region_id"].astype(str).to_numpy()
    matrices = []
    for hour in range(24):
        path = args.mass_matrices / f"hour_{hour:02d}.npz"
        matrix = sparse.load_npz(path).tocsc()
        if matrix.shape != (len(region_ids), len(region_ids)):
            raise ValueError(f"Unexpected shape for {path}: {matrix.shape}")
        matrices.append(matrix)

    indptr = [0]
    target_indices: list[np.ndarray] = []
    stats: list[dict[str, object]] = []
    for hour, matrix in enumerate(matrices):
        for source_index in range(matrix.shape[1]):
            start, end = matrix.indptr[source_index], matrix.indptr[source_index + 1]
            rows = matrix.indices[start:end]
            values = matrix.data[start:end]
            selected = rows[np.asarray(values > args.min_coefficient)]
            selected = np.unique(selected.astype(np.int32))
            target_indices.append(selected)
            indptr.append(indptr[-1] + len(selected))
            receptor_count = int(len(selected) * args.receptor_points_per_region)
            stats.append(
                {
                    "hour_index": hour,
                    "source_matrix_index": source_index,
                    "source_region_id": region_ids[source_index],
                    "candidate_target_region_count": int(len(selected)),
                    "candidate_receptor_count": receptor_count,
                    "candidate_receptor_batches": int(np.ceil(receptor_count / args.max_discrete_receptors)) if receptor_count else 0,
                }
            )

    candidates = np.concatenate(target_indices).astype(np.int32) if target_indices else np.empty(0, dtype=np.int32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "candidate_targets_by_hour_source.npz",
        indptr=np.asarray(indptr, dtype=np.int64),
        target_region_indices=candidates,
        region_ids=region_ids,
        hours_utc=np.asarray([f"2025-06-23T{18 + h:02d}:00:00Z" if 18 + h < 24 else f"2025-06-24T{18 + h - 24:02d}:00:00Z" for h in range(24)]),
    )
    stats_frame = pd.DataFrame(stats)
    stats_frame.to_csv(args.output_dir / "source_hour_candidate_stats.csv", index=False)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "positive support of current sparse emulator used as a screening candidate set",
        "not_official_result": True,
        "region_count": int(len(region_ids)),
        "hours": 24,
        "receptor_points_per_region": args.receptor_points_per_region,
        "max_discrete_receptors_assumed": args.max_discrete_receptors,
        "min_coefficient": args.min_coefficient,
        "source_hour_count": int(len(stats)),
        "total_candidate_target_pairs": int(len(candidates)),
        "candidate_target_count_min": int(stats_frame["candidate_target_region_count"].min()),
        "candidate_target_count_median": float(stats_frame["candidate_target_region_count"].median()),
        "candidate_target_count_p95": float(stats_frame["candidate_target_region_count"].quantile(0.95)),
        "candidate_target_count_max": int(stats_frame["candidate_target_region_count"].max()),
        "candidate_receptor_count_max": int(stats_frame["candidate_receptor_count"].max()),
        "candidate_batch_count_total": int(stats_frame["candidate_receptor_batches"].sum()),
        "validation_required": [
            "Run expanded-receptor CALPUFF smoke tests for representative source regions and hours.",
            "Compare omitted-region response against a predeclared tolerance before accepting zeros.",
            "Record the screening threshold, candidate manifest hash, and CALPUFF output hash.",
        ],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    strategy_path = args.output_dir.parent.parent / "sparse_official_strategy.json"
    if strategy_path.exists():
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        strategy["sparse_candidate_manifest"] = {
            "directory": str(args.output_dir.relative_to(strategy_path.parent)).replace("\\", "/"),
            "source": payload["method"],
            "validated_by_official_calpuff": False,
            "screened_candidate_run_count": payload["candidate_batch_count_total"],
            "screened_candidate_receptor_count_max": payload["candidate_receptor_count_max"],
        }
        strategy.setdefault("sparsity_policy", {})["candidate_manifest_is_screening_only"] = True
        strategy_path.write_text(json.dumps(strategy, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
