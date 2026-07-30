from __future__ import annotations

import argparse
from pathlib import Path

from src.transfer_matrix.assemble_sparse import assemble_sparse_response_matrices


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble official CALPUFF/CALPOST receptor exports into sparse hourly matrices."
    )
    parser.add_argument(
        "--case-root",
        default="official_calpuff/case_20250623_18z_30sqmi/runs/official_sparse_full_fixed",
        help="Directory containing hour_00/source_<region_id>/receptors.csv cases.",
    )
    parser.add_argument(
        "--partition-dir",
        default="population_partitions/area_capped_30sqmi_population_balanced",
    )
    parser.add_argument(
        "--output-dir",
        default="official_calpuff/case_20250623_18z_30sqmi/outputs/matrices_sparse_official_fixed_20250623_18z",
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--start-utc", default="2025-06-23T18:00:00Z")
    parser.add_argument("--value-column", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    out = assemble_sparse_response_matrices(
        case_root=_resolve(args.case_root),
        partition_dir=_resolve(args.partition_dir),
        output_dir=_resolve(args.output_dir),
        hours=args.hours,
        start_utc=args.start_utc,
        value_column=args.value_column,
        allow_missing=args.allow_missing,
    )
    print(out)
    return 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
