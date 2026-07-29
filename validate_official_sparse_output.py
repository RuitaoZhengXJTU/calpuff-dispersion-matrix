from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, save_npz


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate official CALPUFF sparse response matrices without densifying them."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--regions", type=int, default=5042)
    parser.add_argument("--rewrite-zero-storage", action="store_true")
    args = parser.parse_args()

    hourly: list[dict[str, object]] = []
    failures: list[str] = []
    for hour in range(args.hours):
        path = args.output_dir / f"hour_{hour:02d}.npz"
        if not path.exists():
            failures.append(f"missing {path.name}")
            continue
        matrix = load_npz(path).tocsc()
        stored_zero_count = int(np.count_nonzero(matrix.data == 0.0))
        if args.rewrite_zero_storage and stored_zero_count:
            matrix.eliminate_zeros()
            save_npz(path, matrix, compressed=True)
            stored_zero_count = 0
        if matrix.shape != (args.regions, args.regions):
            failures.append(f"{path.name}: shape={matrix.shape}")
        if matrix.nnz and (not np.isfinite(matrix.data).all() or matrix.data.min() < 0):
            failures.append(f"{path.name}: invalid or negative coefficient")
        source_mask = np.asarray(matrix.getnnz(axis=0)).ravel().astype(bool)
        source_column_count = int(source_mask.sum())
        # One-hot direction check: the result must equal the selected source
        # column, which documents row=target and column=source.
        source = min(hour, args.regions - 1)
        one_hot = np.zeros(args.regions, dtype=np.float32)
        one_hot[source] = 1.0
        expected = np.asarray(matrix[:, source].toarray()).ravel()
        observed = np.asarray(matrix @ one_hot).ravel()
        direction_error = float(np.max(np.abs(expected - observed))) if expected.size else 0.0
        if direction_error > 1e-10:
            failures.append(f"{path.name}: one-hot direction error={direction_error}")
        hourly.append(
            {
                "hour_index": hour,
                "shape": list(matrix.shape),
                "stored_nnz": int(matrix.nnz),
                "stored_zero_count": stored_zero_count,
                "positive_nnz": int(np.count_nonzero(matrix.data > 0.0)),
                "source_columns_with_response": source_column_count,
                "zero_source_columns": int(args.regions - source_column_count),
                "coefficient_min": float(matrix.data.min()) if matrix.nnz else 0.0,
                "coefficient_max": float(matrix.data.max()) if matrix.nnz else 0.0,
                "one_hot_direction_error": direction_error,
            }
        )

    metadata_path = args.output_dir / "transfer_sparse_metadata.npz"
    if metadata_path.exists():
        metadata = np.load(metadata_path, allow_pickle=True)
        payload = {key: metadata[key] for key in metadata.files if key != "nnz_by_hour"}
        payload["nnz_by_hour"] = np.asarray([row["stored_nnz"] for row in hourly], dtype=np.int64)
        np.savez_compressed(metadata_path, **payload)

    provenance_path = args.output_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    provenance["response_unit"] = "g/m3 per lb/h source emission rate"
    provenance["matrix_semantics"] = (
        "R[h,j,i] is target-region concentration response per 1 lb/h source-region emission; "
        "it is not a concentration-state transition matrix."
    )
    provenance["hourly_stats_validated"] = hourly
    provenance["validation_failures"] = failures
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    report = {
        "ok": not failures,
        "output_dir": str(args.output_dir.resolve()),
        "region_count": args.regions,
        "hours": args.hours,
        "rewrite_zero_storage": args.rewrite_zero_storage,
        "hourly": hourly,
        "failures": failures,
    }
    (args.output_dir / "official_matrix_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"ok": report["ok"], "hours": len(hourly), "failures": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
