from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, save_npz


def assemble_sparse_response_matrices(
    case_root: Path,
    partition_dir: Path,
    output_dir: Path,
    hours: int,
    start_utc: str,
    value_column: str | None = None,
    allow_missing: bool = False,
) -> Path:
    """Assemble CALPOST-adapted receptor responses without dense allocation.

    Each completed source case must contain ``receptors.csv`` with a target
    ``region_id`` column and one response column. The accepted default response
    columns are ``concentration``, ``response_fraction``, and ``value``. This
    adapter deliberately does not guess how a binary CONC.DAT file should be
    decoded; that conversion must be performed by the verified CALPOST/export
    step and recorded in provenance.
    """
    regions = pd.read_csv(partition_dir / "region_area_population_summary.csv")
    region_ids = regions["region_id"].astype(str).sort_values().to_numpy()
    region_index = {region_id: index for index, region_id in enumerate(region_ids)}
    expected_source_ids = set(region_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_root = case_root.resolve()
    partition_dir = partition_dir.resolve()

    hourly_stats: list[dict[str, object]] = []
    missing_cases: list[str] = []
    invalid_cases: list[str] = []
    for hour in range(hours):
        hour_dir = case_root / f"hour_{hour:02d}"
        source_dirs = sorted(hour_dir.glob("source_*")) if hour_dir.exists() else []
        seen_sources: set[str] = set()
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []

        for case_dir in source_dirs:
            source_id = case_dir.name.removeprefix("source_")
            if source_id not in region_index:
                invalid_cases.append(f"hour_{hour:02d}/{case_dir.name}: unknown source region")
                continue
            seen_sources.add(source_id)
            response_path = case_dir / "receptors.csv"
            if not response_path.exists():
                missing_cases.append(f"hour_{hour:02d}/{case_dir.name}/receptors.csv")
                continue
            try:
                response = _read_response(response_path, value_column)
            except (KeyError, ValueError, RuntimeError) as exc:
                invalid_cases.append(f"{response_path}: {exc}")
                continue
            source_col = region_index[source_id]
            for target_id, value in response.items():
                target_index = region_index.get(target_id)
                if target_index is None:
                    invalid_cases.append(f"{response_path}: unknown target region {target_id}")
                    continue
                rows.append(target_index)
                cols.append(source_col)
                values.append(value)

        missing_sources = expected_source_ids - seen_sources
        missing_cases.extend(f"hour_{hour:02d}/source_{source_id}" for source_id in sorted(missing_sources))
        if (missing_cases or invalid_cases) and not allow_missing:
            raise RuntimeError(_format_failures(missing_cases, invalid_cases))

        matrix = csc_matrix(
            (np.asarray(values, dtype=np.float32), (rows, cols)),
            shape=(len(region_ids), len(region_ids)),
            dtype=np.float32,
        )
        # CALPOST can legitimately export exact zeros for candidate receptors;
        # remove stored zeros so nnz and the on-disk sparsity reflect usable
        # coefficients rather than the candidate-manifest envelope.
        matrix.eliminate_zeros()
        if matrix.nnz and matrix.data.min() < 0:
            raise RuntimeError(f"Negative response found in hour {hour:02d}.")
        save_npz(output_dir / f"hour_{hour:02d}.npz", matrix, compressed=True)
        hourly_stats.append(
            {
                "hour_index": hour,
                "time_utc": _hour_labels(start_utc, hours)[hour],
                "shape": [len(region_ids), len(region_ids)],
                "nnz": int(matrix.nnz),
                "zero_source_columns": int(len(region_ids) - matrix.getnnz(axis=0).astype(bool).sum()),
                "value_min": float(matrix.data.min()) if matrix.nnz else 0.0,
                "value_max": float(matrix.data.max()) if matrix.nnz else 0.0,
            }
        )

    metadata_path = output_dir / "transfer_sparse_metadata.npz"
    np.savez_compressed(
        metadata_path,
        region_ids=region_ids,
        hours_utc=np.asarray(_hour_labels(start_utc, hours)),
        nnz_by_hour=np.asarray([row["nnz"] for row in hourly_stats], dtype=np.int64),
    )
    provenance = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "case_root": str(case_root),
        "partition_dir": str(partition_dir),
        "region_count": int(len(region_ids)),
        "hours": hours,
        "value_column": value_column or "auto: concentration, response_fraction, value",
        "response_unit": "g/m3 per lb/h source emission rate",
        "matrix_semantics": "source-rate-to-target-concentration response; not a concentration-state transition",
        "allow_missing": allow_missing,
        "missing_case_count": len(missing_cases),
        "invalid_case_count": len(invalid_cases),
        "hourly_stats": hourly_stats,
        "response_contract": {
            "file": "receptors.csv",
            "required_column": "region_id",
            "response_columns": ["concentration", "response_fraction", "value"],
            "aggregation": "mean by target region_id",
        },
        "warning": (
            "This assembler consumes a verified CALPOST/export adapter contract; it does not decode binary CONC.DAT."
        ),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output_dir


def _read_response(path: Path, value_column: str | None) -> dict[str, float]:
    table = pd.read_csv(path)
    if "region_id" not in table.columns:
        raise RuntimeError("missing required region_id column")
    candidates = [value_column] if value_column else ["concentration", "response_fraction", "value"]
    selected = next((column for column in candidates if column and column in table.columns), None)
    if selected is None:
        raise RuntimeError(f"no response column; expected one of {candidates}")
    table = table[["region_id", selected]].copy()
    table["region_id"] = table["region_id"].astype(str)
    table[selected] = pd.to_numeric(table[selected], errors="coerce")
    if table[selected].isna().any() or not np.isfinite(table[selected].to_numpy(float)).all():
        raise ValueError("response contains non-numeric or non-finite values")
    if (table[selected] < 0).any():
        raise ValueError("response contains negative values")
    grouped = table.groupby("region_id", sort=False)[selected].mean()
    return {region_id: float(value) for region_id, value in grouped.items()}


def _hour_labels(start_utc: str, hours: int) -> list[str]:
    start = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    return [(start + timedelta(hours=hour)).isoformat() for hour in range(hours)]


def _format_failures(missing: list[str], invalid: list[str]) -> str:
    lines = [
        "Official sparse assembly is incomplete; no final matrix was accepted.",
        f"Missing case/output entries: {len(missing)}",
        f"Invalid case/output entries: {len(invalid)}",
    ]
    for item in (missing + invalid)[:12]:
        lines.append(f"  {item}")
    if len(missing) + len(invalid) > 12:
        lines.append("  ... see the case root and rerun with a complete batch")
    lines.append("Use --allow-missing only for an explicitly labelled smoke test.")
    return "\n".join(lines)
