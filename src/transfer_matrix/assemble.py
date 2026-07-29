from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CaseConfig


def assemble_matrices(config: CaseConfig) -> Path:
    region_ids = _region_ids(config)
    region_index = {region_id: idx for idx, region_id in enumerate(region_ids)}
    hours = config.hours
    n = config.target_regions
    matrix = np.zeros((hours, n, n), dtype=float)

    for hour in range(hours):
        hour_dir = config.case_root() / f"hour_{hour:02d}"
        if not hour_dir.exists():
            continue
        # Keep this aligned with build_calpuff_cases: source IDs may be
        # ``r000`` or namespaced IDs such as ``area_pop_000000``.
        for case_dir in sorted(hour_dir.glob("source_*")):
            source_id = case_dir.name.replace("source_", "")
            if source_id not in region_index:
                continue
            output = case_dir / "receptors.csv"
            if not output.exists():
                continue
            responses = _read_case_response(output)
            source_col = region_index[source_id]
            for target_id, value in responses.items():
                if target_id in region_index:
                    matrix[hour, region_index[target_id], source_col] = value

    normalized = _normalize_columns(matrix)
    out_npz = config.output_path("matrix_npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        T=matrix,
        T_normalized=normalized,
        region_ids=np.asarray(region_ids),
        hours_utc=np.asarray(_hours_utc(config)),
    )
    _write_hour_csvs(config, matrix, region_ids)
    _write_provenance(config, matrix)
    return out_npz


def _region_ids(config: CaseConfig) -> list[str]:
    receptors = config.output_path("receptors_csv")
    if receptors.exists():
        table = pd.read_csv(receptors)
        return sorted(table["region_id"].unique().tolist())
    return [f"r{i:03d}" for i in range(config.target_regions)]


def _read_case_response(path: Path) -> dict[str, float]:
    table = pd.read_csv(path)
    if "region_id" not in table.columns:
        raise RuntimeError(f"{path} must include region_id")
    value_col = None
    for candidate in ("response_fraction", "concentration", "value"):
        if candidate in table.columns:
            value_col = candidate
            break
    if value_col is None:
        raise RuntimeError(f"{path} must include response_fraction, concentration, or value")
    grouped = table.groupby("region_id", as_index=True)[value_col].mean()
    return {str(region_id): float(value) for region_id, value in grouped.items()}


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(matrix)
    col_sums = matrix.sum(axis=1, keepdims=True)
    np.divide(matrix, col_sums, out=normalized, where=col_sums > 0)
    return normalized


def _write_hour_csvs(config: CaseConfig, matrix: np.ndarray, region_ids: list[str]) -> None:
    matrices_dir = config.output_path("matrices_dir")
    matrices_dir.mkdir(parents=True, exist_ok=True)
    for hour in range(matrix.shape[0]):
        frame = pd.DataFrame(matrix[hour], index=region_ids, columns=region_ids)
        frame.index.name = "target_region"
        frame.to_csv(matrices_dir / f"hour_{hour:02d}.csv")


def _hours_utc(config: CaseConfig) -> list[str]:
    start = datetime.fromisoformat(config.data["time"]["start_utc"].replace("Z", "+00:00"))
    return [(start + timedelta(hours=hour)).isoformat() for hour in range(config.hours)]


def _write_provenance(config: CaseConfig, matrix: np.ndarray) -> None:
    census_url = config.data.get("domain", {}).get("census_state_shapefile_url")
    provenance = {
        "case_id": config.case_id,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "case_config": str(config.path),
        "matrix_shape": list(matrix.shape),
        "nonzero_entries": int(np.count_nonzero(matrix)),
        "sources": {
            "calpuff": "https://calpuff.org/",
            "hrrr": "https://registry.opendata.aws/noaa-hrrr-pds/",
            "census_tiger": census_url,
        },
    }
    path = config.output_path("provenance_json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
