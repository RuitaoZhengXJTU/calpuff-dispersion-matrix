from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CaseConfig


def validate_outputs(config: CaseConfig) -> dict[str, object]:
    matrix_path = config.output_path("matrix_npz")
    if not matrix_path.exists():
        raise RuntimeError(f"Missing matrix file: {matrix_path}")

    payload = np.load(matrix_path, allow_pickle=True)
    matrix = payload["T"]
    normalized = payload["T_normalized"]
    region_ids = payload["region_ids"]
    expected_shape = (config.hours, config.target_regions, config.target_regions)

    subregions_path = config.data.get("outputs", {}).get("subregions_geojson")
    checks: dict[str, object] = {
        "matrix_shape": list(matrix.shape),
        "expected_shape": list(expected_shape),
        "shape_ok": tuple(matrix.shape) == expected_shape,
        "normalized_shape_ok": normalized.shape == matrix.shape,
        "nonnegative": bool(np.all(matrix >= 0)),
        "region_count": int(len(region_ids)),
        "direction_check": _direction_check(matrix),
        "subregions_geojson_exists": config.resolve(subregions_path).exists() if subregions_path else None,
    }
    checks["ok"] = all(
        bool(checks[key])
        for key in ["shape_ok", "normalized_shape_ok", "nonnegative", "direction_check"]
    )

    report_path = config.output_path("diagnostics_dir") / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not checks["ok"]:
        raise RuntimeError(f"Validation failed. See {report_path}")
    return checks


def _direction_check(matrix: np.ndarray) -> bool:
    if matrix.size == 0:
        return False
    vector = np.zeros(matrix.shape[2])
    vector[0] = 1.0
    result = matrix[0] @ vector
    return np.allclose(result, matrix[0, :, 0])
