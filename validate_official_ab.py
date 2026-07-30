"""Validate the official CALPUFF A/B concentration-matrix package.

The expected paper contract is:

    c_1 = B0 @ emitted_mass_lb
    c_{h+1} = A_h @ c_h,  h=1,...,23

This validator intentionally checks sparse files without densifying them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)

    root = args.output_root
    contract_path = root / "matrix_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    n_regions = int(contract["region_count"])
    n_generators = int(contract["generator_count"])
    horizon_hours = int(contract.get("horizon_hours", 24))
    expected_hours = [int(hour) for hour in contract.get("a_hour_indices", range(1, horizon_hours))]

    report: dict[str, object] = {
        "output_root": str(root),
        "uses_calpuff": True,
        "not_emulator": True,
        "contract": contract.get("state_equation"),
        "region_count": n_regions,
        "generator_count": n_generators,
        "errors": [],
        "warnings": [],
        "b0": None,
        "a": [],
    }
    errors: list[str] = report["errors"]  # type: ignore[assignment]
    warnings: list[str] = report["warnings"]  # type: ignore[assignment]

    expected_equation = (
        f"c1 = B0 @ emitted_mass_lb; c[h+1] = A[h] @ c[h] "
        f"for h=1..{horizon_hours - 1}"
    )
    if contract.get("state_equation") != expected_equation:
        errors.append("matrix_contract.json has an unexpected state equation")

    b0_path = root / str(contract.get("b0_file", "B0/B0_g_m3_per_lb.npz"))
    if b0_path.exists():
        b0 = load_npz(b0_path).tocsc()
        expected_b0_shape = (n_regions, n_generators)
        if args.allow_partial:
            _check_matrix(b0, (n_regions, b0.shape[1]), "B0", errors, warnings)
            if b0.shape[1] != n_generators:
                warnings.append(f"B0 has {b0.shape[1]} generator columns; contract expects {n_generators} (partial run)")
        else:
            _check_matrix(b0, expected_b0_shape, "B0", errors, warnings)
        report["b0"] = _summary(b0, b0_path)
        _check_provenance(root / "B0" / "provenance.json", "B0", errors)
    elif args.allow_partial:
        warnings.append("B0 file is missing (partial validation)")
    else:
        errors.append(f"missing required file: {b0_path}")

    for hour in expected_hours:
        path = root / "A" / f"hour_{hour:02d}.npz"
        if not path.exists():
            if args.allow_partial:
                warnings.append(f"A hour {hour:02d} is missing (partial validation)")
                continue
            errors.append(f"missing required file: {path}")
            continue
        matrix = load_npz(path).tocsc()
        _check_matrix(matrix, (n_regions, n_regions), f"A[{hour}]", errors, warnings)
        report["a"].append(_summary(matrix, path))  # type: ignore[union-attr]

    if not args.allow_partial:
        if len(report["a"]) != len(expected_hours):  # type: ignore[arg-type]
            errors.append(f"the official package must contain exactly {len(expected_hours)} A matrices")

    # A full run must identify itself as the official route in every matrix
    # provenance file. This blocks accidental promotion of legacy emulator data.
    for label, path in (("B0", root / "B0" / "provenance.json"), ("A", root / "A" / "provenance.json")):
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("uses_calpuff") is not True or payload.get("not_emulator") is not True:
                errors.append(f"{label} provenance does not assert official CALPUFF execution")

    report["ok"] = not errors
    report["warnings"] = warnings
    report["errors"] = errors
    report_path = root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": not errors, "errors": len(errors), "warnings": len(warnings), "report": str(report_path)}))
    return 0 if not errors else 1


def _check_matrix(matrix, shape: tuple[int, int], label: str, errors: list[str], warnings: list[str]) -> None:
    if matrix.shape != shape:
        errors.append(f"{label} shape {matrix.shape} != {shape}")
    if matrix.nnz and not np.isfinite(matrix.data).all():
        errors.append(f"{label} contains non-finite coefficients")
    if matrix.nnz and (matrix.data < 0).any():
        errors.append(f"{label} contains negative coefficients")
    if matrix.nnz == 0:
        warnings.append(f"{label} is entirely zero")
    if matrix.shape[1] and matrix.getnnz(axis=0).min() == 0:
        warnings.append(f"{label} has at least one all-zero source column")


def _summary(matrix, path: Path) -> dict[str, object]:
    column_nnz = matrix.getnnz(axis=0)
    return {
        "file": str(path),
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "max": float(matrix.data.max()) if matrix.nnz else 0.0,
        "min": float(matrix.data.min()) if matrix.nnz else 0.0,
        "all_zero_columns": int((column_nnz == 0).sum()),
    }


def _check_provenance(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing {label} provenance: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
