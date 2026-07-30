from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description="Write provenance for the completed official sparse CALPUFF batch.")
    parser.add_argument("--case-root", type=Path, default=Path("official_calpuff/case_20250623_18z_30sqmi"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    case_root = _resolve(args.case_root)
    output_dir = _resolve(args.output_dir) if args.output_dir else case_root / "outputs"
    matrix_dir = output_dir / "matrices_sparse_official_fixed_20250623_18z"
    validation = json.loads((matrix_dir / "official_matrix_validation.json").read_text(encoding="utf-8"))
    calmet_provenance = json.loads(
        (case_root / "met" / "calmet_hrrr" / "hrrr_calmet_provenance.json").read_text(encoding="utf-8")
    )
    paths = {
        "calmet_exe": Path(os.environ.get("CALMET_EXE", "calmet_v6.5.0.exe")),
        "calpuff_exe": Path(os.environ.get("CALPUFF_EXE", "calpuff_v7.2.1.exe")),
        "calpost_exe": Path(os.environ.get("CALPOST_EXE", "calpost_v7.1.0.exe")),
        "wgrib2_exe": Path(os.environ.get("WGRIB2_EXE", "wgrib2")),
        "calmet_dat": case_root / "met/calmet_hrrr/CALMET.DAT",
        "candidate_manifest": case_root / "inputs/sparse_candidate_manifest/candidate_targets_by_hour_source.npz",
        "matrix_region_index": case_root / "inputs/matrix_region_index.csv",
        "sources_16_per_region": case_root / "inputs/sources_16_per_region.csv",
        "receptors_9_per_region": case_root / "inputs/receptors_9_per_region.csv",
        "official_initial_response_g_m3_per_lb_h": case_root / "runs/official_initial_response_352/official_initial_local_response_g_m3_per_lb_h.npz",
        "official_initial_response_ppb_per_lb_h": case_root / "runs/official_initial_response_352/official_initial_local_response_ppb_per_lb_h.npz",
    }
    hashes = {name: sha256(path) for name, path in paths.items() if path.exists()}
    hourly = validation["hourly"]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": "case_20250623_18z_30sqmi",
        "target_start_utc": "2025-06-23T18:00:00Z",
        "target_end_utc_exclusive": "2025-06-24T18:00:00Z",
        "region_count": 5042,
        "hours": 24,
        "partition": "area-capped 30 sq mi, population-balanced DC/MD/VA partition",
        "meteorology": calmet_provenance,
        "software": {
            "calmet": str(paths["calmet_exe"]),
            "calpuff": str(paths["calpuff_exe"]),
            "calpost": str(paths["calpost_exe"]),
            "wgrib2": str(paths["wgrib2_exe"]),
        },
        "hashes_sha256": hashes,
        "source_setup": {
            "sources_per_region": 16,
            "release_height_m": 15.0,
            "pollutant": "passive NO2-equivalent tracer",
            "chemistry": "off",
            "emission_rate_lb_per_hour": 1.0,
            "receptors_per_region": 9,
        },
        "official_response_package": str(matrix_dir.relative_to(ROOT)).replace("\\", "/"),
        "response_unit": "g/m3 per lb/h source emission rate",
        "matrix_semantics": "R[h,j,i] maps source-region emission rate to target-region average receptor concentration; it is not a concentration-state transition.",
        "hourly_validation_summary": {
            "ok": bool(validation["ok"]),
            "failures": validation["failures"],
            "min_positive_nnz": min(row["positive_nnz"] for row in hourly),
            "max_positive_nnz": max(row["positive_nnz"] for row in hourly),
            "total_positive_nnz": sum(row["positive_nnz"] for row in hourly),
            "max_coefficient": max(row["coefficient_max"] for row in hourly),
            "max_zero_source_columns": max(row["zero_source_columns"] for row in hourly),
        },
        "emulator_and_b0_note": "The state-transition G matrices and 5042x352 B0 packages remain separate, protocol-labelled outputs; they are not silently replaced by this official source-rate response package.",
        "official_initial_response_package": "runs/official_initial_response_352/; direct one-hour host-region CALPUFF responses, not instantaneous B0",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "provenance_official_20250623_18z_30sqmi.json"
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(path)
    return 0


def _resolve(path: Path | None) -> Path:
    if path is None:
        return ROOT
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
