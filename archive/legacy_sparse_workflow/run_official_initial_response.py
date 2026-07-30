from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csc_matrix, save_npz

from official_case_builder import (
    DEFAULT_START,
    CalpuffCaseFactory,
    load_csv_rows,
)
from parse_calpost_tseries import parse_calpost_tseries
from run_official_sparse_matrix import (
    _assert_success,
    _count_csv_rows,
    _parse_start_utc,
    _resolve_executable,
    _run_binary,
    _sha256,
)


ROOT = Path(__file__).resolve().parent
CASE_ROOT = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi"
DEFAULT_CALPOST_EXE = ROOT / "data/raw/official_examples/calpost_v7.1.0_L141010/CALPOST_v7.1.0_L141010/calpost_v7.1.0.exe"
DEFAULT_SEED = CASE_ROOT / "templates/CALPUFF_7.0_seed_from_distribution.INP"
DEFAULT_CALPOST_TEMPLATE = ROOT / "data/raw/official_examples/calpost_v7.1.0_L141010/CALPOST_v7.1.0_L141010/calpost.inp"
# Initial-response cases live one directory shallower than the source-hour
# cases: generator -> output root -> runs -> case root.
CALMET_DAT = r"..\..\..\met\calmet_hrrr\CALMET.DAT"
PPB_PER_GM3_NO2 = 1_000_000.0 * 24.45 / 46.01


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official CALPUFF one-hour local responses for the 352 generators.")
    parser.add_argument("--case-root", type=Path, default=CASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--data-centers", type=Path, default=None)
    parser.add_argument("--receptors", type=Path, default=None)
    parser.add_argument("--region-index", type=Path, default=None)
    parser.add_argument("--seed", type=Path, default=None)
    parser.add_argument("--calpost-template", type=Path, default=None)
    parser.add_argument("--calmet-dat", default=CALMET_DAT)
    parser.add_argument("--start-utc", default="2025-06-23T18:00:00Z")
    parser.add_argument("--calpuff-exe", default=None)
    parser.add_argument("--calpost-exe", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    case_root = args.case_root if args.case_root.is_absolute() else ROOT / args.case_root
    output_root = args.output_root or case_root / "runs/official_initial_response_352"
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    data_centers = args.data_centers or ROOT / "data/data_centers_example.csv"
    receptors_path = args.receptors or case_root / "inputs/receptors_9_per_region.csv"
    region_index_path = args.region_index or case_root / "inputs/matrix_region_index.csv"
    seed = args.seed or case_root / "templates/CALPUFF_7.0_seed_from_distribution.INP"
    calpost_template = args.calpost_template or DEFAULT_CALPOST_TEMPLATE
    start_utc = _parse_start_utc(args.start_utc)
    calpuff_exe = _resolve_executable(args.calpuff_exe, "CALPUFF_EXE", "calpuff_v7.2.1.exe")
    calpost_exe = _resolve_executable(args.calpost_exe, "CALPOST_EXE", "calpost_v7.1.0.exe", DEFAULT_CALPOST_EXE)

    inventory = pd.read_csv(data_centers)
    required = {"generator_id", "facility_id", "site_no", "region_id", "lon", "lat", "stack_height"}
    missing = required - set(inventory.columns)
    if missing:
        raise RuntimeError(f"data-center inventory lacks columns: {sorted(missing)}")
    inventory = inventory.reset_index(drop=True)
    receptors = load_csv_rows(receptors_path)
    receptor_by_region: dict[str, list[dict[str, str]]] = {}
    for row in receptors:
        receptor_by_region.setdefault(str(row["region_id"]), []).append(row)
    to_projected = Transformer.from_crs(
        "EPSG:4326",
        "+proj=lcc +lat_1=33 +lat_2=39.5 +lat_0=37 +lon_0=-77.5 +datum=WGS84 +units=m +no_defs",
        always_xy=True,
    )
    source_rows: list[dict[str, str]] = []
    generator_meta: list[dict[str, object]] = []
    for generator_index, row in inventory.iterrows():
        x_m, y_m = to_projected.transform(float(row["lon"]), float(row["lat"]))
        generator_meta.append(
            {
                "generator_matrix_index": int(generator_index),
                "generator_id": str(row["generator_id"]),
                "facility_id": str(row["facility_id"]),
                "site_no": str(row["site_no"]),
                "region_id": str(row["region_id"]),
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "x_m": float(x_m),
                "y_m": float(y_m),
            }
        )
        for source_index in range(16):
            source_rows.append(
                {
                    "source_id": f"{row['generator_id']}_s{source_index:02d}",
                    "matrix_index": str(generator_index),
                    "x_m": str(x_m),
                    "y_m": str(y_m),
                    "release_fraction": str(1.0 / 16.0),
                }
            )
    factory = CalpuffCaseFactory(seed, calpost_template, source_rows, args.calmet_dat, start_utc=start_utc)

    def task(generator_index: int) -> dict[str, object]:
        meta = generator_meta[generator_index]
        case_dir = output_root / f"generator_{generator_index:03d}_{meta['generator_id']}"
        status_path = case_dir / "run_status.json"
        if args.resume and status_path.exists() and (case_dir / "receptors.csv").exists():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous.get("status") == "completed" and previous.get("calpuff_return_code") == 0 and previous.get("calpost_return_code") == 0:
                previous["status"] = "skipped_completed"
                return previous
        target_receptors = receptor_by_region.get(str(meta["region_id"]), [])
        if not target_receptors:
            raise RuntimeError(f"no receptors for host region {meta['region_id']}")
        case_dir.mkdir(parents=True, exist_ok=True)
        record = {**meta, "case_dir": str(case_dir.resolve()), "receptor_count": len(target_receptors)}
        try:
            control = factory.build_calpuff(case_dir, generator_index, 0, target_receptors, emission_lb_per_hour=1.0)
            calpost_control = factory.build_calpost(case_dir, 0)
            record["calpuff_control_sha256"] = _sha256(control)
            record["calpost_control_sha256"] = _sha256(calpost_control)
            calpuff_log = case_dir / "CALPUFF_RUN.log"
            calpuff_code = _run_binary(calpuff_exe, control, case_dir, calpuff_log, args.timeout_sec)
            record["calpuff_return_code"] = calpuff_code
            _assert_success(case_dir / "CALPUFF.CON", calpuff_log)
            calpost_log = case_dir / "CALPOST_7.1.0_RUN.log"
            calpost_code = _run_binary(calpost_exe, calpost_control, case_dir, calpost_log, args.timeout_sec)
            record["calpost_return_code"] = calpost_code
            tseries = case_dir / "TSERIES_NO2_1HR_CONC.DAT"
            _assert_success(tseries, calpost_log)
            response = case_dir / "receptors.csv"
            parse_calpost_tseries(tseries, case_dir / "receptor_manifest.csv", response, start_utc=start_utc.isoformat().replace("+00:00", "Z"), value_unit="g/m3")
            record["response_rows"] = _count_csv_rows(response)
            record["response_sha256"] = _sha256(response)
            record["status"] = "completed"
            for name in ("CALPUFF.CON", "CALPUFF.LST", "CALPOST.LST", "CALPUFF.INP", "CALPOST.INP", "TSERIES_NO2_1HR_CONC.DAT"):
                path = case_dir / name
                if path.exists():
                    path.unlink()
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
        status_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(task, i) for i in range(len(generator_meta))]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: int(row["generator_matrix_index"]))
    failures = [row for row in results if row.get("status") not in {"completed", "skipped_completed"}]
    columns = pd.DataFrame(generator_meta)
    columns.to_csv(output_root / "generator_columns.csv", index=False)

    region_ids = pd.read_csv(region_index_path)["region_id"].astype(str).to_numpy()
    region_index = {region_id: i for i, region_id in enumerate(region_ids)}
    rows: list[int] = []
    cols: list[int] = []
    values_g_m3_per_lb_h: list[float] = []
    for row in results:
        response_path = Path(str(row["case_dir"])) / "receptors.csv"
        if not response_path.exists():
            continue
        table = pd.read_csv(response_path)
        value_column = "concentration" if "concentration" in table.columns else "value"
        value = float(pd.to_numeric(table[value_column], errors="coerce").mean())
        rows.append(region_index[str(row["region_id"])])
        cols.append(int(row["generator_matrix_index"]))
        values_g_m3_per_lb_h.append(value)
    response_matrix = csc_matrix(
        (np.asarray(values_g_m3_per_lb_h, dtype=np.float32), (rows, cols)),
        shape=(len(region_ids), len(generator_meta)),
    )
    response_matrix.eliminate_zeros()
    save_npz(output_root / "official_initial_local_response_g_m3_per_lb_h.npz", response_matrix, compressed=True)
    save_npz(output_root / "official_initial_local_response_ppb_per_lb_h.npz", response_matrix.multiply(PPB_PER_GM3_NO2).astype(np.float32), compressed=True)
    provenance = {
        "case_id": case_root.name,
        "target_start_utc": start_utc.isoformat().replace("+00:00", "Z"),
        "target_end_utc_exclusive": (start_utc + pd.Timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "generator_count": len(generator_meta),
        "region_count": len(region_ids),
        "source_setup": "16 colocated equal-weight volume sources per data center, 15 m release height, 1 lb/h passive NO2-equivalent tracer",
        "receptor_setup": "nine receptors in the generator host region; arithmetic mean",
        "response_unit": "g/m3 per lb/h and ppb per lb/h",
        "semantic_warning": "This is a direct one-hour local CALPUFF response, not an instantaneous paper-defined B0. Use the protocol B0 package for the initial uniformly mixed state unless this time alignment is explicitly adopted.",
        "output_files": ["official_initial_local_response_g_m3_per_lb_h.npz", "official_initial_local_response_ppb_per_lb_h.npz"],
        "completed_count": len(results) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output_root / "run_report.json").write_text(json.dumps({"results": results, "failures": failures}, indent=2), encoding="utf-8")
    print(json.dumps({"ok": not failures, "generators": len(generator_meta), "responses": response_matrix.nnz, "failures": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
