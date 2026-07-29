from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import numpy as np

from official_case_builder import (
    DEFAULT_START,
    CalpuffCaseFactory,
    load_csv_rows,
)
from parse_calpost_tseries import parse_calpost_tseries


ROOT = Path(__file__).resolve().parent
CASE_ROOT = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi"
DEFAULT_OUTPUT = CASE_ROOT / "runs" / "official_sparse_candidate"
DEFAULT_CANDIDATES = CASE_ROOT / "inputs" / "sparse_candidate_manifest" / "candidate_targets_by_hour_source.npz"
DEFAULT_SOURCES = CASE_ROOT / "inputs" / "sources_16_per_region.csv"
DEFAULT_RECEPTORS = CASE_ROOT / "inputs" / "receptors_9_per_region.csv"
DEFAULT_SEED = CASE_ROOT / "templates" / "CALPUFF_7.0_seed_from_distribution.INP"
DEFAULT_CALPOST_TEMPLATE = ROOT / "data" / "raw" / "official_examples" / "calpost_v7.1.0_L141010" / "CALPOST_v7.1.0_L141010" / "calpost.inp"
DEFAULT_CALPOST_EXE = ROOT / "data" / "raw" / "official_examples" / "calpost_v7.1.0_L141010" / "CALPOST_v7.1.0_L141010" / "calpost_v7.1.0.exe"
HRRR_CALMET = r"..\..\..\..\met\calmet_hrrr\CALMET.DAT"
ERROR_RE = re.compile(r"(?i)error in subr|halted in|fatal|endfile")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official CALPUFF/CALPOST candidate sparse source-hour cases."
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--start-source", type=int, default=0)
    parser.add_argument("--source-count", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--receptors", type=Path, default=DEFAULT_RECEPTORS)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--calpost-template", type=Path, default=DEFAULT_CALPOST_TEMPLATE)
    parser.add_argument("--calmet-dat", default=HRRR_CALMET)
    parser.add_argument("--start-utc", default="2025-06-23T18:00:00Z")
    parser.add_argument("--calpuff-exe", default=None, help="CALPUFF executable; defaults to CALPUFF_EXE or PATH")
    parser.add_argument("--calpost-exe", default=None, help="CALPOST executable; defaults to CALPOST_EXE or the local official example")
    parser.add_argument("--emission-lb-per-hour", type=float, default=1.0)
    parser.add_argument("--retain-case-files", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.hours <= 0 or args.start_hour < 0 or args.start_hour + args.hours > 24:
        raise ValueError("start-hour + hours must be within [0, 24]")
    if args.max_workers <= 0:
        raise ValueError("max-workers must be positive")
    start_utc = _parse_start_utc(args.start_utc)
    calpuff_exe = _resolve_executable(args.calpuff_exe, "CALPUFF_EXE", "calpuff_v7.2.1.exe")
    calpost_exe = _resolve_executable(args.calpost_exe, "CALPOST_EXE", "calpost_v7.1.0.exe", DEFAULT_CALPOST_EXE)
    for path in (args.candidate_manifest, args.sources, args.receptors, args.seed, args.calpost_template):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.dry_run:
        for path in (calpuff_exe, calpost_exe):
            if not path.exists():
                raise FileNotFoundError(path)

    sources = load_csv_rows(args.sources)
    receptors = load_csv_rows(args.receptors)
    by_region: dict[int, list[dict[str, str]]] = {}
    for row in receptors:
        by_region.setdefault(int(row["matrix_index"]), []).append(row)
    # The checked-in candidate builder stores region_ids as an object array.
    # This file is a local generated manifest, not untrusted user input.
    candidate_data = np.load(args.candidate_manifest, allow_pickle=True)
    indptr = candidate_data["indptr"]
    target_indices = candidate_data["target_region_indices"]
    region_ids = candidate_data["region_ids"].astype(str)
    region_count = len(region_ids)
    available_sources = int(region_count)
    if args.start_source >= available_sources:
        raise ValueError(f"start-source must be below {available_sources}")
    source_count = args.source_count
    if source_count is None:
        source_count = available_sources - args.start_source
    if source_count <= 0 or args.start_source + source_count > available_sources:
        raise ValueError("source-count exceeds the region index range")

    factory = CalpuffCaseFactory(
        seed_path=args.seed,
        calpost_template=args.calpost_template,
        source_rows=sources,
        calmet_dat=args.calmet_dat,
        start_utc=start_utc,
    )
    tasks = [
        (hour, source_index)
        for hour in range(args.start_hour, args.start_hour + args.hours)
        for source_index in range(args.start_source, args.start_source + source_count)
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_config.json").write_text(
        json.dumps(
            {
                "hours": args.hours,
                "start_hour": args.start_hour,
                "start_source": args.start_source,
                "source_count": source_count,
                "task_count": len(tasks),
                "max_workers": args.max_workers,
                "candidate_manifest": str(args.candidate_manifest.resolve()),
                "region_count": region_count,
                "emission_lb_per_hour": args.emission_lb_per_hour,
                "response_unit": "g/m3 per lb emitted during one hour",
                "not_final_transition_matrix": True,
                "dry_run": args.dry_run,
                "resume": args.resume,
                "start_utc": _iso(start_utc),
                "calpuff_exe": str(calpuff_exe),
                "calpost_exe": str(calpost_exe),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    progress_lock = Lock()
    completed = 0
    results: list[dict[str, object]] = []

    def emit_progress(result: dict[str, object], count: int) -> None:
        status = result.get("status")
        should_emit = (
            status not in {"completed", "skipped_completed"}
            or count <= 5
            or count == len(tasks)
            or count % 100 == 0
        )
        if should_emit:
            print(json.dumps({"completed": count, "total": len(tasks), **result}))

    def run_task(task: tuple[int, int]) -> dict[str, object]:
        return _run_one(
            task=task,
            factory=factory,
            indptr=indptr,
            target_indices=target_indices,
            region_ids=region_ids,
            receptor_by_region=by_region,
            output_root=args.output_root,
            emission_lb_per_hour=args.emission_lb_per_hour,
            timeout_sec=args.timeout_sec,
            retain_case_files=args.retain_case_files,
            dry_run=args.dry_run,
            resume=args.resume,
            calpuff_exe=calpuff_exe,
            calpost_exe=calpost_exe,
            start_utc=start_utc,
        )

    if args.max_workers == 1:
        for task in tasks:
            result = run_task(task)
            results.append(result)
            completed += 1
            emit_progress(result, completed)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                with progress_lock:
                    completed += 1
                    emit_progress(result, completed)

    accepted_statuses = (
        {"completed", "dry_run_ready"}
        if args.dry_run
        else {"completed", "skipped_completed"}
    )
    failures = [result for result in results if result["status"] not in accepted_statuses]
    report = {
        "task_count": len(tasks),
        "completed_count": len(results) - len(failures),
        "failure_count": len(failures),
        "ok": not failures,
        "results": results,
    }
    (args.output_root / "run_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0 if not failures else 1


def _run_one(
    task: tuple[int, int],
    factory: CalpuffCaseFactory,
    indptr: np.ndarray,
    target_indices: np.ndarray,
    region_ids: np.ndarray,
    receptor_by_region: dict[int, list[dict[str, str]]],
    output_root: Path,
    emission_lb_per_hour: float,
    timeout_sec: int,
    retain_case_files: bool,
    dry_run: bool,
    resume: bool,
    calpuff_exe: Path,
    calpost_exe: Path,
    start_utc: datetime,
) -> dict[str, object]:
    hour, source_index = task
    case_dir = (
        output_root / f"hour_{hour:02d}" / f"source_{region_ids[source_index]}"
    ).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    case_status = case_dir / "run_status.json"
    record = {
        "hour_index": hour,
        "source_matrix_index": source_index,
        "source_region_id": str(region_ids[source_index]),
        "case_dir": str(case_dir.resolve()),
    }
    try:
        if resume and case_status.exists() and (case_dir / "receptors.csv").exists():
            previous = json.loads(case_status.read_text(encoding="utf-8"))
            if (
                previous.get("status") == "completed"
                and previous.get("calpuff_return_code") == 0
                and previous.get("calpost_return_code") == 0
            ):
                previous["status"] = "skipped_completed"
                return previous
        record_index = hour * len(region_ids) + source_index
        start, end = int(indptr[record_index]), int(indptr[record_index + 1])
        selected_targets = np.unique(target_indices[start:end].astype(int))
        selected_receptors: list[dict[str, str]] = []
        for target in selected_targets:
            selected_receptors.extend(receptor_by_region[int(target)])
        if not selected_receptors:
            raise RuntimeError("candidate manifest selected zero receptors")

        control = factory.build_calpuff(
            output_dir=case_dir,
            source_region_index=source_index,
            hour_index=hour,
            receptor_rows=selected_receptors,
            emission_lb_per_hour=emission_lb_per_hour,
        )
        calpost_control = factory.build_calpost(case_dir, hour)
        record.update(
            {
                "candidate_target_count": int(len(selected_targets)),
                "receptor_count": int(len(selected_receptors)),
                "calpuff_control_sha256": _sha256(control),
                "calpost_control_sha256": _sha256(calpost_control),
            }
        )
        if dry_run:
            record["status"] = "dry_run_ready"
            case_status.write_text(json.dumps(record, indent=2), encoding="utf-8")
            return record

        calpuff_log = case_dir / "CALPUFF_RUN.log"
        calpuff_code = _run_binary(calpuff_exe, control, case_dir, calpuff_log, timeout_sec)
        record["calpuff_return_code"] = calpuff_code
        _assert_success(case_dir / "CALPUFF.CON", calpuff_log)

        calpost_log = case_dir / "CALPOST_7.1.0_RUN.log"
        calpost_code = _run_binary(calpost_exe, calpost_control, case_dir, calpost_log, timeout_sec)
        record["calpost_return_code"] = calpost_code
        tseries = case_dir / "TSERIES_NO2_1HR_CONC.DAT"
        _assert_success(tseries, calpost_log)

        response = case_dir / "receptors.csv"
        parse_calpost_tseries(
            input_path=tseries,
            receptor_manifest=case_dir / "receptor_manifest.csv",
            output_path=response,
            start_utc=_iso(start_utc + timedelta(hours=hour)),
            value_unit="g/m3",
        )
        record["response_rows"] = _count_csv_rows(response)
        record["response_sha256"] = _sha256(response)
        record["status"] = "completed"
        case_status.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if not retain_case_files:
            for name in ("CALPUFF.CON", "CALPUFF.LST", "CALPOST.LST", "CALPUFF.INP", "CALPOST.INP", "TSERIES_NO2_1HR_CONC.DAT"):
                path = case_dir / name
                if path.exists():
                    path.unlink()
        return record
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        case_status.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record


def _run_binary(executable: Path, control: Path, cwd: Path, log_path: Path, timeout_sec: int) -> int:
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(
            [str(executable), control.name],
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
    return int(completed.returncode)


def _assert_success(output: Path, log_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if ERROR_RE.search(text):
        raise RuntimeError(f"binary log contains fatal/error marker: {log_path}")
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"expected non-empty output missing: {output}")


def _count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _iso(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse_start_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("start-utc must include a timezone")
    return parsed.astimezone(timezone.utc)


def _resolve_executable(
    explicit: str | None,
    env_name: str,
    command_name: str,
    local_fallback: Path | None = None,
) -> Path:
    value = explicit or os.environ.get(env_name)
    if value:
        return Path(value)
    if local_fallback and local_fallback.exists():
        return local_fallback
    discovered = shutil.which(command_name)
    return Path(discovered) if discovered else Path(command_name)


if __name__ == "__main__":
    raise SystemExit(main())
