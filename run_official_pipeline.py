from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CASE_ROOT = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi"


def _default_tool(env_name: str, command_name: str, local_fallback: Path | None = None) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured)
    if local_fallback and local_fallback.exists():
        return local_fallback
    discovered = shutil.which(command_name)
    return Path(discovered) if discovered else Path(command_name)


DEFAULT_TOOLS = {
    "mmif": _default_tool("MMIF_EXE", "mmif_4.1.1.exe"),
    "calmet": _default_tool("CALMET_EXE", "calmet_v6.5.0.exe"),
    "calpuff": _default_tool("CALPUFF_EXE", "calpuff_v7.2.1.exe"),
    "calpost": _default_tool(
        "CALPOST_EXE",
        "calpost_v7.1.0.exe",
        ROOT / "data" / "raw" / "official_examples" / "calpost_v7.1.0_L141010" / "CALPOST_v7.1.0_L141010" / "calpost_v7.1.0.exe",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the gated MMIF/CALMET/CALPUFF/CALPOST pipeline."
    )
    parser.add_argument("--case-root", default=str(DEFAULT_CASE_ROOT))
    parser.add_argument(
        "--stage",
        choices=("mmif", "calmet", "calpuff", "calpost", "all"),
        default="all",
    )
    parser.add_argument("--mmif-control", default=None)
    parser.add_argument("--calmet-control", default=None)
    parser.add_argument("--calpuff-control", default=None)
    parser.add_argument("--calpost-control", default=None)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the preflight manifest and commands without starting a binary.",
    )
    args = parser.parse_args()

    case_root = _resolve(args.case_root)
    controls = {
        "mmif": _resolve(args.mmif_control) if args.mmif_control else case_root / "met" / "mmif" / "mmif_20250623_18z.inp",
        "calmet": _resolve(args.calmet_control) if args.calmet_control else case_root / "met" / "calmet" / "calmet.inp",
        "calpuff": _resolve(args.calpuff_control) if args.calpuff_control else case_root / "runs" / "smoke" / "calpuff.inp",
        "calpost": _resolve(args.calpost_control) if args.calpost_control else case_root / "runs" / "smoke" / "calpost.inp",
    }
    order = ["mmif", "calmet", "calpuff", "calpost"]
    stages = order if args.stage == "all" else [args.stage]
    manifest_path = case_root / "outputs" / "formal_pipeline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "created_utc": _now(),
        "case_root": str(case_root),
        "requested_stage": args.stage,
        "dry_run": args.dry_run,
        "stages": [],
    }
    overall_ok = True
    for stage in stages:
        stage_report = _preflight_stage(stage, case_root, controls[stage], DEFAULT_TOOLS[stage])
        stage_report["command"] = [str(DEFAULT_TOOLS[stage]), str(controls[stage])]
        if stage_report["preflight_ok"] and not args.dry_run:
            stage_report.update(_execute_stage(stage, case_root, controls[stage], DEFAULT_TOOLS[stage], args.timeout_sec))
        elif stage_report["preflight_ok"]:
            stage_report["status"] = "dry_run_ready"
            stage_report["return_code"] = None
        else:
            stage_report["status"] = "blocked_preflight"
            stage_report["return_code"] = None
        report["stages"].append(stage_report)
        if not stage_report["preflight_ok"] or stage_report.get("return_code") not in (None, 0):
            overall_ok = False
            if args.stage == "all":
                break

    report["ok"] = overall_ok
    report["scientific_result_available"] = bool(
        overall_ok and not args.dry_run and args.stage in ("calpuff", "calpost", "all")
    )
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(manifest_path)
    return 0 if overall_ok else 1


def _preflight_stage(stage: str, case_root: Path, control: Path, executable: Path) -> dict[str, object]:
    errors: list[str] = []
    if not executable.exists():
        errors.append(f"executable does not exist: {executable}")
    if not control.exists():
        errors.append(f"control file does not exist: {control}")
        return {
            "stage": stage,
            "executable": str(executable),
            "control": str(control),
            "preflight_ok": False,
            "errors": errors,
        }
    text = control.read_text(encoding="utf-8", errors="ignore")
    upper = text.upper()
    for marker in ("PLACEHOLDER", "WRF_INPUTS_REQUIRED", "NOT READY TO RUN", "SKELETON"):
        if marker in upper:
            errors.append(f"control contains non-runnable marker: {marker}")
    if stage == "mmif":
        input_paths = _mmif_input_paths(text)
        if not input_paths:
            errors.append("MMIF control contains no active INPUT line")
        for input_path in input_paths:
            resolved = _resolve_from(control.parent, input_path)
            if not resolved.exists():
                errors.append(f"MMIF INPUT file does not exist: {resolved}")
    elif stage == "calmet":
        if "!END!" not in upper:
            errors.append("CALMET control does not contain an !END! terminator")
    elif stage in ("calpuff", "calpost"):
        if "!END!" not in upper:
            errors.append(f"{stage.upper()} control does not contain an !END! terminator")
    return {
        "stage": stage,
        "executable": str(executable),
        "executable_sha256": _sha256(executable) if executable.exists() else None,
        "control": str(control),
        "control_sha256": _sha256(control),
        "preflight_ok": not errors,
        "errors": errors,
    }


def _execute_stage(stage: str, case_root: Path, control: Path, executable: Path, timeout_sec: int) -> dict[str, object]:
    stage_dir = control.parent
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / f"{stage}_run.log"
    started = _now()
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            completed = subprocess.run(
                [str(executable), str(control)],
                cwd=stage_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "started_utc": started,
            "finished_utc": _now(),
            "return_code": completed.returncode,
            "log": str(log_path),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "started_utc": started,
            "finished_utc": _now(),
            "return_code": None,
            "log": str(log_path),
            "errors": [f"stage exceeded timeout of {timeout_sec} seconds"],
        }


def _mmif_input_paths(text: str) -> list[str]:
    paths = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in ";#!":
            continue
        match = re.match(r"^INPUT\s+(.+?)\s*$", stripped, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            paths.append(value)
    return paths


def _resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
