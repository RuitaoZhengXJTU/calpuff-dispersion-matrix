"""Stable command-line interface for the formal CALPUFF matrix workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import calmet, conversion, hrrr, matrices, preparation, validation, weather
from .config import load_case_config, mapping_value, project_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = PROJECT_ROOT / "configs" / "dc_md_va_20250623.yaml"


def _case_path(value: str | None) -> Path:
    return Path(value).resolve() if value else DEFAULT_CASE


def _case_args(argv: list[str], flag: str = "--case") -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(flag, type=Path, default=DEFAULT_CASE)
    known, remainder = parser.parse_known_args(argv)
    return known.__dict__[flag[2:].replace("-", "_")], remainder


def _path(case: dict[str, object], name: str) -> Path:
    value = project_path(PROJECT_ROOT, mapping_value(case, "paths").get(name))
    if value is None:
        raise ValueError(f"case paths.{name} is required")
    return value


def _fetch_hrrr(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Fetch selected HRRR fields configured by a case YAML.")
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--window", choices=("main", "spinup", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--legacy-direct", action="store_true", help=argparse.SUPPRESS)
    args, remainder = parser.parse_known_args(argv)
    if args.legacy_direct:
        return hrrr.main(remainder)
    if remainder:
        parser.error(f"unrecognized arguments: {' '.join(remainder)}")
    case = load_case_config(args.case)
    hrrr_config = mapping_value(case, "hrrr")
    windows = (("main", "main_window", "hrrr_main_raw_dir", "hrrr_main_manifest"), ("spinup", "spinup_window", "hrrr_spinup_raw_dir", "hrrr_spinup_manifest"))
    for name, window_name, raw_name, manifest_name in windows:
        if args.window not in {name, "all"}:
            continue
        window = mapping_value(hrrr_config, window_name)
        command = [
            "--date", str(window["date"]), "--cycle", str(window["cycle_utc"]),
            "--start-hour", str(window["start_hour"]), "--hours", str(window["hours"]),
            "--output-dir", str(_path(case, raw_name)), "--manifest-path", str(_path(case, manifest_name)),
        ]
        if args.force:
            command.append("--force")
        if args.dry_run:
            command.append("--dry-run")
        hrrr.main(command)
    return 0


def _build_weather(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build configured subregional HRRR weather table.")
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--wgrib2", type=Path, default=None)
    parser.add_argument("--legacy-direct", action="store_true", help=argparse.SUPPRESS)
    args, remainder = parser.parse_known_args(argv)
    if args.legacy_direct:
        return weather.main(remainder)
    if remainder:
        parser.error(f"unrecognized arguments: {' '.join(remainder)}")
    case = load_case_config(args.case)
    window = mapping_value(mapping_value(case, "hrrr"), "main_window")
    wgrib2 = args.wgrib2
    if wgrib2 is None:
        from .runtime import _resolve_executable
        wgrib2 = _resolve_executable(None, "WGRIB2_EXE", "wgrib2")
    return weather.main([
        "--grib-dir", str(_path(case, "hrrr_main_raw_dir")),
        "--subregions", str(_path(case, "subregions")),
        "--output", str(_path(case, "weather")),
        "--date", str(window["date"]), "--cycle", str(window["cycle_utc"]),
        "--start-hour", str(window["start_hour"]),
        "--hours", str(int(mapping_value(case, "time")["hours"]) + 1),
        "--wgrib2", str(wgrib2),
    ])


def _prepare(argv: list[str]) -> int:
    case, remainder = _case_args(argv)
    return preparation.main(["--case", str(case), *remainder])


def _run(argv: list[str]) -> int:
    case, remainder = _case_args(argv)
    return matrices.main(["--case", str(case), *remainder])


def _convert(argv: list[str]) -> int:
    return conversion.main(argv)


def _validate(argv: list[str]) -> int:
    return validation.main(argv)


def _build_calmet(argv: list[str]) -> int:
    return calmet.main(argv)


def _verify_hrrr(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Verify HRRR selected files against case manifests without downloading.")
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--window", choices=("main", "spinup", "all"), default="all")
    args = parser.parse_args(argv)
    case = load_case_config(args.case)
    reports = []
    names = (("main", "hrrr_main_manifest"), ("spinup", "hrrr_spinup_manifest"))
    for name, manifest_name in names:
        if args.window in {name, "all"}:
            reports.append(hrrr.verify_manifest(_path(case, manifest_name), PROJECT_ROOT))
    report = {"ok": all(bool(item["ok"]) for item in reports), "reports": reports}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


COMMANDS = {
    "fetch-hrrr": _fetch_hrrr,
    "prepare": _prepare,
    "build-weather": _build_weather,
    "build-calmet": _build_calmet,
    "run": _run,
    "convert-units": _convert,
    "validate": _validate,
    "verify-hrrr": _verify_hrrr,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calpuff-matrix", description=__doc__)
    parser.add_argument("command", nargs="?", choices=tuple(COMMANDS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    if parsed.command is None:
        parser.print_help()
        return 0
    return COMMANDS[parsed.command](list(parsed.arguments))


if __name__ == "__main__":
    raise SystemExit(main())
