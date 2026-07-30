"""Compose HRRR station sampling, CALMET control generation, and CALMET execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import calmet_inputs, station_data
from .config import load_case_config, mapping_value, project_path
from .runtime import _assert_success, _resolve_executable, run_control_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--wgrib2", type=Path, default=None)
    parser.add_argument("--calmet-exe", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    case = load_case_config(args.case)
    paths = mapping_value(case, "paths")
    hrrr = mapping_value(case, "hrrr")
    main_window = mapping_value(hrrr, "main_window")
    spinup_window = mapping_value(hrrr, "spinup_window")
    required = (
        "hrrr_main_raw_dir", "hrrr_spinup_raw_dir", "stations",
        "station_met_main", "station_met_spinup", "calmet_dir",
    )
    missing = [name for name in required if paths.get(name) is None]
    if missing:
        raise ValueError(f"case paths missing CALMET inputs: {', '.join(missing)}")
    if not main_window or not spinup_window:
        raise ValueError("case hrrr.main_window and hrrr.spinup_window are required")

    def path(name: str) -> Path:
        value = project_path(PROJECT_ROOT, paths.get(name))
        assert value is not None
        return value

    wgrib2 = args.wgrib2 or _resolve_executable(None, "WGRIB2_EXE", "wgrib2")
    calmet_dir = path("calmet_dir")
    common = ["--stations-csv", str(path("stations")), "--wgrib2", str(wgrib2)]
    station_commands = [
        [
            "--grib-dir", str(path("hrrr_spinup_raw_dir")), "--output-dir", str(path("station_met_spinup")),
            "--date", str(spinup_window["date"]), "--cycle", str(spinup_window["cycle_utc"]),
            "--start-hour", str(spinup_window["start_hour"]), "--hours", str(spinup_window["hours"]), *common,
        ],
        [
            "--grib-dir", str(path("hrrr_main_raw_dir")), "--output-dir", str(path("station_met_main")),
            "--date", str(main_window["date"]), "--cycle", str(main_window["cycle_utc"]),
            "--start-hour", str(main_window["start_hour"]), "--hours", str(int(case.get("time", {}).get("hours", 24))), *common,
        ],
    ]
    calmet_command = [
        "--surface-csv", str(path("station_met_main") / "hrrr_surface_station_hourly.csv"),
        "--upper-csv", str(path("station_met_main") / "hrrr_upper_air_hourly.csv"),
        "--spinup-surface-csv", str(path("station_met_spinup") / "hrrr_surface_station_hourly.csv"),
        "--spinup-upper-csv", str(path("station_met_spinup") / "hrrr_upper_air_hourly.csv"),
        "--stations-csv", str(path("stations")),
        "--terrain-grib", str(path("hrrr_main_raw_dir") / f"hrrr.t{int(main_window['cycle_utc']):02d}z.wrfsfcf{int(main_window['start_hour']):02d}.grib2.selected.grib2"),
        "--wgrib2", str(wgrib2), "--output-dir", str(calmet_dir),
        "--hours", str(int(case.get("time", {}).get("hours", 24))),
        "--case-config", str(args.case),
    ]
    if args.dry_run:
        print("CALMET dry run: HRRR station extraction and CALMET input generation were not executed.")
        print("station commands:", station_commands)
        print("CALMET input command:", calmet_command)
        return 0

    for command in station_commands:
        station_data.main(command)
    calmet_inputs.main(calmet_command)
    calmet_exe = args.calmet_exe or _resolve_executable(None, "CALMET_EXE", "calmet.exe")
    control = calmet_dir / "CALMET.INP"
    log = calmet_dir / "CALMET_RUN.log"
    run_control_file(calmet_exe, control, calmet_dir, log, args.timeout_sec)
    _assert_success(calmet_dir / "CALMET.DAT", log)
    print(calmet_dir / "CALMET.DAT")
    return 0
