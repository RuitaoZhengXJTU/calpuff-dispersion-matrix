from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_GRIB_DIR = ROOT / "data" / "raw" / "hrrr_20250623_18z"
DEFAULT_STATIONS = (
    ROOT
    / "official_calpuff"
    / "case_20250623_18z_30sqmi"
    / "met"
    / "stations"
    / "surrogate_surface_stations.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "hrrr_station_met_20250623_18z"
DEFAULT_WGRIB2 = Path(os.environ.get("WGRIB2_EXE", "wgrib2"))

SURFACE_FIELDS = {
    "u10_m_s": "UGRD:10 m above ground",
    "v10_m_s": "VGRD:10 m above ground",
    "temperature_2m_k": "TMP:2 m above ground",
    "relative_humidity_pct": "RH:2 m above ground",
    "pressure_surface_pa": "PRES:surface",
    "pbl_height_m": "HPBL:surface",
    "precip_rate_kg_m2_s": "PRATE:surface",
    "total_cloud_pct": "TCDC:entire atmosphere",
}
PRESSURE_LEVELS = (1000, 925, 850, 700, 500)
HEIGHT_LEVELS = (1000, 850, 700, 500)
UPPER_FIELDS = {
    "height_m": "HGT",
    "temperature_k": "TMP",
    "u_m_s": "UGRD",
    "v_m_s": "VGRD",
}
VALUE_PATTERN = re.compile(r"val=([-+0-9.eE]+|NaN|nan)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample selected HRRR GRIB2 messages at the CALMET station locations."
    )
    parser.add_argument("--grib-dir", type=Path, default=DEFAULT_GRIB_DIR)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date", default="20250623")
    parser.add_argument("--cycle", type=int, default=18)
    parser.add_argument(
        "--start-hour",
        type=int,
        default=0,
        help="first HRRR forecast hour represented by output hour_index 0",
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--wgrib2", type=Path, default=DEFAULT_WGRIB2)
    args = parser.parse_args()
    if not args.wgrib2.exists():
        raise FileNotFoundError(args.wgrib2)
    if not 0 <= args.start_hour <= 48 or not 1 <= args.hours <= 48 or args.start_hour + args.hours > 48:
        raise ValueError("start-hour plus hours must be within HRRR f00--f48")

    stations = _read_stations(args.stations_csv)
    if len(stations) != 9:
        raise ValueError("the current CALMET protocol requires exactly nine surface stations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows: list[dict[str, object]] = []
    upper_rows: list[dict[str, object]] = []
    command_log: list[dict[str, object]] = []

    for output_hour in range(args.hours):
        forecast_hour = args.start_hour + output_hour
        grib_path = args.grib_dir / f"hrrr.t{args.cycle:02d}z.wrfsfcf{forecast_hour:02d}.grib2.selected.grib2"
        if not grib_path.exists():
            raise FileNotFoundError(grib_path)
        values_by_field: dict[str, list[float]] = {}
        for output_name, selector in SURFACE_FIELDS.items():
            values, command = _sample_field(args.wgrib2, grib_path, selector, stations)
            values_by_field[output_name] = values
            command_log.append(
                {
                    "output_hour_index": output_hour,
                    "forecast_hour": forecast_hour,
                    "selector": selector,
                    "command": command,
                }
            )

        timestamp = datetime.strptime(args.date, "%Y%m%d").replace(
            hour=args.cycle, tzinfo=timezone.utc
        ) + timedelta(hours=forecast_hour)
        for index, station in enumerate(stations):
            u = values_by_field["u10_m_s"][index]
            v = values_by_field["v10_m_s"][index]
            speed, direction = _wind_speed_direction(u, v)
            surface_rows.append(
                {
                    "hour_index": output_hour,
                    "time_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "station_index": station["station_index"],
                    "station_id": station["station_id"],
                    "name": station["name"],
                    "lon": station["lon"],
                    "lat": station["lat"],
                    "wind_speed_m_s": speed,
                    "wind_direction_deg_from": direction,
                    "temperature_2m_c": values_by_field["temperature_2m_k"][index] - 273.15,
                    "relative_humidity_2m_pct": values_by_field["relative_humidity_pct"][index],
                    "pressure_surface_hpa": values_by_field["pressure_surface_pa"][index] / 100.0,
                    "boundary_layer_height_m": values_by_field["pbl_height_m"][index],
                    "precip_rate_mm_h": max(values_by_field["precip_rate_kg_m2_s"][index], 0.0) * 3600.0,
                    "total_cloud_pct": values_by_field["total_cloud_pct"][index],
                }
            )

        upper_station = stations[len(stations) // 2]
        upper_level_values: dict[int, dict[str, float]] = {
            pressure_mb: {} for pressure_mb in PRESSURE_LEVELS
        }
        for pressure_mb in PRESSURE_LEVELS:
            for output_name, field in UPPER_FIELDS.items():
                # HRRR CONUS surface/pressure products contain no HGT:925 mb
                # message. Keep the observed 925-mb thermodynamic and wind
                # fields, and derive only that missing geopotential height
                # below from neighboring observed heights.
                if output_name == "height_m" and pressure_mb not in HEIGHT_LEVELS:
                    continue
                selector = f"{field}:{pressure_mb} mb"
                values, command = _sample_field(args.wgrib2, grib_path, selector, [upper_station])
                upper_level_values[pressure_mb][output_name] = values[0]
                command_log.append(
                    {
                        "output_hour_index": output_hour,
                        "forecast_hour": forecast_hour,
                        "selector": selector,
                        "command": command,
                    }
                )
        observed_heights = {
            pressure_mb: values["height_m"]
            for pressure_mb, values in upper_level_values.items()
            if "height_m" in values
        }
        if len(observed_heights) != len(HEIGHT_LEVELS):
            raise RuntimeError(
                f"missing HRRR geopotential heights; observed levels={sorted(observed_heights)}"
            )
        for pressure_mb in PRESSURE_LEVELS:
            level_values = upper_level_values[pressure_mb]
            if "height_m" in level_values:
                height_source = "HRRR HGT pressure-level message"
            else:
                level_values["height_m"] = _interpolate_height_log_pressure(
                    pressure_mb, observed_heights
                )
                height_source = "log-pressure interpolation of HRRR HGT 1000/850 mb"
            upper_rows.append(
                {
                    "hour_index": output_hour,
                    "time_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "station_index": upper_station["station_index"],
                    "station_id": upper_station["station_id"],
                    "pressure_mb": pressure_mb,
                    **level_values,
                    "height_source": height_source,
                }
            )
        print(f"f{forecast_hour:02d}: sampled {len(stations)} surface stations and {len(PRESSURE_LEVELS)} upper levels")

    _write_csv(args.output_dir / "hrrr_surface_station_hourly.csv", surface_rows)
    _write_csv(args.output_dir / "hrrr_upper_air_hourly.csv", upper_rows)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "wgrib2 -lon sampling of selected NOAA HRRR GRIB2 messages",
        "date": args.date,
        "cycle_utc": args.cycle,
        "start_forecast_hour": args.start_hour,
        "hours": args.hours,
        "source_grib_dir": str(args.grib_dir),
        "stations_csv": str(args.stations_csv),
        "station_count": len(stations),
        "upper_station_index": stations[len(stations) // 2]["station_index"],
        "surface_fields": SURFACE_FIELDS,
        "pressure_levels_mb": list(PRESSURE_LEVELS),
        "height_levels_with_direct_HRRR_messages_mb": list(HEIGHT_LEVELS),
        "derived_height_rule": "925 mb HGT is log-pressure interpolated between direct HRRR HGT at 1000 and 850 mb",
        "unit_conversions": {
            "temperature": "K to degC for surface CSV; upper air retained in K",
            "pressure": "Pa to hPa for surface CSV",
            "precip_rate": "kg m-2 s-1 to mm h-1",
            "wind_direction": "meteorological direction from, degrees clockwise from north",
        },
        "wgrib2": str(args.wgrib2),
        "wgrib2_sha256": _sha256(args.wgrib2),
        "commands": command_log,
        "warning": "These are HRRR-derived station fields, not WRF wrfout NetCDF and not direct MMIF input.",
    }
    (args.output_dir / "hrrr_station_met_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(args.output_dir)
    return 0


def _read_stations(path: Path) -> list[dict[str, float | int | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "station_index": int(row["station_index"]),
            "station_id": int(row["station_id"]),
            "name": row["name"],
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
        }
        for row in rows
    ]


def _sample_field(
    wgrib2: Path,
    grib_path: Path,
    selector: str,
    stations: list[dict[str, float | int | str]],
) -> tuple[list[float], list[str]]:
    command = [str(wgrib2), str(grib_path), "-match", f":{selector}:" ]
    for station in stations:
        command.extend(["-lon", str(station["lon"]), str(station["lat"])])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    values = [float(value) for value in VALUE_PATTERN.findall(completed.stdout)]
    if len(values) != len(stations):
        raise RuntimeError(
            f"wgrib2 returned {len(values)} values for {selector}, expected {len(stations)}; output={completed.stdout!r}"
        )
    return values, command


def _wind_speed_direction(u: float, v: float) -> tuple[float, float]:
    speed = math.hypot(u, v)
    if speed < 0.1:
        return 0.1, 0.0
    direction = (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0
    return speed, direction


def _interpolate_height_log_pressure(
    pressure_mb: int, observed_heights: dict[int, float]
) -> float:
    """Interpolate geopotential height against log pressure between HRRR levels."""
    if pressure_mb in observed_heights:
        return float(observed_heights[pressure_mb])
    points = sorted(observed_heights.items())
    for (p1, h1), (p2, h2) in zip(points, points[1:]):
        if min(p1, p2) <= pressure_mb <= max(p1, p2):
            x1 = math.log(float(p1))
            x2 = math.log(float(p2))
            x = math.log(float(pressure_mb))
            return float(h1 + (h2 - h1) * (x - x1) / (x2 - x1))
    raise RuntimeError(
        f"cannot interpolate missing HRRR height at {pressure_mb} mb from {sorted(observed_heights)}"
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
