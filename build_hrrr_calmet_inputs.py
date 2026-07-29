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

from pyproj import Transformer


ROOT = Path(__file__).resolve().parent
LCC = "+proj=lcc +lat_1=33 +lat_2=39.5 +lat_0=37 +lon_0=-77.5 +datum=WGS84 +units=m +no_defs"
DEFAULT_SURFACE = ROOT / "data" / "processed" / "hrrr_station_met_20250623_18z" / "hrrr_surface_station_hourly.csv"
DEFAULT_UPPER = ROOT / "data" / "processed" / "hrrr_station_met_20250623_18z" / "hrrr_upper_air_hourly.csv"
DEFAULT_STATIONS = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi" / "met" / "stations" / "surrogate_surface_stations.csv"
DEFAULT_TERRAIN = ROOT / "data" / "raw" / "hrrr_20250623_18z" / "hrrr.t18z.wrfsfcf00.grib2.selected.grib2"
DEFAULT_OUTPUT = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi" / "met" / "calmet_hrrr"
DEFAULT_WGRIB2 = Path(os.environ.get("WGRIB2_EXE", "wgrib2"))
VALUE_PATTERN = re.compile(r"val=([-+0-9.eE]+|NaN|nan)")

NX = 79
NY = 38
NZ = 10
DGRID_KM = 10.0
XORIG_KM = -560.0
YORIG_KM = -60.0
ZFACE = (0.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1200.0, 2000.0, 3000.0, 4000.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build CALMET formatted inputs from sampled NOAA HRRR station fields."
    )
    parser.add_argument("--surface-csv", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--upper-csv", type=Path, default=DEFAULT_UPPER)
    parser.add_argument(
        "--spinup-surface-csv",
        type=Path,
        default=None,
        help="optional earlier HRRR station surface CSV to prepend by absolute UTC time",
    )
    parser.add_argument(
        "--spinup-upper-csv",
        type=Path,
        default=None,
        help="optional earlier HRRR station upper-air CSV to prepend by absolute UTC time",
    )
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--terrain-grib", type=Path, default=DEFAULT_TERRAIN)
    parser.add_argument("--wgrib2", type=Path, default=DEFAULT_WGRIB2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--itest", type=int, choices=(1, 2), default=2)
    parser.add_argument("--irtype", type=int, choices=(0, 1), default=1)
    parser.add_argument("--land-use-code", type=int, default=40)
    parser.add_argument("--terrain-batch-size", type=int, default=50)
    args = parser.parse_args()

    for path in (args.surface_csv, args.upper_csv, args.stations_csv, args.terrain_grib, args.wgrib2):
        if not path.exists():
            raise FileNotFoundError(path)
    if (args.spinup_surface_csv is None) != (args.spinup_upper_csv is None):
        raise ValueError("spinup surface and upper CSVs must be supplied together")
    for path in (args.spinup_surface_csv, args.spinup_upper_csv):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)
    if args.hours < 1:
        raise ValueError("hours must be positive")
    if args.terrain_batch_size < 1:
        raise ValueError("terrain-batch-size must be positive")
    if not 1 <= args.land_use_code <= 99:
        raise ValueError("land-use-code must be a CALMET land-use category")

    target_surface = _read_csv(args.surface_csv)
    target_upper = _read_csv(args.upper_csv)
    if args.spinup_surface_csv is not None:
        surface = _merge_time_rows(_read_csv(args.spinup_surface_csv), target_surface, "surface")
        upper = _merge_time_rows(_read_csv(args.spinup_upper_csv), target_upper, "upper")
    else:
        surface = _reindex_time_rows(target_surface)
        upper = _reindex_time_rows(target_upper)
    stations = _read_stations(args.stations_csv)
    total_hours = len({row["time_utc"] for row in surface})
    _validate_inputs(surface, upper, stations, total_hours)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start = _parse_utc(surface[0]["time_utc"])
    end = start + timedelta(hours=total_hours)
    terrain_points = _grid_points()
    terrain_values = _sample_wgrib2(
        args.wgrib2.resolve(),
        args.terrain_grib.resolve(),
        "HGT:surface",
        terrain_points,
        args.terrain_batch_size,
    )
    if any(not math.isfinite(value) for value in terrain_values):
        raise ValueError("HRRR terrain sampling returned NaN or non-finite values")
    station_points = [(float(row["lon"]), float(row["lat"])) for row in stations]
    station_elevations = _sample_wgrib2(
        args.wgrib2.resolve(),
        args.terrain_grib.resolve(),
        "HGT:surface",
        station_points,
        args.terrain_batch_size,
    )

    _write_geo(output_dir / "GEO.DAT", terrain_values, args.land_use_code)
    _write_surface(output_dir / "SURF.DAT", surface, stations, total_hours)
    _write_upper_air(output_dir / "UP1.DAT", upper, stations[len(stations) // 2], station_elevations[len(stations) // 2], start, total_hours)
    _write_control(
        output_dir / "CALMET.INP",
        start + timedelta(hours=1),
        total_hours,
        args.itest,
        args.irtype,
        len(stations),
    )

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "HRRR-derived CALMET input; not WRF/MMIF and not yet a formal paper result",
        "method": "sample selected HRRR GRIB2 fields at nine stations; sample HRRR HGT surface at CALMET grid-cell centers; write CALMET v6.5 formatted inputs",
        "case_start_utc": start.isoformat().replace("+00:00", "Z"),
        "case_end_utc_exclusive": end.isoformat().replace("+00:00", "Z"),
        "calmet_control_start_utc": (start + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "hours": total_hours,
        "target_hours": args.hours,
        "spinup_hours": total_hours - args.hours,
        "target_start_utc": _parse_utc(target_surface[0]["time_utc"]).isoformat().replace("+00:00", "Z"),
        "surface_csv": str(args.surface_csv.resolve()),
        "upper_csv": str(args.upper_csv.resolve()),
        "stations_csv": str(args.stations_csv.resolve()),
        "terrain_grib": str(args.terrain_grib.resolve()),
        "wgrib2": str(args.wgrib2.resolve()),
        "input_sha256": {
            "surface_csv": _sha256(args.surface_csv),
            "upper_csv": _sha256(args.upper_csv),
            "stations_csv": _sha256(args.stations_csv),
            "terrain_grib": _sha256(args.terrain_grib),
            "wgrib2": _sha256(args.wgrib2),
        },
        "grid": {
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "dgrid_km": DGRID_KM,
            "xorig_km": XORIG_KM,
            "yorig_km": YORIG_KM,
            "projection": "LCC WGS-84 lat_1=33 lat_2=39.5 lat_0=37 lon_0=-77.5",
        },
        "terrain": {
            "field": "HGT:surface",
            "sample_method": "wgrib2 -lon at cell centers",
            "min_m": min(terrain_values),
            "max_m": max(terrain_values),
            "mean_m": sum(terrain_values) / len(terrain_values),
            "land_use_source": "uniform code supplied by protocol, not HRRR vegetation classification",
            "land_use_code": args.land_use_code,
        },
        "upper_air": {
            "station_count": 1,
            "pressure_levels_mb": sorted({int(row["pressure_mb"]) for row in upper}, reverse=True),
            "missing_height_rule": "925 mb height is log-pressure interpolated between direct HRRR 1000 and 850 mb HGT",
            "boundary_hours": "UP1.DAT persists first/last sampled HRRR profiles plus one trailing profile required by the CALMET version-1 reader",
        },
        "calmet": {
            "version": "6.5.0",
            "irtype": args.irtype,
            "itest": args.itest,
            "surface_stations": len(stations),
            "upper_stations": 1,
            "precipitation_stations": 0,
            "chemistry_or_deposition": "not a CALMET setting; downstream CALPUFF smoke controls set chemistry/deposition off",
        },
        "warnings": [
            "HRRR GRIB2 is not relabeled as WRF wrfout and is not passed to MMIF.",
            "The nine-station sampling and uniform land-use category are protocol approximations.",
            "Run CALMET and inspect CALMET.LST before using CALMET.DAT in CALPUFF.",
        ],
    }
    if args.spinup_surface_csv is not None:
        provenance["spinup_surface_csv"] = str(args.spinup_surface_csv.resolve())
        provenance["spinup_upper_csv"] = str(args.spinup_upper_csv.resolve())
        provenance["input_sha256"]["spinup_surface_csv"] = _sha256(args.spinup_surface_csv)
        provenance["input_sha256"]["spinup_upper_csv"] = _sha256(args.spinup_upper_csv)
    (output_dir / "hrrr_calmet_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(output_dir)
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reindex_time_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize hour_index to absolute UTC ordering within one CSV."""
    times = sorted({_parse_utc(row["time_utc"]) for row in rows})
    if not times:
        raise ValueError("meteorology CSV is empty")
    _validate_contiguous_times(times)
    index_by_time = {time: index for index, time in enumerate(times)}
    output = []
    for row in rows:
        copied = dict(row)
        copied["hour_index"] = str(index_by_time[_parse_utc(row["time_utc"])])
        output.append(copied)
    return sorted(output, key=lambda row: (_parse_utc(row["time_utc"]), int(row.get("station_index", 0))))


def _merge_time_rows(
    spinup_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    kind: str,
) -> list[dict[str, str]]:
    """Merge spin-up and target records without relying on local hour_index."""
    if kind not in {"surface", "upper"}:
        raise ValueError(kind)
    combined = list(spinup_rows) + list(target_rows)
    times = sorted({_parse_utc(row["time_utc"]) for row in combined})
    _validate_contiguous_times(times)
    index_by_time = {time: index for index, time in enumerate(times)}
    output = []
    for row in combined:
        copied = dict(row)
        copied["hour_index"] = str(index_by_time[_parse_utc(row["time_utc"])])
        output.append(copied)
    return sorted(output, key=lambda row: (_parse_utc(row["time_utc"]), int(row.get("station_index", 0))))


def _validate_contiguous_times(times: list[datetime]) -> None:
    for previous, current in zip(times, times[1:]):
        if current - previous != timedelta(hours=1):
            raise ValueError(
                f"meteorology records are not hourly-contiguous: {previous.isoformat()} -> {current.isoformat()}"
            )


def _read_stations(path: Path) -> list[dict[str, float | int | str]]:
    rows = _read_csv(path)
    return sorted(
        [
            {
                "station_index": int(row["station_index"]),
                "station_id": int(row["station_id"]),
                "name": row["name"],
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
            }
            for row in rows
        ],
        key=lambda row: int(row["station_index"]),
    )


def _validate_inputs(
    surface: list[dict[str, str]],
    upper: list[dict[str, str]],
    stations: list[dict[str, float | int | str]],
    hours: int,
) -> None:
    if len(stations) != 9:
        raise ValueError(f"expected nine surface stations, found {len(stations)}")
    if len(surface) != hours * len(stations):
        raise ValueError(f"surface row count {len(surface)} != {hours}*{len(stations)}")
    if len(upper) != hours * 5:
        raise ValueError(f"upper row count {len(upper)} != {hours}*5")
    surface_hours = sorted({int(row["hour_index"]) for row in surface})
    upper_hours = sorted({int(row["hour_index"]) for row in upper})
    if surface_hours != list(range(hours)) or upper_hours != list(range(hours)):
        raise ValueError("surface or upper records do not cover consecutive hour_index values")
    for hour in range(hours):
        surface_rows = [row for row in surface if int(row["hour_index"]) == hour]
        if len({int(row["station_index"]) for row in surface_rows}) != len(stations):
            raise ValueError(f"surface stations are incomplete at hour {hour}")
        upper_rows = sorted(
            [row for row in upper if int(row["hour_index"]) == hour],
            key=lambda row: int(row["pressure_mb"]),
            reverse=True,
        )
        if [int(row["pressure_mb"]) for row in upper_rows] != [1000, 925, 850, 700, 500]:
            raise ValueError(f"upper pressure levels are incomplete at hour {hour}")
        heights = [float(row["height_m"]) for row in upper_rows]
        if any(not math.isfinite(value) for value in heights) or any(a >= b for a, b in zip(heights, heights[1:])):
            raise ValueError(f"upper heights are not strictly increasing with altitude at hour {hour}")
    for row in surface:
        for field in ("wind_speed_m_s", "wind_direction_deg_from", "temperature_2m_c", "relative_humidity_2m_pct", "pressure_surface_hpa"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"non-finite surface {field} at hour {row['hour_index']}")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _grid_points() -> list[tuple[float, float]]:
    inverse = Transformer.from_crs(LCC, "EPSG:4326", always_xy=True)
    points: list[tuple[float, float]] = []
    for j in range(NY):
        for i in range(NX):
            x = (XORIG_KM + (i + 0.5) * DGRID_KM) * 1000.0
            y = (YORIG_KM + (j + 0.5) * DGRID_KM) * 1000.0
            lon, lat = inverse.transform(x, y)
            points.append((float(lon), float(lat)))
    return points


def _sample_wgrib2(
    wgrib2: Path,
    grib: Path,
    selector: str,
    points: list[tuple[float, float]],
    batch_size: int,
) -> list[float]:
    values: list[float] = []
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        command = [str(wgrib2), str(grib), "-match", f":{selector}:"]
        for lon, lat in batch:
            command.extend(["-lon", f"{lon:.8f}", f"{lat:.8f}"])
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        batch_values = [float(value) for value in VALUE_PATTERN.findall(completed.stdout)]
        if len(batch_values) != len(batch):
            raise RuntimeError(
                f"wgrib2 returned {len(batch_values)} values for {selector}, expected {len(batch)}; output={completed.stdout!r}"
            )
        values.extend(batch_values)
    return values


def _write_geo(path: Path, terrain_values: list[float], land_use_code: int) -> None:
    if len(terrain_values) != NX * NY:
        raise ValueError("terrain grid size does not match CALMET grid")
    lines = [
        f"{'GEO.DAT':<16}{'6.5.0':<16}{'HRRR-DERIVED TERRAIN; PROVISIONAL LAND USE'}",
        "0",
        "LCC",
        f"{'N37.0':<16}{'W77.5':<16}{'N33.0':<16}{'N39.5':<16}",
        "0.0 0.0",
        f"{'WGS-84':<8}{'HRRR':<12}",
        f"{NX:8d}{NY:8d}{XORIG_KM:12.3f}{YORIG_KM:12.3f}{DGRID_KM:12.3f}{DGRID_KM:12.3f}",
        "KM",
        "0",
    ]
    lines.extend(" ".join([str(land_use_code)] * NX) for _ in range(NY))
    # CALMET's GEO.DAT layout used by the installed distribution expects a
    # scalar roughness length before the elevation grid. The value is kept
    # explicit because land-use-dependent roughness is not available here.
    lines.append("1.0")
    for row in range(NY):
        start = row * NX
        lines.append(" ".join(f"{value:.2f}" for value in terrain_values[start : start + NX]))
    lines.extend(["0", "0", "0", "0", "0", "0"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_surface(
    path: Path,
    surface: list[dict[str, str]],
    stations: list[dict[str, float | int | str]],
    hours: int,
) -> None:
    first = _parse_utc(surface[0]["time_utc"])
    # CALMET version-1 formatted surface files use hour-ending records. The
    # first record at 19Z represents the 18Z--19Z model hour.
    first_record = first + timedelta(hours=1)
    last_record = first + timedelta(hours=hours)
    lines = [
        f"{'SURF.DAT':<16}{'1.0':<16}{'HRRR-DERIVED SURFACE DATA'}",
        "0",
        "LL",
        f"{'WGS-84':<8}{'HRRR':<10}",
        "KM",
        f"{first_record.year:04d} {first_record.timetuple().tm_yday:03d} {first_record.hour:02d} "
        f"{last_record.year:04d} {last_record.timetuple().tm_yday:03d} {last_record.hour:02d} 0 {len(stations)}",
    ]
    for station in stations:
        lines.append(
            f"{int(station['station_id'])} {str(station['name'])[:4]:<4} "
            f"N{float(station['lat']):.4f} W{abs(float(station['lon'])):.4f} 10.0"
        )
    by_hour_station = {
        (int(row["hour_index"]), int(row["station_index"])): row for row in surface
    }
    for hour in range(hours):
        record_time = first + timedelta(hours=hour + 1)
        values: list[str] = []
        for station in stations:
            row = by_hour_station[(hour, int(station["station_index"]))]
            wind = max(float(row["wind_speed_m_s"]), 0.1)
            direction = float(row["wind_direction_deg_from"]) % 360.0
            cloud = int(max(0, min(10, round(float(row["total_cloud_pct"]) / 10.0))))
            rh = int(round(float(row["relative_humidity_2m_pct"])))
            rh = max(1, min(100, rh))
            precip_code = 1 if float(row["precip_rate_mm_h"]) > 0.01 else 0
            values.extend(
                [
                    f"{wind:.3f}",
                    f"{direction:.2f}",
                    "0",  # no cloud-ceiling field was requested; retain CALMET-valid explicit value
                    str(cloud),
                    f"{float(row['temperature_2m_c']) + 273.15:.2f}",
                    str(rh),
                    f"{float(row['pressure_surface_hpa']):.2f}",
                    str(precip_code),
                ]
            )
        lines.append(
            f"{record_time.year:04d} {record_time.timetuple().tm_yday:03d} {record_time.hour:02d} "
            + " ".join(values)
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_upper_air(
    path: Path,
    upper: list[dict[str, str]],
    station: dict[str, float | int | str],
    station_elevation: float,
    start: datetime,
    hours: int,
) -> None:
    # Include one boundary record on each side and one trailing record. The
    # CALMET version-1 reader asks for the next sounding after the final model
    # hour, so the trailing persisted profile prevents an end-of-file read.
    header_start = start - timedelta(hours=1)
    header_end = start + timedelta(hours=hours + 1)
    lines = [
        f"{'UP.DAT':<16}{'1.0':<16}{'HRRR-DERIVED UPPER-AIR DATA'}",
        "0",
        "LL",
        f"{'WGS-84':<8}{'HRRR':<10}",
        "KM",
        f"{header_start.year:5d}{header_start.timetuple().tm_yday:5d}{header_start.hour:5d}"
        f"{header_end.year:5d}{header_end.timetuple().tm_yday:5d}{header_end.hour:5d}"
        f"{500.0:5.0f}{3:5d}{1:5d}",
        "     F    F    F    F",
        f"20001 U001 N{float(station['lat']):.4f} W{abs(float(station['lon'])):.4f} {int(round(station_elevation))}",
    ]
    by_hour = {
        int(hour): sorted(
            [row for row in upper if int(row["hour_index"]) == hour],
            key=lambda row: int(row["pressure_mb"]),
            reverse=True,
        )
        for hour in range(hours)
    }
    for record_hour in range(-1, hours + 2):
        source_hour = min(max(record_hour, 0), hours - 1)
        record_time = start + timedelta(hours=record_hour)
        rows = by_hour[source_hour]
        lines.append(
            f"{'':9}{20001:8d}{'':5}"
            f"{record_time.year % 100:02d}{record_time.month:02d}{record_time.day:02d}{record_time.hour:02d}"
            f"{'':35}{len(rows):5d}"
        )
        records: list[str] = []
        for row in rows:
            temp_c = float(row["temperature_k"]) - 273.15
            speed = math.hypot(float(row["u_m_s"]), float(row["v_m_s"]))
            direction = _wind_direction_from_uv(float(row["u_m_s"]), float(row["v_m_s"]))
            records.append(
                f"   {float(row['pressure_mb']):6.1f} {float(row['height_m']):5.0f} "
                f"{temp_c:5.1f} {direction:3.0f} {max(speed, 0.1):3.0f}"
            )
        lines.extend("".join(records[i : i + 4]) for i in range(0, len(records), 4))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _wind_direction_from_uv(u: float, v: float) -> float:
    speed = math.hypot(u, v)
    if speed < 0.1:
        return 0.0
    return (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0


def _write_control(
    path: Path,
    start: datetime,
    hours: int,
    itest: int,
    irtype: int,
    station_count: int,
) -> None:
    ns = ",".join(["0"] * NZ)
    nintr = ",".join(["9"] * NZ)
    zface = ",".join(f"{value:g}." for value in ZFACE)
    lines = [
        "HRRR-DERIVED CALMET CONTROL; LAND USE IS PROVISIONAL",
        "HRRR GRIB2 fields sampled to formatted SURF.DAT/UP1.DAT; not MMIF input",
        "CALMET MODEL CONTROL FILE",
        "INPUT GROUP: 0 -- Input and Output File Names",
        "! GEODAT=GEO.DAT !",
        "! SRFDAT=SURF.DAT !",
        "! METLST=CALMET.LST !",
        "! METDAT=CALMET.DAT !",
        "! NUSTA=1 !",
        "! NOWSTA=0 !",
        "!END!",
        "! UPDAT=UP1.DAT ! !END!",
        "! DIADAT=DIAG.DAT !",
        "! PRGDAT=PROG.DAT !",
        "!END!",
        "INPUT GROUP: 1 -- General run control parameters",
        f"! IBYR={start.year:04d} !",
        f"! IBMO={start.month} !",
        f"! IBDY={start.day} !",
        f"! IBHR={start.hour} !",
        "! IBTZ=0 !",
        f"! IRLG={hours} !",
        f"! IRTYPE={irtype} !",
        "! LCALGRD=F !",
        "! MREG=0 !",
        f"! ITEST={itest} !",
        "!END!",
        "INPUT GROUP: 2 -- Grid control parameters",
        f"! NX={NX} !",
        f"! NY={NY} !",
        f"! DGRIDKM={DGRID_KM:.3f} !",
        f"! XORIGKM={XORIG_KM:.3f} !",
        f"! YORIGKM={YORIG_KM:.3f} !",
        "! PMAP=LCC !",
        "! DATUM=WGS-84 !",
        "! FEAST=0.0 !",
        "! FNORTH=0.0 !",
        "! IUTMZN=0 !",
        "! UTMHEM=N !",
        "! XLAT1=N33.0 !",
        "! XLAT2=N39.5 !",
        "! RLON0=W77.5 !",
        "! RLAT0=N37.0 !",
        f"! NZ={NZ} !",
        f"! ZFACE={zface} !",
        "!END!",
        "INPUT GROUP: 3 -- Output Options",
        "! LSAVE=T !",
        "! IFORMO=1 !",
        "! LPRINT=F !",
        "! IPRINF=1 !",
        "! LDB=F !",
        "! IOUTD=0 !",
        "!END!",
        "INPUT GROUP: 4 -- Meteorological data options",
        "! NOOBS=0 !",
        f"! NSSTA={station_count} !",
        "! NPSTA=0 !",
        "! ICLOUD=0 !",
        "! IFORMS=2 !",
        "! IFORMP=2 !",
        "!END!",
        "INPUT GROUP: 5 -- Wind Field Options and Parameters",
        "! IWFCOD=1 !",
        "! IFRADJ=0 !",
        "! IKINE=0 !",
        "! IOBR=0 !",
        "! ISLOPE=0 !",
        "! IEXTRP=4 !",
        "! ICALM=0 !",
        "! RMIN2=-1.0 !",
        "! IPROG=0 !",
        "! ISTEPPGS=3600 !",
        "! LVARY=F !",
        "! RMAX1=500.0 !",
        "! RMAX2=500.0 !",
        "! RMAX3=500.0 !",
        "! RMIN=0.1 !",
        "! TERRAD=10.0 !",
        "! R1=100.0 !",
        "! R2=500.0 !",
        "! RPROG=54.0 !",
        "! DIVLIM=5.0E-6 !",
        "! NITER=50 !",
        f"! NSMTH={ns} !",
        f"! NINTR2={nintr} !",
        "! CRITFN=1.0 !",
        "! ALPHA=0.1 !",
        "! NBAR=0 !",
        "! IDIOPT1=0 !",
        "! ISURFT=1 !",
        "! IDIOPT2=0 !",
        "! IUPT=1 !",
        "! ZUPT=200.0 !",
        "! IDIOPT3=0 !",
        "! IUPWND=-1 !",
        "! ZUPWND=1.0,500.0 !",
        "! IDIOPT4=0 !",
        "! IDIOPT5=0 !",
        "! LLBREZE=F !",
        "! NBOX=0 !",
        "!END!",
        "INPUT GROUP: 6 -- Mixing Height, Temperature and Precipitation Parameters",
        "! CONSTB=1.41 !",
        "! CONSTE=0.15 !",
        "! CONSTN=2400.0 !",
        "! CONSTW=0.16 !",
        "! FCORIOL=1.0E-4 !",
        "! IAVEZI=0 !",
        "! MNMDAV=1 !",
        "! HAFANG=30.0 !",
        "! ILEVZI=1 !",
        "! DPTMIN=0.001 !",
        "! DZZI=200.0 !",
        "! ZIMIN=50.0 !",
        "! ZIMAX=3000.0 !",
        "! ZIMINW=50.0 !",
        "! ZIMAXW=3000.0 !",
        "! IRAD=1 !",
        "! TRADKM=500.0 !",
        "! NUMTS=5 !",
        "! IAVET=0 !",
        "! TGDEFB=-0.0098 !",
        "! TGDEFA=-0.0045 !",
        "! JWAT1=999 !",
        "! JWAT2=999 !",
        "! NFLAGP=2 !",
        "! SIGMAP=100.0 !",
        "! CUTP=0.01 !",
        "!END!",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
