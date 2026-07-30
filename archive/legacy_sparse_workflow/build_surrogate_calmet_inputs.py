from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parent
LCC = "+proj=lcc +lat_1=33 +lat_2=39.5 +lat_0=37 +lon_0=-77.5 +datum=WGS84 +units=m +no_defs"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build explicitly labelled surrogate CALMET inputs from the fallback weather CSV."
    )
    parser.add_argument(
        "--weather-csv",
        default="data/processed/weather_by_region_hour_20250623_18z.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="official_calpuff/case_20250623_18z_30sqmi/met/calmet_surrogate",
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--surface-stations", type=int, default=9)
    parser.add_argument("--itest", type=int, choices=(1, 2), default=1)
    parser.add_argument("--irtype", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--start-offset-hours",
        type=int,
        default=0,
        help="shift the generated output window relative to the first weather hour; boundary hours are held constant",
    )
    args = parser.parse_args()
    if args.surface_stations != 9:
        raise ValueError("The current generator intentionally uses a 3x3 surface-station layout")

    weather_path = _resolve(args.weather_csv)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weather = pd.read_csv(weather_path)
    weather["time"] = pd.to_datetime(weather["time_utc"], utc=True)
    weather = weather.sort_values(["hour_index", "region_id"]).reset_index(drop=True)
    if len(weather["hour_index"].unique()) < args.hours:
        raise ValueError("weather CSV does not contain the requested number of hours")
    weather = weather[weather["hour_index"] < args.hours].copy()
    weather = _expand_weather_window(weather, args.hours, args.start_offset_hours)

    to_lcc = Transformer.from_crs("EPSG:4326", LCC, always_xy=True)
    station_regions = _select_station_regions(weather, to_lcc)
    stations = []
    for index, region_id in enumerate(station_regions, start=1):
        row = weather[weather["region_id"].astype(str) == region_id].iloc[0]
        x_m, y_m = to_lcc.transform(float(row["centroid_lon"]), float(row["centroid_lat"]))
        stations.append(
            {
                "station_index": index,
                "region_id": region_id,
                "station_id": 10000 + index,
                "name": f"S{index:03d}",
                "lon": float(row["centroid_lon"]),
                "lat": float(row["centroid_lat"]),
                "x_km": x_m / 1000.0,
                "y_km": y_m / 1000.0,
            }
        )
    station_table = pd.DataFrame(stations)
    station_table.to_csv(output_dir / "surrogate_surface_stations.csv", index=False)

    base_start = pd.to_datetime(weather["source_time_utc"].min(), utc=True).to_pydatetime()
    start = base_start + timedelta(hours=args.start_offset_hours)
    end = start + timedelta(hours=args.hours - 1)
    x0 = -560.0
    y0 = -60.0
    dgrid = 10.0
    nx = 79
    ny = 38
    nz = 10
    _write_geo(output_dir / "GEO.DAT", nx, ny, dgrid, x0, y0)
    _write_surface(output_dir / "SURF.DAT", weather, station_table, start, end, args.hours)
    _write_upper_air(output_dir / "UP1.DAT", weather, start, end, args.hours, station_table.iloc[4])
    _write_control(
        output_dir / "CALMET.INP",
        start=start,
        end=end,
        hours=args.hours,
        itest=args.itest,
        irtype=args.irtype,
        station_table=station_table,
        x0=x0,
        y0=y0,
        nx=nx,
        ny=ny,
        dgrid=dgrid,
        nz=nz,
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "surrogate_calmet_from_open_meteo_fallback_weather",
        "scientific_status": "surrogate_not_formal_WRF_or_MMIF",
        "weather_csv": str(weather_path),
        "start_utc": start.isoformat(),
        "end_utc": (start + timedelta(hours=args.hours)).isoformat(),
        "hours": args.hours,
        "start_offset_hours": args.start_offset_hours,
        "irtype": args.irtype,
        "surface_station_count": len(station_table),
        "surface_station_selection": "3x3 nearest unique region centroids in LCC bounding box",
        "upper_air_profile": "synthetic pressure-height-temperature-wind profile anchored to center surface station",
        "geophysical_input": "flat elevation, default CALMET land-use parameter table, land-use code 40",
        "grid": {"nx": nx, "ny": ny, "nz": nz, "dgrid_km": dgrid, "xorig_km": x0, "yorig_km": y0},
        "itest": args.itest,
        "warning": "This input set is useful only for software-chain and sensitivity testing; it is not a verified target-period CALMET meteorology and must not replace WRF/MMIF in the paper reproduction.",
    }
    (output_dir / "surrogate_calmet_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(output_dir)
    return 0


def _expand_weather_window(weather: pd.DataFrame, hours: int, offset_hours: int) -> pd.DataFrame:
    """Create output-hour rows while explicitly clamping missing boundary hours."""
    source = weather.copy()
    source["source_time_utc"] = source["time"]
    source_by_hour = {
        int(hour): group.copy()
        for hour, group in source.groupby("hour_index", sort=True)
    }
    available = sorted(source_by_hour)
    if not available:
        raise ValueError("weather CSV has no source hours")
    rows: list[pd.DataFrame] = []
    for output_hour in range(hours):
        source_hour = min(max(output_hour + offset_hours, available[0]), available[-1])
        selected = source_by_hour[source_hour].copy()
        selected["hour_index"] = output_hour
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _select_station_regions(weather: pd.DataFrame, to_lcc: Transformer) -> list[str]:
    first = weather[weather["hour_index"] == weather["hour_index"].min()].copy()
    x, y = to_lcc.transform(first["centroid_lon"].to_numpy(), first["centroid_lat"].to_numpy())
    first["x_km"] = np.asarray(x) / 1000.0
    first["y_km"] = np.asarray(y) / 1000.0
    targets = [(x, y) for y in np.linspace(-30, 260, 3) for x in np.linspace(-450, 150, 3)]
    chosen: list[str] = []
    for tx, ty in targets:
        candidate = first.assign(distance=(first["x_km"] - tx) ** 2 + (first["y_km"] - ty) ** 2)
        for row in candidate.sort_values("distance").itertuples():
            region_id = str(row.region_id)
            if region_id not in chosen:
                chosen.append(region_id)
                break
    if len(chosen) != 9:
        raise ValueError("unable to select nine unique surface stations")
    return chosen


def _write_geo(path: Path, nx: int, ny: int, dgrid: float, x0: float, y0: float) -> None:
    lines = [
        f"{'GEO.DAT':<16}{'6.5.0':<16}{'SURROGATE FLAT GEO.DAT FOR SOFTWARE-CHAIN VALIDATION'}",
        "0",
        "LCC",
        f"{'N37.0':<16}{'W77.5':<16}{'N33.0':<16}{'N39.5':<16}",
        "0.0 0.0",
        f"{'WGS-84':<8}{'SURROGATE':<12}",
        f"{nx:8d}{ny:8d}{x0:12.3f}{y0:12.3f}{dgrid:12.3f}{dgrid:12.3f}",
        "KM",
        "0",
    ]
    lines.extend(" ".join(["40"] * nx) for _ in range(ny))
    lines.append("1.0")
    lines.extend(" ".join(["0.0"] * nx) for _ in range(ny))
    lines.extend(["0", "0", "0", "0", "0", "0"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_surface(
    path: Path,
    weather: pd.DataFrame,
    station_table: pd.DataFrame,
    start: datetime,
    end: datetime,
    hours: int,
) -> None:
    lines = [
        f"{'SURF.DAT':<16}{'1.0':<16}{'SURROGATE SURFACE OBSERVATIONS'}",
        "0",
        "LL",
        f"{'WGS-84':<8}{'SURROGATE':<10}",
        "KM",
        f"{start.year:04d} {start.timetuple().tm_yday:03d} {start.hour:02d} "
        f"{end.year:04d} {end.timetuple().tm_yday:03d} {end.hour:02d} 0 {len(station_table)}",
    ]
    for row in station_table.itertuples():
        lines.append(
            f"{int(row.station_id)} {row.name[:4]:<4}  N{float(row.lat):.4f} "
            f"W{abs(float(row.lon)):.4f} 10.0"
        )
    for hour_index in range(hours):
        rows = weather[weather["hour_index"] == hour_index].set_index(weather[weather["hour_index"] == hour_index]["region_id"].astype(str))
        values = []
        timestamp = start + timedelta(hours=hour_index)
        for region_id in station_table["region_id"]:
            row = rows.loc[str(region_id)]
            values.extend(
                [
                    f"{max(float(row['wind_speed_m_s']), 0.1):.3f}",
                    f"{float(row['wind_direction_deg_from']) % 360.0:.2f}",
                    "0",
                    "0",
                    f"{float(row['temperature_2m_c']) + 273.15:.2f}",
                    str(int(np.clip(round(float(row['relative_humidity_2m_pct'])), 1, 100))),
                    "1013.25",
                    "0",
                ]
            )
        lines.append(f"{timestamp.year:04d} {timestamp.timetuple().tm_yday:03d} {timestamp.hour:02d} " + " ".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_upper_air(
    path: Path,
    weather: pd.DataFrame,
    start: datetime,
    end: datetime,
    hours: int,
    station: pd.Series,
) -> None:
    header_start = start - timedelta(hours=1)
    header_end = end + timedelta(hours=1)
    lines = [
        f"{'UP.DAT':<16}{'1.0':<16}{'SURROGATE UPPER-AIR OBSERVATIONS'}",
        "0",
        "LL",
        f"{'WGS-84':<8}{'SURROGATE':<10}",
        "KM",
        f" {header_start.year:5d}{header_start.timetuple().tm_yday:5d}{header_start.hour:5d}"
        f"{header_end.year:5d}{header_end.timetuple().tm_yday:5d}{header_end.hour:5d}"
        f"{500.0:5.0f}{3:5d}{1:5d}",
        "     F    F    F    F",
        f"20001 U001 N{float(station['lat']):.4f} W{abs(float(station['lon'])):.4f} 0",
    ]
    levels = [(1000.0, 0.0, 1.00), (900.0, 1000.0, 1.05), (800.0, 2000.0, 1.10), (700.0, 3000.0, 1.15), (600.0, 4200.0, 1.20), (500.0, 5600.0, 1.25)]
    for hour_index in range(-1, hours + 1):
        timestamp = start + timedelta(hours=hour_index)
        weather_index = min(max(hour_index, 0), hours - 1)
        row = weather[weather["hour_index"] == weather_index]
        row = row.iloc[(row["region_id"].astype(str) == str(station["region_id"])).to_numpy().nonzero()[0][0]]
        nlevels = len(levels)
        lines.append(
            f"{'':9}{20001:8d}{'':5}"
            f"{timestamp.year % 100:02d}{timestamp.month:02d}{timestamp.day:02d}{timestamp.hour:02d}"
            f"{'':35}{nlevels:5d}"
        )
        base_temp = float(row["temperature_2m_c"])
        base_speed = max(float(row["wind_speed_m_s"]), 0.1)
        base_direction = float(row["wind_direction_deg_from"]) % 360.0
        records = []
        for pressure, height, speed_factor in levels:
            temp = base_temp - 0.0065 * height
            speed = base_speed * speed_factor
            records.append(f"   {pressure:6.1f} {height:5.0f} {temp:5.1f} {base_direction:3.0f} {speed:3.0f}")
        lines.extend("".join(records[i : i + 4]) for i in range(0, len(records), 4))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _put(chars: list[str], start: int, end: int, value: str) -> None:
    chars[start:end] = list(value.rjust(end - start)[: end - start])


def _write_control(
    path: Path,
    *,
    start: datetime,
    end: datetime,
    hours: int,
    itest: int,
    irtype: int,
    station_table: pd.DataFrame,
    x0: float,
    y0: float,
    nx: int,
    ny: int,
    dgrid: float,
    nz: int,
) -> None:
    zface = "0.,20.,40.,80.,160.,320.,640.,1200.,2000.,3000.,4000."
    ns = ",".join(["0"] * nz)
    nintr = ",".join(["9"] * nz)
    lines = [
        "SURROGATE CALMET CONTROL - NOT A FORMAL WRF/MMIF METEOROLOGY",
        "Open-Meteo weather fields with flat GEO.DAT and synthetic upper-air profile",
        "CALMET MODEL CONTROL FILE",
        "INPUT GROUP: 0 -- Input and Output File Names",
        f"! GEODAT=GEO.DAT !",
        f"! SRFDAT=SURF.DAT !",
        f"! METLST=CALMET.LST !",
        f"! METDAT=CALMET.DAT !",
        # This is a synthetic upper-air record used only for input-parser
        # smoke testing. It must not be treated as verified WRF/MMIF data.
        "! NUSTA=1 !",
        f"! NOWSTA=0 !",
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
        f"! NX={nx} !",
        f"! NY={ny} !",
        f"! DGRIDKM={dgrid:.3f} !",
        f"! XORIGKM={x0:.3f} !",
        f"! YORIGKM={y0:.3f} !",
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
        f"! NZ={nz} !",
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
        f"! NSSTA={len(station_table)} !",
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
    # Station locations are intentionally supplied by SURF.DAT/UP1.DAT.
    # Keeping SS/US records out of the control file avoids duplicate-location
    # conflicts and mirrors the formal file-driven CALMET workflow.
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
