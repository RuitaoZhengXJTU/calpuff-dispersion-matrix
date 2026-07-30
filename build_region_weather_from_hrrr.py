"""Create the A-matrix mixed-layer-height table directly from raw HRRR GRIB2.

The official A_h experiment needs one boundary-layer height for every
subregion and interval to convert unit concentration to an equivalent source
mass. Gas ppb output additionally needs temperature and pressure at every
state endpoint. This script samples ``HPBL:surface``, ``TMP:2 m above ground``,
and ``PRES:surface`` in the selected HRRR files at an interior representative
point of each GeoJSON subregion. It does not run or approximate a dispersion
model; CALPUFF still receives meteorology through the separately generated
CALMET.DAT file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shapely.geometry import shape

from concentration_units import DEFAULT_NO2_MOLECULAR_WEIGHT_G_MOL, ppb_per_g_m3


ROOT = Path(__file__).resolve().parent
VALUE_PATTERN = re.compile(r"val=([-+0-9.eE]+|NaN|nan)")
DEFAULT_GRIB_DIR = ROOT / "data" / "raw" / "hrrr_20250623_18z"
DEFAULT_SUBREGIONS = (
    ROOT
    / "population_partitions"
    / "area_capped_30sqmi_population_balanced"
    / "subregions.geojson"
)
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "hrrr_region_weather_20250623_18z" / "weather_by_region_hour.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grib-dir", type=Path, default=DEFAULT_GRIB_DIR)
    parser.add_argument("--subregions", type=Path, default=DEFAULT_SUBREGIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--date", default="20250623", help="HRRR cycle date, YYYYMMDD")
    parser.add_argument("--cycle", type=int, default=18, help="HRRR cycle UTC hour")
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--wgrib2", type=Path, required=True)
    parser.add_argument(
        "--molecular-weight-g-mol",
        type=float,
        default=DEFAULT_NO2_MOLECULAR_WEIGHT_G_MOL,
        help="Molecular weight used for the gas mass-concentration to ppb factor.",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args(argv)

    if not args.wgrib2.exists():
        raise FileNotFoundError(args.wgrib2)
    if not args.subregions.exists():
        raise FileNotFoundError(args.subregions)
    if not 0 <= args.start_hour <= 48 or not 1 <= args.hours <= 48 or args.start_hour + args.hours > 48:
        raise ValueError("start-hour plus hours must be within HRRR f00--f48")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    points = _representative_points(args.subregions)
    rows: list[dict[str, object]] = []
    cycle_start = datetime.strptime(args.date, "%Y%m%d").replace(
        hour=args.cycle, tzinfo=timezone.utc
    )
    for hour_index in range(args.hours):
        forecast_hour = args.start_hour + hour_index
        grib = args.grib_dir / f"hrrr.t{args.cycle:02d}z.wrfsfcf{forecast_hour:02d}.grib2.selected.grib2"
        if not grib.exists():
            raise FileNotFoundError(grib)
        heights = _sample_wgrib2(args.wgrib2, grib, "HPBL:surface", points, args.batch_size)
        temperatures = _sample_wgrib2(args.wgrib2, grib, "TMP:2 m above ground", points, args.batch_size)
        pressures = _sample_wgrib2(args.wgrib2, grib, "PRES:surface", points, args.batch_size)
        timestamp = cycle_start + timedelta(hours=forecast_hour)
        for (region_id, lon, lat), height, temperature_k, pressure_pa in zip(
            points, heights, temperatures, pressures, strict=True
        ):
            if height <= 0:
                raise ValueError(f"HPBL must be positive for {region_id} at hour {hour_index}: {height}")
            factor = float(ppb_per_g_m3(temperature_k, pressure_pa, args.molecular_weight_g_mol))
            rows.append(
                {
                    "region_id": region_id,
                    "hour_index": hour_index,
                    "time_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "centroid_lon": lon,
                    "centroid_lat": lat,
                    "boundary_layer_height_m": height,
                    "temperature_k": temperature_k,
                    "pressure_pa": pressure_pa,
                    "ppb_per_g_m3": factor,
                }
            )
        print(f"f{forecast_hour:02d}: sampled HPBL, temperature, and pressure at {len(points)} subregions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "method": "wgrib2 -lon sampling of HPBL:surface, TMP:2 m above ground, and PRES:surface at GeoJSON representative points",
        "subregions": str(args.subregions),
        "grib_dir": str(args.grib_dir),
        "date": args.date,
        "cycle_utc": args.cycle,
        "start_forecast_hour": args.start_hour,
        "hours": args.hours,
        "molecular_weight_g_mol": args.molecular_weight_g_mol,
        "ppb_conversion": "ppb_per_g_m3 = R * temperature_k * 1e9 / (pressure_pa * molecular_weight_g_mol)",
        "region_count": len(points),
        "wgrib2": str(args.wgrib2),
    }
    args.output.with_name("weather_by_region_hour.provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0


def _representative_points(path: Path) -> list[tuple[str, float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points: list[tuple[str, float, float]] = []
    for feature in payload.get("features", []):
        region_id = str(feature.get("properties", {}).get("region_id", ""))
        if not region_id:
            raise ValueError("each GeoJSON feature requires properties.region_id")
        point = shape(feature["geometry"]).representative_point()
        points.append((region_id, float(point.x), float(point.y)))
    if not points:
        raise ValueError(f"no features found in {path}")
    if len({region_id for region_id, _, _ in points}) != len(points):
        raise ValueError("subregion region_id values must be unique")
    return sorted(points)


def _sample_wgrib2(
    wgrib2: Path,
    grib: Path,
    selector: str,
    points: list[tuple[str, float, float]],
    batch_size: int,
) -> list[float]:
    values: list[float] = []
    for offset in range(0, len(points), batch_size):
        batch = points[offset : offset + batch_size]
        command = [str(wgrib2), str(grib), "-match", f":{selector}:"]
        for _, lon, lat in batch:
            command.extend(["-lon", f"{lon:.8f}", f"{lat:.8f}"])
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        batch_values = [float(value) for value in VALUE_PATTERN.findall(completed.stdout)]
        if len(batch_values) != len(batch):
            raise RuntimeError(
                f"wgrib2 returned {len(batch_values)} values for {selector}, expected {len(batch)}"
            )
        values.extend(batch_values)
    return values


if __name__ == "__main__":
    raise SystemExit(main())
