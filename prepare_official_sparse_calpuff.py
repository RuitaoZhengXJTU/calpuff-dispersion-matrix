from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform as shapely_transform

from official_case_builder import CalpuffDomain
from official_case_config import load_case_config, mapping_value, project_path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare official sparse CALPUFF run skeleton.")
    parser.add_argument("--case-config", type=Path, default=None)
    parser.add_argument("--partition-dir", type=Path, default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--case-root", type=Path, default=None)
    parser.add_argument("--start-utc", default=None)
    parser.add_argument("--hours", type=int, default=None)
    parser.add_argument("--receptor-points-per-region", type=int, default=9)
    parser.add_argument("--max-discrete-receptors", type=int, default=10000)
    parser.add_argument("--calpuff-seed", default=os.environ.get("CALPUFF_SEED"), help="Path to the official CALPUFF seed control file.")
    args = parser.parse_args()

    config = load_case_config(args.case_config)
    config_paths = mapping_value(config, "paths")
    config_time = mapping_value(config, "time")
    config_met = mapping_value(config, "meteorology")
    domain = CalpuffDomain.from_mapping(mapping_value(config, "calpuff_domain"))
    partition_dir = args.partition_dir or project_path(ROOT, config_paths.get("partition_dir"))
    if partition_dir is None:
        partition_dir = ROOT / "population_partitions" / "area_capped_30sqmi_population_balanced"
    if not partition_dir.exists():
        raise FileNotFoundError(partition_dir)
    args.case_id = args.case_id or str(config.get("case_id") or "case_20250623_18z_30sqmi")
    root = args.case_root or project_path(ROOT, config_paths.get("case_root")) or ROOT / "official_calpuff" / args.case_id
    start = _parse_utc(args.start_utc or str(config_time.get("start_utc", "2025-06-23T18:00:00Z")))
    hours = args.hours if args.hours is not None else int(config_time.get("hours", 24))
    if hours < 2:
        raise ValueError("hours must be at least 2")
    mmif_grid = config_met.get("mmif_grid_ll")
    if mmif_grid is None:
        mmif_grid = _buffered_geographic_bounds(partition_dir / "subregions.geojson")
    if not isinstance(mmif_grid, list) or len(mmif_grid) != 4:
        raise ValueError("meteorology.mmif_grid_ll must be [south, west, north, east]")
    mmif_grid_ll = tuple(float(value) for value in mmif_grid)
    paths = {
        "root": root,
        "wrf_inputs": root / "met" / "wrf_inputs",
        "mmif": root / "met" / "mmif",
        "inputs": root / "inputs",
        "calpuff_templates": root / "templates",
        "smoke": root / "smoke_tests" / "hour_00_source_0000",
        "outputs": root / "outputs",
        "matrices_sparse": root / "outputs" / "matrices_sparse",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    _write_mmif_draft(paths["mmif"] / "mmif.inp", start, hours, mmif_grid_ll)
    _write_wrf_manifest(paths["wrf_inputs"] / "WRF_INPUTS_REQUIRED.txt", start, hours)
    _copy_calpuff_seed(paths["calpuff_templates"] / "CALPUFF_7.0_seed_from_distribution.INP", args.calpuff_seed)
    regions = _write_case_inputs(
        paths["inputs"], partition_dir, args.max_discrete_receptors,
        domain.projected_crs, args.receptor_points_per_region,
    )
    _write_sparse_strategy(
        paths["root"] / "sparse_official_strategy.json",
        regions,
        args.receptor_points_per_region,
        args.max_discrete_receptors,
        hours,
    )
    _write_readme(
        root, partition_dir, regions, args.receptor_points_per_region,
        args.max_discrete_receptors, start, hours,
    )
    print(root)
    return 0


def _write_mmif_draft(
    path: Path,
    start: datetime,
    hours: int,
    grid_ll: tuple[float, float, float, float],
) -> None:
    end = start + timedelta(hours=hours)
    south, west, north, east = grid_ll
    text = f"""; Draft MMIF control for the configured official CALPUFF matrix case.
; This file is not ready to run until INPUT lines point to real WRF/ARW files.
; MMIF time is interpreted with TimeZone. Use TimeZone 0 for UTC.

start      {start.strftime('%Y-%m-%d_%H:%M:%S')}
stop       {end.strftime('%Y-%m-%d_%H:%M:%S')}
TimeZone   0

; South, west, north, east bounding box, including the configured transport buffer.
grid       LL {south:.6f} {west:.6f} {north:.6f} {east:.6f}

layers top 20 40 80 160 320 640 1200 2000 3000 4000
stability  GOLDER
CLOUDCOVER WRF
CALSCI_MIXHT WRF

Output qaplot     BLN      domain.bln
Output qaplot     KML      qaplot.kml
Output calpuff    useful   calmet.info.txt
Output calpuff    calmet   CALMET.DAT
Output calpuff    terrain  TERRAIN.GRD
Output calpuffv6  useful   calmetv6.info.txt
Output calpuffv6  calmet   CALMETV6.DAT
Output calpuffv6  aux      CALMETV6.AUX
Output calpuffv6  terrain  TERRAINV6.GRD

; Replace these placeholders with WRF/ARW wrfout files covering the target window.
; HRRR GRIB2 files are not accepted here unless converted to MMIF-compatible WRF/ARW.
; INPUT met/wrf_inputs/wrfout_d01_{start.strftime('%Y-%m-%d_%H_%M_%S')}
"""
    path.write_text(text, encoding="ascii")


def _write_wrf_manifest(path: Path, start: datetime, hours: int) -> None:
    end = start + timedelta(hours=hours)
    text = f"""Required meteorology for official MMIF/CALPUFF route

Provide WRF/ARW wrfout files covering:
  {start.isoformat().replace('+00:00', 'Z')} through {end.isoformat().replace('+00:00', 'Z')}

MMIF v4.1.1 expects MM5 or WRF/ARW model output. HRRR archive files are GRIB2
and are not sufficient unless converted into MMIF-compatible WRF/ARW files with
the required variables, dimensions, projection metadata, and time coordinates.

Place files in this directory and uncomment/update INPUT lines in:
  ../mmif/mmif.inp
"""
    path.write_text(text, encoding="ascii")


def _copy_calpuff_seed(path: Path, source_text: str | None) -> None:
    if not source_text:
        return
    source = Path(source_text)
    if source.exists():
        shutil.copyfile(source, path)


def _write_sparse_strategy(
    path: Path,
    regions: pd.DataFrame,
    receptor_points_per_region: int,
    max_discrete_receptors: int,
    hours: int,
) -> None:
    target_receptor_count = int(len(regions) * receptor_points_per_region)
    receptor_batches = int(np.ceil(target_receptor_count / max_discrete_receptors))
    source_hour_cases = int(len(regions) * hours)
    payload = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "region_count": int(len(regions)),
        "hours": hours,
        "source_hour_cases_full": source_hour_cases,
        "receptor_points_per_region": receptor_points_per_region,
        "target_receptor_count": target_receptor_count,
        "verified_max_discrete_receptors_assumed": max_discrete_receptors,
        "receptor_batches": receptor_batches,
        "batched_direct_run_count": source_hour_cases * receptor_batches,
        "recommended_output": "scipy sparse matrices, one file per hour",
        "matrix_shape_per_hour": [int(len(regions)), int(len(regions))],
        "source_model": {
            "pollutant": "inert passive tracer",
            "release_height_m": 15.0,
            "chemistry": "off",
            "dry_deposition": "off",
            "wet_deposition": "off",
            "decay": "off",
        },
        "sparsity_policy": {
            "official_calpuff_still_runs_source_columns": True,
            "parse_nearby_or_nonzero_receptors_only": True,
            "avoid_dense_csv": True,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_case_inputs(
    output_dir: Path,
    partition_dir: Path,
    max_discrete_receptors: int,
    projected_crs: str,
    receptor_points_per_region: int,
) -> pd.DataFrame:
    """Create stable source/receptor point tables for the official case skeleton."""
    payload = json.loads((partition_dir / "subregions.geojson").read_text(encoding="utf-8"))
    raw_features = payload.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError("subregions.geojson must be a nonempty FeatureCollection")
    try:
        features = sorted(raw_features, key=lambda feature: str(feature["properties"]["region_id"]))
    except (KeyError, TypeError) as exc:
        raise ValueError("each subregions.geojson feature must have properties.region_id") from exc
    to_projected = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    )
    to_wgs84 = Transformer.from_crs(
        projected_crs,
        "EPSG:4326",
        always_xy=True,
    )
    source_rows = []
    receptor_rows = []
    region_rows = []
    for feature in features:
        props = feature["properties"]
        region_id = str(props["region_id"])
        geom = shapely_transform(to_projected.transform, shape(feature["geometry"]))
        if not region_id or region_id == "None":
            raise ValueError("each subregions.geojson feature must have properties.region_id")
        region_rows.append(
            {
                "matrix_index": len(region_rows),
                "region_id": region_id,
                "area_m2": float(props.get("area_m2", geom.area)),
            }
        )
        for point_index, point in enumerate(_sample_points(geom, 16)):
            lon, lat = to_wgs84.transform(point.x, point.y)
            source_rows.append(
                {
                    "source_id": f"{region_id}_s{point_index:02d}",
                    "region_id": region_id,
                    "matrix_index": len(region_rows) - 1,
                    "x_m": point.x,
                    "y_m": point.y,
                    "lon": lon,
                    "lat": lat,
                    "release_fraction": 1.0 / 16.0,
                    "release_height_m": 15.0,
                }
            )
        for point_index, point in enumerate(_sample_points(geom, receptor_points_per_region)):
            lon, lat = to_wgs84.transform(point.x, point.y)
            receptor_rows.append(
                {
                    "receptor_id": f"{region_id}_q{point_index:02d}",
                    "region_id": region_id,
                    "matrix_index": len(region_rows) - 1,
                    "x_m": point.x,
                    "y_m": point.y,
                    "lon": lon,
                    "lat": lat,
                    "receptor_height_m": 1.5,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    region_frame = pd.DataFrame(region_rows)
    if not region_frame["region_id"].is_unique:
        raise ValueError("subregions.geojson contains duplicate region_id values")
    region_frame.to_csv(output_dir / "matrix_region_index.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output_dir / "sources_16_per_region.csv", index=False)
    receptors = pd.DataFrame(receptor_rows)
    receptors.to_csv(output_dir / f"receptors_{receptor_points_per_region}_per_region.csv", index=False)
    _write_receptor_batches(output_dir, receptors, max_discrete_receptors)
    return region_frame


def _write_receptor_batches(output_dir: Path, receptors: pd.DataFrame, max_discrete_receptors: int) -> None:
    if max_discrete_receptors <= 0:
        raise ValueError("max_discrete_receptors must be positive")
    batch_dir = output_dir / "receptor_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    # Batches are generated artifacts. Clearing only this known filename
    # pattern prevents a repartitioned case from retaining stale receptors.
    for stale_batch in batch_dir.glob("batch_*.csv"):
        stale_batch.unlink()
    rows: list[dict[str, object]] = []
    current_frames: list[pd.DataFrame] = []
    current_count = 0
    batch_id = 0
    start_region_index: int | None = None

    for _, group in receptors.groupby("matrix_index", sort=False):
        group = group.copy()
        group_count = len(group)
        if group_count > max_discrete_receptors:
            raise ValueError("One target region has more receptors than the compiled MXREC limit")
        if current_frames and current_count + group_count > max_discrete_receptors:
            batch = pd.concat(current_frames, ignore_index=True)
            filename = f"batch_{batch_id:03d}.csv"
            batch.to_csv(batch_dir / filename, index=False)
            rows.append(
                {
                    "batch_id": batch_id,
                    "filename": f"receptor_batches/{filename}",
                    "start_matrix_index": int(batch["matrix_index"].min()),
                    "end_matrix_index": int(batch["matrix_index"].max()),
                    "region_count": int(batch["matrix_index"].nunique()),
                    "receptor_count": int(len(batch)),
                    "max_discrete_receptors": max_discrete_receptors,
                }
            )
            batch_id += 1
            current_frames = []
            current_count = 0
            start_region_index = None
        current_frames.append(group)
        current_count += group_count
        if start_region_index is None:
            start_region_index = int(group["matrix_index"].iloc[0])

    if current_frames:
        batch = pd.concat(current_frames, ignore_index=True)
        filename = f"batch_{batch_id:03d}.csv"
        batch.to_csv(batch_dir / filename, index=False)
        rows.append(
            {
                "batch_id": batch_id,
                "filename": f"receptor_batches/{filename}",
                "start_matrix_index": int(batch["matrix_index"].min()),
                "end_matrix_index": int(batch["matrix_index"].max()),
                "region_count": int(batch["matrix_index"].nunique()),
                "receptor_count": int(len(batch)),
                "max_discrete_receptors": max_discrete_receptors,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "receptor_batch_manifest.csv", index=False)


def _sample_points(geom, count: int) -> list[Point]:
    side = int(np.sqrt(count))
    if side * side != count:
        raise ValueError("source/receptor sample count must be a square")
    minx, miny, maxx, maxy = geom.bounds
    points = []
    for row in range(side):
        y = miny + (row + 0.5) * (maxy - miny) / side
        for col in range(side):
            x = minx + (col + 0.5) * (maxx - minx) / side
            point = Point(x, y)
            if not geom.covers(point):
                point = geom.representative_point()
            points.append(point)
    return points


def _buffered_geographic_bounds(path: Path, buffer_degrees: float = 0.5) -> list[float]:
    """Return south, west, north, east with a minimal transport buffer."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    geometries = [shape(feature["geometry"]) for feature in payload.get("features", [])]
    if not geometries:
        raise ValueError(f"subregion GeoJSON has no features: {path}")
    minx = min(geom.bounds[0] for geom in geometries)
    miny = min(geom.bounds[1] for geom in geometries)
    maxx = max(geom.bounds[2] for geom in geometries)
    maxy = max(geom.bounds[3] for geom in geometries)
    return [miny - buffer_degrees, minx - buffer_degrees, maxy + buffer_degrees, maxx + buffer_degrees]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("start_utc must include a UTC offset or Z")
    return parsed.astimezone(timezone.utc)


def _write_readme(
    root: Path,
    partition_dir: Path,
    regions: pd.DataFrame,
    receptor_points_per_region: int,
    max_discrete_receptors: int,
    start: datetime,
    hours: int,
) -> None:
    receptor_count = len(regions) * receptor_points_per_region
    receptor_batches = int(np.ceil(receptor_count / max_discrete_receptors))
    end = start + timedelta(hours=hours)
    try:
        partition_label = partition_dir.relative_to(ROOT).as_posix()
    except ValueError:
        partition_label = str(partition_dir)
    text = f"""# Official Sparse CALPUFF Skeleton

Partition: `{partition_label}`

Region count: {len(regions):,}

Target window: {start.isoformat().replace('+00:00', 'Z')} through {end.isoformat().replace('+00:00', 'Z')}.

This folder prepares the official sparse CALPUFF route but does not make it
scientifically runnable by itself. Remaining blocker: provide MMIF-compatible
WRF/ARW input files or a verified HRRR-GRIB2-to-WRF conversion.

Files created:

- `met/mmif/mmif.inp`: draft MMIF control file.
- `met/wrf_inputs/WRF_INPUTS_REQUIRED.txt`: meteorology requirements.
- `inputs/matrix_region_index.csv`: stable region matrix ordering.
- `inputs/sources_16_per_region.csv`: 16 equal-weight source points per region at 15 m.
- `inputs/receptors_9_per_region.csv`: {receptor_points_per_region} receptor sample points per region at 1.5 m ({receptor_count:,} total).
- `inputs/receptor_batch_manifest.csv` and `inputs/receptor_batches/`: fixed batch files that preserve complete target regions.
- `inputs/sparse_candidate_manifest/`: emulator-screened candidate targets for the sparse approximation; not official CALPUFF output.
- `templates/CALPUFF_7.0_seed_from_distribution.INP`: official distribution seed
  control file to adapt for inert tracer source-response cases.
- `sparse_official_strategy.json`: sparse matrix strategy and run scale.

The installed CALPUFF v7.2.1 executable was probed at `MXREC=10000`; the
current receptor table therefore requires {receptor_batches} fixed batches.
This is a compile-time limit, so rerun `probe_calpuff_compiled_limits.py` if the
executable changes. Every batch must preserve its receptor-to-region manifest.

Once WRF input is available:

```powershell
cd "{root}"
& $env:MMIF_EXE met\\mmif\\mmif.inp
```

Then inspect `met/mmif/CALMET.DAT` or `CALMETV6.DAT`, adapt the CALPUFF seed
control file, and run a one-source one-hour smoke test before launching the full
matrix.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
