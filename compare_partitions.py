from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from transfer_matrix.config import CaseConfig, load_case
from transfer_matrix.diagnostics import write_diagnostics
from transfer_matrix.fallback_model import compute_advection_diffusion_matrices
from transfer_matrix.grid import (
    _cut_positions,
    _make_equal_area_cells,
    _projection_string,
    _require_geo_stack,
    _sample_points,
)
from transfer_matrix.validate import validate_outputs


CASE_TAG = "20250623_18z"
COMPARISON_ROOT = ROOT / "partition_comparison"
COUNTY_URL = "https://www2.census.gov/geo/tiger/TIGER2025/COUNTY/tl_2025_us_county.zip"

DATA_CENTER_POINTS = [
    ("Ashburn Data Center Alley", -77.4875, 39.0438, 1.00),
    ("Sterling / Dulles Corridor", -77.4291, 39.0062, 0.80),
    ("Herndon", -77.3861, 38.9696, 0.62),
    ("Reston", -77.3570, 38.9586, 0.55),
    ("Chantilly", -77.4311, 38.8943, 0.55),
    ("Manassas", -77.4753, 38.7509, 0.50),
    ("Tysons", -77.2311, 38.9187, 0.42),
    ("Silver Spring", -77.0261, 38.9907, 0.25),
    ("Rockville", -77.1528, 39.0839, 0.28),
    ("Washington DC Core", -77.0369, 38.9072, 0.20),
]

CITY_ANCHORS = [
    ("Baltimore", -76.6122, 39.2904),
    ("Annapolis", -76.4922, 38.9784),
    ("Frederick", -77.4105, 39.4143),
    ("Richmond", -77.4360, 37.5407),
    ("Norfolk", -76.2859, 36.8508),
    ("Virginia Beach", -75.9780, 36.8529),
    ("Charlottesville", -78.4767, 38.0293),
    ("Roanoke", -79.9414, 37.2710),
    ("Harrisonburg", -78.8689, 38.4496),
    ("Hagerstown", -77.7199, 39.6418),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build comparison transfer matrices for five partition schemes.")
    parser.add_argument("--case", default="config/case_20250623_18z.yaml")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["rectangular", "administrative", "hexagonal", "adaptive_grid", "hybrid"],
        help="Partition methods to compute.",
    )
    parser.add_argument(
        "--weather-mode",
        choices=["interpolate"],
        default="interpolate",
        help="Use the existing June 23 sampled weather field and interpolate it to each new centroid.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    base_case_path = Path(args.case)
    if not base_case_path.is_absolute():
        base_case_path = ROOT / base_case_path
    base_config = load_case(base_case_path)

    shapefile, Transformer, factories, unary_union, shapely_transform = _require_geo_stack()
    context = _load_domain_context(base_config, shapefile, Transformer, factories, unary_union, shapely_transform)
    base_weather = pd.read_csv(base_config.output_path("weather_api_csv"))

    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for method in args.methods:
        method_dir = COMPARISON_ROOT / method
        matrix_path = method_dir / f"transfer_matrices_{CASE_TAG}_{method}.npz"
        if args.skip_existing and matrix_path.exists():
            summary_rows.append(_matrix_summary(method, method_dir / "case.yaml", matrix_path))
            continue

        print(f"[{method}] generating partition")
        records = _partition_records(method, base_config, context)
        if len(records) != 100:
            raise RuntimeError(f"{method} generated {len(records)} regions; expected 100")
        _write_partition_files(method_dir, records, context, base_config)

        print(f"[{method}] writing case config and interpolated weather")
        case_path = _write_case_config(base_config, method, method_dir)
        config = load_case(case_path)
        _write_interpolated_weather(config, base_weather)

        print(f"[{method}] computing 24 hourly matrices")
        out = compute_advection_diffusion_matrices(config)
        checks = validate_outputs(config)
        write_diagnostics(config)
        summary = _matrix_summary(method, case_path, out)
        summary["validation_ok"] = checks["ok"]
        summary_rows.append(summary)

    summary_path = COMPARISON_ROOT / f"partition_matrix_summary_{CASE_TAG}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    _write_readme(summary_rows, summary_path)
    print(f"Wrote {summary_path}")
    return 0


def _load_domain_context(base_config, shapefile, Transformer, factories, unary_union, shapely_transform):
    point_factory, box_factory, mapping_factory, shape_factory = factories
    to_projected = Transformer.from_crs("EPSG:4269", _projection_string(base_config), always_xy=True)
    to_wgs84 = Transformer.from_crs(_projection_string(base_config), "EPSG:4326", always_xy=True)

    state_geoms = _read_state_geometries(base_config, shapefile, shape_factory)
    projected_states = {
        statefp: shapely_transform(to_projected.transform, geom)
        for statefp, geom in state_geoms.items()
    }
    union_poly = unary_union(list(projected_states.values()))

    data_centers = []
    for name, lon, lat, weight in DATA_CENTER_POINTS:
        point = shapely_transform(to_projected.transform, point_factory(lon, lat))
        if union_poly.contains(point):
            data_centers.append({"name": name, "point": point, "weight": weight, "lon": lon, "lat": lat})

    return {
        "point": point_factory,
        "box": box_factory,
        "mapping": mapping_factory,
        "shape": shape_factory,
        "unary_union": unary_union,
        "transform": shapely_transform,
        "to_projected": to_projected,
        "to_wgs84": to_wgs84,
        "states": projected_states,
        "union": union_poly,
        "data_centers": data_centers,
        "shapefile": shapefile,
    }


def _read_state_geometries(config: CaseConfig, shapefile, shape_factory) -> dict[str, object]:
    extracted = ROOT / "data" / "raw" / "census" / "tl_2025_us_state"
    shp_path = extracted / "tl_2025_us_state.shp"
    if not shp_path.exists():
        raise RuntimeError(f"Missing Census state shapefile: {shp_path}. Run build-grid first.")
    reader = shapefile.Reader(str(shp_path), encoding="latin1")
    fields = [field[0] for field in reader.fields[1:]]
    statefp_idx = fields.index("STATEFP")
    wanted = set(config.data["domain"]["state_fips"])
    geoms = {}
    for record, shp in zip(reader.records(), reader.shapes()):
        statefp = str(record[statefp_idx])
        if statefp in wanted:
            geoms[statefp] = shape_factory(shp.__geo_interface__)
    return geoms


def _partition_records(method: str, base_config: CaseConfig, context: dict[str, object]) -> list[dict[str, object]]:
    if method == "rectangular":
        return _records_from_cells(
            _make_equal_area_cells(base_config, context["union"], context["box"]),
            "rect",
            context,
            {"scheme": "equal_area_rectangular"},
        )
    if method == "administrative":
        return _administrative_records(base_config, context)
    if method == "hexagonal":
        return _hexagonal_records(context)
    if method == "adaptive_grid":
        return _adaptive_grid_records(context)
    if method == "hybrid":
        return _hybrid_records(context)
    raise ValueError(f"Unknown partition method: {method}")


def _records_from_cells(cells, prefix: str, context: dict[str, object], base_props: dict[str, object]) -> list[dict[str, object]]:
    records = []
    for idx, (_, geom) in enumerate(cells):
        if geom.is_empty:
            continue
        props = dict(base_props)
        records.append(_record(prefix, idx, geom, context, props))
    return records


def _record(prefix: str, idx: int, geom, context: dict[str, object], props: dict[str, object]) -> dict[str, object]:
    centroid = geom.centroid
    lon, lat = context["to_wgs84"].transform(centroid.x, centroid.y)
    geom_wgs84 = context["transform"](context["to_wgs84"].transform, geom)
    region_id = f"{prefix}_{idx:03d}"
    return {
        "region_id": region_id,
        "area_m2": float(geom.area),
        "centroid_x": float(centroid.x),
        "centroid_y": float(centroid.y),
        "centroid_lon": float(lon),
        "centroid_lat": float(lat),
        "geometry_projected": geom,
        "geometry_wgs84": geom_wgs84,
        "properties": props,
    }


def _administrative_records(base_config: CaseConfig, context: dict[str, object]) -> list[dict[str, object]]:
    counties = _read_counties(base_config, context)
    merged = _merge_cells_to_target(counties, 100, context)
    return [_record("admin", idx, item["geom"], context, item["props"]) for idx, item in enumerate(merged)]


def _read_counties(base_config: CaseConfig, context: dict[str, object]) -> list[dict[str, object]]:
    county_dir = _download_county_shapefile()
    shp_path = county_dir / "tl_2025_us_county.shp"
    reader = context["shapefile"].Reader(str(shp_path), encoding="latin1")
    fields = [field[0] for field in reader.fields[1:]]
    statefp_idx = fields.index("STATEFP")
    geoid_idx = fields.index("GEOID")
    name_idx = fields.index("NAME")
    wanted = set(base_config.data["domain"]["state_fips"])
    rows = []
    for record, shp in zip(reader.records(), reader.shapes()):
        statefp = str(record[statefp_idx])
        if statefp not in wanted:
            continue
        geom_wgs84 = context["shape"](shp.__geo_interface__)
        geom = context["transform"](context["to_projected"].transform, geom_wgs84)
        geom = geom.intersection(context["union"])
        if geom.is_empty:
            continue
        geoid = str(record[geoid_idx])
        name = str(record[name_idx])
        rows.append(
            {
                "geom": geom,
                "props": {
                    "scheme": "county_city_administrative_merged",
                    "admin_geoids": geoid,
                    "admin_names": name,
                    "source_count": 1,
                },
            }
        )
    return rows


def _download_county_shapefile() -> Path:
    raw_dir = ROOT / "data" / "raw" / "census"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "tl_2025_us_county.zip"
    extract_dir = raw_dir / "tl_2025_us_county"
    if not zip_path.exists():
        urlretrieve(COUNTY_URL, zip_path)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    return extract_dir


def _merge_cells_to_target(cells: list[dict[str, object]], target: int, context: dict[str, object]) -> list[dict[str, object]]:
    cells = [dict(item) for item in cells]
    while len(cells) > target:
        source_idx = min(range(len(cells)), key=lambda idx: cells[idx]["geom"].area)
        target_idx = _best_merge_target(cells, source_idx)
        first = cells[source_idx]
        second = cells[target_idx]
        merged = {
            "geom": context["unary_union"]([first["geom"], second["geom"]]),
            "props": _merged_props(first["props"], second["props"]),
        }
        for idx in sorted([source_idx, target_idx], reverse=True):
            del cells[idx]
        cells.append(merged)
    cells.sort(key=lambda item: (item["geom"].centroid.y, item["geom"].centroid.x))
    return cells


def _best_merge_target(cells: list[dict[str, object]], source_idx: int) -> int:
    geom = cells[source_idx]["geom"]
    best_idx = None
    best_score = -1.0
    for idx, candidate in enumerate(cells):
        if idx == source_idx:
            continue
        shared = geom.boundary.intersection(candidate["geom"].boundary).length
        distance = geom.distance(candidate["geom"])
        score = shared * 1_000_000.0 - distance
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        raise RuntimeError("No merge target found")
    return best_idx


def _merged_props(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    names = ";".join(filter(None, [str(a.get("admin_names", "")), str(b.get("admin_names", ""))]))
    geoids = ";".join(filter(None, [str(a.get("admin_geoids", "")), str(b.get("admin_geoids", ""))]))
    return {
        "scheme": a.get("scheme", b.get("scheme", "merged")),
        "admin_geoids": geoids,
        "admin_names": names,
        "source_count": int(a.get("source_count", 1)) + int(b.get("source_count", 1)),
    }


def _hexagonal_records(context: dict[str, object]) -> list[dict[str, object]]:
    union_poly = context["union"]
    target = 100
    ideal_radius = math.sqrt(union_poly.area / (target * (3.0 * math.sqrt(3.0) / 2.0)))
    best_cells = None
    best_count = 0
    for scale in np.linspace(0.80, 1.25, 37):
        cells = _make_hex_cells(union_poly, context["point"], ideal_radius * float(scale))
        count = len(cells)
        if count >= target and (best_cells is None or count < best_count):
            best_cells = cells
            best_count = count
    if best_cells is None:
        best_cells = _make_hex_cells(union_poly, context["point"], ideal_radius * 0.75)
    rows = [{"geom": geom, "props": {"scheme": "regular_hexagonal_clipped"}} for geom in best_cells]
    rows = _merge_cells_to_target(rows, target, context)
    return [_record("hex", idx, item["geom"], context, item["props"]) for idx, item in enumerate(rows)]


def _make_hex_cells(union_poly, point_factory, radius: float) -> list[object]:
    from shapely.geometry import Polygon

    minx, miny, maxx, maxy = union_poly.bounds
    dx = math.sqrt(3.0) * radius
    dy = 1.5 * radius
    cells = []
    row = 0
    y = miny - 2 * radius
    while y <= maxy + 2 * radius:
        x_offset = 0.5 * dx if row % 2 else 0.0
        x = minx - 2 * dx + x_offset
        while x <= maxx + 2 * dx:
            vertices = []
            for k in range(6):
                angle = math.radians(60 * k + 30)
                vertices.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
            poly = Polygon(vertices)
            clipped = poly.intersection(union_poly)
            if not clipped.is_empty and clipped.area > 1_000_000:
                cells.append(clipped)
            x += dx
        row += 1
        y += dy
    return cells


def _adaptive_grid_records(context: dict[str, object]) -> list[dict[str, object]]:
    cells = [{"geom": context["union"], "props": {"scheme": "source_weighted_adaptive_grid", "depth": 0}}]
    while len(cells) < 100:
        idx = max(range(len(cells)), key=lambda k: _adaptive_score(cells[k]["geom"], context))
        item = cells.pop(idx)
        parts = _split_equal_area(item["geom"], context)
        if len(parts) != 2:
            cells.append(item)
            break
        depth = int(item["props"].get("depth", 0)) + 1
        for part in parts:
            cells.append({"geom": part, "props": {"scheme": "source_weighted_adaptive_grid", "depth": depth}})
    cells.sort(key=lambda item: (item["geom"].centroid.y, item["geom"].centroid.x))
    return [_record("adapt", idx, item["geom"], context, item["props"]) for idx, item in enumerate(cells)]


def _adaptive_score(geom, context: dict[str, object]) -> float:
    centroid = geom.centroid
    influence = 0.0
    for item in context["data_centers"]:
        distance_km = max(geom.distance(item["point"]) / 1000.0, 1.0)
        influence += float(item["weight"]) * math.exp(-distance_km / 55.0)
    return geom.area * (1.0 + 8.0 * influence) + 0.08 * geom.length * math.sqrt(max(geom.area, 1.0))


def _split_equal_area(geom, context: dict[str, object]) -> list[object]:
    minx, miny, maxx, maxy = geom.bounds
    axis = "x" if (maxx - minx) >= (maxy - miny) else "y"
    cut = _cut_positions(geom, axis, 2, context["box"])[0]
    if axis == "x":
        boxes = [context["box"](minx - 1, miny - 1, cut, maxy + 1), context["box"](cut, miny - 1, maxx + 1, maxy + 1)]
    else:
        boxes = [context["box"](minx - 1, miny - 1, maxx + 1, cut), context["box"](minx - 1, cut, maxx + 1, maxy + 1)]
    parts = [geom.intersection(box) for box in boxes]
    return [part for part in parts if not part.is_empty and part.area > 1_000_000]


def _hybrid_records(context: dict[str, object]) -> list[dict[str, object]]:
    seeds = _hybrid_seed_points(context, 100)
    multipoint = context["unary_union"](seeds)
    from shapely.ops import voronoi_diagram

    diagram = voronoi_diagram(multipoint, envelope=context["union"].envelope, edges=False)
    cells = []
    used = set()
    for seed_idx, seed in enumerate(seeds):
        best_idx = None
        best_distance = float("inf")
        for idx, geom in enumerate(diagram.geoms):
            if idx in used:
                continue
            if geom.contains(seed) or geom.touches(seed):
                best_idx = idx
                best_distance = 0.0
                break
            dist = geom.distance(seed)
            if dist < best_distance:
                best_distance = dist
                best_idx = idx
        if best_idx is None:
            continue
        used.add(best_idx)
        clipped = diagram.geoms[best_idx].intersection(context["union"])
        if not clipped.is_empty:
            cells.append((seed_idx, clipped))
    if len(cells) != 100:
        raise RuntimeError(f"Hybrid Voronoi generated {len(cells)} cells, expected 100")
    return [
        _record("hybrid", idx, geom, context, {"scheme": "data_center_city_voronoi_hybrid", "seed_index": seed_idx})
        for idx, (seed_idx, geom) in enumerate(cells)
    ]


def _hybrid_seed_points(context: dict[str, object], target: int) -> list[object]:
    seeds = [item["point"] for item in context["data_centers"]]
    for _, lon, lat in CITY_ANCHORS:
        point = context["transform"](context["to_projected"].transform, context["point"](lon, lat))
        if context["union"].contains(point):
            seeds.append(point)
    candidates = _candidate_points(context, nx=42, ny=32)
    while len(seeds) < target:
        best = max(candidates, key=lambda point: min(point.distance(seed) for seed in seeds))
        seeds.append(best)
        candidates.remove(best)
    return seeds[:target]


def _candidate_points(context: dict[str, object], nx: int, ny: int) -> list[object]:
    minx, miny, maxx, maxy = context["union"].bounds
    points = []
    for row in range(ny):
        y = miny + (row + 0.5) * (maxy - miny) / ny
        for col in range(nx):
            x = minx + (col + 0.5) * (maxx - minx) / nx
            point = context["point"](x, y)
            if context["union"].contains(point):
                points.append(point)
    return points


def _write_partition_files(method_dir: Path, records: list[dict[str, object]], context: dict[str, object], base_config) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    features = []
    for row in records:
        props = {
            "region_id": row["region_id"],
            "area_m2": row["area_m2"],
            "centroid_lon": row["centroid_lon"],
            "centroid_lat": row["centroid_lat"],
            "centroid_x": row["centroid_x"],
            "centroid_y": row["centroid_y"],
        }
        props.update(row["properties"])
        features.append({"type": "Feature", "properties": props, "geometry": context["mapping"](row["geometry_wgs84"])})
    (method_dir / "subregions.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )
    _write_sample_csvs(method_dir, records, context, base_config)


def _write_sample_csvs(method_dir: Path, records: list[dict[str, object]], context: dict[str, object], base_config) -> None:
    source_count = int(base_config.data["grid"]["source_points_per_region"])
    receptor_count = int(base_config.data["grid"]["receptor_points_per_region"])
    source_rows = []
    receptor_rows = []
    for row in records:
        region_id = row["region_id"]
        for idx, point in enumerate(_sample_points(row["geometry_projected"], context["point"], source_count)):
            lon, lat = context["to_wgs84"].transform(point.x, point.y)
            source_rows.append(
                {
                    "source_id": f"{region_id}_s{idx:02d}",
                    "region_id": region_id,
                    "x_m": point.x,
                    "y_m": point.y,
                    "lon": lon,
                    "lat": lat,
                    "release_fraction": 1.0 / source_count,
                }
            )
        for idx, point in enumerate(_sample_points(row["geometry_projected"], context["point"], receptor_count)):
            lon, lat = context["to_wgs84"].transform(point.x, point.y)
            receptor_rows.append(
                {
                    "receptor_id": f"{region_id}_q{idx:02d}",
                    "region_id": region_id,
                    "x_m": point.x,
                    "y_m": point.y,
                    "lon": lon,
                    "lat": lat,
                    "target_area_m2": row["area_m2"],
                }
            )
    _write_csv(method_dir / "sources.csv", source_rows)
    _write_csv(method_dir / "receptors.csv", receptor_rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_case_config(base_config: CaseConfig, method: str, method_dir: Path) -> Path:
    data = deepcopy(base_config.data)
    data["case_id"] = f"{base_config.case_id}_{method}"
    data["description"] = f"{base_config.data.get('description', '')} Partition comparison: {method}."
    data["outputs"] = {
        "subregions_geojson": _posix_abs(method_dir / "subregions.geojson"),
        "sources_csv": _posix_abs(method_dir / "sources.csv"),
        "receptors_csv": _posix_abs(method_dir / "receptors.csv"),
        "hrrr_manifest": _posix_abs(method_dir / "hrrr_manifest.csv"),
        "weather_api_json": _posix_abs(method_dir / "open_meteo_interpolated_weather_source.json"),
        "weather_api_csv": _posix_abs(method_dir / "weather_by_region_hour.csv"),
        "matrix_npz": _posix_abs(method_dir / f"transfer_matrices_{CASE_TAG}_{method}.npz"),
        "matrices_dir": _posix_abs(method_dir / "matrices"),
        "diagnostics_dir": _posix_abs(method_dir / "diagnostics"),
        "provenance_json": _posix_abs(method_dir / "provenance.json"),
    }
    data["calpuff"]["case_root"] = _posix_abs(method_dir / "calpuff_cases")
    path = method_dir / "case.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _posix_abs(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _write_interpolated_weather(config: CaseConfig, base_weather: pd.DataFrame) -> None:
    subregions = json.loads(config.output_path("subregions_geojson").read_text(encoding="utf-8"))["features"]
    sub_rows = [feature["properties"] for feature in subregions]
    base_points = (
        base_weather[["region_id", "centroid_lon", "centroid_lat"]]
        .drop_duplicates("region_id")
        .sort_values("region_id")
        .reset_index(drop=True)
    )
    base_xy = _lonlat_to_xy(base_points["centroid_lon"].to_numpy(), base_points["centroid_lat"].to_numpy())
    rows = []
    for hour, hour_frame in base_weather.groupby("hour_index"):
        hour_frame = hour_frame.sort_values("region_id").reset_index(drop=True)
        speeds = hour_frame["wind_speed_m_s"].to_numpy(float)
        dirs = hour_frame["wind_direction_deg_from"].to_numpy(float)
        u = -speeds * np.sin(np.deg2rad(dirs))
        v = -speeds * np.cos(np.deg2rad(dirs))
        for region in sub_rows:
            target_xy = _lonlat_to_xy(np.asarray([region["centroid_lon"]]), np.asarray([region["centroid_lat"]]))[0]
            weights = _idw_weights(base_xy, target_xy, k=6)
            wind_u = float(np.dot(weights, u))
            wind_v = float(np.dot(weights, v))
            wind_speed = math.hypot(wind_u, wind_v)
            wind_dir = (math.degrees(math.atan2(-wind_u, -wind_v)) + 360.0) % 360.0
            row = {
                "region_id": region["region_id"],
                "hour_index": int(hour),
                "time_utc": hour_frame["time_utc"].iloc[0],
                "centroid_lon": region["centroid_lon"],
                "centroid_lat": region["centroid_lat"],
                "api_lon": None,
                "api_lat": None,
                "wind_speed_m_s": wind_speed,
                "wind_direction_deg_from": wind_dir,
            }
            for name in ["boundary_layer_height_m", "temperature_2m_c", "relative_humidity_2m_pct"]:
                values = hour_frame[name].to_numpy(float)
                row[name] = float(np.dot(weights, values))
            rows.append(row)
    out = config.output_path("weather_api_csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    raw = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source": "IDW interpolation from data/processed/weather_by_region_hour_20250623_18z.csv",
        "base_region_count": int(base_points.shape[0]),
        "target_region_count": int(len(sub_rows)),
    }
    config.output_path("weather_api_json").write_text(json.dumps(raw, indent=2), encoding="utf-8")


def _lonlat_to_xy(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lat0 = math.radians(38.4)
    x = lon * 111_320.0 * math.cos(lat0)
    y = lat * 110_540.0
    return np.column_stack([x, y])


def _idw_weights(base_xy: np.ndarray, target_xy: np.ndarray, k: int) -> np.ndarray:
    dist = np.linalg.norm(base_xy - target_xy, axis=1)
    order = np.argsort(dist)[:k]
    local = np.maximum(dist[order], 1.0)
    weights = 1.0 / (local * local)
    weights /= weights.sum()
    out = np.zeros(base_xy.shape[0], dtype=float)
    out[order] = weights
    return out


def _matrix_summary(method: str, case_path: Path, matrix_path: Path) -> dict[str, object]:
    payload = np.load(matrix_path, allow_pickle=True)
    matrix = payload["T"]
    col_sums = matrix.sum(axis=1)
    return {
        "method": method,
        "case_path": str(case_path.relative_to(ROOT)).replace("\\", "/"),
        "matrix_path": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
        "shape": "x".join(str(item) for item in matrix.shape),
        "nonzero_entries": int(np.count_nonzero(matrix)),
        "entry_max": float(matrix.max()),
        "column_sum_min": float(col_sums.min()),
        "column_sum_mean": float(col_sums.mean()),
        "column_sum_max": float(col_sums.max()),
        "all_zero_columns": int(np.sum(np.isclose(col_sums, 0.0))),
    }


def _write_readme(summary_rows: list[dict[str, object]], summary_path: Path) -> None:
    lines = [
        "# Partition Comparison Matrices",
        "",
        f"Case window: 2025-06-23 14:00 EDT / {CASE_TAG}.",
        "",
        "Each subfolder contains one 100-region partition, its 24 hourly transfer matrices,",
        "diagnostics, and a case YAML. Weather was interpolated from the already fetched",
        "June 23 Open-Meteo/HRRR historical forecast centroid samples so the five schemes",
        "share the same meteorological field without refetching external data.",
        "",
        "Matrix convention: `T[h, j, i]` is the hour-h transfer coefficient from source",
        "region `i` to target region `j`, so `x_next = T[h] @ x_now`.",
        "",
        f"Summary CSV: `{summary_path.relative_to(ROOT).as_posix()}`",
        "",
        "| method | shape | column_sum_min | column_sum_mean | column_sum_max | max entry |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['shape']} | {row['column_sum_min']:.6f} | "
            f"{row['column_sum_mean']:.6f} | {row['column_sum_max']:.6f} | {row['entry_max']:.6f} |"
        )
    (COMPARISON_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
