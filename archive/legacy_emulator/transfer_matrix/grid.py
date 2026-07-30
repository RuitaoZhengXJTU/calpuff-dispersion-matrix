from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

from .config import CaseConfig


def _require_geo_stack() -> tuple[object, object, object, object, object]:
    try:
        import shapefile
        from pyproj import Transformer
        from shapely.geometry import Point, box, mapping, shape
        from shapely.ops import unary_union
    except ImportError as exc:
        raise RuntimeError(
            "build-grid requires shapely, pyproj, and pyshp. "
            "Install requirements.txt before generating real subregions."
        ) from exc
    return shapefile, Transformer, (Point, box, mapping, shape), unary_union, __import__("shapely.ops").ops.transform


def _projection_string(config: CaseConfig) -> str:
    projection = config.data["domain"]["projection"]
    parts = []
    for key, value in projection.items():
        if value is None:
            parts.append(f"+{key}")
        else:
            parts.append(f"+{key}={value}")
    return " ".join(parts)


def _cut_positions(poly, axis: str, count: int, box_factory) -> list[float]:
    minx, miny, maxx, maxy = poly.bounds
    low = miny if axis == "y" else minx
    high = maxy if axis == "y" else maxx
    total = poly.area
    cuts: list[float] = []

    def area_below(coord: float) -> float:
        if axis == "y":
            clip = box_factory(minx - 1, miny - 1, maxx + 1, coord)
        else:
            clip = box_factory(minx - 1, miny - 1, coord, maxy + 1)
        return poly.intersection(clip).area

    for index in range(1, count):
        target = total * index / count
        left = low
        right = high
        for _ in range(60):
            mid = (left + right) / 2
            if area_below(mid) < target:
                left = mid
            else:
                right = mid
        cuts.append((left + right) / 2)
    return cuts


def _clip_interval(poly, axis: str, lower: float, upper: float, box_factory):
    minx, miny, maxx, maxy = poly.bounds
    if axis == "y":
        clip = box_factory(minx - 1, lower, maxx + 1, upper)
    else:
        clip = box_factory(lower, miny - 1, upper, maxy + 1)
    return poly.intersection(clip)


def _make_equal_area_cells(config: CaseConfig, union_poly, box_factory) -> list[tuple[str, object]]:
    bands = int(config.data["grid"]["north_south_bands"])
    per_band = int(config.data["grid"]["east_west_cells_per_band"])
    minx, miny, maxx, maxy = union_poly.bounds
    y_cuts = [miny - 1, *_cut_positions(union_poly, "y", bands, box_factory), maxy + 1]
    cells: list[tuple[str, object]] = []

    for band_idx in range(bands):
        band = _clip_interval(union_poly, "y", y_cuts[band_idx], y_cuts[band_idx + 1], box_factory)
        x_min, _, x_max, _ = band.bounds
        x_cuts = [x_min - 1, *_cut_positions(band, "x", per_band, box_factory), x_max + 1]
        for cell_idx in range(per_band):
            region_id = f"r{len(cells):03d}"
            cell = _clip_interval(band, "x", x_cuts[cell_idx], x_cuts[cell_idx + 1], box_factory)
            cells.append((region_id, cell))
    return cells


def _sample_points(poly, point_factory, count: int) -> list[object]:
    side = int(count**0.5)
    if side * side != count:
        raise ValueError("sample point count must be a square number")
    minx, miny, maxx, maxy = poly.bounds
    points = []
    for row in range(side):
        y = miny + (row + 0.5) * (maxy - miny) / side
        for col in range(side):
            x = minx + (col + 0.5) * (maxx - minx) / side
            point = point_factory(x, y)
            if not poly.contains(point):
                point = poly.representative_point()
            points.append(point)
    return points


def build_grid(config: CaseConfig) -> Path:
    shapefile, Transformer, factories, unary_union, shapely_transform = _require_geo_stack()
    point_factory, box_factory, mapping_factory, shape_factory = factories

    fips = set(config.data["domain"]["state_fips"])
    extracted = _download_census_state_shapefile(config)
    shp_path = next(extracted.glob("*.shp"))
    reader = shapefile.Reader(str(shp_path), encoding="latin1")
    field_names = [field[0] for field in reader.fields[1:]]
    statefp_idx = field_names.index("STATEFP")
    geoms = []
    found = set()
    for record, shp in zip(reader.records(), reader.shapes()):
        statefp = str(record[statefp_idx])
        if statefp in fips:
            found.add(statefp)
            geoms.append(shape_factory(shp.__geo_interface__))
    if found != fips:
        raise RuntimeError(f"Expected FIPS {sorted(fips)}, found {sorted(found)}")

    to_projected = Transformer.from_crs("EPSG:4269", _projection_string(config), always_xy=True)
    to_wgs84 = Transformer.from_crs(_projection_string(config), "EPSG:4326", always_xy=True)
    projected_geoms = [shapely_transform(to_projected.transform, geom) for geom in geoms]
    union_poly = unary_union(projected_geoms)
    cells = _make_equal_area_cells(config, union_poly, box_factory)
    if len(cells) != config.target_regions:
        raise RuntimeError(f"Expected {config.target_regions} regions, got {len(cells)}")

    records = []
    for region_id, geom in cells:
        centroid = geom.centroid
        lon, lat = to_wgs84.transform(centroid.x, centroid.y)
        geom_wgs84 = shapely_transform(to_wgs84.transform, geom)
        records.append(
            {
                "region_id": region_id,
                "area_m2": geom.area,
                "centroid_x": centroid.x,
                "centroid_y": centroid.y,
                "centroid_lon": lon,
                "centroid_lat": lat,
                "geometry_projected": geom,
                "geometry_wgs84": geom_wgs84,
            }
        )

    subregions = config.output_path("subregions_geojson")
    subregions.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for item in records:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "region_id": item["region_id"],
                    "area_m2": item["area_m2"],
                    "centroid_lon": item["centroid_lon"],
                    "centroid_lat": item["centroid_lat"],
                    "centroid_x": item["centroid_x"],
                    "centroid_y": item["centroid_y"],
                },
                "geometry": mapping_factory(item["geometry_wgs84"]),
            }
        )
    subregions.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )

    _write_sample_points(config, records, point_factory, to_wgs84)
    return subregions


def _download_census_state_shapefile(config: CaseConfig) -> Path:
    url = config.data["domain"]["census_state_shapefile_url"]
    raw_dir = config.root / "data" / "raw" / "census"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / Path(url).name
    extracted = raw_dir / zip_path.stem
    if not zip_path.exists():
        urlretrieve(url, zip_path)
    if not extracted.exists():
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extracted)
    return extracted


def _write_sample_points(config: CaseConfig, records, point_factory, to_wgs84) -> None:
    source_count = int(config.data["grid"]["source_points_per_region"])
    receptor_count = int(config.data["grid"]["receptor_points_per_region"])
    source_rows = []
    receptor_rows = []

    for row in records:
        region_id = row["region_id"]
        for idx, point in enumerate(_sample_points(row["geometry_projected"], point_factory, source_count)):
            lon, lat = to_wgs84.transform(point.x, point.y)
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
        for idx, point in enumerate(_sample_points(row["geometry_projected"], point_factory, receptor_count)):
            lon, lat = to_wgs84.transform(point.x, point.y)
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

    _write_csv(config.output_path("sources_csv"), source_rows)
    _write_csv(config.output_path("receptors_csv"), receptor_rows)


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
