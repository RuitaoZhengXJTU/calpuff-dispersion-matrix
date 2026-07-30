from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlretrieve

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "archive" / "legacy_emulator"))

from transfer_matrix.config import load_case
from transfer_matrix.grid import _projection_string, _require_geo_stack, _sample_points


DEFAULT_STATES = ["11", "24", "51"]
POP_FIELD = "POP20"


@dataclass
class BlockGroup:
    geoid: str
    name: str
    statefp: str
    population: int
    geometry: object
    centroid_x: float
    centroid_y: float
    area_m2: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build equal-population DC/VA/MD subregions from Census population blocks.")
    parser.add_argument("--case", default="config/case_20250623_18z.yaml")
    parser.add_argument("--year", type=int, default=2020, help="TIGER tabulation-block vintage with POP20 population.")
    parser.add_argument("--target-regions", type=int, default=1000)
    parser.add_argument("--states", nargs="*", default=DEFAULT_STATES)
    parser.add_argument("--output-dir", default="population_partitions/equal_population_1000")
    args = parser.parse_args(argv)

    case_path = Path(args.case)
    if not case_path.is_absolute():
        case_path = ROOT / case_path
    config = load_case(case_path)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    shapefile, Transformer, factories, unary_union, shapely_transform = _require_geo_stack()
    point_factory, _, mapping_factory, shape_factory = factories
    to_projected = Transformer.from_crs("EPSG:4269", _projection_string(config), always_xy=True)
    to_wgs84 = Transformer.from_crs(_projection_string(config), "EPSG:4326", always_xy=True)

    print("Downloading/loading TIGER tabulation blocks with POP20 population")
    block_groups = _load_tabulation_blocks(
        year=args.year,
        states=args.states,
        shapefile=shapefile,
        shape_factory=shape_factory,
        shapely_transform=shapely_transform,
        to_projected=to_projected,
    )
    if len(block_groups) < args.target_regions:
        raise RuntimeError(f"Only {len(block_groups)} block groups available; cannot create {args.target_regions} regions")

    print(f"Loaded {len(block_groups)} Census blocks with total population {sum(bg.population for bg in block_groups):,}")
    groups = _recursive_population_split(block_groups, args.target_regions)
    print(f"Built {len(groups)} population-balanced groups")

    records = _write_regions(out_dir, groups, config, unary_union, shapely_transform, to_wgs84, mapping_factory)
    _write_source_receptor_csvs(out_dir, records, config, point_factory, to_wgs84)
    _write_assignment_csv(out_dir, groups)
    stats = _write_validation(out_dir, records, len(block_groups), args.year)
    _write_report(out_dir, stats, args.year)
    print(out_dir)
    return 0


def _fetch_acs_population(year: int, states: list[str], out_dir: Path) -> dict[str, dict[str, object]]:
    cache_path = out_dir / f"acs{year}_block_group_population.csv"
    if cache_path.exists():
        frame = pd.read_csv(cache_path, dtype={"geoid": str, "state": str, "county": str, "tract": str, "block group": str})
        return {
            row["geoid"]: {"population": int(row["population"]), "name": row["name"], "statefp": row["state"]}
            for _, row in frame.iterrows()
        }

    rows = []
    for state in states:
        params = {
            "get": f"NAME,B01003_001E",
            "for": "block group:*",
            "in": f"state:{state} county:* tract:*",
        }
        url = f"https://api.census.gov/data/{year}/acs/acs5?{urlencode(params)}"
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        payload = response.json()
        header = payload[0]
        for values in payload[1:]:
            item = dict(zip(header, values))
            geoid = f"{item['state']}{item['county']}{item['tract']}{item['block group']}"
            pop_raw = item.get("B01003_001E")
            population = int(pop_raw) if pop_raw not in (None, "", "-666666666") else 0
            rows.append(
                {
                    "geoid": geoid,
                    "name": item["NAME"],
                    "population": max(population, 0),
                    "state": item["state"],
                    "county": item["county"],
                    "tract": item["tract"],
                    "block group": item["block group"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)
    return {
        row["geoid"]: {"population": int(row["population"]), "name": row["name"], "statefp": row["state"]}
        for _, row in frame.iterrows()
    }


def _load_tabulation_blocks(
    year: int,
    states: list[str],
    shapefile,
    shape_factory,
    shapely_transform,
    to_projected,
) -> list[BlockGroup]:
    raw_dir = ROOT / "data" / "raw" / "census" / f"tiger{year}_tabblock"
    raw_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    for state in states:
        extract_dir = _download_tiger_tabblock(year, state, raw_dir)
        shp_path = extract_dir / f"tl_{year}_{state}_tabblock20.shp"
        reader = shapefile.Reader(str(shp_path), encoding="latin1")
        fields = [field[0] for field in reader.fields[1:]]
        geoid_idx = fields.index("GEOID20")
        state_idx = fields.index("STATEFP20")
        name_idx = fields.index("NAME20")
        pop_idx = fields.index(POP_FIELD)
        for record, shp in zip(reader.records(), reader.shapes()):
            geoid = str(record[geoid_idx])
            population = int(record[pop_idx] or 0)
            geom_wgs84 = shape_factory(shp.__geo_interface__)
            geom = shapely_transform(to_projected.transform, geom_wgs84)
            if geom.is_empty or geom.area <= 0:
                continue
            centroid = geom.centroid
            blocks.append(
                BlockGroup(
                    geoid=geoid,
                    name=str(record[name_idx]),
                    statefp=str(record[state_idx]),
                    population=max(population, 0),
                    geometry=geom,
                    centroid_x=float(centroid.x),
                    centroid_y=float(centroid.y),
                    area_m2=float(geom.area),
                )
            )
    return blocks


def _download_tiger_tabblock(year: int, state: str, raw_dir: Path) -> Path:
    zip_path = raw_dir / f"tl_{year}_{state}_tabblock20.zip"
    extract_dir = raw_dir / f"tl_{year}_{state}_tabblock20"
    if not zip_path.exists():
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TABBLOCK20/tl_{year}_{state}_tabblock20.zip"
        urlretrieve(url, zip_path)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    return extract_dir


def _download_tiger_bg(year: int, state: str, raw_dir: Path) -> Path:
    zip_path = raw_dir / f"tl_{year}_{state}_bg.zip"
    extract_dir = raw_dir / f"tl_{year}_{state}_bg"
    if not zip_path.exists():
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{state}_bg.zip"
        urlretrieve(url, zip_path)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    return extract_dir


def _recursive_population_split(block_groups: list[BlockGroup], target_regions: int) -> list[list[BlockGroup]]:
    if target_regions <= 0:
        raise ValueError("target_regions must be positive")
    if target_regions == 1:
        return [block_groups]
    if len(block_groups) <= target_regions:
        singles = [[item] for item in block_groups]
        while len(singles) < target_regions:
            singles.append([])
        return singles

    k_left = target_regions // 2
    k_right = target_regions - k_left
    axis = _long_axis(block_groups)
    ordered = sorted(block_groups, key=lambda item: item.centroid_x if axis == "x" else item.centroid_y)
    cut_idx = _population_cut_index(ordered, k_left, k_right)
    return _recursive_population_split(ordered[:cut_idx], k_left) + _recursive_population_split(ordered[cut_idx:], k_right)


def _long_axis(block_groups: list[BlockGroup]) -> str:
    minx = min(item.centroid_x for item in block_groups)
    maxx = max(item.centroid_x for item in block_groups)
    miny = min(item.centroid_y for item in block_groups)
    maxy = max(item.centroid_y for item in block_groups)
    return "x" if (maxx - minx) >= (maxy - miny) else "y"


def _population_cut_index(ordered: list[BlockGroup], k_left: int, k_right: int) -> int:
    total = sum(item.population for item in ordered)
    min_cut = max(1, k_left)
    max_cut = min(len(ordered) - 1, len(ordered) - k_right)
    if min_cut > max_cut:
        return max(1, min(len(ordered) - 1, len(ordered) // 2))
    if total <= 0:
        return max(min_cut, min(max_cut, len(ordered) // 2))

    target = total * k_left / (k_left + k_right)
    running = 0
    best_idx = min_cut
    best_error = float("inf")
    for idx, item in enumerate(ordered, start=1):
        running += item.population
        if idx < min_cut or idx > max_cut:
            continue
        error = abs(running - target)
        if error < best_error:
            best_error = error
            best_idx = idx
    return best_idx


def _write_regions(out_dir: Path, groups, config, unary_union, shapely_transform, to_wgs84, mapping_factory):
    features = []
    simplified_features = []
    records = []
    total_population = sum(item.population for group in groups for item in group)
    target_population = total_population / len(groups)
    for idx, group in enumerate(groups):
        if not group:
            continue
        geom = unary_union([item.geometry for item in group])
        pop = sum(item.population for item in group)
        area = float(geom.area)
        centroid = geom.centroid
        lon, lat = to_wgs84.transform(centroid.x, centroid.y)
        region_id = f"pop_{idx:04d}"
        geoid_list = [item.geoid for item in group]
        record = {
            "region_id": region_id,
            "population": int(pop),
            "target_population": float(target_population),
            "population_ratio": float(pop / target_population) if target_population else None,
            "block_group_count": len(group),
            "area_m2": area,
            "population_density_per_km2": float(pop / (area / 1_000_000.0)) if area else None,
            "centroid_x": float(centroid.x),
            "centroid_y": float(centroid.y),
            "centroid_lon": float(lon),
            "centroid_lat": float(lat),
            "source_geoids": ";".join(geoid_list),
            "geometry_projected": geom,
        }
        records.append(record)
        props = {key: value for key, value in record.items() if key != "geometry_projected"}
        geom_wgs84 = shapely_transform(to_wgs84.transform, geom)
        features.append({"type": "Feature", "properties": props, "geometry": mapping_factory(geom_wgs84)})
        simplified_features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": mapping_factory(geom_wgs84.simplify(0.001, preserve_topology=True)),
            }
        )

    (out_dir / "subregions.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "subregions_simplified.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": simplified_features}, indent=2),
        encoding="utf-8",
    )
    summary_cols = [
        "region_id",
        "population",
        "target_population",
        "population_ratio",
        "block_group_count",
        "area_m2",
        "population_density_per_km2",
        "centroid_lon",
        "centroid_lat",
    ]
    pd.DataFrame([{key: row[key] for key in summary_cols} for row in records]).to_csv(
        out_dir / "region_population_summary.csv",
        index=False,
    )
    return records


def _write_source_receptor_csvs(out_dir: Path, records, config, point_factory, to_wgs84) -> None:
    source_count = int(config.data["grid"]["source_points_per_region"])
    receptor_count = int(config.data["grid"]["receptor_points_per_region"])
    source_rows = []
    receptor_rows = []
    for row in records:
        region_id = row["region_id"]
        geom = row["geometry_projected"]
        for idx, point in enumerate(_sample_points(geom, point_factory, source_count)):
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
        for idx, point in enumerate(_sample_points(geom, point_factory, receptor_count)):
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
                    "target_population": row["population"],
                }
            )
    _write_csv(out_dir / "sources.csv", source_rows)
    _write_csv(out_dir / "receptors.csv", receptor_rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_assignment_csv(out_dir: Path, groups: list[list[BlockGroup]]) -> None:
    rows = []
    for idx, group in enumerate(groups):
        region_id = f"pop_{idx:04d}"
        for item in group:
            rows.append(
                {
                    "region_id": region_id,
                    "geoid": item.geoid,
                    "name": item.name,
                    "statefp": item.statefp,
                    "population": item.population,
                    "area_m2": item.area_m2,
                    "centroid_x": item.centroid_x,
                    "centroid_y": item.centroid_y,
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "census_block_assignments.csv", index=False)


def _write_validation(out_dir: Path, records: list[dict[str, object]], block_group_count: int, year: int) -> dict[str, object]:
    pops = [row["population"] for row in records]
    ratios = [row["population_ratio"] for row in records]
    areas = [row["area_m2"] for row in records]
    target = sum(pops) / len(pops)
    stats = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "census_year": year,
        "target_regions": len(records),
        "source_census_blocks": block_group_count,
        "total_population": int(sum(pops)),
        "target_population_per_region": float(target),
        "population_min": int(min(pops)),
        "population_median": float(pd.Series(pops).median()),
        "population_mean": float(pd.Series(pops).mean()),
        "population_max": int(max(pops)),
        "population_ratio_min": float(min(ratios)),
        "population_ratio_p05": float(pd.Series(ratios).quantile(0.05)),
        "population_ratio_median": float(pd.Series(ratios).median()),
        "population_ratio_p95": float(pd.Series(ratios).quantile(0.95)),
        "population_ratio_max": float(max(ratios)),
        "regions_within_10pct": int(sum(0.9 <= item <= 1.1 for item in ratios)),
        "regions_within_20pct": int(sum(0.8 <= item <= 1.2 for item in ratios)),
        "area_km2_min": float(min(areas) / 1_000_000.0),
        "area_km2_median": float(pd.Series(areas).median() / 1_000_000.0),
        "area_km2_max": float(max(areas) / 1_000_000.0),
        "method": "population_weighted_recursive_kd_split_over_census_tabulation_blocks",
        "notes": [
            "Census tabulation blocks are atomic; no block is split internally.",
            "Regions are geographically local because each recursive split is made along the longer projected axis.",
            "Exact equal population is impossible when whole Census blocks are preserved.",
        ],
    }
    (out_dir / "validation.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _write_report(out_dir: Path, stats: dict[str, object], year: int) -> None:
    lines = [
        "# Equal-Population DC/VA/MD Partition",
        "",
        f"Data source: TIGER/{year} tabulation block boundaries with `POP20` total population.",
        "",
        "Method: population-weighted recursive k-d splitting. Census tabulation blocks are first projected into the",
        "same Lambert Conformal Conic coordinate system used by the transfer-matrix project. The algorithm",
        "recursively splits the current set of blocks along the longer spatial axis at the population",
        "weighted cut that matches the requested child-region count. Whole Census blocks are preserved.",
        "",
        f"Regions: {stats['target_regions']}",
        f"Total population: {stats['total_population']:,}",
        f"Target population per region: {stats['target_population_per_region']:.1f}",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| population min | {stats['population_min']} |",
        f"| population median | {stats['population_median']:.1f} |",
        f"| population max | {stats['population_max']} |",
        f"| ratio p05 | {stats['population_ratio_p05']:.3f} |",
        f"| ratio median | {stats['population_ratio_median']:.3f} |",
        f"| ratio p95 | {stats['population_ratio_p95']:.3f} |",
        f"| within 10 percent | {stats['regions_within_10pct']} |",
        f"| within 20 percent | {stats['regions_within_20pct']} |",
        f"| area km2 min | {stats['area_km2_min']:.3f} |",
        f"| area km2 median | {stats['area_km2_median']:.3f} |",
        f"| area km2 max | {stats['area_km2_max']:.3f} |",
        "",
        "Outputs:",
        "",
        "- `subregions.geojson`: 1000 equal-population regions.",
        "- `subregions_simplified.geojson`: simplified copy for quick visualization.",
        "- `region_population_summary.csv`: region-level population, density, area, and centroid.",
        "- `census_block_assignments.csv`: source Census tabulation-block to region mapping.",
        "- `sources.csv` and `receptors.csv`: source/receptor samples for later transfer-matrix runs.",
        "- `validation.json`: balance and provenance checks.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
