from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import box, mapping
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "archive" / "legacy_emulator"))

from build_population_equal_partition import DEFAULT_STATES, _load_tabulation_blocks
from transfer_matrix.config import load_case
from transfer_matrix.grid import _projection_string, _require_geo_stack


SQ_MILE_M2 = 1609.344**2


@dataclass
class Cell:
    region_id: str
    geometry: object
    estimated_population: float = 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a population-aware partition with every region capped at a maximum area."
    )
    parser.add_argument("--case", default="config/case_20250623_18z.yaml")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--states", nargs="*", default=DEFAULT_STATES)
    parser.add_argument("--max-area-sqmi", type=float, default=1.0)
    parser.add_argument("--target-population", type=float, default=None)
    parser.add_argument("--split-threshold-factor", type=float, default=1.25)
    parser.add_argument("--min-area-sqmi", type=float, default=0.003)
    parser.add_argument("--exact-refinement-passes", type=int, default=2)
    parser.add_argument("--output-dir", default="population_partitions/area_capped_1sqmi_population_balanced")
    args = parser.parse_args(argv)

    case_path = Path(args.case)
    if not case_path.is_absolute():
        case_path = ROOT / case_path
    config = load_case(case_path)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    shapefile, Transformer, factories, unary_union, shapely_transform = _require_geo_stack()
    _, _, mapping_factory, shape_factory = factories
    to_projected = Transformer.from_crs("EPSG:4269", _projection_string(config), always_xy=True)
    to_wgs84 = Transformer.from_crs(_projection_string(config), "EPSG:4326", always_xy=True)

    print("Loading Census tabulation blocks")
    blocks = _load_tabulation_blocks(
        year=args.year,
        states=args.states,
        shapefile=shapefile,
        shape_factory=shape_factory,
        shapely_transform=shapely_transform,
        to_projected=to_projected,
    )
    total_population = sum(block.population for block in blocks)

    print("Building state-domain union")
    domain = _load_state_domain(config, args.states, shapefile, shape_factory, shapely_transform, to_projected, unary_union)
    max_area_m2 = args.max_area_sqmi * SQ_MILE_M2
    min_area_m2 = args.min_area_sqmi * SQ_MILE_M2
    lower_bound_regions = math.ceil(domain.area / max_area_m2)
    target_population = args.target_population or (total_population / lower_bound_regions)

    print(f"Domain area lower bound: {lower_bound_regions:,} regions; target population ~= {target_population:.2f}")
    print(f"Generating {args.max_area_sqmi:g}-square-mile base cells")
    base_cells = _make_base_cells(domain, max_area_m2)
    print(f"Base cells: {len(base_cells):,}")

    print("Allocating population to base cells")
    base_pop = _allocate_population(blocks, base_cells)
    for idx, value in enumerate(base_pop):
        base_cells[idx].estimated_population = value

    print("Splitting high-population cells")
    final_cells = _split_cells(
        base_cells,
        target_population=target_population,
        threshold_factor=args.split_threshold_factor,
        min_area_m2=min_area_m2,
    )
    print(f"Final candidate regions: {len(final_cells):,}")

    print("Reallocating population to final cells")
    final_pop = _allocate_population(blocks, final_cells)
    for pass_idx in range(args.exact_refinement_passes):
        before = len(final_cells)
        for cell, pop in zip(final_cells, final_pop):
            cell.estimated_population = float(pop)
        refined_cells = _split_cells(
            final_cells,
            target_population=target_population,
            threshold_factor=args.split_threshold_factor,
            min_area_m2=min_area_m2,
        )
        if len(refined_cells) == before:
            break
        print(f"Exact refinement pass {pass_idx + 1}: {before:,} -> {len(refined_cells):,} cells")
        final_cells = refined_cells
        final_pop = _allocate_population(blocks, final_cells)
    records = _write_outputs(
        out_dir=out_dir,
        cells=final_cells,
        populations=final_pop,
        blocks=blocks,
        config=config,
        to_wgs84=to_wgs84,
        shapely_transform=shapely_transform,
        mapping_factory=mapping_factory,
        max_area_sqmi=args.max_area_sqmi,
        min_area_sqmi=args.min_area_sqmi,
        target_population=target_population,
        year=args.year,
        lower_bound_regions=lower_bound_regions,
    )
    _write_preview(out_dir, records)
    print(out_dir)
    return 0


def _load_state_domain(config, states, shapefile, shape_factory, shapely_transform, to_projected, unary_union):
    shp_path = ROOT / "data" / "raw" / "census" / "tl_2025_us_state" / "tl_2025_us_state.shp"
    if not shp_path.exists():
        raise RuntimeError(f"Missing state shapefile: {shp_path}. Run the original grid builder first.")
    reader = shapefile.Reader(str(shp_path), encoding="latin1")
    fields = [field[0] for field in reader.fields[1:]]
    state_idx = fields.index("STATEFP")
    wanted = set(states)
    geoms = []
    for record, shp in zip(reader.records(), reader.shapes()):
        if str(record[state_idx]) not in wanted:
            continue
        geom = shape_factory(shp.__geo_interface__)
        geoms.append(shapely_transform(to_projected.transform, geom))
    if len(geoms) != len(wanted):
        raise RuntimeError(f"Expected {sorted(wanted)} state geometries, found {len(geoms)}")
    return unary_union(geoms)


def _make_base_cells(domain, max_area_m2: float) -> list[Cell]:
    side = math.sqrt(max_area_m2)
    minx, miny, maxx, maxy = domain.bounds
    cells: list[Cell] = []
    row = 0
    y = miny
    while y < maxy:
        col = 0
        x = minx
        while x < maxx:
            clipped = box(x, y, min(x + side, maxx), min(y + side, maxy)).intersection(domain)
            if not clipped.is_empty and clipped.area > 1.0:
                cells.append(Cell(region_id=f"base_{len(cells):06d}", geometry=clipped))
            x += side
            col += 1
        y += side
        row += 1
    return cells


def _allocate_population(blocks, cells: list[Cell]) -> np.ndarray:
    geoms = [cell.geometry for cell in cells]
    tree = STRtree(geoms)
    population = np.zeros(len(cells), dtype=float)
    for block_idx, block in enumerate(blocks):
        if block.population <= 0 or block.area_m2 <= 0:
            continue
        hits = tree.query(block.geometry)
        if len(hits) == 0:
            continue
        block_area = block.area_m2
        for cell_idx in hits:
            cell_idx = int(cell_idx)
            inter_area = block.geometry.intersection(geoms[cell_idx]).area
            if inter_area > 0:
                population[cell_idx] += block.population * inter_area / block_area
        if block_idx and block_idx % 50000 == 0:
            print(f"  allocated {block_idx:,} / {len(blocks):,} blocks")
    return population


def _split_cells(
    cells: list[Cell],
    target_population: float,
    threshold_factor: float,
    min_area_m2: float,
) -> list[Cell]:
    threshold = target_population * threshold_factor
    final: list[Cell] = []
    stack = list(cells)
    while stack:
        cell = stack.pop()
        area = cell.geometry.area
        if cell.estimated_population <= threshold or area <= min_area_m2:
            final.append(cell)
            continue
        minx, miny, maxx, maxy = cell.geometry.bounds
        midx = (minx + maxx) / 2.0
        midy = (miny + maxy) / 2.0
        quadrants = [
            box(minx, miny, midx, midy),
            box(midx, miny, maxx, midy),
            box(minx, midy, midx, maxy),
            box(midx, midy, maxx, maxy),
        ]
        made_child = False
        for quad in quadrants:
            geom = cell.geometry.intersection(quad)
            if geom.is_empty or geom.area <= 1.0:
                continue
            made_child = True
            child = Cell(region_id="", geometry=geom, estimated_population=cell.estimated_population * geom.area / area)
            stack.append(child)
        if not made_child:
            final.append(cell)
    final.sort(key=lambda item: (item.geometry.centroid.y, item.geometry.centroid.x))
    for idx, cell in enumerate(final):
        cell.region_id = f"area_pop_{idx:06d}"
    return final


def _write_outputs(
    out_dir: Path,
    cells: list[Cell],
    populations: np.ndarray,
    blocks,
    config,
    to_wgs84,
    shapely_transform,
    mapping_factory,
    max_area_sqmi: float,
    min_area_sqmi: float,
    target_population: float,
    year: int,
    lower_bound_regions: int,
) -> list[dict[str, object]]:
    features = []
    simplified_features = []
    rows = []
    for idx, (cell, pop) in enumerate(zip(cells, populations)):
        geom = cell.geometry
        centroid = geom.centroid
        lon, lat = to_wgs84.transform(centroid.x, centroid.y)
        area_sqmi = geom.area / SQ_MILE_M2
        props = {
            "region_id": cell.region_id,
            "population": float(pop),
            "population_rounded": int(round(float(pop))),
            "target_population": float(target_population),
            "population_ratio": float(pop / target_population) if target_population else None,
            "area_m2": float(geom.area),
            "area_sqmi": float(area_sqmi),
            "population_density_per_sqmi": float(pop / area_sqmi) if area_sqmi > 0 else None,
            "centroid_x": float(centroid.x),
            "centroid_y": float(centroid.y),
            "centroid_lon": float(lon),
            "centroid_lat": float(lat),
        }
        rows.append(props)
        geom_wgs84 = shapely_transform(to_wgs84.transform, geom)
        features.append({"type": "Feature", "properties": props, "geometry": mapping_factory(geom_wgs84)})
        simplified_features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": mapping_factory(geom_wgs84.simplify(0.00035, preserve_topology=True)),
            }
        )

    _write_feature_collection(out_dir / "subregions.geojson", features)
    _write_feature_collection(out_dir / "subregions_simplified.geojson", simplified_features)
    pd.DataFrame(rows).to_csv(out_dir / "region_area_population_summary.csv", index=False)
    _write_validation(
        out_dir=out_dir,
        rows=rows,
        blocks=blocks,
        year=year,
        max_area_sqmi=max_area_sqmi,
        min_area_sqmi=min_area_sqmi,
        target_population=target_population,
        lower_bound_regions=lower_bound_regions,
    )
    _write_report(out_dir, rows, year, max_area_sqmi, min_area_sqmi, target_population, lower_bound_regions)
    return rows


def _write_feature_collection(path: Path, features: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")), encoding="utf-8")


def _write_validation(
    out_dir: Path,
    rows: list[dict[str, object]],
    blocks,
    year: int,
    max_area_sqmi: float,
    min_area_sqmi: float,
    target_population: float,
    lower_bound_regions: int,
) -> None:
    frame = pd.DataFrame(rows)
    populated = frame[frame["population"] > 0].copy()
    validation = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "census_year": year,
        "method": "area_capped_grid_with_recursive_high_population_quadtree_refinement",
        "max_area_sqmi": max_area_sqmi,
        "min_area_sqmi_for_refinement": min_area_sqmi,
        "max_area_m2": max_area_sqmi * SQ_MILE_M2,
        "region_count": int(len(frame)),
        "one_sqmi_lower_bound_region_count": int(lower_bound_regions),
        "source_census_blocks": int(len(blocks)),
        "total_population_allocated": float(frame["population"].sum()),
        "source_total_population": int(sum(block.population for block in blocks)),
        "target_population": float(target_population),
        "area_sqmi_min": float(frame["area_sqmi"].min()),
        "area_sqmi_median": float(frame["area_sqmi"].median()),
        "area_sqmi_max": float(frame["area_sqmi"].max()),
        "area_cap_ok": bool(frame["area_sqmi"].max() <= max_area_sqmi + 1e-6),
        "population_all_regions_min": float(frame["population"].min()),
        "population_all_regions_median": float(frame["population"].median()),
        "population_all_regions_p95": float(frame["population"].quantile(0.95)),
        "population_all_regions_max": float(frame["population"].max()),
        "populated_region_count": int(len(populated)),
        "zero_population_region_count": int((frame["population"] <= 1e-9).sum()),
        "population_populated_median": float(populated["population"].median()) if len(populated) else None,
        "population_populated_p95": float(populated["population"].quantile(0.95)) if len(populated) else None,
        "population_populated_max": float(populated["population"].max()) if len(populated) else None,
        "regions_above_split_threshold_after_exact_allocation": int((frame["population"] > target_population * 1.25).sum()),
        "notes": [
            "All final geometries are clipped to DC/MD/VA and capped at the requested maximum area.",
            "Population is allocated from 2020 Census tabulation blocks to final regions by intersected area share.",
            "Rural and water-dominated areas can have low or zero population; max-area constraints make exact equal population impossible there.",
            "High-population one-square-mile cells are recursively refined into smaller cells to improve population similarity.",
        ],
    }
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")


def _write_report(
    out_dir: Path,
    rows: list[dict[str, object]],
    year: int,
    max_area_sqmi: float,
    min_area_sqmi: float,
    target_population: float,
    lower_bound_regions: int,
) -> None:
    frame = pd.DataFrame(rows)
    validation = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
    lines = [
        "# Area-Capped Population-Aware Partition",
        "",
        f"Data source: TIGER/{year} Census tabulation blocks with `POP20` total population.",
        "",
        f"Constraint: every region area is at most {max_area_sqmi:.3f} square mile.",
        "",
        "Method: create a one-square-mile projected grid over DC/MD/VA, clip it to the state union,",
        "allocate Census-block population by area intersection, then recursively subdivide grid cells",
        "whose estimated population is above the target threshold. The population target is the all-domain",
        "average implied by the area cap, not a fixed requested region count.",
        "",
        f"One-square-mile lower-bound region count: {lower_bound_regions:,}",
        f"Final region count: {len(frame):,}",
        f"Target population: {target_population:.2f}",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| max area sqmi | {validation['area_sqmi_max']:.6f} |",
        f"| area cap ok | {validation['area_cap_ok']} |",
        f"| zero-population regions | {validation['zero_population_region_count']:,} |",
        f"| all-region population median | {validation['population_all_regions_median']:.3f} |",
        f"| all-region population p95 | {validation['population_all_regions_p95']:.3f} |",
        f"| all-region population max | {validation['population_all_regions_max']:.3f} |",
        f"| populated-region median | {validation['population_populated_median']:.3f} |",
        f"| populated-region p95 | {validation['population_populated_p95']:.3f} |",
        f"| populated-region max | {validation['population_populated_max']:.3f} |",
        f"| regions above 1.25x target after exact allocation | {validation['regions_above_split_threshold_after_exact_allocation']:,} |",
        "",
        "Outputs:",
        "",
        "- `subregions.geojson`: full final regions.",
        "- `subregions_simplified.geojson`: lighter geometry for previews.",
        "- `region_area_population_summary.csv`: area, allocated population, density, and centroids.",
        "- `validation.json`: quality checks and assumptions.",
        "- `area_capped_population_preview.png`: quick visual check.",
        "",
        "Important limitation: equal population and max-area constraints conflict in low-density rural/water areas.",
        "Those areas cannot reach the target population without exceeding the one-square-mile cap.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preview(out_dir: Path, rows: list[dict[str, object]]) -> None:
    payload = json.loads((out_dir / "subregions_simplified.geojson").read_text(encoding="utf-8"))
    validation = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
    patches = []
    values = []
    for feature in payload["features"]:
        geom = _shape(feature["geometry"])
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            if poly.is_empty:
                continue
            patches.append(MplPolygon(np.asarray(poly.exterior.coords), closed=True))
            values.append(float(feature["properties"]["population"]))

    fig, ax = plt.subplots(figsize=(12, 8), dpi=170)
    nonzero = [value for value in values if value > 0]
    vmax = np.percentile(nonzero, 99) if nonzero else 1.0
    collection = PatchCollection(
        patches,
        cmap="magma",
        edgecolor=(1, 1, 1, 0.10),
        linewidth=0.025,
    )
    collection.set_array(np.minimum(np.asarray(values, dtype=float), vmax))
    collection.set_clim(0, vmax)
    ax.add_collection(collection)
    ax.set_xlim(-83.8, -74.7)
    ax.set_ylim(36.3, 40.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"DC/VA/MD area-capped population-aware partition, max area <= {validation['max_area_sqmi']:g} sq mi",
        fontsize=12,
    )
    cbar = fig.colorbar(collection, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("allocated population, clipped at 99th percentile")
    fig.tight_layout()
    fig.savefig(out_dir / "area_capped_population_preview.png")
    plt.close(fig)


def _shape(geometry: dict[str, object]):
    from shapely.geometry import shape

    return shape(geometry)


if __name__ == "__main__":
    raise SystemExit(main())
