from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Polygon as MplPolygon, Rectangle
from shapely.geometry import shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[2]
FONT_FAMILY = "Times New Roman"
DETAIL_EXTENT = (-77.90, -76.95, 38.60, 39.20)

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.serif": [FONT_FAMILY],
        "font.size": 14,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the current region partition map without heatmap coloring.")
    parser.add_argument(
        "--partition-dir",
        default="population_partitions/area_capped_30sqmi_population_balanced",
    )
    parser.add_argument(
        "--output",
        default=(
            "population_partitions/area_capped_30sqmi_population_balanced/"
            "area_capped_30sqmi_partition_map.png"
        ),
    )
    parser.add_argument(
        "--data-centers",
        default="data/examples/dc_md_va_20250623/generators.csv",
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    partition_dir = _resolve(args.partition_dir)
    output = _resolve(args.output)
    data_centers_path = _resolve(args.data_centers)
    geojson = partition_dir / "subregions_simplified.geojson"
    if not geojson.exists():
        geojson = partition_dir / "subregions.geojson"
    payload = json.loads(geojson.read_text(encoding="utf-8"))
    geoms = [shape(feature["geometry"]) for feature in payload["features"]]
    fills, interior_lines = _geometry_artists(geoms)
    outline_lines = _outline_artists(unary_union(geoms))
    data_centers = _read_data_centers(data_centers_path)
    hosting_region_ids = set(data_centers.get("region_id", pd.Series(dtype=str)).dropna().astype(str))

    fig = plt.figure(figsize=(16.2, 8.4), dpi=args.dpi)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.28, 0.72], wspace=0.03)
    ax = fig.add_subplot(grid[0, 0])
    detail_ax = fig.add_subplot(grid[0, 1])
    fig.patch.set_facecolor("#f8f7f2")
    ax.set_facecolor("#eef2f3")
    detail_ax.set_facecolor("#eef2f3")

    fill_collection = PatchCollection(
        fills,
        facecolor="#f5efe4",
        edgecolor="none",
        linewidth=0,
        zorder=1,
    )
    ax.add_collection(fill_collection)

    boundary_collection = LineCollection(
        interior_lines,
        colors="#858580",
        linewidths=0.34,
        zorder=2,
    )
    ax.add_collection(boundary_collection)

    outline_collection = LineCollection(
        outline_lines,
        colors="#4a4a44",
        linewidths=0.95,
        zorder=3,
    )
    ax.add_collection(outline_collection)

    hosted_fills, hosted_lines = _host_region_artists(payload["features"], hosting_region_ids)
    ax.add_collection(
        PatchCollection(
            hosted_fills,
            facecolor="#f2bd4a",
            edgecolor="#bd6f18",
            linewidth=0.45,
            alpha=0.30,
            zorder=3.4,
        )
    )
    ax.add_collection(
        LineCollection(hosted_lines, colors="#bd6f18", linewidths=0.5, alpha=0.75, zorder=3.5)
    )

    _plot_data_centers(ax, data_centers)
    _add_title(fig)
    _add_legend(fig)
    _draw_detail_map(detail_ax, payload["features"], data_centers, hosting_region_ids)
    _add_detail_link(fig, ax, detail_ax, DETAIL_EXTENT)

    ax.set_xlim(-83.85, -74.65)
    ax.set_ylim(36.25, 40.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.012, right=0.992, top=0.90, bottom=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(output)
    return 0


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def _geometry_artists(geoms) -> tuple[list[MplPolygon], list[np.ndarray]]:
    fills: list[MplPolygon] = []
    lines: list[np.ndarray] = []
    for geom in geoms:
        polygons = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polygons:
            if poly.is_empty:
                continue
            exterior = np.asarray(poly.exterior.coords)
            fills.append(MplPolygon(exterior, closed=True))
            lines.append(exterior)
            for ring in poly.interiors:
                lines.append(np.asarray(ring.coords))
    return fills, lines


def _outline_artists(geom) -> list[np.ndarray]:
    lines: list[np.ndarray] = []
    polygons = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polygons:
        if poly.is_empty:
            continue
        lines.append(np.asarray(poly.exterior.coords))
    return lines


def _host_region_artists(features: list[dict], hosting_region_ids: set[str]) -> tuple[list[MplPolygon], list[np.ndarray]]:
    geoms = [shape(feature["geometry"]) for feature in features if feature["properties"]["region_id"] in hosting_region_ids]
    return _geometry_artists(geoms)


def _read_data_centers(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["lon", "lat"])
    data = pd.read_csv(path)
    if "lon" not in data.columns or "lat" not in data.columns:
        raise RuntimeError(f"Data-center file must contain lon and lat columns: {path}")
    data["lon"] = pd.to_numeric(data["lon"], errors="coerce")
    data["lat"] = pd.to_numeric(data["lat"], errors="coerce")
    data = data.dropna(subset=["lon", "lat"]).copy()
    return data[
        data["lon"].between(-83.85, -74.65)
        & data["lat"].between(36.25, 40.05)
    ].reset_index(drop=True)


def _plot_data_centers(ax, data_centers: pd.DataFrame) -> None:
    if data_centers.empty:
        return
    ax.scatter(
        data_centers["lon"],
        data_centers["lat"],
        s=22,
        marker="o",
        facecolor="#9fd3e6",
        edgecolor="#f9fbfb",
        linewidth=0.55,
        alpha=0.92,
        zorder=4,
    )


def _add_place_labels(ax) -> None:
    labels = [
        ("Maryland", -76.85, 39.12),
        ("Virginia", -78.55, 37.60),
        ("DC", -77.04, 38.90),
        ("Chesapeake Bay", -76.12, 38.30),
        ("Atlantic Coast", -75.35, 37.35),
    ]
    for text, x, y in labels:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=9 if text != "DC" else 8,
            color="#595851",
            zorder=4,
        )


def _add_title(fig) -> None:
    title = "Area-Capped Population-Aware Partition"
    fig.text(
        0.5,
        0.975,
        title,
        ha="center",
        va="top",
        fontsize=28,
        color="#22221f",
        fontweight="bold",
        zorder=5,
    )


def _add_legend(fig) -> None:
    handles = [
        Line2D([0], [0], color="#858580", lw=3.6, label="Subregion boundary"),
        Line2D([0], [0], color="#bd6f18", lw=6.2, alpha=0.80, label="Generator-hosting subregion"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=13.5,
            markerfacecolor="#9fd3e6",
            markeredgecolor="#f9fbfb",
            markeredgewidth=0.65,
            label="Data center site",
        ),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=True,
        framealpha=0.86,
        facecolor="#f8f7f2",
        edgecolor="#d9d6cc",
        fontsize=20,
        handlelength=3.4,
        borderpad=0.75,
        columnspacing=2.7,
    )
    legend.set_zorder(6)


def _draw_detail_map(ax, features: list[dict], data_centers: pd.DataFrame, hosting_region_ids: set[str]) -> None:
    """Render the Northern Virginia/DC cluster as the dominant panel."""
    geoms = [shape(feature["geometry"]) for feature in features]
    fills, lines = _geometry_artists(geoms)
    ax.add_collection(PatchCollection(fills, facecolor="#f5efe4", edgecolor="none", zorder=1))
    ax.add_collection(LineCollection(lines, colors="#777772", linewidths=0.72, zorder=2))
    hosted_fills, hosted_lines = _host_region_artists(features, hosting_region_ids)
    ax.add_collection(
        PatchCollection(hosted_fills, facecolor="#f2bd4a", edgecolor="#bd6f18", linewidth=1.15, alpha=0.38, zorder=3)
    )
    ax.add_collection(LineCollection(hosted_lines, colors="#bd6f18", linewidths=1.25, alpha=0.95, zorder=4))
    if not data_centers.empty:
        ax.scatter(
            data_centers["lon"], data_centers["lat"], s=38, marker="o",
            facecolor="#54b8df", edgecolor="white", linewidth=0.80, alpha=0.95, zorder=5,
        )
    xmin, xmax, ymin, ymax = DETAIL_EXTENT
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a4a44")
        spine.set_linewidth(0.9)


def _add_detail_link(fig, overview_ax, detail_ax, extent: tuple[float, float, float, float]) -> None:
    """Mark the overview extent and connect it to the enlarged panel."""
    xmin, xmax, ymin, ymax = extent
    locator = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor="#527c91",
        linewidth=1.05,
        zorder=5.5,
    )
    overview_ax.add_patch(locator)

    # The two diagonal guides map the overview rectangle's right edge to the
    # left edge of the enlarged panel, so the spatial relationship stays clear.
    for start, end in [((xmax, ymax), (xmin, ymax)), ((xmax, ymin), (xmin, ymin))]:
        fig.add_artist(
            ConnectionPatch(
                xyA=start,
                xyB=end,
                coordsA="data",
                coordsB="data",
                axesA=overview_ax,
                axesB=detail_ax,
                color="#527c91",
                linewidth=0.85,
                alpha=0.82,
                zorder=4.8,
                clip_on=False,
            )
        )


def _add_scale_bar(ax) -> None:
    y = 36.42
    x0 = -83.25
    miles = 100
    deg_lon = miles / (69.0 * np.cos(np.deg2rad(38.0)))
    ax.plot([x0, x0 + deg_lon], [y, y], color="#383833", lw=1.4, zorder=5)
    ax.plot([x0, x0], [y - 0.025, y + 0.025], color="#383833", lw=1.0, zorder=5)
    ax.plot([x0 + deg_lon, x0 + deg_lon], [y - 0.025, y + 0.025], color="#383833", lw=1.0, zorder=5)
    ax.text(x0 + deg_lon / 2, y + 0.055, "100 miles", ha="center", va="bottom", fontsize=8, color="#383833")


if __name__ == "__main__":
    raise SystemExit(main())
