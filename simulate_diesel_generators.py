from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point, shape


ROOT = Path(__file__).resolve().parent
CASE_TAG = "20250623_18z"


@dataclass(frozen=True)
class GeneratorSite:
    name: str
    lon: float
    lat: float
    emission_units: float
    note: str


GENERATOR_SITES = [
    GeneratorSite("Ashburn Data Center Alley", -77.4875, 39.0438, 140.0, "largest local data-center cluster"),
    GeneratorSite("Sterling / Dulles Corridor", -77.4291, 39.0062, 95.0, "Northern Virginia data-center corridor"),
    GeneratorSite("Herndon", -77.3861, 38.9696, 70.0, "Dulles Technology Corridor"),
    GeneratorSite("Reston", -77.3570, 38.9586, 65.0, "Dulles Technology Corridor"),
    GeneratorSite("Chantilly", -77.4311, 38.8943, 60.0, "Northern Virginia data-center corridor"),
    GeneratorSite("Manassas", -77.4753, 38.7509, 75.0, "Prince William data-center growth area"),
    GeneratorSite("Tysons", -77.2311, 38.9187, 45.0, "Northern Virginia digital infrastructure"),
    GeneratorSite("Silver Spring", -77.0261, 38.9907, 25.0, "Maryland/DC edge node"),
    GeneratorSite("Rockville", -77.1528, 39.0839, 30.0, "Maryland I-270 corridor node"),
    GeneratorSite("Washington DC core", -77.0369, 38.9072, 20.0, "urban colocation/edge node"),
]


def main() -> int:
    out_dir = ROOT / "outputs" / f"diesel_heatmaps_{CASE_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    features = _load_features(ROOT / "outputs" / "subregions.geojson")
    matrix_payload = np.load(ROOT / "outputs" / f"transfer_matrices_{CASE_TAG}.npz", allow_pickle=True)
    transfer = matrix_payload["T"]
    region_ids = matrix_payload["region_ids"].astype(str).tolist()
    hours_utc = matrix_payload["hours_utc"].astype(str).tolist()

    region_index = {region_id: idx for idx, region_id in enumerate(region_ids)}
    initial, mapped_sites = _build_initial_vector(features, region_index)
    concentrations = _propagate(transfer, initial)

    _write_outputs(out_dir, region_ids, hours_utc, initial, concentrations, mapped_sites)
    frame_paths = _render_frames(out_dir, features, region_index, concentrations, hours_utc, mapped_sites)
    _render_contact_sheet(out_dir, frame_paths)
    _render_gif(out_dir, frame_paths)
    print(out_dir)
    return 0


def _load_features(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload["features"]
    features.sort(key=lambda feature: str(feature["properties"]["region_id"]))
    return features


def _build_initial_vector(features: list[dict[str, object]], region_index: dict[str, int]):
    geoms = [(feature["properties"]["region_id"], shape(feature["geometry"])) for feature in features]
    initial = np.zeros(len(region_index), dtype=float)
    mapped_sites = []

    for site in GENERATOR_SITES:
        point = Point(site.lon, site.lat)
        region_id = None
        for candidate_id, geom in geoms:
            if geom.contains(point) or geom.touches(point):
                region_id = candidate_id
                break
        if region_id is None:
            region_id = min(geoms, key=lambda item: item[1].distance(point))[0]
        initial[region_index[region_id]] += site.emission_units
        mapped_sites.append(
            {
                "name": site.name,
                "lon": site.lon,
                "lat": site.lat,
                "emission_units": site.emission_units,
                "region_id": region_id,
                "note": site.note,
            }
        )
    return initial, mapped_sites


def _propagate(transfer: np.ndarray, initial: np.ndarray) -> np.ndarray:
    concentrations = np.zeros((transfer.shape[0] + 1, transfer.shape[1]), dtype=float)
    concentrations[0] = initial
    current = initial.copy()
    for hour in range(transfer.shape[0]):
        current = transfer[hour] @ current
        concentrations[hour + 1] = current
    return concentrations


def _write_outputs(
    out_dir: Path,
    region_ids: list[str],
    hours_utc: list[str],
    initial: np.ndarray,
    concentrations: np.ndarray,
    mapped_sites: list[dict[str, object]],
) -> None:
    import pandas as pd

    pd.DataFrame(mapped_sites).to_csv(out_dir / "diesel_generator_sites.csv", index=False)
    pd.DataFrame({"region_id": region_ids, "initial_concentration": initial}).to_csv(
        out_dir / "initial_concentration_by_region.csv",
        index=False,
    )
    rows = []
    labels = ["initial"] + [f"after_hour_{i:02d}" for i in range(1, concentrations.shape[0])]
    time_labels = ["2025-06-23T18:00:00+00:00"] + hours_utc
    for step, (label, time_label) in enumerate(zip(labels, time_labels)):
        for region_id, value in zip(region_ids, concentrations[step]):
            rows.append(
                {
                    "step": step,
                    "label": label,
                    "time_utc": time_label,
                    "region_id": region_id,
                    "concentration_units": value,
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "hourly_concentration_by_region.csv", index=False)
    summary = {
        "case_tag": CASE_TAG,
        "generator_count": len(mapped_sites),
        "total_initial_emission_units": float(initial.sum()),
        "max_concentration": float(concentrations.max()),
        "final_total_within_domain_units": float(concentrations[-1].sum()),
        "frames": int(concentrations.shape[0]),
    }
    (out_dir / "simulation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _render_frames(
    out_dir: Path,
    features: list[dict[str, object]],
    region_index: dict[str, int],
    concentrations: np.ndarray,
    hours_utc: list[str],
    mapped_sites: list[dict[str, object]],
) -> list[Path]:
    patches = []
    values_order = []
    for feature in features:
        region_id = feature["properties"]["region_id"]
        geom = shape(feature["geometry"])
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            coords = np.asarray(poly.exterior.coords)
            patches.append(MplPolygon(coords, closed=True))
            values_order.append(region_index[region_id])

    nonzero = concentrations[concentrations > 0]
    vmin = max(float(nonzero.min()) if nonzero.size else 1e-6, 1e-4)
    vmax = float(concentrations.max()) if concentrations.max() > 0 else 1.0
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("inferno")
    xlim = (-83.8, -74.7)
    ylim = (36.3, 40.0)
    frame_paths = []

    for step in range(concentrations.shape[0]):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
        values = np.asarray([concentrations[step, idx] for idx in values_order])
        collection = PatchCollection(
            patches,
            cmap=cmap,
            norm=norm,
            edgecolor=(0.96, 0.96, 0.96, 0.18),
            linewidth=0.20,
        )
        collection.set_array(np.maximum(values, vmin))
        ax.add_collection(collection)
        _plot_generators(ax, mapped_sites)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#f5f7f8")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        _plot_place_labels(ax)

        if step == 0:
            title = "Initial diesel generator emission, 2025-06-23 18Z"
        else:
            title = f"Diesel pollution concentration after hour {step:02d}, {hours_utc[step - 1]}"
        ax.set_title(title, fontsize=12, pad=10)
        cbar = fig.colorbar(collection, ax=ax, fraction=0.028, pad=0.02)
        cbar.set_label("relative concentration units")
        fig.text(
            0.015,
            0.015,
            "DC/VA/MD subregions. Light lines show subregion boundaries; colors use shared log scale.",
            fontsize=8,
            color="#555555",
        )
        out = out_dir / f"heatmap_step_{step:02d}.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        frame_paths.append(out)
    return frame_paths


def _plot_generators(ax, mapped_sites: list[dict[str, object]]) -> None:
    lons = [float(site["lon"]) for site in mapped_sites]
    lats = [float(site["lat"]) for site in mapped_sites]
    sizes = [20 + float(site["emission_units"]) * 0.45 for site in mapped_sites]
    ax.scatter(lons, lats, s=sizes, marker="^", c="#00d1ff", edgecolors="#111111", linewidths=0.45, zorder=5)


def _plot_place_labels(ax) -> None:
    label_style = {
        "fontsize": 9,
        "color": "#d8d8d8",
        "alpha": 0.85,
        "ha": "center",
        "va": "center",
        "weight": "bold",
    }
    ax.text(-78.9, 37.55, "Virginia", **label_style)
    ax.text(-76.7, 39.18, "Maryland", **label_style)
    ax.text(-77.04, 38.88, "DC", fontsize=7, color="#e6e6e6", alpha=0.9, ha="center", va="center", weight="bold")


def _render_contact_sheet(out_dir: Path, frame_paths: list[Path]) -> None:
    images = [imageio.imread(path) for path in frame_paths[1:25]]
    rows, cols = 4, 6
    h, w = images[0].shape[:2]
    sheet = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for idx, image in enumerate(images):
        r = idx // cols
        c = idx % cols
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w, :3] = image[:, :, :3]
    imageio.imwrite(out_dir / "diesel_heatmap_24_hour_contact_sheet.png", sheet)


def _render_gif(out_dir: Path, frame_paths: list[Path]) -> None:
    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(out_dir / "diesel_heatmap_animation.gif", images, duration=0.55)


if __name__ == "__main__":
    raise SystemExit(main())
