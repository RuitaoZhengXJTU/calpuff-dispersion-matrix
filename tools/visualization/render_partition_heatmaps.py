from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.visualization.simulate_diesel_generators import (
    GENERATOR_SITES,
    _build_initial_vector,
    _load_features,
    _plot_generators,
    _plot_place_labels,
    _propagate,
    _render_contact_sheet,
    _render_gif,
    _write_outputs,
)


CASE_TAG = "20250623_18z"
METHODS = ["rectangular", "administrative", "hexagonal", "adaptive_grid", "hybrid"]
METHOD_LABELS = {
    "rectangular": "Equal-area rectangular",
    "administrative": "Administrative merged county/city",
    "hexagonal": "Hexagonal clipped grid",
    "adaptive_grid": "Source-weighted adaptive grid",
    "hybrid": "Data-center/city hybrid Voronoi",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render diesel-emission heatmaps for partition-comparison matrices.")
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)

    comparison_root = ROOT / "partition_comparison"
    runs = {}
    for method in args.methods:
        method_dir = comparison_root / method
        runs[method] = _load_and_propagate(method, method_dir)

    global_nonzero = np.concatenate([run["concentrations"][run["concentrations"] > 0] for run in runs.values()])
    vmin = max(float(global_nonzero.min()) if global_nonzero.size else 1e-6, 1e-4)
    vmax = max(float(run["concentrations"].max()) for run in runs.values())

    summary_rows = []
    for method, run in runs.items():
        method_dir = comparison_root / method
        out_dir = method_dir / "diesel_heatmaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_outputs(
            out_dir,
            run["region_ids"],
            run["hours_utc"],
            run["initial"],
            run["concentrations"],
            run["mapped_sites"],
        )
        frame_paths = _render_frames(
            method=method,
            out_dir=out_dir,
            features=run["features"],
            region_index=run["region_index"],
            concentrations=run["concentrations"],
            hours_utc=run["hours_utc"],
            mapped_sites=run["mapped_sites"],
            vmin=vmin,
            vmax=vmax,
            dpi=args.dpi,
        )
        _render_contact_sheet(out_dir, frame_paths)
        _render_gif(out_dir, frame_paths)
        summary_rows.append(
            {
                "method": method,
                "heatmap_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
                "frames": len(frame_paths),
                "initial_total_units": float(run["initial"].sum()),
                "final_total_units": float(run["concentrations"][-1].sum()),
                "max_concentration": float(run["concentrations"].max()),
                "nonzero_region_steps": int(np.count_nonzero(run["concentrations"])),
                "color_vmin": vmin,
                "color_vmax": vmax,
            }
        )
        print(out_dir)

    summary_path = comparison_root / f"diesel_heatmap_summary_{CASE_TAG}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    _write_report(comparison_root, summary_rows, vmin, vmax)
    print(f"Wrote {summary_path}")
    return 0


def _load_and_propagate(method: str, method_dir: Path) -> dict[str, object]:
    matrix_path = method_dir / f"transfer_matrices_{CASE_TAG}_{method}.npz"
    subregions_path = method_dir / "subregions.geojson"
    if not matrix_path.exists():
        raise RuntimeError(f"Missing matrix file: {matrix_path}")
    if not subregions_path.exists():
        raise RuntimeError(f"Missing subregions file: {subregions_path}")

    features = _load_features(subregions_path)
    matrix_payload = np.load(matrix_path, allow_pickle=True)
    transfer = matrix_payload["T"]
    region_ids = matrix_payload["region_ids"].astype(str).tolist()
    hours_utc = matrix_payload["hours_utc"].astype(str).tolist()
    region_index = {region_id: idx for idx, region_id in enumerate(region_ids)}
    initial, mapped_sites = _build_initial_vector(features, region_index)
    concentrations = _propagate(transfer, initial)
    return {
        "features": features,
        "transfer": transfer,
        "region_ids": region_ids,
        "hours_utc": hours_utc,
        "region_index": region_index,
        "initial": initial,
        "mapped_sites": mapped_sites,
        "concentrations": concentrations,
    }


def _render_frames(
    method: str,
    out_dir: Path,
    features: list[dict[str, object]],
    region_index: dict[str, int],
    concentrations: np.ndarray,
    hours_utc: list[str],
    mapped_sites: list[dict[str, object]],
    vmin: float,
    vmax: float,
    dpi: int,
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

    norm = colors.LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10.0))
    cmap = plt.get_cmap("inferno")
    xlim = (-83.8, -74.7)
    ylim = (36.3, 40.0)
    frame_paths = []
    method_label = METHOD_LABELS.get(method, method)

    for step in range(concentrations.shape[0]):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
        values = np.asarray([concentrations[step, idx] for idx in values_order])
        collection = PatchCollection(
            patches,
            cmap=cmap,
            norm=norm,
            edgecolor=(0.96, 0.96, 0.96, 0.18),
            linewidth=0.18,
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
            title = f"{method_label}: initial diesel generator emission, 2025-06-23 18Z"
        else:
            title = f"{method_label}: concentration after hour {step:02d}, {hours_utc[step - 1]}"
        ax.set_title(title, fontsize=11, pad=10)
        cbar = fig.colorbar(collection, ax=ax, fraction=0.028, pad=0.02)
        cbar.set_label("relative concentration units")
        fig.text(
            0.015,
            0.015,
            "Shared log color scale across all five partitions. Light lines show subregion boundaries.",
            fontsize=8,
            color="#555555",
        )
        out = out_dir / f"heatmap_step_{step:02d}.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        frame_paths.append(out)
    return frame_paths


def _write_report(comparison_root: Path, summary_rows: list[dict[str, object]], vmin: float, vmax: float) -> None:
    lines = [
        "# Diesel Heatmaps By Partition",
        "",
        f"Case window: {CASE_TAG}. Initial sources are the same data-center diesel generator sites used by",
        "`simulate_diesel_generators.py`; each partition maps those sites into its own regions and propagates",
        "the concentration vector through its own 24 hourly transfer matrices.",
        "",
        f"All rendered maps use a shared log color scale: vmin={vmin:.6g}, vmax={vmax:.6g}.",
        "",
        "| method | frames | initial total | final total | max concentration | output |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['frames']} | {row['initial_total_units']:.3f} | "
            f"{row['final_total_units']:.3f} | {row['max_concentration']:.3f} | `{row['heatmap_dir']}` |"
        )
    lines.append("")
    lines.append("Each output folder contains 25 PNG frames, one 24-hour contact sheet, one GIF,")
    lines.append("the mapped generator sites, and hourly region-level concentration CSVs.")
    (comparison_root / f"diesel_heatmap_report_{CASE_TAG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
