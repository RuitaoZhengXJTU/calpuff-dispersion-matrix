from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from transfer_matrix.config import load_case


CASE_TAG = "20250623_18z"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a pseudo-CALMET meteorology package from hourly region weather."
    )
    parser.add_argument("--case", default="config/case_20250623_18z.yaml")
    parser.add_argument(
        "--partition-dir",
        default="population_partitions/area_capped_30sqmi_population_balanced",
    )
    parser.add_argument(
        "--weather-csv",
        default=(
            "population_partitions/area_capped_30sqmi_population_balanced/"
            "sparse_transfer_matrices_20250623_18z/weather_by_region_hour.csv"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-horizontal-diffusivity", type=float, default=None)
    parser.add_argument("--diffusivity-pbl-wind-factor", type=float, default=None)
    args = parser.parse_args(argv)

    case_path = _resolve(args.case)
    partition_dir = _resolve(args.partition_dir)
    weather_csv = _resolve(args.weather_csv)
    out_dir = _resolve(
        args.output_dir
        or f"{args.partition_dir}/pseudo_meteorology_{CASE_TAG}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_case(case_path)
    fallback = config.data["fallback_model"]
    min_kh = (
        float(args.min_horizontal_diffusivity)
        if args.min_horizontal_diffusivity is not None
        else float(fallback["min_horizontal_diffusivity_m2_s"])
    )
    kh_factor = (
        float(args.diffusivity_pbl_wind_factor)
        if args.diffusivity_pbl_wind_factor is not None
        else float(fallback["diffusivity_pbl_wind_factor"])
    )

    regions = pd.read_csv(partition_dir / "region_area_population_summary.csv")
    regions = regions.sort_values("region_id").reset_index(drop=True)
    weather = pd.read_csv(weather_csv)
    fields = _build_fields(config, regions, weather, min_kh, kh_factor)

    csv_path = out_dir / "pseudo_met_by_region_hour.csv"
    fields.to_csv(csv_path, index=False)

    npz_path = out_dir / f"pseudo_met_fields_{CASE_TAG}.npz"
    np.savez_compressed(
        npz_path,
        region_ids=fields[fields["hour_index"] == 0]["region_id"].astype(str).to_numpy(),
        hours_utc=sorted(fields["time_utc"].unique()),
        wind_u_towards_m_s=_pivot(fields, "wind_u_towards_m_s"),
        wind_v_towards_m_s=_pivot(fields, "wind_v_towards_m_s"),
        wind_speed_m_s=_pivot(fields, "wind_speed_m_s"),
        wind_direction_deg_from=_pivot(fields, "wind_direction_deg_from"),
        boundary_layer_height_m=_pivot(fields, "boundary_layer_height_m"),
        horizontal_diffusivity_m2_s=_pivot(fields, "horizontal_diffusivity_m2_s"),
        vertical_diffusivity_m2_s=_pivot(fields, "vertical_diffusivity_m2_s"),
        sigma_horizontal_1h_m=_pivot(fields, "sigma_horizontal_1h_m"),
        sigma_vertical_1h_m=_pivot(fields, "sigma_vertical_1h_m"),
        temperature_2m_c=_pivot(fields, "temperature_2m_c"),
        relative_humidity_2m_pct=_pivot(fields, "relative_humidity_2m_pct"),
    )

    hourly_summary = (
        fields.groupby("hour_index")
        .agg(
            time_utc=("time_utc", "first"),
            wind_speed_min=("wind_speed_m_s", "min"),
            wind_speed_mean=("wind_speed_m_s", "mean"),
            wind_speed_max=("wind_speed_m_s", "max"),
            pbl_min=("boundary_layer_height_m", "min"),
            pbl_mean=("boundary_layer_height_m", "mean"),
            pbl_max=("boundary_layer_height_m", "max"),
            kh_mean=("horizontal_diffusivity_m2_s", "mean"),
            sigma_h_mean=("sigma_horizontal_1h_m", "mean"),
            sigma_z_mean=("sigma_vertical_1h_m", "mean"),
        )
        .reset_index()
    )
    hourly_summary.to_csv(out_dir / "pseudo_met_hourly_summary.csv", index=False)
    _write_provenance(out_dir, config, partition_dir, weather_csv, min_kh, kh_factor, fields)
    _write_readme(out_dir, csv_path, npz_path)
    print(out_dir)
    return 0


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def _build_fields(
    config,
    regions: pd.DataFrame,
    weather: pd.DataFrame,
    min_kh: float,
    kh_factor: float,
) -> pd.DataFrame:
    dt = float(config.data["fallback_model"]["timestep_seconds"])
    merged = weather.merge(
        regions[
            [
                "region_id",
                "area_m2",
                "population",
                "centroid_x",
                "centroid_y",
                "centroid_lon",
                "centroid_lat",
            ]
        ],
        on="region_id",
        how="left",
        suffixes=("", "_partition"),
    )
    if merged["area_m2"].isna().any():
        missing = merged.loc[merged["area_m2"].isna(), "region_id"].head(5).tolist()
        raise RuntimeError(f"Weather rows contain unknown region_id values: {missing}")

    pbl = merged["boundary_layer_height_m"].astype(float).fillna(800.0).clip(lower=100.0)
    speed = merged["wind_speed_m_s"].astype(float).fillna(0.0).clip(lower=0.0)
    wind_from = merged["wind_direction_deg_from"].astype(float).fillna(0.0)
    wind_rad = np.deg2rad(wind_from)
    wind_u = -speed * np.sin(wind_rad)
    wind_v = -speed * np.cos(wind_rad)

    kh = np.maximum(min_kh, kh_factor * pbl * np.maximum(speed, 0.5))
    kz = np.maximum(1.0, 0.025 * pbl * np.maximum(speed, 0.5))
    eq_radius = np.sqrt(merged["area_m2"].astype(float).to_numpy() / math.pi)
    initial_sigma = 0.65 * eq_radius
    sigma_h = np.sqrt(initial_sigma * initial_sigma + 2.0 * kh * dt)
    sigma_z = np.minimum(pbl.to_numpy(), np.sqrt(20.0 * 20.0 + 2.0 * kz * dt))

    out = merged.copy()
    out["wind_u_towards_m_s"] = wind_u
    out["wind_v_towards_m_s"] = wind_v
    out["advection_dx_1h_m"] = wind_u * dt
    out["advection_dy_1h_m"] = wind_v * dt
    out["horizontal_diffusivity_m2_s"] = kh
    out["vertical_diffusivity_m2_s"] = kz
    out["sigma_horizontal_1h_m"] = sigma_h
    out["sigma_vertical_1h_m"] = sigma_z
    out["stability_class"] = [
        _stability_class(row.time_utc, row.wind_speed_m_s, row.boundary_layer_height_m)
        for row in out.itertuples(index=False)
    ]
    keep = [
        "region_id",
        "hour_index",
        "time_utc",
        "centroid_lon",
        "centroid_lat",
        "centroid_x",
        "centroid_y",
        "area_m2",
        "population",
        "wind_speed_m_s",
        "wind_direction_deg_from",
        "wind_u_towards_m_s",
        "wind_v_towards_m_s",
        "advection_dx_1h_m",
        "advection_dy_1h_m",
        "boundary_layer_height_m",
        "temperature_2m_c",
        "relative_humidity_2m_pct",
        "horizontal_diffusivity_m2_s",
        "vertical_diffusivity_m2_s",
        "sigma_horizontal_1h_m",
        "sigma_vertical_1h_m",
        "stability_class",
    ]
    return out[keep].sort_values(["hour_index", "region_id"]).reset_index(drop=True)


def _stability_class(time_utc: str, wind_speed_m_s: float, pbl_m: float) -> str:
    label = str(time_utc)
    if label.endswith("Z"):
        parsed = datetime.fromisoformat(label.replace("Z", "+00:00"))
    else:
        parsed = datetime.fromisoformat(label)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    local_hour = (parsed - timedelta(hours=4)).hour
    daytime = 7 <= local_hour <= 18
    speed = float(wind_speed_m_s)
    pbl = float(pbl_m) if math.isfinite(float(pbl_m)) else 800.0
    if daytime:
        if pbl >= 1200.0 and speed < 4.0:
            return "B"
        if pbl >= 700.0:
            return "C"
        return "D"
    if speed < 2.0 and pbl < 300.0:
        return "F"
    if speed < 3.5 and pbl < 500.0:
        return "E"
    return "D"


def _pivot(fields: pd.DataFrame, name: str) -> np.ndarray:
    return (
        fields.pivot(index="hour_index", columns="region_id", values=name)
        .sort_index(axis=0)
        .sort_index(axis=1)
        .to_numpy(np.float32)
    )


def _write_provenance(
    out_dir: Path,
    config,
    partition_dir: Path,
    weather_csv: Path,
    min_kh: float,
    kh_factor: float,
    fields: pd.DataFrame,
) -> None:
    stability_counts = fields["stability_class"].value_counts().sort_index().to_dict()
    payload = {
        "case_id": config.case_id,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "purpose": "pseudo_calmet_like_meteorology_for_non_regulatory_transfer_matrix",
        "not_official_calmet_dat": True,
        "time_window_utc": {
            "start": config.data["time"]["start_utc"],
            "hours": config.hours,
        },
        "partition_dir": str(partition_dir.relative_to(ROOT)).replace("\\", "/"),
        "source_weather_csv": str(weather_csv.relative_to(ROOT)).replace("\\", "/"),
        "region_count": int(fields["region_id"].nunique()),
        "row_count": int(len(fields)),
        "derived_parameters": {
            "min_horizontal_diffusivity_m2_s": float(min_kh),
            "diffusivity_pbl_wind_factor": float(kh_factor),
            "vertical_diffusivity_factor": 0.025,
            "initial_sigma_region_radius_factor": 0.65,
            "release_height_m": float(config.data["pollutant"]["release_height_m"]),
            "timestep_seconds": float(config.data["fallback_model"]["timestep_seconds"]),
        },
        "stability_class_counts": stability_counts,
        "official_route_note": (
            "This package is not a CALMET.DAT binary file. It is the meteorology "
            "layer used by the sparse CALPUFF-style emulator when WRF/MMIF/CALMET "
            "inputs are unavailable."
        ),
    }
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_readme(out_dir: Path, csv_path: Path, npz_path: Path) -> None:
    text = f"""# Pseudo Meteorology Package, {CASE_TAG}

This folder contains an engineering approximation to the CALMET meteorology
layer for the DC/VA/MD transfer-matrix case.

It is not an official `CALMET.DAT` file and should not be described as a
regulatory CALPUFF/CALMET run. It is suitable for the current research goal:
weather-conditioned relative pollutant-transfer matrices when WRF/MMIF inputs
are unavailable.

Files:

- `{csv_path.name}`: long table with one row per region-hour.
- `{npz_path.name}`: compact NumPy arrays with shape `(24, n_regions)`.
- `pseudo_met_hourly_summary.csv`: hourly wind, PBL and dispersion summaries.
- `provenance.json`: parameter choices and source file references.

Key derived fields:

- `wind_u_towards_m_s`, `wind_v_towards_m_s`: transport vector in projected
  x/y directions.
- `advection_dx_1h_m`, `advection_dy_1h_m`: one-hour advective displacement.
- `horizontal_diffusivity_m2_s`: turbulence proxy from wind speed and PBL height.
- `sigma_horizontal_1h_m`, `sigma_vertical_1h_m`: one-hour puff spreading scales.
- `stability_class`: simple Pasquill-style class inferred from local hour,
  wind speed and boundary-layer height.

Use this package together with the sparse matrices in
`../sparse_transfer_matrices_20250623_18z/`.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
