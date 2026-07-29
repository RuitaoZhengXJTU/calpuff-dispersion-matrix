from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CaseConfig
from .diagnostics import write_diagnostics


def compute_advection_diffusion_matrices(config: CaseConfig) -> Path:
    """Compute a documented fallback transfer tensor from forecast wind fields."""

    sources = pd.read_csv(config.output_path("sources_csv"))
    receptors = pd.read_csv(config.output_path("receptors_csv"))
    weather = pd.read_csv(config.output_path("weather_api_csv"))
    subregions = _read_subregion_table(config.output_path("subregions_geojson"))

    region_ids = sorted(subregions["region_id"].tolist())
    region_index = {region_id: idx for idx, region_id in enumerate(region_ids)}
    n = len(region_ids)
    hours = config.hours
    fallback = config.data["fallback_model"]
    dt = float(fallback["timestep_seconds"])
    matrix = np.zeros((hours, n, n), dtype=float)
    clamp_events = []

    receptor_groups = {
        region_id: frame.copy()
        for region_id, frame in receptors.groupby("region_id", sort=False)
    }
    source_groups = {
        region_id: frame.copy()
        for region_id, frame in sources.groupby("region_id", sort=False)
    }
    weather_index = weather.set_index(["hour_index", "region_id"])
    area_by_region = subregions.set_index("region_id")["area_m2"].to_dict()

    for hour in range(hours):
        for source_region in region_ids:
            source_col = region_index[source_region]
            source_weather = _weather_for(weather_index, hour, source_region)
            speed = float(source_weather["wind_speed_m_s"])
            direction = float(source_weather["wind_direction_deg_from"])
            pbl = source_weather.get("boundary_layer_height_m")
            if pd.isna(pbl):
                pbl = 800.0
            sigma = _sigma_for_region(config, float(area_by_region[source_region]), float(pbl), speed)
            u, v = _wind_to_uv_towards(speed, direction)
            dx = u * dt
            dy = v * dt
            norm = 1.0 / (2.0 * math.pi * sigma * sigma)

            for _, source in source_groups[source_region].iterrows():
                center_x = float(source["x_m"]) + dx
                center_y = float(source["y_m"]) + dy
                source_fraction = float(source["release_fraction"])
                for target_region in region_ids:
                    target_row = receptor_groups[target_region]
                    d2 = (target_row["x_m"].to_numpy() - center_x) ** 2 + (
                        target_row["y_m"].to_numpy() - center_y
                    ) ** 2
                    mean_density = float(np.exp(-d2 / (2.0 * sigma * sigma)).mean() * norm)
                    probability = mean_density * float(area_by_region[target_region])
                    matrix[hour, region_index[target_region], source_col] += source_fraction * probability

            col_sum = matrix[hour, :, source_col].sum()
            max_sum = float(fallback["max_raw_column_sum"])
            if col_sum > max_sum > 0:
                matrix[hour, :, source_col] *= max_sum / col_sum
                clamp_events.append(
                    {
                        "hour": hour,
                        "source_region": source_region,
                        "unclamped_sum": float(col_sum),
                        "clamped_sum": max_sum,
                    }
                )

    normalized = _normalize_columns(matrix)
    out = config.output_path("matrix_npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        T=matrix,
        T_normalized=normalized,
        region_ids=np.asarray(region_ids),
        hours_utc=np.asarray(_hours_utc(config)),
        method=np.asarray(["open_meteo_historical_forecast_advection_diffusion"]),
    )
    _write_hour_csvs(config, matrix, region_ids)
    _write_weather_summary(config, weather)
    _write_provenance(config, matrix, clamp_events)
    write_diagnostics(config)
    return out


def _read_subregion_table(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [feature["properties"] for feature in payload["features"]]
    return pd.DataFrame(rows)


def _weather_for(weather_index: pd.DataFrame, hour: int, region_id: str) -> pd.Series:
    try:
        return weather_index.loc[(hour, region_id)]
    except KeyError as exc:
        raise RuntimeError(f"Missing weather for hour={hour}, region={region_id}") from exc


def _sigma_for_region(config: CaseConfig, area_m2: float, pbl_m: float, speed_m_s: float) -> float:
    fallback = config.data["fallback_model"]
    eq_radius = math.sqrt(area_m2 / math.pi)
    kh = max(
        float(fallback["min_horizontal_diffusivity_m2_s"]),
        float(fallback["diffusivity_pbl_wind_factor"]) * max(pbl_m, 100.0) * max(speed_m_s, 0.5),
    )
    return math.sqrt((float(fallback["region_radius_sigma_factor"]) * eq_radius) ** 2 + 2.0 * kh * float(fallback["timestep_seconds"]))


def _wind_to_uv_towards(speed_m_s: float, direction_deg_from: float) -> tuple[float, float]:
    radians = math.radians(direction_deg_from)
    # Meteorological direction is where wind comes from. Convert to east/north motion.
    u = -speed_m_s * math.sin(radians)
    v = -speed_m_s * math.cos(radians)
    return u, v


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(matrix)
    col_sums = matrix.sum(axis=1, keepdims=True)
    np.divide(matrix, col_sums, out=normalized, where=col_sums > 0)
    return normalized


def _hours_utc(config: CaseConfig) -> list[str]:
    start = datetime.fromisoformat(config.data["time"]["start_utc"].replace("Z", "+00:00"))
    return [(start + timedelta(hours=hour)).isoformat() for hour in range(config.hours)]


def _write_hour_csvs(config: CaseConfig, matrix: np.ndarray, region_ids: list[str]) -> None:
    matrices_dir = config.output_path("matrices_dir")
    matrices_dir.mkdir(parents=True, exist_ok=True)
    for hour in range(matrix.shape[0]):
        frame = pd.DataFrame(matrix[hour], index=region_ids, columns=region_ids)
        frame.index.name = "target_region"
        frame.to_csv(matrices_dir / f"hour_{hour:02d}.csv")


def _write_weather_summary(config: CaseConfig, weather: pd.DataFrame) -> None:
    diag = config.output_path("diagnostics_dir")
    diag.mkdir(parents=True, exist_ok=True)
    summary = (
        weather.groupby("hour_index")
        .agg(
            wind_speed_mean_m_s=("wind_speed_m_s", "mean"),
            wind_speed_min_m_s=("wind_speed_m_s", "min"),
            wind_speed_max_m_s=("wind_speed_m_s", "max"),
            pbl_mean_m=("boundary_layer_height_m", "mean"),
            temperature_mean_c=("temperature_2m_c", "mean"),
            relative_humidity_mean_pct=("relative_humidity_2m_pct", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(diag / "weather_hourly_summary.csv", index=False)


def _write_provenance(config: CaseConfig, matrix: np.ndarray, clamp_events: list[dict[str, object]]) -> None:
    provenance = {
        "case_id": config.case_id,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "method": "fallback_open_meteo_historical_forecast_advection_diffusion",
        "not_calpuff": True,
        "reason": "Official CALPUFF/CALMET/wgrib2 executables were not available in the local environment.",
        "matrix_shape": list(matrix.shape),
        "nonzero_entries": int(np.count_nonzero(matrix)),
        "column_sum_min": float(matrix.sum(axis=1).min()),
        "column_sum_max": float(matrix.sum(axis=1).max()),
        "clamped_columns": clamp_events,
        "sources": {
            "calpuff": "https://calpuff.org/",
            "open_meteo_historical_forecast": "https://open-meteo.com/en/docs/historical-forecast-api",
            "census_tiger": config.data["domain"]["census_state_shapefile_url"],
        },
        "parameters": config.data["fallback_model"],
    }
    path = config.output_path("provenance_json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

