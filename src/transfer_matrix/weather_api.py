from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from .config import CaseConfig


def fetch_open_meteo_forecast(config: CaseConfig) -> Path:
    """Fetch hourly historical forecast weather for each region centroid."""

    subregions = _read_subregion_properties(config.output_path("subregions_geojson"))
    fallback = config.data["fallback_model"]
    batch_size = int(fallback["batch_size"])
    start = datetime.fromisoformat(config.data["time"]["start_utc"].replace("Z", "+00:00"))
    end = start + timedelta(hours=config.hours - 1)
    start_date = start.date().isoformat()
    end_date = end.date().isoformat()
    raw_responses = []
    rows = []

    for offset in range(0, len(subregions), batch_size):
        batch = subregions[offset : offset + batch_size]
        params = {
            "latitude": ",".join(f"{item['centroid_lat']:.6f}" for item in batch),
            "longitude": ",".join(f"{item['centroid_lon']:.6f}" for item in batch),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(fallback["variables"]),
            "wind_speed_unit": fallback["wind_speed_unit"],
            "timezone": fallback["timezone"],
        }
        response = requests.get(fallback["api_url"], params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = [payload]
        raw_responses.extend(payload)
        for item, weather in zip(batch, payload):
            rows.extend(_weather_rows_for_region(config, item, weather, start, end))

    raw_path = config.output_path("weather_api_json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw_responses, indent=2), encoding="utf-8")

    csv_path = config.output_path("weather_api_csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _read_subregion_properties(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for feature in payload["features"]:
        props = dict(feature["properties"])
        rows.append(props)
    rows.sort(key=lambda item: str(item["region_id"]))
    return rows


def _weather_rows_for_region(
    config: CaseConfig,
    region: dict[str, object],
    weather: dict[str, object],
    start: datetime,
    end: datetime,
) -> list[dict[str, object]]:
    hourly = weather["hourly"]
    rows = []
    valid_times = {
        (start + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M"): hour
        for hour in range(config.hours)
    }
    for index, time_label in enumerate(hourly["time"]):
        if time_label not in valid_times:
            continue
        rows.append(
            {
                "region_id": region["region_id"],
                "hour_index": valid_times[time_label],
                "time_utc": time_label,
                "centroid_lon": region["centroid_lon"],
                "centroid_lat": region["centroid_lat"],
                "api_lon": weather.get("longitude"),
                "api_lat": weather.get("latitude"),
                "wind_speed_m_s": hourly["wind_speed_10m"][index],
                "wind_direction_deg_from": hourly["wind_direction_10m"][index],
                "boundary_layer_height_m": hourly.get("boundary_layer_height", [None])[index],
                "temperature_2m_c": hourly.get("temperature_2m", [None])[index],
                "relative_humidity_2m_pct": hourly.get("relative_humidity_2m", [None])[index],
            }
        )
    return rows

