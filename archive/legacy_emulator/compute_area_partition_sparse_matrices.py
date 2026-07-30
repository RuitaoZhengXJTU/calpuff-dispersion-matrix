from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, save_npz
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from transfer_matrix.config import load_case


SQ_MILE_M2 = 1609.344**2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute sparse 24h transfer matrices for the area-capped partition.")
    parser.add_argument("--case", default="config/case_20250623_18z.yaml")
    parser.add_argument(
        "--partition-dir",
        default="population_partitions/area_capped_20sqmi_population_balanced",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cutoff-sigma", type=float, default=4.5)
    parser.add_argument("--max-raw-column-sum", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--case-tag", default=None, help="Output tag; defaults to the case start timestamp.")
    args = parser.parse_args(argv)

    case_path = Path(args.case)
    if not case_path.is_absolute():
        case_path = ROOT / case_path
    config = load_case(case_path)
    case_tag = args.case_tag or _case_tag(config)
    partition_dir = ROOT / args.partition_dir
    out_dir = ROOT / (args.output_dir or f"{args.partition_dir}/sparse_transfer_matrices_{case_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    regions = pd.read_csv(partition_dir / "region_area_population_summary.csv")
    regions = regions.sort_values("region_id").reset_index(drop=True)
    region_ids = regions["region_id"].astype(str).to_numpy()
    xy = regions[["centroid_x", "centroid_y"]].to_numpy(float)
    area = regions["area_m2"].to_numpy(float)
    eq_radius = np.sqrt(area / math.pi)
    target_tree = cKDTree(xy)

    base_weather = pd.read_csv(config.output_path("weather_api_csv"))
    weather = _interpolate_weather(regions, base_weather)
    weather.to_csv(out_dir / "weather_by_region_hour.csv", index=False)

    dtype = np.float32 if args.dtype == "float32" else np.float64
    hours_utc = _hours_utc(config)
    hourly_stats = []
    matrices_dir = out_dir / "matrices_sparse"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    for hour in range(config.hours):
        print(f"hour {hour:02d}")
        frame = weather[weather["hour_index"] == hour].sort_values("region_id").reset_index(drop=True)
        matrix, stats = _compute_hour_sparse(
            config=config,
            xy=xy,
            area=area,
            eq_radius=eq_radius,
            wind_speed=frame["wind_speed_m_s"].to_numpy(float),
            wind_dir_from=frame["wind_direction_deg_from"].to_numpy(float),
            pbl=frame["boundary_layer_height_m"].to_numpy(float),
            target_tree=target_tree,
            cutoff_sigma=args.cutoff_sigma,
            max_raw_column_sum=args.max_raw_column_sum,
            dtype=dtype,
        )
        path = matrices_dir / f"hour_{hour:02d}.npz"
        save_npz(path, matrix, compressed=True)
        stats.update({"hour_index": hour, "matrix_path": str(path.relative_to(ROOT)).replace("\\", "/")})
        hourly_stats.append(stats)

    np.savez_compressed(
        out_dir / f"transfer_sparse_metadata_{case_tag}.npz",
        region_ids=region_ids,
        hours_utc=np.asarray(hours_utc),
        area_m2=area.astype(dtype),
        centroid_x=xy[:, 0].astype(dtype),
        centroid_y=xy[:, 1].astype(dtype),
        population=regions["population"].to_numpy(dtype),
        method=np.asarray(["sparse_calpuff_style_advection_diffusion_emulator"]),
    )
    pd.DataFrame(hourly_stats).to_csv(out_dir / "matrix_hourly_summary.csv", index=False)
    _write_provenance(out_dir, config, partition_dir, args, regions, hourly_stats)
    print(out_dir)
    return 0


def _interpolate_weather(regions: pd.DataFrame, base_weather: pd.DataFrame) -> pd.DataFrame:
    base_points = (
        base_weather[["region_id", "centroid_lon", "centroid_lat"]]
        .drop_duplicates("region_id")
        .sort_values("region_id")
        .reset_index(drop=True)
    )
    base_xy = _lonlat_to_xy(base_points["centroid_lon"].to_numpy(), base_points["centroid_lat"].to_numpy())
    target_xy = _lonlat_to_xy(regions["centroid_lon"].to_numpy(), regions["centroid_lat"].to_numpy())
    rows = []
    for hour, hour_frame in base_weather.groupby("hour_index"):
        hour_frame = hour_frame.sort_values("region_id").reset_index(drop=True)
        speeds = hour_frame["wind_speed_m_s"].to_numpy(float)
        dirs = hour_frame["wind_direction_deg_from"].to_numpy(float)
        u = -speeds * np.sin(np.deg2rad(dirs))
        v = -speeds * np.cos(np.deg2rad(dirs))
        values = {
            "boundary_layer_height_m": hour_frame["boundary_layer_height_m"].to_numpy(float),
            "temperature_2m_c": hour_frame["temperature_2m_c"].to_numpy(float),
            "relative_humidity_2m_pct": hour_frame["relative_humidity_2m_pct"].to_numpy(float),
        }
        for idx, region in regions.iterrows():
            weights = _idw_weights(base_xy, target_xy[idx], k=8)
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
                "wind_speed_m_s": wind_speed,
                "wind_direction_deg_from": wind_dir,
            }
            for name, arr in values.items():
                row[name] = float(np.dot(weights, arr))
            rows.append(row)
    return pd.DataFrame(rows)


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


def _compute_hour_sparse(
    config,
    xy: np.ndarray,
    area: np.ndarray,
    eq_radius: np.ndarray,
    wind_speed: np.ndarray,
    wind_dir_from: np.ndarray,
    pbl: np.ndarray,
    target_tree: cKDTree,
    cutoff_sigma: float,
    max_raw_column_sum: float,
    dtype,
) -> tuple[csc_matrix, dict[str, object]]:
    fallback = config.data["fallback_model"]
    dt = float(fallback["timestep_seconds"])
    min_kh = float(fallback["min_horizontal_diffusivity_m2_s"])
    kh_factor = float(fallback["diffusivity_pbl_wind_factor"])
    radius_factor = float(fallback["region_radius_sigma_factor"])
    n = xy.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    col_sums = np.zeros(n, dtype=float)
    zero_columns = 0

    for source_idx in range(n):
        speed = float(wind_speed[source_idx])
        direction = float(wind_dir_from[source_idx])
        pbl_m = float(pbl[source_idx]) if np.isfinite(pbl[source_idx]) else 800.0
        kh = max(min_kh, kh_factor * max(pbl_m, 100.0) * max(speed, 0.5))
        sigma = math.sqrt((radius_factor * float(eq_radius[source_idx])) ** 2 + 2.0 * kh * dt)
        u, v = _wind_to_uv_towards(speed, direction)
        center = xy[source_idx] + np.asarray([u * dt, v * dt])
        search_radius = cutoff_sigma * sigma + float(np.percentile(eq_radius, 90))
        targets = target_tree.query_ball_point(center, search_radius)
        if not targets:
            targets = [source_idx]
        targets_arr = np.asarray(targets, dtype=int)
        d2 = np.sum((xy[targets_arr] - center) ** 2, axis=1)
        keep = d2 <= (cutoff_sigma * sigma) ** 2
        targets_arr = targets_arr[keep]
        d2 = d2[keep]
        if targets_arr.size == 0:
            targets_arr = np.asarray([source_idx], dtype=int)
            d2 = np.asarray([0.0])
        norm = 1.0 / (2.0 * math.pi * sigma * sigma)
        values = np.exp(-d2 / (2.0 * sigma * sigma)) * norm * area[targets_arr]
        col_sum = float(values.sum())
        if col_sum > max_raw_column_sum > 0:
            values *= max_raw_column_sum / col_sum
            col_sum = max_raw_column_sum
        rows.extend(targets_arr.tolist())
        cols.extend([source_idx] * int(targets_arr.size))
        data.extend(values.astype(float).tolist())
        col_sums[source_idx] = col_sum

    matrix = csc_matrix((np.asarray(data, dtype=dtype), (rows, cols)), shape=(n, n), dtype=dtype)
    stats = {
        "shape": f"{n}x{n}",
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (n * n)),
        "column_sum_min": float(col_sums.min()),
        "column_sum_mean": float(col_sums.mean()),
        "column_sum_max": float(col_sums.max()),
        "zero_columns": int(zero_columns),
        "entry_max": float(matrix.data.max()) if matrix.nnz else 0.0,
        "cutoff_sigma": float(cutoff_sigma),
    }
    return matrix, stats


def _wind_to_uv_towards(speed_m_s: float, direction_deg_from: float) -> tuple[float, float]:
    radians = math.radians(direction_deg_from)
    return -speed_m_s * math.sin(radians), -speed_m_s * math.cos(radians)


def _hours_utc(config) -> list[str]:
    start = datetime.fromisoformat(config.data["time"]["start_utc"].replace("Z", "+00:00"))
    return [(start + timedelta(hours=hour)).isoformat() for hour in range(config.hours)]


def _case_tag(config) -> str:
    value = config.data["time"]["start_utc"].replace("-", "").replace(":", "")
    date, time = value.split("T", 1)
    hour = time[:2]
    return f"{date}_{hour}z"


def _write_provenance(out_dir: Path, config, partition_dir: Path, args, regions: pd.DataFrame, hourly_stats: list[dict[str, object]]) -> None:
    stats = pd.DataFrame(hourly_stats)
    dense_entries = config.hours * len(regions) * len(regions)
    provenance = {
        "case_id": config.case_id,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "time_window_utc": {
            "start": config.data["time"]["start_utc"],
            "hours": config.hours,
        },
        "partition_dir": str(partition_dir.relative_to(ROOT)).replace("\\", "/"),
        "region_count": int(len(regions)),
        "dense_matrix_entries_if_materialized": int(dense_entries),
        "dense_float32_gib_if_materialized": float(dense_entries * 4 / (1024**3)),
        "method": "sparse_calpuff_style_advection_diffusion_emulator",
        "not_official_calpuff": True,
        "official_calpuff_blocker": (
            "CALPUFF/CALMET executables exist locally, but the project still has only a placeholder "
            "CALPUFF control template and no verified CALMET-ready meteorological input/control-file pipeline."
        ),
        "sparse_summary": {
            "total_nnz": int(stats["nnz"].sum()),
            "mean_hourly_nnz": float(stats["nnz"].mean()),
            "max_hourly_nnz": int(stats["nnz"].max()),
            "column_sum_min": float(stats["column_sum_min"].min()),
            "column_sum_mean": float(stats["column_sum_mean"].mean()),
            "column_sum_max": float(stats["column_sum_max"].max()),
            "zero_columns_total": int(stats["zero_columns"].sum()),
        },
        "parameters": {
            "cutoff_sigma": args.cutoff_sigma,
            "dtype": args.dtype,
            "max_raw_column_sum": args.max_raw_column_sum,
            "fallback_model": config.data["fallback_model"],
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
