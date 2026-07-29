"""Run the official CALPUFF B0 and A_h matrix experiments.

The paper contract implemented here is:

    c_1 = B0 @ emitted_mass_lb
    c_{h+1} = A_h @ c_h,  h=1,...,23

B0 is the one-hour CALPUFF response to a 1 lb/h generator release during
[t0, t1).  Each A_h is a CALPUFF response to a unit concentration represented
by an equivalent one-hour mass release during [t_h, t_{h+1}).  The source
experiment is deliberately one hour long; no preceding emission period is
silently added.

This is an official CALPUFF/CALPOST route. It does not call the fallback
emulator. B0 uses all fixed receptor batches; A uses the documented default
150 km geometric sparse-receptor rule unless that radius is disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, save_npz

from official_case_builder import CalpuffCaseFactory, CalpuffDomain, load_csv_rows
from official_case_config import load_case_config, mapping_value, project_path
from parse_calpost_tseries import parse_calpost_tseries
from run_official_sparse_matrix import (
    _assert_success,
    _parse_start_utc,
    _resolve_executable,
    _sha256,
)


ROOT = Path(__file__).resolve().parent
CASE_ROOT = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi"
PARTITION = ROOT / "population_partitions" / "area_capped_30sqmi_population_balanced"
DEFAULT_OUTPUT = ROOT / "outputs" / "official_ab_20250623_18z"
DEFAULT_SOURCES = CASE_ROOT / "inputs" / "sources_16_per_region.csv"
DEFAULT_RECEPTORS = CASE_ROOT / "inputs" / "receptors_9_per_region.csv"
DEFAULT_BATCH_DIR = CASE_ROOT / "inputs" / "receptor_batches"
DEFAULT_REGION_INDEX = CASE_ROOT / "inputs" / "matrix_region_index.csv"
DEFAULT_GENERATORS = ROOT / "data" / "data_centers_example.csv"
DEFAULT_SEED = Path(os.environ.get(
    "CALPUFF_SEED",
    CASE_ROOT / "templates" / "CALPUFF_7.0_seed_from_distribution.INP",
))
DEFAULT_CALPOST_TEMPLATE = Path(os.environ.get(
    "CALPOST_TEMPLATE",
    ROOT / "data" / "raw" / "official_examples" / "calpost_v7.1.0_L141010" / "CALPOST_v7.1.0_L141010" / "calpost.inp",
))
DEFAULT_CALPOST_EXE = ROOT / "data" / "raw" / "official_examples" / "calpost_v7.1.0_L141010" / "CALPOST_v7.1.0_L141010" / "calpost_v7.1.0.exe"
DEFAULT_CALMET = Path(os.environ.get(
    "CALMET_DAT",
    CASE_ROOT / "met" / "calmet_hrrr" / "CALMET.DAT",
))
LB_TO_G = 453.59237
UG_TO_G = 1.0e-6
ERROR_MARKERS = ("error in subr", "fatal", "halted in", "endfile")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("b0", "a", "all"), default="all")
    parser.add_argument(
        "--case-config", type=Path, default=None,
        help="portable official_case YAML; explicit CLI paths override its paths section",
    )
    parser.add_argument("--case-root", type=Path, default=None)
    parser.add_argument("--partition-dir", type=Path, default=None)
    parser.add_argument("--sources", type=Path, default=None)
    parser.add_argument("--receptors", type=Path, default=None)
    parser.add_argument("--receptor-batch-dir", type=Path, default=None)
    parser.add_argument("--region-index", type=Path, default=None)
    parser.add_argument("--generators", type=Path, default=None)
    parser.add_argument("--weather", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=Path, default=None)
    parser.add_argument("--calpost-template", type=Path, default=None)
    parser.add_argument("--calmet-dat", type=Path, default=None)
    parser.add_argument("--start-utc", default=None)
    parser.add_argument("--hours", type=int, default=None)
    parser.add_argument("--calpuff-exe", default=None)
    parser.add_argument("--calpost-exe", default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--b0-source-count", type=int, default=None)
    parser.add_argument("--a-source-count", type=int, default=None)
    parser.add_argument("--a-start-hour", type=int, default=1)
    parser.add_argument("--a-hours", type=int, default=None)
    parser.add_argument(
        "--a-sparse-radius-km",
        type=float,
        default=None,
        help="Keep target receptors within this projected distance of each A source region; use 0 to disable",
    )
    parser.add_argument("--unit-concentration-ug-m3", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record failed CALPUFF cases and continue other sources; rerun failures with --resume",
    )
    parser.add_argument("--retain-case-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    case_config = load_case_config(args.case_config)
    _apply_case_config(args, case_config)

    root = ROOT
    paths = _resolve_paths(args, root)
    for path in (paths["sources"], paths["receptors"], paths["region_index"], paths["generators"], paths["seed"], paths["calpost_template"]):
        if not path.exists():
            raise FileNotFoundError(path)
    if not paths["calmet_dat"].exists() and not args.dry_run:
        raise FileNotFoundError(f"CALMET.DAT is required: {paths['calmet_dat']}")
    if not args.dry_run:
        for executable in (paths["calpuff_exe"], paths["calpost_exe"]):
            if not executable.exists():
                raise FileNotFoundError(executable)
    if args.max_workers <= 0:
        raise ValueError("max-workers must be positive")
    if args.hours < 2:
        raise ValueError("hours must be at least 2: B0 covers hour 0 and A covers later transitions")
    if args.a_start_hour < 1 or args.a_hours <= 0 or args.a_start_hour + args.a_hours > args.hours:
        raise ValueError(f"A hours must be within transition hours 1..{args.hours - 1}")
    if args.unit_concentration_ug_m3 <= 0:
        raise ValueError("unit-concentration-ug-m3 must be positive")
    if args.a_sparse_radius_km < 0:
        raise ValueError("a-sparse-radius-km must be nonnegative")

    paths["output_root"].mkdir(parents=True, exist_ok=True)
    start_utc = _parse_start_utc(args.start_utc)
    region_index = pd.read_csv(paths["region_index"]).sort_values("matrix_index").reset_index(drop=True)
    region_ids = region_index["region_id"].astype(str).to_numpy()
    if len(region_ids) != len(set(region_ids)):
        raise ValueError("region index contains duplicate region_id values")
    receptor_batches = _load_receptor_batches(paths["receptor_batch_dir"], paths["receptors"])
    source_rows = load_csv_rows(paths["sources"])
    factory = CalpuffCaseFactory(
        paths["seed"],
        paths["calpost_template"],
        source_rows,
        str(paths["calmet_dat"]),
        start_utc=start_utc,
        domain=args.calpuff_domain,
    )

    weather = None
    if args.mode in {"a", "all"}:
        weather_path = paths["weather"] or _default_weather_path(paths["partition_dir"])
        weather = _load_weather(weather_path, region_ids, args.hours)
    output = paths["output_root"]
    contract = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": args.case_id,
        "start_utc": _iso(start_utc),
        "end_utc_exclusive": _iso(start_utc + pd.Timedelta(hours=args.hours).to_pytimedelta()),
        "horizon_hours": int(args.hours),
        "a_hour_indices": list(range(1, args.hours)),
        "state_equation": _state_equation(args.hours),
        "matrix_convention": "rows=target regions; columns=source regions or generators",
        "region_count": int(len(region_ids)),
        "generator_count": int(len(pd.read_csv(paths["generators"]))),
        "receptor_batches": int(len(receptor_batches)),
        "receptor_count": int(sum(len(batch) for batch in receptor_batches)),
        "source_points_per_region": 16,
        "release_height_m": 15.0,
        "chemistry": "off",
        "deposition": "off",
        "calmet_dat_sha256": _sha256(paths["calmet_dat"]) if paths["calmet_dat"].exists() else None,
        "unit_concentration_ug_m3": float(args.unit_concentration_ug_m3),
        "a_sparse_radius_km": float(args.a_sparse_radius_km) if args.a_sparse_radius_km else None,
        "a_sparsification": (
            "geometric target-receptor radius around each source-region centroid in projected coordinates"
            if args.a_sparse_radius_km else "disabled; all receptor batches"
        ),
        "calpuff_domain": _domain_contract(args.calpuff_domain),
        "outputs": {"b0": "B0", "a": "A"},
    }
    (output / "matrix_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    if args.mode in {"b0", "all"}:
        _run_b0(args, paths, factory, receptor_batches, region_ids, output / "B0", start_utc)
    if args.mode in {"a", "all"}:
        assert weather is not None
        _run_a(
            args,
            paths,
            factory,
            receptor_batches,
            region_ids,
            region_index,
            weather,
            output / "A",
            start_utc,
        )
    return 0


def _apply_case_config(args: argparse.Namespace, payload: dict[str, object]) -> None:
    """Apply a portable case manifest while preserving explicit CLI precedence."""
    paths = mapping_value(payload, "paths")
    model = mapping_value(payload, "model")
    time = mapping_value(payload, "time")

    defaults: dict[str, Path | None] = {
        "case_root": CASE_ROOT,
        "partition_dir": PARTITION,
        "sources": DEFAULT_SOURCES,
        "receptors": DEFAULT_RECEPTORS,
        "receptor_batch_dir": DEFAULT_BATCH_DIR,
        "region_index": DEFAULT_REGION_INDEX,
        "generators": DEFAULT_GENERATORS,
        "weather": None,
        "output_root": DEFAULT_OUTPUT,
        "seed": DEFAULT_SEED,
        "calpost_template": DEFAULT_CALPOST_TEMPLATE,
        "calmet_dat": DEFAULT_CALMET,
    }
    for name, fallback in defaults.items():
        configured = project_path(ROOT, paths.get(name))
        if getattr(args, name) is None:
            setattr(args, name, configured or fallback)
    args.start_utc = args.start_utc or str(time.get("start_utc", "2025-06-23T18:00:00Z"))
    args.hours = args.hours if args.hours is not None else int(time.get("hours", 24))
    args.a_hours = args.a_hours if args.a_hours is not None else args.hours - args.a_start_hour
    if args.a_sparse_radius_km is None:
        args.a_sparse_radius_km = float(model.get("a_sparse_radius_km", 150.0))
    args.case_id = str(payload.get("case_id") or Path(args.case_root).name)
    args.calpuff_domain = CalpuffDomain.from_mapping(mapping_value(payload, "calpuff_domain"))


def _state_equation(hours: int) -> str:
    return f"c1 = B0 @ emitted_mass_lb; c[h+1] = A[h] @ c[h] for h=1..{hours - 1}"


def _domain_contract(domain: CalpuffDomain) -> dict[str, object]:
    return {
        "projected_crs": domain.projected_crs,
        "pmap": domain.pmap,
        "datum": domain.datum,
        "nx": domain.nx,
        "ny": domain.ny,
        "nz": domain.nz,
        "dgrid_km": domain.dgrid_km,
        "xorig_km": domain.xorig_km,
        "yorig_km": domain.yorig_km,
    }


def _resolve_paths(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    return {
        "case_root": resolve(args.case_root),
        "partition_dir": resolve(args.partition_dir),
        "sources": resolve(args.sources),
        "receptors": resolve(args.receptors),
        "receptor_batch_dir": resolve(args.receptor_batch_dir),
        "region_index": resolve(args.region_index),
        "generators": resolve(args.generators),
        "weather": resolve(args.weather) if args.weather else None,
        "output_root": resolve(args.output_root),
        "seed": resolve(args.seed),
        "calpost_template": resolve(args.calpost_template),
        "calmet_dat": resolve(args.calmet_dat),
        "calpuff_exe": _resolve_executable(args.calpuff_exe, "CALPUFF_EXE", "calpuff_v7.2.1.exe"),
        "calpost_exe": _resolve_executable(args.calpost_exe, "CALPOST_EXE", "calpost_v7.1.0.exe", DEFAULT_CALPOST_EXE),
    }


def _load_receptor_batches(batch_dir: Path, receptor_path: Path) -> list[list[dict[str, str]]]:
    files = sorted(batch_dir.glob("batch_*.csv"))
    if not files:
        raise FileNotFoundError(f"No receptor batches found under {batch_dir}")
    batches = [load_csv_rows(path) for path in files]
    all_ids = [row["receptor_id"] for batch in batches for row in batch]
    expected = [row["receptor_id"] for row in load_csv_rows(receptor_path)]
    if sorted(all_ids) != sorted(expected):
        raise ValueError("receptor batches do not exactly cover the receptor table")
    for batch in batches:
        if len(batch) == 0:
            raise ValueError("empty receptor batch")
    return batches


def _default_weather_path(partition_dir: Path) -> Path:
    candidates = sorted(partition_dir.glob("sparse_transfer_matrices_*/weather_by_region_hour.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError("Pass --weather explicitly when more than one weather table exists")
    return candidates[0]


def _load_weather(path: Path, region_ids: np.ndarray, hours: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    weather = pd.read_csv(path)
    required = {"region_id", "hour_index", "boundary_layer_height_m"}
    missing = required - set(weather.columns)
    if missing:
        raise ValueError(f"weather table lacks columns: {sorted(missing)}")
    weather["region_id"] = weather["region_id"].astype(str)
    expected = {(hour, region_id) for hour in range(hours) for region_id in region_ids}
    observed = set(zip(weather["hour_index"].astype(int), weather["region_id"]))
    if expected - observed:
        raise ValueError(f"weather table is missing {len(expected - observed)} region-hour values")
    return weather


def _run_b0(
    args: argparse.Namespace,
    paths: dict[str, Path],
    factory: CalpuffCaseFactory,
    receptor_batches: list[list[dict[str, str]]],
    region_ids: np.ndarray,
    output_dir: Path,
    start_utc: datetime,
) -> None:
    inventory = pd.read_csv(paths["generators"]).reset_index(drop=True)
    required = {"generator_id", "facility_id", "site_no", "region_id", "lon", "lat", "stack_height"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"generator inventory lacks columns: {sorted(missing)}")
    if args.b0_source_count is not None:
        inventory = inventory.iloc[: args.b0_source_count].copy()
    region_index = {region_id: i for i, region_id in enumerate(region_ids)}
    to_projected = _projected_transformer(factory.domain.projected_crs)
    generator_sources: list[dict[str, str]] = []
    generator_meta: list[dict[str, object]] = []
    for generator_index, row in inventory.iterrows():
        x_m, y_m = to_projected(float(row["lon"]), float(row["lat"]))
        generator_meta.append({
            "generator_matrix_index": int(generator_index),
            "generator_id": str(row["generator_id"]),
            "facility_id": str(row["facility_id"]),
            "site_no": str(row["site_no"]),
            "region_id": str(row["region_id"]),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "x_m": float(x_m),
            "y_m": float(y_m),
        })
        if str(row["region_id"]) not in region_index:
            raise ValueError(f"generator region is not in the partition: {row['region_id']}")
        for point_index in range(16):
            generator_sources.append({
                "source_id": f"{row['generator_id']}_s{point_index:02d}",
                "matrix_index": str(generator_index),
                "x_m": str(x_m),
                "y_m": str(y_m),
                "release_fraction": str(1.0 / 16.0),
            })
    generator_factory = CalpuffCaseFactory(
        paths["seed"], paths["calpost_template"], generator_sources,
        str(paths["calmet_dat"]), start_utc=start_utc, domain=factory.domain,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(generator_meta).to_csv(output_dir / "generator_columns.csv", index=False)
    responses = _run_cases_by_source(
        args=args,
        paths=paths,
        factory=generator_factory,
        receptor_batches=receptor_batches,
        source_count=len(inventory),
        hour_index=0,
        emission_rate=lambda _source: 1.0,
        mode="B0",
        output_dir=output_dir / "cases",
        start_utc=start_utc,
    )
    rows, cols, values = [], [], []
    for source_index, target_values in responses.items():
        for target_index, value in target_values.items():
            if value > 0.0:
                rows.append(target_index)
                cols.append(source_index)
                values.append(value)
    matrix = csc_matrix((np.asarray(values, dtype=np.float32), (rows, cols)), shape=(len(region_ids), len(inventory)))
    save_npz(output_dir / "B0_g_m3_per_lb.npz", matrix, compressed=True)
    (output_dir / "provenance.json").write_text(json.dumps({
        "matrix": "B0",
        "shape": list(matrix.shape),
        "unit": "g/m3 per lb emitted during [t0,t1)",
        "time_start_utc": _iso(start_utc),
        "time_end_utc_exclusive": _iso(start_utc + pd.Timedelta(hours=1).to_pytimedelta()),
        "source_experiment": "1 lb/h from each generator represented by 16 equal-weight 15 m volume sources",
        "receptor_aggregation": "mean of 9 receptors per target region",
        "uses_calpuff": True,
        "not_emulator": True,
        "nnz": int(matrix.nnz),
    }, indent=2), encoding="utf-8")


def _run_a(
    args: argparse.Namespace,
    paths: dict[str, Path],
    factory: CalpuffCaseFactory,
    receptor_batches: list[list[dict[str, str]]],
    region_ids: np.ndarray,
    region_index: pd.DataFrame,
    weather: pd.DataFrame,
    output_dir: Path,
    start_utc: datetime,
) -> None:
    source_count = len(region_ids) if args.a_source_count is None else min(args.a_source_count, len(region_ids))
    area = region_index["area_m2"].to_numpy(float)
    pbl_by_hour = _pbl_array(weather, region_ids, args.hours)
    source_centroids = _region_source_centroids(load_csv_rows(paths["sources"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly = []
    for hour_index in range(args.a_start_hour, args.a_start_hour + args.a_hours):
        volumes = area * pbl_by_hour[hour_index]
        unit_mass_lb = volumes * args.unit_concentration_ug_m3 * UG_TO_G / LB_TO_G
        responses = _run_cases_by_source(
            args=args,
            paths=paths,
            factory=factory,
            receptor_batches=receptor_batches,
            source_count=source_count,
            hour_index=hour_index,
            emission_rate=lambda source, rates=unit_mass_lb: float(rates[source]),
            mode=f"A_hour_{hour_index:02d}",
            output_dir=output_dir / f"hour_{hour_index:02d}" / "cases",
            start_utc=start_utc,
            source_centroids=source_centroids,
            sparse_radius_km=args.a_sparse_radius_km,
        )
        rows, cols, values = [], [], []
        for source_index, target_values in responses.items():
            for target_index, value in target_values.items():
                coefficient = value / (args.unit_concentration_ug_m3 * UG_TO_G)
                if coefficient > 0.0:
                    rows.append(target_index)
                    cols.append(source_index)
                    values.append(coefficient)
        matrix = csc_matrix((np.asarray(values, dtype=np.float32), (rows, cols)), shape=(len(region_ids), len(region_ids)))
        save_npz(output_dir / f"hour_{hour_index:02d}.npz", matrix, compressed=True)
        hourly.append({
            "hour_index": hour_index,
            "time_start_utc": _iso(start_utc + pd.Timedelta(hours=hour_index).to_pytimedelta()),
            "time_end_utc_exclusive": _iso(start_utc + pd.Timedelta(hours=hour_index + 1).to_pytimedelta()),
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
            "coefficient_max": float(matrix.data.max()) if matrix.nnz else 0.0,
            "column_count": source_count,
            "unit": "dimensionless concentration-state transfer",
        })
    (output_dir / "provenance.json").write_text(json.dumps({
        "matrix": "A",
        "hours": hourly,
        "source_experiment": "unit regional concentration represented by equivalent one-hour CALPUFF volume-source mass release",
        "unit_concentration_ug_m3": args.unit_concentration_ug_m3,
        "mass_conversion": "M_lb = V_m3 * unit_concentration_ug_m3 * 1e-6 / 453.59237",
        "uses_calpuff": True,
        "not_emulator": True,
        "receptor_aggregation": "mean of 9 receptors per target region",
        "sparse_radius_km": float(args.a_sparse_radius_km) if args.a_sparse_radius_km else None,
        "sparsification": (
            "target receptors within source-region centroid radius in configured projected coordinates"
            if args.a_sparse_radius_km else "disabled; all receptor batches"
        ),
    }, indent=2), encoding="utf-8")


def _run_cases_by_source(
    args: argparse.Namespace,
    paths: dict[str, Path],
    factory: CalpuffCaseFactory,
    receptor_batches: list[list[dict[str, str]]],
    source_count: int,
    hour_index: int,
    emission_rate,
    mode: str,
    output_dir: Path,
    start_utc: datetime,
    source_centroids: dict[int, tuple[float, float]] | None = None,
    sparse_radius_km: float = 0.0,
) -> dict[int, dict[int, float]]:
    tasks = list(range(source_count))
    results: dict[int, dict[int, float]] = {}

    def run_source(source_index: int) -> tuple[int, dict[int, float]]:
        combined: dict[int, float] = {}
        active_batches = _select_receptor_batches(
            receptor_batches,
            source_centroids.get(source_index) if source_centroids else None,
            sparse_radius_km,
        )
        for batch_index, receptor_rows in enumerate(active_batches):
            case_dir = output_dir / f"source_{source_index:05d}" / f"batch_{batch_index:03d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            status_path = case_dir / "run_status.json"
            if args.resume and status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                response_file = case_dir / "region_response.npz"
                if status.get("status") == "completed" and response_file.exists():
                    saved = np.load(response_file)
                    for key, value in zip(saved["indices"], saved["values"]):
                        combined[int(key)] = float(value)
                    continue
            try:
                rate = float(emission_rate(source_index))
                local_calmet = _stage_calmet_links(paths["calmet_dat"], case_dir)
                control = factory.build_calpuff(
                    case_dir,
                    source_index,
                    hour_index,
                    receptor_rows,
                    emission_lb_per_hour=rate,
                    include_preceding_met_period=False,
                    initialization_mode="one_hour_box",
                    calmet_dat_override=_relative_ascii_path(local_calmet, case_dir),
                )
                calpost_control = factory.build_calpost(case_dir, hour_index)
                if args.dry_run:
                    status_path.write_text(json.dumps({
                        "status": "dry_run_ready",
                        "source_index": source_index,
                        "hour_index": hour_index,
                        "batch_index": batch_index,
                        "emission_lb_per_hour": rate,
                    }, indent=2), encoding="utf-8")
                    continue
                _run_binary(paths["calpuff_exe"], control, case_dir, case_dir / "CALPUFF_RUN.log", args.timeout_sec)
                _assert_success(case_dir / "CALPUFF.CON", case_dir / "CALPUFF_RUN.log")
                _run_binary(paths["calpost_exe"], calpost_control, case_dir, case_dir / "CALPOST_RUN.log", args.timeout_sec)
                tseries = case_dir / "TSERIES_NO2_1HR_CONC.DAT"
                _assert_success(tseries, case_dir / "CALPOST_RUN.log")
                response_path = case_dir / "receptors.csv"
                parse_calpost_tseries(
                    tseries,
                    case_dir / "receptor_manifest.csv",
                    response_path,
                    start_utc=_iso(start_utc + pd.Timedelta(hours=hour_index).to_pytimedelta()),
                    value_unit="g/m3",
                )
                table = pd.read_csv(response_path)
                table["concentration"] = pd.to_numeric(table["concentration"], errors="raise")
                grouped = table.groupby("matrix_index", sort=True)["concentration"].mean()
                batch_response = {int(key): float(value) for key, value in grouped.items() if np.isfinite(value) and value > 0}
                combined.update(batch_response)
                np.savez_compressed(
                    case_dir / "region_response.npz",
                    indices=np.asarray(sorted(batch_response), dtype=np.int32),
                    values=np.asarray([batch_response[key] for key in sorted(batch_response)], dtype=np.float32),
                )
                status_path.write_text(json.dumps({
                    "status": "completed",
                    "source_index": source_index,
                    "hour_index": hour_index,
                    "batch_index": batch_index,
                    "emission_lb_per_hour": rate,
                    "response_file": "region_response.npz",
                    "response_nnz": len(batch_response),
                }, indent=2), encoding="utf-8")
                if not args.retain_case_files:
                    _remove_case_intermediates(case_dir)
            except Exception as exc:
                status_path.write_text(json.dumps({
                    "status": "failed",
                    "source_index": source_index,
                    "hour_index": hour_index,
                    "batch_index": batch_index,
                    "error": str(exc),
                }, indent=2), encoding="utf-8")
                raise
        return source_index, combined

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_source, source_index) for source_index in tasks]
        for future in as_completed(futures):
            try:
                source_index, values = future.result()
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(json.dumps({"mode": mode, "status": "failed_source", "error": str(exc)}))
                continue
            results[source_index] = values
            if len(results) <= 5 or len(results) % 100 == 0 or len(results) == len(tasks):
                print(json.dumps({"mode": mode, "hour_index": hour_index, "completed_sources": len(results), "total_sources": len(tasks)}))
    return results


def _region_source_centroids(source_rows: list[dict[str, str]]) -> dict[int, tuple[float, float]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for row in source_rows:
        grouped.setdefault(int(row["matrix_index"]), []).append(
            (float(row["x_m"]), float(row["y_m"]))
        )
    return {
        index: (float(np.mean([point[0] for point in points])), float(np.mean([point[1] for point in points])))
        for index, points in grouped.items()
    }


def _select_receptor_batches(
    receptor_batches: list[list[dict[str, str]]],
    source_centroid: tuple[float, float] | None,
    radius_km: float,
) -> list[list[dict[str, str]]]:
    if not radius_km or source_centroid is None:
        return receptor_batches
    radius_m = radius_km * 1000.0
    source_x, source_y = source_centroid
    selected = []
    for batch in receptor_batches:
        for row in batch:
            dx = float(row["x_m"]) - source_x
            dy = float(row["y_m"]) - source_y
            if dx * dx + dy * dy <= radius_m * radius_m:
                selected.append(row)
    if not selected:
        raise ValueError(f"sparse receptor radius selected no receptors for source centroid {source_centroid}")
    return [selected[offset : offset + 10000] for offset in range(0, len(selected), 10000)]


def _stage_calmet_links(calmet_dat: Path, case_dir: Path) -> Path:
    """Give each CALPUFF case private hard links to CALMET and its aux file.

    CALPUFF may create/delete the ``.aux`` sidecar during a run. Sharing the
    sidecar across concurrent cases causes Fortran runtime DeleteFile races.
    Hard links keep disk use low while making the directory entry private to
    each case. The source CALMET files are never modified or removed.
    """
    local_dat = case_dir / "CALMET.DAT"
    local_aux = case_dir / "CALMET.DAT.aux"
    source_aux = Path(str(calmet_dat) + ".aux")
    for source, target in ((calmet_dat, local_dat), (source_aux, local_aux)):
        if not source.exists():
            if source == source_aux:
                continue
            raise FileNotFoundError(source)
        if target.exists():
            continue
        try:
            os.link(source, target)
        except OSError as exc:
            raise RuntimeError(
                f"cannot create private CALMET hard link {target}; "
                "parallel CALPUFF runs require a filesystem supporting hard links"
            ) from exc
    return local_dat


def _pbl_array(weather: pd.DataFrame, region_ids: np.ndarray, hours: int) -> np.ndarray:
    index = weather.assign(region_id=weather["region_id"].astype(str)).set_index(["hour_index", "region_id"])
    values = np.zeros((hours, len(region_ids)), dtype=float)
    for hour in range(hours):
        values[hour] = [float(index.loc[(hour, region_id), "boundary_layer_height_m"]) for region_id in region_ids]
    return values


def _projected_transformer(projected_crs: str):
    from pyproj import Transformer

    return Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    ).transform


def _run_binary(executable: Path, control: Path, cwd: Path, log_path: Path, timeout_sec: int) -> None:
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        result = subprocess.run(
            [str(executable), control.name],
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{executable.name} returned {result.returncode}; see {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    if any(marker in text for marker in ERROR_MARKERS):
        raise RuntimeError(f"binary log contains an error marker: {log_path}")


def _remove_case_intermediates(case_dir: Path) -> None:
    for path in case_dir.iterdir():
        # Keep the compact parsed response and status so --resume can reuse a
        # completed CALPUFF/CALPOST case without retaining the large CON file.
        if path.name not in {"run_status.json", "region_response.npz"}:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _relative_ascii_path(target: Path, case_dir: Path) -> str:
    """Return a CALPUFF-safe relative path without non-ASCII parent names."""
    return os.path.relpath(target, case_dir).replace("/", "\\")


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
