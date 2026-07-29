from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from transfer_matrix.assemble import assemble_matrices
from transfer_matrix.calpuff import build_calpuff_cases, check_external_tools, run_calpuff_cases
from transfer_matrix.config import load_case
from transfer_matrix.diagnostics import write_diagnostics
from transfer_matrix.grid import build_grid
from transfer_matrix.hrrr import fetch_hrrr
from transfer_matrix.weather_api import fetch_open_meteo_forecast
from transfer_matrix.fallback_model import compute_advection_diffusion_matrices
from transfer_matrix.paths import ensure_project_dirs
from transfer_matrix.validate import validate_outputs


def main(argv: list[str] | None = None) -> int:
    case_parent = argparse.ArgumentParser(add_help=False)
    case_parent.add_argument(
        "--case",
        default=argparse.SUPPRESS,
        help="Path to case YAML, relative to this project by default.",
    )
    parser = argparse.ArgumentParser(description="DC/VA/MD CALPUFF transfer-matrix harness")
    parser.add_argument("--case", default=None, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "prepare",
        parents=[case_parent],
        help="Create runtime dirs and check optional external tool paths.",
    )
    sub.add_parser(
        "build-grid",
        parents=[case_parent],
        help="Build 100 subregions plus source/receptor sample CSVs.",
    )

    fetch = sub.add_parser(
        "fetch-met",
        parents=[case_parent],
        help="Fetch HRRR meteorology files or write their manifest.",
    )
    fetch.add_argument("--manifest-only", action="store_true", help="Write URLs without downloading GRIB2 files.")

    sub.add_parser(
        "fetch-weather",
        parents=[case_parent],
        help="Fetch Open-Meteo historical forecast weather for subregion centroids.",
    )

    build_cases = sub.add_parser(
        "build-calpuff-cases",
        parents=[case_parent],
        help="Render per-hour/per-source CALPUFF cases.",
    )
    build_cases.add_argument("--hours", type=int, default=None)
    build_cases.add_argument("--sources", type=int, default=None)
    build_cases.add_argument("--dry-run", action="store_true")

    run_cases = sub.add_parser(
        "run-calpuff",
        parents=[case_parent],
        help="Run rendered CALPUFF cases.",
    )
    run_cases.add_argument("--hours", type=int, default=None)
    run_cases.add_argument("--sources", type=int, default=None)
    run_cases.add_argument("--max-workers", type=int, default=1)

    sub.add_parser("assemble", parents=[case_parent], help="Assemble receptor outputs into matrices.")
    sub.add_parser(
        "compute-fallback",
        parents=[case_parent],
        help="Compute non-CALPUFF advection-diffusion matrices from fetched weather.",
    )
    sub.add_parser("validate", parents=[case_parent], help="Validate assembled matrices.")

    all_cmd = sub.add_parser("all", parents=[case_parent], help="Run full workflow.")
    all_cmd.add_argument("--max-workers", type=int, default=1)
    all_cmd.add_argument("--manifest-only", action="store_true")
    all_cmd.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    case_path = Path(args.case or "config/case_20250624_18z.yaml")
    if not case_path.is_absolute():
        case_path = ROOT / case_path
    config = load_case(case_path)

    if args.command == "prepare":
        dirs = ensure_project_dirs(config)
        print("Created/verified directories:")
        for path in dirs:
            print(f"  {path}")
        try:
            tools = check_external_tools(config, require_template=False)
            print("External tools found:")
            for name, path in tools.items():
                print(f"  {name}: {path}")
        except RuntimeError as exc:
            print(f"External tools not ready: {exc}")
        return 0

    if args.command == "build-grid":
        ensure_project_dirs(config)
        print(build_grid(config))
        return 0

    if args.command == "fetch-met":
        ensure_project_dirs(config)
        print(fetch_hrrr(config, manifest_only=args.manifest_only))
        return 0

    if args.command == "fetch-weather":
        ensure_project_dirs(config)
        print(fetch_open_meteo_forecast(config))
        return 0

    if args.command == "build-calpuff-cases":
        ensure_project_dirs(config)
        count = build_calpuff_cases(config, hours=args.hours, sources=args.sources, dry_run=args.dry_run)
        print(f"Rendered {count} case directories")
        return 0

    if args.command == "run-calpuff":
        count = run_calpuff_cases(config, max_workers=args.max_workers, hours=args.hours, sources=args.sources)
        print(f"Ran {count} CALPUFF cases")
        return 0

    if args.command == "assemble":
        print(assemble_matrices(config))
        print("Diagnostics:")
        for path in write_diagnostics(config):
            print(f"  {path}")
        return 0

    if args.command == "compute-fallback":
        print(compute_advection_diffusion_matrices(config))
        return 0

    if args.command == "validate":
        print(validate_outputs(config))
        return 0

    if args.command == "all":
        ensure_project_dirs(config)
        build_grid(config)
        fetch_hrrr(config, manifest_only=args.manifest_only)
        build_calpuff_cases(config, dry_run=args.dry_run)
        if not args.dry_run:
            run_calpuff_cases(config, max_workers=args.max_workers)
        out = assemble_matrices(config)
        write_diagnostics(config)
        validate_outputs(config)
        print(out)
        return 0

    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
