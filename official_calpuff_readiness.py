from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
PARTITION_DIR = ROOT / "population_partitions" / "area_capped_30sqmi_population_balanced"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check readiness for official CALPUFF matrix production.")
    parser.add_argument("--case", default="config/case_20250623_18z.yaml")
    parser.add_argument("--partition-dir", default=str(PARTITION_DIR.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--workers", type=int, nargs="*", default=[8, 16, 32, 64])
    parser.add_argument("--seconds-per-case", type=float, nargs="*", default=[5, 15, 30, 60, 120])
    parser.add_argument("--receptor-points-per-region", type=int, default=9)
    parser.add_argument(
        "--max-discrete-receptors",
        type=int,
        default=10000,
        help="Compiled MXREC value; verified for the installed CALPUFF 7.2.1 executable by probe_calpuff_compiled_limits.py.",
    )
    args = parser.parse_args()

    case_path = ROOT / args.case
    config = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    partition_dir = ROOT / args.partition_dir
    regions = pd.read_csv(partition_dir / "region_area_population_summary.csv")
    region_count = int(len(regions))
    hours = int(config["time"]["hours"])
    source_hour_case_count = hours * region_count
    receptor_count = region_count * args.receptor_points_per_region
    receptor_batches = math.ceil(receptor_count / args.max_discrete_receptors)
    batched_run_count = source_hour_case_count * receptor_batches

    tools = _tool_status(config)
    template_status = _template_status(config)
    estimates = _runtime_estimates(batched_run_count, args.seconds_per_case, args.workers)

    mmif_direct_met = (partition_dir / "official_calpuff" / "met" / "mmif" / "CALMET.DAT").exists()
    project_mmif_direct_met = (
        ROOT
        / "official_calpuff"
        / "case_20250623_18z_30sqmi"
        / "met"
        / "mmif"
        / "CALMET.DAT"
    ).exists()
    has_calmet_dat = (
        (partition_dir / "official_calpuff" / "met" / "calmet_outputs" / "CALMET.DAT").exists()
        or mmif_direct_met
        or project_mmif_direct_met
    )
    calmet_route_ready = template_status["calmet_template_ready"]
    mmif_route_ready = tools["calwrf_or_mmif"]["exists"]
    candidate_provenance_path = (
        ROOT
        / "official_calpuff"
        / "case_20250623_18z_30sqmi"
        / "inputs"
        / "sparse_candidate_manifest"
        / "provenance.json"
    )
    candidate_provenance = (
        json.loads(candidate_provenance_path.read_text(encoding="utf-8"))
        if candidate_provenance_path.exists()
        else None
    )

    missing = []
    if not (calmet_route_ready or mmif_route_ready):
        missing.append("verified CALMET.INP template or MMIF/CALWRF meteorology converter")
    if not template_status["calpuff_template_ready"]:
        missing.append("verified CALPUFF.INP template")
    if not template_status["calpost_template_ready"]:
        missing.append("verified CALPOST.INP template or direct CALPUFF receptor parser")
    if not has_calmet_dat:
        missing.append("CALMET.DAT or MMIF CALPUFF-format met file generated for the target window")

    report = {
        "case": args.case,
        "partition_dir": args.partition_dir,
        "hours": hours,
        "region_count": region_count,
        "receptor_count": receptor_count,
        "receptor_points_per_region": args.receptor_points_per_region,
        "max_discrete_receptors_assumed": args.max_discrete_receptors,
        "receptor_batches": receptor_batches,
        "source_hour_cases": source_hour_case_count,
        "batched_calpuff_run_count": batched_run_count,
        "screened_sparse_candidate": {
            "available": candidate_provenance is not None,
            "validated_by_official_calpuff": False if candidate_provenance else None,
            "candidate_batch_count_total": candidate_provenance.get("candidate_batch_count_total") if candidate_provenance else None,
            "candidate_receptor_count_max": candidate_provenance.get("candidate_receptor_count_max") if candidate_provenance else None,
            "provenance": str(candidate_provenance_path) if candidate_provenance else None,
        },
        "dense_coefficients": hours * region_count * region_count,
        "dense_float32_gib": hours * region_count * region_count * 4 / (1024**3),
        "tools": tools,
        "templates": template_status,
        "meteorology_routes": {
            "calmet_control_file_route_ready": calmet_route_ready,
            "mmif_or_calwrf_converter_ready": mmif_route_ready,
            "target_window_calmet_dat_exists": has_calmet_dat,
        },
        "missing_required_for_official_run": missing,
        "runtime_estimates": estimates,
        "screened_candidate_runtime_estimates": (
            _runtime_estimates(int(candidate_provenance["candidate_batch_count_total"]), args.seconds_per_case, args.workers)
            if candidate_provenance
            else []
        ),
        "notes": [
            "source_hour_cases counts the direct impulse cases before receptor batching.",
            "batched_calpuff_run_count multiplies source-hour cases by ceil(target_receptors / assumed MXREC).",
            "MXREC is a compile-time array limit; verify the installed executable or recompile before using the assumed value.",
            "CALMET is run once per meteorological window; CALPUFF is run once per source-hour and receptor batch unless tagged-species source apportionment is implemented.",
            "The screened sparse candidate manifest has not been validated by official CALPUFF and cannot be used to claim omitted responses are zero.",
            "Dense matrix storage should be avoided; use sparse per-hour matrices.",
        ],
    }
    out = partition_dir / "official_calpuff_readiness.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(out)
    return 0


def _tool_status(config: dict[str, object]) -> dict[str, object]:
    paths = config["calpuff"].get("executable_paths", {})
    names = {
        "calmet": paths.get("calmet"),
        "calpuff": paths.get("calpuff"),
        "calpost": paths.get("calpost"),
        "wgrib2": paths.get("wgrib2"),
        "calwrf": paths.get("calwrf"),
        "mmif": paths.get("mmif"),
    }
    status = {}
    for name, configured in names.items():
        candidates = []
        if configured:
            candidates.append(Path(configured))
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
        existing = [str(path) for path in candidates if path.exists()]
        status[name] = {"configured": configured, "exists": bool(existing), "resolved": existing[:1]}
    status["calwrf_or_mmif"] = {
        "exists": bool(status["calwrf"]["exists"] or status["mmif"]["exists"]),
        "resolved": status["calwrf"]["resolved"] + status["mmif"]["resolved"],
    }
    return status


def _template_status(config: dict[str, object]) -> dict[str, object]:
    calpuff_template = ROOT / str(config["calpuff"].get("template_file", ""))
    calmet_template = ROOT / "config" / "calmet_20250623_18z.inp.tpl"
    calpost_template = ROOT / "config" / "calpost_20250623_18z.inp.tpl"
    return {
        "calmet_template": str(calmet_template),
        "calmet_template_ready": calmet_template.exists() and "PLACEHOLDER" not in calmet_template.read_text(encoding="utf-8", errors="ignore"),
        "calpuff_template": str(calpuff_template),
        "calpuff_template_ready": calpuff_template.exists() and "PLACEHOLDER" not in calpuff_template.read_text(encoding="utf-8", errors="ignore"),
        "calpost_template": str(calpost_template),
        "calpost_template_ready": calpost_template.exists() and "PLACEHOLDER" not in calpost_template.read_text(encoding="utf-8", errors="ignore"),
    }


def _runtime_estimates(case_count: int, seconds_per_case: list[float], workers: list[int]) -> list[dict[str, float]]:
    rows = []
    for sec in seconds_per_case:
        cpu_days = case_count * sec / 86400.0
        for worker_count in workers:
            rows.append(
                {
                    "seconds_per_case": sec,
                    "workers": worker_count,
                    "cpu_days": cpu_days,
                    "wall_days_ideal": cpu_days / worker_count,
                    "wall_days_with_25pct_overhead": cpu_days / worker_count * 1.25,
                    "cases_per_day_required": case_count / max(cpu_days / worker_count * 1.25, 1e-9),
                }
            )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
