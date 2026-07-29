from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import CaseConfig


def check_external_tools(config: CaseConfig, require_template: bool = True) -> dict[str, str]:
    calpuff = config.data["calpuff"]
    home = os.environ.get(calpuff["calpuff_home_env"])
    home_path = Path(home) if home else None
    found: dict[str, str] = {}
    missing: list[str] = []
    explicit_paths = calpuff.get("executable_paths", {})

    for tool, name in calpuff["executable_names"].items():
        explicit = explicit_paths.get(tool)
        candidate = Path(explicit) if explicit else None
        env_name = calpuff.get("executable_env_vars", {}).get(tool)
        env_value = os.environ.get(env_name) if env_name else None
        env_candidate = Path(env_value) if env_value else None
        home_candidate = home_path / name if home_path else None
        resolved = None
        if candidate and candidate.exists():
            resolved = candidate
        elif env_candidate and env_candidate.exists():
            resolved = env_candidate
        elif home_candidate and home_candidate.exists():
            resolved = home_candidate
        else:
            resolved = shutil.which(name)
        if not resolved:
            expected = []
            if candidate:
                expected.append(str(candidate))
            if env_name:
                expected.append(f"{env_name}")
            if home_candidate:
                expected.append(str(home_candidate))
            expected.append(f"{name} on PATH")
            missing.append(f"{tool}: expected one of {', '.join(expected)}")
            continue
        found[tool] = str(resolved)

    explicit_wgrib2 = explicit_paths.get("wgrib2")
    wgrib2 = None
    if explicit_wgrib2 and Path(explicit_wgrib2).exists():
        wgrib2 = explicit_wgrib2
    else:
        wgrib2_env_name = calpuff.get("executable_env_vars", {}).get("wgrib2", calpuff["wgrib2_env"])
        env_wgrib2 = os.environ.get(wgrib2_env_name)
        if env_wgrib2 and Path(env_wgrib2).exists():
            wgrib2 = env_wgrib2
        else:
            wgrib2 = shutil.which("wgrib2")
    if not wgrib2:
        missing.append(
            f"wgrib2: set calpuff.executable_paths.wgrib2, set {calpuff['wgrib2_env']}, "
            "or add wgrib2.exe to PATH"
        )
    else:
        found["wgrib2"] = str(wgrib2)

    if missing:
        found_text = "\n".join(f"  {name}: {path}" for name, path in sorted(found.items()))
        missing_text = "\n".join(missing)
        if found_text:
            raise RuntimeError(f"Found external tools:\n{found_text}\nMissing external tools:\n  {missing_text}")
        raise RuntimeError("Missing external tools:\n  " + missing_text)

    template = config.resolve(calpuff["template_file"])
    if require_template and _template_is_placeholder(template) and not calpuff.get("allow_placeholder_template", False):
        raise RuntimeError(
            f"{template} is a placeholder. Replace it with a verified CALPUFF input template "
            "before running real CALPUFF simulations."
        )

    return found


def _template_is_placeholder(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "PLACEHOLDER CALPUFF CONTROL TEMPLATE" in text


def build_calpuff_cases(
    config: CaseConfig,
    hours: int | None = None,
    sources: int | None = None,
    dry_run: bool = False,
) -> int:
    hours = config.hours if hours is None else hours
    sources = config.target_regions if sources is None else sources
    case_root = config.case_root()
    case_root.mkdir(parents=True, exist_ok=True)

    sources_csv = config.output_path("sources_csv")
    receptors_csv = config.output_path("receptors_csv")
    if not sources_csv.exists() or not receptors_csv.exists():
        raise RuntimeError("Build subregions first: sources.csv and receptors.csv are required.")

    source_table = pd.read_csv(sources_csv)
    source_regions = sorted(source_table["region_id"].unique())[:sources]
    start = datetime.fromisoformat(config.data["time"]["start_utc"].replace("Z", "+00:00"))
    template_path = config.resolve(config.data["calpuff"]["template_file"])
    template = template_path.read_text(encoding="utf-8")
    if _template_is_placeholder(template_path) and not dry_run:
        check_external_tools(config, require_template=True)

    count = 0
    for hour in range(hours):
        start_utc = start + timedelta(hours=hour)
        end_utc = start_utc + timedelta(hours=1)
        for region_id in source_regions:
            case_dir = case_root / f"hour_{hour:02d}" / f"source_{region_id}"
            case_dir.mkdir(parents=True, exist_ok=True)
            out_csv = case_dir / "receptors.csv"
            met_file = _met_file_for_hour(config, hour)
            rendered = _render_template(
                template,
                {
                    "CASE_ID": config.case_id,
                    "HOUR_INDEX": f"{hour:02d}",
                    "SOURCE_REGION_ID": region_id,
                    "START_UTC": start_utc.isoformat(),
                    "END_UTC": end_utc.isoformat(),
                    "SOURCES_CSV": str(sources_csv),
                    "RECEPTORS_CSV": str(receptors_csv),
                    "MET_FILE": str(met_file),
                    "OUTPUT_RECEPTOR_CSV": str(out_csv),
                },
            )
            (case_dir / "calpuff.inp").write_text(rendered, encoding="utf-8")
            (case_dir / "case_manifest.txt").write_text(
                "\n".join(
                    [
                        f"case_id={config.case_id}",
                        f"hour_index={hour}",
                        f"source_region_id={region_id}",
                        f"start_utc={start_utc.isoformat()}",
                        f"end_utc={end_utc.isoformat()}",
                        f"output_receptor_csv={out_csv}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            count += 1
    return count


def _met_file_for_hour(config: CaseConfig, hour: int) -> Path:
    run_date = config.data["hrrr"]["run_date"].replace("-", "")
    cycle = int(config.data["hrrr"]["cycle_utc"])
    return config.root / "data" / "raw" / "hrrr" / f"hrrr.{run_date}" / f"hrrr.t{cycle:02d}z.wrfprsf{hour:02d}.grib2"


def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def run_calpuff_cases(
    config: CaseConfig,
    max_workers: int = 1,
    hours: int | None = None,
    sources: int | None = None,
) -> int:
    tools = check_external_tools(config, require_template=True)
    calpuff_exe = tools["calpuff"]
    case_dirs = list(_iter_case_dirs(config, hours=hours, sources=sources))
    if not case_dirs:
        raise RuntimeError("No case directories found. Run build-calpuff-cases first.")

    failures: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one_case, calpuff_exe, case_dir): case_dir for case_dir in case_dirs}
        for future in as_completed(futures):
            case_dir = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - preserve all failing case context.
                failures.append((case_dir, str(exc)))

    if failures:
        sample = "\n".join(f"{path}: {message}" for path, message in failures[:5])
        raise RuntimeError(f"{len(failures)} CALPUFF cases failed. First failures:\n{sample}")
    return len(case_dirs)


def _iter_case_dirs(config: CaseConfig, hours: int | None, sources: int | None):
    hour_limit = config.hours if hours is None else hours
    source_limit = config.target_regions if sources is None else sources
    for hour in range(hour_limit):
        hour_dir = config.case_root() / f"hour_{hour:02d}"
        if not hour_dir.exists():
            continue
        # Region identifiers are not required to start with ``r``. The current
        # population partition uses identifiers such as ``area_pop_000000``.
        for case_dir in sorted(hour_dir.glob("source_*"))[:source_limit]:
            if (case_dir / "calpuff.inp").exists():
                yield case_dir


def _run_one_case(calpuff_exe: str, case_dir: Path) -> None:
    result = subprocess.run(
        [calpuff_exe],
        cwd=case_dir,
        text=True,
        input="calpuff.inp\n",
        capture_output=True,
        check=False,
    )
    (case_dir / "calpuff.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (case_dir / "calpuff.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"return code {result.returncode}")
