from __future__ import annotations

import csv
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import CaseConfig


@dataclass(frozen=True)
class HrrrFile:
    product: str
    forecast_hour: int
    url: str
    local_path: Path


def build_hrrr_inventory(config: CaseConfig) -> list[HrrrFile]:
    hrrr = config.data["hrrr"]
    run_date = hrrr["run_date"].replace("-", "")
    cycle = int(hrrr["cycle_utc"])
    base_url = hrrr["aws_base_url"].rstrip("/")
    domain = hrrr["domain"]
    raw_dir = config.root / "data" / "raw" / "hrrr" / f"hrrr.{run_date}"
    files: list[HrrrFile] = []

    for forecast_hour in range(int(hrrr["forecast_hours"])):
        for product in hrrr["products"]:
            name = f"hrrr.t{cycle:02d}z.{product}{forecast_hour:02d}.grib2"
            url = f"{base_url}/hrrr.{run_date}/{domain}/{name}"
            files.append(
                HrrrFile(
                    product=product,
                    forecast_hour=forecast_hour,
                    url=url,
                    local_path=raw_dir / name,
                )
            )
    return files


def write_manifest(config: CaseConfig, files: list[HrrrFile]) -> Path:
    manifest = config.output_path("hrrr_manifest")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["product", "forecast_hour", "url", "local_path", "exists"],
        )
        writer.writeheader()
        for item in files:
            writer.writerow(
                {
                    "product": item.product,
                    "forecast_hour": item.forecast_hour,
                    "url": item.url,
                    "local_path": str(item.local_path),
                    "exists": item.local_path.exists(),
                }
            )
    return manifest


def fetch_hrrr(config: CaseConfig, manifest_only: bool = False) -> Path:
    files = build_hrrr_inventory(config)
    if not manifest_only:
        for item in files:
            item.local_path.parent.mkdir(parents=True, exist_ok=True)
            if item.local_path.exists():
                continue
            urllib.request.urlretrieve(item.url, item.local_path)
    return write_manifest(config, files)

