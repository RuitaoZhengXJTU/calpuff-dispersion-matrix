from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "hrrr_20250623_18z"
DEFAULT_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
LEVELS = ("1000 mb", "925 mb", "850 mb", "700 mb", "500 mb")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download only selected indexed HRRR GRIB2 messages using HTTP byte ranges."
    )
    parser.add_argument("--date", default="20250623")
    parser.add_argument("--cycle", type=int, default=18)
    parser.add_argument(
        "--start-hour",
        type=int,
        default=0,
        help="first forecast hour to download from the selected HRRR cycle",
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="write the inventory here instead of <output-dir>/hrrr_selected_manifest.json",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload selected GRIB2 files that already exist",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.cycle <= 23:
        raise ValueError("cycle must be between 0 and 23")
    if not 0 <= args.start_hour <= 48:
        raise ValueError("start-hour must be between 0 and 48")
    if not 1 <= args.hours <= 48 or args.start_hour + args.hours > 48:
        raise ValueError("hours must be between 1 and 48")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_records: list[dict[str, object]] = []
    for offset_hour in range(args.hours):
        forecast_hour = args.start_hour + offset_hour
        filename = f"hrrr.t{args.cycle:02d}z.wrfsfcf{forecast_hour:02d}.grib2"
        url = hrrr_surface_url(args.base_url, args.date, args.cycle, forecast_hour)
        idx_path = args.output_dir / f"{filename}.idx"
        if not idx_path.exists():
            _download(url + ".idx", idx_path)
        entries = _read_index(idx_path)
        source_size = _content_length(url)
        for index, entry in enumerate(entries):
            entry["end"] = (
                int(entries[index + 1]["offset"]) - 1
                if index + 1 < len(entries)
                else source_size - 1
            )
        selected = [entry for entry in entries if _is_selected(entry["description"])]
        if forecast_hour == 0:
            terrain_selected = [entry for entry in entries if _is_terrain(entry["description"])]
            selected.extend(entry for entry in terrain_selected if entry not in selected)
        selected.sort(key=lambda entry: entry["offset"])
        if not selected:
            raise RuntimeError(f"no required GRIB2 messages found in {idx_path}")
        record: dict[str, object] = {
            "forecast_hour": forecast_hour,
            "filename": filename,
            "source_url": url,
            "index_path": idx_path.as_posix(),
            "source_size_bytes": source_size,
            "selected_messages": [
                {
                    "line": item["line"],
                    "offset": item["offset"],
                    "end": item["end"],
                    "byte_range": [item["offset"], item["end"]],
                    "description": item["description"],
                }
                for item in selected
            ],
        }
        if not args.dry_run:
            subset_path = args.output_dir / f"{filename}.selected.grib2"
            if not subset_path.exists() or args.force:
                _download_ranges(url, selected, source_size, subset_path)
            record["subset_path"] = subset_path.as_posix()
            record["subset_size_bytes"] = subset_path.stat().st_size
            record["subset_sha256"] = _sha256(subset_path)
        selected_records.append(record)
        print(f"f{forecast_hour:02d}: {len(selected)} messages; source={source_size:,} bytes")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "NOAA HRRR AWS byte-range extraction using GRIB2 index files",
        "date": args.date,
        "cycle_utc": args.cycle,
        "forecast_hours": args.hours,
        "base_url": args.base_url,
        "selection": {
            "surface": [
                "UGRD:10 m above ground",
                "VGRD:10 m above ground",
                "TMP:2 m above ground",
                "RH:2 m above ground",
                "PRES:surface",
                "HPBL:surface",
                "PRATE:surface",
                "TCDC:entire atmosphere",
            ],
            "upper_air_levels": list(LEVELS),
            "terrain": "HGT:surface from forecast hour 0",
        },
        "dry_run": args.dry_run,
        "files": selected_records,
        "warning": "Selected GRIB2 messages are HRRR model fields. They are not WRF wrfout NetCDF and are not direct MMIF input.",
    }
    manifest_path = args.manifest_path or args.output_dir / "hrrr_selected_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)
    return 0


def verify_manifest(manifest_path: Path, project_root: Path) -> dict[str, object]:
    """Verify locally available selected HRRR files without network access."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"HRRR manifest has no files list: {manifest_path}")
    report: dict[str, object] = {
        "manifest": str(manifest_path),
        "checked": 0,
        "errors": [],
        "files": [],
    }
    errors: list[str] = report["errors"]  # type: ignore[assignment]
    checks: list[dict[str, object]] = report["files"]  # type: ignore[assignment]
    for record in files:
        if not isinstance(record, dict):
            errors.append("manifest contains a non-mapping file record")
            continue
        raw_path = record.get("subset_path")
        if not raw_path:
            errors.append(f"f{record.get('forecast_hour', '?')}: missing subset_path")
            continue
        path = Path(str(raw_path).replace("\\", "/"))
        if not path.is_absolute():
            path = project_root / path
        expected_size = record.get("subset_size_bytes")
        expected_hash = record.get("subset_sha256")
        entry: dict[str, object] = {"path": str(path), "ok": False}
        if not path.exists():
            entry["error"] = "missing"
            errors.append(f"missing HRRR selected file: {path}")
        elif expected_size is not None and path.stat().st_size != int(expected_size):
            entry["error"] = f"size {path.stat().st_size} != expected {expected_size}"
            errors.append(f"HRRR size mismatch: {path}")
        elif expected_hash and _sha256(path) != str(expected_hash):
            entry["error"] = "sha256 mismatch"
            errors.append(f"HRRR SHA-256 mismatch: {path}")
        else:
            entry["ok"] = True
        checks.append(entry)
        report["checked"] = int(report["checked"]) + 1
    report["ok"] = not errors
    return report


def enrich_manifest(manifest_path: Path, project_root: Path) -> None:
    """Add byte ranges and normalize paths to a pre-existing selected-file manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"HRRR manifest has no files list: {manifest_path}")
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("HRRR manifest contains a non-mapping record")
        index_path = Path(str(record["index_path"]).replace("\\", "/"))
        if not index_path.is_absolute():
            index_path = project_root / index_path
        source_size = int(record["source_size_bytes"])
        entries = _read_index(index_path)
        offsets = {int(entry["offset"]): int(entries[index + 1]["offset"]) - 1 if index + 1 < len(entries) else source_size - 1 for index, entry in enumerate(entries)}
        messages = record.get("selected_messages")
        if not isinstance(messages, list):
            raise ValueError(f"HRRR manifest record has no selected_messages: {index_path}")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"invalid selected message in {index_path}")
            start = int(message["offset"])
            end = offsets[start]
            message["end"] = end
            message["byte_range"] = [start, end]
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "dc-va-md-calpuff-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as handle:
        handle.write(response.read())


def hrrr_surface_url(base_url: str, date: str, cycle: int, forecast_hour: int) -> str:
    """Return the NOAA HRRR surface-product URL used by the raw-input workflow."""
    return (
        f"{base_url.rstrip('/')}/hrrr.{date}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsfcf{forecast_hour:02d}.grib2"
    )


def _content_length(url: str) -> int:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "dc-va-md-calpuff-reproduction/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
    if not value:
        raise RuntimeError(f"HRRR server did not return Content-Length for {url}")
    return int(value)


def _read_index(path: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for raw in path.read_text(encoding="ascii").splitlines():
        parts = raw.split(":")
        if len(parts) < 6:
            raise ValueError(f"unexpected HRRR index line: {raw}")
        entries.append(
            {
                "line": raw,
                "offset": int(parts[1]),
                "description": f"{parts[3]}:{parts[4]}",
            }
        )
    return entries


def _is_selected(description: str) -> bool:
    surface = (
        "UGRD:10 m above ground",
        "VGRD:10 m above ground",
        "TMP:2 m above ground",
        "RH:2 m above ground",
        "PRES:surface",
        "HPBL:surface",
        "PRATE:surface",
        "TCDC:entire atmosphere",
    )
    upper = tuple(f"{field}:{level}" for level in LEVELS for field in ("HGT", "TMP", "UGRD", "VGRD"))
    return description in surface or description in upper


def _is_terrain(description: str) -> bool:
    return description == "HGT:surface"


def _download_ranges(url: str, selected: list[dict[str, object]], source_size: int, output: Path) -> None:
    with output.open("wb") as handle:
        for index, entry in enumerate(selected):
            start = int(entry["offset"])
            end = int(entry["end"])
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "dc-va-md-calpuff-reproduction/1.0",
                    "Range": f"bytes={start}-{end}",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                status = getattr(response, "status", None)
                if status not in (206, 200):
                    raise RuntimeError(f"unexpected HTTP status {status} for byte range {start}-{end}")
                payload = response.read()
            expected = end - start + 1
            if len(payload) != expected:
                raise RuntimeError(
                    f"short HRRR range response for {start}-{end}: {len(payload)} != {expected}"
                )
            handle.write(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
