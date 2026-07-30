from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")
TIME_RE = re.compile(r"^\s*(\d{4})\s+(\d{1,3})\s+(\d{4})\s+(.*)$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a CALPOST time-series text file into explicit receptor rows."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--receptor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--value-unit", default="g/m3")
    parser.add_argument(
        "--matrix-index",
        type=int,
        default=None,
        help="explicitly select one region from a multi-region manifest (smoke tests only)",
    )
    args = parser.parse_args(argv)

    output = parse_calpost_tseries(
        input_path=args.input,
        receptor_manifest=args.receptor_manifest,
        output_path=args.output,
        start_utc=args.start_utc,
        value_unit=args.value_unit,
        matrix_index=args.matrix_index,
    )
    print(output)
    return 0


def parse_calpost_tseries(
    input_path: Path,
    receptor_manifest: Path,
    output_path: Path,
    start_utc: str,
    value_unit: str = "g/m3",
    matrix_index: int | None = None,
) -> Path:
    manifest = pd.read_csv(receptor_manifest)
    required = {"receptor_id", "region_id", "matrix_index"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"receptor manifest missing columns: {missing}")
    if manifest["receptor_id"].duplicated().any():
        raise ValueError("receptor manifest contains duplicate receptor_id values")
    if manifest.empty:
        raise ValueError("receptor manifest is empty")
    if matrix_index is not None:
        manifest = manifest.loc[manifest["matrix_index"] == matrix_index].copy()
        if manifest.empty:
            raise ValueError(f"manifest contains no matrix_index={matrix_index}")

    expected_time = _parse_utc(start_utc)
    expected_count = len(manifest)
    values = _read_target_values(input_path, expected_time, expected_count)
    if len(values) != expected_count:
        raise ValueError(
            f"CALPOST value count {len(values)} does not match manifest count {expected_count}"
        )

    result = manifest[["receptor_id", "region_id", "matrix_index"]].copy()
    result["concentration"] = values
    result["value_unit"] = value_unit
    result["source_time_utc"] = expected_time.isoformat().replace("+00:00", "Z")
    result["calpost_file"] = str(input_path.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return output_path


def _read_target_values(path: Path, expected_time: datetime, expected_count: int) -> list[float]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    target_values: list[float] | None = None
    active_time: datetime | None = None
    active_values: list[float] = []

    def flush() -> None:
        nonlocal target_values
        if active_time == expected_time:
            if target_values is not None:
                raise ValueError("CALPOST contains duplicate target time rows")
            if len(active_values) != expected_count:
                raise ValueError(
                    f"CALPOST target row has {len(active_values)} values; expected {expected_count}"
                )
            target_values = list(active_values)

    for line in lines:
        match = TIME_RE.match(line)
        if match:
            flush()
            active_time = _from_jdy(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            active_values = _float_tokens(match.group(4))
            continue
        if active_time is None or len(active_values) >= expected_count:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if tokens and all(FLOAT_RE.fullmatch(token) for token in tokens):
            active_values.extend(float(token) for token in tokens)
            if len(active_values) > expected_count:
                raise ValueError(
                    f"CALPOST target row contains more than {expected_count} values"
                )
    flush()

    if target_values is None:
        raise ValueError(f"CALPOST file contains no row at {expected_time.isoformat()}")
    if any(value < 0 for value in target_values):
        raise ValueError("CALPOST output contains negative concentrations")
    return target_values


def _float_tokens(text: str) -> list[float]:
    tokens = text.split()
    if not tokens or not all(FLOAT_RE.fullmatch(token) for token in tokens):
        return []
    return [float(token) for token in tokens]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("start-utc must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _from_jdy(year: int, jday: int, hhmm: int) -> datetime:
    hour, minute = divmod(hhmm, 100)
    return datetime(year, 1, 1, hour, minute, tzinfo=timezone.utc).replace(
        day=1
    ) + pd.Timedelta(days=jday - 1).to_pytimedelta()


if __name__ == "__main__":
    raise SystemExit(main())
