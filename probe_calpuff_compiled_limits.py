"""Probe compiled CALPUFF array limits without requiring meteorology.

The probe uses the official distribution seed, changes only NREC, and runs in
a temporary directory. It never writes to the CALPUFF installation directory.
If the executable rejects NREC before opening CALMET.DAT, its error message
reveals the compiled MXREC value. Otherwise the result is explicitly reported
as inconclusive rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi" / "templates" / "CALPUFF_7.0_seed_from_distribution.INP"
DEFAULT_EXE = Path(os.environ.get("CALPUFF_EXE", "calpuff_v7.2.1.exe"))


def _make_control(seed: str, nrec: int) -> str:
    pattern = re.compile(r"(!\s*NREC\s*=\s*)\d+(\s*!\s*)")
    updated, count = pattern.subn(rf"\g<1>{nrec}\g<2>", seed, count=1)
    if count != 1:
        raise RuntimeError("Could not locate the discrete-receptor NREC assignment in the seed file.")
    return updated


def _probe(exe: Path, seed: Path, requested_nrec: int) -> dict[str, object]:
    seed_text = seed.read_text(encoding="ascii", errors="ignore")
    with tempfile.TemporaryDirectory(prefix="calpuff_limit_probe_") as temp_name:
        temp_dir = Path(temp_name)
        control = temp_dir / "CALPUFF.INP"
        control.write_text(_make_control(seed_text, requested_nrec), encoding="ascii")
        completed = subprocess.run(
            [str(exe), str(control)],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
        list_files = list(temp_dir.glob("*.LST")) + list(temp_dir.glob("*.lst"))
        list_text = "\n".join(path.read_text(encoding="latin-1", errors="ignore") for path in list_files)
        combined += "\n" + list_text
        match = re.search(r"MXREC\s*=\s*(\d+)", combined, re.IGNORECASE)
        too_many = bool(re.search(r"too many discrete", combined, re.IGNORECASE))
        if too_many and match:
            status = "compiled_limit_observed"
            compiled_mxrec = int(match.group(1))
        elif re.search(r"CALMET|METDAT|opening.*control|file.*not found", combined, re.IGNORECASE):
            status = "inconclusive_reached_other_input_check"
            compiled_mxrec = int(match.group(1)) if match else None
        else:
            status = "inconclusive"
            compiled_mxrec = int(match.group(1)) if match else None
        return {
            "requested_nrec": requested_nrec,
            "return_code": completed.returncode,
            "status": status,
            "compiled_mxrec_if_reported": compiled_mxrec,
            "message_tail": combined[-2000:],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--requested-nrec", type=int, default=45378)
    parser.add_argument("--output", type=Path, default=ROOT / "official_calpuff" / "case_20250623_18z_30sqmi" / "outputs" / "calpuff_limit_probe_20260727.json")
    args = parser.parse_args()
    if not args.exe.exists():
        raise FileNotFoundError(args.exe)
    if not args.seed.exists():
        raise FileNotFoundError(args.seed)
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "executable": str(args.exe),
        "seed": str(args.seed),
        "probe": _probe(args.exe, args.seed, args.requested_nrec),
        "scientific_use": "Do not use the result as a CALPUFF run; it is only a compiled-array-capacity check.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
