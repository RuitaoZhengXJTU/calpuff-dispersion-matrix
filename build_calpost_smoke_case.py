from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = Path(os.environ.get("CALPOST_TEMPLATE", "data/raw/official_examples/calpost_v7.1.0_L141010/CALPOST_v7.1.0_L141010/calpost.inp"))
DEFAULT_CASE = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi" / "smoke_no2_1h"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a one-hour CALPOST discrete-receptor smoke case.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE)
    args = parser.parse_args()

    lines = args.template.read_text(encoding="ascii", errors="replace").splitlines()
    assignments = {
        "MODDAT": "CALPUFF.CON",
        "PSTLST": "CALPOST.LST",
        "LCFILES": "F",
        "METRUN": "0",
        "ISYR": "2025",
        "ISMO": "6",
        "ISDY": "23",
        "ISHR": "18",
        "ISMIN": "0",
        "ISSEC": "0",
        "IEYR": "2025",
        "IEMO": "6",
        "IEDY": "23",
        "IEHR": "19",
        "IEMIN": "0",
        "IESEC": "0",
        "ABTZ": "UTC+0000",
        "NREP": "1",
        "NSPEC": "1",
        "NO2CALC": "0",
        "LG": "F",
        "LD": "T",
        "NDRECP": "-1",
        "NDRGRP": "0",
        "LTOPN": "F",
        "LEXCD": "F",
        "LTIME": "T",
        "LPEAK": "F",
        "LPLT": "F",
        "IECHO": "173*0, 1, 192*0",
        "ASPEC": "NO2",
        "ILAYER": "1",
        "IPRTU": "0",
        "A": "0.0",
        "B": "0.0",
        "L1PD": "F",
        "L1HR": "T",
        "L3HR": "F",
        "L24HR": "F",
        "LRUNL": "F",
        "NAVGH": "0",
        "NAVGM": "0",
        "NAVGS": "0",
    }
    for key, value in assignments.items():
        _replace_first_assignment(lines, key, value)

    args.case_dir.mkdir(parents=True, exist_ok=True)
    output = args.case_dir / "CALPOST.INP"
    output.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
    (args.case_dir / "CALPOST_CASE_README.txt").write_text(
        "This control file is for a one-source/9-receptor/1-hour CALPOST smoke test.\n"
        "It processes NO2 concentration from CALPUFF.CON in the source case directory.\n",
        encoding="ascii",
    )
    print(output)
    return 0


def _replace_first_assignment(lines: list[str], key: str, value: str) -> None:
    pattern = re.compile(rf"(!\s*{re.escape(key)}\s*=)(.*?)(!)", re.IGNORECASE)
    for index, line in enumerate(lines):
        if pattern.search(line):
            lines[index] = pattern.sub(
                lambda match: f"{match.group(1)} {value} {match.group(3)}",
                line,
                count=1,
            )
            return
    raise ValueError(f"assignment not found in CALPOST template: {key}")


if __name__ == "__main__":
    raise SystemExit(main())
