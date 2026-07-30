from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEED = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi" / "templates" / "CALPUFF_7.0_seed_from_distribution.INP"
DEFAULT_OUT = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi" / "smoke_no2_1h"
SOURCES = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi" / "inputs" / "sources_16_per_region.csv"
RECEPTORS = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi" / "inputs" / "receptors_9_per_region.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a gated one-source CALPUFF smoke case from the official seed control.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--region-index", type=int, default=0)
    parser.add_argument("--emission-g-s", type=float, default=1.0)
    parser.add_argument("--source-receptor-index", type=int, default=None)
    parser.add_argument("--sigma-y-m", type=float, default=50.0)
    parser.add_argument("--sigma-z-m", type=float, default=50.0)
    parser.add_argument(
        "--calmet-dat",
        default=r"..\\met\\calmet_surrogate\\CALMET.DAT",
        help="CALMET.DAT path as written into CALPUFF.INP, relative to --output-dir",
    )
    args = parser.parse_args()

    source_rows = _read_rows(SOURCES)
    receptor_rows = _read_rows(RECEPTORS)
    source = next(row for row in source_rows if int(row["matrix_index"]) == args.region_index)
    receptors = [row for row in receptor_rows if int(row["matrix_index"]) == args.region_index]
    if len(receptors) < 9:
        raise ValueError("the selected region does not have nine receptor samples")
    source = source_rows[args.region_index * 16 + args.source_index]
    source_x_m = float(source["x_m"])
    source_y_m = float(source["y_m"])
    if args.source_receptor_index is not None:
        if not 0 <= args.source_receptor_index < 9:
            raise ValueError("--source-receptor-index must be in [0, 8]")
        source_x_m = float(receptors[args.source_receptor_index]["x_m"])
        source_y_m = float(receptors[args.source_receptor_index]["y_m"])
    if args.sigma_y_m <= 0 or args.sigma_z_m <= 0:
        raise ValueError("initial sigmas must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lines = SEED.read_text(encoding="utf-8", errors="replace").splitlines()
    replacements = {
        "METDAT": args.calmet_dat,
        "PUFLST": "CALPUFF.LST",
        "CONDAT": "CALPUFF.CON",
        "DFDAT": "",
        "WFDAT": "",
        "VISDAT": "",
        "OZDAT": "",
        "AUXEXT": "aux",
        "METRUN": "0",
        "IBYR": "2025",
        "IBMO": "6",
        "IBDY": "23",
        "IBHR": "18",
        "IBMIN": "0",
        "IBSEC": "0",
        "IEYR": "2025",
        "IEMO": "6",
        "IEDY": "23",
        "IEHR": "19",
        "IEMIN": "0",
        "IESEC": "0",
        "ABTZ": "UTC+0000",
        "NSECDT": "3600",
        "NSPEC": "7",
        "NSE": "1",
        "ITEST": "2",
        "METFM": "1",
        "MPRFFM": "1",
        "MCHEM": "0",
        "MWET": "0",
        "MDRY": "0",
        "MREG": "0",
        "MBCON": "0",
        "MFOG": "0",
        "MRESTART": "0",
        "NRESPD": "0",
        "PMAP": "LCC",
        "DATUM": "WGS-84",
        "FEAST": "0.0",
        "FNORTH": "0.0",
        "RLAT0": "N37.0",
        "RLON0": "W77.5",
        "XLAT1": "N33.0",
        "XLAT2": "N39.5",
        "NX": "79",
        "NY": "38",
        "NZ": "10",
        "ZFACE": "0.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1200.0, 2000.0, 3000.0, 4000.0",
        "DGRIDKM": "10.0",
        "XORIGKM": "-560.0",
        "YORIGKM": "-60.0",
        "IBCOMP": "1",
        "JBCOMP": "1",
        "IECOMP": "79",
        "JECOMP": "38",
        "LSAMP": "F",
        "IBSAMP": "1",
        "JBSAMP": "1",
        "IESAMP": "1",
        "JESAMP": "1",
        "MESHDN": "1",
        "ICON": "1",
        "IDRY": "0",
        "IWET": "0",
        "IVIS": "0",
        "ICPRT": "0",
        "IDPRT": "0",
        "IWPRT": "0",
        "IMFLX": "0",
        "IMBAL": "0",
        "IQAPLOT": "0",
        "IPFTRAK": "0",
        "IT2D": "0",
        "IRHO": "0",
        "NPT1": "0",
        "NSPT1": "0",
        "NPT2": "0",
        "NAR1": "0",
        "NSAR1": "0",
        "NAR2": "0",
        "NLN2": "0",
        "NLINES": "0",
        "NSLN1": "0",
        "NVL1": "1",
        "IVLU": "1",
        "NSVL1": "0",
        "NVL2": "0",
        "NREC": "9",
        "NRGRP": "0",
    }
    for key, value in replacements.items():
        _replace_first_assignment(lines, key, value)

    lines = _replace_between(
        lines,
        "Subgroup (16b)",
        "Subgroup (16c)",
        [
            "Subgroup (16b)",
            "VOLUME SOURCE: CONSTANT DATA",
            f"  1 ! SRCNAM = V001 !",
            "  1 ! X = {x:.6f}, {y:.6f}, 15.0, 0.0, {sy:.6f}, {sz:.6f}, 0.0, 0.0, 0.0, {no2:.8g}, 0.0, 0.0, 0.0 ! !END!".format(
                x=source_x_m / 1000.0,
                y=source_y_m / 1000.0,
                sy=args.sigma_y_m,
                sz=args.sigma_z_m,
                no2=args.emission_g_s,
            ),
            "",
            "Subgroup (16c)",
        ],
    )

    # With NPT1=0 the reader skips point-source subgroups entirely. Remove
    # the distribution seed's example point-source assignments so they are
    # not mistaken for the next source-type dictionary.
    lines = _replace_between(
        lines,
        "Subgroup (13b)",
        "INPUT GROUPS: 14a",
        [
            "Subgroup (13b)",
            "POINT SOURCE: CONSTANT DATA (none in this smoke case)",
            "Subgroup (13c)",
            "POINT SOURCE BUILDING DATA (none in this smoke case)",
            "Subgroup (13d)",
            "POINT SOURCE SCALING DATA (none in this smoke case)",
            "",
        ],
    )

    # With NRGRP=0 the reader does not expect any receptor-group names.
    # Remove the distribution seed's active RGRPNAM example before group 20c.
    lines = _replace_between(
        lines,
        "Subgroup (20b)",
        "Subgroup (20c)",
        [
            "Subgroup (20b)",
            "RECEPTOR GROUP DATA (none in this smoke case; NRGRP=0)",
            "",
            "Subgroup (20c)",
        ],
    )

    receptor_lines = [
        "Subgroup (20c)",
        "NON-GRIDDED (DISCRETE) RECEPTOR DATA",
    ]
    for number, row in enumerate(receptors[:9], start=1):
        receptor_lines.append(
            f" {number:2d} ! X = {float(row['x_m']) / 1000.0:.6f}, {float(row['y_m']) / 1000.0:.6f}, 0.0, {float(row['receptor_height_m']):.3f} ! !END!"
        )
    lines = _replace_between(lines, "Subgroup (20c)", None, receptor_lines)

    # Only NO2 is emitted. Restrict this edit to species-definition group 3a;
    # the output and chemistry groups use different field counts.
    group3_start = next(index for index, line in enumerate(lines) if "Subgroup (3a)" in line)
    group4_start = next(index for index, line in enumerate(lines) if "INPUT GROUP: 4" in line)
    species_values = {
        "SO2": "!          SO2  =         1,               0,           0,                 0   !",
        "SO4": "!          SO4  =         1,               0,           0,                 0   !",
        "NO": "!           NO  =         1,               0,           0,                 0   !",
        "NO2": "!          NO2  =         1,               1,           0,                 0   !",
        "HNO3": "!         HNO3  =         1,               0,           0,                 0   !",
        "NO3": "!          NO3  =         1,               0,           0,                 0   !",
        "PM10": "!         PM10  =         1,               0,           0,                 0   !",
    }
    for index in range(group3_start, group4_start):
        for species, replacement in species_values.items():
            if re.search(rf"!\s*{re.escape(species)}\s*=", lines[index]):
                lines[index] = replacement

    out = args.output_dir / "CALPUFF.INP"
    out.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
    met_kind = "surrogate" if "surrogate" in args.calmet_dat.lower() else "external/derived"
    (args.output_dir / "SMOKE_CASE_README.txt").write_text(
        "This is a gated one-source/9-receptor/1-hour CALPUFF smoke test.\n"
        f"CALMET input: {args.calmet_dat} ({met_kind}).\n"
        "This is not a formal paper result until the meteorology and controls are validated.\n",
        encoding="ascii",
    )
    print(out)
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _replace_first_assignment(lines: list[str], key: str, value: str) -> None:
    pattern = re.compile(rf"(!\s*{re.escape(key)}\s*=)(.*?)(!)", re.IGNORECASE)
    for index, line in enumerate(lines):
        if pattern.search(line):
            lines[index] = pattern.sub(lambda match: f"{match.group(1)} {value} {match.group(3)}", line, count=1)
            return
    raise ValueError(f"assignment not found in seed: {key}")


def _replace_between(lines: list[str], start_marker: str, end_marker: str | None, replacement: list[str]) -> list[str]:
    start = next(index for index, line in enumerate(lines) if start_marker in line)
    if end_marker is None:
        return lines[:start] + replacement
    end = next(index for index in range(start + 1, len(lines)) if end_marker in lines[index])
    return lines[:start] + replacement + lines[end:]


if __name__ == "__main__":
    raise SystemExit(main())
