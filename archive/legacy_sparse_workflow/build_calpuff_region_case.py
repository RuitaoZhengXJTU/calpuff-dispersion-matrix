from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from build_calpuff_smoke_case import _read_rows, _replace_between, _replace_first_assignment


ROOT = Path(__file__).resolve().parent
CASE_ROOT = ROOT / "official_calpuff" / "case_20250623_18z_30sqmi"
SEED = CASE_ROOT / "templates" / "CALPUFF_7.0_seed_from_distribution.INP"
SOURCES = CASE_ROOT / "inputs" / "sources_16_per_region.csv"
RECEPTORS = CASE_ROOT / "inputs" / "receptors_9_per_region.csv"
LB_PER_HOUR_TO_G_S = 453.59237 / 3600.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one formal-baseline CALPUFF source-region/hour case."
    )
    parser.add_argument("--source-region-index", type=int, required=True)
    parser.add_argument("--hour-index", type=int, required=True)
    parser.add_argument("--receptor-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calmet-dat",
        default=r"..\met\calmet_hrrr\CALMET.DAT",
        help="CALMET.DAT path written into CALPUFF.INP, relative to output-dir",
    )
    parser.add_argument("--emission-lb-per-hour", type=float, default=1.0)
    parser.add_argument("--release-height-m", type=float, default=15.0)
    parser.add_argument("--sigma-y-m", type=float, default=250.0)
    parser.add_argument("--sigma-z-m", type=float, default=20.0)
    args = parser.parse_args()

    if args.source_region_index < 0 or args.hour_index < 0 or args.hour_index >= 24:
        raise ValueError("source-region-index must be nonnegative and hour-index must be in [0, 23]")
    if args.emission_lb_per_hour <= 0 or args.release_height_m < 0:
        raise ValueError("emission and release height must be nonnegative, with positive emission")
    if args.sigma_y_m <= 0 or args.sigma_z_m <= 0:
        raise ValueError("initial sigmas must be positive")

    sources = [
        row
        for row in _read_rows(SOURCES)
        if int(row["matrix_index"]) == args.source_region_index
    ]
    if len(sources) != 16:
        raise ValueError(
            f"source region {args.source_region_index} has {len(sources)} rows; expected 16"
        )
    receptors = _read_rows(args.receptor_manifest)
    if not receptors:
        raise ValueError("receptor manifest is empty")
    required_receptors = {"receptor_id", "region_id", "matrix_index", "x_m", "y_m"}
    missing = sorted(required_receptors - set(receptors[0]))
    if missing:
        raise ValueError(f"receptor manifest missing columns: {missing}")

    start = datetime(2025, 6, 23, 18, tzinfo=timezone.utc) + timedelta(hours=args.hour_index)
    end = start + timedelta(hours=1)
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
        "IBYR": str(start.year),
        "IBMO": str(start.month),
        "IBDY": str(start.day),
        "IBHR": str(start.hour),
        "IBMIN": str(start.minute),
        "IBSEC": str(start.second),
        "IEYR": str(end.year),
        "IEMO": str(end.month),
        "IEDY": str(end.day),
        "IEHR": str(end.hour),
        "IEMIN": str(end.minute),
        "IESEC": str(end.second),
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
        "NVL1": str(len(sources)),
        "IVLU": "1",
        "NSVL1": "0",
        "NVL2": "0",
        "NREC": str(len(receptors)),
        "NRGRP": "0",
    }
    for key, value in replacements.items():
        _replace_first_assignment(lines, key, value)

    # The 16b dictionary is read repeatedly until NVL1 source blocks have been
    # consumed. Keep each SRCNAM/X pair adjacent; the one-source seed hides
    # this ordering requirement.
    source_lines: list[str] = ["Subgroup (16b)", "VOLUME SOURCE: CONSTANT DATA"]
    total_g_s = args.emission_lb_per_hour * LB_PER_HOUR_TO_G_S
    source_name_map: dict[str, str] = {}
    for number, source in enumerate(sources, start=1):
        fraction = float(source.get("release_fraction", 1.0 / len(sources)))
        q_no2 = total_g_s * fraction
        control_source_name = f"V{number:04d}"
        source_name_map[control_source_name] = str(source["source_id"])
        source_lines.append(f"  {number} ! SRCNAM = {control_source_name} !")
        source_lines.append(
            f"  {number} ! X = {float(source['x_m']) / 1000.0:.6f}, "
            f"{float(source['y_m']) / 1000.0:.6f}, {args.release_height_m:.3f}, "
            f"0.0, {args.sigma_y_m:.3f}, {args.sigma_z_m:.3f}, "
            f"0.0, 0.0, 0.0, {q_no2:.10g}, 0.0, 0.0, 0.0 ! !END!"
        )
    lines = _replace_between(
        lines, "Subgroup (16b)", "Subgroup (16c)", source_lines + ["", "Subgroup (16c)"]
    )

    # Remove point-source examples because NPT1=0.
    lines = _replace_between(
        lines,
        "Subgroup (13b)",
        "INPUT GROUPS: 14a",
        [
            "Subgroup (13b)",
            "POINT SOURCE: CONSTANT DATA (none in this case)",
            "Subgroup (13c)",
            "POINT SOURCE BUILDING DATA (none in this case)",
            "Subgroup (13d)",
            "POINT SOURCE SCALING DATA (none in this case)",
            "",
        ],
    )
    lines = _replace_between(
        lines,
        "Subgroup (20b)",
        "Subgroup (20c)",
        ["Subgroup (20b)", "RECEPTOR GROUP DATA (none in this case; NRGRP=0)", "", "Subgroup (20c)"],
    )
    receptor_lines = ["Subgroup (20c)", "NON-GRIDDED (DISCRETE) RECEPTOR DATA"]
    for number, receptor in enumerate(receptors, start=1):
        receptor_lines.append(
            f" {number:5d} ! X = {float(receptor['x_m']) / 1000.0:.6f}, "
            f"{float(receptor['y_m']) / 1000.0:.6f}, 0.0, "
            f"{float(receptor.get('receptor_height_m', 1.5)):.3f} ! !END!"
        )
    lines = _replace_between(lines, "Subgroup (20c)", None, receptor_lines)

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

    # Persist the modeled NO2 group for CALPOST while keeping chemistry off.
    output_start = next(
        index
        for index, line in enumerate(lines)
        if "SPECIES (or GROUP for combined species)" in line
    )
    output_end = next(index for index in range(output_start, len(lines)) if "!END!" in lines[index])
    for index in range(output_start, output_end):
        if "NO2 =" in lines[index] and "ioutop" not in lines[index].lower():
            lines[index] = "!          NO2 =     0,           1,           0,           1,           0,           1,           0   !"

    output = args.output_dir / "CALPUFF.INP"
    output.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
    metadata = {
        "source_region_index": args.source_region_index,
        "hour_index": args.hour_index,
        "start_utc": start.isoformat().replace("+00:00", "Z"),
        "end_utc_exclusive": end.isoformat().replace("+00:00", "Z"),
        "source_count": len(sources),
        "source_name_map": source_name_map,
        "receptor_count": len(receptors),
        "total_emission_lb_per_hour": args.emission_lb_per_hour,
        "emission_unit_in_control": "g/s",
        "total_emission_g_per_s": total_g_s,
        "release_height_m": args.release_height_m,
        "sigma_y_m": args.sigma_y_m,
        "sigma_z_m": args.sigma_z_m,
        "pollutant": "passive NO2-equivalent tracer",
        "chemistry": "off",
        "receptor_manifest": str(args.receptor_manifest.resolve()),
        "calmet_dat": args.calmet_dat,
    }
    (args.output_dir / "CASE_METADATA.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
