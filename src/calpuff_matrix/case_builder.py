from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping


LB_PER_HOUR_TO_G_S = 453.59237 / 3600.0
DEFAULT_START = datetime(2025, 6, 23, 18, tzinfo=timezone.utc)


@dataclass(frozen=True)
class CalpuffDomain:
    """Projection and grid settings shared by CALMET and each CALPUFF case."""

    projected_crs: str = (
        "+proj=lcc +lat_1=33 +lat_2=39.5 +lat_0=37 +lon_0=-77.5 "
        "+datum=WGS84 +units=m +no_defs"
    )
    pmap: str = "LCC"
    datum: str = "WGS-84"
    feast_km: float = 0.0
    fnorth_km: float = 0.0
    rlat0: str = "N37.0"
    rlon0: str = "W77.5"
    xlat1: str = "N33.0"
    xlat2: str = "N39.5"
    nx: int = 79
    ny: int = 38
    nz: int = 10
    zface_m: tuple[float, ...] = (
        0.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1200.0,
        2000.0, 3000.0, 4000.0,
    )
    dgrid_km: float = 10.0
    xorig_km: float = -560.0
    yorig_km: float = -60.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "CalpuffDomain":
        """Create a domain from an ``official_case`` YAML ``calpuff_domain`` block."""
        if values is None:
            return cls()
        fields = set(cls.__dataclass_fields__)
        unknown = set(values) - fields
        if unknown:
            raise ValueError(f"unknown calpuff_domain keys: {sorted(unknown)}")
        payload = dict(values)
        if "zface_m" in payload:
            payload["zface_m"] = tuple(float(value) for value in payload["zface_m"])
        for key in ("feast_km", "fnorth_km", "dgrid_km", "xorig_km", "yorig_km"):
            if key in payload:
                payload[key] = float(payload[key])
        for key in ("nx", "ny", "nz"):
            if key in payload:
                payload[key] = int(payload[key])
        domain = cls(**payload)
        if domain.nx < 1 or domain.ny < 1 or domain.nz < 1:
            raise ValueError("CALPUFF domain nx, ny, and nz must be positive")
        if len(domain.zface_m) != domain.nz + 1:
            raise ValueError("zface_m must contain nz + 1 layer faces")
        if domain.dgrid_km <= 0:
            raise ValueError("dgrid_km must be positive")
        return domain

    def control_assignments(self) -> dict[str, str]:
        """Return the CALPUFF control assignments that must match CALMET.DAT."""
        return {
            "PMAP": self.pmap,
            "DATUM": self.datum,
            "FEAST": str(self.feast_km),
            "FNORTH": str(self.fnorth_km),
            "RLAT0": self.rlat0,
            "RLON0": self.rlon0,
            "XLAT1": self.xlat1,
            "XLAT2": self.xlat2,
            "NX": str(self.nx),
            "NY": str(self.ny),
            "NZ": str(self.nz),
            "ZFACE": ", ".join(str(value) for value in self.zface_m),
            "DGRIDKM": str(self.dgrid_km),
            "XORIGKM": str(self.xorig_km),
            "YORIGKM": str(self.yorig_km),
            "IBCOMP": "1",
            "JBCOMP": "1",
            "IECOMP": str(self.nx),
            "JECOMP": str(self.ny),
        }


class CalpuffCaseFactory:
    """Build deterministic CALPUFF/CALPOST controls from cached case inputs."""

    def __init__(
        self,
        seed_path: Path,
        calpost_template: Path,
        source_rows: list[dict[str, str]],
        calmet_dat: str,
        start_utc: datetime = DEFAULT_START,
        domain: CalpuffDomain | Mapping[str, object] | None = None,
    ) -> None:
        self.seed_lines = seed_path.read_text(encoding="utf-8", errors="replace").splitlines()
        self.calpost_lines = calpost_template.read_text(
            encoding="ascii", errors="replace"
        ).splitlines()
        self.source_by_region: dict[int, list[dict[str, str]]] = {}
        for row in source_rows:
            self.source_by_region.setdefault(int(row["matrix_index"]), []).append(row)
        self.calmet_dat = calmet_dat
        if start_utc.tzinfo is None:
            raise ValueError("start_utc must be timezone-aware")
        self.start_utc = start_utc.astimezone(timezone.utc)
        self.domain = (
            domain if isinstance(domain, CalpuffDomain) else CalpuffDomain.from_mapping(domain)
        )

    def build_calpuff(
        self,
        output_dir: Path,
        source_region_index: int,
        hour_index: int,
        receptor_rows: list[dict[str, str]],
        emission_lb_per_hour: float = 1.0,
        release_height_m: float = 15.0,
        sigma_y_m: float = 250.0,
        sigma_z_m: float = 20.0,
        include_preceding_met_period: bool = False,
        initialization_mode: str = "one_hour_box",
        calmet_dat_override: str | None = None,
    ) -> Path:
        sources = self.source_by_region.get(source_region_index, [])
        if len(sources) != 16:
            raise ValueError(
                f"source region {source_region_index} has {len(sources)} rows; expected 16"
            )
        if not receptor_rows:
            raise ValueError("receptor_rows is empty")
        if emission_lb_per_hour <= 0 or sigma_y_m <= 0 or sigma_z_m <= 0:
            raise ValueError("emission and initial sigmas must be positive")

        target_start = self.start_utc + timedelta(hours=hour_index)
        target_end = target_start + timedelta(hours=1)
        # The paper matrix convention uses a one-hour source/state experiment.
        # Do not silently add a preceding emission period: that would contaminate
        # B0/A with puffs emitted before the requested interval. The legacy
        # runner can opt into the old behavior explicitly for audit comparisons.
        start = target_start - timedelta(hours=1) if include_preceding_met_period else target_start
        end = target_end
        output_dir.mkdir(parents=True, exist_ok=True)
        lines = list(self.seed_lines)
        assignments = {
            "METDAT": calmet_dat_override or self.calmet_dat,
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
            "NREC": str(len(receptor_rows)),
            "NRGRP": "0",
        }
        assignments.update(self.domain.control_assignments())
        for key, value in assignments.items():
            _replace_first_assignment(lines, key, value)

        source_blocks = ["Subgroup (16b)", "VOLUME SOURCE: CONSTANT DATA"]
        total_g_s = emission_lb_per_hour * LB_PER_HOUR_TO_G_S
        source_name_map: dict[str, str] = {}
        for number, source in enumerate(sources, start=1):
            control_name = f"V{number:04d}"
            source_name_map[control_name] = str(source["source_id"])
            fraction = float(source.get("release_fraction", 1.0 / len(sources)))
            q_no2 = total_g_s * fraction
            source_blocks.append(f"  {number} ! SRCNAM = {control_name} !")
            source_blocks.append(
                f"  {number} ! X = {float(source['x_m']) / 1000.0:.6f}, "
                f"{float(source['y_m']) / 1000.0:.6f}, {release_height_m:.3f}, "
                f"0.0, {sigma_y_m:.3f}, {sigma_z_m:.3f}, "
                f"0.0, 0.0, 0.0, {q_no2:.10g}, 0.0, 0.0, 0.0 ! !END!"
            )
        lines = _replace_between(lines, "Subgroup (16b)", "Subgroup (16c)", source_blocks)

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
            [
                "Subgroup (20b)",
                "RECEPTOR GROUP DATA (none in this case; NRGRP=0)",
                "",
            ],
        )
        receptor_blocks = ["Subgroup (20c)", "NON-GRIDDED (DISCRETE) RECEPTOR DATA"]
        for number, receptor in enumerate(receptor_rows, start=1):
            receptor_blocks.append(
                f" {number:5d} ! X = {float(receptor['x_m']) / 1000.0:.6f}, "
                f"{float(receptor['y_m']) / 1000.0:.6f}, 0.0, "
                f"{float(receptor.get('receptor_height_m', 1.5)):.3f} ! !END!"
            )
        lines = _replace_between(lines, "Subgroup (20c)", None, receptor_blocks)
        _set_species(lines)

        control_path = output_dir / "CALPUFF.INP"
        control_path.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
        metadata = {
            "source_region_index": source_region_index,
            "hour_index": hour_index,
            "start_utc": _iso(target_start),
            "end_utc_exclusive": _iso(target_end),
            "calpuff_model_start_utc": _iso(start),
            "calpuff_model_end_utc": _iso(end),
            "source_count": len(sources),
            "receptor_count": len(receptor_rows),
            "total_emission_lb_per_hour": emission_lb_per_hour,
            "total_emission_g_per_s": total_g_s,
            "release_height_m": release_height_m,
            "sigma_y_m": sigma_y_m,
            "sigma_z_m": sigma_z_m,
            "pollutant": "passive NO2-equivalent tracer",
            "chemistry": "off",
            "initialization_mode": initialization_mode,
            "include_preceding_met_period": include_preceding_met_period,
            "source_schedule": f"constant over [{_iso(start)}, {_iso(end)})",
            "source_name_map": source_name_map,
        }
        (output_dir / "CASE_METADATA.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        _write_rows(output_dir / "receptor_manifest.csv", receptor_rows)
        return control_path

    def build_calpost(self, output_dir: Path, hour_index: int) -> Path:
        start = self.start_utc + timedelta(hours=hour_index)
        end = start + timedelta(hours=1)
        lines = list(self.calpost_lines)
        assignments = {
            "MODDAT": "CALPUFF.CON",
            "PSTLST": "CALPOST.LST",
            "LCFILES": "F",
            "METRUN": "0",
            "ISYR": str(start.year),
            "ISMO": str(start.month),
            "ISDY": str(start.day),
            "ISHR": str(start.hour),
            "ISMIN": str(start.minute),
            "ISSEC": str(start.second),
            "IEYR": str(end.year),
            "IEMO": str(end.month),
            "IEDY": str(end.day),
            "IEHR": str(end.hour),
            "IEMIN": str(end.minute),
            "IESEC": str(end.second),
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
            # Select both Julian days in the 24-hour window so CALPOST emits
            # records after the UTC midnight boundary as well as before it.
            "IECHO": "173*0, 2*1, 191*0",
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
        output = output_dir / "CALPOST.INP"
        output.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")
        return output


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _set_species(lines: list[str]) -> None:
    group_start = next(i for i, line in enumerate(lines) if "Subgroup (3a)" in line)
    group_end = next(i for i in range(group_start, len(lines)) if "INPUT GROUP: 4" in lines[i])
    replacements = {
        "SO2": "!          SO2  =         1,               0,           0,                 0   !",
        "SO4": "!          SO4  =         1,               0,           0,                 0   !",
        "NO": "!           NO  =         1,               0,           0,                 0   !",
        "NO2": "!          NO2  =         1,               1,           0,                 0   !",
        "HNO3": "!         HNO3  =         1,               0,           0,                 0   !",
        "NO3": "!          NO3  =         1,               0,           0,                 0   !",
        "PM10": "!         PM10  =         1,               0,           0,                 0   !",
    }
    for i in range(group_start, group_end):
        for species, replacement in replacements.items():
            if re.search(rf"!\s*{re.escape(species)}\s*=", lines[i]):
                lines[i] = replacement
    output_start = next(
        i for i, line in enumerate(lines)
        if "SPECIES (or GROUP for combined species)" in line
    )
    output_end = next(i for i in range(output_start, len(lines)) if "!END!" in lines[i])
    for i in range(output_start, output_end):
        if re.search(r"!\s*NO2\s*=", lines[i]):
            lines[i] = "!          NO2 =     0,           1,           0,           1,           0,           1,           0   !"


def _replace_first_assignment(lines: list[str], key: str, value: str) -> None:
    pattern = re.compile(rf"(!\s*{re.escape(key)}\s*=)(.*?)(!)", re.IGNORECASE)
    for i, line in enumerate(lines):
        if pattern.search(line):
            lines[i] = pattern.sub(
                lambda match: f"{match.group(1)} {value} {match.group(3)}",
                line,
                count=1,
            )
            return
    raise ValueError(f"assignment not found in seed: {key}")


def _replace_between(
    lines: list[str],
    start_marker: str,
    end_marker: str | None,
    replacement: list[str],
) -> list[str]:
    start = next(i for i, line in enumerate(lines) if start_marker in line)
    if end_marker is None:
        return lines[:start] + replacement
    end = next(i for i in range(start + 1, len(lines)) if end_marker in lines[i])
    return lines[:start] + replacement + lines[end:]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
