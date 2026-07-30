from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def adapt_calpost_csv(
    input_path: Path,
    receptor_manifest: Path,
    output_path: Path,
    *,
    value_column: str = "concentration",
    value_unit: str | None = None,
) -> Path:
    """Normalize a verified CALPOST export to the assembler receptor contract.

    The input must already contain an explicit ``receptor_id``. Coordinates or
    row order are deliberately not used to infer receptor identity.
    """
    raw = pd.read_csv(input_path)
    manifest = pd.read_csv(receptor_manifest)
    required_manifest = {"receptor_id", "region_id", "matrix_index"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(f"receptor manifest missing columns: {sorted(missing_manifest)}")
    if "receptor_id" not in raw.columns or value_column not in raw.columns:
        raise ValueError(
            f"CALPOST export must contain explicit receptor_id and {value_column} columns"
        )
    if not manifest["receptor_id"].is_unique:
        raise ValueError("receptor manifest receptor_id values must be unique")
    if raw["receptor_id"].duplicated().any():
        duplicate_ids = raw.loc[raw["receptor_id"].duplicated(), "receptor_id"].astype(str).tolist()
        raise ValueError(f"CALPOST export contains duplicate receptor_id values: {duplicate_ids[:5]}")

    raw = raw[["receptor_id", value_column]].copy()
    raw["receptor_id"] = raw["receptor_id"].astype(str)
    raw[value_column] = pd.to_numeric(raw[value_column], errors="coerce")
    if raw[value_column].isna().any() or not np.isfinite(raw[value_column].to_numpy(float)).all():
        raise ValueError("CALPOST values contain non-finite or non-numeric entries")
    if (raw[value_column] < 0).any():
        raise ValueError("CALPOST values contain negative concentrations")

    expected = manifest[["receptor_id", "region_id", "matrix_index"]].copy()
    merged = expected.merge(raw, on="receptor_id", how="left", validate="one_to_one")
    missing = merged.loc[merged[value_column].isna(), "receptor_id"].tolist()
    unexpected = sorted(set(raw["receptor_id"]) - set(expected["receptor_id"]))
    if missing or unexpected:
        raise ValueError(
            f"receptor mapping incomplete: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    merged = merged.rename(columns={value_column: "concentration"})
    merged["value_unit"] = value_unit or "unspecified; verify CALPOST output units"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path
