from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from transfer_matrix.calpost_adapter import adapt_calpost_csv


def test_calpost_adapter_requires_explicit_complete_receptor_mapping(tmp_path: Path) -> None:
    manifest = tmp_path / "batch.csv"
    pd.DataFrame(
        {
            "receptor_id": ["r0_q0", "r0_q1"],
            "region_id": ["r0", "r0"],
            "matrix_index": [0, 0],
        }
    ).to_csv(manifest, index=False)
    raw = tmp_path / "calpost.csv"
    pd.DataFrame({"receptor_id": ["r0_q0", "r0_q1"], "concentration": [1.0, 2.0]}).to_csv(raw, index=False)

    output = adapt_calpost_csv(raw, manifest, tmp_path / "receptors.csv", value_unit="ppb")
    result = pd.read_csv(output)
    assert result["receptor_id"].tolist() == ["r0_q0", "r0_q1"]
    assert result["concentration"].tolist() == [1.0, 2.0]
    assert result["value_unit"].unique().tolist() == ["ppb"]

    incomplete = tmp_path / "incomplete.csv"
    pd.DataFrame({"receptor_id": ["r0_q0"], "concentration": [1.0]}).to_csv(incomplete, index=False)
    with pytest.raises(ValueError, match="mapping incomplete"):
        adapt_calpost_csv(incomplete, manifest, tmp_path / "bad.csv")
