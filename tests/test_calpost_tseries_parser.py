from pathlib import Path

import pandas as pd

from parse_calpost_tseries import parse_calpost_tseries


def test_parse_calpost_tseries_preserves_manifest_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "receptor_id": ["r0", "r1", "r2"],
            "region_id": ["a", "a", "b"],
            "matrix_index": [0, 0, 1],
        }
    ).to_csv(manifest, index=False)
    tseries = tmp_path / "TSERIES.DAT"
    tseries.write_text(
        "header\n"
        " YYYY JDY HHMM (START time)\n"
        " 2025 174 1800    1.0E-06  0.0E+00\n"
        " 2.5E-06\n",
        encoding="ascii",
    )
    output = parse_calpost_tseries(
        tseries,
        manifest,
        tmp_path / "receptors.csv",
        "2025-06-23T18:00:00Z",
    )
    result = pd.read_csv(output)
    assert result["receptor_id"].tolist() == ["r0", "r1", "r2"]
    assert result["concentration"].tolist() == [1e-6, 0.0, 2.5e-6]
