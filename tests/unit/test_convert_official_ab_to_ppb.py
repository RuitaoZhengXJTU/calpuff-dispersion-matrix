from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, load_npz, save_npz

from calpuff_matrix.units import ppb_factor_array
from calpuff_matrix.conversion import main


def test_converter_applies_endpoint_diagonal_transforms(tmp_path: Path) -> None:
    source = tmp_path / "g_m3"
    (source / "B0").mkdir(parents=True)
    (source / "A").mkdir()
    contract = {
        "horizon_hours": 2,
        "a_hour_indices": [1],
        "region_count": 2,
        "generator_count": 1,
    }
    (source / "matrix_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    b0 = csc_matrix([[1.0], [2.0]])
    a1 = csc_matrix([[0.2, 0.3], [0.4, 0.5]])
    save_npz(source / "B0" / "B0_g_m3_per_lb.npz", b0)
    save_npz(source / "A" / "hour_01.npz", a1)
    (source / "B0" / "provenance.json").write_text("{}", encoding="utf-8")
    (source / "A" / "provenance.json").write_text("{}", encoding="utf-8")

    region_index = tmp_path / "region_index.csv"
    pd.DataFrame({"matrix_index": [0, 1], "region_id": ["a", "b"]}).to_csv(region_index, index=False)
    weather = pd.DataFrame({
        "hour_index": [0, 0, 1, 1, 2, 2],
        "region_id": ["a", "b", "a", "b", "a", "b"],
        "temperature_k": [290.0, 291.0, 292.0, 293.0, 294.0, 295.0],
        "pressure_pa": [100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0],
    })
    weather_path = tmp_path / "weather.csv"
    weather.to_csv(weather_path, index=False)
    output = tmp_path / "ppb"

    assert main([
        "--input-root", str(source),
        "--output-root", str(output),
        "--region-index", str(region_index),
        "--weather", str(weather_path),
    ]) == 0

    factors = ppb_factor_array(weather, np.array(["a", "b"]), 2, 46.0055)
    expected_b0 = np.diag(factors[1]) @ b0.toarray()
    expected_a1 = np.diag(factors[2]) @ a1.toarray() @ np.diag(1.0 / factors[1])
    assert np.allclose(load_npz(output / "B0" / "B0_ppb_per_lb.npz").toarray(), expected_b0)
    assert np.allclose(load_npz(output / "A" / "hour_01.npz").toarray(), expected_a1)
    converted_contract = json.loads((output / "matrix_contract.json").read_text(encoding="utf-8"))
    assert converted_contract["concentration_unit"] == "ppb"
    assert converted_contract["b0_file"] == "B0/B0_ppb_per_lb.npz"
