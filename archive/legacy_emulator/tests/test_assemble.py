from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transfer_matrix.assemble import assemble_matrices
from transfer_matrix.config import load_case
from transfer_matrix.validate import validate_outputs


def test_fake_receptor_outputs_assemble_expected_direction(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_dir = project / "config"
    config_dir.mkdir()
    case_file = config_dir / "case.yaml"
    case_file.write_text(
        """
case_id: fake
time:
  start_utc: "2025-06-24T18:00:00Z"
  hours: 1
grid:
  target_regions: 2
calpuff:
  case_root: runs/cases
outputs:
  receptors_csv: outputs/receptors.csv
  matrix_npz: outputs/matrix.npz
  matrices_dir: outputs/matrices
  diagnostics_dir: outputs/diagnostics
  provenance_json: outputs/provenance.json
""",
        encoding="utf-8",
    )
    config = load_case(case_file)
    (project / "outputs").mkdir()
    pd.DataFrame(
        {
            "receptor_id": ["r000_q00", "r001_q00"],
            "region_id": ["area_pop_000000", "area_pop_000001"],
        }
    ).to_csv(project / "outputs" / "receptors.csv", index=False)

    case_root = project / "runs" / "cases" / "hour_00"
    (case_root / "source_area_pop_000000").mkdir(parents=True)
    (case_root / "source_area_pop_000001").mkdir(parents=True)
    pd.DataFrame(
        {
            "receptor_id": ["r000_q00", "r001_q00"],
            "region_id": ["area_pop_000000", "area_pop_000001"],
            "response_fraction": [0.7, 0.2],
        }
    ).to_csv(case_root / "source_area_pop_000000" / "receptors.csv", index=False)
    pd.DataFrame(
        {
            "receptor_id": ["r000_q00", "r001_q00"],
            "region_id": ["area_pop_000000", "area_pop_000001"],
            "response_fraction": [0.1, 0.6],
        }
    ).to_csv(case_root / "source_area_pop_000001" / "receptors.csv", index=False)

    out = assemble_matrices(config)
    payload = np.load(out, allow_pickle=True)
    matrix = payload["T"]
    assert matrix.shape == (1, 2, 2)
    np.testing.assert_allclose(matrix[0], np.array([[0.7, 0.1], [0.2, 0.6]]))

    vector = np.array([1.0, 0.0])
    np.testing.assert_allclose(matrix[0] @ vector, np.array([0.7, 0.2]))
    assert validate_outputs(config)["ok"] is True
