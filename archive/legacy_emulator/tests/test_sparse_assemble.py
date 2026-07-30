from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from transfer_matrix.assemble_sparse import assemble_sparse_response_matrices


def test_sparse_assembler_accepts_namespaced_region_ids(tmp_path: Path) -> None:
    partition = tmp_path / "partition"
    partition.mkdir()
    pd.DataFrame({"region_id": ["area_pop_000000", "area_pop_000001"]}).to_csv(
        partition / "region_area_population_summary.csv", index=False
    )
    case_root = tmp_path / "cases"
    for source_id, values in {
        "area_pop_000000": [0.7, 0.2],
        "area_pop_000001": [0.1, 0.6],
    }.items():
        case_dir = case_root / "hour_00" / f"source_{source_id}"
        case_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "region_id": ["area_pop_000000", "area_pop_000001"],
                "concentration": values,
            }
        ).to_csv(case_dir / "receptors.csv", index=False)

    output = assemble_sparse_response_matrices(
        case_root=case_root,
        partition_dir=partition,
        output_dir=tmp_path / "out",
        hours=1,
        start_utc="2025-06-23T18:00:00Z",
    )
    matrix = load_npz(output / "hour_00.npz").toarray()
    np.testing.assert_allclose(matrix, np.array([[0.7, 0.1], [0.2, 0.6]]))
