import json
from pathlib import Path

import numpy as np

from calpuff_matrix.matrices import _load_completed_response


def test_completed_response_is_reused_only_with_completed_status(tmp_path: Path) -> None:
    np.savez_compressed(tmp_path / "region_response.npz", indices=[2], values=[1.25])
    (tmp_path / "run_status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert _load_completed_response(tmp_path) is None

    (tmp_path / "run_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert _load_completed_response(tmp_path) == {2: 1.25}
