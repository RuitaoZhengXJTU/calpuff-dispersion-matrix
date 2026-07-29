import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, save_npz

from validate_official_ab import main


def _write_package(root: Path, a_hours: range) -> None:
    (root / "B0").mkdir(parents=True)
    (root / "A").mkdir(parents=True)
    contract = {
        "state_equation": "c1 = B0 @ emitted_mass_lb; c[h+1] = A[h] @ c[h] for h=1..23",
        "region_count": 3,
        "generator_count": 2,
    }
    (root / "matrix_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    save_npz(root / "B0" / "B0_g_m3_per_lb.npz", csc_matrix(np.eye(3, 2)))
    (root / "B0" / "provenance.json").write_text(
        json.dumps({"uses_calpuff": True, "not_emulator": True}), encoding="utf-8"
    )
    for hour in a_hours:
        save_npz(root / "A" / f"hour_{hour:02d}.npz", csc_matrix(np.eye(3)))
    (root / "A" / "provenance.json").write_text(
        json.dumps({"uses_calpuff": True, "not_emulator": True}), encoding="utf-8"
    )


def test_official_ab_validator_accepts_complete_package(tmp_path: Path):
    _write_package(tmp_path, range(1, 24))
    assert main(["--output-root", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "validation_report.json").read_text(encoding="utf-8"))
    assert report["ok"] is True


def test_official_ab_validator_allows_smoke_package(tmp_path: Path):
    _write_package(tmp_path, range(1, 2))
    assert main(["--output-root", str(tmp_path), "--allow-partial"]) == 0


def test_official_ab_validator_uses_horizon_from_contract(tmp_path: Path):
    _write_package(tmp_path, range(1, 3))
    contract_path = tmp_path / "matrix_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update({
        "horizon_hours": 3,
        "a_hour_indices": [1, 2],
        "state_equation": "c1 = B0 @ emitted_mass_lb; c[h+1] = A[h] @ c[h] for h=1..2",
    })
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    assert main(["--output-root", str(tmp_path)]) == 0
