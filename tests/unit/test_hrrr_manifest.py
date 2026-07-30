import hashlib
import json
from pathlib import Path

from calpuff_matrix.hrrr import verify_manifest


def test_verify_manifest_checks_size_and_sha256_without_network(tmp_path: Path) -> None:
    selected = tmp_path / "data" / "sample.grib2"
    selected.parent.mkdir()
    selected.write_bytes(b"selected hrrr bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{
        "subset_path": "data/sample.grib2",
        "subset_size_bytes": selected.stat().st_size,
        "subset_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }]}), encoding="utf-8")

    report = verify_manifest(manifest, tmp_path)

    assert report["ok"] is True
    selected.write_bytes(b"tampered")
    assert verify_manifest(manifest, tmp_path)["ok"] is False
