from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from prepare_official_sparse_calpuff import _write_case_inputs, _write_receptor_batches


def test_receptor_batches_keep_regions_intact(tmp_path: Path) -> None:
    rows = []
    for matrix_index in range(10):
        for point_index in range(3):
            rows.append(
                {
                    "receptor_id": f"r{matrix_index:02d}_q{point_index:02d}",
                    "region_id": f"r{matrix_index:02d}",
                    "matrix_index": matrix_index,
                }
            )
    receptors = pd.DataFrame(rows)
    _write_receptor_batches(tmp_path, receptors, max_discrete_receptors=10)

    manifest = pd.read_csv(tmp_path / "receptor_batch_manifest.csv")
    combined = pd.concat(
        [pd.read_csv(tmp_path / filename) for filename in manifest["filename"]],
        ignore_index=True,
    )
    assert len(manifest) == 4
    assert len(combined) == len(receptors)
    assert combined["receptor_id"].is_unique
    assert set(combined["receptor_id"]) == set(receptors["receptor_id"])
    assert (combined.groupby("region_id").size() == 3).all()
    assert (manifest["receptor_count"] <= 10).all()


def test_case_input_writer_accepts_portable_geojson_and_computes_area(tmp_path: Path) -> None:
    partition = tmp_path / "partition"
    partition.mkdir()
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"region_id": "west"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-77.1, 38.8], [-77.0, 38.8], [-77.0, 38.9], [-77.1, 38.9], [-77.1, 38.8]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"region_id": "east"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-77.0, 38.8], [-76.9, 38.8], [-76.9, 38.9], [-77.0, 38.9], [-77.0, 38.8]]],
                },
            },
        ],
    }
    (partition / "subregions.geojson").write_text(json.dumps(payload), encoding="utf-8")

    regions = _write_case_inputs(
        tmp_path / "case" / "inputs",
        partition,
        max_discrete_receptors=10,
        projected_crs="EPSG:3857",
        receptor_points_per_region=9,
    )

    assert list(regions["region_id"]) == ["east", "west"]
    assert (regions["area_m2"] > 0).all()
    assert len(pd.read_csv(tmp_path / "case" / "inputs" / "sources_16_per_region.csv")) == 32
    assert len(pd.read_csv(tmp_path / "case" / "inputs" / "receptors_9_per_region.csv")) == 18
