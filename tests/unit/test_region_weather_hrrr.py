from __future__ import annotations

import json
from pathlib import Path

from calpuff_matrix.weather import _representative_points


def test_representative_points_are_stable_and_require_unique_ids(tmp_path: Path) -> None:
    geojson = tmp_path / "subregions.geojson"
    geojson.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"region_id": "b"},
                "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]]},
            },
            {
                "type": "Feature",
                "properties": {"region_id": "a"},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            },
        ],
    }), encoding="utf-8")

    points = _representative_points(geojson)

    assert [region_id for region_id, _, _ in points] == ["a", "b"]
    assert all(0.0 <= lon <= 2.0 and 0.0 <= lat <= 2.0 for _, lon, lat in points)
