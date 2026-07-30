from pathlib import Path

from calpuff_matrix.config import load_case_config


def test_case_config_extends_and_recursively_merges(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text("paths:\n  raw: data/raw\n  nested:\n    keep: true\ntime:\n  hours: 24\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\npaths:\n  nested:\n    replace: yes\ntime:\n  start_utc: '2025-06-23T18:00:00Z'\n", encoding="utf-8")

    loaded = load_case_config(child)

    assert loaded["paths"] == {"raw": "data/raw", "nested": {"keep": True, "replace": True}}
    assert loaded["time"] == {"hours": 24, "start_utc": "2025-06-23T18:00:00Z"}
