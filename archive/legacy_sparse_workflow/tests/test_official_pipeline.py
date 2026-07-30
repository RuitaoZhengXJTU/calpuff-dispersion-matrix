from pathlib import Path

from run_official_pipeline import _mmif_input_paths, _preflight_stage


def test_mmif_input_parser_ignores_comments_and_preserves_order():
    text = """
    ; INPUT commented-out.wrf
    INPUT first.wrf
    INPUT \"folder with spaces/second.wrf\"
    ! INPUT ignored.wrf
    """
    assert _mmif_input_paths(text) == ["first.wrf", "folder with spaces/second.wrf"]


def test_draft_mmif_control_is_blocked(tmp_path: Path):
    control = tmp_path / "mmif.inp"
    control.write_text(
        "; draft control\n"
        "; Replace these placeholders\n"
        "start 2025 06 23 18\n"
        "stop 2025 06 24 18\n",
        encoding="ascii",
    )
    report = _preflight_stage(
        "mmif",
        tmp_path,
        control,
        tmp_path / "missing-mmif.exe",
    )
    assert report["preflight_ok"] is False
    assert any("no active INPUT" in error for error in report["errors"])
