from pathlib import Path

import pytest

from calpuff_matrix import cli


def test_cli_lists_stable_commands(capsys) -> None:
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    for command in ("fetch-hrrr", "prepare", "build-weather", "build-calmet", "run", "convert-units", "validate", "verify-hrrr"):
        assert command in output


def test_each_core_subcommand_exposes_help() -> None:
    for command in cli.COMMANDS:
        with pytest.raises(SystemExit) as result:
            cli.main([command, "--help"])
        assert result.value.code == 0


def test_compatibility_wrappers_are_explicitly_deprecated() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("fetch_hrrr_selected_messages.py", "prepare_official_sparse_calpuff.py", "run_official_ab_matrices.py", "validate_official_ab.py"):
        assert "deprecated" in (root / name).read_text(encoding="utf-8").lower()
