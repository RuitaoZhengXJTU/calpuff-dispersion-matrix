from pathlib import Path

import pytest

from calpuff_matrix.runtime import ExternalCommandError, _assert_success, _parse_start_utc


def test_runtime_error_context_detects_fatal_log(tmp_path: Path) -> None:
    output = tmp_path / "output.dat"
    output.write_text("data", encoding="utf-8")
    log = tmp_path / "run.log"
    log.write_text("FATAL processing error", encoding="utf-8")

    with pytest.raises(ExternalCommandError, match="fatal/error marker"):
        _assert_success(output, log)


def test_start_time_requires_timezone() -> None:
    assert _parse_start_utc("2025-06-23T18:00:00Z").hour == 18
    with pytest.raises(ValueError, match="timezone"):
        _parse_start_utc("2025-06-23T18:00:00")
