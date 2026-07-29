from __future__ import annotations

from pathlib import Path

from .config import CaseConfig


def ensure_project_dirs(config: CaseConfig) -> list[Path]:
    """Create runtime directories used by the harness."""

    dirs = [
        config.root / "data" / "raw",
        config.root / "data" / "processed",
        config.root / "runs",
        config.case_root(),
        config.output_path("matrices_dir"),
        config.output_path("diagnostics_dir"),
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
    return dirs

