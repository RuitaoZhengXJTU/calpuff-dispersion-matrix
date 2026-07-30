from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CaseConfig:
    """Thin wrapper around the YAML case file."""

    path: Path
    root: Path
    data: dict[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.data["case_id"])

    @property
    def hours(self) -> int:
        return int(self.data["time"]["hours"])

    @property
    def target_regions(self) -> int:
        return int(self.data["grid"]["target_regions"])

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    def output_path(self, key: str) -> Path:
        return self.resolve(self.data["outputs"][key])

    def case_root(self) -> Path:
        return self.resolve(self.data["calpuff"]["case_root"])


def load_case(path: str | Path) -> CaseConfig:
    case_path = Path(path).resolve()
    with case_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    root = case_path.parent.parent.resolve()
    return CaseConfig(path=case_path, root=root, data=data)

