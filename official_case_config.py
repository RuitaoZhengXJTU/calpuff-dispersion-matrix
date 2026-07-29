"""Shared helpers for portable official CALPUFF matrix case manifests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


def load_case_config(path: Path | None) -> dict[str, object]:
    """Read an optional YAML case manifest and require mapping-shaped sections."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"case config must contain a YAML mapping: {path}")
    return payload


def mapping_value(payload: Mapping[str, object], *keys: str) -> Mapping[str, object]:
    """Return a nested mapping or an empty mapping when the section is absent."""
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key, {})
    if not isinstance(current, Mapping):
        raise ValueError(f"{'.'.join(keys)} must be a mapping")
    return current


def scalar_value(payload: Mapping[str, object], *keys: str, default: object = None) -> object:
    """Return a nested scalar, treating absent keys as the provided default."""
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def project_path(project_root: Path, value: object | None) -> Path | None:
    """Resolve a manifest path against the repository root, never its YAML folder."""
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path
