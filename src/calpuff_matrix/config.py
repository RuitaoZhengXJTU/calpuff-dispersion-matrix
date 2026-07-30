"""Shared helpers for portable official CALPUFF matrix case manifests."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml


def load_case_config(path: Path | None) -> dict[str, object]:
    """Read a case YAML and resolve its optional repository-relative ``extends`` chain.

    Child mappings override parent mappings recursively.  The returned mapping
    keeps no implicit dependence on the YAML file location: paths are resolved
    through :func:`project_path` against the repository root.
    """
    if path is None:
        return {}
    return _load_case_config(path.resolve(), seen=set())


def _load_case_config(path: Path, seen: set[Path]) -> dict[str, object]:
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ValueError(f"cyclic case-config extends chain: {chain}")
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"case config must contain a YAML mapping: {path}")
    parent_value = payload.pop("extends", None)
    if parent_value is None:
        return payload
    parent = Path(str(parent_value))
    if not parent.is_absolute():
        parent = path.parent / parent
    parent_payload = _load_case_config(parent.resolve(), seen={*seen, path})
    return _deep_merge(parent_payload, payload)


def _deep_merge(parent: Mapping[str, object], child: Mapping[str, object]) -> dict[str, object]:
    """Merge nested YAML mappings without mutating either configuration."""
    merged: dict[str, object] = deepcopy(dict(parent))
    for key, child_value in child.items():
        parent_value = merged.get(key)
        if isinstance(parent_value, Mapping) and isinstance(child_value, Mapping):
            merged[key] = _deep_merge(parent_value, child_value)
        else:
            merged[key] = deepcopy(child_value)
    return merged


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
