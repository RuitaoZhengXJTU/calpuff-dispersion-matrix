"""Shared runtime helpers for formal external CALPUFF workflow commands."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ERROR_RE = re.compile(r"(?i)error in subr|halted in|fatal|endfile")


class ExternalCommandError(RuntimeError):
    """A required executable failed, including command context for diagnosis."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_start_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("start-utc must include a timezone, for example 2025-06-23T18:00:00Z")
    return parsed.astimezone(timezone.utc)


def _resolve_executable(
    explicit: str | Path | None,
    environment_variable: str,
    default_name: str,
    fallback: str | Path | None = None,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a named executable and fail with the exact configuration needed."""
    candidates = [explicit, os.environ.get(environment_variable), fallback]
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            if path.exists() or not must_exist:
                return path.resolve() if path.exists() else path
            raise FileNotFoundError(f"{environment_variable} points to a missing executable: {path}")
    discovered = shutil.which(default_name)
    if discovered:
        return Path(discovered).resolve()
    if must_exist:
        raise FileNotFoundError(
            f"Cannot find {default_name}. Set {environment_variable} to its executable path "
            "or pass the corresponding CLI option."
        )
    return Path(default_name)


def _assert_success(output: Path, log_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if ERROR_RE.search(text):
        raise ExternalCommandError(f"binary log contains fatal/error marker: {log_path}")
    if not output.exists() or output.stat().st_size == 0:
        raise ExternalCommandError(f"expected non-empty output missing: {output}")


def run_control_file(
    executable: Path,
    control: Path,
    cwd: Path,
    log_path: Path,
    timeout_sec: int,
) -> int:
    """Run one CALPUFF-family executable and leave its complete log on disk."""
    command = [str(executable), control.name]
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            completed = subprocess.run(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise ExternalCommandError(
            f"command timed out after {timeout_sec}s: {command}; log={log_path}"
        ) from exc
    if completed.returncode != 0:
        raise ExternalCommandError(
            f"command returned {completed.returncode}: {command}; log={log_path}"
        )
    return int(completed.returncode)
