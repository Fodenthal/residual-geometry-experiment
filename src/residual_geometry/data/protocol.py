from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from residual_geometry.utils.io import load_json, save_json


def get_git_commit(repo_root: str | None = None) -> str | None:
    cwd = repo_root or str(Path.cwd())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def initialize_protocol_log(path: str, payload: dict[str, Any]) -> None:
    base = {"git_commit": get_git_commit(), **payload}
    save_json(base, path)


def update_protocol_log(path: str, payload: dict[str, Any]) -> None:
    existing = load_json(path) if Path(path).exists() else {}
    existing.update(payload)
    save_json(existing, path)

