from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from residual_geometry.script_common import (  # noqa: E402
    artifact_metadata,
    git_commit_hash,
    load_config_and_store,
    require_residual_geometry_config,
    residual_batches,
    split_contexts,
    token_matrix,
)

__all__ = [
    "artifact_metadata",
    "git_commit_hash",
    "load_config_and_store",
    "require_residual_geometry_config",
    "residual_batches",
    "split_contexts",
    "token_matrix",
]
