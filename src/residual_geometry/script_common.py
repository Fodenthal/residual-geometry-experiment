from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from residual_geometry.config.loader import config_hash
from residual_geometry.config.loader import get_run_dir, load_persistent_state_config, save_config_to_run_dir
from residual_geometry.config.schema import PersistentStateConfig, ResidualGeometryConfig
from residual_geometry.data.artifact_store import PersistentStateArtifactStore
from residual_geometry.residuals.provider import ResidualStreamProvider

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config_and_store(config_path: str) -> tuple[PersistentStateConfig, PersistentStateArtifactStore]:
    config = load_persistent_state_config(config_path)
    run_dir = get_run_dir(config)
    save_config_to_run_dir(config, run_dir)
    return config, PersistentStateArtifactStore(run_dir)


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def artifact_metadata(config: PersistentStateConfig, stage: str) -> dict[str, object]:
    return {
        "stage": stage,
        "config_hash": config_hash(config),
        "git_commit": git_commit_hash(),
        "mode": config.mode,
        "run_name": config.run_name,
    }


def require_residual_geometry_config(config: PersistentStateConfig) -> ResidualGeometryConfig:
    if config.residual_geometry is None:
        raise ValueError("config.residual_geometry is required for residual-geometry scripts")
    if config.context_processing.add_special_tokens:
        raise ValueError("residual geometry requires context_processing.add_special_tokens=false")
    if config.context_processing.max_tokens != config.residual_geometry.max_tokens:
        raise ValueError(
            "context_processing.max_tokens must match residual_geometry.max_tokens "
            f"({config.context_processing.max_tokens} != {config.residual_geometry.max_tokens})"
        )
    if config.context_processing.min_tokens != config.residual_geometry.max_tokens:
        raise ValueError(
            "context_processing.min_tokens must match residual_geometry.max_tokens "
            f"({config.context_processing.min_tokens} != {config.residual_geometry.max_tokens})"
        )
    return config.residual_geometry


def split_contexts(store: PersistentStateArtifactStore, split_name: str) -> pd.DataFrame:
    pool = pd.read_parquet(store.context_pool_path)
    splits = pd.read_parquet(store.context_split_path)
    df = pool.merge(splits, on="context_id", how="inner")
    return df[df["split"] == split_name].reset_index(drop=True)


def token_matrix(contexts: pd.DataFrame) -> np.ndarray:
    return np.asarray([list(tokens) for tokens in contexts["token_ids"]], dtype=np.int64)


def residual_batches(
    provider: ResidualStreamProvider,
    tokens: np.ndarray,
    batch_size: int,
):
    for start in range(0, len(tokens), batch_size):
        batch = torch.as_tensor(tokens[start : start + batch_size], dtype=torch.long, device=provider.device)
        residuals = provider.compute_sequence_residuals(batch).float().detach().cpu().numpy().astype(np.float32)
        yield start, residuals
