from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import yaml

from residual_geometry.config.schema import PersistentStateConfig, mode_defaults
from residual_geometry.utils.io import ensure_dir, save_json


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_persistent_state_config(path: str) -> PersistentStateConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} did not contain a mapping.")
    mode = raw.get("mode")
    if mode not in {"smoke", "pilot", "full"}:
        raise ValueError("persistent_state config mode must be smoke, pilot, or full")
    merged = _deep_merge(mode_defaults(mode), raw)
    return PersistentStateConfig.model_validate(merged)


def config_hash(config: PersistentStateConfig) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_run_dir(config: PersistentStateConfig) -> str:
    return f"{config.output_base}/{config.mode}/{config.run_name}"


def save_config_to_run_dir(config: PersistentStateConfig, run_dir: str) -> None:
    ensure_dir(f"{run_dir}/configs")
    save_json(config.model_dump(), f"{run_dir}/configs/config.json")
    save_json({"hash": config_hash(config)}, f"{run_dir}/configs/config_hash.json")

