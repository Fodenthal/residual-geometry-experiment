from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import numpy as np

from src.utils.io import ensure_dir


@dataclass(frozen=True)
class ResidualProbeSet:
    directions: np.ndarray
    probe_ids: np.ndarray
    probe_family: np.ndarray

    def __post_init__(self) -> None:
        if self.directions.ndim != 2:
            raise ValueError("directions must have shape (probes, d_model)")
        if len(self.probe_ids) != self.directions.shape[0]:
            raise ValueError("probe_ids length must match directions")
        if len(self.probe_family) != self.directions.shape[0]:
            raise ValueError("probe_family length must match directions")

    @property
    def n_probes(self) -> int:
        return int(self.directions.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.directions.shape[1])


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms < eps):
        raise ValueError("cannot normalize a zero-norm direction")
    return arr / norms


def random_residual_probe_set(d_model: int, count: int, seed: int, family: str = "random") -> ResidualProbeSet:
    rng = np.random.default_rng(seed)
    directions = normalize_rows(rng.normal(size=(count, d_model)).astype(np.float32))
    probe_ids = np.array([f"random_residual_{idx:05d}" for idx in range(count)], dtype=object)
    probe_family = np.full(count, family, dtype=object)
    return ResidualProbeSet(directions=directions, probe_ids=probe_ids, probe_family=probe_family)


def probe_set_from_directions(directions: np.ndarray, family: str, id_prefix: str) -> ResidualProbeSet:
    normalized = normalize_rows(directions)
    probe_ids = np.array([f"{id_prefix}_{idx:05d}" for idx in range(normalized.shape[0])], dtype=object)
    probe_family = np.full(normalized.shape[0], family, dtype=object)
    return ResidualProbeSet(directions=normalized, probe_ids=probe_ids, probe_family=probe_family)


def concatenate_probe_sets(probe_sets: Iterable[ResidualProbeSet]) -> ResidualProbeSet:
    sets = list(probe_sets)
    if not sets:
        raise ValueError("need at least one probe set")
    d_model = sets[0].d_model
    if any(probes.d_model != d_model for probes in sets):
        raise ValueError("all probe sets must have the same d_model")
    return ResidualProbeSet(
        directions=np.concatenate([probes.directions for probes in sets], axis=0).astype(np.float32),
        probe_ids=np.concatenate([probes.probe_ids for probes in sets], axis=0),
        probe_family=np.concatenate([probes.probe_family for probes in sets], axis=0),
    )


def save_probe_set(probes: ResidualProbeSet, path: str, **metadata: object) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.savez_compressed(
        path,
        directions=probes.directions.astype(np.float32),
        probe_ids=probes.probe_ids.astype(str),
        probe_family=probes.probe_family.astype(str),
        **metadata,
    )


def load_probe_set(path: str) -> ResidualProbeSet:
    with np.load(path, allow_pickle=False) as data:
        return ResidualProbeSet(
            directions=data["directions"].astype(np.float32),
            probe_ids=data["probe_ids"].astype(str),
            probe_family=data["probe_family"].astype(str),
        )


def load_many_probe_sets(paths: Iterable[str]) -> ResidualProbeSet:
    return concatenate_probe_sets(load_probe_set(path) for path in paths)
