from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def iter_activation_chunks(split_dir: str) -> Iterator[dict[str, np.ndarray]]:
    for path in sorted(Path(split_dir).glob("chunk_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            yield {key: data[key] for key in data.files}


def save_activation_chunk(
    path: str,
    context_ids: np.ndarray,
    tokens: np.ndarray,
    feature_indices: np.ndarray,
    activations: np.ndarray,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        context_ids=context_ids,
        tokens=tokens,
        feature_indices=feature_indices,
        activations=activations.astype(np.float16, copy=False),
    )

