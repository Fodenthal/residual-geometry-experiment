from __future__ import annotations

import numpy as np


def max_abs_alignment(direction: np.ndarray, basis: np.ndarray) -> float:
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    return float(np.max(np.abs(basis @ direction)))

