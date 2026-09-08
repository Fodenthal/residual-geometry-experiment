from __future__ import annotations

import numpy as np


def orthonormal_basis(directions: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(directions)
    return q


def project_out(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return vectors - (vectors @ basis) @ basis.T

