"""Prospective constants and integrity helpers for Arm D R2.6-M-r3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL_REVISION = "arm_d_r2_6_m_r4"
MODEL_NAME = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
SPECIFICATION_PATH = "docs/specifications/arm_d/arm_d_r2_6_m_r4.md"

WIDTH = 256
SEED_RANK = 48
CANDIDATE_RANKS = (4, 8, 16, 32)
MAX_NEW_COMPONENTS = 4
MAX_TOTAL_RANK = 128
LAGS = (1, 2, 4, 8, 16, 32, 64)
RIDGE_GRID = (1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8, 1e10)
ESTIMATOR_RHO = 0.1
RIDGE_RHOS = (0.03, 0.1, 0.3)
RANDOM_DRAWS = 200
BOOTSTRAP_REPLICATES = 2000
STABILITY_NULL_DRAWS = 200
STABILITY_BOOTSTRAPS = 20
SEED = 42
ALPHA = 0.05
ORTHOGONALITY_ATOL = 1e-10
EIGEN_EPSILON = 1e-12
RANK_GAIN_FRACTION = 0.90
RIDGE_GAP_RATIO_MIN = 0.50

Q1_SHA256 = "df938a4e875e324ccaf452d6cb3f6036e8f05f67b16b07e99a28878c84bfa8c7"
Q2_SHA256 = "d2aa4443c1529360af57f0528c4afd0b58f31e891f1d71f01a84facd78d72f59"
R25_PCA64_GAIN = 0.2209871657266234
PRACTICAL_EFFECT = 0.025 * R25_PCA64_GAIN

SPLIT_SALT = "arm_d_r2_6_m_r4_confirm"
SPLIT_FRACTIONS = {"fitA": 0.35, "fitB": 0.35, "validation": 0.15, "test": 0.15}
SAMPLE_SIZES = (4000, 8000, 12000)
# R2.2 used 100--107, R2.4 used 300--307, donors 500--507, R2.5 used 700--707,
# and development used 900--901.  These sixteen untouched shards provide headroom for N=12k.
FRESH_SHARDS = tuple(range(800, 816))

W3_SHARING = (0.5, 0.8, 0.95)
W3_EFFECT_MULTIPLIERS = (1.0, 2.0)
W3_REPLICATES = 50
STANDARD_REPLICATES = 50
W5_FAMILIES = ("exponential", "power_law", "curved_log", "broken_slope")
W5_REPLICATES = 50
# Prospective known-answer calibration from the quarantined first qualification screen.
# The analytical signal-energy plant overpredicted the realized matched-ridge increment by
# 1/0.61 because the score averages normalized seven-lag gains after baseline fitting.
SYNTHETIC_EFFECT_CALIBRATION = 1.70


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array) -> str:
    import numpy as np

    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def write_json(path: str | Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def namespaced_seed(namespace: str, *parts: object) -> int:
    value = ":".join([namespace, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")
