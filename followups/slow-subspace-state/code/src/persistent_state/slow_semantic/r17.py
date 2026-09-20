"""Pure helpers for the faithful signed-TICA R1.7 experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.persistent_state.residuals.probes import (
    ResidualProbeSet,
    concatenate_probe_sets,
    random_residual_probe_set,
)
from src.persistent_state.subspace.residual_geometry import deduplicate_ranked_probes, orthonormal_basis


LAGS = (8, 16, 32, 64, 128)
RANKS = (8, 13, 21, 31)
RANDOM_IN_SPAN_COUNT = 512
RANDOM_IN_SPAN_SEED = 31
SPLIT_SEED = 170031
EXPECTED_CANONICAL_SHA256 = "966736e0346dd354559f1354efa86f71c4e6c01cbb929ff70b18f7a9420cce6c"
EXPECTED_CONFIG_SHA256 = "9fa0bed92af4b2da3ca0e91c3f3056594a8c95d21dccf5bfe7e5bca08b97e3dd"


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_half_indices(document_count: int = 4000, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    if document_count != 4000:
        raise ValueError("R1.7 requires the frozen 4,000-document training corpus")
    order = np.random.default_rng(seed).permutation(document_count)
    return np.sort(order[:2000]), np.sort(order[2000:])


def sampled_original_positions(
    document_indices: np.ndarray,
    *,
    positions_per_document: int = 1024,
    max_positions: int = 1_000_000,
    seed: int = 4,
) -> np.ndarray:
    """Return original-train-flat indices in canonical sample output order."""
    docs = np.asarray(document_indices, dtype=np.int64)
    local_total = len(docs) * positions_per_document
    selected = np.sort(
        np.random.default_rng(seed).choice(local_total, size=min(max_positions, local_total), replace=False)
    )
    local_docs, positions = np.divmod(selected, positions_per_document)
    return docs[local_docs] * positions_per_document + positions


def raw_statistics_to_centered(payload: dict[str, np.ndarray], lags: tuple[int, ...] = LAGS):
    count = int(np.asarray(payload["token_count"]).item())
    total = np.asarray(payload["sum"], dtype=np.float64)
    mean = total / count
    sigma0 = (
        np.asarray(payload["second_moment_sum"], dtype=np.float64)
        - np.outer(total, mean)
        - np.outer(mean, total)
        + count * np.outer(mean, mean)
    ) / count
    sigma0 = 0.5 * (sigma0 + sigma0.T)
    sigma_lag = np.zeros_like(sigma0)
    pair_count_total = 0
    for lag in lags:
        n = int(np.asarray(payload[f"lag_{lag}_count"]).item())
        raw = np.asarray(payload[f"lag_{lag}_cross_sum"], dtype=np.float64)
        left = np.asarray(payload[f"lag_{lag}_left_sum"], dtype=np.float64)
        right = np.asarray(payload[f"lag_{lag}_right_sum"], dtype=np.float64)
        centered = raw - np.outer(left, mean) - np.outer(mean, right) + n * np.outer(mean, mean)
        sigma_lag += 0.5 * (centered + centered.T) / n
        pair_count_total += n
    sigma_lag /= len(lags)
    return mean.astype(np.float32), sigma0, sigma_lag, pair_count_total, count


def combine_raw_statistics(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(a) != set(b):
        raise ValueError("A/B sufficient-statistic keys differ")
    return {key: np.asarray(a[key], dtype=np.float64) + np.asarray(b[key], dtype=np.float64) for key in a}


def candidate_probe_set(time_lagged: ResidualProbeSet, pca: ResidualProbeSet) -> ResidualProbeSet:
    random = random_residual_probe_set(d_model=2304, count=512, seed=4)
    return concatenate_probe_sets([time_lagged, pca, random])


def select_nested_bases(
    probes: ResidualProbeSet,
    validation_timescales: pd.DataFrame,
    ranks: tuple[int, ...] = RANKS,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    table = validation_timescales.copy()
    eligible = table[table["tau_valid_within"].astype(bool)].copy()
    if "gk_positive_validation_persistence" in eligible:
        eligible = eligible[
            (eligible["probe_family"] != "time_lagged")
            | eligible["gk_positive_validation_persistence"].astype(bool)
        ]
    ranked = deduplicate_ranked_probes(probes, eligible, threshold=0.95)
    if len(ranked) < max(ranks):
        raise ValueError(f"only {len(ranked)} candidates survived; need {max(ranks)}")
    id_to_index = {str(pid): i for i, pid in enumerate(probes.probe_ids.astype(str))}
    directions = np.stack([probes.directions[id_to_index[str(pid)]] for pid in ranked.probe_id])
    bases = {rank: orthonormal_basis(directions, rank) for rank in ranks}
    return bases, ranked


def random_in_span(basis: np.ndarray, count: int = RANDOM_IN_SPAN_COUNT, seed: int = RANDOM_IN_SPAN_SEED):
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(count, basis.shape[1]))
    coefficients /= np.linalg.norm(coefficients, axis=1, keepdims=True)
    return (coefficients @ basis.T).astype(np.float32), coefficients.astype(np.float32)


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> tuple[float, np.ndarray]:
    singular = np.linalg.svd(left.astype(np.float64).T @ right.astype(np.float64), compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return float(np.mean(singular**2)), singular


def geometry_classification(s31: float) -> str:
    if s31 >= 0.70:
        return "GEOMETRY_HIGH"
    if s31 < 0.40:
        return "GEOMETRY_LOW"
    return "GEOMETRY_INTERMEDIATE"


def functional_classification(median_a: float, median_b: float) -> str:
    if median_a >= 15 and median_b >= 15:
        return "FUNCTION_PRESERVED"
    if median_a <= 5 or median_b <= 5:
        return "FUNCTION_COLLAPSED"
    return "FUNCTION_INTERMEDIATE"


def frozen_outcome(geometry: str, function: str) -> str:
    if geometry == "GEOMETRY_HIGH" and function == "FUNCTION_PRESERVED":
        return "SIGNED_TICA_IDENTIFIABLE"
    if geometry == "GEOMETRY_LOW" and function == "FUNCTION_PRESERVED":
        return "NONCANONICAL_FAT_SLOW_FAMILY"
    if geometry == "GEOMETRY_LOW" and function == "FUNCTION_COLLAPSED":
        return "SIGNED_TICA_SAMPLE_DEPENDENT"
    return "SIGNED_TICA_IDENTIFIABILITY_MIXED"


def summarize_tau(table: pd.DataFrame) -> dict[str, float | int]:
    values = table.loc[table["tau_valid_within"].astype(bool), "tau_within"].to_numpy(dtype=float)
    if not len(values):
        return {"count": 0}
    q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.9, 0.95])
    return {
        "count": int(len(values)), "min": float(values.min()), "q05": float(q[0]),
        "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]),
        "q90": float(q[4]), "q95": float(q[5]), "max": float(values.max()),
        "mean": float(values.mean()),
        "right_censoring_fraction": float(table["right_censored_within"].astype(bool).mean()),
    }
