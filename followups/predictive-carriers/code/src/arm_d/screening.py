"""Natural-text screening: local-window nuisance and raw cross-lag fit (R1 §8, §14.4).

This is the analysis R0 made primary and R1 demotes.  It is retained because
spliced sequences are counterfactual and mildly off-distribution, so the report
needs to show what the same estimator sees on ordinary text -- but it never
carries a D1 verdict.

The nuisance contract is R1 §8: features from information available at or
before the source token only, a frozen 32-token local window represented by a
frozen input-embedding PCA at seven offsets plus pooled class and frequency
summaries, and document-grouped ridge selection.  Token-random CV is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_d.scoring import document_folds, equal_document_r2

LOCAL_WINDOW = 32
LOCAL_OFFSETS = (0, -1, -2, -4, -8, -16, -31)
FREQUENCY_BINS = 8


@dataclass(frozen=True)
class LocalFeatureSpec:
    embedding_dimension: int
    offsets: tuple[int, ...] = LOCAL_OFFSETS
    window: int = LOCAL_WINDOW
    frequency_bins: int = FREQUENCY_BINS

    def to_dict(self) -> dict[str, object]:
        return {
            "embedding_dimension_per_offset": int(self.embedding_dimension),
            "offsets": list(self.offsets),
            "window": int(self.window),
            "frequency_bins": int(self.frequency_bins),
            "information_set": "tokens at or before the source position only",
        }


def fit_embedding_pca(embedding_matrix: np.ndarray, dimension: int) -> np.ndarray:
    """Frozen PCA of the input-embedding table, fitted once and reused."""

    matrix = np.asarray(embedding_matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / centered.shape[0]
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1][: int(dimension)]
    return vectors[:, order]


def build_local_features(
    token_ids: np.ndarray,
    positions: np.ndarray,
    embedding_matrix: np.ndarray,
    embedding_basis: np.ndarray,
    frequency_rank: np.ndarray,
    spec: LocalFeatureSpec,
) -> np.ndarray:
    """Local-window features for one document at the requested source positions.

    ``token_ids`` is one document.  Every feature reads tokens at or before the
    source position; :func:`assert_no_future_leakage` is the test that keeps it
    that way.
    """

    document = np.asarray(token_ids)
    codes = np.asarray(embedding_matrix, dtype=np.float64) @ np.asarray(embedding_basis, dtype=np.float64)
    rows = []
    for position in np.asarray(positions):
        pieces = []
        for offset in spec.offsets:
            index = max(int(position) + int(offset), 0)
            pieces.append(codes[document[index]])
        window = document[max(int(position) - spec.window + 1, 0) : int(position) + 1]
        ranks = frequency_rank[window]
        histogram = np.bincount(
            np.clip((ranks * spec.frequency_bins).astype(int), 0, spec.frequency_bins - 1),
            minlength=spec.frequency_bins,
        ).astype(np.float64)
        histogram /= max(float(window.size), 1.0)
        pieces.append(histogram)
        pieces.append(
            np.array(
                [
                    float(np.unique(window).size / max(window.size, 1)),
                    float(ranks.mean()),
                    float(position) / 1024.0,
                    (float(position) / 1024.0) ** 2,
                ]
            )
        )
        rows.append(np.concatenate(pieces))
    return np.stack(rows)


def assert_no_future_leakage(
    token_ids: np.ndarray,
    positions: np.ndarray,
    build: callable,
    rng: np.random.Generator,
) -> bool:
    """Perturb every token strictly after the source and require no change."""

    document = np.asarray(token_ids).copy()
    reference = build(document, positions)
    latest = int(np.max(positions))
    if latest + 1 >= document.shape[0]:
        return True
    corrupted = document.copy()
    corrupted[latest + 1 :] = rng.integers(0, 1000, size=document.shape[0] - latest - 1)
    return bool(np.allclose(reference, build(corrupted, positions)))


def residualize(
    features_train: np.ndarray,
    targets_train: np.ndarray,
    documents_train: np.ndarray,
    features_eval: np.ndarray,
    targets_eval: np.ndarray,
    *,
    ridge_grid: tuple[float, ...] = (1e-2, 1.0, 1e2, 1e4, 1e6),
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Train-only nuisance residualization with document-grouped ridge CV."""

    design_train = np.column_stack([features_train, np.ones(features_train.shape[0])])
    design_eval = np.column_stack([features_eval, np.ones(features_eval.shape[0])])
    folds = document_folds(documents_train, n_folds=min(n_folds, len(np.unique(documents_train))), seed=seed)
    best_ridge, best_score = float(ridge_grid[0]), -np.inf
    for ridge in ridge_grid:
        scores = []
        for held_out in folds:
            mask = np.ones(design_train.shape[0], dtype=bool)
            mask[held_out] = False
            coefficients = np.linalg.solve(
                design_train[mask].T @ design_train[mask] + ridge * np.eye(design_train.shape[1]),
                design_train[mask].T @ targets_train[mask],
            )
            residual = targets_train[held_out] - design_train[held_out] @ coefficients
            energy = float(np.sum(targets_train[held_out] ** 2))
            if energy > 0:
                scores.append(1.0 - float(np.sum(residual**2)) / energy)
        if scores and float(np.mean(scores)) > best_score:
            best_score, best_ridge = float(np.mean(scores)), float(ridge)
    coefficients = np.linalg.solve(
        design_train.T @ design_train + best_ridge * np.eye(design_train.shape[1]),
        design_train.T @ targets_train,
    )
    predicted_train = design_train @ coefficients
    predicted_eval = design_eval @ coefficients
    explained = 1.0 - float(np.sum((targets_eval - predicted_eval) ** 2)) / float(np.sum(targets_eval**2))
    return {
        "ridge": best_ridge,
        "held_out_nuisance_r2": explained,
        "residual_variance_fraction": float(
            np.sum((targets_eval - predicted_eval) ** 2) / np.sum(targets_eval**2)
        ),
        "residual_train": targets_train - predicted_train,
        "residual_eval": targets_eval - predicted_eval,
    }


def trivial_baseline_r2(
    positions_train: np.ndarray,
    targets_train: np.ndarray,
    positions_eval: np.ndarray,
    targets_eval: np.ndarray,
) -> float:
    """Position-and-intercept-only comparator for ``test_nuisance_baseline_strength``."""

    design_train = np.column_stack(
        [positions_train / 1024.0, (positions_train / 1024.0) ** 2, np.ones(positions_train.shape[0])]
    )
    design_eval = np.column_stack(
        [positions_eval / 1024.0, (positions_eval / 1024.0) ** 2, np.ones(positions_eval.shape[0])]
    )
    coefficients = np.linalg.lstsq(design_train, targets_train, rcond=None)[0]
    residual = targets_eval - design_eval @ coefficients
    return 1.0 - float(np.sum(residual**2)) / float(np.sum(targets_eval**2))


def permute_within_document(values: np.ndarray, document_ids: np.ndarray, rng: np.random.Generator, block: int = 1) -> np.ndarray:
    """Token-order (``block=1``) or block-order destruction inside each document."""

    out = values.copy()
    for document in np.unique(document_ids):
        rows = np.flatnonzero(document_ids == document)
        if rows.size < 2:
            continue
        if block <= 1:
            out[rows] = values[rows][rng.permutation(rows.size)]
        else:
            n_blocks = int(np.ceil(rows.size / block))
            order = rng.permutation(n_blocks)
            pieces = [values[rows][index * block : (index + 1) * block] for index in order]
            out[rows] = np.concatenate(pieces, axis=0)[: rows.size]
    return out


def screening_score(
    source: np.ndarray,
    destination: np.ndarray,
    documents: np.ndarray,
    basis: np.ndarray,
    operator: np.ndarray,
) -> float:
    prediction = (source @ basis) @ operator.T
    return equal_document_r2(destination @ basis, prediction, documents).r2
