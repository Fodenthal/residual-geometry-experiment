from __future__ import annotations

import numpy as np
import pandas as pd

from residual_geometry.autocorr.estimators import compute_document_autocorr, extract_tau
from residual_geometry.utils.seed import get_rng


def matched_sparsity_sample(activations: np.ndarray, seed: int) -> np.ndarray:
    rng = get_rng(seed)
    docs, positions, features = activations.shape
    out = np.zeros_like(activations, dtype=np.float32)
    flat = activations.reshape(-1, features)
    for feature_i in range(features):
        positive = flat[:, feature_i][flat[:, feature_i] > 0]
        p = len(positive) / flat.shape[0]
        fires = rng.random(docs * positions) < p
        if len(positive) and fires.any():
            out.reshape(-1, features)[fires, feature_i] = rng.choice(positive, size=int(fires.sum()), replace=True)
    return out


def document_permutation_sample(activations: np.ndarray, seed: int) -> np.ndarray:
    rng = get_rng(seed)
    out = activations.copy()
    docs, _, features = out.shape
    for doc_i in range(docs):
        for feature_i in range(features):
            out[doc_i, :, feature_i] = out[doc_i, rng.permutation(out.shape[1]), feature_i]
    return out


def null_tau_table(
    activations: np.ndarray,
    feature_indices: np.ndarray,
    null_name: str,
    replicates: int,
    max_lag: int,
    min_valid_docs: int,
    min_valid_lag_fraction: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for replicate in range(replicates):
        sample_seed = seed + replicate
        if null_name == "matched_sparsity":
            sample = matched_sparsity_sample(activations, sample_seed)
        elif null_name == "document_permutation":
            sample = document_permutation_sample(activations, sample_seed)
        else:
            raise ValueError(f"Unknown null: {null_name}")
        for estimator in ["raw", "within", "binary"]:
            thresholds = None
            if estimator == "binary":
                flat = sample.reshape(-1, sample.shape[-1])
                thresholds = np.array(
                    [
                        np.quantile(values[values > 0], 0.99) if np.any(values > 0) else np.inf
                        for values in flat.T
                    ]
                )
            result = compute_document_autocorr(sample, max_lag=max_lag, estimator=estimator, thresholds=thresholds)
            for selected_i, feature_index in enumerate(feature_indices):
                stats = extract_tau(
                    result.profiles[selected_i],
                    max_lag=max_lag,
                    valid_counts=result.valid_doc_counts[selected_i],
                    min_valid_docs=min_valid_docs,
                    min_valid_lag_fraction=min_valid_lag_fraction,
                )
                rows.append(
                    {
                        "null": null_name,
                        "replicate": replicate,
                        "estimator": estimator,
                        "feature_index": int(feature_index),
                        **stats,
                    }
                )
    return pd.DataFrame(rows)

