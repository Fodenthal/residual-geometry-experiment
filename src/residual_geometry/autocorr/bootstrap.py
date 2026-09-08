from __future__ import annotations

import numpy as np
import pandas as pd

from residual_geometry.autocorr.estimators import compute_document_autocorr_matrix, extract_tau
from residual_geometry.utils.seed import get_rng


def bootstrap_timescales_for_array(
    activations: np.ndarray,
    feature_indices: np.ndarray,
    estimator: str,
    max_lag: int,
    min_valid_docs: int,
    min_valid_lag_fraction: float,
    replicates: int,
    seed: int,
    thresholds: np.ndarray | None = None,
    ci_lags: list[int] | None = None,
    smoothing_width: int = 5,
) -> pd.DataFrame:
    corr_matrix, valid_matrix = compute_document_autocorr_matrix(
        activations,
        max_lag=max_lag,
        estimator=estimator,
        thresholds=thresholds,
    )
    docs, features, lags = corr_matrix.shape
    ci_lags = [lag for lag in (ci_lags or [1, 2, 4, 8, 16, 32, 64, 128, 256]) if lag < lags]
    rng = get_rng(seed)
    tau_samples = np.zeros((replicates, features), dtype=np.int32)
    censored_samples = np.zeros((replicates, features), dtype=bool)
    profile_samples = {lag: np.full((replicates, features), np.nan, dtype=np.float32) for lag in ci_lags}

    for replicate in range(replicates):
        sample_indices = rng.integers(0, docs, size=docs)
        sample_corr = corr_matrix[sample_indices]
        sample_valid = valid_matrix[sample_indices]
        valid_counts = sample_valid.sum(axis=0)
        summed = np.where(sample_valid, sample_corr, 0.0).sum(axis=0)
        profiles = np.full((features, lags), np.nan, dtype=np.float64)
        valid = valid_counts > 0
        profiles[valid] = summed[valid] / valid_counts[valid]
        profiles[:, 0] = 1.0
        for feature_i in range(features):
            stats = extract_tau(
                profiles[feature_i],
                max_lag=max_lag,
                valid_counts=valid_counts[feature_i],
                min_valid_docs=min_valid_docs,
                min_valid_lag_fraction=min_valid_lag_fraction,
                smoothing_width=smoothing_width,
            )
            tau_samples[replicate, feature_i] = int(stats["tau"])
            censored_samples[replicate, feature_i] = bool(stats["right_censored"])
        for lag in ci_lags:
            profile_samples[lag][replicate] = profiles[:, lag]

    rows = []
    for feature_i, feature_index in enumerate(feature_indices):
        row = {
            "feature_index": int(feature_index),
            "estimator": estimator,
            "tau_ci_low": float(np.quantile(tau_samples[:, feature_i], 0.025)),
            "tau_ci_high": float(np.quantile(tau_samples[:, feature_i], 0.975)),
            "tau_bootstrap_mean": float(np.mean(tau_samples[:, feature_i])),
            "right_censoring_rate": float(np.mean(censored_samples[:, feature_i])),
        }
        for lag in ci_lags:
            values = profile_samples[lag][:, feature_i]
            row[f"R_lag_{lag}_ci_low"] = float(np.nanquantile(values, 0.025))
            row[f"R_lag_{lag}_ci_high"] = float(np.nanquantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def flag_high_confidence_slow(timescales: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    within_bootstrap = bootstrap[bootstrap["estimator"] == "within"][
        ["feature_index", "tau_ci_low", "tau_ci_high", "right_censoring_rate"]
    ].rename(
        columns={
            "tau_ci_low": "tau_within_ci_low",
            "tau_ci_high": "tau_within_ci_high",
            "right_censoring_rate": "within_right_censoring_rate",
        }
    )
    out = timescales.merge(within_bootstrap, on="feature_index", how="left")
    median_tau = float(out.loc[out["tau_valid_within"], "tau_within"].median())
    top_decile = out["tau_within"].rank(pct=True) >= 0.9
    binary_median = float(out.loc[out["tau_valid_binary"], "tau_binary"].median()) if "tau_valid_binary" in out else 0.0
    out["high_confidence_slow"] = (
        top_decile
        & (out["tau_within_ci_low"] > median_tau)
        & (out["tau_binary"] > binary_median)
        & out["tau_valid_within"]
    )
    return out[out["high_confidence_slow"]].copy().reset_index(drop=True)
