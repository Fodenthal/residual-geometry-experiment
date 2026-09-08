from __future__ import annotations

import numpy as np

from residual_geometry.autocorr.bootstrap import bootstrap_timescales_for_array
from residual_geometry.autocorr.estimators import AutocorrAccumulator, build_timescale_table, compute_document_autocorr, extract_tau, smooth_profile
from residual_geometry.autocorr.nulls import document_permutation_sample, matched_sparsity_sample


def test_compute_document_autocorr_shapes() -> None:
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(5, 16, 3)).astype(np.float32)
    result = compute_document_autocorr(acts, max_lag=4, estimator="raw")
    assert result.profiles.shape == (3, 5)
    assert result.valid_doc_counts.shape == (3, 5)
    assert np.all(result.profiles[:, 0] == 1.0)


def test_within_autocorr_demeans_each_document_and_probe() -> None:
    base = np.tile(np.arange(8, dtype=np.float32), (2, 1))
    acts = np.stack(
        [
            np.stack([base[0] + 100.0, base[0] * 2.0 - 50.0], axis=-1),
            np.stack([base[1] - 200.0, base[1] * 2.0 + 25.0], axis=-1),
        ],
        axis=0,
    )
    within = compute_document_autocorr(acts, max_lag=2, estimator="within")
    shifted = acts.copy()
    shifted[0, :, 0] += 1000.0
    shifted[1, :, 1] -= 3000.0
    shifted_within = compute_document_autocorr(shifted, max_lag=2, estimator="within")
    np.testing.assert_allclose(within.profiles, shifted_within.profiles, atol=1e-10)


def test_tau_crossing_starts_at_lag_one_and_smoothing_excludes_lag_zero() -> None:
    profile = np.array([1.0, 0.1, 0.1, 0.1, 0.9], dtype=np.float64)
    smoothed = smooth_profile(profile, width=5)
    assert smoothed[0] == 1.0
    assert smoothed[1] < 1.0 / np.e
    stats = extract_tau(
        profile,
        max_lag=4,
        valid_counts=np.ones(5, dtype=np.int64) * 10,
        min_valid_docs=1,
        min_valid_lag_fraction=0.8,
        smoothing_width=5,
    )
    assert stats["tau"] == 1
    assert stats["right_censored"] is False


def test_tau_ignores_lag_zero_even_if_profile_zero_is_small() -> None:
    profile = np.array([0.0, 0.9, 0.2], dtype=np.float64)
    stats = extract_tau(
        profile,
        max_lag=2,
        valid_counts=np.ones(3, dtype=np.int64) * 10,
        min_valid_docs=1,
        min_valid_lag_fraction=0.8,
        smoothing_width=1,
    )
    assert stats["tau"] == 2


def test_autocorr_accumulator_matches_full_batch_average() -> None:
    rng = np.random.default_rng(3)
    acts = rng.normal(size=(8, 16, 2)).astype(np.float32)
    full = compute_document_autocorr(acts, max_lag=3, estimator="raw")
    accumulator = AutocorrAccumulator(n_features=2, max_lag=3)
    accumulator.update(compute_document_autocorr(acts[:4], max_lag=3, estimator="raw"))
    accumulator.update(compute_document_autocorr(acts[4:], max_lag=3, estimator="raw"))
    streamed = accumulator.finalize()
    np.testing.assert_allclose(streamed.profiles, full.profiles, equal_nan=True)


def test_build_timescale_table_adds_slow_score() -> None:
    rng = np.random.default_rng(1)
    acts = rng.normal(size=(6, 18, 2)).astype(np.float32)
    raw = compute_document_autocorr(acts, max_lag=5, estimator="raw")
    within = compute_document_autocorr(acts, max_lag=5, estimator="within")
    binary = compute_document_autocorr(acts, max_lag=5, estimator="binary", thresholds=np.zeros(2))
    table = build_timescale_table(
        np.array([10, 20]),
        {"raw": raw, "within": within, "binary": binary},
        max_lag=5,
        min_valid_docs=1,
        min_valid_lag_fraction=0.5,
    )
    assert "slow_score" in table.columns
    assert table["feature_index"].tolist() == [10, 20]


def test_build_timescale_table_allows_within_only() -> None:
    rng = np.random.default_rng(2)
    acts = rng.normal(size=(6, 18, 2)).astype(np.float32)
    within = compute_document_autocorr(acts, max_lag=5, estimator="within")
    table = build_timescale_table(
        np.array([10, 20]),
        {"within": within},
        max_lag=5,
        min_valid_docs=1,
        min_valid_lag_fraction=0.5,
    )
    assert "slow_score" in table.columns
    assert table["slow_score"].notna().all()


def test_build_timescale_table_preserves_empty_schema() -> None:
    result = AutocorrAccumulator(n_features=0, max_lag=5).finalize()
    table = build_timescale_table(
        np.array([], dtype=np.int64),
        {"before": result},
        max_lag=5,
        min_valid_docs=1,
        min_valid_lag_fraction=0.5,
    )
    assert table.empty
    assert {"feature_index", "tau_before", "tau_valid_before", "slow_score"}.issubset(table.columns)


def test_null_samples_preserve_shape() -> None:
    acts = np.array([[[0.0, 1.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 3.0]]], dtype=np.float32)
    assert matched_sparsity_sample(acts, seed=0).shape == acts.shape
    assert document_permutation_sample(acts, seed=0).shape == acts.shape


def test_bootstrap_timescales_returns_feature_rows() -> None:
    rng = np.random.default_rng(4)
    acts = rng.normal(size=(8, 20, 3)).astype(np.float32)
    table = bootstrap_timescales_for_array(
        acts,
        feature_indices=np.array([1, 2, 3]),
        estimator="within",
        max_lag=4,
        min_valid_docs=1,
        min_valid_lag_fraction=0.5,
        replicates=3,
        seed=0,
        ci_lags=[1, 2],
    )
    assert table["feature_index"].tolist() == [1, 2, 3]
    assert "tau_ci_low" in table.columns
