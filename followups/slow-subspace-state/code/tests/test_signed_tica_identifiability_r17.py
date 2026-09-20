from __future__ import annotations

import numpy as np

from src.persistent_state.slow_semantic.r17 import (
    combine_raw_statistics,
    frozen_half_indices,
    frozen_outcome,
    functional_classification,
    geometry_classification,
    random_in_span,
    raw_statistics_to_centered,
    subspace_overlap,
)


def _raw(x: np.ndarray, lags=(1, 2)) -> dict[str, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    out = {"token_count": np.array(len(flat)), "sum": flat.sum(0), "second_moment_sum": flat.T @ flat}
    for lag in lags:
        left = x[:, :-lag].reshape(-1, x.shape[-1]).astype(np.float64)
        right = x[:, lag:].reshape(-1, x.shape[-1]).astype(np.float64)
        out[f"lag_{lag}_count"] = np.array(len(left))
        out[f"lag_{lag}_left_sum"] = left.sum(0)
        out[f"lag_{lag}_right_sum"] = right.sum(0)
        out[f"lag_{lag}_cross_sum"] = left.T @ right
    return out


def test_raw_statistics_reproduce_direct_centered_covariances() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 8, 4))
    mean, sigma0, sigma_lag, pairs, count = raw_statistics_to_centered(_raw(x), lags=(1, 2))
    centered = x - x.reshape(-1, 4).mean(0)
    expected0 = centered.reshape(-1, 4).T @ centered.reshape(-1, 4) / 40
    expected_lag = sum(
        0.5 * ((centered[:, :-lag].reshape(-1, 4).T @ centered[:, lag:].reshape(-1, 4)) +
               (centered[:, lag:].reshape(-1, 4).T @ centered[:, :-lag].reshape(-1, 4))) /
        (5 * (8-lag)) for lag in (1, 2)
    ) / 2
    np.testing.assert_allclose(mean, x.reshape(-1, 4).mean(0), rtol=1e-6)
    np.testing.assert_allclose(sigma0, expected0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sigma_lag, expected_lag, rtol=1e-12, atol=1e-12)
    assert (pairs, count) == (65, 40)


def test_additive_raw_statistics_equal_concatenated_documents() -> None:
    rng = np.random.default_rng(8)
    a, b = rng.normal(size=(2, 8, 4)), rng.normal(size=(3, 8, 4))
    merged = combine_raw_statistics(_raw(a), _raw(b))
    got = raw_statistics_to_centered(merged, lags=(1, 2))
    expected = raw_statistics_to_centered(_raw(np.concatenate([a, b])), lags=(1, 2))
    for left, right in zip(got[:3], expected[:3]):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


def test_frozen_split_and_span_sampler_are_deterministic() -> None:
    a1, b1 = frozen_half_indices(); a2, b2 = frozen_half_indices()
    assert len(a1) == len(b1) == 2000
    np.testing.assert_array_equal(a1, a2); np.testing.assert_array_equal(b1, b2)
    assert len(np.intersect1d(a1, b1)) == 0
    q = np.eye(10, 3)
    d1, c1 = random_in_span(q); d2, c2 = random_in_span(q)
    np.testing.assert_array_equal(d1, d2); np.testing.assert_array_equal(c1, c2)
    np.testing.assert_allclose(np.linalg.norm(d1, axis=1), 1.0, rtol=1e-6)


def test_overlap_and_frozen_decisions() -> None:
    q = np.eye(8, 3)
    score, singular = subspace_overlap(q, q)
    assert score == 1.0 and np.all(singular == 1)
    assert geometry_classification(.7) == "GEOMETRY_HIGH"
    assert geometry_classification(.39) == "GEOMETRY_LOW"
    assert functional_classification(15, 15) == "FUNCTION_PRESERVED"
    assert functional_classification(5, 20) == "FUNCTION_COLLAPSED"
    assert frozen_outcome("GEOMETRY_LOW", "FUNCTION_PRESERVED") == "NONCANONICAL_FAT_SLOW_FAMILY"
