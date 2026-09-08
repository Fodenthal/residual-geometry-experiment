from __future__ import annotations

import numpy as np

from residual_geometry.residuals.pca import fit_residual_pca, sample_residual_positions
from residual_geometry.residuals.time_lagged import fit_time_lagged_residual_directions


def test_fit_residual_pca_returns_probe_directions_and_diagnostics() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(80, 6)).astype(np.float32)
    fit = fit_residual_pca(samples, n_components=3, seed=0)
    assert fit.probes.directions.shape == (3, 6)
    assert "effective_rank" in fit.diagnostics
    assert len(fit.eigenvalues) == 3


def test_sample_residual_positions_is_deterministic_without_replacement() -> None:
    batches = [
        np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2),
        np.arange(12, 24, dtype=np.float32).reshape(2, 3, 2),
    ]

    def factory():
        yield from batches

    a = sample_residual_positions(factory, total_positions=12, max_positions=5, seed=7, d_model=2)
    b = sample_residual_positions(factory, total_positions=12, max_positions=5, seed=7, d_model=2)
    assert a.shape == (5, 2)
    np.testing.assert_allclose(a, b)


def test_time_lagged_fit_recovers_persistent_synthetic_direction() -> None:
    rng = np.random.default_rng(1)
    docs, positions, d_model = 32, 48, 5
    state = rng.normal(size=(docs, 1, 1)).astype(np.float32)
    noise = 0.05 * rng.normal(size=(docs, positions, d_model)).astype(np.float32)
    residuals = noise
    residuals[:, :, 0:1] += state
    fit = fit_time_lagged_residual_directions(
        residuals,
        lag_set=[1, 2, 4],
        whitening_pcs=4,
        output_directions=2,
        ridge_scale=1e-4,
        max_ridge_scale=1e-1,
        condition_threshold=1e4,
    )
    assert fit.probes.directions.shape == (2, d_model)
    assert abs(float(fit.probes.directions[0, 0])) > 0.8
    assert "epsilon_scale" in fit.summary
