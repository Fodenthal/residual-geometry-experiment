from __future__ import annotations

import numpy as np

from residual_geometry.slow_semantic.protocol import deterministic_document_split, orthonormal_random_controls
from residual_geometry.slow_semantic.readout import cross_entropy_rows, fit_offset_multinomial


def test_frozen_document_split_counts_and_determinism():
    ids = [f"doc-{i}" for i in range(1200)]
    first = deterministic_document_split(ids)
    second = deterministic_document_split(ids)
    assert first == second
    assert {x: first.count(x) for x in set(first)} == {"fit": 800, "validation": 200, "test": 200}


def test_random_controls_are_inside_parent_and_orthonormal():
    rng = np.random.default_rng(4)
    parent, _ = np.linalg.qr(rng.standard_normal((64, 20)))
    controls = orthonormal_random_controls(parent, rank=7, count=3, seed=9)
    projector = parent @ parent.T
    for basis in controls:
        np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=1e-10)
        np.testing.assert_allclose(projector @ basis, basis, atol=1e-10)


def test_offset_multinomial_recovers_signal_and_zero_sum_gauge():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((500, 3))
    true = np.asarray([[1.2, -1.2], [-0.8, 0.8], [0.4, -0.4]])
    logits = x @ true
    p = np.exp(logits - np.logaddexp(logits[:, 0], logits[:, 1])[:, None])
    y = np.asarray([rng.choice(2, p=row) for row in p])
    offset = np.zeros((len(x), 2))
    fit = fit_offset_multinomial(offset, x, y, l2=1e-3)
    assert fit.success
    np.testing.assert_allclose(fit.coefficients.mean(axis=1), 0.0, atol=1e-10)
    assert cross_entropy_rows(fit.logits(offset, x), y).mean() < np.log(2)
