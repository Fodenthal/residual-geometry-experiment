"""The two multi-horizon carrier estimators, and the three predictive ceilings.

Source section 8 fits two metrics deliberately, so that a high-variance objective cannot
silently define "the state" by construction:

``Q-E``   energy-weighted -- source directions whose linear information predicts large
          Euclidean future-block variance;
``Q-W``   covariance-whitened -- destination directions are variance-normalized before
          their predictability contributes, which is the primary scientific estimator.

Two amendments change how the operators are assembled.  A1 builds every cross-covariance
from the persistence-residualized target, because the score is a gain over persistence and
a raw-target objective is rewarded for predicting the present.  A2 trace-normalizes each
lag's term before summing, because cross-covariance magnitude decays with lag and equal
weights apply to objective terms rather than to scores.

Neither estimator sees a composition, semigroup or spectral-simplicity term.  That
prohibition is the load-bearing rule of the whole specification: a carrier fitted to
compose builds a later autonomy result into its own estimator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from arm_d_r24.contract import LAGS_Q, MAX_CONDITION_NUMBER, RANK_GRID
from arm_d_r24.scoring import Frame, symmetric_inverse_sqrt

FORBIDDEN_TERMS = (
    "composition",
    "semigroup consistency",
    "spectral simplicity",
    "causal transport",
)


def operator_terms(
    frame: Frame,
    *,
    metric: str,
    lags: Sequence[int] | None = None,
    normalize: bool = True,
    use_raw_cross: bool = False,
) -> dict[str, object]:
    """The per-lag terms of ``G_E`` or ``G_W`` before they are summed.

    Returned separately so the A2 sensitivity (unnormalized) and the A1 diagnostic (raw
    cross-covariance) are the same code with two flags rather than three implementations.
    """

    if metric not in {"energy", "whitened"}:
        raise ValueError(f"unknown metric {metric!r}")
    lags = tuple(int(value) for value in (frame.lags if lags is None else lags))
    eps_source = float(frame.rho) * float(np.trace(frame.source_covariance)) / (
        frame.source_covariance.shape[0]
    )
    w0 = symmetric_inverse_sqrt(frame.source_covariance, eps_source)

    terms: dict[int, np.ndarray] = {}
    traces: dict[int, float] = {}
    cross_source = frame.raw_cross_covariance if use_raw_cross else frame.cross_covariance
    for lag in lags:
        c = cross_source[lag]
        left = w0 @ c
        if metric == "whitened":
            left = left @ frame.whitener[lag]
        term = left @ left.T
        term = 0.5 * (term + term.T)
        trace = float(np.trace(term))
        terms[lag] = term
        traces[lag] = trace
    return {"terms": terms, "traces": traces, "whitener_source": w0, "lags": lags,
            "normalize": bool(normalize)}


def multi_horizon_operator(
    frame: Frame,
    *,
    metric: str,
    lags: Sequence[int] | None = None,
    normalize: bool = True,
    use_raw_cross: bool = False,
) -> dict[str, object]:
    """``G_E`` or ``G_W`` of unified section 4, with its assembly recorded."""

    parts = operator_terms(
        frame, metric=metric, lags=lags, normalize=normalize, use_raw_cross=use_raw_cross
    )
    lags = parts["lags"]  # type: ignore[assignment]
    weight = 1.0 / float(len(lags))
    operator = np.zeros_like(next(iter(parts["terms"].values())))  # type: ignore[union-attr]
    contributions: dict[str, float] = {}
    for lag in lags:  # type: ignore[union-attr]
        term = parts["terms"][lag]  # type: ignore[index]
        trace = parts["traces"][lag]  # type: ignore[index]
        scale = weight / trace if (normalize and trace > 0) else weight
        operator += scale * term
        contributions[str(lag)] = float(scale * trace)
    values, vectors = np.linalg.eigh(0.5 * (operator + operator.T))
    order = np.argsort(values)[::-1]
    return {
        "metric": metric,
        "operator": operator,
        "eigenvalues": values[order],
        "eigenvectors": vectors[:, order],
        "whitener_source": parts["whitener_source"],
        "lag_trace": {str(lag): float(parts["traces"][lag]) for lag in lags},  # type: ignore[index]
        "lag_contribution_after_weighting": contributions,
        "normalized": bool(normalize),
        "raw_cross_covariance": bool(use_raw_cross),
        "lags": [int(lag) for lag in lags],  # type: ignore[union-attr]
    }


def carrier_from_operator(operator: Mapping[str, object], rank: int) -> np.ndarray:
    """``Q(r) = orth(W_0 V^{(r)})`` -- the source functionals, in aperture coordinates."""

    vectors = np.asarray(operator["eigenvectors"], dtype=np.float64)[:, : int(rank)]
    mapped = np.asarray(operator["whitener_source"], dtype=np.float64) @ vectors
    return np.linalg.qr(mapped)[0]


def carrier_family(
    frame: Frame,
    *,
    metric: str,
    ranks: Sequence[int] = RANK_GRID,
    lags: Sequence[int] | None = None,
    normalize: bool = True,
    use_raw_cross: bool = False,
) -> dict[str, object]:
    """One eigensystem, every rank in the grid, nested by construction."""

    operator = multi_horizon_operator(
        frame, metric=metric, lags=lags, normalize=normalize, use_raw_cross=use_raw_cross
    )
    bases = {int(rank): carrier_from_operator(operator, int(rank)) for rank in ranks}
    return {
        "metric": metric,
        "bases": bases,
        "eigenvalues": operator["eigenvalues"],
        "lag_trace": operator["lag_trace"],
        "lag_contribution_after_weighting": operator["lag_contribution_after_weighting"],
        "normalized": operator["normalized"],
        "raw_cross_covariance": operator["raw_cross_covariance"],
        "lags": operator["lags"],
        "operator": operator,
    }


# --------------------------------------------------------------------------
# Ceilings (unified section 6)
# --------------------------------------------------------------------------


def reduced_rank_source_subspace(
    frame: Frame, *, lag: int, rank: int, ridge: float | None = None
) -> np.ndarray:
    """The per-lag free-source reduced-rank bottleneck of source section 7.2.

    ``min_{rank(M)<=r} E||y - x M||^2`` factors as ``M = U B``, so the optimal rank-r SOURCE
    bottleneck is the row space of the rank-r approximation of the ridge solution measured
    in the ``Sigma_0`` metric.
    """

    lag = int(lag)
    sigma = frame.source_covariance
    eps = (
        float(ridge)
        if ridge is not None
        else float(frame.rho) * float(np.trace(sigma)) / sigma.shape[0]
    )
    solution = np.linalg.solve(sigma + eps * np.eye(sigma.shape[0]), frame.cross_covariance[lag])
    root = symmetric_inverse_sqrt(sigma, eps)
    sigma_sqrt = np.linalg.inv(root)
    left, _, _ = np.linalg.svd(sigma_sqrt @ solution, full_matrices=False)
    return np.linalg.qr(root @ left[:, : int(rank)])[0]


def shared_source_subspace(
    frame: Frame, *, rank: int, lags: Sequence[int] | None = None, ridge: float | None = None
) -> np.ndarray:
    """Amendment A7: ONE source subspace optimal for the score-weighted sum over lags.

    Each lag's explained energy is normalized by that lag's residual target energy -- the
    denominator the reported score divides by -- so this is the best single rank-r
    bottleneck for the quantity the report averages, under the shared-subspace constraint
    every candidate carrier carries.
    """

    sigma = frame.source_covariance
    eps = (
        float(ridge)
        if ridge is not None
        else float(frame.rho) * float(np.trace(sigma)) / sigma.shape[0]
    )
    root = symmetric_inverse_sqrt(sigma, eps)
    lags = tuple(int(value) for value in (frame.lags if lags is None else lags))
    operator = np.zeros((sigma.shape[0], sigma.shape[0]), dtype=np.float64)
    for lag in lags:
        energy = float(np.trace(frame.residual_covariance[lag]))
        left = root @ frame.cross_covariance[lag]
        operator += (left @ left.T) / max(energy, 1e-300)
    values, vectors = np.linalg.eigh(0.5 * (operator + operator.T))
    order = np.argsort(values)[::-1]
    return np.linalg.qr(root @ vectors[:, order][:, : int(rank)])[0]


def full_aperture_projection(width: int) -> np.ndarray:
    return np.eye(int(width), dtype=np.float64)


def conditioning_report(frame: Frame) -> dict[str, object]:
    """Source section 17's conditioning audit, with the instability verdict."""

    record = dict(frame.conditioning)
    worst_destination = max(
        float(value) for value in record["destination_condition_numbers"].values()  # type: ignore[union-attr]
    )
    record["worst_destination_condition_number"] = worst_destination
    record["max_condition_number"] = float(MAX_CONDITION_NUMBER)
    record["numerically_stable"] = bool(
        float(record["source_condition_number"]) <= MAX_CONDITION_NUMBER
        and worst_destination <= MAX_CONDITION_NUMBER
    )
    record["verdict"] = (
        "NUMERICALLY_STABLE" if record["numerically_stable"] else "NUMERICALLY_UNSTABLE"
    )
    return record


def eigensystem_replay(operator: Mapping[str, object], *, tolerance: float = 1e-8) -> dict[str, object]:
    """Source section 17: replay ``G V = V L`` and the orthonormality of ``V``."""

    matrix = np.asarray(operator["operator"], dtype=np.float64)
    vectors = np.asarray(operator["eigenvectors"], dtype=np.float64)
    values = np.asarray(operator["eigenvalues"], dtype=np.float64)
    residual = matrix @ vectors - vectors * values
    scale = max(float(np.max(np.abs(matrix))), 1e-300)
    return {
        "max_eigen_residual": float(np.max(np.abs(residual)) / scale),
        "max_orthonormality_error": float(
            np.max(np.abs(vectors.T @ vectors - np.eye(vectors.shape[1])))
        ),
        "symmetry_error": float(np.max(np.abs(matrix - matrix.T)) / scale),
        "tolerance": float(tolerance),
        "passes": bool(
            float(np.max(np.abs(residual)) / scale) < tolerance
            and float(np.max(np.abs(vectors.T @ vectors - np.eye(vectors.shape[1])))) < tolerance
        ),
    }


def block_form_equivalence(frame: Frame, *, metric: str = "energy") -> dict[str, object]:
    """Source section 17: the summed per-lag operator equals the direct block form.

    Stacking the lags into one wide destination block and forming ``W0 C C^T W0`` once must
    reproduce the per-lag sum exactly (up to rounding) when the weights are uniform and the
    normalization is off.
    """

    eps = float(frame.rho) * float(np.trace(frame.source_covariance)) / (
        frame.source_covariance.shape[0]
    )
    w0 = symmetric_inverse_sqrt(frame.source_covariance, eps)
    blocks = []
    for lag in frame.lags:
        c = frame.cross_covariance[lag]
        blocks.append(w0 @ c if metric == "energy" else (w0 @ c) @ frame.whitener[lag])
    stacked = np.concatenate(blocks, axis=1)
    direct = stacked @ stacked.T / float(len(frame.lags))
    summed = multi_horizon_operator(frame, metric=metric, normalize=False)["operator"]
    scale = max(float(np.max(np.abs(direct))), 1e-300)
    return {
        "metric": metric,
        "max_absolute_difference": float(np.max(np.abs(direct - summed))),
        "max_relative_difference": float(np.max(np.abs(direct - summed)) / scale),
        "passes": bool(float(np.max(np.abs(direct - summed)) / scale) < 1e-10),
    }
