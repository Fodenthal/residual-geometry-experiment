"""One target, one baseline, one scoring path -- for every carrier, ceiling and deletion.

Source section 3 fixes the score: the held-out gain of a carrier's unrestricted decoder
over a frozen, carrier-independent baseline, on the SAME 256-dimensional future aperture
target.  This module is that score and nothing else, so a ceiling, a deletion refit and a
candidate carrier are all scored by the same code path on the same documents.

Three numbers come out of one set of fits (unified section 3):

``gamma_raw``    the source section 3 score exactly -- decoder reads ``z_t`` alone
``gamma_aug``    the baseline plus a decoder of ``z_t``, which is the quantity amendment
                 A1's residualized objective actually optimizes
``gamma_white``  the same augmented contrast in the destination-whitened metric

Everything is returned as per-document energies rather than as scores, so the paired
document bootstrap resamples every carrier together without refitting anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from arm_d_r4.common_target import (  # noqa: F401
    GroupedRidge,
    bootstrap_weights,
    gain_replicates,
    per_document_energy,
    r2_from_energy,
)

from arm_d_r24.contract import (
    LAGS_Q,
    PRIMARY_SCORE,
    BASELINE_MARGIN,
    BASELINE_ORDER,
    RIDGE_GRID,
    SEED,
    unavailable,
)
from arm_d_r24.data import Pool


def document_codes(documents: np.ndarray) -> tuple[np.ndarray, int]:
    labels, codes = np.unique(np.asarray(documents), return_inverse=True)
    return codes, int(labels.shape[0])


def symmetric_inverse_sqrt(matrix: np.ndarray, ridge: float) -> np.ndarray:
    """``(M + ridge I)^{-1/2}`` by eigendecomposition, in float64."""

    array = np.asarray(matrix, dtype=np.float64)
    array = 0.5 * (array + array.T)
    values, vectors = np.linalg.eigh(array)
    shifted = np.clip(values + float(ridge), a_min=1e-300, a_max=None)
    return (vectors * shifted ** -0.5) @ vectors.T


def condition_number(matrix: np.ndarray, ridge: float) -> float:
    array = np.asarray(matrix, dtype=np.float64)
    values = np.linalg.eigvalsh(0.5 * (array + array.T)) + float(ridge)
    low = float(np.min(values))
    return float(np.max(values) / low) if low > 0 else float("inf")


@dataclass
class Frame:
    """Everything estimated on fit documents: means, baselines, whiteners, targets.

    A frame is fitted once per (pool, fit mask, rho) and then applied unchanged to every
    other split.  No validation or test row enters any quantity stored here, which is the
    numerical contract's central invariant rather than a convention.
    """

    lags: tuple[int, ...]
    fit_mask: np.ndarray
    source: np.ndarray                      # centred x_t, all rows
    increment: np.ndarray                   # centred u_t = x_t - x_{t-1}, all rows
    target: dict[int, np.ndarray]           # centred y_k, all rows
    residual: dict[int, np.ndarray]         # y_k - b_k(x_t), all rows
    baseline: dict[int, dict[str, object]]  # the per-lag baseline record
    whitener: dict[int, np.ndarray]         # destination whitener of the residual target
    source_covariance: np.ndarray
    residual_covariance: dict[int, np.ndarray]
    cross_covariance: dict[int, np.ndarray]
    raw_cross_covariance: dict[int, np.ndarray]
    rho: float
    conditioning: dict[str, object] = field(default_factory=dict)
    #: Optional orthonormal source map the ESTIMATOR sees (the complement refit of source
    #: section 12).  Scoring always uses the full aperture source and the full target, so a
    #: complement carrier is compared against every other carrier on one common object.
    projection: np.ndarray | None = None

    @property
    def n_rows(self) -> int:
        return int(self.source.shape[0])

    def baseline_prediction(self, lag: int) -> np.ndarray:
        return self.target[int(lag)] - self.residual[int(lag)]


def select_baseline(scores: dict[str, float], margin: float = BASELINE_MARGIN) -> str:
    """Amendment A14: the simplest baseline wins unless a richer one beats it by ``margin``.

    ``scores`` are pooled held-out R-squared values, so differences far below the margin are
    noise rather than evidence that a richer predictor is doing work.
    """

    winner = BASELINE_ORDER[0]
    for name in BASELINE_ORDER[1:]:
        if name in scores and float(scores[name]) > float(scores[winner]) + float(margin):
            winner = name
    return winner


def fit_frame(
    pool: Pool,
    fit_mask: np.ndarray,
    *,
    lags: Sequence[int] = LAGS_Q,
    rho: float = 1e-2,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    seed: int = SEED,
    projection: np.ndarray | None = None,
    solver: GroupedRidge | None = None,
) -> Frame:
    """Fit means, the baseline bank, the residualized targets and the whiteners.

    ``solver`` lets a caller that refits the same fit documents many times -- the nulls, the
    bootstrap ceiling -- build the increment baseline's ridge machinery once.  It depends only
    on the increment design and the document grouping, both of which a derangement of the
    FUTURE leaves untouched, so sharing it changes no number.

    ``projection`` restricts what the ESTIMATOR sees without touching the baseline, the
    target or the scoring path: it is how source section 12's complement refit reruns the
    whole estimator inside ``(I - QQ')`` while still predicting the unchanged
    256-dimensional future.
    """

    fit_mask = np.asarray(fit_mask, dtype=bool)
    documents = pool.documents
    x = pool.offsets[0]
    u = pool.offsets[0] - pool.offsets[-1]
    mu_x = x[fit_mask].mean(axis=0, keepdims=True)
    mu_u = u[fit_mask].mean(axis=0, keepdims=True)
    source = x - mu_x
    increment = u - mu_u

    if solver is None:
        solver = GroupedRidge(
            increment[fit_mask], documents[fit_mask], seed=seed, ridge_grid=ridge_grid
        )
    target: dict[int, np.ndarray] = {}
    residual: dict[int, np.ndarray] = {}
    baselines: dict[int, dict[str, object]] = {}
    for lag in lags:
        lag = int(lag)
        y = pool.offsets[lag]
        centred = y - y[fit_mask].mean(axis=0, keepdims=True)
        target[lag] = centred
        fit_target = centred[fit_mask]
        energy = float(np.sum(fit_target**2))
        scores = {"B0": 0.0}
        scores["B1"] = float(
            1.0 - float(np.sum((fit_target - source[fit_mask]) ** 2)) / energy
        )
        b2_ridge = solver.select(fit_target)
        out_of_fold = solver.cv_predictions(fit_target, b2_ridge)
        scores["B2"] = float(1.0 - float(np.sum((fit_target - out_of_fold) ** 2)) / energy)
        winner = select_baseline(scores)
        if winner == "B0":
            prediction = np.zeros_like(centred)
        elif winner == "B1":
            prediction = source
        else:
            prediction = increment @ solver.fit(fit_target, b2_ridge)
        residual[lag] = centred - prediction
        baselines[lag] = {
            "lag": lag,
            "baseline": winner,
            "fit_cv_scores": scores,
            "baseline_margin": float(BASELINE_MARGIN),
            "b2_ridge": float(b2_ridge),
        }

    n_fit = int(fit_mask.sum())
    estimation = source if projection is None else source @ np.asarray(projection, dtype=np.float64)
    source_cov = (estimation[fit_mask].T @ estimation[fit_mask]) / max(n_fit - 1, 1)
    eps_source = float(rho) * float(np.trace(source_cov)) / source_cov.shape[0]
    whitener: dict[int, np.ndarray] = {}
    residual_cov: dict[int, np.ndarray] = {}
    cross: dict[int, np.ndarray] = {}
    raw_cross: dict[int, np.ndarray] = {}
    destination_conditions: dict[str, float] = {}
    for lag in lags:
        lag = int(lag)
        r_fit = residual[lag][fit_mask]
        cov = (r_fit.T @ r_fit) / max(n_fit - 1, 1)
        eps = float(rho) * float(np.trace(cov)) / cov.shape[0]
        residual_cov[lag] = cov
        whitener[lag] = symmetric_inverse_sqrt(cov, eps)
        cross[lag] = (estimation[fit_mask].T @ r_fit) / max(n_fit - 1, 1)
        raw_cross[lag] = (estimation[fit_mask].T @ target[lag][fit_mask]) / max(n_fit - 1, 1)
        destination_conditions[str(lag)] = condition_number(cov, eps)

    conditioning = {
        "rho": float(rho),
        "epsilon_source": eps_source,
        "source_condition_number": condition_number(source_cov, eps_source),
        "destination_condition_numbers": destination_conditions,
        "source_covariance_symmetry": float(
            np.max(np.abs(source_cov - source_cov.T)) / max(float(np.max(np.abs(source_cov))), 1e-300)
        ),
        "fit_documents": n_fit,
        "estimation_dimension": int(estimation.shape[1]),
        "estimation_projected": bool(projection is not None),
    }
    return Frame(
        lags=tuple(int(lag) for lag in lags),
        fit_mask=fit_mask,
        source=source,
        increment=increment,
        target=target,
        residual=residual,
        baseline=baselines,
        whitener=whitener,
        source_covariance=source_cov,
        residual_covariance=residual_cov,
        cross_covariance=cross,
        raw_cross_covariance=raw_cross,
        rho=float(rho),
        conditioning=conditioning,
        projection=None if projection is None else np.asarray(projection, dtype=np.float64),
    )


def score_projection(
    frame: Frame,
    projection: np.ndarray,
    *,
    documents: np.ndarray,
    score_mask: np.ndarray,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    seed: int = SEED,
    lags: Sequence[int] | None = None,
) -> dict[str, object]:
    """Fit decoders on the frame's fit rows and score them on ``score_mask``.

    ``projection`` is any linear source map ``z_t = x_t P`` -- an orthonormal carrier, a
    deletion projector's surviving coordinates, the identity for the full-aperture ceiling.
    The decoders are unrestricted and fitted separately per lag, as source section 3
    requires.
    """

    fit_mask = frame.fit_mask
    score_mask = np.asarray(score_mask, dtype=bool)
    p = np.asarray(projection, dtype=np.float64)
    z = frame.source @ p
    z_fit, z_score = z[fit_mask], z[score_mask]
    solver = GroupedRidge(z_fit, documents[fit_mask], seed=seed, ridge_grid=ridge_grid)
    codes, n_documents = document_codes(documents[score_mask])

    per_lag: dict[str, object] = {}
    ridges: dict[str, object] = {}
    for lag in (frame.lags if lags is None else [int(value) for value in lags]):
        lag = int(lag)
        y = frame.target[lag]
        r = frame.residual[lag]
        w = frame.whitener[lag]

        d_raw, ridge_raw = solver.select_and_fit(y[fit_mask])
        d_res, ridge_res = solver.select_and_fit(r[fit_mask])
        white_target = r @ w
        d_white, ridge_white = solver.select_and_fit(white_target[fit_mask])

        y_score = y[score_mask]
        r_score = r[score_mask]
        wt_score = white_target[score_mask]
        raw_prediction = z_score @ d_raw
        # The augmented and whitened paths add a decoder of z on top of the baseline, so
        # what is stored is directly the residual THAT PATH leaves.
        augmented_residual = r_score - z_score @ d_res
        white_residual = wt_score - z_score @ d_white

        per_lag[str(lag)] = {
            "lag": lag,
            "target_energy": per_document_energy(y_score, codes, n_documents),
            "baseline_residual": per_document_energy(r_score, codes, n_documents),
            "raw_residual": per_document_energy(y_score - raw_prediction, codes, n_documents),
            "augmented_residual": per_document_energy(augmented_residual, codes, n_documents),
            "white_target_energy": per_document_energy(
                y_score @ w, codes, n_documents
            ),
            "white_baseline_residual": per_document_energy(wt_score, codes, n_documents),
            "white_residual": per_document_energy(white_residual, codes, n_documents),
        }
        ridges[str(lag)] = {
            "raw": float(ridge_raw),
            "residual": float(ridge_res),
            "whitened": float(ridge_white),
            # Amendment A13: an edge selection means the grid, not the data, chose the
            # regularizer, so it travels with every score rather than being inferred later.
            "at_grid_edge": {
                name: bool(value in (float(min(ridge_grid)), float(max(ridge_grid))))
                for name, value in (
                    ("raw", float(ridge_raw)),
                    ("residual", float(ridge_res)),
                    ("whitened", float(ridge_white)),
                )
            },
        }
    return {
        "dimension": int(p.shape[1]),
        "n_documents": n_documents,
        "documents": np.unique(documents[score_mask]),
        "per_lag": per_lag,
        "decoder_ridges": ridges,
    }


def _gain(residual: np.ndarray, base: np.ndarray, energy: np.ndarray, weights=None) -> float:
    if weights is None:
        numerator = float(np.mean(base) - np.mean(residual))
        denominator = float(np.mean(energy))
    else:
        total = float(np.sum(weights))
        numerator = float(np.dot(weights, base - residual) / total)
        denominator = float(np.dot(weights, energy) / total)
    return numerator / denominator if denominator > 0 else float("nan")


def gains(record: Mapping[str, object], *, weights: np.ndarray | None = None) -> dict[str, object]:
    """The three scores of unified section 3, per lag plus the multi-horizon mean."""

    per_lag: dict[str, object] = {}
    for key, entry in record["per_lag"].items():  # type: ignore[index]
        per_lag[key] = {
            "lag": int(entry["lag"]),
            "gamma_raw": _gain(
                entry["raw_residual"], entry["baseline_residual"], entry["target_energy"]
            ),
            "gamma_augmented": _gain(
                entry["augmented_residual"], entry["baseline_residual"], entry["target_energy"]
            ),
            "gamma_white": _gain(
                entry["white_residual"],
                entry["white_baseline_residual"],
                entry["white_target_energy"],
            ),
            "baseline_r2": r2_from_energy(entry["baseline_residual"], entry["target_energy"]),
            "raw_r2": r2_from_energy(entry["raw_residual"], entry["target_energy"]),
        }
    summary = {
        name: float(np.mean([per_lag[key][name] for key in per_lag]))
        for name in ("gamma_raw", "gamma_augmented", "gamma_white")
    }
    out: dict[str, object] = {"per_lag": per_lag, "multi_horizon_mean": summary}
    if weights is not None:
        out["bootstrap"] = bootstrap_summary(record, weights)
    return out


def bootstrap_summary(record: Mapping[str, object], weights: np.ndarray) -> dict[str, object]:
    """Paired document-bootstrap replicates of the multi-horizon mean of each score."""

    fields = {
        "gamma_raw": ("raw_residual", "baseline_residual", "target_energy"),
        "gamma_augmented": ("augmented_residual", "baseline_residual", "target_energy"),
        "gamma_white": ("white_residual", "white_baseline_residual", "white_target_energy"),
    }
    out: dict[str, object] = {"replicates": int(weights.shape[0])}
    for name, (residual, base, energy) in fields.items():
        stack = []
        for entry in record["per_lag"].values():  # type: ignore[union-attr]
            stack.append(
                gain_replicates(entry[residual], entry[base], entry[energy], weights)
            )
        mean = np.mean(np.stack(stack, axis=0), axis=0)
        out[name] = {
            "mean": float(np.mean(mean)),
            "standard_error": float(np.std(mean, ddof=1)),
            "q05": float(np.quantile(mean, 0.05)),
            "q50": float(np.quantile(mean, 0.50)),
            "q95": float(np.quantile(mean, 0.95)),
        }
    return out


def score_carrier(
    frame: Frame,
    basis: np.ndarray,
    *,
    documents: np.ndarray,
    score_mask: np.ndarray,
    weights: np.ndarray | None = None,
    **kwargs,
) -> dict[str, object]:
    record = score_projection(
        frame, basis, documents=documents, score_mask=score_mask, **kwargs
    )
    summary = gains(record, weights=weights)
    summary["rank"] = int(np.asarray(basis).shape[1])
    summary["decoder_ridges"] = record["decoder_ridges"]
    return {"record": record, "summary": summary}


def to_aperture(frame: Frame, basis: np.ndarray) -> np.ndarray:
    """Map a carrier estimated under a projection back into aperture coordinates."""

    if frame.projection is None:
        return np.asarray(basis, dtype=np.float64)
    return frame.projection @ np.asarray(basis, dtype=np.float64)


def orthonormality_error(basis: np.ndarray) -> float:
    q = np.asarray(basis, dtype=np.float64)
    return float(np.max(np.abs(q.T @ q - np.eye(q.shape[1]))))


def positive_gain_mask(gain_by_lag: Mapping[str, object], floor: float) -> dict[str, bool]:
    return {
        key: bool(float(entry[PRIMARY_SCORE]) >= float(floor))
        for key, entry in gain_by_lag.items()
    }


def surviving_fraction(
    numerator: Mapping[str, object], denominator: Mapping[str, object], *, floor: float
) -> dict[str, object]:
    """``S_D(k)`` with the frozen positive-gain floor on the denominator."""

    out: dict[str, object] = {}
    for key, entry in denominator.items():
        base = float(entry[PRIMARY_SCORE])
        if base < float(floor):
            out[key] = {
                "unavailable": unavailable(
                    f"surviving_fraction_lag_{key}",
                    f"the full-aperture gain {base:.6f} is below the frozen positive-gain "
                    f"floor {floor}",
                )
            }
            continue
        out[key] = {
            "surviving_fraction": float(numerator[key][PRIMARY_SCORE]) / base,  # type: ignore[index]
            "numerator_primary_gain": float(numerator[key][PRIMARY_SCORE]),  # type: ignore[index]
            "denominator_primary_gain": base,
        }
    return out
