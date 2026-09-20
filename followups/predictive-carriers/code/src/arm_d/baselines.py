"""Mandatory trajectory baselines (R1 §15).

R1 addition A5 makes these the operative null for the paired object: because
Delta z is a difference that decays, "the difference just decays smoothly" is
the competitor hypothesis, and the headline is incremental over the strongest
*applicable* baseline, never over zero.

Applicability is not decoration.  The baselines split into two groups by the
information they consume, and mixing them makes the headline uninterpretable:

* **source-state baselines** -- persistence and the update-only comparator --
  see exactly what the operator sees, the state at the source token.
  Persistence is literally the operator with ``A = I``, so "the operator beats
  persistence" is a pure test of whether the fitted transition differs usefully
  from doing nothing.  These set the D1 headline.
* **trajectory baselines** -- constant velocity, local linear extrapolation,
  local autoregression -- consume five slots of the state trajectory.  Five
  noisy measurements of a decaying difference average out measurement noise
  that a single slot cannot, so these beat the operator whenever the
  measurement noise is comparable to the signal, whether or not any dynamics
  exist.  That advantage is not evidence about dynamics; it is evidence about
  whether the state at ``t`` is a sufficient statistic, which is the D5 and
  Markov-order question (R1 §24, §25).  They are reported at D1 as context and
  consumed formally at D5.

This split was found by running the estimator on the synthetic worlds: on
``S-D1`` and ``S-D2``, where the planted process is exactly first-order, the
local-AR baseline beat the true operator by roughly 0.1 in ``R^2`` purely by
denoising, which would have made every signal world look like a null.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_d.scoring import DocumentScore, document_folds, equal_document_r2

BASELINE_NAMES = (
    "persistence",
    "constant_velocity",
    "local_linear",
    "local_ar",
    "update_only",
)

#: Baselines that see exactly the operator's information set (R1 §15).  The
#: zero predictor -- "no remote-history effect propagates" -- is included
#: because it is always available and is the natural origin for a difference.
#: Without it a baseline can score a large NEGATIVE R^2 (persistence does, when
#: nothing propagates), and an incremental statistic measured against that
#: negative number would reward an operator for correctly predicting nothing.
SOURCE_STATE_BASELINES = ("zero", "persistence", "update_only")

#: Baselines that additionally see earlier state history.
TRAJECTORY_BASELINES = ("constant_velocity", "local_linear", "local_ar")


@dataclass(frozen=True)
class BaselineResult:
    name: str
    score: DocumentScore
    detail: dict[str, object]

    @property
    def r2(self) -> float:
        return self.score.r2


def persistence(history: np.ndarray) -> np.ndarray:
    """``pred = Delta z_t`` (the last history slot)."""

    return np.asarray(history, dtype=np.float64)[:, -1, :]


def constant_velocity(history: np.ndarray, lag: int) -> np.ndarray:
    """``pred = Delta z_t + k (Delta z_t - Delta z_{t-1})``."""

    stack = np.asarray(history, dtype=np.float64)
    if stack.shape[1] < 2:
        return stack[:, -1, :]
    velocity = stack[:, -1, :] - stack[:, -2, :]
    return stack[:, -1, :] + float(lag) * velocity


def local_linear(history: np.ndarray, lag: int, offsets: np.ndarray) -> np.ndarray:
    """Per-row least-squares line through the history, evaluated at ``+lag``.

    ``offsets`` are the (negative or zero) positions of the history slots
    relative to the source token.
    """

    stack = np.asarray(history, dtype=np.float64)
    x = np.asarray(offsets, dtype=np.float64).reshape(-1)
    if stack.shape[1] != x.shape[0]:
        raise ValueError("history slots and offsets must agree")
    if x.shape[0] < 2:
        return stack[:, -1, :]
    x_mean = x.mean()
    centered = x - x_mean
    denominator = float(np.sum(centered**2))
    y_mean = stack.mean(axis=1)
    slope = np.einsum("j,ijk->ik", centered, stack) / denominator
    return y_mean + slope * (float(lag) - x_mean)


def fit_local_ar(
    history_train: np.ndarray,
    target_train: np.ndarray,
    document_ids: np.ndarray,
    *,
    ridge_grid: tuple[float, ...] = (1e-4, 1e-2, 1.0, 1e2, 1e4),
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Train-fitted linear map from the flattened history stack to the target."""

    design = np.asarray(history_train, dtype=np.float64).reshape(history_train.shape[0], -1)
    target = np.asarray(target_train, dtype=np.float64)
    folds = document_folds(document_ids, n_folds=min(n_folds, len(np.unique(document_ids))), seed=seed)
    best_ridge, best_score = float(ridge_grid[0]), -np.inf
    for ridge in ridge_grid:
        fold_scores = []
        for held_out in folds:
            mask = np.ones(design.shape[0], dtype=bool)
            mask[held_out] = False
            if mask.sum() <= design.shape[1] // 4 or held_out.size < 1:
                continue
            coefficients = _ridge_solve(design[mask], target[mask], ridge)
            residual = target[held_out] - design[held_out] @ coefficients
            energy = float(np.sum(target[held_out] ** 2))
            if energy > 0:
                fold_scores.append(1.0 - float(np.sum(residual**2)) / energy)
        if fold_scores and float(np.mean(fold_scores)) > best_score:
            best_score, best_ridge = float(np.mean(fold_scores)), float(ridge)
    return _ridge_solve(design, target, best_ridge), best_ridge


def _ridge_solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ target)


def apply_local_ar(history: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.asarray(history, dtype=np.float64).reshape(history.shape[0], -1)
    return design @ coefficients


def fit_update_only(
    features_train: np.ndarray,
    target_train: np.ndarray,
    document_ids: np.ndarray,
    *,
    ridge_grid: tuple[float, ...] = (1e-2, 1.0, 1e2, 1e4),
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """R1 §15.7: predict the future state from local update information alone.

    For the *paired* object the update sequence is identical between the two
    runs by construction, so this predictor should recover approximately
    nothing.  That near-zero is the expected, correct behaviour and is reported
    rather than skipped: it is the direct measurement that the paired design
    removed the update channel.
    """

    return fit_local_ar(
        features_train[:, None, :],
        target_train,
        document_ids,
        ridge_grid=ridge_grid,
        n_folds=n_folds,
        seed=seed,
    )


def evaluate_baselines(
    *,
    history_train: np.ndarray,
    target_train: np.ndarray,
    documents_train: np.ndarray,
    history_eval: np.ndarray,
    target_eval: np.ndarray,
    documents_eval: np.ndarray,
    offsets: np.ndarray,
    lag: int,
    update_features_train: np.ndarray | None = None,
    update_features_eval: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, BaselineResult]:
    """Score every mandatory trajectory baseline on the evaluation rows."""

    results: dict[str, BaselineResult] = {}
    results["zero"] = BaselineResult(
        "zero",
        equal_document_r2(target_eval, np.zeros_like(target_eval), documents_eval),
        {"note": "predicts no remote-history effect; R^2 is zero by construction"},
    )
    results["persistence"] = BaselineResult(
        "persistence",
        equal_document_r2(target_eval, persistence(history_eval), documents_eval),
        {},
    )
    results["constant_velocity"] = BaselineResult(
        "constant_velocity",
        equal_document_r2(target_eval, constant_velocity(history_eval, lag), documents_eval),
        {},
    )
    results["local_linear"] = BaselineResult(
        "local_linear",
        equal_document_r2(target_eval, local_linear(history_eval, lag, offsets), documents_eval),
        {},
    )
    coefficients, ridge = fit_local_ar(history_train, target_train, documents_train, seed=seed)
    results["local_ar"] = BaselineResult(
        "local_ar",
        equal_document_r2(target_eval, apply_local_ar(history_eval, coefficients), documents_eval),
        {"ridge": ridge, "order": int(history_train.shape[1])},
    )
    if update_features_train is None or update_features_eval is None:
        # R1 §15.7 / addition A2.  The local update at a position is a function of
        # the tokens at and before it and of the incoming state.  After the splice
        # boundary the two runs read identical tokens, so the token-driven part of
        # the update is bitwise identical and cancels in the paired difference:
        # the update-only predictor has no non-zero feature to regress on and its
        # score is zero by construction, not by omission.  Reported rather than
        # skipped, because that zero is the direct measurement that the paired
        # design removed the update channel.
        results["update_only"] = BaselineResult(
            "update_only",
            results["zero"].score,
            {
                "structurally_zero": True,
                "reason": (
                    "token-driven local update is identical between the spliced and "
                    "unspliced runs and cancels in the paired difference"
                ),
            },
        )
    if update_features_train is not None and update_features_eval is not None:
        update_coefficients, update_ridge = fit_update_only(
            update_features_train, target_train, documents_train, seed=seed
        )
        prediction = update_features_eval @ update_coefficients
        results["update_only"] = BaselineResult(
            "update_only",
            equal_document_r2(target_eval, prediction, documents_eval),
            {"ridge": update_ridge, "features": int(update_features_train.shape[1])},
        )
    return results


def strongest_baseline(
    results: dict[str, BaselineResult],
    names: tuple[str, ...] = SOURCE_STATE_BASELINES,
) -> BaselineResult:
    """The competitor the D1 headline must beat (R1 §14.2, §15).

    Restricted by default to the baselines that consume the operator's own
    information set.  Pass ``names=TRAJECTORY_BASELINES`` for the richer
    comparison that feeds the Markov-order label.
    """

    applicable = {name: result for name, result in results.items() if name in names}
    if not applicable:
        raise ValueError(f"no applicable baselines among {names}")
    return max(applicable.values(), key=lambda item: item.r2)
