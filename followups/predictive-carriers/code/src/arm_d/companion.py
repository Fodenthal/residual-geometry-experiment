"""Companion-form operators at state history order p, and the Markov-order curve.

Why this module exists
----------------------
The R1.1 order-1 trajectory criterion compared an operator that reads a single
difference slot (rank ``r`` inputs, ``r**2`` free parameters in the ``full``
class) against a local autoregressive comparator that reads the whole projected
history stack (``p*r`` inputs, ``p*r**2`` parameters).  On the confirmatory
validation split the comparator won, which the original label read as a local
context explanation.  That reading is wrong on this design: the paired
construction subtracts local context exactly, so every input to the comparator
is itself a difference ``dz`` that is identically zero when the two prefixes
agree.  The comparator is therefore a *higher-order model of the same remote
history state*, and the honest question is whether a fixed law of matched
capacity recovers the same skill.

This module fits the operator family at state history order ``p``, predicting
``dz_{t+k}`` from the stacked stateeee ``[dz_t, ..., dz_{t-p+1}]`` with one
coefficient block per lag slot, and keeps the nested constraint classes so the
mixing test survives at every order:

``gain``       ``A_j = c_j I``                    -> ``p`` parameters
``diagonal``   ``A_j`` diagonal                   -> ``p*r`` parameters
``symmetric``  ``A_j = A_j^T``                    -> ``p*r*(r+1)/2`` parameters
``full``       ``A_j`` unconstrained              -> ``p*r**2`` parameters

At matched ``p`` the ``full`` class and the order-``p`` autoregressive
comparator are the *same model class* on the same design matrix, so their
difference measures ridge and fold noise rather than structure.  That is a
property of the comparison, not a result, and
:func:`companion_criterion` records it explicitly in
``full_class_is_tied_by_construction`` so no reader mistakes a near-zero
difference for evidence.  The substantive content at matched order is whether
the *constrained* classes retain the skill, and the shape of the Markov-order
curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arm_d.scoring import document_folds, equal_document_r2

COMPANION_FAMILY = ("gain", "diagonal", "symmetric", "full")
DEFAULT_RIDGE_GRID = (1e-4, 1e-2, 1.0, 1e2, 1e4)
#: R1.1 amendment 8 freezes the two capacity-matched orders.  They are not searched.
MATCHED_ORDERS = (1, 5)
#: R1 markov-order analysis: the curve is reported at these orders.
MARKOV_ORDERS = (1, 2, 3, 5)


def _stack(history: np.ndarray, order: int) -> np.ndarray:
    """The ``order`` most recent slots, flattened, most recent first.

    ``history`` is ``(n, n_offsets, r)`` with the *last* offset the current slot,
    matching ``HISTORY_OFFSETS = [-4, -3, -2, -1, 0]``.
    """

    array = np.asarray(history, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("history must be (rows, offsets, rank)")
    if order < 1 or order > array.shape[1]:
        raise ValueError(f"order {order} outside available offsets {array.shape[1]}")
    recent = array[:, array.shape[1] - order :, :][:, ::-1, :]
    return recent.reshape(array.shape[0], -1)


def parameter_count(family: str, order: int, rank: int) -> int:
    """Free parameters, so a matched comparison can be verified rather than asserted."""

    if family == "gain":
        return int(order)
    if family == "diagonal":
        return int(order * rank)
    if family == "symmetric":
        return int(order * rank * (rank + 1) // 2)
    if family == "full":
        return int(order * rank * rank)
    raise ValueError(f"unknown family {family}")


def _solve_full(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ target)


def _predict_full(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return design @ weights


def _solve_diagonal(design: np.ndarray, target: np.ndarray, ridge: float, order: int, rank: int) -> np.ndarray:
    """One independent ``order``-parameter regression per output coordinate."""

    weights = np.zeros((order, rank), dtype=np.float64)
    for m in range(rank):
        columns = design[:, [j * rank + m for j in range(order)]]
        gram = columns.T @ columns + ridge * np.eye(order)
        weights[:, m] = np.linalg.solve(gram, columns.T @ target[:, m])
    return weights


def _predict_diagonal(design: np.ndarray, weights: np.ndarray, order: int, rank: int) -> np.ndarray:
    prediction = np.zeros((design.shape[0], rank), dtype=np.float64)
    for j in range(order):
        prediction += design[:, j * rank : (j + 1) * rank] * weights[j][None, :]
    return prediction


def _solve_gain(design: np.ndarray, target: np.ndarray, ridge: float, order: int, rank: int) -> np.ndarray:
    """``order`` scalars: ``dz_{t+k} = sum_j c_j dz_{t-j+1}``."""

    blocks = [design[:, j * rank : (j + 1) * rank] for j in range(order)]
    gram = np.zeros((order, order), dtype=np.float64)
    rhs = np.zeros(order, dtype=np.float64)
    for a in range(order):
        rhs[a] = float(np.sum(blocks[a] * target))
        for b in range(order):
            gram[a, b] = float(np.sum(blocks[a] * blocks[b]))
    gram += ridge * np.eye(order)
    return np.linalg.solve(gram, rhs)


def _predict_gain(design: np.ndarray, weights: np.ndarray, order: int, rank: int) -> np.ndarray:
    prediction = np.zeros((design.shape[0], rank), dtype=np.float64)
    for j in range(order):
        prediction += weights[j] * design[:, j * rank : (j + 1) * rank]
    return prediction


def _symmetric_basis(rank: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(rank) for b in range(a, rank)]


def _solve_symmetric(design: np.ndarray, target: np.ndarray, ridge: float, order: int, rank: int) -> np.ndarray:
    """Least squares over symmetric blocks, accumulated one output coordinate at a time.

    Parameter ``(j, a, b)`` with ``a <= b`` contributes ``x_{j,b}`` to output
    ``a`` and ``x_{j,a}`` to output ``b``, which is what makes ``A_j`` symmetric.
    """

    pairs = _symmetric_basis(rank)
    n_params = order * len(pairs)
    gram = np.zeros((n_params, n_params), dtype=np.float64)
    rhs = np.zeros(n_params, dtype=np.float64)
    rows = design.shape[0]
    for m in range(rank):
        block = np.zeros((rows, n_params), dtype=np.float64)
        for j in range(order):
            columns = design[:, j * rank : (j + 1) * rank]
            for index, (a, b) in enumerate(pairs):
                slot = j * len(pairs) + index
                if a == m:
                    block[:, slot] += columns[:, b]
                if b == m and a != b:
                    block[:, slot] += columns[:, a]
        gram += block.T @ block
        rhs += block.T @ target[:, m]
    gram += ridge * np.eye(n_params)
    return np.linalg.solve(gram, rhs)


def _predict_symmetric(design: np.ndarray, theta: np.ndarray, order: int, rank: int) -> np.ndarray:
    pairs = _symmetric_basis(rank)
    prediction = np.zeros((design.shape[0], rank), dtype=np.float64)
    for j in range(order):
        columns = design[:, j * rank : (j + 1) * rank]
        matrix = np.zeros((rank, rank), dtype=np.float64)
        for index, (a, b) in enumerate(pairs):
            value = theta[j * len(pairs) + index]
            matrix[a, b] = value
            matrix[b, a] = value
        prediction += columns @ matrix.T
    return prediction


def _fit_family(
    family: str, design: np.ndarray, target: np.ndarray, ridge: float, order: int, rank: int
):
    if family == "full":
        weights = _solve_full(design, target, ridge)
        return weights, lambda d: _predict_full(d, weights)
    if family == "diagonal":
        weights = _solve_diagonal(design, target, ridge, order, rank)
        return weights, lambda d: _predict_diagonal(d, weights, order, rank)
    if family == "gain":
        weights = _solve_gain(design, target, ridge, order, rank)
        return weights, lambda d: _predict_gain(d, weights, order, rank)
    if family == "symmetric":
        theta = _solve_symmetric(design, target, ridge, order, rank)
        return theta, lambda d: _predict_symmetric(d, theta, order, rank)
    raise ValueError(f"unknown family {family}")


def _select_ridge(
    family: str,
    design: np.ndarray,
    target: np.ndarray,
    documents: np.ndarray,
    order: int,
    rank: int,
    ridge_grid: tuple[float, ...],
    n_folds: int,
    seed: int,
) -> float:
    unique = len(np.unique(documents))
    folds = document_folds(documents, n_folds=min(n_folds, max(unique, 2)), seed=seed)
    best_ridge, best_score = float(ridge_grid[0]), -np.inf
    for ridge in ridge_grid:
        scores: list[float] = []
        for held_out in folds:
            mask = np.ones(design.shape[0], dtype=bool)
            mask[held_out] = False
            if mask.sum() <= design.shape[1] // 4 or held_out.size < 1:
                continue
            try:
                _, predict = _fit_family(family, design[mask], target[mask], ridge, order, rank)
            except np.linalg.LinAlgError:
                continue
            residual = target[held_out] - predict(design[held_out])
            energy = float(np.sum(target[held_out] ** 2))
            if energy > 0:
                scores.append(1.0 - float(np.sum(residual**2)) / energy)
        if scores and float(np.mean(scores)) > best_score:
            best_score, best_ridge = float(np.mean(scores)), float(ridge)
    return best_ridge


@dataclass
class CompanionFit:
    """One (family, order) cell, scored with equal document weight."""

    family: str
    order: int
    rank: int
    r2: float
    ridge: float
    parameters: int


@dataclass
class CompanionResult:
    order: int
    rank: int
    families: dict[str, CompanionFit] = field(default_factory=dict)
    comparator_r2: float = float("nan")
    comparator_parameters: int = 0

    @property
    def best_family_r2(self) -> float:
        values = [fit.r2 for fit in self.families.values() if np.isfinite(fit.r2)]
        return max(values) if values else float("nan")

    def to_dict(self) -> dict:
        return {
            "order": int(self.order),
            "rank": int(self.rank),
            "comparator_r2": float(self.comparator_r2),
            "comparator_parameters": int(self.comparator_parameters),
            "best_family_r2": float(self.best_family_r2),
            "families": {
                name: {
                    "r2": float(fit.r2),
                    "ridge": float(fit.ridge),
                    "parameters": int(fit.parameters),
                }
                for name, fit in self.families.items()
            },
        }


def fit_companion_family(
    history_train: np.ndarray,
    target_train: np.ndarray,
    documents_train: np.ndarray,
    history_eval: np.ndarray,
    target_eval: np.ndarray,
    documents_eval: np.ndarray,
    *,
    order: int,
    families: tuple[str, ...] = COMPANION_FAMILY,
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
    n_folds: int = 5,
    seed: int = 42,
) -> CompanionResult:
    """Fit every nested family at state history ``order`` and score held out.

    The comparator is the unconstrained map on the *same* design, which is what
    makes the capacity match exact: it has the same input count and the same
    parameter count as the ``full`` class.
    """

    design_train = _stack(history_train, order)
    design_eval = _stack(history_eval, order)
    target_train = np.asarray(target_train, dtype=np.float64)
    target_eval = np.asarray(target_eval, dtype=np.float64)
    rank = target_train.shape[1]

    result = CompanionResult(order=int(order), rank=int(rank))
    for family in families:
        ridge = _select_ridge(
            family, design_train, target_train, documents_train, order, rank, ridge_grid, n_folds, seed
        )
        try:
            _, predict = _fit_family(family, design_train, target_train, ridge, order, rank)
        except np.linalg.LinAlgError:
            result.families[family] = CompanionFit(
                family, order, rank, float("nan"), ridge, parameter_count(family, order, rank)
            )
            continue
        r2 = equal_document_r2(target_eval, predict(design_eval), documents_eval).r2
        result.families[family] = CompanionFit(
            family, order, rank, float(r2), float(ridge), parameter_count(family, order, rank)
        )

    ridge = _select_ridge(
        "full", design_train, target_train, documents_train, order, rank, ridge_grid, n_folds, seed
    )
    weights = _solve_full(design_train, target_train, ridge)
    result.comparator_r2 = float(
        equal_document_r2(target_eval, design_eval @ weights, documents_eval).r2
    )
    result.comparator_parameters = parameter_count("full", order, rank)
    return result


def companion_criterion(
    history_train: np.ndarray,
    target_train: np.ndarray,
    documents_train: np.ndarray,
    history_eval: np.ndarray,
    target_eval: np.ndarray,
    documents_eval: np.ndarray,
    *,
    orders: tuple[int, ...] = MATCHED_ORDERS,
    seed: int = 42,
) -> dict:
    """R1.1 amendment 8: the capacity-matched trajectory criterion at frozen orders.

    Reports, per order, the operator family skill and the matched-capacity
    comparator, plus the constrained-class margins that carry the substantive
    content.  ``full_class_is_tied_by_construction`` is always true and is
    stated so a near-zero ``full`` margin is not read as a finding.
    """

    per_order: dict[str, dict] = {}
    for order in orders:
        fit = fit_companion_family(
            history_train,
            target_train,
            documents_train,
            history_eval,
            target_eval,
            documents_eval,
            order=order,
            seed=seed,
        )
        payload = fit.to_dict()
        comparator = fit.comparator_r2
        payload["margins_over_matched_comparator"] = {
            name: (float(entry["r2"] - comparator) if np.isfinite(entry["r2"]) else float("nan"))
            for name, entry in payload["families"].items()
        }
        constrained = {
            name: payload["margins_over_matched_comparator"][name]
            for name in ("gain", "diagonal", "symmetric")
            if name in payload["margins_over_matched_comparator"]
        }
        finite = [value for value in constrained.values() if np.isfinite(value)]
        payload["best_constrained_margin"] = max(finite) if finite else float("nan")
        payload["constrained_class_retains_skill"] = bool(finite and max(finite) >= 0.0)
        per_order[str(order)] = payload

    return {
        "orders": [int(order) for order in orders],
        "orders_frozen": True,
        "per_order": per_order,
        "full_class_is_tied_by_construction": True,
        "tie_note": (
            "at matched order the unconstrained companion operator and the "
            "order-p autoregressive comparator are the same model class on the "
            "same design matrix, so the full-class margin measures ridge and "
            "fold noise; the constrained classes carry the structural content"
        ),
    }


def markov_order_curve(
    history_train: np.ndarray,
    target_train: np.ndarray,
    documents_train: np.ndarray,
    history_eval: np.ndarray,
    target_eval: np.ndarray,
    documents_eval: np.ndarray,
    *,
    orders: tuple[int, ...] = MARKOV_ORDERS,
    saturation_tolerance: float = 0.01,
    seed: int = 42,
) -> dict:
    """R1 markov-order analysis: skill as a function of state history order.

    Returns the skill curve, the increment from each additional order, whether
    the increments are decaying, and the order at which skill saturates within
    ``saturation_tolerance``.  The label is emitted from the curve rather than
    from a single comparison.
    """

    ordered = sorted(int(order) for order in orders)
    skill: dict[str, float] = {}
    for order in ordered:
        fit = fit_companion_family(
            history_train,
            target_train,
            documents_train,
            history_eval,
            target_eval,
            documents_eval,
            order=order,
            families=("full",),
            seed=seed,
        )
        skill[str(order)] = float(fit.families["full"].r2)

    increments: dict[str, float] = {}
    for previous, current in zip(ordered, ordered[1:]):
        a, b = skill[str(previous)], skill[str(current)]
        increments[f"{previous}->{current}"] = (
            float(b - a) if np.isfinite(a) and np.isfinite(b) else float("nan")
        )

    finite_increments = [value for value in increments.values() if np.isfinite(value)]
    decaying = bool(
        len(finite_increments) >= 2
        and all(
            later <= earlier + 1e-12
            for earlier, later in zip(finite_increments, finite_increments[1:])
        )
    )

    saturation_order: int | None = None
    best = max((value for value in skill.values() if np.isfinite(value)), default=float("nan"))
    if np.isfinite(best):
        for order in ordered:
            value = skill[str(order)]
            if np.isfinite(value) and value >= best - saturation_tolerance:
                saturation_order = int(order)
                break

    if not np.isfinite(best):
        label = "MARKOV_ORDER_UNAVAILABLE"
    elif saturation_order == 1:
        label = "FIRST_ORDER_SUFFICIENT"
    elif saturation_order is not None and saturation_order <= 3:
        label = f"LOW_ORDER_SUFFICIENT_P{saturation_order}"
    else:
        label = "HIGH_ORDER_REQUIRED"

    return {
        "orders": ordered,
        "skill_by_order": skill,
        "increment_by_order": increments,
        "increments_decaying": decaying,
        "saturation_tolerance": float(saturation_tolerance),
        "saturation_order": saturation_order,
        "markov_order_label": label,
        "note": (
            "skill is the held-out equal-document-weight R^2 of the "
            "unconstrained companion operator at each state history order; the "
            "curve, not any single order, carries the interpretation"
        ),
    }


def first_order_skill_fraction(order_1_r2: float, order_p_r2: float) -> dict:
    """The interpretable framing of the original order-1 comparison.

    The fraction of the higher-order model's skill that the order-1 operator
    recovers.  This is a quantitative statement about approximate first-order
    state; it is deliberately not a pass criterion.
    """

    a, b = float(order_1_r2), float(order_p_r2)
    fraction = a / b if np.isfinite(a) and np.isfinite(b) and b != 0.0 else float("nan")
    return {
        "order_1_r2": a,
        "higher_order_r2": b,
        "skill_fraction_recovered": float(fraction),
        "is_pass_criterion": False,
        "note": (
            "the fraction of the higher-order model's skill recovered by the "
            "order-1 fixed law; reported as a statement about approximate "
            "first-order state, not as a threshold test"
        ),
    }


def degenerate_baseline_report(baseline_r2: dict, *, lag: int, threshold: float = -1.0) -> dict:
    """R1.1: unfitted extrapolators diverge at long lags and constrain nothing.

    ``constant_velocity`` and ``local_linear`` extrapolate without being fitted,
    so over tens of tokens they diverge and return large negative R^2.  Clearing
    them is not evidence of anything, and a maximum-over-baselines statistic is
    decided entirely by the fitted comparator.
    """

    unfitted = ("constant_velocity", "local_linear")
    values = {
        name: float(baseline_r2.get(name, float("nan")))
        for name in unfitted
        if name in baseline_r2
    }
    degenerate = {
        name: bool(np.isfinite(value) and value < threshold) for name, value in values.items()
    }
    fitted = float(baseline_r2.get("local_ar", float("nan")))
    return {
        "lag": int(lag),
        "unfitted_extrapolator_r2": values,
        "is_degenerate": degenerate,
        "all_unfitted_degenerate": bool(values) and all(degenerate.values()),
        "fitted_comparator_r2": fitted,
        "criterion_decided_by": "local_ar" if np.isfinite(fitted) else "none",
        "threshold": float(threshold),
        "note": (
            "unfitted extrapolation is inappropriate at this lag: these "
            "baselines diverge rather than competing, so they are inert in a "
            "maximum-over-baselines statistic and clearing them is not evidence"
        ),
    }
