from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def cross_entropy_rows(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    return logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]


def decision_logits(model: LogisticRegression, x: np.ndarray, classes: int) -> np.ndarray:
    logits = np.asarray(model.decision_function(x), dtype=np.float64)
    if logits.ndim == 1:
        logits = np.column_stack([-logits / 2, logits / 2])
    if logits.shape[1] != classes:
        raise ValueError("class count changed between fit and evaluation")
    return logits - logits.mean(axis=1, keepdims=True)


def choose_baseline_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray, grid: list[float], folds: int) -> float:
    splitter = GroupKFold(n_splits=folds)
    scores = np.zeros(len(grid), dtype=np.float64)
    for train, val in splitter.split(x, y, groups):
        for i, c in enumerate(grid):
            model = LogisticRegression(C=c, max_iter=500, solver="lbfgs", tol=1e-7)
            model.fit(x[train], y[train])
            scores[i] += cross_entropy_rows(decision_logits(model, x[val], len(np.unique(y))), y[val]).mean()
    return float(grid[int(np.argmin(scores))])


@dataclass
class OffsetFit:
    coefficients: np.ndarray
    l2: float
    success: bool
    iterations: int

    def logits(self, offset: np.ndarray, x: np.ndarray) -> np.ndarray:
        return offset + x @ self.coefficients


def fit_offset_multinomial(offset: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float) -> OffsetFit:
    x = np.asarray(x, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    n, d = x.shape
    classes = offset.shape[1]

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coef = flat.reshape(d, classes)
        coef = coef - coef.mean(axis=1, keepdims=True)
        logits = offset + x @ coef
        log_norm = logsumexp(logits, axis=1)
        loss = float(np.mean(log_norm - logits[np.arange(n), y]) + 0.5 * l2 * np.sum(coef * coef))
        probs = np.exp(logits - log_norm[:, None])
        probs[np.arange(n), y] -= 1.0
        grad = x.T @ probs / n + l2 * coef
        grad -= grad.mean(axis=1, keepdims=True)
        return loss, grad.ravel()

    result = minimize(
        objective, np.zeros(d * classes, dtype=np.float64), method="L-BFGS-B", jac=True,
        options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-7},
    )
    coef = result.x.reshape(d, classes)
    coef -= coef.mean(axis=1, keepdims=True)
    return OffsetFit(coef, l2, bool(result.success), int(result.nit))


def choose_offset_l2(
    nuisance: np.ndarray, representation: np.ndarray, y: np.ndarray, groups: np.ndarray,
    baseline_c: float, grid: list[float], folds: int,
) -> float:
    splitter = GroupKFold(n_splits=folds)
    scores = np.zeros(len(grid), dtype=np.float64)
    classes = len(np.unique(y))
    for train, val in splitter.split(representation, y, groups):
        baseline = LogisticRegression(C=baseline_c, max_iter=500, solver="lbfgs", tol=1e-7)
        baseline.fit(nuisance[train], y[train])
        train_offset = decision_logits(baseline, nuisance[train], classes)
        val_offset = decision_logits(baseline, nuisance[val], classes)
        for i, l2 in enumerate(grid):
            fit = fit_offset_multinomial(train_offset, representation[train], y[train], l2)
            scores[i] += cross_entropy_rows(fit.logits(val_offset, representation[val]), y[val]).mean()
    return float(grid[int(np.argmin(scores))])


def standardized_fit_apply(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    scaler = StandardScaler().fit(train)
    return (scaler.transform(train), *(scaler.transform(x) for x in others))


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
