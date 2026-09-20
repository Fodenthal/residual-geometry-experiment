"""Pure analyses for R1.8 semantic and variance characterization."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import orth
from scipy.optimize import linear_sum_assignment


def variance_components(coordinates: np.ndarray) -> dict[str, float]:
    """Population-moment decomposition for equal-position document arrays."""
    z = np.asarray(coordinates, dtype=np.float64)
    if z.ndim != 3 or z.shape[0] < 2 or z.shape[1] < 2:
        raise ValueError("coordinates must be (documents, positions, dimensions)")
    grand = z.mean(axis=(0, 1), keepdims=True)
    doc_means = z.mean(axis=1)
    total_trace = float(np.mean(np.sum((z - grand) ** 2, axis=2)))
    between_trace = float(np.mean(np.sum((doc_means - doc_means.mean(axis=0)) ** 2, axis=1)))
    within_trace = float(np.mean(np.sum((z - doc_means[:, None, :]) ** 2, axis=2)))
    closure = abs(total_trace - between_trace - within_trace) / max(total_trace, np.finfo(float).tiny)
    return {
        "total_trace": total_trace, "between_trace": between_trace,
        "within_trace": within_trace, "r_between": between_trace / total_trace,
        "r_within": within_trace / total_trace, "relative_closure_error": closure,
    }


def bootstrap_variance(
    coordinates: dict[str, np.ndarray], random_labels: list[str], replicates: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = list(coordinates)
    docs = next(iter(coordinates.values())).shape[0]
    if any(value.shape[0] != docs for value in coordinates.values()):
        raise ValueError("basis coordinate arrays have different document counts")
    rng = np.random.default_rng(seed)
    point = {label: variance_components(value) for label, value in coordinates.items()}
    rows = []
    for replicate in range(replicates):
        draw = rng.integers(0, docs, size=docs)
        values = {label: variance_components(coordinates[label][draw])["r_between"] for label in labels}
        random_median = float(np.median([values[label] for label in random_labels]))
        slow_median = float(np.median([values[label] for label in ("canonical", "A", "B")]))
        for label, value in values.items():
            rows.append({"replicate": replicate, "basis": label, "statistic": "r_between", "value": value})
        rows.append({"replicate": replicate, "basis": "slow", "statistic": "slow_minus_random_median", "value": slow_median-random_median})
    boot = pd.DataFrame(rows)
    summaries = []
    requested = ["canonical", "A", "B", "pca", "slow"]
    for label in requested:
        statistic = "slow_minus_random_median" if label == "slow" else "r_between"
        values = boot.loc[(boot.basis == label) & (boot.statistic == statistic), "value"].to_numpy(float)
        point_value = (
            np.median([point[x]["r_between"] for x in ("canonical", "A", "B")])
            - np.median([point[x]["r_between"] for x in random_labels])
            if label == "slow" else point[label]["r_between"]
        )
        summaries.append({"basis": label, "statistic": statistic, "point": float(point_value),
                          "ci_low": float(np.quantile(values,.025)), "ci_high": float(np.quantile(values,.975))})
    return boot, pd.DataFrame(summaries)


def discriminant_basis(ambient_coefficients: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    matrix = np.asarray(ambient_coefficients, dtype=np.float64)
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular) or singular[0] <= 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    rank = int(np.sum(singular > tolerance * singular[0]))
    return u[:, :rank]


def readout_overlap(left: np.ndarray, right: np.ndarray) -> tuple[float, np.ndarray, int]:
    ql, qr = discriminant_basis(left), discriminant_basis(right)
    rank = min(ql.shape[1], qr.shape[1])
    if rank == 0:
        return 0.0, np.zeros(0), 0
    singular = np.clip(np.linalg.svd(ql.T @ qr, compute_uv=False)[:rank], 0, 1)
    return float(np.mean(singular**2)), singular, rank


def match_axes(left: np.ndarray, right: np.ndarray) -> pd.DataFrame:
    cosine = np.abs(np.asarray(left, float) @ np.asarray(right, float).T)
    rows, cols = linear_sum_assignment(-cosine)
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    return pd.DataFrame({"a_selected_rank": rows+1, "b_selected_rank": cols+1,
                         "matched_abs_cosine": cosine[rows, cols],
                         "rank_displacement": np.abs(rows-cols)})


def local_eigengap(values: np.ndarray, index: int) -> float:
    x = np.asarray(values, float)
    distances = []
    if index > 0: distances.append(abs(x[index]-x[index-1]))
    if index+1 < len(x): distances.append(abs(x[index]-x[index+1]))
    return float(min(distances))


def axis_decision(matches: pd.DataFrame) -> str:
    stable = int((matches.head(5).matched_abs_cosine >= .80).sum())
    return "AXES_INTERPRETABLE" if stable >= 3 else "AXES_NOT_INTERPRETABLE"


def plus_one_p(observed: float, null: np.ndarray) -> float:
    values = np.asarray(null, float)
    return float((1 + np.sum(values >= observed)) / (1 + len(values)))
