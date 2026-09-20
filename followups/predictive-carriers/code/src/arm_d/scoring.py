"""Equal-document-weight scoring and the document bootstrap (R1 §30, §14).

R1 §30 is categorical: every confirmatory predictive score, including R^2, uses
equal document weight, and documents -- never tokens -- are the inferential
replicates.  A plain pooled R^2 silently weights each document by how many rows
it contributed, which for Arm D varies with position band and splice condition.
Every score in this package routes through :func:`equal_document_r2`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def document_index(document_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map document labels to contiguous integer codes, preserving first order."""

    labels = np.asarray(document_ids).reshape(-1)
    unique, inverse = np.unique(labels, return_inverse=True)
    return unique, inverse.astype(np.int64)


def per_document_sums(
    values: np.ndarray,
    document_codes: np.ndarray,
    n_documents: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-document (sum of squared row norms, row count)."""

    array = _as_2d(values, "values")
    energy = np.sum(array**2, axis=1)
    sums = np.bincount(document_codes, weights=energy, minlength=n_documents)
    counts = np.bincount(document_codes, minlength=n_documents).astype(np.float64)
    return sums, counts


@dataclass(frozen=True)
class DocumentScore:
    """An equal-document-weight R^2 with the per-document parts kept.

    ``numerator_by_document`` and ``denominator_by_document`` are retained so a
    bootstrap can resample documents without refitting anything.
    """

    r2: float
    numerator_by_document: np.ndarray
    denominator_by_document: np.ndarray
    document_labels: np.ndarray

    @property
    def n_documents(self) -> int:
        return int(self.document_labels.shape[0])


def equal_document_r2(
    target: np.ndarray,
    prediction: np.ndarray,
    document_ids: np.ndarray,
) -> DocumentScore:
    r"""Equal-document-weight coefficient of determination.

    .. math::
        R^2 = 1 - \frac{\frac1D\sum_d \frac1{n_d}\sum_{i\in d}\|e_i\|^2}
                       {\frac1D\sum_d \frac1{n_d}\sum_{i\in d}\|y_i\|^2}

    The target is not re-centred: for paired differences the meaningful origin
    is zero (no remote-history effect), so the denominator is the raw energy of
    the difference, not its variance about a fitted mean.
    """

    y = _as_2d(target, "target")
    p = _as_2d(prediction, "prediction")
    if y.shape != p.shape:
        raise ValueError("target and prediction must have the same shape")
    labels, codes = document_index(document_ids)
    if codes.shape[0] != y.shape[0]:
        raise ValueError("document_ids must have one entry per row")
    n_documents = int(labels.shape[0])
    if n_documents < 2:
        raise ValueError("equal-document weighting requires at least two documents")

    residual_sums, counts = per_document_sums(y - p, codes, n_documents)
    target_sums, _ = per_document_sums(y, codes, n_documents)
    if np.any(counts <= 0):
        raise AssertionError("document code without rows")
    numerator = residual_sums / counts
    denominator = target_sums / counts
    total_denominator = float(np.mean(denominator))
    if total_denominator <= 0:
        raise ValueError("target energy is zero; R^2 is undefined")
    r2 = 1.0 - float(np.mean(numerator)) / total_denominator
    return DocumentScore(
        r2=r2,
        numerator_by_document=numerator,
        denominator_by_document=denominator,
        document_labels=labels,
    )


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float
    replicates: np.ndarray

    @property
    def excludes_zero(self) -> bool:
        return bool(self.lower > 0.0 or self.upper < 0.0)

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "point": float(self.point),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "excludes_zero": self.excludes_zero,
            "replicates": int(self.replicates.shape[0]),
        }


def bootstrap_r2_difference(
    left: DocumentScore,
    right: DocumentScore,
    *,
    replicates: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Document bootstrap of ``left.r2 - right.r2`` with shared resamples.

    The two scores must be over the same document set in the same order --
    resampling them jointly is what makes the interval an interval on the
    *difference* rather than on two independently noisy numbers.
    """

    if left.n_documents != right.n_documents or not np.array_equal(
        left.document_labels, right.document_labels
    ):
        raise ValueError("scores must share an identical document set and order")
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("replicates must be positive and confidence must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    n = left.n_documents
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        take = rng.integers(0, n, size=n)
        left_r2 = 1.0 - left.numerator_by_document[take].mean() / left.denominator_by_document[take].mean()
        right_r2 = 1.0 - right.numerator_by_document[take].mean() / right.denominator_by_document[take].mean()
        values[replicate] = left_r2 - right_r2
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        point=float(left.r2 - right.r2),
        lower=float(np.quantile(values, alpha)),
        upper=float(np.quantile(values, 1.0 - alpha)),
        replicates=values,
    )


def bootstrap_scalar_by_document(
    values_by_document: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Document bootstrap of a mean over per-document scalars."""

    array = np.asarray(values_by_document, dtype=np.float64).reshape(-1)
    if array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError("need at least two finite per-document values")
    rng = np.random.default_rng(seed)
    n = array.shape[0]
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        draws[replicate] = array[rng.integers(0, n, size=n)].mean()
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        point=float(array.mean()),
        lower=float(np.quantile(draws, alpha)),
        upper=float(np.quantile(draws, 1.0 - alpha)),
        replicates=draws,
    )


def empirical_p_value(observed: float, null_values: np.ndarray) -> float:
    """One-sided empirical p-value with the conventional +1 correction.

    With ``B`` null replicates the smallest attainable value is ``1/(B+1)``,
    which is reported honestly rather than as zero.
    """

    nulls = np.asarray(null_values, dtype=np.float64).reshape(-1)
    if nulls.size == 0:
        raise ValueError("null bank is empty")
    exceed = int(np.sum(nulls >= observed))
    return float((exceed + 1) / (nulls.size + 1))


def document_folds(document_labels: np.ndarray, n_folds: int, seed: int = 42) -> list[np.ndarray]:
    """Deterministic document-grouped folds returning row-index arrays.

    Token-random cross-validation is forbidden by R1 §8.3; every ridge or rank
    selection in this package uses these folds.
    """

    labels, codes = document_index(document_labels)
    n_documents = int(labels.shape[0])
    if n_folds < 2 or n_folds > n_documents:
        raise ValueError("n_folds must lie in [2, number of documents]")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_documents)
    assignment = np.empty(n_documents, dtype=np.int64)
    assignment[order] = np.arange(n_documents) % n_folds
    return [np.flatnonzero(assignment[codes] == fold) for fold in range(n_folds)]
