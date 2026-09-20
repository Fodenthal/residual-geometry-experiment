"""The nested operator family, whitened screening, and the shared-space gate.

Three R1 sections live here.

* §11.1 asymmetric VAMP screening, reusing the researcher's frozen Stage 06C
  primitives (``fit_pca_whitener``, ``resolve_whitening_ridge``) so the ridge
  doubling schedule and the source/destination separation are identical.
* §11.2 the nested operator family -- gain, diagonal, symmetric, full.  This is
  R1 addition A2: without it, a process in which each fixed coordinate decays
  independently reads as a "transition law" even though nothing mixes.
* §12 the shared-space gate.  R0's ``orth[U_r, V_r]`` is forbidden; the gated
  construction takes the leading eigenspace of ``(P_U + P_V) / 2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.utils.extmath import randomized_svd

from src.persistent_state.live_state.temporal_code import (
    principal_angles,
    resolve_whitening_ridge,
)

from arm_d.scoring import DocumentScore, document_folds, equal_document_r2

OPERATOR_FAMILY = ("gain", "diagonal", "symmetric", "full")
DEFAULT_RIDGE_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _covariance(values: np.ndarray) -> np.ndarray:
    return (values.T @ values) / float(values.shape[0])


# --------------------------------------------------------------------------
# §11.2 nested operator family
# --------------------------------------------------------------------------


def fit_gain(source: np.ndarray, destination: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """A = rho I with a single fitted scalar (least squares in rho)."""

    x, y = _matrix(source, "source"), _matrix(destination, "destination")
    denominator = float(np.sum(x * x)) + ridge
    if denominator <= 0:
        return np.zeros((x.shape[1], x.shape[1]))
    rho = float(np.sum(x * y)) / denominator
    return rho * np.eye(x.shape[1])


def fit_diagonal(source: np.ndarray, destination: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """A = diag(a) -- independent per-coordinate decay, no mixing."""

    x, y = _matrix(source, "source"), _matrix(destination, "destination")
    denominator = np.sum(x * x, axis=0) + ridge
    numerator = np.sum(x * y, axis=0)
    coefficients = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    return np.diag(coefficients)


def fit_full(source: np.ndarray, destination: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Ridge-regularized unconstrained operator, ``y ~ A x``."""

    x, y = _matrix(source, "source"), _matrix(destination, "destination")
    rank = x.shape[1]
    gram = x.T @ x + ridge * np.eye(rank)
    cross = x.T @ y
    return np.linalg.solve(gram, cross).T


def fit_symmetric(source: np.ndarray, destination: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Ridge-regularized operator constrained to ``A = A^T``.

    Solved exactly in the symmetric parameterization rather than by
    symmetrizing the full solution: ``(1/2)(A + A^T)`` of a least-squares fit is
    not the least-squares fit under the symmetry constraint, and using it would
    make the nesting ``full >= symmetric`` false by construction.
    The normal equations for ``min_S ||Y - X S||_F^2 + lam ||S||_F^2`` with
    ``S`` symmetric are ``G S + S G + 2 lam S = C + C^T`` where ``G = X^T X``
    and ``C = X^T Y``; this is a Sylvester/Lyapunov system solved in the
    eigenbasis of ``G``.
    """

    x, y = _matrix(source, "source"), _matrix(destination, "destination")
    gram = x.T @ x
    cross = x.T @ y
    rhs = cross + cross.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    transformed = vectors.T @ rhs @ vectors
    denominator = eigenvalues[:, None] + eigenvalues[None, :] + 2.0 * ridge
    if np.any(denominator <= 0):
        denominator = np.where(denominator <= 0, np.finfo(float).eps, denominator)
    solution = transformed / denominator
    return vectors @ solution @ vectors.T


_FITTERS = {
    "gain": fit_gain,
    "diagonal": fit_diagonal,
    "symmetric": fit_symmetric,
    "full": fit_full,
}


@dataclass(frozen=True)
class OperatorFit:
    family: str
    operator: np.ndarray
    ridge: float
    train_r2: float
    score: DocumentScore | None = None

    @property
    def r2(self) -> float:
        return self.score.r2 if self.score is not None else float("nan")


@dataclass(frozen=True)
class OperatorFamilyFit:
    fits: dict[str, OperatorFit]
    lag: int
    rank: int
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __getitem__(self, family: str) -> OperatorFit:
        return self.fits[family]

    def best(self) -> OperatorFit:
        return max(self.fits.values(), key=lambda fit: fit.r2)


def select_ridge(
    source: np.ndarray,
    destination: np.ndarray,
    document_ids: np.ndarray,
    family: str,
    *,
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
    n_folds: int = 5,
    seed: int = 42,
) -> float:
    """Document-grouped cross-validated ridge for one operator family."""

    folds = document_folds(document_ids, n_folds=min(n_folds, len(np.unique(document_ids))), seed=seed)
    fitter = _FITTERS[family]
    scores = np.full(len(ridge_grid), -np.inf)
    for index, ridge in enumerate(ridge_grid):
        fold_scores: list[float] = []
        for held_out in folds:
            mask = np.ones(source.shape[0], dtype=bool)
            mask[held_out] = False
            if mask.sum() < 2 or held_out.size < 1:
                continue
            operator = fitter(source[mask], destination[mask], ridge)
            residual = destination[held_out] - source[held_out] @ operator.T
            energy = float(np.sum(destination[held_out] ** 2))
            if energy <= 0:
                continue
            fold_scores.append(1.0 - float(np.sum(residual**2)) / energy)
        if fold_scores:
            scores[index] = float(np.mean(fold_scores))
    return float(ridge_grid[int(np.argmax(scores))])


def fit_operator_family(
    source: np.ndarray,
    destination: np.ndarray,
    document_ids: np.ndarray,
    *,
    lag: int,
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
    n_folds: int = 5,
    seed: int = 42,
    families: tuple[str, ...] = OPERATOR_FAMILY,
) -> OperatorFamilyFit:
    """Fit every nested family on the same rows with per-family CV ridge."""

    x, y = _matrix(source, "source"), _matrix(destination, "destination")
    if x.shape != y.shape:
        raise ValueError("source and destination must have the same shape")
    fits: dict[str, OperatorFit] = {}
    for family in families:
        ridge = select_ridge(
            x, y, document_ids, family, ridge_grid=ridge_grid, n_folds=n_folds, seed=seed
        )
        operator = _FITTERS[family](x, y, ridge)
        prediction = x @ operator.T
        train_energy = float(np.sum(y**2))
        train_r2 = 1.0 - float(np.sum((y - prediction) ** 2)) / train_energy if train_energy > 0 else float("nan")
        fits[family] = OperatorFit(family=family, operator=operator, ridge=ridge, train_r2=train_r2)
    return OperatorFamilyFit(fits=fits, lag=int(lag), rank=int(x.shape[1]))


def score_operator(
    operator: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    document_ids: np.ndarray,
) -> DocumentScore:
    """Held-out equal-document-weight R^2 for a frozen operator."""

    prediction = _matrix(source, "source") @ np.asarray(operator, dtype=np.float64).T
    return equal_document_r2(destination, prediction, document_ids)


def score_family(
    family_fit: OperatorFamilyFit,
    source: np.ndarray,
    destination: np.ndarray,
    document_ids: np.ndarray,
) -> OperatorFamilyFit:
    """Attach held-out scores to every fit in a family."""

    scored = {
        name: OperatorFit(
            family=fit.family,
            operator=fit.operator,
            ridge=fit.ridge,
            train_r2=fit.train_r2,
            score=score_operator(fit.operator, source, destination, document_ids),
        )
        for name, fit in family_fit.fits.items()
    }
    return OperatorFamilyFit(
        fits=scored,
        lag=family_fit.lag,
        rank=family_fit.rank,
        diagnostics=dict(family_fit.diagnostics),
    )


def mixing_evidence(
    family_fit: OperatorFamilyFit,
    *,
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """R1 §11.2: does the full operator beat independent per-coordinate decay?

    Returns the bootstrap interval on ``R^2(full) - R^2(diagonal)`` and the
    verdict that gates every mixing statement in the report.

    The scores passed in must be HELD-OUT scores.  In sample the full operator
    has more free parameters and always wins by a small positive margin, which
    a bootstrap over documents will happily certify as ``mixing_supported``.
    ``run_cell`` fits on train rows and scores on evaluation rows for exactly
    this reason.
    """

    from arm_d.scoring import bootstrap_r2_difference

    full = family_fit["full"].score
    diagonal = family_fit["diagonal"].score
    if full is None or diagonal is None:
        raise ValueError("mixing evidence requires scored fits")
    interval = bootstrap_r2_difference(full, diagonal, replicates=replicates, seed=seed)
    off_diagonal = family_fit["full"].operator.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    total_norm = float(np.linalg.norm(family_fit["full"].operator, ord="fro"))
    return {
        "lag": family_fit.lag,
        "r2_full": float(full.r2),
        "r2_diagonal": float(diagonal.r2),
        "r2_symmetric": float(family_fit["symmetric"].r2),
        "r2_gain": float(family_fit["gain"].r2),
        "full_minus_diagonal": interval.to_dict(),
        "mixing_supported": bool(interval.lower > 0.0),
        "off_diagonal_frobenius_fraction": (
            float(np.linalg.norm(off_diagonal, ord="fro") / total_norm) if total_norm > 0 else 0.0
        ),
    }


# --------------------------------------------------------------------------
# §11.1 whitened asymmetric screening
# --------------------------------------------------------------------------


def whitening_dimension(aperture: int, rank: int, n_rows: int) -> int:
    """Train-fitted PCA width used before whitening (R1 §11.1).

    Whitening a ``p``-dimensional covariance estimated from ``n`` rows inverts
    the estimator's own noise when ``p`` approaches ``n``: on the synthetic
    fixed-coordinate worlds the unreduced kernel returned singular values
    saturated near 0.93 and source/destination spaces whose overlap sat *below*
    the chance level ``r/p``.  Reducing to the leading ``4r`` principal
    directions of the pooled paired difference first -- the same
    ``whitening_dimension`` idiom Stage 06C uses -- restores it.  The factor
    four also fixes the chance overlap at ``r/(4r) = 0.25`` independently of
    rank, so one frozen gate threshold applies at every rank.
    """

    ceiling = max(int(n_rows) // 4, int(rank) + 1)
    return int(max(min(int(aperture), 4 * int(rank), ceiling), int(rank) + 1))


def gate_width_is_calibrated(width: int, rank: int) -> bool:
    """Whether ``MSC_CHANCE_LEVEL`` actually applies to this cell.

    The 0.25 chance level holds only at the full ``4r`` reduction width.  When a
    cell has too few rows the width is capped at ``n_rows // 4``, chance rises to
    ``r / width`` -- 0.53 at rank 16 with 120 rows, 0.97 at rank 32 -- and a
    frozen threshold calibrated at 0.25 would pass at chance and award
    ``FIXED_COORDINATE_DYNAMICS``, unlocking eigenvalue and frequency language,
    to a cell with no shared structure.  The gate is refused instead.
    """

    return int(width) >= 4 * int(rank)


MSC_CHANCE_LEVEL = 0.25


@dataclass(frozen=True)
class ScreeningFit:
    source_space: np.ndarray
    destination_space: np.ndarray
    singular_values: np.ndarray
    whitening_dimension: int
    source_epsilon: float
    destination_epsilon: float
    source_condition: float
    destination_condition: float
    source_healthy: bool
    destination_healthy: bool

    @property
    def healthy(self) -> bool:
        return self.source_healthy and self.destination_healthy

    @property
    def whitening_asymmetry(self) -> float:
        """R1 §10 ``H_eps = |log(eps_src / eps_dst)|``."""

        if self.source_epsilon <= 0 or self.destination_epsilon <= 0:
            return float("inf")
        return float(abs(np.log(self.source_epsilon / self.destination_epsilon)))

    def health_report(self) -> dict[str, float | bool]:
        return {
            "epsilon_source_scale": float(self.source_epsilon),
            "epsilon_destination_scale": float(self.destination_epsilon),
            "source_condition_number": float(self.source_condition),
            "destination_condition_number": float(self.destination_condition),
            "source_healthy": bool(self.source_healthy),
            "destination_healthy": bool(self.destination_healthy),
            "whitening_asymmetry_h_eps": self.whitening_asymmetry,
            "whitening_dimension": int(self.whitening_dimension),
        }


def pooled_reducer(source: np.ndarray, destination: np.ndarray, width: int) -> np.ndarray:
    """Train-fitted PCA basis of the pooled source/destination rows.

    Computed once per (aperture, lag, suffix length) and sliced by every rank in
    the grid: the basis itself does not depend on rank, only the width taken
    from it does, and recomputing it per rank made the null bank six times more
    expensive than it needed to be.  Randomized SVD because only the leading
    ``width`` directions are ever used.
    """

    pooled = np.concatenate([_matrix(source, "source"), _matrix(destination, "destination")], axis=0)
    pooled = pooled - pooled.mean(axis=0, keepdims=True)
    width = int(min(width, pooled.shape[1], pooled.shape[0] - 1))
    if width >= min(pooled.shape) - 1:
        _, _, right = np.linalg.svd(pooled, full_matrices=False)
        return right[:width].T
    _, _, right = randomized_svd(pooled, n_components=width, n_iter=4, random_state=0)
    return right.T


def fit_screening_operator(
    source: np.ndarray,
    destination: np.ndarray,
    rank: int,
    *,
    reduction: int | None = None,
    reducer: np.ndarray | None = None,
) -> ScreeningFit:
    r"""Whitened asymmetric kernel ``K = C00^{-1/2} C01 C11^{-1/2}`` (R1 §11.1).

    Computed inside a train-fitted PCA reduction shared by both roles (see
    :func:`whitening_dimension`).  Source and destination ridges are then
    resolved independently through the frozen doubling schedule; their
    imbalance is reported as ``H_eps`` rather than quietly equalized.
    """

    x, y = _matrix(source, "source"), _matrix(destination, "destination")
    if x.shape != y.shape:
        raise ValueError("source and destination must have the same shape")
    rank = int(min(rank, x.shape[1]))
    width = int(reduction or whitening_dimension(x.shape[1], rank, x.shape[0]))
    width = int(min(width, x.shape[1]))
    if float(np.sum(x * x)) <= 0.0 or float(np.sum(y * y)) <= 0.0:
        # A condition whose paired difference is identically zero -- the exact
        # no-op splice -- has no covariance to whiten.  That is the correct
        # measurement (the floor is exactly zero), not an error, so it is
        # reported as an unhealthy fit and the caller drops the cell.
        empty = np.zeros((x.shape[1], 0))
        return ScreeningFit(
            source_space=empty,
            destination_space=empty,
            singular_values=np.zeros(0),
            whitening_dimension=0,
            source_epsilon=0.0,
            destination_epsilon=0.0,
            source_condition=float("inf"),
            destination_condition=float("inf"),
            source_healthy=False,
            destination_healthy=False,
        )
    if reducer is None:
        reducer = pooled_reducer(x, y, width)
    reducer = np.asarray(reducer, dtype=np.float64)[:, :width]
    width = int(reducer.shape[1])
    x, y = x @ reducer, y @ reducer
    source_cov = _covariance(x)
    destination_cov = _covariance(y)
    source_health = resolve_whitening_ridge(source_cov)
    destination_health = resolve_whitening_ridge(destination_cov)

    def inverse_sqrt(covariance: np.ndarray, epsilon: float) -> np.ndarray:
        values, vectors = np.linalg.eigh(covariance + epsilon * np.eye(covariance.shape[0]))
        values = np.clip(values, np.finfo(float).tiny, None)
        return (vectors * values**-0.5) @ vectors.T

    if not (source_health.healthy and destination_health.healthy):
        empty = np.zeros((reducer.shape[0], 0))
        return ScreeningFit(
            source_space=empty,
            destination_space=empty,
            singular_values=np.zeros(0),
            whitening_dimension=width,
            source_epsilon=source_health.epsilon,
            destination_epsilon=destination_health.epsilon,
            source_condition=source_health.condition_number,
            destination_condition=destination_health.condition_number,
            source_healthy=False,
            destination_healthy=False,
        )

    source_inverse = inverse_sqrt(source_cov, source_health.epsilon)
    destination_inverse = inverse_sqrt(destination_cov, destination_health.epsilon)
    cross = (x.T @ y) / float(x.shape[0])
    kernel = source_inverse @ cross @ destination_inverse
    left, singular_values, right_t = np.linalg.svd(kernel, full_matrices=False)

    # In-sample singular values of a whitened kernel are not a usable ordering:
    # whitening rescales every direction to unit variance, so a direction
    # carrying almost no variance can show a near-unit canonical correlation
    # that is entirely estimation noise.  On the synthetic worlds this produced
    # singular values saturated near 0.93 in every direction and a recovered
    # subspace no closer to the planted one than chance.  Re-order the
    # canonical directions by their HELD-OUT correlation on a document-disjoint
    # half of the training rows and keep the top ``rank``.
    order = _held_out_canonical_order(x, y, left, right_t.T, rank)
    left = left[:, order]
    right = right_t.T[:, order]
    singular_values = singular_values[order]
    source_space = np.linalg.qr(reducer @ source_inverse @ left[:, :rank])[0]
    destination_space = np.linalg.qr(reducer @ destination_inverse @ right[:, :rank])[0]
    return ScreeningFit(
        source_space=source_space,
        destination_space=destination_space,
        singular_values=singular_values[:rank],
        whitening_dimension=width,
        source_epsilon=source_health.epsilon,
        destination_epsilon=destination_health.epsilon,
        source_condition=source_health.condition_number,
        destination_condition=destination_health.condition_number,
        source_healthy=True,
        destination_healthy=True,
    )


# --------------------------------------------------------------------------
# §12 shared-space gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedSpace:
    basis: np.ndarray
    mean_subspace_correlation: float
    principal_angles_radians: np.ndarray
    union_rank: int
    gate_passed: bool
    label: str
    threshold: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_subspace_correlation": float(self.mean_subspace_correlation),
            "principal_angles_degrees": [
                float(np.degrees(angle)) for angle in self.principal_angles_radians
            ],
            "union_rank": int(self.union_rank),
            "rank": int(self.basis.shape[1]),
            "gate_passed": bool(self.gate_passed),
            "shared_space_label": self.label,
            "gate_threshold_msc": float(self.threshold),
        }


def mean_subspace_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """``MSC(U, V) = (1/r) ||U^T V||_F^2`` for orthonormal column bases."""

    u, v = _matrix(left, "left"), _matrix(right, "right")
    if u.shape[1] == 0 or v.shape[1] == 0:
        return 0.0
    rank = min(u.shape[1], v.shape[1])
    return float(np.sum((u.T @ v) ** 2) / rank)


def build_shared_space(
    source_space: np.ndarray,
    destination_space: np.ndarray,
    *,
    rank: int,
    msc_threshold: float,
) -> SharedSpace:
    """R1 §12.  Report overlap first; build the shared support only if gated.

    The orthogonal union ``orth[U, V]`` is never constructed.  When the gate
    fails, the returned basis is the source space, the label is
    ``SOURCE_DESTINATION_TRANSPORT``, and downstream code must refuse to report
    eigenvalues, decay times, or frequencies.
    """

    u = _matrix(source_space, "source_space")
    v = _matrix(destination_space, "destination_space")
    if u.shape[0] != v.shape[0]:
        raise ValueError("source and destination spaces need a common ambient width")
    msc = mean_subspace_correlation(u, v)
    angles = principal_angles(u, v)
    union_rank = int(np.linalg.matrix_rank(np.concatenate([u, v], axis=1), tol=1e-8))
    passed = bool(msc >= msc_threshold)
    if not passed:
        return SharedSpace(
            basis=u[:, :rank],
            mean_subspace_correlation=msc,
            principal_angles_radians=angles,
            union_rank=union_rank,
            gate_passed=False,
            label="SOURCE_DESTINATION_TRANSPORT",
            threshold=float(msc_threshold),
        )
    mean_projector = 0.5 * (u @ u.T + v @ v.T)
    values, vectors = np.linalg.eigh(mean_projector)
    order = np.argsort(values)[::-1][: int(rank)]
    basis = np.linalg.qr(vectors[:, order])[0]
    return SharedSpace(
        basis=basis,
        mean_subspace_correlation=msc,
        principal_angles_radians=angles,
        union_rank=union_rank,
        gate_passed=True,
        label="FIXED_COORDINATE_DYNAMICS",
        threshold=float(msc_threshold),
    )


def _held_out_canonical_order(
    source: np.ndarray,
    destination: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Rank canonical directions by held-out correlation, best first."""

    n = source.shape[0]
    if n < 8:
        return np.arange(left.shape[1])
    half = n // 2
    a = source[half:] @ left
    b = destination[half:] @ right
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(a**2, axis=0) * np.sum(b**2, axis=0))
    correlation = np.divide(
        np.sum(a * b, axis=0), denominator, out=np.zeros(left.shape[1]), where=denominator > 1e-12
    )
    return np.argsort(np.abs(correlation))[::-1]


def haar_subspace(width: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-random orthonormal column basis of the requested rank."""

    return np.linalg.qr(rng.normal(size=(int(width), int(rank))))[0]
