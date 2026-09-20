"""Subspace comparison, deletion bases, and the aperture eigensystem.

Every similarity number here goes through R2.2's invariant I3: the two compared objects are
asserted to differ before any agreement statistic is formed, because a diagnostic that can
only return perfect agreement is not a diagnostic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from arm_d_r4.carriers import (  # noqa: F401
    aperture_eigenbasis,
    containment_in_parent,
    orthonormal,
    subspace_overlap,
    variance_captured,
)

from arm_d_r24.contract import DELETION_SEED_NAMESPACE, namespaced_seed


def mean_subspace_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """``||U^T V||_F^2 / r`` for equal-rank orthonormal bases (source section 10.1)."""

    u = np.asarray(left, dtype=np.float64)
    v = np.asarray(right, dtype=np.float64)
    if u.shape[1] != v.shape[1]:
        raise ValueError(f"MSC needs equal ranks, got {u.shape[1]} and {v.shape[1]}")
    return float(np.sum((u.T @ v) ** 2) / u.shape[1])


def principal_angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cosines = np.clip(
        np.linalg.svd(np.asarray(left).T @ np.asarray(right), compute_uv=False), -1.0, 1.0
    )
    return np.degrees(np.arccos(cosines))


def haar_subspace(width: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(int(width), int(rank))))[0]


def complement_basis(deleted: np.ndarray, *, width: int) -> np.ndarray:
    """An orthonormal basis of the surviving coordinates after deleting ``span(D)``.

    Refitting in the surviving coordinates is the same model as refitting the rank-deficient
    projected source, and is what "fully refit the predictive model from the remaining
    source coordinates" means numerically.
    """

    d = orthonormal(deleted, width=int(width))
    _, _, right = np.linalg.svd(d.T, full_matrices=True)
    return np.ascontiguousarray(right[d.shape[1]:].T)


def deleted_variance(covariance: np.ndarray, deleted: np.ndarray) -> float:
    d = np.asarray(deleted, dtype=np.float64)
    sigma = np.asarray(covariance, dtype=np.float64)
    return float(np.trace(d.T @ sigma @ d) / np.trace(sigma))


def rotate_toward(basis: np.ndarray, targets: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate every column of ``basis`` by ``degrees`` toward the matching target column."""

    angle = np.radians(float(degrees))
    columns = []
    for index in range(basis.shape[1]):
        u = basis[:, index]
        v = targets[:, index % targets.shape[1]]
        v = v - u * float(u @ v)
        norm = float(np.linalg.norm(v))
        columns.append(u if norm <= 0 else np.cos(angle) * u + np.sin(angle) * (v / norm))
    return np.linalg.qr(np.stack(columns, axis=1))[0]


def deletion_families(
    eigen: dict[str, np.ndarray],
    *,
    width: int,
    rank: int,
    haar_draws: int,
    mixed_m: Sequence[int],
    rotation_angles: Sequence[float],
    dose_ranks: Sequence[int],
    dose_grid: Sequence[float],
    target_variance: float,
    cumulative: np.ndarray,
) -> dict[str, dict[str, object]]:
    """The deletion families of unified section 7, family F included and required."""

    basis = np.asarray(eigen["basis"], dtype=np.float64)
    families: dict[str, dict[str, object]] = {}

    families["A_pca_1_16"] = {
        "family": "A",
        "role": "target: the top-variance rank-16 subspace",
        "basis": np.ascontiguousarray(basis[:, :rank]),
    }
    families["B_pca_17_32"] = {
        "family": "B",
        "role": "the next 16 principal directions",
        "basis": np.ascontiguousarray(basis[:, rank : 2 * rank]),
    }
    for index in range(int(haar_draws)):
        rng = np.random.default_rng(namespaced_seed(DELETION_SEED_NAMESPACE, rank, index))
        families[f"C_haar_{index}"] = {
            "family": "C",
            "role": "ambient Haar rank-16 deletion",
            "basis": haar_subspace(width, rank, rng),
        }
    complement = np.ascontiguousarray(basis[:, rank:])
    for m in mixed_m:
        rng = np.random.default_rng(namespaced_seed(DELETION_SEED_NAMESPACE + "_mixed", rank, m))
        inner = haar_subspace(complement.shape[1], int(rank - m), rng) if m < rank else None
        parts = [basis[:, : int(m)]] if m > 0 else []
        if inner is not None:
            parts.append(complement @ inner)
        families[f"D_mixed_m{int(m)}"] = {
            "family": "D",
            "role": f"{int(m)} top PCs plus {int(rank - m)} complement-Haar directions",
            "basis": np.linalg.qr(np.concatenate(parts, axis=1))[0],
        }
    lower = np.ascontiguousarray(basis[:, rank : 2 * rank])
    for angle in rotation_angles:
        families[f"E_rotated_{angle:g}"] = {
            "family": "E",
            "role": f"PCA1:16 rotated {angle:g} degrees toward PCs 17:32",
            "basis": rotate_toward(basis[:, :rank], lower, float(angle)),
        }

    # Family F (amendments A3 and A12).  A3 assumed the target's captured variance could be
    # matched by taking enough LOWER principal directions.  On this aperture it cannot: the
    # top 16 directions hold about 0.672 of the variance, so every other direction together
    # holds only about 0.328.  The family is built anyway, up to the maximum attainable dose,
    # and the shortfall is recorded rather than papered over -- and A12 adds the comparison
    # that IS attainable: top-PC against lower-direction deletions at a COMMON dose.
    tail = np.asarray(cumulative, dtype=np.float64)
    fractions = np.diff(np.concatenate([[0.0], tail]))
    attainable = float(np.sum(fractions[rank:]))
    running, take = 0.0, 0
    for index in range(rank, fractions.shape[0]):
        running += float(fractions[index])
        take += 1
        if running >= float(target_variance):
            break
    families["F_energy_matched"] = {
        "family": "F",
        "role": (
            f"lower principal directions PCs {rank + 1}..{rank + take}, reaching captured "
            f"variance {running:.6f} against the target's {float(target_variance):.6f}"
        ),
        "basis": np.ascontiguousarray(basis[:, rank : rank + take]),
        "matched_rank": int(take),
        "attained_variance": float(running),
        "target_variance": float(target_variance),
        "energy_match_attained": bool(running >= float(target_variance) - 1e-9),
        "maximum_attainable_lower_variance": attainable,
    }

    # A12: the dose-matched geometry contrast.  At each attainable dose, one deletion built
    # from the TOP principal directions and one built from the LOWER ones, carrying the same
    # fraction of aperture variance.  Rank is deliberately not matched -- matching rank and
    # dose at once is what the impossibility argument forbids.
    for target in dose_grid:
        target = float(target)
        if target > attainable + 1e-9:
            continue
        top_take = int(np.searchsorted(tail, target) + 1)
        lower_running, lower_take = 0.0, 0
        for index in range(fractions.shape[0] - 1, -1, -1):
            lower_running += float(fractions[index])
            lower_take += 1
            if lower_running >= target:
                break
        key = f"{target:g}"
        families[f"H_dose{key}_top"] = {
            "family": "H_top",
            "role": f"top principal directions reaching deleted variance {target:g}",
            "basis": np.ascontiguousarray(basis[:, :top_take]),
            "dose_target": target,
        }
        families[f"H_dose{key}_lower"] = {
            "family": "H_lower",
            "role": f"lowest principal directions reaching deleted variance {target:g}",
            "basis": np.ascontiguousarray(basis[:, basis.shape[1] - lower_take :]),
            "dose_target": target,
        }
    for r in dose_ranks:
        if int(r) >= basis.shape[1]:
            continue
        families[f"G_pca_top{int(r)}"] = {
            "family": "G",
            "role": f"dose context: the top {int(r)} principal directions",
            "basis": np.ascontiguousarray(basis[:, : int(r)]),
        }
    return families


def local_linear_prediction(
    x: np.ndarray, y: np.ndarray, at: float, *, bandwidth: float
) -> float:
    """Kernel-weighted local linear fit of the dose-response curve at one dose."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.exp(-0.5 * ((x - float(at)) / float(bandwidth)) ** 2)
    if float(np.sum(weights)) <= 0:
        return float("nan")
    design = np.stack([np.ones_like(x), x - float(at)], axis=1)
    gram = design.T @ (weights[:, None] * design)
    cross = design.T @ (weights * y)
    try:
        return float(np.linalg.solve(gram, cross)[0])
    except np.linalg.LinAlgError:
        return float(np.sum(weights * y) / np.sum(weights))
