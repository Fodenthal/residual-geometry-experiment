"""Eigenvalue conditioning: is the spectrum of ``A_k`` a portable invariant?

R1 §20-§22 report eigenvalue moduli, decay times, and frequencies of the fitted
operator.  Those quantities are similarity-invariant, but similarity invariance
is algebraic and says nothing about conditioning: under strong non-normality a
single eigenvalue moves by ``kappa_j ||dA||`` to first order, so a perturbation
of ``A_k`` far smaller than the estimation noise can carry a mode anywhere.
This module is the numerical-trust gate that ``arm_d.spectra`` presumes.

* :func:`eigenvalue_conditioning` -- Bauer-Fike per-mode condition numbers
  ``kappa_j = 1 / |<u_j, v_j>|`` over unit-norm left/right eigenvectors, as a
  full distribution rather than only its maximum.
* :func:`departure_from_normality` -- Henrici's Frobenius departure, the
  scale-free summary of how far ``A`` is from a matrix whose eigenvectors are
  orthogonal.
* :func:`bootstrap_eigenvalues` -- the empirical counterpart: how far the modes
  actually move when the documents are resampled.
* :func:`singular_value_invariants` -- the Weyl-stable fallback.  Singular
  values move by at most ``||dA||_2`` for *any* matrix, so they remain reportable
  exactly where the eigenvalues do not.
* :func:`schur_block_separation` -- an invariant subspace is only better
  behaved than its individual eigenvalues when the retained block is separated
  from the remainder.  A small gap is unsafe in the same way a large condition
  number is.
* :func:`spectral_label` -- the verdict the report is allowed to quote.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.linalg

TINY = float(np.finfo(np.float64).tiny)

#: Default ceiling on ``max_j kappa_j`` above which eigenvalues stop being
#: treated as portable.  R1 §21 quotes eigenvalue moduli to two decimals; a
#: condition number of 100 lets a relative perturbation of 1e-3 move a mode by
#: 0.1, which is larger than the reported precision.
DEFAULT_MAX_CONDITION_NUMBER = 100.0

#: Heuristic used by :func:`spectral_label` to flag analytic/bootstrap
#: disagreement.  See that function's docstring.
BOOTSTRAP_SPREAD_FLOOR = 0.05
BOOTSTRAP_SPREAD_PER_CONDITION = 0.01


def _square_matrix(values: np.ndarray, name: str = "a") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _unit_columns(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    return vectors / norms


# --------------------------------------------------------------------------
# analytic conditioning
# --------------------------------------------------------------------------


def eigenvalue_conditioning(a: np.ndarray) -> dict[str, object]:
    """Per-eigenvalue condition numbers of a real square operator.

    ``kappa_j = 1 / |<u_j, v_j>|`` with ``u_j`` and ``v_j`` the unit-norm left
    and right eigenvectors of mode ``j``.  A normal matrix has ``kappa_j = 1``
    for every mode; a defective matrix has ``kappa_j = inf`` where the left and
    right eigenvectors become orthogonal.  The whole distribution is returned
    because one badly conditioned mode does not condemn the rest, and the
    median is what says whether the spectrum as a whole is trustworthy.
    """

    matrix = _square_matrix(a)
    eigenvalues, left, right = scipy.linalg.eig(matrix, left=True, right=True)
    left = _unit_columns(np.asarray(left))
    right = _unit_columns(np.asarray(right))
    condition_numbers = np.empty(eigenvalues.size, dtype=np.float64)
    for index in range(eigenvalues.size):
        overlap = abs(np.vdot(left[:, index], right[:, index]))
        condition_numbers[index] = np.inf if overlap <= 0.0 else 1.0 / float(overlap)
    return {
        "eigenvalues_real": [float(value) for value in np.real(eigenvalues)],
        "eigenvalues_imag": [float(value) for value in np.imag(eigenvalues)],
        "condition_numbers": [float(value) for value in condition_numbers],
        "max_condition_number": float(np.max(condition_numbers)) if condition_numbers.size else float("nan"),
        "median_condition_number": float(np.median(condition_numbers)) if condition_numbers.size else float("nan"),
        # ``method="higher"`` picks an observed condition number rather than
        # interpolating, which keeps the quantile defined when a defective mode
        # contributes an infinite entry.
        "q90_condition_number": (
            float(np.quantile(condition_numbers, 0.90, method="higher"))
            if condition_numbers.size
            else float("nan")
        ),
        "n_modes": int(eigenvalues.size),
    }


def departure_from_normality(a: np.ndarray) -> float:
    r"""Henrici's Frobenius departure from normality, relative to ``||A||_F``.

    ``sqrt(max(0, ||A||_F^2 - sum_j |lambda_j|^2)) / ||A||_F``.  Zero exactly
    when ``A`` is normal (Schur's inequality is tight), one when the operator
    carries all of its energy in the strictly triangular part of its Schur
    form.  The ``max(0, ...)`` clamp only absorbs rounding: the quantity under
    the root is non-negative for every matrix.
    """

    matrix = _square_matrix(a)
    frobenius_squared = float(np.sum(matrix * matrix))
    if frobenius_squared <= 0.0:
        return 0.0
    eigenvalues = scipy.linalg.eigvals(matrix)
    spectral_energy = float(np.sum(np.abs(eigenvalues) ** 2))
    return float(np.sqrt(max(0.0, frobenius_squared - spectral_energy)) / np.sqrt(frobenius_squared))


# --------------------------------------------------------------------------
# empirical conditioning
# --------------------------------------------------------------------------


def bootstrap_eigenvalues(
    fit_operator: Callable[[np.ndarray], np.ndarray | None],
    document_index: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Resample documents and measure how far each mode actually moves.

    ``fit_operator`` receives one resampled array of document ids and returns
    the refitted square operator, or ``None`` when that resample is degenerate
    (too few documents, a singular Gram matrix).  Degenerate replicates are
    skipped rather than raised on, and ``n_successful`` records how many
    survived so the caller can refuse a spread computed from a handful of fits.

    Eigenvalues are sorted by descending modulus within each replicate.  That
    ordering is the only thing making mode ``j`` comparable across replicates,
    and it is itself unreliable when two moduli are close -- a near-degenerate
    pair swaps positions between replicates and inflates the spread at both
    positions.  The reported spread is therefore an upper bound on per-mode
    instability, which is the safe direction for a trust gate.
    """

    documents = np.asarray(document_index)
    rng = np.random.default_rng(int(seed))
    moduli: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    for _ in range(int(replicates)):
        resample = rng.choice(documents, size=documents.size, replace=True)
        operator = fit_operator(resample)
        if operator is None:
            continue
        matrix = np.asarray(operator, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
            continue
        eigenvalues = scipy.linalg.eigvals(matrix)
        order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[order]
        moduli.append(np.abs(eigenvalues))
        angles.append(np.angle(eigenvalues))
    if not moduli:
        return {
            "n_successful": 0,
            "modulus_spread": [],
            "modulus_q05": [],
            "modulus_q95": [],
            "angle_spread": [],
        }
    width = min(int(values.size) for values in moduli)
    modulus_matrix = np.stack([values[:width] for values in moduli], axis=0)
    angle_matrix = np.stack([values[:width] for values in angles], axis=0)
    modulus_iqr = np.quantile(modulus_matrix, 0.75, axis=0) - np.quantile(modulus_matrix, 0.25, axis=0)
    angle_iqr = np.quantile(angle_matrix, 0.75, axis=0) - np.quantile(angle_matrix, 0.25, axis=0)
    return {
        "n_successful": int(modulus_matrix.shape[0]),
        "modulus_spread": [float(value) for value in modulus_iqr],
        "modulus_q05": [float(value) for value in np.quantile(modulus_matrix, 0.05, axis=0)],
        "modulus_q95": [float(value) for value in np.quantile(modulus_matrix, 0.95, axis=0)],
        "angle_spread": [float(value) for value in angle_iqr],
    }


# --------------------------------------------------------------------------
# stable fallbacks
# --------------------------------------------------------------------------


def singular_value_invariants(a: np.ndarray) -> dict[str, object]:
    """Singular values and their norms -- the invariants Weyl's bound protects.

    ``|sigma_j(A + dA) - sigma_j(A)| <= ||dA||_2`` holds for every matrix with
    no conditioning caveat, so these remain reportable in exactly the cells
    where :func:`spectral_label` withholds the eigenvalues.  They are invariant
    under orthogonal similarity but *not* under a general similarity, which is
    the price paid for the stability.
    """

    matrix = _square_matrix(a)
    singular_values = np.asarray(scipy.linalg.svdvals(matrix), dtype=np.float64)
    singular_values = np.sort(singular_values)[::-1]
    spectral_norm = float(singular_values[0]) if singular_values.size else 0.0
    frobenius_squared = float(np.sum(singular_values**2))
    return {
        "singular_values": [float(value) for value in singular_values],
        "spectral_norm": spectral_norm,
        "nuclear_norm": float(np.sum(singular_values)),
        "stable_rank": float(frobenius_squared / spectral_norm**2) if spectral_norm > 0 else 0.0,
    }


def schur_block_separation(
    a: np.ndarray,
    block_size: int,
    *,
    min_separation: float = 0.10,
) -> dict[str, object]:
    """Modulus gap between the leading invariant subspace and the remainder.

    An invariant subspace can be far better conditioned than the individual
    eigenvalues inside it, but only when it is separated from the rest of the
    spectrum: the perturbation of the subspace scales as ``||dA|| / sep``.  The
    real Schur form is reordered so the ``block_size`` largest-modulus
    eigenvalues occupy the leading block, and the gap is measured between the
    smallest retained modulus and the largest discarded one.  A small
    separation is unsafe in exactly the way a large condition number is, and
    ``well_separated`` is the flag that says so.
    """

    matrix = _square_matrix(a)
    width = matrix.shape[0]
    block = int(np.clip(int(block_size), 0, width))
    moduli = np.sort(np.abs(scipy.linalg.eigvals(matrix)))[::-1]
    schur_form = _reordered_schur(matrix, moduli, block)
    retained = np.abs(scipy.linalg.eigvals(schur_form[:block, :block])) if block > 0 else np.zeros(0)
    remainder = np.abs(scipy.linalg.eigvals(schur_form[block:, block:])) if block < width else np.zeros(0)
    retained = np.sort(retained)[::-1]
    remainder = np.sort(remainder)[::-1]
    minimum_retained = float(retained.min()) if retained.size else 0.0
    maximum_remainder = float(remainder.max()) if remainder.size else 0.0
    separation = minimum_retained - maximum_remainder
    scale = max(float(retained.max()) if retained.size else 0.0, TINY)
    return {
        "block_size": block,
        "retained_eigenvalue_moduli": [float(value) for value in retained],
        "remainder_eigenvalue_moduli": [float(value) for value in remainder],
        "separation": float(separation),
        "separation_relative": float(separation / scale),
        "well_separated": bool(separation / scale >= float(min_separation)),
    }


def _reordered_schur(matrix: np.ndarray, moduli: np.ndarray, block: int) -> np.ndarray:
    """Real Schur form with the ``block`` largest-modulus eigenvalues first.

    ``scipy.linalg.schur`` leaves the diagonal order unspecified, so the
    leading block of an unsorted factorization is arbitrary.  When the sort
    would split a complex conjugate pair -- LAPACK cannot separate the two
    halves of a 2x2 real block -- the reorder is refused and the unsorted form
    is returned.  The two diagonal blocks then no longer hold the largest and
    smallest moduli, and the reported separation is small or negative, which
    lands on ``well_separated=False``: a block boundary that cuts a conjugate
    pair is not a usable invariant subspace and must not read as safe.
    """

    if block <= 0 or block >= matrix.shape[0]:
        return np.asarray(scipy.linalg.schur(matrix, output="real")[0], dtype=np.float64)
    threshold = float(moduli[block - 1])

    def keep(real: float, imaginary: float) -> bool:
        return bool(np.hypot(real, imaginary) >= threshold * (1.0 - 1e-12))

    try:
        schur_form, _, kept = scipy.linalg.schur(matrix, output="real", sort=keep)
        if int(kept) == block:
            return np.asarray(schur_form, dtype=np.float64)
    except (ValueError, scipy.linalg.LinAlgError):
        pass
    return np.asarray(scipy.linalg.schur(matrix, output="real")[0], dtype=np.float64)


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------


def spectral_label(
    conditioning: dict[str, object],
    *,
    max_condition_number: float = DEFAULT_MAX_CONDITION_NUMBER,
    bootstrap: dict[str, object] | None = None,
    shared_space_forbids_eigenvalues: bool = False,
) -> dict[str, object]:
    """Whether the eigenvalues of this operator may be quoted as invariants.

    Two things can withhold them.  The shared-space gate comes first: under
    ``SOURCE_DESTINATION_TRANSPORT`` (R1 §12) the operator maps one space into
    a different one, so its eigenvalues are not the eigenvalues of any
    endomorphism and no amount of good conditioning rescues them.  Otherwise
    the analytic conditioning decides: ``max_j kappa_j > max_condition_number``
    means a perturbation smaller than the estimation noise can move a reported
    mode further than the reported precision.

    When ``bootstrap`` is supplied its worst ``modulus_spread`` is compared
    against what the analytic conditioning predicts, using one deliberately
    simple heuristic: the two DISAGREE when the worst modulus IQR exceeds both
    ``BOOTSTRAP_SPREAD_FLOOR`` (0.05, the reporting precision of R1 §21) and
    ``BOOTSTRAP_SPREAD_PER_CONDITION * max_j kappa_j`` (0.01 per unit of
    condition number).  A disagreement is appended to ``reasons`` and does not
    flip the label: the analytic number is a first-order bound on perturbation
    of a fixed matrix, while the bootstrap spread also absorbs mode reordering
    and genuine sampling variance, so the two measure different things and the
    discrepancy is evidence for a human rather than a rule.
    """

    reasons: list[str] = []
    maximum = float(conditioning.get("max_condition_number", float("nan")))
    if shared_space_forbids_eigenvalues:
        label = "SPECTRUM_ILL_CONDITIONED"
        portable = False
        reasons.append(
            "shared-space gate failed: the operator is a source-destination transport between "
            "two different spaces, so its eigenvalues are not invariants of any single space"
        )
    elif not (maximum <= float(max_condition_number)):
        label = "SPECTRUM_ILL_CONDITIONED"
        portable = False
        reasons.append(
            f"max eigenvalue condition number {maximum:.4g} exceeds the ceiling {float(max_condition_number):.4g}; "
            "individual eigenvalues can move further than the reported precision under a tiny perturbation"
        )
    else:
        label = "SPECTRUM_WELL_CONDITIONED"
        portable = True
        reasons.append(
            f"max eigenvalue condition number {maximum:.4g} is within the ceiling {float(max_condition_number):.4g}"
        )
    if bootstrap is not None:
        spreads = _as_float_sequence(bootstrap.get("modulus_spread", []))
        if spreads:
            worst = float(np.max(spreads))
            budget = BOOTSTRAP_SPREAD_PER_CONDITION * maximum if np.isfinite(maximum) else np.inf
            if worst > BOOTSTRAP_SPREAD_FLOOR and worst > budget:
                reasons.append(
                    f"analytic/bootstrap disagreement: worst bootstrap modulus IQR {worst:.4g} exceeds both the "
                    f"{BOOTSTRAP_SPREAD_FLOOR:.4g} reporting floor and the {budget:.4g} implied by "
                    f"max kappa {maximum:.4g}; label unchanged, the two diagnostics measure different things"
                )
    return {"label": label, "eigenvalues_are_portable": portable, "reasons": reasons}


def _as_float_sequence(values: object) -> list[float]:
    if values is None or isinstance(values, (str, bytes)):
        return []
    array = np.atleast_1d(np.asarray(values, dtype=np.float64))
    return [float(value) for value in array]
