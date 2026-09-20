"""Frozen-chain transfer across document groups (R1 §33, Amendment 3).

Amendment 2 (:mod:`arm_d.strata`) asks whether objects *refitted independently*
per stratum look similar.  Similarity of two independent fits is a statement
about the estimator's stability, not about the objects themselves: two strata
can admit near-identical rank curves while the actual aperture, whitener, and
operator fitted on one are useless on the other.  This module asks the stronger
question -- do the **same frozen objects** still predict on a different
population -- and reports where transfer fails when it does.

Three constructions carry the claim.

**The whole chain is frozen, not just the operator.**  :class:`FrozenChain`
carries the mean, the nuisance model, the aperture ``Q``, both whiteners, the
ridge epsilons, and the per-lag operators as one bundle.  Refitting any one of
them on the evaluation group -- a whitener re-estimated "just to standardize
scale", a nuisance model re-residualized "because the domain differs" -- lets
the estimator adapt to the new distribution through the back door, and a
transfer result obtained that way measures nothing.  The three-cell design of
:func:`transfer_cells` unfreezes objects *deliberately and one stage at a time*
instead.

**A difference, not a ratio.**  :func:`transfer_gap` is the primary statistic.
:func:`transfer_ratio` exists as a secondary readout and refuses to report at
all when the denominator is small, because that is exactly where a ratio is
unbounded and where the science is most delicate.

**Signed skill, never clipped.**  :func:`skill_over_baseline` is a difference of
two absolute ``R^2`` values against the target's own mean, so it lives on the
same scale as the headline cell statistic and a frozen chain that actively
mispredicts returns a negative number.  That negative number is a result --
the frozen law is wrong on the new population, not merely uninformative -- and
clipping it to zero would erase the distinction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

COMPONENT_NAMES: tuple[str, ...] = (
    "mean",
    "nuisance_model",
    "projector",
    "source_whitener",
    "destination_whitener",
    "ridge_epsilons",
    "operators",
)

# A transfer verdict is only issued when the objects it rests on were actually
# computed, and only described as evidence when the null band is tight enough to
# discriminate.  Both constants are preregistered here rather than chosen after
# seeing a band.
TRANSFER_UNAVAILABLE = "TRANSFER_UNAVAILABLE"
MIN_BAND_REPLICATES = 20
WIDE_BAND_WIDTH = 0.05

TRANSFER_CELLS: tuple[str, ...] = (
    "frozen_q_frozen_a",
    "frozen_q_refit_a",
    "refit_q_refit_a",
)


def _vector_block(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return array


# --------------------------------------------------------------------------
# The frozen bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenChain:
    """Every fitted object of one estimator chain, frozen together.

    The bundle is deliberately whole.  A transfer claim is a claim that the
    objects fitted on the fitting group are the objects that work on the
    evaluation group, and *every* stage of the chain is a fitted object: the
    centering mean, the nuisance residualization model, the aperture ``Q``, the
    source and destination whiteners, the ridge epsilons chosen by the
    whitening health schedule, and the per-lag operators.  Silently refitting
    any of them on the evaluation population -- a whitener re-estimated so the
    coordinates "match", a nuisance model re-fitted because the new domain has
    different surface statistics -- lets the estimator absorb the distribution
    shift before the operator ever sees it.  The measured gap then shrinks for
    a reason that has nothing to do with the transition law, and the claim is
    quietly destroyed rather than tested.

    ``operators`` and ``ridge_epsilons`` are keyed by lag ``k``.
    ``nuisance_model`` may be ``None`` when the chain was fitted without
    residualization; it is held as an opaque object because this module never
    applies it, only records that it was frozen.
    """

    mean: np.ndarray
    nuisance_model: object | None
    projector: np.ndarray
    source_whitener: np.ndarray
    destination_whitener: np.ndarray
    ridge_epsilons: Mapping[int, float]
    operators: Mapping[int, np.ndarray]

    def components_frozen(self) -> list[str]:
        """Names of the frozen components, in chain order."""

        return list(COMPONENT_NAMES)


def apply_frozen(chain: FrozenChain, source: np.ndarray, lag: int) -> np.ndarray:
    """Predict at ``lag`` with the frozen chain, in destination-whitened coordinates.

    The source rows are centered by the frozen mean, projected through the
    frozen aperture ``Q``, whitened by the frozen source whitener, and pushed
    through the frozen operator for ``lag``.  No statistic of ``source`` is
    estimated anywhere in the path, which is the point: the returned prediction
    is comparable to a target expressed in the *frozen* destination-whitened
    coordinates, and nothing about the evaluation population has entered the
    transform.
    """

    key = int(lag)
    if key not in chain.operators:
        raise KeyError(f"frozen chain has no operator for lag {key}")
    rows = _vector_block(source, "source")
    mean = np.asarray(chain.mean, dtype=np.float64).reshape(-1)
    if mean.shape[0] != rows.shape[1]:
        raise ValueError("frozen mean width does not match the source width")
    centered = rows - mean[None, :]
    projected = centered @ np.asarray(chain.projector, dtype=np.float64)
    whitened = projected @ np.asarray(chain.source_whitener, dtype=np.float64)
    operator = np.asarray(chain.operators[key], dtype=np.float64)
    return whitened @ operator.T


# --------------------------------------------------------------------------
# Skill on an absolute scale
# --------------------------------------------------------------------------


def _sum_squares(values: np.ndarray) -> float:
    return float(np.sum(np.asarray(values, dtype=np.float64) ** 2))


def _r2_against_own_mean(prediction: np.ndarray, target: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    total = _sum_squares(y - y.mean(axis=0, keepdims=True))
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    return 1.0 - _sum_squares(y - np.asarray(prediction, dtype=np.float64)) / total


def skill_over_baseline(
    prediction: np.ndarray,
    target: np.ndarray,
    baseline_prediction: np.ndarray,
) -> float:
    """``R^2(prediction) - R^2(baseline)``, both against the target's own mean.

    The alternative construction ``1 - SSE(prediction) / SSE(baseline)`` is a
    *relative* improvement measured against the baseline's error rather than
    against the target's variance.  It is zero when the two predictors tie and
    one when the prediction is exact, so it looks like a skill score, but its
    units depend on how good the baseline happens to be: the same absolute
    improvement reads large next to a weak baseline and small next to a strong
    one, and the number is not comparable with the headline cell statistic.
    Differencing two absolute ``R^2`` values keeps the result on one fixed
    scale across groups, lags, and baselines.

    A worse-than-baseline prediction returns a negative value and it is
    returned as-is.  A frozen chain that actively mispredicts on a new
    population is a strong interpretable result -- the fitted law is wrong
    there, not merely absent -- and clipping at zero would collapse it into the
    "no signal" case.
    """

    return float(
        _r2_against_own_mean(prediction, target)
        - _r2_against_own_mean(baseline_prediction, target)
    )


# --------------------------------------------------------------------------
# The three-cell design
# --------------------------------------------------------------------------


def transfer_cells(
    fit_frozen: Callable[[], np.ndarray],
    fit_refit_a: Callable[[], np.ndarray],
    fit_refit_both: Callable[[], np.ndarray],
    evaluate: Callable[[np.ndarray], float],
) -> dict[str, float]:
    """Skill of the three transfer cells, evaluated on the same rows.

    The cells are

    * ``frozen_q_frozen_a`` -- the whole chain frozen;
    * ``frozen_q_refit_a`` -- the frozen aperture ``Q``, with the operator
      ``A`` refitted on the evaluation group inside those frozen coordinates;
    * ``refit_q_refit_a`` -- both refitted, the oracle for this group.

    The expected ordering is (a) <= (b) <= (c).  Each step relaxes one
    constraint, so each can only help a well-behaved estimator.  The ordering
    is what localizes *where* transfer fails, which a two-cell frozen-versus-
    refit comparison cannot do: if (a) is far below (b) but (b) reaches (c),
    the coordinates carry over and the transition law does not; if (b) is
    already far below (c), the aperture itself is group-specific and no
    operator fitted inside it can recover.  A violation of the ordering is
    informative too, and is reported rather than sorted away -- it means the
    refit is noisier than the frozen object, usually because the refitting
    group is small (see :func:`match_fitting_documents`).

    There is deliberately **no fourth cell** pairing a refitted ``Q`` with the
    frozen ``A``.  ``A`` is expressed in ``Q``'s own coordinates, so a frozen
    ``A`` has no defined action on a differently-fitted basis: making the
    pairing well posed requires fitting an alignment between the two apertures,
    and that alignment is itself an adaptation to the evaluation population --
    the exact adaptation the frozen cell exists to forbid.  The cell would
    measure the alignment fit, not transfer.
    """

    return {
        "frozen_q_frozen_a": float(evaluate(fit_frozen())),
        "frozen_q_refit_a": float(evaluate(fit_refit_a())),
        "refit_q_refit_a": float(evaluate(fit_refit_both())),
    }


def transfer_gap(skill_refit: float, skill_frozen: float) -> float:
    """Primary transfer statistic: ``skill_refit - skill_frozen``.

    A difference, not a ratio.  Both terms are already absolute ``R^2``
    improvements over the same baseline on the same rows, so their difference
    is on that same interpretable scale and stays finite and signed however
    small either term becomes.
    """

    return float(skill_refit) - float(skill_frozen)


def transfer_ratio(
    skill_refit: float,
    skill_frozen: float,
    *,
    min_refit_skill: float,
) -> float | None:
    """Secondary readout ``skill_frozen / skill_refit``, or ``None`` when unusable.

    Ratios are reported only where the refit (oracle) skill clears
    ``min_refit_skill``.  A ratio is unbounded and unstable exactly where its
    denominator approaches zero, which here is the regime where the estimator
    barely works on the evaluation group at all -- the place the science is
    most delicate and where a headline "retains 300% of the refit skill" would
    be pure division noise.  Below the floor the honest answer is that the
    ratio is undefined, so ``None`` is returned and the gap carries the claim.
    """

    if not np.isfinite(skill_refit) or float(skill_refit) < float(min_refit_skill):
        return None
    denominator = float(skill_refit)
    if denominator == 0.0:
        return None
    return float(skill_frozen) / denominator


# --------------------------------------------------------------------------
# The identical-distribution null band
# --------------------------------------------------------------------------


def null_band(
    gaps: Sequence[float],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> dict[str, float]:
    """Band of gaps produced when fitting and evaluation groups are the same population.

    A transfer gap is essentially never exactly zero: refitting on a different
    random half of the *same* distribution already moves the skill, so "the gap
    is positive" is not evidence of a distribution-specific law.  ``gaps`` are
    the gaps measured under identical-distribution splits, and the returned
    interval is the noise floor any cross-group gap must clear.  Non-finite
    entries are dropped; an empty band returns ``nan`` bounds with ``n = 0``.
    """

    draws = np.asarray(list(gaps), dtype=np.float64).reshape(-1)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return {
            "lower": float("nan"), "upper": float("nan"), "median": float("nan"),
            "n": 0.0, "width": float("nan"), "reportable": False,
            "unreportable_because": "no finite identical-distribution gaps",
        }
    lower = float(np.quantile(draws, float(lower_quantile)))
    upper = float(np.quantile(draws, float(upper_quantile)))
    # The width travels with the band everywhere.  A verdict of "the gap is
    # inside the band" means nothing without it: a band of +/-0.12 accepts
    # almost any gap, so the verdict reflects low discriminating power rather
    # than demonstrated transfer, and must be labelled that way.
    return {
        "lower": lower,
        "upper": upper,
        "median": float(np.median(draws)),
        "n": float(draws.size),
        "width": float(upper - lower),
        "reportable": bool(draws.size >= MIN_BAND_REPLICATES),
        "minimum_replicates": float(MIN_BAND_REPLICATES),
        "unreportable_because": (
            None
            if draws.size >= MIN_BAND_REPLICATES
            else (
                f"only {int(draws.size)} identical-distribution replicates, below the "
                f"preregistered minimum of {MIN_BAND_REPLICATES}; report per-lag bands "
                "with their replicate counts instead of an aggregate"
            )
        ),
    }


def inside_band(gap: float, band: Mapping[str, float]) -> bool:
    """Whether ``gap`` falls inside the identical-distribution band, inclusive.

    A non-finite gap, or a band with non-finite bounds (an empty null), is not
    inside: absence of a usable null is not evidence of transfer.
    """

    value = float(gap)
    lower = float(band.get("lower", float("nan")))
    upper = float(band.get("upper", float("nan")))
    if not (np.isfinite(value) and np.isfinite(lower) and np.isfinite(upper)):
        return False
    return bool(lower <= value <= upper)


# --------------------------------------------------------------------------
# Size matching and the document bootstrap
# --------------------------------------------------------------------------


def match_fitting_documents(
    document_ids_by_group: Mapping[object, Sequence],
    rng: np.random.Generator,
) -> dict[object, np.ndarray]:
    """Subsample every group's fitting documents to the smallest group's count.

    The refit cells are fitted per group, and a refit on a small group is a
    noisier estimate than a refit on a large one.  Unmatched counts therefore
    let the frozen chain beat the refit -- or the refit beat the frozen chain
    -- for a reason that is entirely sample size and nothing to do with
    distribution shift.  Matching first removes that route to both a false
    positive and a false negative.  Selected ids are returned sorted for
    reproducibility; an empty group yields an empty selection for every group.
    """

    arrays = {
        group: np.asarray(list(ids)) for group, ids in document_ids_by_group.items()
    }
    if not arrays:
        return {}
    smallest = int(min(array.shape[0] for array in arrays.values()))
    if smallest <= 0:
        return {group: array[:0] for group, array in arrays.items()}
    matched: dict[object, np.ndarray] = {}
    for group, array in arrays.items():
        take = rng.choice(array.shape[0], size=smallest, replace=False)
        matched[group] = array[np.sort(take)]
    return matched


def bootstrap_gap(
    gap_fn: Callable[[np.ndarray], float | None],
    evaluation_documents: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Document-level bootstrap interval on a transfer gap.

    Documents, not rows, are resampled: rows within a document are strongly
    dependent, so a row-level bootstrap would report an interval several times
    too narrow.  Resampling is over the **evaluation** documents only -- the
    frozen objects are fixed by construction, and the question is how much the
    measured gap moves with the evaluation sample.

    ``gap_fn`` receives an array of resampled document ids (with replacement,
    so ids repeat) and returns a gap.  Replicates that return ``None`` or a
    non-finite value are skipped rather than imputed; the count that survived
    is not returned, so callers wanting it should count themselves.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    ids = np.asarray(evaluation_documents)
    if ids.shape[0] == 0:
        return {"median": float("nan"), "q05": float("nan"), "q95": float("nan")}
    draws: list[float] = []
    for _ in range(int(replicates)):
        resampled = ids[rng.integers(0, ids.shape[0], size=ids.shape[0])]
        value = gap_fn(resampled)
        if value is None:
            continue
        gap = float(value)
        if np.isfinite(gap):
            draws.append(gap)
    if not draws:
        return {"median": float("nan"), "q05": float("nan"), "q95": float("nan")}
    array = np.asarray(draws, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def transfer_label(
    cell_a_gap: float,
    cell_b_gap: float,
    cell_c_skill: float,
    band: Mapping[str, float],
    *,
    min_oracle_skill: float,
) -> str:
    """Backwards-compatible wrapper returning only the label string."""

    return transfer_verdict(
        cell_a_gap, cell_b_gap, cell_c_skill, band, min_oracle_skill=min_oracle_skill
    )["label"]


def transfer_verdict(
    cell_a_gap: float,
    cell_b_gap: float,
    cell_c_skill: float,
    band: Mapping[str, float],
    *,
    min_oracle_skill: float,
) -> dict[str, object]:
    """Transfer verdict for one (fitting group, evaluation group) pair.

    One of four labels:

    * ``NO_RECURRENCE`` -- the oracle cell ``refit_q_refit_a`` itself fails to
      clear ``min_oracle_skill`` on this group;
    * ``INTERFACE_AND_LAW_TRANSFER`` -- the fully frozen gap sits inside the
      identical-distribution band;
    * ``INTERFACE_TRANSFERS_LAW_SPECIFIC`` -- the frozen-``Q`` gap is inside
      the band but the fully frozen gap is not, so the aperture carries over
      and the operator does not;
    * ``STRUCTURE_RECURS_OBJECTS_DO_NOT`` -- neither gap is inside the band,
      but the oracle works, so the structure exists in this group and the
      fitted objects are group-specific.

    ``NO_RECURRENCE`` is checked **first**.  Where the oracle fails there is no
    structure in the evaluation group to transfer *to*, and a frozen chain that
    predicts nothing produces a gap near zero for that reason alone -- both
    skills are near zero, so their difference is too.  Checking the band first
    would award ``INTERFACE_AND_LAW_TRANSFER`` to exactly the cells with no
    signal at all.
    """

    # A MISSING COMPUTATION IS NOT A FINDING.  The first implementation returned
    # NO_RECURRENCE whenever the oracle was non-finite, so an uncomputed refit
    # was reported as a scientific claim of no transfer.  Every non-finite input
    # now produces TRANSFER_UNAVAILABLE naming the cell that failed.
    oracle = float(cell_c_skill)
    unavailable: list[str] = []
    if not np.isfinite(oracle):
        unavailable.append("cell_c refit_q_refit_a (the oracle) is not finite")
    if not np.isfinite(float(cell_a_gap)):
        unavailable.append("cell_a fully frozen gap is not finite")
    if not np.isfinite(float(cell_b_gap)):
        unavailable.append("cell_b frozen-Q gap is not finite")
    if unavailable:
        return {
            "label": TRANSFER_UNAVAILABLE,
            "unavailable_cells": unavailable,
            "reason": "; ".join(unavailable),
            "band_width": band.get("width"),
            "power_limited": None,
        }

    band_width = band.get("width", float("nan"))
    power_limited = bool(
        np.isfinite(float(band_width or float("nan")))
        and float(band_width) > WIDE_BAND_WIDTH
    )
    if oracle < float(min_oracle_skill):
        label = "NO_RECURRENCE"
    elif inside_band(cell_a_gap, band):
        label = "INTERFACE_AND_LAW_TRANSFER"
    elif inside_band(cell_b_gap, band):
        label = "INTERFACE_TRANSFERS_LAW_SPECIFIC"
    else:
        label = "STRUCTURE_RECURS_OBJECTS_DO_NOT"
    return {
        "label": label,
        "band_width": band_width,
        "band_replicates": band.get("n"),
        "power_limited": power_limited,
        "power_limited_note": (
            (
                f"the identical-distribution band spans {float(band_width):.4f}, wider than "
                f"{WIDE_BAND_WIDTH}, so a gap falling inside it reflects low discriminating "
                "power rather than demonstrated transfer"
            )
            if power_limited
            else None
        ),
    }


# --------------------------------------------------------------------------
# The full directional matrix
# --------------------------------------------------------------------------


def transfer_matrix(skill_by_pair: Mapping[tuple, float]) -> dict[str, object]:
    """Full (fitting group x evaluation group) skill matrix, both directions.

    ``skill_by_pair`` is keyed ``(fitting_group, evaluation_group)``.  Transfer
    need not be symmetric -- a chain fitted on a broad group can carry to a
    narrow one while the reverse fails -- so both off-diagonal entries are
    reported rather than averaged into one "similarity" number.  Missing pairs
    become ``nan`` and never raise, because an unevaluated pair is not a zero.

    Returns ``groups`` (sorted by string form, so mixed key types stay
    orderable), ``matrix`` as nested lists, ``diagonal`` (each group's
    fit-and-evaluate-on-itself skill), and ``asymmetry``, the maximum
    ``|M[i][j] - M[j][i]|`` over pairs where both directions are finite
    (``nan`` when no such pair exists).
    """

    keys = set()
    for pair in skill_by_pair:
        fitting, evaluation = pair
        keys.add(fitting)
        keys.add(evaluation)
    groups = sorted(keys, key=lambda group: (str(type(group)), str(group)))
    index_of = {group: position for position, group in enumerate(groups)}
    size = len(groups)
    matrix = np.full((size, size), np.nan, dtype=np.float64)
    for (fitting, evaluation), skill in skill_by_pair.items():
        matrix[index_of[fitting], index_of[evaluation]] = float(skill)

    asymmetries: list[float] = []
    for i in range(size):
        for j in range(i + 1, size):
            forward, backward = matrix[i, j], matrix[j, i]
            if np.isfinite(forward) and np.isfinite(backward):
                asymmetries.append(abs(float(forward) - float(backward)))

    return {
        "groups": groups,
        "matrix": [[float(value) for value in row] for row in matrix],
        "diagonal": [float(matrix[i, i]) for i in range(size)],
        "asymmetry": float(max(asymmetries)) if asymmetries else float("nan"),
    }
