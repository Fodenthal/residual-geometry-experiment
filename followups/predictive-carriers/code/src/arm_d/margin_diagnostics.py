"""Why the governing margin interval is wide: measure it, do not assert it.

The recorded section 14.3 margin picks each condition's strongest SOURCE-STATE
baseline once on the full sample and then bootstraps documents with those
choices held fixed.  That is the preregistered statistic and nothing here
changes it.

The report has been asserting a mechanism for the interval's width: main is
scored against the zero predictor while the floor conditions are scored against
persistence, so two of three conditions carry a separately estimated per-document
baseline whose noise the increment inherits.  This module turns that assertion
into a measurement.  It re-runs the same document bootstrap while carrying every
source-state baseline through the resampling, so it can report

* the joint distribution of the raw margin and the increment margin, and their
  covariance, rather than two intervals side by side;
* the exact additive decomposition
  ``increment_margin = raw_margin_against_the_same_floor - baseline_difference``
  and how much of the increment's variance the baseline difference contributes;
* how often document resampling changes which baseline wins, per condition and
  between conditions.  A maximum over several close, noisy estimators is not a
  smooth function of the sample, so the winner can switch; that is a quantity to
  report, not to assume away.

Everything here is diagnostic.  None of it is eligible to change a verdict, and
the labels say so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

CITATION = (
    "R1 section 14.3 content floor; diagnostic decomposition of its bootstrap, "
    "not a gate"
)
DIAGNOSTIC_ONLY = (
    "DIAGNOSTIC ONLY: this quantity explains the recorded interval and is not "
    "eligible to change the assigned rung"
)
SOURCE_STATE_NAMES: tuple[str, ...] = ("zero", "persistence", "update_only")
UNAVAILABLE = "MARGIN_DECOMPOSITION_UNAVAILABLE"
MEASURED = "MARGIN_DECOMPOSITION_MEASURED"


def _r2(numerator: np.ndarray, denominator: np.ndarray, picks: np.ndarray) -> float:
    """Document-weighted R-squared over a resampled document index."""

    total = float(np.sum(denominator[picks]))
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum(numerator[picks])) / total


def _align(
    conditions: Mapping[str, Mapping[str, object]],
) -> tuple[list[str], np.ndarray, dict[str, dict[str, object]]]:
    """Restrict every condition to the documents all of them share.

    The bootstrap is paired across conditions, so a replicate must resample one
    document set and score every condition on it.
    """

    names = sorted(conditions)
    shared: np.ndarray | None = None
    for name in names:
        labels = np.asarray(conditions[name]["document_labels"])
        shared = labels if shared is None else np.intersect1d(shared, labels)
    shared = np.asarray(shared if shared is not None else [])

    aligned: dict[str, dict[str, object]] = {}
    for name in names:
        entry = conditions[name]
        labels = np.asarray(entry["document_labels"])
        order = {label: index for index, label in enumerate(labels)}
        take = np.asarray([order[label] for label in shared], dtype=int)
        baselines = {
            baseline_name: {
                "numerator": np.asarray(payload["numerator"])[take],
                "denominator": np.asarray(payload["denominator"])[take],
            }
            for baseline_name, payload in entry["baselines"].items()
        }
        aligned[name] = {
            "operator_numerator": np.asarray(entry["operator_numerator"])[take],
            "operator_denominator": np.asarray(entry["operator_denominator"])[take],
            "baselines": baselines,
        }
    return names, shared, aligned


def _interval(draws: np.ndarray) -> dict[str, object]:
    finite = draws[np.isfinite(draws)]
    if finite.size < 2:
        return {
            "q05": float("nan"),
            "q95": float("nan"),
            "excludes_zero": False,
            "usable_replicates": int(finite.size),
        }
    low = float(np.quantile(finite, 0.05))
    high = float(np.quantile(finite, 0.95))
    return {
        "q05": low,
        "q95": high,
        "standard_deviation": float(np.std(finite, ddof=1)),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "positive_beyond_interval": bool(low > 0.0),
        "usable_replicates": int(finite.size),
    }


def margin_decomposition(
    conditions: Mapping[str, Mapping[str, object]],
    *,
    replicates: int = 2000,
    seed: int = 42,
    main: str = "main",
    source_state_names: Sequence[str] = SOURCE_STATE_NAMES,
) -> dict[str, object]:
    """Decompose the increment margin's bootstrap into its two moving parts.

    ``conditions`` maps a condition name to per-document operator numerators and
    denominators, ``document_labels``, and a ``baselines`` mapping from baseline
    name to its own per-document numerators and denominators.  Every source-state
    baseline is carried, so the winner is recomputed inside each replicate.
    """

    if main not in conditions:
        return {
            "label": UNAVAILABLE,
            "unavailable_because": f"the {main} condition is absent",
            "citation": CITATION,
            "status": DIAGNOSTIC_ONLY,
        }
    floors = sorted(name for name in conditions if name != main)
    if not floors:
        return {
            "label": UNAVAILABLE,
            "unavailable_because": "no floor condition was supplied",
            "citation": CITATION,
            "status": DIAGNOSTIC_ONLY,
        }

    names, shared, aligned = _align(conditions)
    n_documents = int(shared.size)
    if n_documents < 2:
        return {
            "label": UNAVAILABLE,
            "unavailable_because": (
                f"only {n_documents} documents are shared across conditions"
            ),
            "citation": CITATION,
            "status": DIAGNOSTIC_ONLY,
        }

    usable = [
        name
        for name in source_state_names
        if all(name in aligned[condition]["baselines"] for condition in names)
    ]
    if not usable:
        return {
            "label": UNAVAILABLE,
            "unavailable_because": (
                "no source-state baseline is present in every condition"
            ),
            "citation": CITATION,
            "status": DIAGNOSTIC_ONLY,
        }

    def evaluate(picks: np.ndarray) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for name in names:
            entry = aligned[name]
            operator = _r2(
                entry["operator_numerator"], entry["operator_denominator"], picks
            )
            scores = {
                baseline: _r2(
                    entry["baselines"][baseline]["numerator"],
                    entry["baselines"][baseline]["denominator"],
                    picks,
                )
                for baseline in usable
            }
            finite = {k: v for k, v in scores.items() if np.isfinite(v)}
            winner = max(finite, key=lambda k: finite[k]) if finite else usable[0]
            out[name] = {
                "operator": operator,
                "baseline_scores": scores,
                "winner": winner,
                "baseline": scores[winner],
                "increment": operator - scores[winner],
            }
        return out

    def margins(state: dict[str, dict[str, object]]) -> dict[str, object]:
        raw_floor = max(floors, key=lambda c: state[c]["operator"])
        inc_floor = max(floors, key=lambda c: state[c]["increment"])
        raw_margin = state[main]["operator"] - state[raw_floor]["operator"]
        increment_margin = state[main]["increment"] - state[inc_floor]["increment"]
        # Exact identity: against the SAME floor the increment margin is the raw
        # difference minus the difference of the two winning baselines.
        raw_same_floor = state[main]["operator"] - state[inc_floor]["operator"]
        baseline_difference = state[main]["baseline"] - state[inc_floor]["baseline"]
        return {
            "raw_margin": raw_margin,
            "increment_margin": increment_margin,
            "raw_against_increment_floor": raw_same_floor,
            "baseline_difference": baseline_difference,
            "raw_floor": raw_floor,
            "increment_floor": inc_floor,
        }

    observed_state = evaluate(np.arange(n_documents))
    observed = margins(observed_state)

    rng = np.random.default_rng(seed)
    raw_draws = np.empty(replicates, dtype=np.float64)
    inc_draws = np.empty(replicates, dtype=np.float64)
    same_floor_draws = np.empty(replicates, dtype=np.float64)
    bdiff_draws = np.empty(replicates, dtype=np.float64)
    winner_counts = {name: dict.fromkeys(usable, 0) for name in names}
    switched_from_observed = dict.fromkeys(names, 0)
    winners_differ_between = 0
    floor_choice_differs = 0

    for replicate in range(replicates):
        picks = rng.integers(0, n_documents, size=n_documents)
        state = evaluate(picks)
        drawn = margins(state)
        raw_draws[replicate] = drawn["raw_margin"]
        inc_draws[replicate] = drawn["increment_margin"]
        same_floor_draws[replicate] = drawn["raw_against_increment_floor"]
        bdiff_draws[replicate] = drawn["baseline_difference"]
        for name in names:
            winner = state[name]["winner"]
            winner_counts[name][winner] += 1
            if winner != observed_state[name]["winner"]:
                switched_from_observed[name] += 1
        if state[main]["winner"] != state[drawn["increment_floor"]]["winner"]:
            winners_differ_between += 1
        if drawn["raw_floor"] != drawn["increment_floor"]:
            floor_choice_differs += 1

    finite = np.isfinite(raw_draws) & np.isfinite(inc_draws)
    if int(finite.sum()) >= 2:
        covariance = float(np.cov(raw_draws[finite], inc_draws[finite])[0, 1])
        correlation = float(np.corrcoef(raw_draws[finite], inc_draws[finite])[0, 1])
    else:
        covariance = float("nan")
        correlation = float("nan")

    pair = np.isfinite(same_floor_draws) & np.isfinite(bdiff_draws)
    if int(pair.sum()) >= 2:
        var_raw_same = float(np.var(same_floor_draws[pair], ddof=1))
        var_bdiff = float(np.var(bdiff_draws[pair], ddof=1))
        cov_pair = float(np.cov(same_floor_draws[pair], bdiff_draws[pair])[0, 1])
        var_increment = var_raw_same + var_bdiff - 2.0 * cov_pair
        share = var_bdiff / var_increment if var_increment > 0 else float("nan")
    else:
        var_raw_same = var_bdiff = cov_pair = var_increment = share = float("nan")

    return {
        "label": MEASURED,
        "citation": CITATION,
        "status": DIAGNOSTIC_ONLY,
        "n_documents": n_documents,
        "replicates": int(replicates),
        "main_condition": main,
        "floor_conditions": floors,
        "source_state_baselines_carried": list(usable),
        "observed": {
            "raw_margin": float(observed["raw_margin"]),
            "increment_margin": float(observed["increment_margin"]),
            "raw_against_increment_floor": float(
                observed["raw_against_increment_floor"]
            ),
            "baseline_difference": float(observed["baseline_difference"]),
            "raw_floor": observed["raw_floor"],
            "increment_floor": observed["increment_floor"],
            "winner_by_condition": {
                name: observed_state[name]["winner"] for name in names
            },
            "baseline_scores_by_condition": {
                name: {k: float(v) for k, v in observed_state[name]["baseline_scores"].items()}
                for name in names
            },
        },
        "raw_margin_interval": _interval(raw_draws),
        "increment_margin_interval": _interval(inc_draws),
        "baseline_difference_interval": _interval(bdiff_draws),
        "covariance_raw_increment": covariance,
        "correlation_raw_increment": correlation,
        "variance_decomposition": {
            "variance_raw_against_same_floor": var_raw_same,
            "variance_baseline_difference": var_bdiff,
            "covariance": cov_pair,
            "variance_increment_margin": var_increment,
            "baseline_difference_share_of_increment_variance": share,
            "identity": (
                "increment_margin = raw_against_increment_floor - baseline_difference, "
                "so Var(increment) = Var(raw) + Var(baseline difference) - 2 Cov"
            ),
        },
        "winner_stability": {
            "counts_by_condition": {
                name: dict(winner_counts[name]) for name in names
            },
            "switch_fraction_by_condition": {
                name: switched_from_observed[name] / float(replicates)
                for name in names
            },
            "fraction_main_and_floor_win_different_baselines": (
                winners_differ_between / float(replicates)
            ),
            "fraction_raw_and_increment_choose_different_floor": (
                floor_choice_differs / float(replicates)
            ),
            "note": (
                "A maximum over several close, noisy baseline estimators is not a "
                "smooth function of the sample, so resampling documents can change "
                "which baseline wins. The recorded section 14.3 statistic fixes the "
                "winner on the full sample; these counts say how unstable that "
                "choice is."
            ),
        },
    }


def fixed_baseline_margins(
    conditions: Mapping[str, Mapping[str, object]],
    *,
    replicates: int = 2000,
    seed: int = 42,
    main: str = "main",
    source_state_names: Sequence[str] = SOURCE_STATE_NAMES,
) -> dict[str, object]:
    """The margin with one baseline held fixed across every condition.

    Holding the baseline fixed removes the maximum, and with it both the winner
    switching and the mismatch between a condition scored against zero and one
    scored against persistence.  Reported as a sensitivity: it is a different
    estimand from the preregistered margin and cannot change the verdict.
    """

    if main not in conditions:
        return {
            "label": UNAVAILABLE,
            "unavailable_because": f"the {main} condition is absent",
            "status": DIAGNOSTIC_ONLY,
        }
    names, shared, aligned = _align(conditions)
    floors = sorted(name for name in names if name != main)
    n_documents = int(shared.size)
    if n_documents < 2 or not floors:
        return {
            "label": UNAVAILABLE,
            "unavailable_because": "too few shared documents or no floor condition",
            "status": DIAGNOSTIC_ONLY,
        }

    everything = np.arange(n_documents)
    out: dict[str, object] = {}
    for baseline in source_state_names:
        if not all(baseline in aligned[name]["baselines"] for name in names):
            out[baseline] = {
                "label": UNAVAILABLE,
                "unavailable_because": f"{baseline} is missing in some condition",
            }
            continue

        def increment(name: str, picks: np.ndarray, baseline=baseline) -> float:
            entry = aligned[name]
            operator = _r2(
                entry["operator_numerator"], entry["operator_denominator"], picks
            )
            reference = _r2(
                entry["baselines"][baseline]["numerator"],
                entry["baselines"][baseline]["denominator"],
                picks,
            )
            return operator - reference

        observed = increment(main, everything) - max(
            increment(name, everything) for name in floors
        )
        rng = np.random.default_rng(seed)
        draws = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            picks = rng.integers(0, n_documents, size=n_documents)
            draws[replicate] = increment(main, picks) - max(
                increment(name, picks) for name in floors
            )
        out[baseline] = {
            "margin": float(observed),
            "interval": _interval(draws),
        }

    return {
        "label": "FIXED_BASELINE_SENSITIVITY_MEASURED",
        "status": DIAGNOSTIC_ONLY,
        "citation": CITATION,
        "n_documents": n_documents,
        "replicates": int(replicates),
        "by_baseline": out,
        "reading": (
            "Each row scores every condition against the same source-state baseline, "
            "so no maximum is taken and no condition inherits a different baseline's "
            "noise. These are sensitivities, not the preregistered statistic, which "
            "section 15.1 defines over each condition's strongest source-state "
            "baseline."
        ),
    }


def document_support_curve(
    *,
    margin: float,
    q05: float,
    q95: float,
    n_documents: int,
    targets: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> dict[str, object]:
    """How many documents this margin needs for its interval to clear zero.

    The bootstrap interval gives a standard error at the observed support; the
    error of a document-weighted mean falls as one over the square root of the
    document count, so the support at which a 5-95 interval just reaches zero is
    ``n * (1.645 * se / |margin|)^2``.  This holds the point estimate fixed, which
    is exactly why it must not be applied to a pooled margin that averages over a
    sign change: there is no single estimand for it to converge to.
    """

    if not np.isfinite(margin) or not np.isfinite(q05) or not np.isfinite(q95):
        return {"label": UNAVAILABLE, "unavailable_because": "non-finite inputs"}
    standard_error = (q95 - q05) / (2.0 * 1.6448536269514722)
    if standard_error <= 0.0 or margin == 0.0:
        return {
            "label": UNAVAILABLE,
            "unavailable_because": "degenerate standard error or zero margin",
        }

    required = float(n_documents) * (1.6448536269514722 * standard_error / abs(margin)) ** 2
    already = bool(q05 > 0.0 or q95 < 0.0)
    return {
        "label": "DOCUMENT_SUPPORT_CURVE_MEASURED",
        "status": DIAGNOSTIC_ONLY,
        "margin": float(margin),
        "margin_sign": "positive" if margin > 0 else "negative",
        "standard_error_at_observed_support": float(standard_error),
        "observed_documents": int(n_documents),
        "documents_required_to_exclude_zero": required,
        "interval_already_excludes_zero": already,
        "separation_target": (
            "hold the positive interval away from zero"
            if margin > 0
            else "separate the observed negative margin from zero"
        ),
        "documents_at_multiples_of_observed": {
            f"{multiple:g}x": {
                "documents": float(n_documents) * multiple,
                "expected_half_width": float(
                    1.6448536269514722
                    * standard_error
                    / np.sqrt(multiple)
                ),
                "would_exclude_zero": bool(
                    abs(margin)
                    > 1.6448536269514722 * standard_error / np.sqrt(multiple)
                ),
            }
            for multiple in targets
        },
        "assumption": (
            "the point estimate is held at its observed value and the standard "
            "error falls as one over the square root of the document count"
        ),
    }
