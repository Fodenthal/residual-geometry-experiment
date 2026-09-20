"""The R1 section 14.3 splice floor margin, with the bootstrap interval it requires.

Section 14.3 defines

    M_splice = dR2_pred(main) - max over c in {no-op, sham, matched} dR2_pred(c)

and requires ``M_splice`` positive BEYOND its document-bootstrap interval. The
confirmatory run recorded the point estimate and never computed the interval, so
the D1 content-floor condition was being read off a number with no uncertainty
attached.

Two quantities are reported side by side, because they answer different questions
and it must be visible when they disagree:

* the RAW margin, on per-condition operator R-squared;
* the dR2 margin that section 14.3 actually specifies, on each condition's
  increment over ITS OWN strongest source-state baseline (section 15.1).

These coincide exactly when the zero predictor is the strongest source-state
baseline in every condition, which happens whenever persistence and the
update-only comparator both score at or below zero. That is a fact to verify per
condition and record, not to assume: if some condition has a positive persistence
score, the two margins genuinely differ and the section 14.3 quantity governs.

The bootstrap is PAIRED across conditions. Every condition is measured on the
same spliced documents, so resampling documents independently per condition would
break that pairing and inflate the interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

SECTION_14_3 = "R1 section 14.3 (splice floor margin, positive beyond its bootstrap interval)"
SECTION_15_1 = "R1 section 15.1 (the source-state baselines set the D1 headline)"
FLOOR_CONDITIONS = ("noop", "sham", "matched")


def _r2_from_documents(
    numerators: np.ndarray, denominators: np.ndarray, picks: np.ndarray
) -> float:
    """Equal-document R-squared over a (possibly resampled) set of documents."""

    denominator = float(np.sum(denominators[picks]))
    if denominator <= 0:
        return float("nan")
    return 1.0 - float(np.sum(numerators[picks])) / denominator


def _aligned(
    conditions: Mapping[str, Mapping[str, object]]
) -> tuple[list[str], np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Restrict every condition to the documents all of them share, in one order."""

    label_sets = []
    for payload in conditions.values():
        labels = np.asarray(payload["document_labels"]).reshape(-1)
        label_sets.append(set(labels.tolist()))
    shared = sorted(set.intersection(*label_sets)) if label_sets else []
    shared_array = np.asarray(shared)

    aligned: dict[str, dict[str, np.ndarray]] = {}
    for name, payload in conditions.items():
        labels = np.asarray(payload["document_labels"]).reshape(-1)
        order = {label: index for index, label in enumerate(labels.tolist())}
        rows = np.asarray([order[label] for label in shared], dtype=np.int64)
        aligned[name] = {
            "operator_numerator": np.asarray(payload["operator_numerator"])[rows],
            "operator_denominator": np.asarray(payload["operator_denominator"])[rows],
            "baseline_numerator": np.asarray(payload["baseline_numerator"])[rows],
            "baseline_denominator": np.asarray(payload["baseline_denominator"])[rows],
        }
    return list(conditions), shared_array, aligned


def splice_floor_margin(
    conditions: Mapping[str, Mapping[str, object]],
    *,
    baseline_names: Mapping[str, str] | None = None,
    replicates: int = 2000,
    seed: int = 42,
    main: str = "main",
) -> dict[str, object]:
    """Both margins with paired document-bootstrap intervals.

    ``conditions`` maps a condition name to per-document numerators and
    denominators for the fitted operator and for that condition's strongest
    SOURCE-STATE baseline, plus ``document_labels``.
    """

    if main not in conditions:
        return {
            "label": "SPLICE_MARGIN_UNAVAILABLE",
            "unavailable_because": f"the {main} condition is absent",
            "citation": SECTION_14_3,
        }
    floors = [name for name in conditions if name != main]
    if not floors:
        return {
            "label": "SPLICE_MARGIN_UNAVAILABLE",
            "unavailable_because": "no floor condition was supplied",
            "citation": SECTION_14_3,
        }

    names, shared, aligned = _aligned(conditions)
    n_documents = int(shared.size)
    if n_documents < 2:
        return {
            "label": "SPLICE_MARGIN_UNAVAILABLE",
            "unavailable_because": (
                f"only {n_documents} documents are shared across conditions, so no "
                "document bootstrap is possible"
            ),
            "citation": SECTION_14_3,
        }

    everything = np.arange(n_documents)

    def scores(picks: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        raw: dict[str, float] = {}
        increment: dict[str, float] = {}
        for name in names:
            entry = aligned[name]
            operator = _r2_from_documents(
                entry["operator_numerator"], entry["operator_denominator"], picks
            )
            baseline = _r2_from_documents(
                entry["baseline_numerator"], entry["baseline_denominator"], picks
            )
            raw[name] = operator
            increment[name] = operator - baseline
        return raw, increment

    observed_raw, observed_increment = scores(everything)
    raw_margin = observed_raw[main] - max(observed_raw[name] for name in floors)
    increment_margin = observed_increment[main] - max(
        observed_increment[name] for name in floors
    )

    rng = np.random.default_rng(seed)
    raw_draws = np.empty(replicates, dtype=np.float64)
    increment_draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        picks = rng.integers(0, n_documents, size=n_documents)
        drawn_raw, drawn_increment = scores(picks)
        raw_draws[replicate] = drawn_raw[main] - max(drawn_raw[name] for name in floors)
        increment_draws[replicate] = drawn_increment[main] - max(
            drawn_increment[name] for name in floors
        )

    def interval(draws: np.ndarray) -> dict[str, object]:
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
            "excludes_zero": bool(low > 0.0 or high < 0.0),
            "positive_beyond_interval": bool(low > 0.0),
            "usable_replicates": int(finite.size),
        }

    raw_interval = interval(raw_draws)
    increment_interval = interval(increment_draws)

    # The two quantities coincide exactly when every condition's strongest
    # source-state baseline scores zero, which is the zero predictor winning.
    zero_baselines = {
        name: bool(
            abs(
                _r2_from_documents(
                    aligned[name]["baseline_numerator"],
                    aligned[name]["baseline_denominator"],
                    everything,
                )
            )
            < 1e-12
        )
        for name in names
    }
    agree_in_sign = bool(np.sign(raw_margin) == np.sign(increment_margin))
    agree_on_exclusion = bool(
        raw_interval["excludes_zero"] == increment_interval["excludes_zero"]
    )

    verdict = bool(increment_interval.get("positive_beyond_interval", False))
    return {
        "label": "SPLICE_MARGIN_MEASURED",
        "citation": SECTION_14_3,
        "baseline_rule_citation": SECTION_15_1,
        "n_documents": n_documents,
        "replicates": int(replicates),
        "main_condition": main,
        "floor_conditions": sorted(floors),
        "strongest_source_state_baseline_by_condition": (
            dict(baseline_names) if baseline_names else None
        ),
        "baseline_scores_zero_by_condition": zero_baselines,
        "raw_r2_by_condition": observed_raw,
        "delta_r2_by_condition": observed_increment,
        "raw_margin": float(raw_margin),
        "raw_margin_interval": raw_interval,
        "delta_r2_margin": float(increment_margin),
        "delta_r2_margin_interval": increment_interval,
        "quantities_agree_in_sign": agree_in_sign,
        "quantities_agree_on_excluding_zero": agree_on_exclusion,
        "quantities_disagree": bool(not (agree_in_sign and agree_on_exclusion)),
        # Section 14.3 defines the margin over dR2_pred, so that is the quantity
        # the verdict uses. The raw margin is reported beside it, never instead.
        "verdict_uses": "delta_r2_margin",
        "clears_content_floor": verdict,
        "reading": (
            "the two quantities disagree, either in sign or in whether their interval "
            "excludes zero; section 14.3 defines the margin over the increment, so the "
            "increment governs, and the disagreement is reported rather than resolved "
            "silently"
            if not (agree_in_sign and agree_on_exclusion)
            else "the raw and increment margins agree, which is expected here because the "
            "zero predictor is the strongest source-state baseline in every condition, so "
            "the increment equals the operator score"
            if all(zero_baselines.values())
            else "the raw and increment margins agree in sign and in whether they exclude "
            "zero"
        ),
    }


def scores_from_baselines(
    *,
    operator_score,
    baseline_results: Mapping[str, object],
    source_state_names: Sequence[str] = ("zero", "persistence", "update_only"),
) -> tuple[dict[str, object], str]:
    """Package one condition for :func:`splice_floor_margin`.

    Picks that condition's strongest SOURCE-STATE baseline (R1 section 15.1) and
    returns its per-document numerators and denominators alongside the operator's.
    """

    applicable = {
        name: result
        for name, result in baseline_results.items()
        if name in tuple(source_state_names)
    }
    if not applicable:
        raise ValueError(f"no source-state baseline among {tuple(source_state_names)}")
    best_name = max(applicable, key=lambda name: float(applicable[name].score.r2))
    best = applicable[best_name].score
    return (
        {
            "operator_numerator": np.asarray(operator_score.numerator_by_document),
            "operator_denominator": np.asarray(operator_score.denominator_by_document),
            "baseline_numerator": np.asarray(best.numerator_by_document),
            "baseline_denominator": np.asarray(best.denominator_by_document),
            "document_labels": np.asarray(operator_score.document_labels),
        },
        best_name,
    )
