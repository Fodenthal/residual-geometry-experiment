"""Splice-validity diagnostics: is the effect content, or is it push?

R1.1 amendment 10.  The splice is what buys causal isolation of remote history:
every token after the boundary is held fixed, so a difference at the source
position can only have come through the state.  The price is that the spliced
run is off-manifold, and a generic "context mismatch" response has its own
structured relaxation dynamics.  Averaging does not remove it, because the bias
term ``E[eps_{t+k} Delta_z_t^T] Sigma^-1`` need not vanish.

This module carries the diagnostics that separate the two readings using data
the confirmatory capture already paid for.

The central one is the intervention-strength confound.  The four conditions
differ in how hard they push as well as in what they push with: mean
destination difference norms run main 36.04, matched 33.61, sham 19.81, no-op 0,
and the scores track that ordering closely.  A content-floor margin computed by
pooling every evaluation row therefore compares conditions at different
intervention magnitudes.  ``norm_stratified_margin`` compares them at MATCHED
magnitude instead, by binning evaluation rows on ``||Delta z_t||`` and rescoring
inside each band.

Two design points are load-bearing.

The band edges come from the pooled distribution across the pushing conditions,
not from each condition separately.  Per-condition deciles would put main's
weakest rows and sham's strongest rows in the same nominal band while their
actual magnitudes differ by a factor of two, which is the confound restated
rather than removed.

A band with too few rows or too few documents returns an explicit UNAVAILABLE
naming what was missing, and never a number.  The no-op condition pushes with
magnitude exactly zero by construction, so it cannot appear in a matched-norm
comparison at all; its floor stays zero and the code says so rather than
silently treating band 0 as a comparison.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from arm_d.baselines import persistence
from arm_d.scoring import document_index, equal_document_r2

UNAVAILABLE = "SPLICE_VALIDITY_UNAVAILABLE"
SURVIVES = "CONTENT_MARGIN_SURVIVES_NORM_MATCHING"
DOES_NOT_SURVIVE = "CONTENT_MARGIN_DOES_NOT_SURVIVE_NORM_MATCHING"
AMBIGUOUS = "CONTENT_MARGIN_AMBIGUOUS_AT_MATCHED_NORM"

# Section 15.1 source-state baselines.  The trajectory baselines are deliberately
# absent: section 14.3 defines the content floor over the source-state increment,
# and importing a trajectory baseline here would repeat the D1/D5 confusion that
# amendment 9 withdrew.
SOURCE_STATE_BASELINES = ("zero", "persistence")

CITATION = "R1 section 14.3 content floor; R1.1 amendment 10 splice validity"


def _r2_from_documents(
    numerator: np.ndarray, denominator: np.ndarray, picks: np.ndarray
) -> float:
    """Equal-document-weight R^2 over a bootstrap resample of documents."""

    num = float(np.mean(numerator[picks]))
    den = float(np.mean(denominator[picks]))
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0:
        return float("nan")
    return 1.0 - num / den


def band_edges(values: np.ndarray, n_bands: int = 10) -> np.ndarray:
    """Quantile edges over the pooled magnitudes of the pushing conditions."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite intervention magnitudes to band")
    quantiles = np.linspace(0.0, 1.0, int(n_bands) + 1)
    edges = np.quantile(finite, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _band_of(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def _source_state_scores(entry: Mapping[str, object], rows: np.ndarray) -> dict[str, object]:
    """Operator and source-state baseline scores on a subset of evaluation rows.

    Predictions come from the fitted operator handed back by ``run_cell`` and
    from the public baseline predictors, so nothing is re-implemented here.
    """

    zs = np.asarray(entry["source_projected"], dtype=np.float64)[rows]
    zd = np.asarray(entry["destination_projected"], dtype=np.float64)[rows]
    zh = np.asarray(entry["history_projected"], dtype=np.float64)[rows]
    documents = np.asarray(entry["documents"])[rows]
    operator = np.asarray(entry["operator"], dtype=np.float64)

    operator_score = equal_document_r2(zd, zs @ operator.T, documents)
    candidates = {
        "zero": equal_document_r2(zd, np.zeros_like(zd), documents),
        "persistence": equal_document_r2(zd, persistence(zh), documents),
    }
    best_name = max(candidates, key=lambda name: candidates[name].r2)
    best = candidates[best_name]
    return {
        "operator": operator_score,
        "baseline": best,
        "baseline_name": best_name,
        "n_rows": int(rows.sum() if rows.dtype == bool else rows.size),
        "n_documents": int(np.unique(documents).size),
    }


def implied_noise_floor(
    condition_scores: Mapping[str, float],
    difference_norms: Mapping[str, float],
    *,
    main: str = "main",
) -> dict[str, object]:
    """Back out the residual unpredictability each condition carries.

    Under ``R2 = S / (S + N)`` with signal ``S`` proportional to the squared
    difference norm, ``N = S (1 - R2) / R2``.  A condition whose implied floor is
    far BELOW main's is carrying an extra component that is largely
    unpredictable rather than structured, which is evidence against the
    mismatch-recovery-dynamics reading.  The crude prediction column anchors the
    noise floor on ``main`` and asks what each condition's score would be if the
    only thing that differed were how hard it pushed.

    The model is deliberately crude and is reported as such: it exists to be
    compared against the norm-matched measurement, not to replace it.
    """

    rows: dict[str, dict[str, float]] = {}
    main_floor = float("nan")
    for name, r2 in condition_scores.items():
        norm = float(difference_norms.get(name, float("nan")))
        signal = norm * norm
        if not np.isfinite(r2) or not np.isfinite(signal) or r2 <= 0.0 or signal <= 0.0:
            rows[name] = {
                "r2": float(r2),
                "signal": float(signal),
                "implied_noise_floor": float("nan"),
            }
            continue
        floor = signal * (1.0 - r2) / r2
        rows[name] = {"r2": float(r2), "signal": float(signal), "implied_noise_floor": float(floor)}
        if name == main:
            main_floor = floor

    for name, row in rows.items():
        signal = row["signal"]
        if np.isfinite(main_floor) and np.isfinite(signal) and signal + main_floor > 0:
            row["predicted_r2_at_main_noise_floor"] = float(signal / (signal + main_floor))
            row["observed_minus_predicted"] = float(row["r2"] - row["predicted_r2_at_main_noise_floor"])
        else:
            row["predicted_r2_at_main_noise_floor"] = float("nan")
            row["observed_minus_predicted"] = float("nan")
        row["noise_floor_relative_to_main"] = (
            float(row["implied_noise_floor"] / main_floor)
            if np.isfinite(main_floor) and main_floor > 0 and np.isfinite(row["implied_noise_floor"])
            else float("nan")
        )

    unpredictable = [
        name
        for name, row in rows.items()
        if name != main and np.isfinite(row["noise_floor_relative_to_main"])
        and row["noise_floor_relative_to_main"] < 0.75
    ]
    usable = [
        row
        for row in rows.values()
        if isinstance(row, Mapping) and np.isfinite(float(row.get("implied_noise_floor", np.nan)))
    ]
    label = "NOISE_FLOOR_MEASURED" if len(usable) >= 2 else UNAVAILABLE
    return {
        "label": label,
        "reason": (
            None
            if label == "NOISE_FLOOR_MEASURED"
            else "fewer than two conditions had a finite difference norm and score"
        ),
        "citation": CITATION,
        "model": "R2 = S/(S+N) with S proportional to the squared difference norm",
        "model_is_crude": True,
        "per_condition": rows,
        "conditions_with_lower_noise_floor_than_main": sorted(unpredictable),
        "reading": (
            "A condition whose implied noise floor is well below main's carries an extra "
            "component that is largely unpredictable rather than structured, which argues "
            "against a structured mismatch-recovery explanation of its score."
            if unpredictable
            else "No condition's implied noise floor sits materially below main's."
        ),
    }


def norm_stratified_margin(
    captures: Mapping[str, Mapping[str, object]],
    *,
    main: str = "main",
    floors: Sequence[str] = ("matched", "sham"),
    zero_magnitude_conditions: Sequence[str] = ("noop",),
    n_bands: int = 10,
    min_rows: int = 24,
    min_documents: int = 8,
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Recompute the content-floor margin at matched intervention strength.

    ``captures`` maps a condition name to the ``capture`` dict ``run_cell``
    returns.  Bands are cut on the pooled ``||Delta z_t||`` of ``main`` and the
    floor conditions; each condition is then rescored inside each band, so the
    comparison holds intervention magnitude fixed and varies only what the
    intervention pushed with.
    """

    missing = [name for name in (main, *floors) if name not in captures]
    if missing:
        return {
            "label": UNAVAILABLE,
            "citation": CITATION,
            "reason": f"no capture for condition(s): {', '.join(sorted(missing))}",
        }

    pooled = np.concatenate(
        [np.asarray(captures[name]["source_norm"], dtype=np.float64) for name in (main, *floors)]
    )
    edges = band_edges(pooled, n_bands=n_bands)

    per_band: list[dict[str, object]] = []
    usable_margins: list[float] = []
    rng = np.random.default_rng(seed)

    for band in range(len(edges) - 2 + 1):
        if band >= n_bands:
            break
        entries: dict[str, dict[str, object]] = {}
        shortfalls: dict[str, str] = {}
        for name in (main, *floors):
            norms = np.asarray(captures[name]["source_norm"], dtype=np.float64)
            rows = _band_of(norms, edges) == band
            if rows.sum() < min_rows:
                shortfalls[name] = f"{int(rows.sum())} rows < {min_rows}"
                continue
            scored = _source_state_scores(captures[name], rows)
            if scored["n_documents"] < min_documents:
                shortfalls[name] = f"{scored['n_documents']} documents < {min_documents}"
                continue
            entries[name] = scored

        record: dict[str, object] = {
            "band": int(band),
            "magnitude_lower": float(edges[band]) if np.isfinite(edges[band]) else None,
            "magnitude_upper": float(edges[band + 1]) if np.isfinite(edges[band + 1]) else None,
            "conditions_scored": sorted(entries),
        }
        if main not in entries or not any(name in entries for name in floors):
            record["label"] = UNAVAILABLE
            record["missing"] = shortfalls
            per_band.append(record)
            continue

        present_floors = [name for name in floors if name in entries]
        increments = {
            name: entries[name]["operator"].r2 - entries[name]["baseline"].r2
            for name in (main, *present_floors)
        }
        margin = increments[main] - max(increments[name] for name in present_floors)

        # Paired document bootstrap inside the band.  Documents are resampled
        # once per replicate and applied to every condition, so the draw carries
        # the correlation between conditions that share a document.
        documents = {
            name: document_index(np.asarray(captures[name]["documents"])[
                _band_of(np.asarray(captures[name]["source_norm"], dtype=np.float64), edges) == band
            ])[0]
            for name in (main, *present_floors)
        }
        shared = sorted(set.intersection(*(set(value.tolist()) for value in documents.values())))
        draws = np.full(replicates, np.nan, dtype=np.float64)
        if len(shared) >= min_documents:
            aligned: dict[str, dict[str, np.ndarray]] = {}
            for name in (main, *present_floors):
                labels = documents[name]
                keep = np.array([np.where(labels == doc)[0][0] for doc in shared], dtype=int)
                aligned[name] = {
                    "operator_numerator": np.asarray(entries[name]["operator"].numerator_by_document)[keep],
                    "operator_denominator": np.asarray(entries[name]["operator"].denominator_by_document)[keep],
                    "baseline_numerator": np.asarray(entries[name]["baseline"].numerator_by_document)[keep],
                    "baseline_denominator": np.asarray(entries[name]["baseline"].denominator_by_document)[keep],
                }
            for replicate in range(replicates):
                picks = rng.integers(0, len(shared), size=len(shared))
                drawn = {
                    name: _r2_from_documents(
                        aligned[name]["operator_numerator"], aligned[name]["operator_denominator"], picks
                    )
                    - _r2_from_documents(
                        aligned[name]["baseline_numerator"], aligned[name]["baseline_denominator"], picks
                    )
                    for name in (main, *present_floors)
                }
                draws[replicate] = drawn[main] - max(drawn[name] for name in present_floors)

        finite = draws[np.isfinite(draws)]
        interval = (
            {
                "q05": float(np.quantile(finite, 0.05)),
                "q95": float(np.quantile(finite, 0.95)),
                "positive_beyond_interval": bool(float(np.quantile(finite, 0.05)) > 0.0),
                "usable_replicates": int(finite.size),
                "shared_documents": len(shared),
            }
            if finite.size >= 2
            else {
                "q05": float("nan"),
                "q95": float("nan"),
                "positive_beyond_interval": False,
                "usable_replicates": int(finite.size),
                "shared_documents": len(shared),
            }
        )

        record.update(
            {
                "label": "BAND_MEASURED",
                "margin": float(margin),
                "margin_interval": interval,
                "increments": {name: float(value) for name, value in increments.items()},
                "operator_r2": {
                    name: float(entries[name]["operator"].r2) for name in (main, *present_floors)
                },
                "baseline_used": {
                    name: entries[name]["baseline_name"] for name in (main, *present_floors)
                },
                "n_rows": {name: entries[name]["n_rows"] for name in (main, *present_floors)},
                "mean_magnitude": {
                    name: float(
                        np.mean(
                            np.asarray(captures[name]["source_norm"], dtype=np.float64)[
                                _band_of(
                                    np.asarray(captures[name]["source_norm"], dtype=np.float64), edges
                                )
                                == band
                            ]
                        )
                    )
                    for name in (main, *present_floors)
                },
            }
        )
        if shortfalls:
            record["missing"] = shortfalls
        usable_margins.append(float(margin))
        per_band.append(record)

    # How much do the conditions even overlap in magnitude?  If a floor condition
    # pushes so much more weakly that it shares almost no band with main, then no
    # matched-magnitude comparison against it is possible at this design, and
    # saying so is more honest than reporting a margin over the thin overlap.
    overlap: dict[str, object] = {}
    for name in (main, *floors):
        norms = np.asarray(captures[name]["source_norm"], dtype=np.float64)
        bands_here = _band_of(norms, edges)
        bands_with_this_condition = {
            int(row["band"])
            for row in per_band
            if row.get("label") == "BAND_MEASURED" and name in (row.get("conditions_scored") or [])
        }
        # Rows in bands where THIS condition was actually scored.  Counting rows
        # in bands measured for other conditions would credit a condition with
        # overlap it does not have.
        shared_rows = np.isin(bands_here, sorted(bands_with_this_condition))
        overlap[name] = {
            "rows_in_bands_where_scored": float(np.mean(shared_rows)) if norms.size else float("nan"),
            "bands_present": sorted(bands_with_this_condition),
            "median_magnitude": float(np.median(norms)) if norms.size else float("nan"),
        }
    thin = [
        name
        for name in floors
        if isinstance(overlap.get(name), dict)
        and len(overlap[name]["bands_present"]) < 2
    ]

    measured = [row for row in per_band if row.get("label") == "BAND_MEASURED"]
    if not measured:
        return {
            "label": UNAVAILABLE,
            "citation": CITATION,
            "reason": "no magnitude band had enough rows and documents in main and a floor condition",
            "per_band": per_band,
            "magnitude_overlap": overlap,
            "n_bands": n_bands,
        }

    equalized = float(np.mean(usable_margins))
    positive_bands = sum(1 for row in measured if float(row["margin"]) > 0.0)
    excluding_zero = sum(
        1 for row in measured if bool(row["margin_interval"]["positive_beyond_interval"])
    )

    if excluding_zero >= max(2, len(measured) // 2) and equalized > 0.0:
        label = SURVIVES
        reading = (
            "The content-floor margin holds when conditions are compared at matched "
            "intervention magnitude, so the main-versus-matched separation is not "
            "explained by how hard the splice pushed."
        )
    elif equalized <= 0.0 or positive_bands <= len(measured) // 3:
        label = DOES_NOT_SURVIVE
        reading = (
            "The content-floor margin does not survive matching on intervention "
            "magnitude. The main-versus-matched separation then reflects how hard the "
            "intervention pushed rather than what it pushed with, and the D1 content "
            "floor must be read with that stated."
        )
    else:
        label = AMBIGUOUS
        reading = (
            "The margin stays positive on average at matched magnitude but few bands "
            "exclude zero on their own, so the norm-matched comparison neither confirms "
            "nor refutes the content reading at this power."
        )

    return {
        "label": label,
        "citation": CITATION,
        "reading": reading,
        "equalized_margin": equalized,
        "pooled_margin_is_band_weighted": True,
        "bands_measured": len(measured),
        "bands_total": int(n_bands),
        "bands_with_positive_margin": positive_bands,
        "bands_excluding_zero": excluding_zero,
        "per_band": per_band,
        "magnitude_overlap": overlap,
        "floors_too_weak_to_compare_at_matched_magnitude": thin,
        "zero_magnitude_conditions": {
            name: "pushes with magnitude zero by construction, so it cannot enter a "
            "matched-magnitude comparison; its floor stays zero"
            for name in zero_magnitude_conditions
        },
    }


def washout_curve(
    per_length: Mapping[int, Mapping[str, object]],
    *,
    main: str = "main",
    floors: Sequence[str] = ("matched", "sham"),
) -> dict[str, object]:
    """Score, magnitude and content margin as functions of suffix length.

    Every suffix length was captured for every condition, so this costs no new
    forward passes.  A decaying curve on its own does not discriminate: both a
    boundary shock and a genuine remote-history component decay as more real
    target tokens intervene.  What discriminates is the TIMESCALE, which is why
    this is reported next to the recovery curve rather than alone.
    """

    rows: list[dict[str, object]] = []
    for length in sorted(per_length):
        entry = per_length[length]
        scores = {name: float(value) for name, value in (entry.get("scores") or {}).items()}
        norms = {name: float(value) for name, value in (entry.get("difference_norms") or {}).items()}
        present = [name for name in floors if name in scores]
        margin = (
            scores[main] - max(scores[name] for name in present)
            if main in scores and present
            else float("nan")
        )
        rows.append(
            {
                "suffix_length": int(length),
                "scores": scores,
                "difference_norms": norms,
                "margin": float(margin),
                "margin_interval": entry.get("margin_interval"),
            }
        )

    finite = [row for row in rows if np.isfinite(row["margin"])]
    if len(finite) < 2:
        return {
            "label": UNAVAILABLE,
            "citation": CITATION,
            "reason": "fewer than two suffix lengths produced a finite margin",
            "per_length": rows,
        }

    lengths = np.array([row["suffix_length"] for row in finite], dtype=np.float64)
    main_scores = np.array([row["scores"].get(main, np.nan) for row in finite], dtype=np.float64)
    usable = np.isfinite(main_scores)
    slope = (
        float(np.polyfit(np.log2(lengths[usable]), main_scores[usable], 1)[0])
        if usable.sum() >= 2
        else float("nan")
    )

    return {
        "label": "WASHOUT_CURVE_MEASURED",
        "citation": CITATION,
        "per_length": rows,
        "main_score_slope_per_doubling": slope,
        "decays_with_suffix_length": bool(np.isfinite(slope) and slope < 0.0),
        "interpretation_requires_recovery_curve": True,
        "reading": (
            "Both a boundary shock and a genuine remote-history component decay as the "
            "suffix grows, so this curve constrains the timescale rather than settling "
            "the reading; it is interpretable only next to the loss-recovery curve."
        ),
    }


STATE_PERSISTS = "STATE_PERSISTS_BEYOND_LOSS_RECOVERY"
STATE_DECAYS_WITH_LOSS = "BOUNDARY_SHOCK_NOT_EXCLUDED"


def _by_document(values: np.ndarray, documents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean per document at each offset, and the document labels."""

    labels, codes = document_index(np.asarray(documents))
    out = np.full((labels.size, values.shape[1]), np.nan, dtype=np.float64)
    for index in range(labels.size):
        rows = codes == index
        if rows.any():
            out[index] = np.nanmean(values[rows], axis=0)
    return out, labels


def recovery_crossings(
    offsets: np.ndarray,
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    documents: Mapping[str, np.ndarray],
    *,
    main: str = "main",
    floor: str = "noop",
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Where does each curve become indistinguishable from the no-op floor?

    ``metrics`` maps a metric name to ``{condition: array (rows, offsets)}``.
    For each metric the main condition's per-document mean is compared against
    the no-op condition's at every offset, and the crossing is the first offset
    whose document-bootstrap interval includes zero.

    The verdict compares the crossing of the next-token excess loss against the
    crossing of the aperture-projected state difference.  Loss recovering while
    the projected state difference is still separated is the picture a remote
    state component predicts.  The two recovering together does not exclude a
    boundary shock, and is reported as such rather than as support.
    """

    window = np.asarray(offsets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    per_metric: dict[str, object] = {}

    for metric, by_condition in metrics.items():
        if main not in by_condition or floor not in by_condition:
            per_metric[metric] = {
                "label": UNAVAILABLE,
                "reason": f"missing condition for {metric}: need {main} and {floor}",
            }
            continue
        main_doc, main_labels = _by_document(
            np.asarray(by_condition[main], dtype=np.float64), documents[main]
        )
        floor_doc, floor_labels = _by_document(
            np.asarray(by_condition[floor], dtype=np.float64), documents[floor]
        )
        shared = np.intersect1d(main_labels, floor_labels)
        if shared.size < 8:
            per_metric[metric] = {
                "label": UNAVAILABLE,
                "reason": f"only {int(shared.size)} documents shared between {main} and {floor}",
            }
            continue
        main_rows = np.array([np.where(main_labels == doc)[0][0] for doc in shared])
        floor_rows = np.array([np.where(floor_labels == doc)[0][0] for doc in shared])
        difference = main_doc[main_rows] - floor_doc[floor_rows]

        observed = np.nanmean(difference, axis=0)
        draws = np.empty((replicates, window.size), dtype=np.float64)
        for replicate in range(replicates):
            picks = rng.integers(0, shared.size, size=shared.size)
            draws[replicate] = np.nanmean(difference[picks], axis=0)
        q05 = np.nanquantile(draws, 0.05, axis=0)
        q95 = np.nanquantile(draws, 0.95, axis=0)
        separated = (q05 > 0.0) | (q95 < 0.0)

        crossing = None
        for index in range(window.size):
            if not separated[index] and bool(separated[:index].any()):
                crossing = int(window[index])
                break

        per_metric[metric] = {
            "label": "RECOVERY_MEASURED",
            "offsets": window.tolist(),
            "observed_difference_from_floor": [float(value) for value in observed],
            "q05": [float(value) for value in q05],
            "q95": [float(value) for value in q95],
            "separated_from_floor": [bool(value) for value in separated],
            "crossing_offset": crossing,
            "still_separated_at_last_offset": bool(separated[-1]),
            "documents": int(shared.size),
        }

    loss = per_metric.get("excess_nll", {})
    state = per_metric.get("projected_difference_norm", {})
    if not isinstance(loss, dict) or not isinstance(state, dict):
        return {"label": UNAVAILABLE, "citation": CITATION, "per_metric": per_metric}
    if loss.get("label") != "RECOVERY_MEASURED" or state.get("label") != "RECOVERY_MEASURED":
        return {
            "label": UNAVAILABLE,
            "citation": CITATION,
            "reason": "the loss curve and the projected-state curve were not both measured",
            "per_metric": per_metric,
        }

    loss_crossing = loss.get("crossing_offset")
    state_crossing = state.get("crossing_offset")
    state_outlasts_loss = bool(
        loss_crossing is not None
        and (state_crossing is None or int(state_crossing) > int(loss_crossing))
    )

    # Scale-free fallback.  The no-op floor is identically zero by construction,
    # so "statistically indistinguishable from the floor" is a criterion neither
    # curve can meet, and both crossings come back null.  Comparing how far each
    # curve has decayed towards its own peak is well defined regardless.
    decay = {
        name: _relative_decay(row.get("offsets") or [], row.get("observed_difference_from_floor") or [])
        for name, row in (("excess_nll", loss), ("projected_difference_norm", state))
    }
    neither_reached_floor = loss_crossing is None and state_crossing is None
    loss_quarter = decay["excess_nll"]["offset_at_quarter_peak"]
    state_quarter = decay["projected_difference_norm"]["offset_at_quarter_peak"]
    state_outlasts_by_decay = bool(
        loss_quarter is not None
        and (state_quarter is None or int(state_quarter) > int(loss_quarter))
    )

    if neither_reached_floor:
        loss_remaining = decay["excess_nll"]["fraction_of_peak_at_last_offset"]
        state_remaining = decay["projected_difference_norm"]["fraction_of_peak_at_last_offset"]
        label = STATE_PERSISTS if state_outlasts_by_decay else STATE_DECAYS_WITH_LOSS
        reading = (
            "Neither curve reaches the no-op floor inside the measured window, so the "
            "preregistered crossing comparison is not available: the floor is "
            "identically zero by construction and no curve can become indistinguishable "
            "from it. Compared instead on decay towards their own peaks, next-token loss "
            f"falls to {loss_remaining:.4f} of its peak by the last offset while the "
            f"aperture-projected state difference falls only to {state_remaining:.4f}. "
            + (
                "The state difference therefore outlasts local surprise, which a boundary "
                "shock that had relaxed would not produce."
                if state_outlasts_by_decay
                else "The state difference decays at least as fast as local surprise, so a "
                "boundary shock is not excluded."
            )
        )
        return {
            "label": label,
            "citation": CITATION,
            "reading": reading,
            "loss_recovery_offset": loss_crossing,
            "state_recovery_offset": state_crossing,
            "state_outlasts_loss": state_outlasts_by_decay,
            "crossing_comparison_available": False,
            "crossing_unavailable_because": (
                "the no-op floor is identically zero, so no curve can become "
                "statistically indistinguishable from it"
            ),
            "verdict_rests_on": "relative_decay",
            "relative_decay": decay,
            "per_metric": per_metric,
        }

    if state_outlasts_loss:
        label = STATE_PERSISTS
        reading = (
            "Next-token loss returns to the no-op floor while the aperture-projected "
            "state difference is still separated from it. A boundary shock that had "
            "relaxed would not leave that residue, so the persistent component is not "
            "explained by local surprise recovery alone."
        )
    else:
        label = STATE_DECAYS_WITH_LOSS
        reading = (
            "The projected state difference decays no more slowly than next-token loss "
            "recovers, so a boundary shock with its own relaxation dynamics is not "
            "excluded as the driver of the effect."
        )

    return {
        "label": label,
        "citation": CITATION,
        "reading": reading,
        "loss_recovery_offset": loss_crossing,
        "state_recovery_offset": state_crossing,
        "state_outlasts_loss": state_outlasts_loss,
        "crossing_comparison_available": True,
        "verdict_rests_on": "crossing",
        "relative_decay": decay,
        "per_metric": per_metric,
    }


def _relative_decay(offsets, values, *, quarter: float = 0.25, tenth: float = 0.10) -> dict:
    """How far a curve has fallen towards its own peak, and where.

    Scale-free, so a loss in nats and a norm in activation units are comparable.
    """

    offsets = [int(value) for value in offsets]
    values = [float(value) for value in values]
    if not offsets or len(offsets) != len(values):
        return {
            "label": UNAVAILABLE,
            "reason": "no curve to measure",
            "fraction_of_peak": [],
            "offset_at_quarter_peak": None,
            "offset_at_tenth_peak": None,
            "fraction_of_peak_at_last_offset": float("nan"),
        }
    peak = max(abs(value) for value in values)
    if not np.isfinite(peak) or peak <= 0:
        return {
            "label": UNAVAILABLE,
            "reason": "the curve has no positive peak to normalise by",
            "fraction_of_peak": [],
            "offset_at_quarter_peak": None,
            "offset_at_tenth_peak": None,
            "fraction_of_peak_at_last_offset": float("nan"),
        }
    fractions = [value / peak for value in values]

    def _first_below(threshold: float):
        for offset, fraction in zip(offsets, fractions):
            if fraction < threshold:
                return int(offset)
        return None

    return {
        "label": "RELATIVE_DECAY_MEASURED",
        "peak": peak,
        "offsets": offsets,
        "fraction_of_peak": fractions,
        "offset_at_quarter_peak": _first_below(quarter),
        "offset_at_tenth_peak": _first_below(tenth),
        "fraction_of_peak_at_last_offset": fractions[-1],
    }
