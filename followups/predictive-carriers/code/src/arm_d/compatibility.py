"""Donor compatibility as a stratifier (R1.1 amendment 10 item 3).

The matched condition was built to hold document content roughly fixed while
replacing the exact remote history.  Whether it succeeded is a matter of degree,
and that degree is measurable per pair: how much of the replaced remote prefix's
vocabulary the donor's prefix reproduces.  Freezing that as a compatibility
score ``c(A, B)`` turns a binary condition label into a continuous stratifier.

Two uses, and the second is the sharper one.

Within magnitude bands, asking whether the effect survives among the most
compatible donors separates "the model noticed the content changed" from "the
model's state carries the specific history".  A plain per-stratum comparison,
though, cannot test the prediction that distinguishes them.  Mismatch recovery
is an ASYMMETRIC hypothesis: a law fitted where donors are incompatible is
fitted partly on mismatch-response dynamics, and should generalise poorly to
compatible pairs that provoke little mismatch, while the reverse direction need
not fail the same way.  Running compatibility through the frozen-transfer
machinery as a stratum yields the full fitting-by-evaluation matrix, and that
matrix's asymmetry is the quantity the hypothesis speaks to.

The scores are computed from the frozen token pool and the frozen splice plan,
so no forward pass is needed and no cap applies: every target document gets a
score, not the 512 the donor audit samples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from arm_d.donor_audit import token_overlap

UNAVAILABLE = "COMPATIBILITY_UNAVAILABLE"
CITATION = "R1 section 50 frozen transfer; R1.1 amendment 10 item 3"

# The compatibility score is the type Jaccard between the replaced remote prefix
# and the donor prefix that took its place.  Coverage and positional identity
# are recorded alongside it, but Jaccard is the score used for stratification:
# it is symmetric in the two spans and does not reward a donor merely for being
# longer, which coverage does.
SCORE_KEY = "type_jaccard"


def pair_compatibility(
    target_tokens: np.ndarray,
    spliced_tokens: np.ndarray,
    boundary: int,
) -> dict[str, float]:
    """Similarity between the replaced remote prefix and what replaced it.

    The window is ``[0, boundary)`` -- the region the splice actually changed.
    Scoring ``[boundary, source_position)`` would measure a region the splice
    invariant holds fixed and return a tautology.
    """

    if boundary <= 0:
        raise ValueError(f"invalid splice boundary {boundary}")
    replaced = np.asarray(target_tokens, dtype=np.int64)[:boundary]
    donor = np.asarray(spliced_tokens, dtype=np.int64)[:boundary]
    if np.array_equal(replaced, donor):
        raise ValueError(
            "the replaced and donor spans are identical, so this window is not the "
            "one the splice changed; refusing to score it"
        )
    overlap = token_overlap(replaced, donor)
    return {key: float(value) for key, value in overlap.items()}


def compatibility_by_document(
    tokens: np.ndarray,
    context_ids: Sequence[str],
    specs: Sequence[object],
    *,
    condition: str,
    build_spliced,
) -> dict[str, dict[str, float]]:
    """Compatibility per target document for one condition, at one suffix length.

    ``specs`` must already be filtered to the suffix length of interest; a
    document appearing at several lengths would otherwise get several scores and
    the last would silently win.
    """

    index_of = {str(cid): row for row, cid in enumerate(context_ids)}
    scores: dict[str, dict[str, float]] = {}
    for spec in specs:
        if getattr(spec, "condition", None) != condition:
            continue
        target_id = str(getattr(spec, "target_id"))
        if target_id in scores:
            raise ValueError(
                f"document {target_id} appears twice for condition {condition}; filter "
                "the specs to a single suffix length before scoring"
            )
        target = tokens[index_of[target_id]]
        donor = tokens[index_of[str(getattr(spec, "donor_id"))]]
        spliced = build_spliced(target, donor, spec)
        scores[target_id] = pair_compatibility(target, spliced, int(spec.boundary))
    return scores


def compatibility_labels(
    scores: Mapping[str, Mapping[str, float]],
    documents: Sequence[str],
    *,
    n_groups: int = 2,
    score_key: str = SCORE_KEY,
    min_spread: float = 1e-6,
    min_per_group: int = 20,
) -> dict[str, object]:
    """Split documents into compatibility groups, or refuse to.

    Returns ``labels`` aligned to ``documents`` with -1 where a document has no
    score.  A stratification whose groups do not actually differ in the quantity
    they are named for is not a stratification, so a degenerate spread returns
    UNAVAILABLE rather than a split that reads as one.
    """

    values = np.array(
        [float(scores[str(doc)][score_key]) if str(doc) in scores else np.nan for doc in documents],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    if finite.size < n_groups * min_per_group:
        return {
            "label": UNAVAILABLE,
            "citation": CITATION,
            "reason": (
                f"{int(finite.size)} scored documents cannot support {n_groups} groups of "
                f"at least {min_per_group}"
            ),
        }
    spread = float(np.nanmax(finite) - np.nanmin(finite))
    if spread < min_spread:
        return {
            "label": UNAVAILABLE,
            "citation": CITATION,
            "reason": (
                f"compatibility scores span only {spread:.3e}; the groups would not differ "
                "in the quantity they are named for"
            ),
            "score_spread": spread,
        }

    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_groups + 1)[1:-1])
    labels = np.full(values.size, -1, dtype=np.int64)
    scored = np.isfinite(values)
    labels[scored] = np.searchsorted(edges, values[scored], side="right")

    groups = []
    for group in range(n_groups):
        rows = labels == group
        groups.append(
            {
                "group": int(group),
                "documents": int(rows.sum()),
                "median_score": float(np.median(values[rows])) if rows.any() else float("nan"),
                "min_score": float(np.min(values[rows])) if rows.any() else float("nan"),
                "max_score": float(np.max(values[rows])) if rows.any() else float("nan"),
            }
        )
    smallest = min(row["documents"] for row in groups)
    if smallest < min_per_group:
        return {
            "label": UNAVAILABLE,
            "citation": CITATION,
            "reason": f"smallest compatibility group holds {smallest} documents",
            "groups": groups,
        }

    medians = [row["median_score"] for row in groups]
    return {
        "label": "COMPATIBILITY_STRATIFIED",
        "citation": CITATION,
        "score_key": score_key,
        "labels": labels,
        "groups": groups,
        "score_spread": spread,
        "median_gap_low_to_high": float(medians[-1] - medians[0]),
        "reading": (
            "group 0 holds the least compatible donors and group "
            f"{n_groups - 1} the most; median score rises from {medians[0]:.4f} to "
            f"{medians[-1]:.4f}"
        ),
    }


def asymmetry_reading(matrix: Mapping[str, object]) -> dict[str, object]:
    """Interpret a compatibility fitting-by-evaluation matrix.

    Mismatch recovery predicts that a law fitted on incompatible donors, where
    the mismatch response is strongest, transfers poorly to compatible donors,
    while the reverse direction survives better.  The symmetric case does not
    support that reading and is reported as not supporting it, rather than as
    evidence for the alternative.
    """

    asymmetry = matrix.get("asymmetry")
    if asymmetry is None or not np.isfinite(float(asymmetry)):
        return {
            "label": UNAVAILABLE,
            "citation": CITATION,
            "reason": "no donor-compatibility pair had both transfer directions finite",
        }
    value = float(asymmetry)
    return {
        "label": "COMPATIBILITY_TRANSFER_ASYMMETRIC"
        if abs(value) > 0.0
        else "COMPATIBILITY_TRANSFER_SYMMETRIC",
        "citation": CITATION,
        "asymmetry": value,
        "reading": (
            "The two fitting directions differ, which is the shape a mismatch-recovery "
            "component predicts; the sign says which population carries the "
            "population-specific part."
            if abs(value) > 0.0
            else "Transfer is symmetric between compatibility groups, which does not "
            "support a mismatch-recovery component concentrated in one of them."
        ),
    }


def compatibility_within_bands(
    capture: Mapping[str, object],
    scores: Mapping[str, Mapping[str, float]],
    *,
    n_bands: int = 6,
    n_groups: int = 2,
    score_key: str = SCORE_KEY,
    min_rows: int = 24,
) -> dict[str, object]:
    """Does the effect survive among the most compatible donors, at fixed push?

    Item 3(a).  Compatibility and intervention magnitude are not independent -- a
    donor whose prefix resembles the replaced one displaces the state less -- so
    comparing compatibility groups without holding magnitude fixed would restate
    the magnitude confound in another variable.  Both are conditioned on here.

    This is the weaker of item 3's two analyses and is reported as such: it
    cannot test the asymmetric prediction, which is what the frozen-transfer
    matrix over the same groups is for.
    """

    from arm_d.validity import UNAVAILABLE as VALIDITY_UNAVAILABLE
    from arm_d.validity import _band_of, _source_state_scores, band_edges

    documents = [str(value) for value in np.asarray(capture["documents"])]
    magnitudes = np.asarray(capture["source_norm"], dtype=np.float64)
    labelled = compatibility_labels(scores, documents, n_groups=n_groups, min_per_group=1)
    if labelled["label"] != "COMPATIBILITY_STRATIFIED":
        return {"label": UNAVAILABLE, "citation": CITATION, "reason": labelled.get("reason")}
    groups = np.asarray(labelled["labels"])

    edges = band_edges(magnitudes, n_bands=n_bands)
    bands = _band_of(magnitudes, edges)

    rows: list[dict[str, object]] = []
    for band in range(n_bands):
        for group in range(n_groups):
            selected = np.flatnonzero((bands == band) & (groups == group))
            if selected.size < min_rows:
                rows.append(
                    {
                        "band": int(band),
                        "compatibility_group": int(group),
                        "label": VALIDITY_UNAVAILABLE,
                        "n_rows": int(selected.size),
                    }
                )
                continue
            scored = _source_state_scores(capture, selected)
            rows.append(
                {
                    "band": int(band),
                    "compatibility_group": int(group),
                    "label": "BAND_GROUP_MEASURED",
                    "operator_r2": float(scored["operator"].r2),
                    "baseline_r2": float(scored["baseline"].r2),
                    "baseline_name": str(scored["baseline_name"]),
                    "increment_over_baseline": float(scored["operator"].r2)
                    - float(scored["baseline"].r2),
                    "n_rows": int(selected.size),
                    "mean_magnitude": float(np.mean(magnitudes[selected])),
                    "median_compatibility": float(
                        np.median(
                            [
                                float(scores[documents[row]][score_key])
                                for row in selected
                                if documents[row] in scores
                            ]
                        )
                    ),
                }
            )

    measured = [row for row in rows if row.get("label") == "BAND_GROUP_MEASURED"]
    if not measured:
        return {
            "label": UNAVAILABLE,
            "citation": CITATION,
            "reason": "no band held enough rows in both compatibility groups",
            "per_band_group": rows,
        }

    # Within each band, is the effect present in the MOST compatible group?
    high = [row for row in measured if row["compatibility_group"] == n_groups - 1]
    low = [row for row in measured if row["compatibility_group"] == 0]
    high_increment = float(np.mean([row["increment_over_baseline"] for row in high])) if high else float("nan")
    low_increment = float(np.mean([row["increment_over_baseline"] for row in low])) if low else float("nan")
    survives = bool(np.isfinite(high_increment) and high_increment > 0.0)

    return {
        "label": "COMPATIBILITY_BANDS_MEASURED",
        "citation": CITATION,
        "per_band_group": rows,
        "mean_increment_most_compatible": high_increment,
        "mean_increment_least_compatible": low_increment,
        "effect_present_among_most_compatible_donors": survives,
        "groups": labelled["groups"],
        "reading": (
            "Conditioning on intervention magnitude, the operator still beats its "
            "source-state baseline among the most compatible donors."
            if survives
            else "Among the most compatible donors the operator does not beat its "
            "source-state baseline once magnitude is held fixed."
        ),
    }
