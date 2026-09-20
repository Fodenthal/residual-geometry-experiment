"""How similar the matched-content donors are to the prefix they replaced.

R1.1 amendment 5a asked whether the matched-content condition behaves as a
content floor.  The first implementation could not answer that question, and its
answer has been withdrawn:

    THE FIRST AUDIT MEASURED THE WRONG WINDOW.  It compared
    ``target[t - L : t]`` against ``spliced[t - L : t]``.  That span is the
    SHARED SUFFIX, which the splice invariant guarantees identical between the
    two sequences -- ``verify_pair`` asserts exactly that.  The audit therefore
    compared the identical-by-construction region against itself, which is why
    every surface statistic came back at exactly 1.0 with q05 = q95 = 1.0, and
    why the embedding path reported NaN over zero pairs.  The resulting
    ``DONORS_NEAR_DUPLICATE`` label carried no information about the donors, and
    it contradicted the capture data: positional identity of 1.0 would mean the
    matched condition *is* the no-op condition, yet matched sits at 0.9328 of
    the main condition's difference norm while no-op sits at 0.0.

The replaced region is the remote prefix ``[0, boundary)`` where
``boundary = t - L``.  ``build_spliced_tokens`` writes ``donor[:boundary]`` over
``target[:boundary]``, and ``verify_pair`` separately asserts that those two
spans DIFFER.  This module audits that span, refuses to run on a span that is
identical by construction, and reports the matched donors against an
unrelated-prefix reference distribution built from the main condition's donors.

A single absolute similarity number cannot answer the question.  "Matched donors
have median type Jaccard 0.31" means nothing until it is compared with what an
arbitrary same-stratum donor scores on the same statistic, because token-type
overlap between any two English documents is already substantial.  The verdict
is therefore relative, and the main condition supplies the reference.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

UNAVAILABLE = "DONOR_AUDIT_UNAVAILABLE"
INVALID_IDENTICAL_SPAN = "DONOR_AUDIT_INVALID_IDENTICAL_SPAN"
CONTENT_MATCHED = "DONORS_CONTENT_MATCHED"
WEAKLY_CONTENT_MATCHED = "DONORS_WEAKLY_CONTENT_MATCHED"
NOT_CONTENT_MATCHED = "DONORS_NOT_CONTENT_MATCHED"

# The withdrawn label, kept only so that a reader of an old artifact can find
# out what happened to it.  Never emitted.
WITHDRAWN_LABELS = {
    "DONORS_NEAR_DUPLICATE": (
        "withdrawn: produced by auditing the shared suffix, which is identical "
        "between the two sequences by construction, so the statistic could only "
        "ever return 1.0"
    )
}


def token_overlap(replaced: Sequence[int], donor: Sequence[int]) -> dict[str, float]:
    """Surface overlap between the replaced remote prefix and the donor prefix.

    Reports the Jaccard index over token *types*, the fraction of donor tokens
    whose type also occurs in the replaced prefix, and the fraction of positions
    holding the identical token.  Positional identity is the strictest of the
    three and is the one that exposes the window bug: on the correct span it is
    near zero, and on the shared suffix it is exactly 1.0.
    """

    left = np.asarray(list(replaced), dtype=np.int64)
    right = np.asarray(list(donor), dtype=np.int64)
    left_types, right_types = set(left.tolist()), set(right.tolist())
    union = left_types | right_types
    intersection = left_types & right_types
    width = int(min(left.size, right.size))
    return {
        "type_jaccard": float(len(intersection) / len(union)) if union else 0.0,
        "donor_token_coverage": (
            float(np.mean([token in left_types for token in right.tolist()])) if right.size else 0.0
        ),
        "positional_identity": float(np.mean(left[:width] == right[:width])) if width else 0.0,
        "n_replaced_tokens": int(left.size),
        "n_donor_tokens": int(right.size),
    }


def _distribution(pairs: Sequence[Mapping[str, object]], *, name: str) -> dict[str, object]:
    """Summarise one condition's donor similarity, refusing an identical span.

    Raises rather than returning a verdict when the two spans coincide.  An
    identical span means the caller passed a region the splice invariant holds
    fixed, and every statistic computed on it is a tautology.  Failing loudly is
    the whole point: the previous silent 1.0 survived review precisely because it
    looked like a measurement.
    """

    overlaps = [token_overlap(row["replaced_tokens"], row["donor_tokens"]) for row in pairs]
    identical = [
        index
        for index, row in enumerate(pairs)
        if np.array_equal(
            np.asarray(list(row["replaced_tokens"]), dtype=np.int64),
            np.asarray(list(row["donor_tokens"]), dtype=np.int64),
        )
    ]
    if identical:
        raise ValueError(
            f"{len(identical)} of {len(pairs)} {name} donor pairs have an IDENTICAL "
            f"replaced/donor span (first at index {identical[0]}). The audit window is "
            "wrong: it is measuring a region the splice invariant holds fixed, not the "
            "replaced remote prefix [0, boundary). Refusing to report a verdict."
        )
    jaccard = np.array([row["type_jaccard"] for row in overlaps], dtype=np.float64)
    coverage = np.array([row["donor_token_coverage"] for row in overlaps], dtype=np.float64)
    positional = np.array([row["positional_identity"] for row in overlaps], dtype=np.float64)
    return {
        "n_pairs": int(len(pairs)),
        "median_type_jaccard": float(np.median(jaccard)),
        "q05_type_jaccard": float(np.quantile(jaccard, 0.05)),
        "q95_type_jaccard": float(np.quantile(jaccard, 0.95)),
        "median_donor_token_coverage": float(np.median(coverage)),
        "median_positional_identity": float(np.median(positional)),
        "max_positional_identity": float(np.max(positional)),
        "_jaccard": jaccard,
    }


def _rank_probability(treatment: np.ndarray, reference: np.ndarray) -> float:
    """P(a matched pair scores above an unrelated pair), ties counted as half.

    A rank statistic rather than a difference of medians, because the two
    distributions have different spreads and the question is about separation,
    not location.  0.5 means the matched donors are indistinguishable from
    arbitrary same-stratum donors.
    """

    if treatment.size == 0 or reference.size == 0:
        return float("nan")
    comparisons = treatment[:, None] - reference[None, :]
    wins = float(np.count_nonzero(comparisons > 0))
    ties = float(np.count_nonzero(comparisons == 0))
    return (wins + 0.5 * ties) / float(treatment.size * reference.size)


def audit_matched_donors(
    matched_pairs: Sequence[Mapping[str, object]],
    reference_pairs: Sequence[Mapping[str, object]] | None = None,
    *,
    content_matched_probability: float = 0.75,
    not_matched_probability: float = 0.55,
) -> dict[str, object]:
    """Classify the matched-content condition against an unrelated-prefix reference.

    ``matched_pairs`` and ``reference_pairs`` each carry ``replaced_tokens`` and
    ``donor_tokens`` for the REPLACED REMOTE PREFIX ``[0, boundary)``.  The
    reference is the main condition's donors: same construction, same stratum,
    no content matching.  Without it no verdict is issued, because an absolute
    overlap number cannot distinguish content matching from the base rate of
    token overlap between two English documents.

    The embedding-cosine path of the previous implementation has been REMOVED
    rather than left returning NaN.  It fed a disjunctive rule, so a NaN made the
    rule silently one-sided; the previous run reported a cosine over zero pairs.
    ``embedding_similarity_available`` records its absence explicitly.
    """

    if not matched_pairs:
        return {"label": UNAVAILABLE, "n_pairs": 0, "reason": "no matched donor pairs supplied"}

    matched = _distribution(matched_pairs, name="matched")
    if not reference_pairs:
        return {
            "label": UNAVAILABLE,
            "reason": (
                "no unrelated-prefix reference distribution was supplied, and an absolute "
                "overlap number cannot separate content matching from the base rate of "
                "token overlap between arbitrary documents"
            ),
            "audit_window": "replaced remote prefix [0, boundary)",
            "matched": {k: v for k, v in matched.items() if not k.startswith("_")},
            "embedding_similarity_available": False,
        }

    reference = _distribution(reference_pairs, name="main (unrelated-prefix reference)")
    probability = _rank_probability(matched["_jaccard"], reference["_jaccard"])
    separation = matched["median_type_jaccard"] - reference["median_type_jaccard"]
    above_reference_tail = float(
        np.mean(matched["_jaccard"] > reference["q95_type_jaccard"])
    )

    if not np.isfinite(probability):
        label = UNAVAILABLE
        reading = "the rank comparison was not computable"
    elif probability >= content_matched_probability:
        label = CONTENT_MATCHED
        reading = (
            "The matched donors are measurably more similar to the replaced prefix than "
            "arbitrary same-stratum donors are, so the condition does carry content "
            "information. Its difference norm sitting close to the main condition is "
            "then informative about how much of the remote effect survives content "
            "matching, and is reported as such."
        )
    elif probability <= not_matched_probability:
        label = NOT_CONTENT_MATCHED
        reading = (
            "The matched donors are not distinguishable from arbitrary same-stratum "
            "donors on this statistic. The condition is a second unrelated-prefix arm "
            "rather than a content floor, and the sham and no-op conditions remain the "
            "binding floors."
        )
    else:
        label = WEAKLY_CONTENT_MATCHED
        reading = (
            "Matched donors are slightly more similar to the replaced prefix than "
            "arbitrary donors, but the separation is small. The condition is reported as "
            "a weak content floor and is not treated as a clean one."
        )

    return {
        "label": label,
        "reading": reading,
        "audit_window": "replaced remote prefix [0, boundary), boundary = source_position - suffix_length",
        "withdrawn_label": "DONORS_NEAR_DUPLICATE",
        "withdrawn_because": WITHDRAWN_LABELS["DONORS_NEAR_DUPLICATE"],
        "probability_matched_more_similar_than_unrelated": probability,
        "median_type_jaccard_separation": float(separation),
        "fraction_matched_above_reference_q95": above_reference_tail,
        "matched": {k: v for k, v in matched.items() if not k.startswith("_")},
        "unrelated_reference": {k: v for k, v in reference.items() if not k.startswith("_")},
        "embedding_similarity_available": False,
        "embedding_similarity_note": (
            "the embedding-cosine metric was removed rather than reported as NaN: it fed a "
            "disjunctive label rule, so a missing value silently made the rule one-sided"
        ),
        "thresholds": {
            "content_matched_probability": content_matched_probability,
            "not_matched_probability": not_matched_probability,
            "statistic": "P(matched pair type-Jaccard > unrelated pair type-Jaccard), ties half",
        },
    }
