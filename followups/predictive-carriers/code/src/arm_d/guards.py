"""Guards for comparisons that are only meaningful between distinct things.

A similarity, overlap, transfer or calibration number answers a *relative*
question: how close is A to B.  When A and B are the same object, every such
number takes its extremal value for reasons that have nothing to do with the
science, and the extremal value is exactly what a strong result looks like.  The
donor audit in this run shipped such a number as a finding.

These helpers make that case loud instead of silent.  They are deliberately
small and deliberately used at the call site rather than inside a metric, so the
refusal names the two populations in the caller's own vocabulary.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "IdenticalPopulationsError",
    "require_distinct_arrays",
    "require_distinct_labels",
    "distinctness_report",
]


class IdenticalPopulationsError(AssertionError):
    """Raised when a relative comparison is asked for between one thing and itself."""


def _as_array(value: object) -> np.ndarray:
    return np.asarray(value)


def require_distinct_arrays(
    left: object,
    right: object,
    *,
    what: str,
    left_name: str = "left",
    right_name: str = "right",
) -> None:
    """Refuse a comparison between two numerically identical spans.

    Identity is checked exactly rather than by tolerance: the failure this
    guards against is a caller passing the same region twice, which produces bit
    equality, not near equality.  A tolerance test would additionally reject
    genuinely close populations, which are a legitimate scientific finding.
    """

    first = _as_array(left)
    second = _as_array(right)
    if first.shape != second.shape:
        return
    if first.size == 0:
        raise IdenticalPopulationsError(
            f"{what}: {left_name} and {right_name} are both empty, so any "
            "similarity between them is undefined rather than high."
        )
    if np.array_equal(first, second):
        raise IdenticalPopulationsError(
            f"{what}: {left_name} and {right_name} are element-wise identical "
            f"(shape {first.shape}). A similarity between a span and itself takes "
            "its extremal value by construction and cannot answer a relative "
            "question. Check that the intended comparison window was passed."
        )


def require_distinct_labels(
    left: object,
    right: object,
    *,
    what: str,
    left_name: str = "left",
    right_name: str = "right",
) -> None:
    """Refuse a comparison between two identical label sequences or identifiers.

    Used where the two populations are named rather than numeric, for example a
    source and destination layer, or two stratification schemes.
    """

    # Sequences are checked first: a list compares equal to a list, so the scalar
    # branch would otherwise claim the whole sequence is "both <repr>", which is
    # true but unreadable for anything longer than a couple of entries.
    left_seq = list(left) if isinstance(left, (list, tuple, np.ndarray)) else None
    right_seq = list(right) if isinstance(right, (list, tuple, np.ndarray)) else None
    if left_seq is not None and right_seq is not None:
        if left_seq == right_seq:
            raise IdenticalPopulationsError(
                f"{what}: {left_name} and {right_name} are the same sequence of "
                f"{len(left_seq)} entries, so the comparison is against itself."
            )
        return
    if left == right:
        raise IdenticalPopulationsError(
            f"{what}: {left_name} and {right_name} are both {left!r}. Comparing a "
            "population against itself reports identity, not agreement."
        )


def distinctness_report(
    left: object,
    right: object,
    *,
    what: str,
) -> dict[str, object]:
    """Non-raising variant, for a payload that must record that the check was made.

    A diagnostic that silently omits the check is indistinguishable from one that
    made it and passed, so the report is written into the artifact.
    """

    first = _as_array(left)
    second = _as_array(right)
    same_shape = first.shape == second.shape
    identical = bool(same_shape and first.size and np.array_equal(first, second))
    return {
        "comparison": what,
        "shapes": [list(first.shape), list(second.shape)],
        "same_shape": same_shape,
        "populations_are_identical": identical,
        "both_empty": bool(first.size == 0 and second.size == 0),
        "interpretable": bool(not identical and first.size and second.size),
        "note": (
            "identical by construction: any similarity is extremal for structural "
            "reasons and carries no evidence"
            if identical
            else "populations differ, so a similarity between them is interpretable"
        ),
    }
