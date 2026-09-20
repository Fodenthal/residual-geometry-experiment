"""Attention-routing diagnostic for the causal battery (R1.1 amendment 4).

Linear-interface arguments cover what a position *encodes*: attention OV
circuits and the unembedding read their input linearly, so a feature consumed
downstream has to be linearly decodable.  They do not cover attention *scores*,
which are ``score(t, s) = h_t^T W_Q^T W_K h_s`` -- a bilinear form across two
positions.

That gap matters for a causal claim.  If remote sequence memory is implemented
by transporting content forward, a patch at the source should move the
destination's content and leave the attention pattern roughly alone.  If it is
implemented by routing -- the patched state changes *where* the destination
reads from -- the attention rows at the evaluated positions move too.  The
destination coordinate shift already measured cannot separate those two
mechanisms, so this module measures the attention side of it.

The diagnostic is descriptive.  It does not enter the claim ladder and gates
nothing; it says which mechanism the observed causal effect looks like, not
whether the effect exists.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

ROUTING_MEDIATED = "ROUTING_MEDIATED"
CONTENT_MEDIATED = "CONTENT_MEDIATED"
MIXED = "MIXED"


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    """Renormalize attention rows so each sums to one over the key axis."""

    total = np.sum(rows, axis=-1, keepdims=True)
    return rows / np.clip(total, 1e-12, None)


def attention_row_l1(clean: np.ndarray, patched: np.ndarray) -> np.ndarray:
    """Per-head L1 distance between attention rows, halved to lie in [0, 1].

    Both inputs have shape ``(rows, heads, keys)``.  Returns ``(rows, heads)``.
    """

    left = _normalize_rows(np.asarray(clean, dtype=np.float64))
    right = _normalize_rows(np.asarray(patched, dtype=np.float64))
    return 0.5 * np.sum(np.abs(left - right), axis=-1)


def attention_row_jensen_shannon(clean: np.ndarray, patched: np.ndarray) -> np.ndarray:
    """Per-head Jensen-Shannon divergence between attention rows (base 2).

    Bounded in [0, 1], and unlike the L1 distance it is insensitive to mass
    moving between two keys that were already both near zero.  Reporting both
    guards against reading a routing change off one metric's quirk.
    """

    left = _normalize_rows(np.asarray(clean, dtype=np.float64))
    right = _normalize_rows(np.asarray(patched, dtype=np.float64))
    mean = 0.5 * (left + right)

    def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        safe = np.where(p > 0, p, 1.0)
        ratio = np.where(p > 0, np.log2(safe / np.clip(q, 1e-12, None)), 0.0)
        return np.sum(np.where(p > 0, p * ratio, 0.0), axis=-1)

    return 0.5 * _kl(left, mean) + 0.5 * _kl(right, mean)


def routing_diagnostic(
    clean_attention: np.ndarray,
    patched_attention: np.ndarray,
    noop_attention: np.ndarray | None,
    destination_shift: float,
    noop_destination_shift: float,
    *,
    routing_floor_multiple: float = 3.0,
) -> dict[str, object]:
    """Compare the attention movement against the no-op floor.

    Every quantity is referenced against the no-op patch, which is the same
    machinery applied with no change.  Without that reference an attention
    distance is uninterpretable: some movement is just numerical noise from
    re-running the forward pass.

    ``routing_floor_multiple`` is how far above the no-op floor a change has to
    sit before it counts as real movement.  A condition counts as showing
    routing when the attention rows move that far, and as showing content
    transport when the destination coordinates do; both together are ``MIXED``.
    """

    l1 = attention_row_l1(clean_attention, patched_attention)
    jsd = attention_row_jensen_shannon(clean_attention, patched_attention)

    if noop_attention is not None:
        floor_l1 = float(np.max(attention_row_l1(clean_attention, noop_attention)))
        floor_jsd = float(np.max(attention_row_jensen_shannon(clean_attention, noop_attention)))
    else:
        floor_l1 = 0.0
        floor_jsd = 0.0

    mean_l1 = float(np.mean(l1))
    mean_jsd = float(np.mean(jsd))
    attention_moved = mean_l1 > routing_floor_multiple * max(floor_l1, 1e-6)
    content_moved = abs(destination_shift) > routing_floor_multiple * max(
        abs(noop_destination_shift), 1e-6
    )

    if attention_moved and content_moved:
        label = MIXED
    elif attention_moved:
        label = ROUTING_MEDIATED
    elif content_moved:
        label = CONTENT_MEDIATED
    else:
        label = MIXED

    return {
        "label": label,
        "attention_moved": bool(attention_moved),
        "content_moved": bool(content_moved),
        "mean_l1": mean_l1,
        "mean_jensen_shannon": mean_jsd,
        "max_l1": float(np.max(l1)),
        "max_jensen_shannon": float(np.max(jsd)),
        "noop_floor_l1": floor_l1,
        "noop_floor_jensen_shannon": floor_jsd,
        "destination_shift": float(destination_shift),
        "noop_destination_shift": float(noop_destination_shift),
        "per_head_mean_l1": [float(value) for value in np.mean(l1, axis=0)],
        "per_head_mean_jensen_shannon": [float(value) for value in np.mean(jsd, axis=0)],
        "routing_floor_multiple": float(routing_floor_multiple),
        "enters_claim_ladder": False,
    }


def summarize_routing(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate per-batch routing diagnostics into one reported verdict."""

    if not rows:
        return {"label": "ROUTING_DIAGNOSTIC_UNAVAILABLE", "n_batches": 0}
    labels = [str(row["label"]) for row in rows]
    counts = {name: labels.count(name) for name in sorted(set(labels))}
    majority = max(counts.items(), key=lambda item: item[1])[0]
    return {
        "label": majority,
        "label_counts": counts,
        "n_batches": len(rows),
        "mean_l1": float(np.mean([float(row["mean_l1"]) for row in rows])),
        "mean_jensen_shannon": float(
            np.mean([float(row["mean_jensen_shannon"]) for row in rows])
        ),
        "mean_noop_floor_l1": float(np.mean([float(row["noop_floor_l1"]) for row in rows])),
        "enters_claim_ladder": False,
    }
