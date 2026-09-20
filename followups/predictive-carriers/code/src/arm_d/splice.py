"""The paired remote-prefix splice construction (R1 §16).

This is the primary object.  Every guarantee the analysis depends on is
enforced here and re-checked by :func:`verify_pair`:

* absolute token positions are preserved -- the spliced sequence has exactly
  the same length as the reference, and position ``i`` means the same thing in
  both;
* the ``L`` tokens immediately before the source position are identical;
* every token from the source position through the last evaluated lag is
  identical;
* the splice boundary at ``t - L`` never lies inside the 32-token nuisance
  window ending at ``t``, which ``L >= 32`` guarantees.

Four conditions share the construction and differ only in where the remote
prefix comes from (R1 §16.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONTEXT_LENGTH = 1024
MAX_LAG = 128
MAX_SUFFIX = 256
NUISANCE_WINDOW = 32

# Source positions must leave a remote prefix worth changing.  With
# ``L <= 256`` and a required prefix of at least 256 tokens the lower bound is
# 512; ``t + 128 <= 1023`` gives the upper bound.  Every Arm D position claim is
# scoped to this range and the report says so.
MIN_SOURCE_POSITION = 512
MAX_SOURCE_POSITION = CONTEXT_LENGTH - MAX_LAG - 1
N_POSITION_BANDS = 8

SUFFIX_LENGTHS = (32, 64, 128, 256)
CONDITIONS = ("main", "noop", "sham", "matched")


def position_band_edges(n_bands: int = N_POSITION_BANDS) -> np.ndarray:
    return np.linspace(MIN_SOURCE_POSITION, MAX_SOURCE_POSITION + 1, n_bands + 1).astype(np.int64)


def assign_source_position(context_id: str, band: int, n_bands: int = N_POSITION_BANDS) -> int:
    """Deterministic per-document source position inside its frozen band."""

    edges = position_band_edges(n_bands)
    low, high = int(edges[band]), int(edges[band + 1])
    if high <= low:
        raise ValueError("empty position band")
    offset = int(int(context_id[:16], 16) % (high - low))
    return low + offset


def assign_bands(context_ids: list[str], n_bands: int = N_POSITION_BANDS) -> np.ndarray:
    """Balanced band assignment: equal document counts per band, deterministic."""

    order = np.argsort([int(context_id[:16], 16) for context_id in context_ids])
    bands = np.empty(len(context_ids), dtype=np.int64)
    bands[order] = np.arange(len(context_ids)) % n_bands
    return bands


@dataclass(frozen=True)
class SpliceSpec:
    target_id: str
    donor_id: str
    condition: str
    suffix_length: int
    source_position: int
    band: int

    @property
    def boundary(self) -> int:
        return self.source_position - self.suffix_length

    def row_key(self) -> str:
        return f"{self.target_id}:{self.condition}:{self.suffix_length}"


def build_spliced_tokens(
    target_tokens: np.ndarray,
    donor_tokens: np.ndarray,
    spec: SpliceSpec,
) -> np.ndarray:
    """Return the spliced token sequence for one condition.

    ``main`` and ``matched`` take the remote prefix from another document.
    ``noop`` reconstructs the target exactly -- run as an independent forward
    pass it measures the numerical floor including batch-position
    nondeterminism, which is what the floor test needs.
    ``sham`` rotates the target's own prefix, preserving document-level topic
    and token statistics while destroying the exact remote history.
    """

    target = np.asarray(target_tokens, dtype=np.int64)
    donor = np.asarray(donor_tokens, dtype=np.int64)
    if target.shape[0] != CONTEXT_LENGTH or donor.shape[0] != CONTEXT_LENGTH:
        raise ValueError("both documents must be exactly CONTEXT_LENGTH tokens")
    boundary = spec.boundary
    if boundary < MIN_SOURCE_POSITION - MAX_SUFFIX or boundary <= 0:
        raise ValueError(f"invalid splice boundary {boundary}")

    spliced = target.copy()
    if spec.condition == "noop":
        return spliced
    if spec.condition == "sham":
        prefix = target[:boundary]
        pivot = boundary // 2
        spliced[:boundary] = np.concatenate([prefix[pivot:], prefix[:pivot]])
        return spliced
    if spec.condition in {"main", "matched"}:
        spliced[:boundary] = donor[:boundary]
        return spliced
    raise ValueError(f"unknown splice condition: {spec.condition!r}")


def verify_pair(
    reference_tokens: np.ndarray,
    spliced_tokens: np.ndarray,
    spec: SpliceSpec,
    *,
    max_lag: int = MAX_LAG,
) -> dict[str, bool]:
    """Check every invariant R1 §16.2 requires, as booleans (all must be True)."""

    reference = np.asarray(reference_tokens, dtype=np.int64)
    spliced = np.asarray(spliced_tokens, dtype=np.int64)
    t = spec.source_position
    boundary = spec.boundary
    checks = {
        "same_length": reference.shape == spliced.shape,
        "absolute_positions_preserved": reference.shape[0] == CONTEXT_LENGTH,
        "identical_suffix_before_source": bool(np.array_equal(reference[boundary:t], spliced[boundary:t])),
        "identical_source_and_future": bool(
            np.array_equal(reference[t : t + max_lag + 1], spliced[t : t + max_lag + 1])
        ),
        "identical_from_boundary_onwards": bool(np.array_equal(reference[boundary:], spliced[boundary:])),
        "boundary_outside_nuisance_window": bool(boundary <= t - NUISANCE_WINDOW),
        "remote_prefix_nonempty": bool(boundary >= 1),
    }
    if spec.condition == "noop":
        checks["noop_is_exact_reconstruction"] = bool(np.array_equal(reference, spliced))
    else:
        checks["remote_prefix_actually_changed"] = bool(
            not np.array_equal(reference[:boundary], spliced[:boundary])
        )
    return checks


def choose_matched_content_donor(
    target_index: int,
    profiles: np.ndarray,
    boundary_lengths: np.ndarray,
    excluded: set[int] | None = None,
) -> int:
    """Most content-similar other document, by hashed token-profile cosine."""

    similarity = profiles @ profiles[target_index]
    similarity[target_index] = -np.inf
    for index in excluded or set():
        similarity[index] = -np.inf
    return int(np.argmax(similarity))


def choose_stratum_donor(
    target_index: int,
    strata: np.ndarray,
    rng: np.random.Generator,
    excluded: set[int] | None = None,
) -> int:
    """Uniform donor from the target's frozen document-level nuisance stratum.

    Falls back to the whole pool when a stratum has no other member, and the
    fallback is recorded so the report can say how often it happened.
    """

    excluded = set(excluded or set()) | {target_index}
    candidates = np.flatnonzero(strata == strata[target_index])
    candidates = np.array([index for index in candidates if index not in excluded])
    if candidates.size == 0:
        candidates = np.array([index for index in range(strata.shape[0]) if index not in excluded])
    return int(rng.choice(candidates))


def build_splice_plan(
    context_ids: list[str],
    strata: np.ndarray,
    profiles: np.ndarray,
    *,
    suffix_lengths: tuple[int, ...] = SUFFIX_LENGTHS,
    conditions: tuple[str, ...] = CONDITIONS,
    seed: int = 42,
) -> tuple[list[SpliceSpec], dict[str, object]]:
    """One source position per document, all conditions and suffix lengths.

    The donor for the ``main`` condition is fixed per (target, suffix length)
    so the same remote prefix change is compared across lags, and is drawn from
    the target's nuisance stratum while excluding the matched-content donor, so
    ``main`` and ``matched`` are genuinely different comparisons.
    """

    rng = np.random.default_rng(seed)
    bands = assign_bands(context_ids)
    boundary_lengths = np.array(
        [assign_source_position(context_id, int(band)) for context_id, band in zip(context_ids, bands)]
    )
    specs: list[SpliceSpec] = []
    fallbacks = 0
    for index, context_id in enumerate(context_ids):
        band = int(bands[index])
        position = int(boundary_lengths[index])
        matched_index = choose_matched_content_donor(index, profiles.copy(), boundary_lengths)
        main_index = choose_stratum_donor(index, strata, rng, excluded={matched_index})
        if strata[main_index] != strata[index]:
            fallbacks += 1
        for suffix_length in suffix_lengths:
            for condition in conditions:
                donor_index = {
                    "main": main_index,
                    "matched": matched_index,
                    "sham": index,
                    "noop": index,
                }[condition]
                specs.append(
                    SpliceSpec(
                        target_id=context_id,
                        donor_id=context_ids[donor_index],
                        condition=condition,
                        suffix_length=int(suffix_length),
                        source_position=position,
                        band=band,
                    )
                )
    summary = {
        "documents": len(context_ids),
        "sequences_per_document": len(suffix_lengths) * len(conditions) + 1,
        "total_spliced_sequences": len(specs),
        "position_band_edges": [int(value) for value in position_band_edges()],
        "source_position_range": [MIN_SOURCE_POSITION, MAX_SOURCE_POSITION],
        "stratum_fallbacks": int(fallbacks),
        "suffix_lengths": list(suffix_lengths),
        "conditions": list(conditions),
    }
    return specs, summary
