"""Document strata and the stability-versus-null test for fitted invariants (R1 §33).

The question this module answers is whether an invariant fitted per stratum --
a rank-versus-predictive-variance curve, an eigenvalue summary, a cell
statistic -- is *stable* across document strata.  Two constructions make the
answer honest.

**A within-stratum null, not a zero null.**  Between-stratum variation is
essentially never exactly zero, so "the strata differ" is not evidence of
instability.  :func:`within_stratum_null` measures how much the same statistic
moves when a *single* stratum is split in half at random, which is the noise
floor any between-stratum spread must clear.  :func:`permutation_test` supplies
the second reference: the spread that survives when stratum labels are shuffled
across documents.  :func:`stratum_label` reports
``SPECTRUM_STRATUM_VARIABLE`` only when the observed spread exceeds *both*.

**Size-matched strata.**  Spectral and rank estimates are biased by sample
size: a small stratum yields a systematically different spectrum than a large
one from the identical generating process.  :func:`match_stratum_sizes`
subsamples every stratum to the smallest stratum's document count, so unmatched
``n`` cannot manufacture apparent instability.

**Curves, not thresholded ranks.**  Stability of the rank profile is compared
with :func:`curve_distance`, a mean absolute difference between two
rank-versus-predictive-variance curves.  An integer "effective rank" is never
compared instead: it is a cutoff applied to a continuous spectrum, so a pair of
near-identical curves straddling the cutoff reports a whole unit of
disagreement while a pair of very different curves on the same side of it
reports none.  The thresholded integer is brittle by construction; the curve is
the object the estimator actually produces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np

SCHEMES: tuple[str, ...] = ("length_band", "c4_source_domain", "register_proxy")

_TERCILE_NAMES: tuple[str, ...] = ("low", "mid", "high")
_SENTENCE_ENDINGS: frozenset[str] = frozenset(".!?")
_PUNCTUATION_OR_DIGIT_MARKS: frozenset[str] = frozenset("0123456789.,;:!?()[]{}\"'`/\\|-_=+*&^%$#@~<>")


# --------------------------------------------------------------------------
# Stratum assignment (R1 §33.1)
# --------------------------------------------------------------------------


def _tercile_labels(values: np.ndarray) -> np.ndarray:
    """Split a scalar vector into terciles, tolerating ties and degeneracy.

    ``np.quantile`` on a heavily tied vector can return equal cut points; the
    resulting label set then has fewer than three distinct values, which is
    reported as-is rather than raised on.  Constant input yields a single
    stratum.
    """

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return np.zeros(0, dtype=np.int64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape[0], dtype=np.int64)
    lower = float(np.quantile(finite, 1.0 / 3.0))
    upper = float(np.quantile(finite, 2.0 / 3.0))
    cuts = np.array(sorted({lower, upper}), dtype=np.float64)
    filled = np.where(np.isfinite(array), array, float(np.median(finite)))
    return np.searchsorted(cuts, filled, side="right").astype(np.int64)


def _host_of(record: Mapping[str, object]) -> str:
    """Registered host of a record's ``url``, or ``''`` when absent/unparsable."""

    url = record.get("url")
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        host = urlsplit(url.strip()).hostname
    except ValueError:
        return ""
    if not host:
        return ""
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def _domain_order(hosts: Sequence[str], min_documents_per_group: int) -> list[str]:
    """Frequent hosts in descending count order; ``'other'`` last when used."""

    unique, counts = np.unique(np.asarray(hosts, dtype=object).astype("U256"), return_counts=True)
    frequent = [
        str(host)
        for host, count in sorted(
            zip(unique.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0])
        )
        if host != "" and int(count) >= int(min_documents_per_group)
    ]
    if len(frequent) < len(set(hosts)) or "" in set(hosts):
        frequent.append("other")
    return frequent


def _register_scores(records: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Continuous register proxy: sentence length against symbol density.

    For each document with usable ``text`` we compute two surface statistics:

    * ``mean_sentence_length`` -- characters per sentence, splitting on
      ``.``/``!``/``?``;
    * ``symbol_fraction`` -- the fraction of characters that are punctuation
      marks or digits.

    The score is ``z(mean_sentence_length) - z(symbol_fraction)``, standardized
    across the supplied records.  High scores mean long sentences with few
    symbols (running prose); low scores mean short segments dense in
    punctuation and digits (lists, tables, navigation, code-like text).

    This is a *proxy*, not a validated register classifier.  It has no training
    data, no register taxonomy, and no measured agreement with any human or
    published register annotation.  It is used only to cut the corpus into
    three groups that plausibly differ in surface style, so that stability can
    be tested across them; no claim about the linguistic register of any
    document follows from it.  Documents with missing or empty ``text`` receive
    the median score of the documents that have text.
    """

    lengths = np.full(len(records), np.nan, dtype=np.float64)
    symbols = np.full(len(records), np.nan, dtype=np.float64)
    for index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        n_sentences = max(1, sum(1 for character in text if character in _SENTENCE_ENDINGS))
        lengths[index] = len(text) / float(n_sentences)
        symbols[index] = sum(
            1 for character in text if character in _PUNCTUATION_OR_DIGIT_MARKS
        ) / float(len(text))

    scores = np.zeros(len(records), dtype=np.float64)
    present = np.isfinite(lengths) & np.isfinite(symbols)
    if not present.any():
        return scores
    for column, target in ((lengths, "length"), (symbols, "symbol")):
        values = column[present]
        spread = float(values.std())
        standardized = (values - float(values.mean())) / (spread if spread > 0 else 1.0)
        scores[present] += standardized if target == "length" else -standardized
    scores[~present] = float(np.median(scores[present]))
    return scores


def _suffix_of(host: str) -> str:
    """Last label of a host, e.g. ``'com'`` for ``news.bbc.com``.

    C4 draws from an enormous number of distinct hosts, so at 2400 documents
    essentially no single host reaches a usable group size and host-level
    grouping degenerates to one bucket.  The public suffix is a coarser cut of
    the same underlying quantity -- where the document came from -- that the
    corpus can actually support.
    """

    if not host:
        return ""
    parts = host.split(".")
    return parts[-1] if parts else ""


def domain_labels_with_granularity(
    hosts: Sequence[str], min_documents_per_group: int
) -> tuple[np.ndarray, str]:
    """Domain labels at the finest granularity yielding at least two groups.

    Returns the labels and the granularity actually used.  A scheme that
    collapses to a single group cannot support a between-stratum comparison, so
    the granularity is reported rather than left implicit: a reader must be able
    to see that ``c4_source_domain`` means hosts in one run and public suffixes
    in another.
    """

    for granularity, keys in (
        ("host", list(hosts)),
        ("public_suffix", [_suffix_of(host) for host in hosts]),
    ):
        order = _domain_order(keys, min_documents_per_group)
        index_of = {key: position for position, key in enumerate(order)}
        fallback = index_of.get("other", 0)
        labels = np.array([index_of.get(key, fallback) for key in keys], dtype=np.int64)
        if np.unique(labels).size >= 2:
            return labels, granularity
    return np.zeros(len(hosts), dtype=np.int64), "degenerate_single_group"


def assign_strata(
    records: Sequence[Mapping[str, object]],
    *,
    min_documents_per_group: int = 25,
) -> dict[str, np.ndarray]:
    """Stratum labels for every scheme in :data:`SCHEMES`, aligned to ``records``.

    ``records`` are per-document mappings carrying ``document_id`` and
    ``n_tokens``, and optionally ``url`` and ``text``.  A missing ``url`` sends
    the document to the ``'other'`` domain bucket; a missing ``text`` gives it
    the median register-proxy score.  Neither raises.
    """

    n_tokens = np.array(
        [float(record.get("n_tokens", 0) or 0) for record in records], dtype=np.float64
    )
    hosts = [_host_of(record) for record in records]
    domain_labels, _granularity = domain_labels_with_granularity(
        hosts, min_documents_per_group
    )
    return {
        "length_band": _tercile_labels(n_tokens),
        "c4_source_domain": domain_labels,
        "register_proxy": _tercile_labels(_register_scores(records)),
    }


def describe_strata(
    records: Sequence[Mapping[str, object]],
    scheme: str,
    labels: np.ndarray,
) -> dict[int, str]:
    """Human-readable name for every label present in ``labels``."""

    if scheme not in SCHEMES:
        raise ValueError(f"unknown stratum scheme: {scheme!r}")
    codes = np.asarray(labels, dtype=np.int64).reshape(-1)
    present = [int(code) for code in np.unique(codes)]

    if scheme == "c4_source_domain":
        hosts = [_host_of(record) for record in records]
        domain_names: dict[int, str] = {}
        for code in present:
            members = [hosts[row] for row in np.flatnonzero(codes == code)]
            distinct = sorted({host for host in members if host})
            if len(distinct) == 1 and all(members):
                domain_names[code] = f"{distinct[0]} (n={len(members)})"
            else:
                domain_names[code] = f"other: rare or unknown hosts (n={len(members)})"
        return domain_names

    if scheme == "length_band":
        values = np.array(
            [float(record.get("n_tokens", 0) or 0) for record in records], dtype=np.float64
        )
        unit = "tokens"
    else:
        values = _register_scores(records)
        unit = "register-proxy score"

    names: dict[int, str] = {}
    for position, code in enumerate(present):
        member_values = values[codes == code]
        band = _TERCILE_NAMES[position] if len(present) == 3 else f"band {position}"
        if member_values.size == 0:
            names[code] = f"{scheme} {band} (empty)"
            continue
        names[code] = (
            f"{scheme} {band}: {unit} in "
            f"[{float(member_values.min()):.4g}, {float(member_values.max()):.4g}], "
            f"n={int(member_values.size)}"
        )
    return names


# --------------------------------------------------------------------------
# Size matching (R1 §33.2)
# --------------------------------------------------------------------------


def match_stratum_sizes(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of a size-matched subsample: every stratum at the smallest size.

    Spectral and rank estimates vary with the number of documents they are
    fitted on, so comparing a 900-document stratum with a 200-document stratum
    conflates the estimator's sample-size bias with a genuine difference in the
    fitted invariant.  Matching first removes that route to a false positive.
    """

    codes = np.asarray(labels, dtype=np.int64).reshape(-1)
    if codes.size == 0:
        return np.zeros(0, dtype=np.int64)
    unique = np.unique(codes)
    smallest = int(min(int(np.sum(codes == code)) for code in unique))
    if smallest <= 0:
        return np.zeros(0, dtype=np.int64)
    selected: list[np.ndarray] = []
    for code in unique:
        members = np.flatnonzero(codes == code)
        take = rng.choice(members.shape[0], size=smallest, replace=False)
        selected.append(members[np.sort(take)])
    return np.sort(np.concatenate(selected)).astype(np.int64)


# --------------------------------------------------------------------------
# Curve comparison (R1 §33.3)
# --------------------------------------------------------------------------


def curve_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """Mean absolute difference between two curves over their shared prefix.

    The curves are rank-versus-predictive-variance profiles.  Unequal lengths
    arise whenever two strata admit different maximum ranks, so the comparison
    is truncated to the shorter length rather than raising or padding.  An
    empty overlap returns ``nan``.
    """

    left = np.asarray(curve_a, dtype=np.float64).reshape(-1)
    right = np.asarray(curve_b, dtype=np.float64).reshape(-1)
    shared = int(min(left.shape[0], right.shape[0]))
    if shared == 0:
        return float("nan")
    return float(np.mean(np.abs(left[:shared] - right[:shared])))


# --------------------------------------------------------------------------
# Nulls and the stability statistic (R1 §33.4)
# --------------------------------------------------------------------------


def _group_arrays(
    values_by_stratum: Mapping[object, np.ndarray],
) -> list[np.ndarray]:
    groups = []
    for values in values_by_stratum.values():
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        array = array[np.isfinite(array)]
        if array.size:
            groups.append(array)
    return groups


def within_stratum_null(
    values_by_stratum: Mapping[object, np.ndarray],
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, float]:
    """Split-half noise floor, pooled over strata.

    Each replicate splits one stratum's documents in half at random and records
    ``|mean(A) - mean(B)|``.  The returned ``median`` and ``q95`` say how large
    a between-stratum spread can be while still being explained by ordinary
    within-stratum sampling noise at the matched sample size.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    groups = _group_arrays(values_by_stratum)
    differences: list[float] = []
    for array in groups:
        if array.size < 2:
            continue
        half = array.size // 2
        for _ in range(int(replicates)):
            order = rng.permutation(array.size)
            differences.append(
                abs(float(array[order[:half]].mean()) - float(array[order[half : 2 * half]].mean()))
            )
    if not differences:
        return {"median": float("nan"), "q95": float("nan")}
    draws = np.asarray(differences, dtype=np.float64)
    return {"median": float(np.median(draws)), "q95": float(np.quantile(draws, 0.95))}


def between_stratum_statistic(values_by_stratum: Mapping[object, np.ndarray]) -> float:
    """Spread of the per-stratum means: ``max - min``."""

    groups = _group_arrays(values_by_stratum)
    if len(groups) < 2:
        return 0.0
    means = np.array([float(array.mean()) for array in groups], dtype=np.float64)
    return float(means.max() - means.min())


def _statistic_from_labels(values: np.ndarray, labels: np.ndarray) -> float:
    grouped = {int(code): values[labels == code] for code in np.unique(labels)}
    return between_stratum_statistic(grouped)


def permutation_test(
    values: np.ndarray,
    labels: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Label-permutation null for the between-stratum spread.

    Stratum labels are shuffled across documents, which preserves the marginal
    distribution of the per-document values and the stratum sizes while
    destroying any association between the two.  The p-value carries the
    conventional add-one correction, so its floor is ``1/(replicates + 1)``.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    codes = np.asarray(labels, dtype=np.int64).reshape(-1)
    if array.shape[0] != codes.shape[0]:
        raise ValueError("values and labels must have one entry per document")
    observed = _statistic_from_labels(array, codes)
    draws = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        draws[replicate] = _statistic_from_labels(array, codes[rng.permutation(codes.shape[0])])
    exceed = int(np.sum(draws >= observed))
    return {
        "observed": float(observed),
        "null_q95": float(np.quantile(draws, 0.95)),
        "p_value": float((exceed + 1) / (draws.shape[0] + 1)),
    }


def stratum_label(observed: float, null_q95: float, within_q95: float) -> str:
    """Stability verdict for one invariant under one stratification scheme.

    ``SPECTRUM_STRATUM_VARIABLE`` requires the observed between-stratum spread
    to exceed the permutation null's 95th percentile *and* the within-stratum
    split-half null's 95th percentile.  Clearing only the permutation null
    means the strata differ by more than label shuffling predicts but no more
    than the estimator's own within-stratum noise, which is not evidence that
    the invariant depends on the stratum.
    """

    # "Stable" is a substantive claim about the invariant.  An earlier version
    # returned it whenever an input was non-finite, so a stratification that was
    # never measured reported as stable.  A missing computation must never
    # produce a substantive label.
    if not np.isfinite(observed):
        return "SPECTRUM_STRATUM_UNAVAILABLE"
    thresholds = [value for value in (null_q95, within_q95) if np.isfinite(value)]
    if len(thresholds) < 2:
        return "SPECTRUM_STRATUM_UNAVAILABLE"
    if all(float(observed) > float(threshold) for threshold in thresholds):
        return "SPECTRUM_STRATUM_VARIABLE"
    return "SPECTRUM_STRATUM_STABLE"
