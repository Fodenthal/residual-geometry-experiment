"""Composed retention, predictive gain, and their document bootstraps, per carrier.

The composition numbers themselves come from R2.2's
:func:`arm_d_r2.dynamics.composition_analysis`, unchanged, called once per carrier
with that carrier's rank.  Nothing about the estimator, the new-information
projection, the direct-fit ceiling or the estimation envelope is re-implemented
here; what this module adds is (1) an aperture-space predictive gain that is
comparable across carriers of the same width, and (2) a document bootstrap of
retention and of the paired retention DIFFERENCE between two carriers.

The bootstrap reuses the ridge values the analysis itself selected rather than
re-selecting them, and then asserts that the point estimates it reconstructs equal
the analysis's own to 1e-12.  Without that assertion the bootstrap could be
centred on a slightly different estimator than the one being reported, which is
exactly the class of mismatch R2.2's invariants exist to catch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from arm_d.companion import DEFAULT_RIDGE_GRID, _select_ridge, _solve_full
from arm_d.scoring import document_index, equal_document_r2

from arm_d_r2.dynamics import augmented_state, composition_analysis, retention_fractions

from arm_d_r3.contract import (
    APERTURE_WIDTH,
    COMPOSITION_ENVELOPE_REPLICATES,
    HISTORY_OFFSETS,
    LAGS,
    RETENTION_BOOTSTRAP,
    SECTION_COMPARE,
    SEED,
    STATE_ORDER,
    label,
    unavailable,
)

SECTION = SECTION_COMPARE
COMPONENT_TOLERANCE = 1e-12


def carrier_history_and_futures(
    capture, basis: np.ndarray, *, width: int = APERTURE_WIDTH
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """``(history, {offset: state})`` in carrier coordinates at the source hook."""

    states = capture.source_states(basis, width=width)
    history = np.stack([states[offset] for offset in HISTORY_OFFSETS], axis=1)
    return history, states


def composition_for_carrier(
    *,
    history: np.ndarray,
    states: Mapping[int, np.ndarray],
    train: np.ndarray,
    evaluate: np.ndarray,
    documents: np.ndarray,
    rank: int,
    order: int = STATE_ORDER,
    lags: Sequence[int] = LAGS,
    seed: int = SEED,
    envelope_replicates: int = COMPOSITION_ENVELOPE_REPLICATES,
) -> dict[str, object]:
    """R2.2's composition analysis on one carrier, called exactly as R2.2 calls it."""

    state = augmented_state(history, order)
    return composition_analysis(
        state_train=state[train],
        state_eval=state[evaluate],
        documents_train=documents[train],
        documents_eval=documents[evaluate],
        future_train={offset: values[train] for offset, values in states.items()},
        future_eval={offset: values[evaluate] for offset, values in states.items()},
        order=order,
        rank=int(rank),
        lags=lags,
        seed=seed,
        envelope_replicates=envelope_replicates,
    )


def _document_energy(values: np.ndarray, codes: np.ndarray, n_documents: int) -> np.ndarray:
    """Mean squared row norm per document, the unit ``equal_document_r2`` averages."""

    rows = np.sum(np.asarray(values, dtype=np.float64) ** 2, axis=1)
    counts = np.bincount(codes, minlength=n_documents).astype(np.float64)
    return np.bincount(codes, weights=rows, minlength=n_documents) / counts


def _r2_from_energies(residual: np.ndarray, target: np.ndarray, take: np.ndarray | None = None) -> float:
    if take is None:
        numerator, denominator = residual.mean(), target.mean()
    else:
        numerator, denominator = residual[take].mean(), target[take].mean()
    if denominator <= 0:
        return float("nan")
    return float(1.0 - numerator / denominator)


def retention_components(
    *,
    analysis: Mapping[str, object],
    history: np.ndarray,
    states: Mapping[int, np.ndarray],
    train: np.ndarray,
    evaluate: np.ndarray,
    documents: np.ndarray,
    rank: int,
    order: int = STATE_ORDER,
    lags: Sequence[int] = LAGS,
) -> dict[str, object]:
    """Per-document energies behind ``p_comp``, verified against the analysis.

    Returns, per lag, the four per-document energy vectors (target, composed
    residual, direct residual, persistence residual) in one fixed document order,
    so a bootstrap over documents can recompute ``p_comp`` without refitting and
    two carriers can be resampled together.
    """

    state = augmented_state(history, order)
    state_train, state_eval = state[train], state[evaluate]
    documents_eval = documents[evaluate]
    labels, codes = document_index(documents_eval)
    n_documents = int(labels.shape[0])
    companion = np.asarray(analysis["companion"], dtype=np.float64)
    size = order * int(rank)

    out: dict[str, object] = {"documents": labels, "n_documents": n_documents, "per_lag": {}}
    for lag in lags:
        key = str(int(lag))
        entry = analysis["per_lag"][key]  # type: ignore[index]
        if "unavailable" in entry:
            out["per_lag"][key] = {"unavailable": entry["unavailable"]}
            continue
        width = min(int(lag), order)
        offsets = [int(lag) - shift for shift in range(width)]
        target_train = np.concatenate([states[offset][train] for offset in offsets], axis=1)
        target_eval = np.concatenate([states[offset][evaluate] for offset in offsets], axis=1)
        projection = np.zeros((width * int(rank), size), dtype=np.float64)
        projection[:, : width * int(rank)] = np.eye(width * int(rank))
        powered = np.linalg.matrix_power(companion, int(lag))
        composed = state_eval @ (projection @ powered).T
        direct_weights = _solve_full(state_train, target_train, float(entry["direct_ridge"]))
        direct = state_eval @ direct_weights
        persistence = np.tile(state_eval[:, : int(rank)], (1, width))

        target_energy = _document_energy(target_eval, codes, n_documents)
        record = {
            "target_energy": target_energy,
            "composed_residual": _document_energy(target_eval - composed, codes, n_documents),
            "direct_residual": _document_energy(target_eval - direct, codes, n_documents),
            "persistence_residual": _document_energy(target_eval - persistence, codes, n_documents),
        }
        # Invariant: the reconstruction must reproduce the analysis exactly.
        composed_r2 = _r2_from_energies(record["composed_residual"], target_energy)
        direct_r2 = _r2_from_energies(record["direct_residual"], target_energy)
        persistence_r2 = _r2_from_energies(record["persistence_residual"], target_energy)
        checks = {
            "composed_r2": (composed_r2, float(entry["composed"]["r2"])),  # type: ignore[index]
            "direct_r2": (direct_r2, float(entry["direct"]["r2"])),  # type: ignore[index]
            "persistence_r2": (persistence_r2, float(entry["direct"]["persistence_r2"])),  # type: ignore[index]
        }
        deviation = max(abs(a - b) for a, b in checks.values())
        if deviation > COMPONENT_TOLERANCE:
            raise AssertionError(
                f"bootstrap components disagree with the composition analysis at lag {lag} "
                f"by {deviation:.3e}: " + ", ".join(
                    f"{name} {a!r} vs {b!r}" for name, (a, b) in checks.items()
                )
            )
        record["reconstruction_deviation"] = float(deviation)
        out["per_lag"][key] = record
    return out


def _p_comp_from(components: Mapping[str, np.ndarray], take: np.ndarray | None, gamma_min: float) -> float:
    target = components["target_energy"]
    composed = _r2_from_energies(components["composed_residual"], target, take)
    direct = _r2_from_energies(components["direct_residual"], target, take)
    persistence = _r2_from_energies(components["persistence_residual"], target, take)
    baseline = max(0.0, persistence)
    direct_gain = direct - baseline
    if not np.isfinite(direct_gain) or direct_gain < gamma_min:
        return float("nan")
    return float((composed - baseline) / direct_gain)


def _composed_gain_from(
    components: Mapping[str, np.ndarray], take: np.ndarray | None
) -> float:
    """Composed gain against the carrier's OWN target energy: not a cross-carrier scale.

    ``p_comp`` divides by the carrier's own direct-fit ceiling, so a carrier whose
    in-carrier ceiling is low can post a high ratio while delivering little.  Dropping
    that denominator removes the ratio artifact, but what is left is still an R2
    against this carrier's own projected-future energy, which differs between carriers.
    It is therefore NOT an absolute cross-carrier quantity and no decision rule treats
    it as one; it is reported as a within-carrier check on the ratio.  The only
    genuinely carrier-comparable predictive quantity in this run is
    ``aperture_target_gain``, measured against the fixed 256-dimensional aperture, and
    the control that carries the composition comparison is the rank-matched Haar floor.
    """

    target = components["target_energy"]
    composed = _r2_from_energies(components["composed_residual"], target, take)
    persistence = _r2_from_energies(components["persistence_residual"], target, take)
    return float(composed - max(0.0, persistence))


def composed_gain_bootstrap(
    components: Mapping[str, object],
    *,
    replicates: int = RETENTION_BOOTSTRAP,
    seed: int = SEED,
    lags: Sequence[int] = LAGS,
) -> dict[str, object]:
    """Document bootstrap of the ABSOLUTE composed gain, ``p_comp x direct_gain``.

    Composed retention is a ratio against each carrier's own direct fit, so it is
    scale-free: a carrier with almost no signal can post a ratio near 1 while predicting
    almost nothing, and two carriers whose captured variance differs by an order of
    magnitude cannot be ranked by it.  This is the same quantity without that denominator:
    how much R2 over persistence the composed operator actually delivers in that carrier's
    coordinates.  It is still an R2 against that carrier's own projected-future energy, so
    the fully carrier-comparable predictive quantity remains ``aperture_target_gain``
    against the fixed 256-dimensional aperture; this one removes the ratio artifact and is
    reported beside retention at every lag.

    Resampling unit and seed match ``retention_bootstrap``, so the two intervals are
    drawn on the same document resamples and may be read together.
    """

    n_documents = int(components["n_documents"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_documents, size=(int(replicates), n_documents))
    out: dict[str, object] = {}
    for lag in lags:
        key = str(int(lag))
        entry = components["per_lag"].get(key)  # type: ignore[union-attr]
        if entry is None or "unavailable" in entry:
            out[key] = {
                "unavailable": unavailable(
                    f"composed_gain_bootstrap_lag_{key}", "no components"
                )
            }
            continue
        point = _composed_gain_from(entry, None)
        values = np.array(
            [_composed_gain_from(entry, draws[index]) for index in range(int(replicates))]
        )
        usable = values[np.isfinite(values)]
        if usable.size < int(replicates) // 2:
            out[key] = {
                "point": float(point) if np.isfinite(point) else None,
                "unavailable": unavailable(
                    f"composed_gain_interval_lag_{key}",
                    f"only {int(usable.size)} of {int(replicates)} replicates were finite",
                ),
                "usable_replicates": int(usable.size),
            }
            continue
        out[key] = {
            "point": float(point),
            "q05": float(np.quantile(usable, 0.05)),
            "median": float(np.median(usable)),
            "q95": float(np.quantile(usable, 0.95)),
            "q025": float(np.quantile(usable, 0.025)),
            "q975": float(np.quantile(usable, 0.975)),
            "std": float(usable.std(ddof=1)),
            "usable_replicates": int(usable.size),
        }
    return out


def retention_bootstrap(
    components: Mapping[str, object],
    *,
    gamma_min: float,
    replicates: int = RETENTION_BOOTSTRAP,
    seed: int = SEED,
    lags: Sequence[int] = LAGS,
) -> dict[str, object]:
    """Document bootstrap of ``p_comp``, holding the fitted operators fixed.

    The resampling unit is the evaluation document, matching R2.2's bootstrap
    contract.  The interval therefore covers evaluation sampling only; training
    noise is what the estimation envelope covers, and the two are reported apart.
    """

    n_documents = int(components["n_documents"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_documents, size=(int(replicates), n_documents))
    out: dict[str, object] = {}
    for lag in lags:
        key = str(int(lag))
        entry = components["per_lag"].get(key)  # type: ignore[union-attr]
        if entry is None or "unavailable" in entry:
            out[key] = {"unavailable": unavailable(f"p_comp_bootstrap_lag_{key}", "no components")}
            continue
        point = _p_comp_from(entry, None, gamma_min)
        values = np.array([_p_comp_from(entry, draws[index], gamma_min) for index in range(int(replicates))])
        usable = values[np.isfinite(values)]
        if usable.size < int(replicates) // 2:
            out[key] = {
                "point": float(point) if np.isfinite(point) else unavailable(
                    f"p_comp_lag_{key}", f"direct gain below the frozen minimum {gamma_min:.5f}"
                ),
                "unavailable": unavailable(
                    f"p_comp_interval_lag_{key}",
                    f"only {int(usable.size)} of {int(replicates)} replicates cleared the "
                    f"frozen minimum direct gain {gamma_min:.5f}",
                ),
                "usable_replicates": int(usable.size),
            }
            continue
        out[key] = {
            "point": float(point),
            "q05": float(np.quantile(usable, 0.05)),
            "q25": float(np.quantile(usable, 0.25)),
            "median": float(np.median(usable)),
            "q75": float(np.quantile(usable, 0.75)),
            "q95": float(np.quantile(usable, 0.95)),
            "q025": float(np.quantile(usable, 0.025)),
            "q975": float(np.quantile(usable, 0.975)),
            "std": float(usable.std(ddof=1)),
            "usable_replicates": int(usable.size),
        }
    return out


def paired_retention_difference(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    gamma_min: float,
    replicates: int = RETENTION_BOOTSTRAP,
    seed: int = SEED,
    lags: Sequence[int] = LAGS,
    left_name: str,
    right_name: str,
) -> dict[str, object]:
    """Paired document bootstrap of ``p_comp(left) - p_comp(right)``.

    Both carriers are scored on the same evaluation documents, so one resampling
    of documents applies to both and the difference is paired.  An unpaired
    interval would be wider for no reason and would not answer whether the two
    carriers differ ON THE SAME documents.
    """

    if not np.array_equal(np.asarray(left["documents"]), np.asarray(right["documents"])):
        return {
            "unavailable": unavailable(
                "paired_difference", "the two carriers were scored on different documents"
            )
        }
    n_documents = int(left["n_documents"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_documents, size=(int(replicates), n_documents))
    out: dict[str, object] = {"left": left_name, "right": right_name}
    for lag in lags:
        key = str(int(lag))
        left_entry = left["per_lag"].get(key)  # type: ignore[union-attr]
        right_entry = right["per_lag"].get(key)  # type: ignore[union-attr]
        if not left_entry or not right_entry or "unavailable" in left_entry or "unavailable" in right_entry:
            out[key] = {"unavailable": unavailable(f"paired_difference_lag_{key}", "missing components")}
            continue
        point = _p_comp_from(left_entry, None, gamma_min) - _p_comp_from(right_entry, None, gamma_min)
        values = np.array(
            [
                _p_comp_from(left_entry, draws[index], gamma_min)
                - _p_comp_from(right_entry, draws[index], gamma_min)
                for index in range(int(replicates))
            ]
        )
        gain_point = _composed_gain_from(left_entry, None) - _composed_gain_from(right_entry, None)
        gain_values = np.array(
            [
                _composed_gain_from(left_entry, draws[index])
                - _composed_gain_from(right_entry, draws[index])
                for index in range(int(replicates))
            ]
        )
        gain_usable = gain_values[np.isfinite(gain_values)]
        usable = values[np.isfinite(values)]
        if usable.size < int(replicates) // 2:
            out[key] = {
                "unavailable": unavailable(
                    f"paired_difference_lag_{key}",
                    f"only {int(usable.size)} of {int(replicates)} replicates usable",
                )
            }
            continue
        out[key] = {
            "point": float(point),
            "q05": float(np.quantile(usable, 0.05)),
            "q95": float(np.quantile(usable, 0.95)),
            "q025": float(np.quantile(usable, 0.025)),
            "q975": float(np.quantile(usable, 0.975)),
            "median": float(np.median(usable)),
            "std": float(usable.std(ddof=1)),
            "excludes_zero_two_sided": bool(
                float(np.quantile(usable, 0.025)) > 0.0 or float(np.quantile(usable, 0.975)) < 0.0
            ),
            "positive_beyond_one_sided_95": bool(float(np.quantile(usable, 0.05)) > 0.0),
            "usable_replicates": int(usable.size),
            "composed_gain_difference_own_target": {
                "scale": (
                    "R2 against each carrier's own projected-future energy; comparable"
                    " within a carrier, not across carriers"
                ),
                "point": float(gain_point),
                "q05": float(np.quantile(gain_usable, 0.05)),
                "q95": float(np.quantile(gain_usable, 0.95)),
                "q025": float(np.quantile(gain_usable, 0.025)),
                "q975": float(np.quantile(gain_usable, 0.975)),
                "positive_beyond_one_sided_95": bool(float(np.quantile(gain_usable, 0.05)) > 0.0),
            },
        }
    return out


def aperture_target_gain(
    *,
    history: np.ndarray,
    aperture_futures: Mapping[int, np.ndarray],
    aperture_source: np.ndarray,
    train: np.ndarray,
    evaluate: np.ndarray,
    documents: np.ndarray,
    rank: int,
    order: int = STATE_ORDER,
    lags: Sequence[int] = LAGS,
    seed: int = SEED,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
) -> dict[str, object]:
    """How much of the FIXED aperture-space future the carrier's state predicts.

    The target is the full 256-dimensional aperture difference at lag ``k``, which
    no choice of carrier can redefine; only the predictor's coordinates change.
    This is the carrier-comparable predictive quantity, and it is the same fixed
    target R2.2 amendment A7 introduced for the deletion controls.  Capacity is
    ``order * rank`` and is reported, because a wider carrier is a bigger model.
    """

    state = augmented_state(history, order)
    state_train, state_eval = state[train], state[evaluate]
    documents_train, documents_eval = documents[train], documents[evaluate]
    out: dict[str, object] = {"capacity": int(order * int(rank))}
    for lag in lags:
        key = str(int(lag))
        if int(lag) not in aperture_futures:
            out[key] = {"unavailable": unavailable(f"aperture_future_{lag}")}
            continue
        target_train = aperture_futures[int(lag)][train]
        target_eval = aperture_futures[int(lag)][evaluate]
        ridge = _select_ridge(
            "full", state_train, target_train, documents_train, order, int(rank),
            tuple(ridge_grid), 5, seed,
        )
        weights = _solve_full(state_train, target_train, ridge)
        r2 = float(equal_document_r2(target_eval, state_eval @ weights, documents_eval).r2)
        persistence = float(
            equal_document_r2(target_eval, aperture_source[evaluate], documents_eval).r2
        )
        out[key] = {
            "r2": r2,
            "persistence_r2": persistence,
            "gain": float(r2 - max(0.0, persistence)),
            "ridge": float(ridge),
        }
    return out


def carrier_report(
    *,
    key: str,
    carrier: Mapping[str, object],
    analysis: Mapping[str, object],
    retention: Mapping[str, object],
    bootstrap: Mapping[str, object],
    gain_bootstrap: Mapping[str, object] | None = None,
    predictive: Mapping[str, object],
    split_half: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    """One carrier's complete record, ready to write."""

    payload = {
        key_name: value for key_name, value in analysis.items()
        if key_name not in {"companion", "blocks"}
    }
    payload["retention"] = dict(retention)
    payload["retention_bootstrap"] = dict(bootstrap)
    #: Absolute composed gain, the same statistic without the within-carrier denominator.
    payload["composed_gain_bootstrap"] = dict(gain_bootstrap or {})
    payload["aperture_target_gain"] = dict(predictive)
    payload["split_half_companion"] = dict(split_half)
    payload["carrier"] = {
        "key": key,
        "family": carrier["family"],
        "rank": int(carrier["rank"]),
        "construction": carrier["construction"],
        "fitted_on": carrier["fitted_on"],
    }
    payload["thresholds"] = dict(thresholds)
    return payload


def retention_table(
    per_carrier: Mapping[str, Mapping[str, object]], *, lags: Sequence[int] = LAGS
) -> dict[str, object]:
    """A small plot-ready table: retention by lag for every carrier."""

    rows = []
    for key, record in per_carrier.items():
        entry: dict[str, object] = {
            "carrier": key,
            "family": record["carrier"]["family"],  # type: ignore[index]
            "rank": record["carrier"]["rank"],  # type: ignore[index]
        }
        for lag in lags:
            lag_key = str(int(lag))
            retention = record["retention"].get(lag_key, {})  # type: ignore[union-attr]
            bootstrap = record["retention_bootstrap"].get(lag_key, {})  # type: ignore[union-attr]
            value = retention.get("p_comp")
            entry[f"p_comp_{lag_key}"] = value if not isinstance(value, str) else None
            entry[f"p_comp_{lag_key}_q05"] = bootstrap.get("q05")
            entry[f"p_comp_{lag_key}_q95"] = bootstrap.get("q95")
            entry[f"direct_gain_{lag_key}"] = retention.get("direct_gain")
            per_lag = record["per_lag"].get(lag_key, {})  # type: ignore[union-attr]
            entry[f"inside_envelope_{lag_key}"] = per_lag.get("inside_envelope")
        rows.append(entry)
    return {"rows": rows, "lags": [int(lag) for lag in lags]}
