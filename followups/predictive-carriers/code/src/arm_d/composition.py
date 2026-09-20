"""Chapman-Kolmogorov composition with an estimation envelope (R1 §19).

Three things distinguish this from R0's CK test.

* The base lag is ``delta in {4, 8}``, chosen by synthetic qualification.  At
  ``delta = 1`` the operator is near-identity and estimator noise compounds
  multiplicatively over ``m = k / delta`` applications, so the test becomes a
  test of the estimator.
* Radial (gain) and directional (shape) mismatch are reported separately.  They
  are different scientific facts and are never pooled.
* D3 must clear the document-bootstrap envelope: propagate *every* bootstrap
  ``A_delta`` to lag ``k`` and measure how much CK error pure estimation
  uncertainty already produces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from arm_d.operators import fit_full, select_ridge
from arm_d.scoring import DocumentScore, document_index, equal_document_r2


@dataclass(frozen=True)
class CompositionResult:
    lag: int
    base_lag: int
    multiplicity: int
    action_error: float
    radial_error: float
    directional_error: float
    r2_direct: float
    r2_propagated: float
    r2_loss: float
    envelope_median: float
    envelope_upper: float
    clears_envelope: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "lag": int(self.lag),
            "base_lag": int(self.base_lag),
            "multiplicity": int(self.multiplicity),
            "ck_action_error": float(self.action_error),
            "ck_radial_error": float(self.radial_error),
            "ck_directional_error": float(self.directional_error),
            "r2_direct": float(self.r2_direct),
            "r2_propagated": float(self.r2_propagated),
            "r2_loss": float(self.r2_loss),
            "bootstrap_envelope_median": float(self.envelope_median),
            "bootstrap_envelope_upper": float(self.envelope_upper),
            "clears_estimation_envelope": bool(self.clears_envelope),
        }


def action_error(direct: np.ndarray, propagated: np.ndarray, source: np.ndarray) -> float:
    r"""``E_CK = E||(A_direct - A_prop) z||^2 / (E||A_direct z||^2 + eps)``."""

    x = np.asarray(source, dtype=np.float64)
    difference = x @ (np.asarray(direct) - np.asarray(propagated)).T
    reference = x @ np.asarray(direct).T
    denominator = float(np.mean(np.sum(reference**2, axis=1)))
    if denominator <= 0:
        return float("inf")
    return float(np.mean(np.sum(difference**2, axis=1)) / denominator)


def radial_and_directional_error(
    direct: np.ndarray,
    propagated: np.ndarray,
    source: np.ndarray,
    document_ids: np.ndarray,
) -> tuple[float, float]:
    """Separate gain mismatch from shape mismatch, equal document weight."""

    x = np.asarray(source, dtype=np.float64)
    a = x @ np.asarray(direct).T
    b = x @ np.asarray(propagated).T
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    usable = (norm_a > 0) & (norm_b > 0)
    if not usable.any():
        return float("inf"), float("inf")
    radial_rows = np.abs(np.log(norm_b[usable] / norm_a[usable]))
    cosine = np.sum(a[usable] * b[usable], axis=1) / (norm_a[usable] * norm_b[usable])
    directional_rows = 1.0 - np.clip(cosine, -1.0, 1.0)
    labels, codes = document_index(np.asarray(document_ids)[usable])
    n_documents = int(labels.shape[0])
    counts = np.bincount(codes, minlength=n_documents).astype(float)
    radial = np.bincount(codes, weights=radial_rows, minlength=n_documents) / counts
    directional = np.bincount(codes, weights=directional_rows, minlength=n_documents) / counts
    return float(np.mean(radial)), float(np.mean(directional))


def bootstrap_operator_envelope(
    source_train: np.ndarray,
    destination_train: np.ndarray,
    documents_train: np.ndarray,
    *,
    base_operator: np.ndarray,
    multiplicity: int,
    ridge: float,
    source_eval: np.ndarray,
    replicates: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """CK error attributable to estimation noise in ``A_delta`` alone (R1 §19.3).

    Each replicate resamples documents, refits ``A_delta``, propagates it to
    lag ``k``, and measures its action error against the *point-estimate*
    propagated operator.  If the observed direct-versus-propagated error sits
    inside this band, the CK "failure" is estimator noise, not a broken
    semigroup -- and equally, a CK "success" inside the band is not evidence.
    """

    labels, codes = document_index(documents_train)
    n_documents = int(labels.shape[0])
    rows_by_document = [np.flatnonzero(codes == index) for index in range(n_documents)]
    reference = np.linalg.matrix_power(np.asarray(base_operator, dtype=np.float64), multiplicity)
    rng = np.random.default_rng(seed)
    errors = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        take = rng.integers(0, n_documents, size=n_documents)
        rows = np.concatenate([rows_by_document[index] for index in take])
        operator = fit_full(source_train[rows], destination_train[rows], ridge)
        propagated = np.linalg.matrix_power(operator, multiplicity)
        errors[replicate] = action_error(reference, propagated, source_eval)
    return float(np.median(errors)), float(np.quantile(errors, 0.95))


def evaluate_composition(
    *,
    coordinates: dict[int, dict[str, np.ndarray]],
    base_lag: int,
    lags: tuple[int, ...],
    ridge_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0, 1e2),
    envelope_replicates: int = 200,
    seed: int = 42,
    base_operator_transform=None,
    diagnostics: dict | None = None,
) -> list[CompositionResult]:
    """Fit direct operators at every lag and test ``A_k`` against ``A_delta^m``.

    ``coordinates[k]`` supplies ``source_train``, ``destination_train``,
    ``documents_train``, ``source_eval``, ``destination_eval``,
    ``documents_eval``.
    """

    base = coordinates[base_lag]
    base_ridge = select_ridge(
        base["source_train"], base["destination_train"], base["documents_train"], "full", ridge_grid=ridge_grid, seed=seed
    )
    base_operator = fit_full(base["source_train"], base["destination_train"], base_ridge)
    # The operator that is actually iterated.  Its spectral radius is reported
    # because a radius above one amplifies magnitudes geometrically with
    # multiplicity, which can produce an enormous composition error while
    # leaving direction largely intact.
    raw_radius = spectral_radius(base_operator)
    if base_operator_transform is not None:
        base_operator = base_operator_transform(base_operator)
    if diagnostics is not None:
        diagnostics["base_lag"] = int(base_lag)
        diagnostics["base_ridge"] = float(base_ridge)
        diagnostics["base_ridge_grid"] = [float(value) for value in ridge_grid]
        diagnostics["base_spectral_radius"] = float(raw_radius)
        diagnostics["base_spectral_radius_after_transform"] = float(
            spectral_radius(base_operator)
        )
        diagnostics["base_operator_is_expansive"] = bool(
            np.isfinite(raw_radius) and raw_radius > 1.0
        )
        diagnostics["shape_only_r2_by_multiplicity"] = {}

    results: list[CompositionResult] = []
    for lag in lags:
        if lag == base_lag or lag % base_lag != 0:
            continue
        multiplicity = lag // base_lag
        entry = coordinates[lag]
        ridge = select_ridge(
            entry["source_train"], entry["destination_train"], entry["documents_train"], "full", ridge_grid=ridge_grid, seed=seed
        )
        direct = fit_full(entry["source_train"], entry["destination_train"], ridge)
        propagated = np.linalg.matrix_power(base_operator, multiplicity)

        direct_score: DocumentScore = equal_document_r2(
            entry["destination_eval"], entry["source_eval"] @ direct.T, entry["documents_eval"]
        )
        propagated_score: DocumentScore = equal_document_r2(
            entry["destination_eval"], entry["source_eval"] @ propagated.T, entry["documents_eval"]
        )
        if diagnostics is not None:
            diagnostics["shape_only_r2_by_multiplicity"][str(int(multiplicity))] = (
                shape_only_score(
                    entry["destination_eval"],
                    entry["source_eval"] @ propagated.T,
                    entry["documents_eval"],
                )
            )
        error = action_error(direct, propagated, entry["source_eval"])
        radial, directional = radial_and_directional_error(
            direct, propagated, entry["source_eval"], entry["documents_eval"]
        )
        envelope_median, envelope_upper = bootstrap_operator_envelope(
            base["source_train"],
            base["destination_train"],
            base["documents_train"],
            base_operator=base_operator,
            multiplicity=multiplicity,
            ridge=base_ridge,
            source_eval=entry["source_eval"],
            replicates=envelope_replicates,
            seed=seed + lag,
        )
        results.append(
            CompositionResult(
                lag=lag,
                base_lag=base_lag,
                multiplicity=multiplicity,
                action_error=error,
                radial_error=radial,
                directional_error=directional,
                r2_direct=direct_score.r2,
                r2_propagated=propagated_score.r2,
                r2_loss=direct_score.r2 - propagated_score.r2,
                envelope_median=envelope_median,
                envelope_upper=envelope_upper,
                clears_envelope=bool(error <= envelope_upper),
            )
        )
    return results


# --------------------------------------------------------------------------
# R1 section 19.2 diagnosis: gain versus shape, and whether an expansive fit
# explains the collapse
# --------------------------------------------------------------------------

SECTION_19_2 = "R1 section 19.2 (radial and directional error are never pooled)"
SECTION_19_3 = "R1 section 19.3 (operator-estimation bootstrap envelope)"

# Above this multiplicative magnitude error the composed prediction is wrong
# about scale by more than a factor of two, which is what "gain does not
# compose" means quantitatively.
GAIN_FAILURE_LOG_RATIO = float(np.log(2.0))
# Above this cosine the composed direction still carries most of the shape.
SHAPE_RETAINED_COSINE = 0.25
# An envelope this wide cannot discriminate: the verdict is power-limited.
WIDE_ENVELOPE_UPPER = 1.0


def spectral_radius(operator: np.ndarray) -> float:
    """Largest eigenvalue modulus of a square operator."""

    array = np.asarray(operator, dtype=np.float64)
    if array.size == 0 or array.ndim != 2 or array.shape[0] != array.shape[1]:
        return float("nan")
    return float(np.max(np.abs(np.linalg.eigvals(array))))


def clip_spectral_radius(operator: np.ndarray, *, maximum: float = 0.99) -> np.ndarray:
    """Rescale an operator so its spectral radius is at most ``maximum``.

    A DIAGNOSTIC ONLY, never the primary fit.  Iterating an operator whose
    spectral radius exceeds one amplifies the excess geometrically: at
    multiplicity 32 a radius of 1.15 multiplies magnitudes by about 80 on its
    own, which says nothing about whether the model's dynamics compose.  Uniform
    rescaling is the least invasive way to ask whether the composition failure
    survives removing that amplification, because it changes gain only and
    leaves every eigenvector direction untouched.
    """

    radius = spectral_radius(operator)
    array = np.asarray(operator, dtype=np.float64)
    if not np.isfinite(radius) or radius <= maximum or radius <= 0:
        return array
    return array * (maximum / radius)


def shape_only_score(
    destination: np.ndarray,
    prediction: np.ndarray,
    documents: np.ndarray,
) -> float:
    """Equal-document R-squared after rescaling each prediction to matched norm.

    Isolates shape agreement from gain agreement: every prediction row is
    rescaled to its target row's norm, so only direction is scored.  A normalized
    companion to the raw statistic and never a replacement, because a model that
    gets direction right and magnitude wrong has not composed.
    """

    target = np.asarray(destination, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    usable = predicted_norm > 0
    rescaled = np.zeros_like(predicted)
    rescaled[usable] = (
        predicted[usable] * (target_norm[usable] / predicted_norm[usable])[:, None]
    )
    return float(equal_document_r2(target, rescaled, documents).r2)


def composition_verdict(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """State the composition outcome in the two channels R1 section 19.2 separates.

    The pooled ``action_error`` verdict says only "composition fails", which
    hides the actual finding.  On this run the composed prediction is wrong about
    MAGNITUDE by a large, geometrically growing factor while still holding a
    positive direction cosine.  Those are different claims about the model, and
    section 19.2 requires that the two channels never be pooled.
    """

    rows: list[dict[str, object]] = []
    for entry in results:
        radial = float(entry.get("ck_radial_error", float("nan")))
        directional = float(entry.get("ck_directional_error", float("nan")))
        upper = float(entry.get("bootstrap_envelope_upper", float("nan")))
        cosine = 1.0 - directional if np.isfinite(directional) else float("nan")
        rows.append(
            {
                "multiplicity": int(entry.get("multiplicity", 0)),
                "lag": int(entry.get("lag", 0)),
                "radial_error_log_ratio": radial,
                "magnitude_error_factor": (
                    float(np.exp(radial)) if np.isfinite(radial) else float("nan")
                ),
                "directional_error": directional,
                "direction_cosine": cosine,
                "gain_fails": bool(np.isfinite(radial) and radial > GAIN_FAILURE_LOG_RATIO),
                "shape_retained": bool(np.isfinite(cosine) and cosine > SHAPE_RETAINED_COSINE),
                "envelope_upper": upper,
                # A non-finite envelope is an unmeasured envelope.  Treating it as
                # narrow would advertise power the run does not have.
                "envelope_is_wide": bool(
                    (not np.isfinite(upper)) or upper > WIDE_ENVELOPE_UPPER
                ),
                "envelope_measured": bool(np.isfinite(upper)),
                "measured": bool(np.isfinite(radial) and np.isfinite(directional)),
                "r2_direct": entry.get("r2_direct"),
                "r2_propagated": entry.get("r2_propagated"),
            }
        )

    # gain_fails is False for a non-finite radial error, so a table of rows that
    # were never measured would fall through to COMPOSITION_CONSISTENT: a
    # substantive positive claim produced by the absence of a measurement.  The
    # verdict is computed over measured rows only, and says so when there are none.
    measured = [row for row in rows if row["measured"]]
    gain_failures = [row for row in measured if row["gain_fails"]]
    shape_kept = [row for row in measured if row["shape_retained"]]
    discriminating = [
        row["multiplicity"]
        for row in rows
        if row["envelope_measured"] and not row["envelope_is_wide"]
    ]
    if not rows or not measured:
        label = "COMPOSITION_UNAVAILABLE"
    elif not gain_failures:
        label = "COMPOSITION_CONSISTENT"
    elif len(shape_kept) >= len(measured) - 1:
        label = "GAIN_DOES_NOT_COMPOSE_SHAPE_LARGELY_RETAINED"
    else:
        label = "GAIN_AND_SHAPE_BOTH_FAIL"

    power_note = None
    if rows:
        power_note = (
            f"the estimation envelope upper bound grows from {rows[0]['envelope_upper']:.5g} at "
            f"multiplicity {rows[0]['multiplicity']} to {rows[-1]['envelope_upper']:.5g} at "
            f"multiplicity {rows[-1]['multiplicity']}. A failure against the narrow "
            f"low-multiplicity envelope is a far stronger statement than one against the wide "
            f"high-multiplicity envelope; the test discriminates only at multiplicities "
            f"{discriminating}"
        )
    return {
        "label": label,
        "citation": SECTION_19_2,
        "envelope_citation": SECTION_19_3,
        "per_multiplicity": rows,
        "multiplicities_with_discriminating_envelope": discriminating,
        "rows_measured": len(measured),
        "rows_total": len(rows),
        "power_note": power_note,
    }
