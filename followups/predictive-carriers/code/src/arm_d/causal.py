"""Causal battery: perturbation, multi-horizon propagation, controls (R1 §27-§29).

One perturbation is applied at the frozen source hook and followed through
every frozen horizon in a *single* patched forward pass, because a one-step
check cannot distinguish propagation from immediate erasure.  Every patch
reports its health (relative residual change, destination norm change,
next-token NLL), and the hooked path is checked against the model-free
reference formula before any causal number is read.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

from src.persistent_state.live_state.patching import apply_residual_patch

CONTROL_NAMES = (
    "noop",
    "full_residual",
    "q_subspace",
    "pca_subspace",
    "random_subspace",
    "sign_reversed",
    "norm_matched",
    "shuffled_donor",
    "same_suffix_donor",
)


@dataclass(frozen=True)
class PropagationMetrics:
    lag: int
    cosine: float
    gain: float
    normalized_error: float
    observed_norm: float
    predicted_norm: float
    correction_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "lag": int(self.lag),
            "directional_agreement_cosine": float(self.cosine),
            "gain_agreement": float(self.gain),
            "normalized_prediction_error": float(self.normalized_error),
            "observed_shift_norm": float(self.observed_norm),
            "predicted_shift_norm": float(self.predicted_norm),
            "correction_toward_unpatched": float(self.correction_ratio),
        }


def propagation_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    lag: int,
    source_perturbation: np.ndarray,
    epsilon: float = 1e-12,
) -> PropagationMetrics:
    """R1 §27.2 metrics for one horizon, averaged over rows.

    ``correction_toward_unpatched`` is the observed destination shift relative
    to the injected source shift.  A value decaying toward zero with lag while
    the cosine stays high is propagation-then-decay; a value at zero with a
    meaningless cosine is erasure.
    """

    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    delta = np.asarray(source_perturbation, dtype=np.float64)
    obs_norm = np.linalg.norm(obs, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    usable = (obs_norm > 0) & (pred_norm > 0)
    if not usable.any():
        return PropagationMetrics(lag, 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0)
    cosine = np.sum(obs[usable] * pred[usable], axis=1) / (obs_norm[usable] * pred_norm[usable])
    gain = obs_norm[usable] / (pred_norm[usable] + epsilon)
    error = np.sum((obs[usable] - pred[usable]) ** 2, axis=1) / (np.sum(pred[usable] ** 2, axis=1) + epsilon)
    return PropagationMetrics(
        lag=int(lag),
        cosine=float(np.mean(cosine)),
        gain=float(np.median(gain)),
        normalized_error=float(np.median(error)),
        observed_norm=float(np.mean(obs_norm)),
        predicted_norm=float(np.mean(pred_norm)),
        correction_ratio=float(np.mean(obs_norm / (np.linalg.norm(delta, axis=1) + epsilon))),
    )


def verify_patch_against_reference(
    base: np.ndarray,
    donor: np.ndarray,
    basis_columns: np.ndarray,
    alpha: float,
    hooked_result: np.ndarray,
    tolerance: float = 1e-3,
) -> dict[str, float | bool]:
    """Check the hooked patch reproduces the model-free reference formula.

    The reference is the researcher's own ``apply_residual_patch``, which is
    exactly ``h' = h + alpha Q Q^T (h_src - h)`` for an orthonormal row basis.
    R1 §28 requires this before any causal result is interpreted.
    """

    reference = apply_residual_patch(
        np.asarray(base, dtype=np.float64),
        np.asarray(donor, dtype=np.float64),
        basis_rows=np.asarray(basis_columns, dtype=np.float64).T,
        alpha=float(alpha),
    )
    hooked = np.asarray(hooked_result, dtype=np.float64)
    scale = float(np.linalg.norm(reference)) or 1.0
    deviation = float(np.linalg.norm(hooked - reference) / scale)
    return {
        "relative_deviation_from_reference": deviation,
        "tolerance": float(tolerance),
        "matches_reference": bool(deviation <= tolerance),
    }


def patch_health(
    base_source: np.ndarray,
    patched_source: np.ndarray,
    base_destination: np.ndarray,
    patched_destination: np.ndarray,
    base_nll: np.ndarray,
    patched_nll: np.ndarray,
) -> dict[str, float]:
    """R1 §27.3 perturbation health, reported for every patch condition."""

    base_source = np.asarray(base_source, dtype=np.float64)
    delta = np.asarray(patched_source, dtype=np.float64) - base_source
    source_norm = np.linalg.norm(base_source, axis=-1)
    destination_base = np.linalg.norm(np.asarray(base_destination, dtype=np.float64), axis=-1)
    destination_patched = np.linalg.norm(np.asarray(patched_destination, dtype=np.float64), axis=-1)
    return {
        "relative_source_change": float(np.mean(np.linalg.norm(delta, axis=-1) / np.clip(source_norm, 1e-12, None))),
        "destination_norm_ratio": float(np.mean(destination_patched / np.clip(destination_base, 1e-12, None))),
        # NaN marks an offset whose next-token target falls past the end of the
        # sequence and so has no likelihood to score; those entries are skipped
        # rather than treated as zero loss.
        "mean_nll_unpatched": float(np.nanmean(base_nll)),
        "mean_nll_patched": float(np.nanmean(patched_nll)),
        "mean_nll_increase": float(np.nanmean(patched_nll) - np.nanmean(base_nll)),
    }


def relay_label(
    *,
    positive_control_healthy: bool,
    q_propagates: bool,
    q_above_controls: bool,
    aligned_with_transition: bool,
    prefix_gain_large: bool,
    multiple_subspaces_propagate: bool,
) -> str:
    """R1 §29.1.  Never assigned from observational composition alone."""

    # bool(float("nan")) is True, so a non-finite quantity coerced on the way in
    # would arrive as a satisfied gate and could only ever push the verdict
    # toward the positive labels.  Require genuine booleans.
    supplied = {
        "positive_control_healthy": positive_control_healthy,
        "q_propagates": q_propagates,
        "q_above_controls": q_above_controls,
        "aligned_with_transition": aligned_with_transition,
        "prefix_gain_large": prefix_gain_large,
        "multiple_subspaces_propagate": multiple_subspaces_propagate,
    }
    unusable = [
        name
        for name, value in supplied.items()
        if not isinstance(value, (bool, np.bool_))
    ]
    if unusable:
        return "RELAY_UNAVAILABLE"

    if not positive_control_healthy:
        return "NOT_EVALUATED"
    if q_propagates and q_above_controls and aligned_with_transition:
        if multiple_subspaces_propagate and prefix_gain_large:
            return "MULTIROUTE_STATE"
        return "CAUSAL_RELAY_SUPPORTED"
    if prefix_gain_large or not q_propagates:
        return "RECONSTRUCTION_DOMINANT"
    return "NOT_EVALUATED"


def storage_locus_label(
    current_token_effect: float,
    preceding_window_effect: float,
    combined_effect: float,
    *,
    ratio_threshold: float = 3.0,
) -> str:
    """R1 §29.2 descriptive storage label from the three intervention scopes."""

    # A non-finite scope is an unmeasured scope, not a zero effect.  Coercing it
    # through max(x, 0.0) turned NaN into a substantive MULTIROUTE_STATE verdict.
    missing = [
        name
        for name, value in (
            ("current_token_effect", current_token_effect),
            ("preceding_window_effect", preceding_window_effect),
            ("combined_effect", combined_effect),
        )
        if not math.isfinite(float(value))
    ]
    if missing:
        return "STORAGE_LOCUS_UNAVAILABLE"
    current = max(float(current_token_effect), 0.0)
    window = max(float(preceding_window_effect), 0.0)
    if current <= 0 and window <= 0:
        return "NOT_EVALUATED"
    if window <= 0 or current / max(window, 1e-12) >= ratio_threshold:
        return "CURRENT_TOKEN_STATE"
    if current <= 0 or window / max(current, 1e-12) >= ratio_threshold:
        return "DISTRIBUTED_HISTORY_STATE"
    return "MULTIROUTE_STATE"
