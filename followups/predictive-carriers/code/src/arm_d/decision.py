"""Rung assignment, the label inventory, and the allowed language (R1 §29.3, §37).

Exactly one highest-supported rung is assigned.  Labels downstream of an
unreached rung are ``NOT_EVALUATED``, never guessed (R1 addition A7), and
spectral language is withheld whenever the shared-space gate fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RUNGS = (
    "D0_NO_REMOTE_HISTORY_STATE_DYNAMICS",
    "D1_REMOTE_HISTORY_STATE_DYNAMICS",
    "D2_STABLE_FIXED_STATE_SUBSPACE",
    "D3_COMPOSABLE_STATE_DYNAMICS",
    "D4_CAUSAL_STATE_PROPAGATION",
    "D5_APPROXIMATE_PREDICTIVE_STATE",
)

ALLOWED_LANGUAGE = {
    "D0_NO_REMOTE_HISTORY_STATE_DYNAMICS": (
        "Under this design, remote-prefix-induced residual differences showed no reproducible "
        "low-dimensional predictive structure beyond the strongest trajectory baseline and the "
        "splice floor."
    ),
    "D1_REMOTE_HISTORY_STATE_DYNAMICS": (
        "After holding the local and intervening token sequence fixed, a change in remote prefix "
        "induces a residual difference with reproducible low-dimensional linear structure across "
        "sequence time."
    ),
    "D2_STABLE_FIXED_STATE_SUBSPACE": (
        "That structure is localized to a reproducible fixed residual subspace rather than an "
        "unstable fitted map."
    ),
    "D3_COMPOSABLE_STATE_DYNAMICS": (
        "Dynamics inside the fixed residual subspace are approximately consistent across lags with "
        "a shared transition law."
    ),
    "D4_CAUSAL_STATE_PROPAGATION": (
        "Perturbations within the fixed predictive subspace propagate through the model in the "
        "direction predicted by its fitted transition law, at every tested horizon."
    ),
    "D5_APPROXIMATE_PREDICTIVE_STATE": (
        "After holding the local and intervening token sequence fixed, changes in remote prefix "
        "induce a low-dimensional residual-state difference that evolves through a stable fixed "
        "coordinate system. The transition law composes across lags beyond estimation uncertainty, "
        "current state plus incoming update information captures most of the measured future-state "
        "evolution, and causal perturbations of the same coordinates propagate to future positions "
        "according to the fitted dynamics."
    ),
    "D_TECHNICAL_FAILURE": (
        "The Arm D instrument did not qualify on this run; no scientific rung is claimed."
    ),
}

FORBIDDEN_PHRASES = ("the transformer is an rnn", "the kv cache is redundant")


@dataclass
class LabelInventory:
    claim_rung: str = "NOT_EVALUATED"
    shared_space: str = "NOT_EVALUATED"
    position_generality: str = "NOT_EVALUATED"
    splice_reading: str = "NOT_EVALUATED"
    relay_reconstruction: str = "NOT_EVALUATED"
    markov_order: str = "NOT_EVALUATED"
    storage_locus: str = "NOT_EVALUATED"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_rung": self.claim_rung,
            "shared_space": self.shared_space,
            "position_generality": self.position_generality,
            "splice_reading": self.splice_reading,
            "relay_reconstruction": self.relay_reconstruction,
            "markov_order": self.markov_order,
            "storage_locus": self.storage_locus,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RungEvidence:
    """Every boolean below is decided elsewhere and only assembled here."""

    qualified: bool
    noop_floor_ok: bool
    d1_detected: bool
    d1_beats_baseline: bool
    d1_clears_splice_floor: bool
    d2_split_half_stable: bool
    d3_composes: bool
    d4_positive_control_healthy: bool
    d4_propagates: bool
    d5_prefix_gain_small: bool
    d5_first_order: bool
    # d1_clears_splice_floor is the R1 section 14.3 content-floor margin and
    # nothing else.  d1_clears_trajectory_floor is retained ONLY so that the
    # quantity can be reported; it is deliberately NOT consulted by any D1 gate.
    # See the correction note on assign_rung.
    d1_clears_content_floor: bool | None = None
    d1_clears_trajectory_floor: bool | None = None


# Every gate below carries the R1 section that mandates that quantity for that
# rung.  A gate without a citation is a bug: this experiment assigned D0 from a
# condition R1 does not contain, because a D5 comparison was wired into the D1
# conjunction and nothing checked that the rule had a source.
GATE_CITATIONS: dict[str, str] = {
    "qualified": "R1 section 13 (synthetic qualification authorizes production)",
    "noop_floor_ok": "R1 section 16.2 (no-op splice returns the numerical floor)",
    "d1_detected": "R1 section 17, section 32 (full-refit null bank)",
    "d1_beats_baseline": (
        "R1 section 15.1 (incremental performance beyond the strongest applicable "
        "SOURCE-STATE baseline sets the D1 headline)"
    ),
    "d1_clears_splice_floor": "R1 section 14.3 (splice floor margin over no-op, sham, matched)",
    "d2_split_half_stable": "R1 section 18 (split-half subspace reproducibility)",
    "d3_composes": "R1 section 19.3 (composition against the operator-estimation envelope)",
    "d4_positive_control_healthy": "R1 section 21 (full-residual positive control)",
    "d4_propagates": "R1 section 21 (subspace-specific causal propagation)",
    "d5_prefix_gain_small": "R1 section 24 (richer-prefix information gain)",
    "d5_first_order": "R1 section 25 (Markov order of the state)",
}

# Gates that R1 does NOT assign to D1, recorded so that re-adding one is a
# visible change rather than a silent drift.
NOT_D1_GATES: dict[str, str] = {
    "d1_clears_trajectory_floor": (
        "R1 section 15.1 assigns the trajectory baselines to D5 and the Markov-order "
        "question (sections 24, 25); they are reported at D1 as CONTEXT only. Five noisy "
        "measurements of a decaying difference average out measurement noise that a "
        "single slot cannot, so they beat the operator whenever measurement noise is "
        "comparable to signal, whether or not any dynamics exist."
    )
}


def assign_rung(evidence: RungEvidence) -> tuple[str, list[str]]:
    """Return the highest supported rung and the reason the ladder stopped.

    The D1 gate is exactly what R1 section 15.1 and section 14.3 specify:
    detection against the null families, incremental performance beyond the
    strongest applicable SOURCE-STATE baseline, and the content-floor margin.

    CORRECTION.  An earlier revision also required the trajectory-relative
    statistic to clear a floor, which assigned D0 on a comparison R1 places at
    D5.  Section 15.1 states that the source-state baselines "set the D1
    headline" and that the trajectory baselines are "reported at D1 as context,
    and consumed formally at D5", and gives the reason: a five-slot comparator
    averages out measurement noise a single slot cannot, so it wins whenever
    noise is comparable to signal whether or not any dynamics exist. That
    advantage is not evidence about dynamics. The trajectory quantity is still
    computed and reported, as an input to D5 and the Markov-order analysis.
    """

    reasons: list[str] = []
    if not evidence.qualified:
        return "D_TECHNICAL_FAILURE", ["synthetic qualification did not pass its frozen bars"]
    if not evidence.noop_floor_ok:
        return "D_TECHNICAL_FAILURE", [
            "the exact no-op splice did not return a numerical-noise-level difference, so the "
            "primary object is not measurable"
        ]
    if not (evidence.d1_detected and evidence.d1_beats_baseline and evidence.d1_clears_splice_floor):
        if not evidence.d1_detected:
            reasons.append("the observed pipeline maximum did not clear the full-refit null bank")
        if not evidence.d1_beats_baseline:
            reasons.append(
                "the operator did not beat the strongest applicable source-state baseline "
                "(zero, persistence, update-only)"
            )
        if not evidence.d1_clears_splice_floor:
            reasons.append(
                "the effect did not clear the no-op, sham and matched-content splice floor "
                "margin required by R1 section 14.3"
            )
        return "D0_NO_REMOTE_HISTORY_STATE_DYNAMICS", reasons
    if not evidence.d2_split_half_stable:
        return "D1_REMOTE_HISTORY_STATE_DYNAMICS", [
            "the fitted subspace was not reproducible across independent train halves"
        ]
    if not evidence.d3_composes:
        return "D2_STABLE_FIXED_STATE_SUBSPACE", [
            "operators fitted independently at different lags did not compose beyond the "
            "bootstrap estimation envelope"
        ]
    if not evidence.d4_positive_control_healthy:
        return "D3_COMPOSABLE_STATE_DYNAMICS", [
            "the full-residual positive control was not healthy, so no subspace-specific causal "
            "result is interpretable"
        ]
    if not evidence.d4_propagates:
        return "D3_COMPOSABLE_STATE_DYNAMICS", [
            "perturbing the fitted coordinates did not shift future coordinates in the predicted "
            "direction above the matched controls"
        ]
    if not (evidence.d5_prefix_gain_small and evidence.d5_first_order):
        reasons.append(
            "richer prefix information or older state history still added materially to prediction"
        )
        return "D4_CAUSAL_STATE_PROPAGATION", reasons
    return "D5_APPROXIMATE_PREDICTIVE_STATE", []


def cumulative_language(rung: str, shared_space_label: str) -> list[str]:
    """The verbatim allowed sentences for the assigned rung and everything below."""

    if rung not in RUNGS:
        return [ALLOWED_LANGUAGE["D_TECHNICAL_FAILURE"]]
    index = RUNGS.index(rung)
    if index == 0:
        sentences = [ALLOWED_LANGUAGE[RUNGS[0]]]
    else:
        sentences = [ALLOWED_LANGUAGE[name] for name in RUNGS[1 : index + 1]]
    if shared_space_label == "SOURCE_DESTINATION_TRANSPORT":
        sentences = [
            sentence.replace("fixed coordinate system", "source-to-destination transport").replace(
                "fixed residual subspace", "source-to-destination transport geometry"
            )
            for sentence in sentences
        ]
        sentences.append(
            "The shared-space gate did not pass, so this run reports transport geometry only: no "
            "transition eigenvalues, decay times, or frequencies are given."
        )
    return sentences


def check_forbidden(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in lowered]


def spectral_language_permitted(shared_space_label: str, frequency_gate_passed: bool) -> dict[str, bool]:
    """R1 §12 and §20: what the report may and may not say about spectra."""

    fixed = shared_space_label == "FIXED_COORDINATE_DYNAMICS"
    return {
        "eigenvalues_permitted": fixed,
        "decay_times_permitted": fixed,
        "frequencies_permitted": bool(fixed and frequency_gate_passed),
    }
