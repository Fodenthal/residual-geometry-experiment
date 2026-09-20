"""The R2.4 contract: what is inherited, what this run freezes, and the amendment ledger.

Everything normative here is quoted from
``docs/specifications/arm_d/arm_d_r2_4_unified_specification.md``, which is the source
specification folded with the ten prospective amendments of its section 0.1.  Nothing in
this module may be recomputed from data.

The identifier discipline of amendment A10 is enforced here: ``S0..S6`` name development
steps, ``C0..C9b`` name confirmatory steps, ``D-AUDIT`` names the variance-deletion audit,
and ``P0_``/``P1_`` remain P-axis RESULT labels rather than step names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from arm_d_r2.contract import (  # noqa: F401
    APERTURE_WIDTH,
    BOOTSTRAP_REPLICATES,
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_REVISION,
    HISTORY_OFFSETS,
    InvariantError,
    MODEL_NAME,
    MODEL_REVISION,
    PRIMARY_LAYER,
    PRIMARY_RANK,
    REFERENCE_LAG,
    RunTree,
    SEED,
    SOURCE_OFFSETS,
    STATE_ORDER,
    assert_distinct,
    environment_note,
    is_unavailable,
    label as _label,
    propagate_unavailable,
    read_json,
    sha256_path,
    unavailable,
    write_json,
)

PROTOCOL_REVISION = "arm_d_r2_4_predictive_carrier_resolution_unified"
SPECIFICATION_PATH = "docs/specifications/arm_d/arm_d_r2_4_unified_specification.md"
SOURCE_SPECIFICATION_PATH = "docs/specifications/arm_d/arm_d_r2_4_source_specification.md"
SOURCE_SPECIFICATION_SHA256 = (
    "a566209ee26feaa74776801523a101368d249280973df896710e8ef69c9f26b2"
)

SOURCE_EXPERIMENT_R1 = "exp_01kzmd2x1hfaas5are07m3bqpe"
SOURCE_EXPERIMENT_R2_2 = "exp_01kzs0q57aet88bm4svwasaxex"
SOURCE_EXPERIMENT_R2_3 = "exp_01kzsrwrndepmbc5jd18t1npf7"
SOURCE_EXPERIMENT_R2_3C = "exp_01kzt4r3zfexna89meakdkxr3r"

# --------------------------------------------------------------------------
# Section references, used by every label (invariant I1)
# --------------------------------------------------------------------------

SECTION_LEDGER = "R2.4 unified section 0.1 (amendment ledger)"
SECTION_OBJECT = "R2.4 unified section 2 (the analysis object)"
SECTION_SCORE = "R2.4 unified section 3 (the score, amendment A1)"
SECTION_OPERATOR = "R2.4 unified section 4 (lag normalization, amendment A2)"
SECTION_RIDGE = "R2.4 unified section 5 (ridge scales, amendment A5)"
SECTION_CEILING = "R2.4 unified section 6 (ceilings, amendment A7)"
SECTION_DELETION = "R2.4 unified section 7 (deletion, amendments A3 and A4)"
SECTION_STABILITY = "R2.4 unified section 8 (stability, nulls, metric geometry)"
SECTION_SIZING = "R2.4 unified section 9 (sizing and reproduction, amendment A8)"
SECTION_WORLDS = "R2.4 unified section 10 (required worlds, amendment A6)"
SECTION_HANDOFF = "R2.4 unified section 11 (the M-stage data rule, amendment A9)"
SECTION_DATA = "R2.4 unified section 12 (data boundary)"
SECTION_NUMERICAL = "R2.4 unified section 13 (numerical contract)"
SECTION_DIRECTION = "R2.4 source section 6 (direction versus magnitude)"
SECTION_COMPARATORS = "R2.4 source section 11 (descriptive comparators)"
SECTION_LABELS = "R2.4 source section 13 (independent result axes)"
SECTION_PROMOTION = "R2.4 source section 15 (confirmatory promotion rule)"

# --------------------------------------------------------------------------
# The analysis object (unified section 2)
# --------------------------------------------------------------------------

ANALYSIS_CONDITION = "main"
ANALYSIS_SUFFIX = 64
LAGS_Q = (1, 2, 4, 8, 16, 32, 64)
DESCRIPTIVE_LAG = 128
LONG_HORIZON_LAGS = (8, 16, 32, 64)
RANK_GRID = (1, 2, 4, 8, 16, 32, 64)
LARGEST_RANK = 64
REFERENCE_RANK = 16

#: The four-way confirmatory document split of unified section 12.
CONFIRM_SPLIT_FRACTIONS = {"fitA": 0.35, "fitB": 0.35, "validation": 0.15, "test": 0.15}
CONFIRM_CANDIDATE_SIZES = (2000, 4000, 8000)

# --------------------------------------------------------------------------
# Estimator constants (unified sections 3 to 6)
# --------------------------------------------------------------------------

RHO_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
#: The fixed metric every ridge candidate is SCORED through, so the candidates are compared
#: on one yardstick rather than each being graded by the metric its own ridge defines.
REFERENCE_SCORING_RHO = 1e-2
#: Amendment A13.  The inherited grid stopped at 1e4, and on DEV-Q every decoder in the
#: scoring path selected exactly that top value -- an edge-selected regularizer in the
#: quantity that decides the run.  The grid is extended upward so the choice is interior,
#: and every scored carrier now records whether its selection still sits at an edge.
RIDGE_GRID = (1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8, 1e10)
#: Amendment A14.  The baseline bank is ordered from the least presumptuous prediction
#: upward, and a richer candidate must beat the incumbent by this much held-out R-squared to
#: displace it.  Without a margin the bank's winner can flip on noise: on DEV-Q the increment
#: baseline took lag 32 with +1.933e-5 against the zero baseline's exact zero, at a ridge of
#: 1e6 where its prediction is numerically nil.  The label mattered, the prediction did not.
BASELINE_MARGIN = 1e-3
BASELINE_ORDER = ("B0", "B1", "B2")
MAX_CONDITION_NUMBER = 1e8
ESTIMATORS = ("Q_E", "Q_W")
#: Amendment A11.  Every label, ratio and curve rides on the baseline-augmented score; the
#: literal source-section-3 form is co-reported under ``gamma_raw``.
PRIMARY_SCORE = "gamma_augmented"
CO_REPORTED_SCORES = ("gamma_raw", "gamma_white")
#: Positive-gain floor below which a surviving-fraction ratio is not formed.
POSITIVE_GAIN_FLOOR = 0.02

# --------------------------------------------------------------------------
# Deletion families (unified section 7)
# --------------------------------------------------------------------------

DELETION_HAAR_DRAWS = 20
DELETION_MIXED_M = (0, 2, 4, 8, 12, 16)
DELETION_ROTATION_ANGLES = (5.0, 10.0, 20.0, 40.0, 60.0, 90.0)
DELETION_DOSE_RANKS = (2, 4, 8, 16, 24, 32, 48, 64, 96)
#: Amendment A12: the doses at which the top-versus-lower geometry contrast is run.  Every
#: value has to be attainable by BOTH constructions, so the grid stops inside the variance
#: the non-leading directions hold.
DELETION_DOSE_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
DELETION_SEED_NAMESPACE = "arm_d_r2_4_deletion"
#: R2.3c measured these on the same aperture; they are context, not inputs.
R23C_PCA16_VARIANCE_CAPTURED = 0.6722328162158252
R23C_AMBIENT_HAAR_VARIANCE_CAPTURED = 0.06806629096444501
R23C_NEAREST_ENERGY_MATCHED_MSC = 0.9375

# --------------------------------------------------------------------------
# Stability and nulls (unified section 8)
# --------------------------------------------------------------------------

NULL_REPLICATES = 400
BOOTSTRAP_OVERLAP_REPLICATES = 200
NULL_SEED_NAMESPACE = "arm_d_r2_4_pairing_null"
BOOTSTRAP_SEED_NAMESPACE = "arm_d_r2_4_bootstrap_overlap"
GAIN_BOOTSTRAP = 2000
LOWER_BOUND_QUANTILE = 0.05

# --------------------------------------------------------------------------
# Thresholds frozen at S6.  Every value here is a DEFAULT that the S6 freeze
# record either confirms or replaces; nothing downstream may read them from
# this module without also recording the frozen value it used.
# --------------------------------------------------------------------------

COMPRESSION_FRACTION = 0.90  # source section 9, not amendable
DEFAULT_THRESHOLDS = {
    "compression_fraction": COMPRESSION_FRACTION,
    "metric_concordance_fraction": None,
    "ridge_stability_min": None,
    "deletion_margin_min": None,
    "redundancy_relative_high": None,
    "redundancy_relative_low": None,
    "sizing_msc_tolerance": None,
    "sizing_se_max": None,
    "reproduction_min_fraction": None,
    "stability_alpha": 1.0 / (NULL_REPLICATES + 1),
}

# --------------------------------------------------------------------------
# The amendment ledger (unified section 0.1)
# --------------------------------------------------------------------------

AMENDMENT_DATE = "2026-08-12"
AMENDMENT_WHEN = "before any R2.4 development number was computed"
AMENDMENT_LEDGER = (
    {
        "id": "A1",
        "date": AMENDMENT_DATE,
        "when": AMENDMENT_WHEN,
        "amends": "source sections 8.1 and 8.2",
        "change": (
            "the multi-horizon operators are built from the persistence-residualized "
            "cross-covariance C_k = E[x_t (y_{t,k} - b_k(x_t))^T], and Q-W whitens by the "
            "covariance of that same residualized object; the raw-C_k form is a diagnostic"
        ),
        "rationale": (
            "the score is a gain over persistence, so an objective built on the raw target "
            "is rewarded for predicting the present, which dominates lags 1 and 2 and "
            "readmits the energy confound the whitened metric was chosen to exclude"
        ),
    },
    {
        "id": "A2",
        "date": AMENDMENT_DATE,
        "when": AMENDMENT_WHEN,
        "amends": "source section 8.3",
        "change": "each lag's operator term is trace-normalized before summing",
        "rationale": (
            "equal weights apply to objective terms rather than to scores and "
            "cross-covariance magnitude decays with lag, so the unnormalized sum yields a "
            "short-lag carrier wearing a multi-horizon label"
        ),
    },
    {
        "id": "A3",
        "date": AMENDMENT_DATE,
        "when": AMENDMENT_WHEN,
        "amends": "source sections 5.3, 5.4 and 12.1",
        "change": (
            "deletion family F (energy-matched at varying rank) is REQUIRED and is the "
            "primary L-axis comparator, with the dose-response curve as continuous context"
        ),
        "rationale": (
            "the impossibility argument forbids matching rank and captured variance at "
            "once near the variance-maximizing subspace, not matching variance at higher rank"
        ),
    },
    {
        "id": "A4",
        "date": AMENDMENT_DATE,
        "when": AMENDMENT_WHEN,
        "amends": "source sections 5.4 and 19",
        "change": (
            "step C6b replays the frozen deletion family on CONFIRM-Q; until it runs, "
            "L1_GEOMETRY_SPECIFIC_DELETION may not be emitted"
        ),
        "rationale": (
            "section 5.4 requires the replay but the section 19 C-sequence has no step for "
            "it, and C6 answers a different question"
        ),
    },
    {
        "id": "A5",
        "date": AMENDMENT_DATE,
        "when": AMENDMENT_WHEN,
        "amends": "source sections 8 and 17",
        "change": (
            "ridge stability is a reported axis: both estimators are refit across the frozen "
            "rho grid and the subspace correlation between variants decides R1/R0"
        ),
        "rationale": (
            "a ridge artifact appears identically in both document halves and therefore "
            "passes split-half stability, so that machinery structurally cannot detect it"
        ),
    },
    {
        "id": "A6",
        "date": AMENDMENT_DATE,
        "when": AMENDMENT_WHEN,
        "amends": "source section 18",
        "change": "an anisotropic no-prediction world Q-S6 is added to the required worlds",
        "rationale": (
            "Q-S0 is isotropic and Q-S3 only shows Q-W can find a real low-variance carrier; "
            "nothing shows it does not manufacture one from a poorly estimated tail"
        ),
    },
    {
        "id": "A7",
        "date": AMENDMENT_DATE,
        "when": AMENDMENT_WHEN,
        "amends": "source sections 7.2 and 7.3",
        "change": (
            "eta_rank is taken against a multi-horizon SHARED-source ceiling; the per-lag "
            "free-source reduced-rank ceiling is kept and labelled as a looser upper bound"
        ),
        "rationale": (
            "the per-lag ceiling is not subject to the shared-subspace constraint every "
            "candidate carrier carries, and the penalty grows with the spread of K_Q"
        ),
    },
    {
        "id": "A8",
        "date": AMENDMENT_DATE,
        "when": AMENDMENT_WHEN,
        "amends": "source sections 4.3 and 15",
        "change": (
            "sizing is decided by split-half MSC at rank 64, the slowest-converging endpoint, "
            "and the reproduction requirement gets a numeric threshold frozen at S6"
        ),
        "rationale": (
            "the stability endpoint gates the label that decides the handoff and converges "
            "far more slowly than predictive gain, and 'material magnitude' would otherwise "
            "be settled after reading the test"
        ),
    },
    {
        "id": "A9",
        "date": AMENDMENT_DATE,
        "when": AMENDMENT_WHEN,
        "amends": "source sections 14 and 25",
        "change": (
            "the autonomy stage receiving this carrier either draws a fresh pool or takes "
            "composition statistics only from R2.4's untouched test split"
        ),
        "rationale": (
            "composition retention's denominator is the carrier's own predictive gain, so "
            "computing it where the carrier was fitted reintroduces the R2.3 denominator "
            "problem one stage later"
        ),
    },
    {
        "id": "A12",
        "date": AMENDMENT_DATE,
        "when": (
            "decided from the DEV-Q deletion audit's variance arithmetic, before any "
            "confirmatory number existed"
        ),
        "amends": "amendment A3 and source sections 5.3 and 5.4",
        "change": (
            "the energy-matched deletion family is unattainable in the direction A3 assumed, "
            "and is replaced as primary L-axis comparator by a DOSE-MATCHED geometry "
            "contrast: at each attainable deleted-variance dose, a top-principal-direction "
            "deletion against a lowest-principal-direction deletion of the same dose"
        ),
        "rationale": (
            "the leading 16 directions hold about 0.672 of aperture variance, so every other "
            "direction together holds only about 0.328: no set of lower directions can reach "
            "the target's dose at any rank.  Matching dose downward instead of upward keeps "
            "A3's intent -- hold the energy fixed and vary the geometry -- inside what the "
            "spectrum allows, and it yields a curve rather than a single point"
        ),
    },
    {
        "id": "A11",
        "date": AMENDMENT_DATE,
        "when": (
            "decided from a synthetic implementation smoke world, before any DEV-Q or "
            "CONFIRM-Q number was computed"
        ),
        "amends": "source sections 3, 7.3 and 9",
        "change": (
            "the primary score is the baseline-augmented gain -- the predictor is "
            "b_k(x_t) + D_{Q,k} Q^T x_t rather than D_{Q,k} Q^T x_t alone -- and source "
            "section 3's literal form is co-reported as gamma_raw"
        ),
        "rationale": (
            "under the literal form a rank-r carrier must also RECONSTRUCT the persistent "
            "component from r dimensions, so its gain over persistence is large and negative "
            "at every rank below the aperture; every efficiency ratio of source section 7.3 "
            "then divides a negative number by a positive one and the compression summary of "
            "section 9 is undefined.  The augmented form is also exactly the quantity "
            "amendment A1's residualized objective optimizes, so estimator and score agree"
        ),
    },
    {
        "id": "A10",
        "date": AMENDMENT_DATE,
        "when": AMENDMENT_WHEN,
        "amends": "source sections 5, 13.1 and 19",
        "change": (
            "the P0 identifier collision is resolved: S0..S6 name development steps, "
            "C0..C9b name confirmatory steps, D-AUDIT names the variance-deletion audit, and "
            "P0_/P1_ remain P-axis result labels"
        ),
        "rationale": (
            "the same class of drift made an earlier specification's execution order "
            "disagree with its own section labels and produced a real defect"
        ),
    },
)

STEP_NAMES = {
    "S0": "lock the specification and the existing-run provenance",
    "S1": "D-AUDIT, the variance-dose deletion audit on DEV-Q",
    "S2": "direction-versus-magnitude diagnostic on DEV-Q",
    "S3": "implement and unit-test the full and reduced-rank ceilings",
    "S4": "implement and unit-test Q-E and Q-W on the known-answer worlds",
    "S5": "DEV rank curves, stability learning curves, ridge sweep, sizing",
    "S6": "freeze lags, weights, ridges, rank rule, nulls, thresholds, sample size",
    "C0": "capture fresh CONFIRM-Q documents",
    "C1": "fit full and rank-r predictive ceilings on fit documents",
    "C2": "fit Q-E and Q-W rank curves",
    "C3": "independent-half stability and null refits",
    "C4": "validation rank freeze and metric-geometry classification",
    "C5": "descriptive existing-carrier comparison",
    "C6": "Q-W complement refit and deletion controls",
    "C6b": "replay the frozen D-AUDIT deletion family on CONFIRM-Q (amendment A4)",
    "C7": "freeze P/G/I/L labels and all carrier artifacts",
    "C8": "exactly one test evaluation",
    "C9": "freeze the reference carrier object for the next M stage",
}

# --------------------------------------------------------------------------
# Result labels (source section 13)
# --------------------------------------------------------------------------

P_LABELS = (
    "P0_NO_COMPACT_PREDICTIVE_COMPRESSION",
    "P1_COMPACT_PREDICTIVE_COMPRESSION",
    "P_TECHNICAL_FAILURE",
)
G_LABELS = ("G0_METRIC_DEPENDENT_CARRIERS", "G1_METRIC_CONCORDANT_CARRIER", "G_UNDERRESOLVED")
I_LABELS = (
    "I0_NONIDENTIFIED_CARRIER",
    "I1_STABLE_CARRIER",
    "I2_STABLE_CORE_BROAD_SHOULDER",
    "I3_REDUNDANT_CARRIER_FAMILY",
    "I4_DOMINANT_CHANNEL",
)
L_LABELS = ("L0_ENERGY_EXPLAINS_DELETION", "L1_GEOMETRY_SPECIFIC_DELETION", "L_UNDERRESOLVED")
RIDGE_LABELS = ("R0_RIDGE_DEPENDENT_CARRIER", "R1_RIDGE_STABLE_CARRIER")

EVIDENTIAL_DEV = "DEVELOPMENT_ONLY_SPENT_POOL"
EVIDENTIAL_DEV_NOTE = (
    "computed on the 2000-document R2.2 pool whose sealed fraction was opened by R2.2; "
    "method development only, and no carrier estimated here is promoted"
)
EVIDENTIAL_CONFIRM = "CONFIRMATORY_FRESH_POOL"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@dataclass
class R24Tree(RunTree):
    """The R2.4 artifact tree of source section 20, with the A4 replay directory."""

    SUBDIRECTORIES = (
        "spec",
        "preflight",
        "ceilings",
        "carriers",
        "stability",
        "comparators",
        "redundancy",
        "worlds",
        "sizing",
        "confirmation",
        "decisions",
        "figures",
    )


def label(name: str, *, section: str, evidence: Mapping[str, object]) -> dict[str, object]:
    """A decision record with its mandating section (invariant I1)."""

    record = _label(name, section=section, evidence=dict(evidence))
    record["protocol_revision"] = PROTOCOL_REVISION
    return record


def development(record: dict[str, object]) -> dict[str, object]:
    record["evidential_status"] = EVIDENTIAL_DEV
    record["evidential_note"] = EVIDENTIAL_DEV_NOTE
    return record


def confirmatory(record: dict[str, object]) -> dict[str, object]:
    record["evidential_status"] = EVIDENTIAL_CONFIRM
    return record


def namespaced_seed(namespace: str, *parts: int) -> int:
    import hashlib

    payload = namespace + ":" + ":".join(str(int(part)) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def carrier_key(family: str, rank: int, index: int | None = None) -> str:
    return f"{family}_r{int(rank)}" + ("" if index is None else f"_{int(index)}")


def as_floats(values) -> list[float]:
    return [float(value) for value in np.asarray(values).ravel()]
