"""Locked Arm D run namespace and frozen decision rules (R1 §31.1, §35).

Mirrors the researcher's ``arm_b/protocol.py`` idiom: a fail-closed run
directory carrying ``run_manifest.json``, ``source_hashes.json``,
``decision_rules.json``, a byte-identical copy of the normative specification
with its hash, and a one-shot test-access seal.  Thresholds live here, in the
arm's own protocol module, not in a shared config schema.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

RUN_PREFIX = "residual_sequence_memory_arm_d_r1_"
PROTOCOL_REVISION = "arm_d_r1_1"

DECISION_RULES: dict[str, object] = {
    "protocol_revision": PROTOCOL_REVISION,
    "model_name": "google/gemma-2-2b",
    "model_revision": "c5ebcd40d208330abc697524c919956e692655cf",
    "primary_layer": 12,
    "context_length": 1024,
    "source_hook": "decoder_layer_12_forward_pre_hook_hidden_states",
    "destination_hook": "decoder_layer_12_forward_output_hidden_states",
    "aperture_candidates": [256, 512, 1024],
    "aperture_selection_rule": "smallest_candidate_passing_every_frozen_gate",
    "predictive_rank_candidates": [1, 2, 4, 8, 16, 32],
    "lags": [1, 2, 4, 8, 16, 32, 64, 128],
    "suffix_lengths": [32, 64, 128, 256],
    "splice_conditions": ["main", "noop", "sham", "matched"],
    "position_bands": 8,
    "source_position_range": [512, 895],
    "history_offsets": [-4, -3, -2, -1, 0],
    "composition_base_lag_candidates": [4, 8],
    "confirmatory_full_refit_null_replicates": 199,
    "random_subspace_replicates": 200,
    "document_bootstrap_replicates": 2000,
    "train_halves": 2,
    "synthetic_replicates_per_world": 200,
    "synthetic_required_power": 0.80,
    "synthetic_max_false_positive_rate": 0.05,
    "null_families": ["lag_shuffle", "lag_block_shuffle", "xdoc_derangement"],
    "subspace_controls": ["pca", "random"],
    "family_wise_alpha": 0.05,
    "mode_cluster_local_gap": 0.05,
    "mode_cluster_max_span": 0.10,
    "mode_cluster_max_size": 4,
    "ridge_grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    "ridge_document_folds": 5,
    "whitening_initial_ridge_fraction": 1e-4,
    "whitening_max_ridge_fraction": 1e-1,
    "whitening_max_condition_number": 1e4,
    "equal_document_weighting": True,
    "token_random_cv_forbidden": True,
    "qualification_required": True,
    "test_access_requires_frozen_validation_decision": True,
    "development_pool_split": "validation",
    "confirmatory_pool_split": "train",
    "legacy_artifacts_reachable": False,
    "legacy_slow_basis_label": "SLOW_BASIS_REFIT",
    "d5_max_prefix_unique_gain": 0.10,
    "d5_max_higher_order_increment": 0.05,
    "causal_patch_alphas": [0.25, 0.5, 1.0],
    "causal_reference_tolerance": 1e-3,
    "frequency_branch_conditional": True,
    "forbidden_phrases": ["the transformer is an RNN", "the KV cache is redundant"],
    # ---- R1.1 amendment 1: non-normality gate on spectral invariants -------
    # Similarity-invariance of the spectrum is algebraic and says nothing about
    # conditioning.  Under strong non-normality an individual eigenvalue moves
    # arbitrarily far under a small perturbation of A_k, so it is not a portable
    # invariant.  This threshold is fixed here BEFORE any confirmatory operator
    # fit is inspected; the git history of this file is the preregistration
    # record.
    "spectral_max_eigenvalue_condition_number": 10.0,
    "spectral_condition_number_definition": "kappa(lambda_j) = 1 / |y_j^H x_j|, unit-norm left and right eigenvectors",
    "spectral_departure_from_normality": "henrici_frobenius",
    "spectral_eigenvalue_bootstrap_replicates": 2000,
    "spectral_bootstrap_unit": "document",
    "spectral_well_conditioned_label": "SPECTRUM_WELL_CONDITIONED",
    "spectral_ill_conditioned_label": "SPECTRUM_ILL_CONDITIONED",
    "spectral_fallback_invariants": ["singular_values", "real_schur_blocks", "invariant_subspace_principal_angles"],
    "schur_block_min_separation": 0.10,
    # ---- R1.1 amendment 2: document-stratum stability ----------------------
    "stratum_definitions": ["length_band", "c4_source_domain", "register_proxy"],
    "stratum_matched_document_count": True,
    "stratum_permutation_replicates": 999,
    "stratum_stable_label": "SPECTRUM_STRATUM_STABLE",
    "stratum_variable_label": "SPECTRUM_STRATUM_VARIABLE",
    "stratum_gate_scope": "spectral_frequency_branch_and_portability_language_only",
    "stratum_gate_halts_confirmatory_production": False,
    "rank_reported_as_curve_not_threshold": True,
    # ---- R1.1 amendment 3: layer sweep -------------------------------------
    "layer_sweep": [10, 12, 14],
    "confirmatory_layer": 12,
    "secondary_layers": [10, 14],
    "secondary_layer_null_replicates": 99,
    "layer_recurrent_label": "LAYER_RECURRENT",
    "layer_specific_label": "LAYER_SPECIFIC",
    # ---- R1.1 amendment 4: attention-routing diagnostic --------------------
    "routing_diagnostic_metrics": ["l1", "jensen_shannon"],
    "routing_labels": ["ROUTING_MEDIATED", "CONTENT_MEDIATED", "MIXED"],
    "routing_diagnostic_enters_claim_ladder": False,
    # ---- R1.1 amendment 5: sensitivity arms --------------------------------
    "aperture_sensitivity_arm": 512,
    "matched_donor_audit_required": True,
    "aperture_score_retention_deviation_recorded": True,
    # ---- R1.1 amendment 7: frozen transfer versus refit --------------------
    # Amendment 2 asks whether the same KIND of structure exists everywhere.
    # This asks the load-bearing question: do the SAME frozen objects work
    # elsewhere.  Every threshold here is fixed BEFORE any confirmatory
    # transfer number is computed; the git history of this file is the record.
    "transfer_frozen_components": [
        "train_token_mean",
        "nuisance_residualization_model",
        "aperture_projector",
        "source_whitener",
        "destination_whitener",
        "ridge_epsilons",
        "operators_all_lags",
    ],
    "transfer_cells": ["frozen_q_frozen_a", "frozen_q_refit_a", "refit_q_refit_a"],
    "transfer_primary_statistic": "difference_skill_refit_minus_skill_frozen",
    "transfer_skill_scale": "predictive_improvement_over_strongest_trajectory_baseline",
    # The ratio is unbounded and unstable exactly where the denominator is
    # small, so it is secondary and reported only above this floor.
    "transfer_ratio_secondary_only": True,
    "transfer_ratio_min_refit_skill": 0.02,
    "transfer_allow_negative_frozen_skill": True,
    "transfer_null_band_design": "reference_split_half_fit_half1_refit_half2_evaluate_heldout",
    "transfer_matched_fitting_documents": True,
    "transfer_evaluation_documents_always_held_out": True,
    "transfer_bootstrap_replicates": 2000,
    # Each (condition, suffix length) cell holds one row per document, so the
    # fitting-group size IS the row count for the refit.  A rank-16 fit through a
    # 64-dimensional whitening is singular at 75 rows, which is what produced the
    # withdrawn NO_RECURRENCE label.  Fewer, coarser strata are preferred over an
    # undefined refit.
    "transfer_minimum_group_documents": 600,
    "transfer_sample_size_probe": [75, 150, 300, 450, 600],
    "transfer_minimum_band_replicates": 20,
    "transfer_wide_band_width": 0.05,
    # Replicated reference splits per lag for the identical-distribution band.
    # Fixed here before any confirmatory transfer number was computed; a single
    # split yields one gap per lag, whose q05/q95 is effectively min/max.
    "transfer_null_band_replicates_per_lag": 64,
    "transfer_null_band_is_per_lag": True,
    "transfer_permutation_replicates": 999,
    "transfer_labels": [
        "INTERFACE_AND_LAW_TRANSFER",
        "INTERFACE_TRANSFERS_LAW_SPECIFIC",
        "STRUCTURE_RECURS_OBJECTS_DO_NOT",
        "NO_RECURRENCE",
    ],
    "transfer_cross_layer": True,
    "transfer_gates_portability_language_only": True,
    "transfer_halts_confirmatory_production": False,
    # ---- R1.1 amendment 8: capacity-matched trajectory criterion -----------
    # Committed 2026-08-10, AFTER the validation-split result and BEFORE any
    # test-split quantity was computed.  The preregistered order-1 outcome on
    # validation stands as the primary result and is never restated; this is a
    # separate, later-preregistered confirmatory test.
    #
    # Why it is a legitimate refinement rather than a second bite: the failed
    # comparison was capacity-mismatched.  The operator read a single
    # difference slot (rank r inputs, r^2 parameters); the comparator read the
    # whole projected stack (p*r inputs, p*r^2 parameters).  Requiring an
    # order-1 map on rank-16 coordinates to beat an order-5 autoregression is
    # requiring the projection to preserve the Markov property, which a
    # rank-16 projection of a higher-dimensional state generically destroys --
    # qualification world S-D8 is exactly that case.
    "matched_criterion_committed_utc": "2026-08-10T15:10:00Z",
    "matched_criterion_committed_before_any_test_read": True,
    "matched_trajectory_orders": [1, 5],
    "matched_trajectory_orders_frozen": True,
    "matched_trajectory_order_search_forbidden": True,
    "matched_trajectory_requires_matched_parameter_count": True,
    "matched_trajectory_constraint_classes": [
        "gain",
        "diagonal",
        "symmetric",
        "full",
    ],
    "matched_trajectory_reports_order_1_alongside": True,
    "matched_trajectory_order_1_result_is_primary": True,
    # At matched order the unconstrained companion operator and the order-p
    # autoregressive comparator are the SAME model class on the SAME design, so
    # the full-class margin measures ridge and fold noise.  This is recorded
    # now, before the test read, so a near-zero margin is never presented as a
    # finding and the constrained classes carry the structural content.
    "matched_trajectory_full_class_tied_by_construction": True,
    "matched_trajectory_substantive_classes": ["gain", "diagonal", "symmetric"],
    # ---- R1.1 amendment 8: markov-order analysis is the primary route -------
    "markov_order_curve_orders": [1, 2, 3, 5],
    "markov_order_curve_is_primary_route": True,
    "markov_order_saturation_tolerance": 0.01,
    "markov_order_labels": [
        "FIRST_ORDER_SUFFICIENT",
        "LOW_ORDER_SUFFICIENT_P2",
        "LOW_ORDER_SUFFICIENT_P3",
        "HIGH_ORDER_REQUIRED",
        "MARKOV_ORDER_UNAVAILABLE",
    ],
    "first_order_skill_fraction_is_pass_criterion": False,
    # ---- R1.1 amendment 8: degenerate baseline diagnostic ------------------
    "unfitted_extrapolator_baselines": ["constant_velocity", "local_linear"],
    "unfitted_extrapolator_degenerate_threshold": -1.0,
    "unfitted_extrapolators_are_inert_evidence": True,
    # ---- R1.1 amendment 8: the corrected splice-reading label ---------------
    # LOCAL_CONTEXT_EXPLAINED is withdrawn as the outcome for this
    # configuration because it misstates the data: the paired design subtracts
    # local context by construction, so every comparator input is itself a
    # difference that is identically zero when the prefixes agree.
    "splice_reading_labels": [
        "REMOTE_PREFIX_STATE",
        "FIRST_ORDER_LAW_INSUFFICIENT",
    ],
    "local_context_explained_label_withdrawn": True,
    "label_rename_alters_assigned_rung": False,
    "single_sealed_test_read": True,
}

# Selections that only synthetic qualification is allowed to fix (R1 §31.1).
QUALIFICATION_SELECTED_KEYS = (
    "primary_aperture",
    "composition_base_lag",
    "suffix_lengths_retained",
    "local_window_embedding_dimension",
    "local_ar_order",
    "shared_space_msc_threshold",
    "mode_cluster_thresholds",
    "max_whitening_asymmetry",
    "full_refit_lag_subset",
    "prospective_power",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: str | Path) -> str:
    resolved = Path(path)
    if resolved.is_file():
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
        return digest.hexdigest()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files = sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file())
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(candidate.relative_to(resolved).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_path(candidate).encode("ascii"))
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def initialize_locked_arm_d_run(
    run_directory: str | Path,
    *,
    specification: str | Path,
    source_paths: Mapping[str, str | Path],
    model_config: Mapping[str, object],
    environment_lock: Mapping[str, object],
    software_commit: str,
    normative_documents: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    """Create the fail-closed Arm D namespace.  Refuses to overwrite."""

    root = Path(run_directory)
    if not root.name.startswith(RUN_PREFIX):
        raise ValueError(f"run directory must start with {RUN_PREFIX!r}")
    if root.exists():
        raise FileExistsError(f"locked run directory already exists: {root}")
    specification_path = Path(specification)
    if not specification_path.is_file():
        raise FileNotFoundError(specification_path)

    specification_text = specification_path.read_text(encoding="utf-8")
    manifest = {
        "run_id": root.name,
        "protocol_revision": PROTOCOL_REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software_commit": software_commit,
        "qualified": False,
        "held_out_test_opened": False,
        "specification_sha256": sha256_bytes(specification_text.encode("utf-8")),
        "legacy_artifact_status": "UNREACHABLE_CLUSTER_FAILED_STATE",
    }
    root.mkdir(parents=True)
    _write_json(root / "manifests" / "run_manifest.json", manifest)
    _write_json(
        root / "manifests" / "source_hashes.json",
        {
            label: {"path": str(Path(path)), "sha256": sha256_path(path)}
            for label, path in sorted(source_paths.items())
        },
    )
    _write_json(root / "manifests" / "model_config.json", dict(model_config))
    _write_json(root / "manifests" / "decision_rules.json", DECISION_RULES)
    _write_json(root / "manifests" / "environment_lock.json", dict(environment_lock))
    (root / "arm_d_r1_specification.md").write_text(specification_text, encoding="utf-8")
    (root / "arm_d_r1_specification.sha256").write_text(
        manifest["specification_sha256"] + "\n", encoding="ascii"
    )
    if normative_documents:
        _write_json(
            root / "manifests" / "normative_specification_hashes.json",
            {
                label: {"path": str(Path(path)), "sha256": sha256_path(path)}
                for label, path in sorted(normative_documents.items())
            },
        )
    return manifest


def verify_locked_arm_d_run(run_directory: str | Path) -> dict[str, object]:
    root = Path(run_directory)
    manifest = json.loads((root / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("run manifest is not an Arm D R1 namespace")
    specification = (root / "arm_d_r1_specification.md").read_text(encoding="utf-8")
    expected = (root / "arm_d_r1_specification.sha256").read_text(encoding="ascii").strip()
    if sha256_bytes(specification.encode("utf-8")) != expected:
        raise ValueError("Arm D R1 specification hash mismatch")
    return manifest


def record_qualification_decision(
    run_directory: str | Path,
    decision: Mapping[str, object],
) -> dict[str, object]:
    """Write the immutable qualification decision and flip the run's gate."""

    root = Path(run_directory)
    path = root / "manifests" / "qualification_decision.json"
    if path.exists():
        raise FileExistsError(f"qualification decision already recorded: {path}")
    missing = [key for key in QUALIFICATION_SELECTED_KEYS if key not in decision.get("selections", {})]
    if decision.get("qualified") and missing:
        raise ValueError(f"a passing qualification must freeze every selection; missing {missing}")
    _write_json(path, dict(decision))
    manifest_path = root / "manifests" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["qualified"] = bool(decision.get("qualified"))
    manifest["qualification_decided_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    return dict(decision)


def require_qualification(run_directory: str | Path) -> dict[str, object]:
    """Fail closed before any confirmatory access (R1 §31.2)."""

    root = Path(run_directory)
    path = root / "manifests" / "qualification_decision.json"
    if not path.is_file():
        raise ValueError("confirmatory access requires a recorded qualification decision")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("qualified") is not True:
        raise ValueError("ARM_D_R1_QUALIFIED is not TRUE; confirmatory access is refused")
    return decision


def freeze_validation_artifacts(run_directory: str | Path, relative_paths: list[str]) -> dict[str, object]:
    root = Path(run_directory)
    marker = root / "validation_frozen.json"
    if marker.exists():
        raise FileExistsError(f"validation is already frozen: {marker}")
    hashes = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"cannot freeze missing validation artifact: {path}")
        hashes[relative] = sha256_path(path)
    payload = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": hashes,
        "test_access_authorized": True,
    }
    _write_json(marker, payload)
    return payload


def mark_one_shot_test_access(run_directory: str | Path) -> dict[str, object]:
    """Irreversibly consume the single test-split allowance (R1 §34 D-P14)."""

    root = Path(run_directory)
    frozen = root / "validation_frozen.json"
    if not frozen.is_file():
        raise ValueError("test access requires a frozen validation decision")
    for relative, expected in json.loads(frozen.read_text(encoding="utf-8"))["artifact_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256_path(path) != expected:
            raise ValueError(f"frozen validation artifact changed or is missing: {relative}")
    marker = root / "test_access_started.json"
    if marker.exists():
        raise FileExistsError(f"test access was already consumed: {marker}")
    payload = {"started_utc": datetime.now(timezone.utc).isoformat(), "one_shot": True}
    _write_json(marker, payload)
    manifest_path = root / "manifests" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["held_out_test_opened"] = True
    manifest["held_out_test_opened_utc"] = payload["started_utc"]
    _write_json(manifest_path, manifest)
    return payload
