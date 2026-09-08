from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from residual_geometry.residuals.probes import ResidualProbeSet, probe_set_from_directions
from residual_geometry.subspace.fat_subspace import (
    autocorr_result_from_document_matrices,
    bootstrap_family_quantiles,
    coupled_nested_span_probe_set,
    geometric_participation_ratio,
    load_residual_basis_artifact,
    pca_axis_loading_profile,
    pca_embedding_geometry,
    positive_profile_area,
    participation_ratio,
    sample_unit_directions_in_span,
    save_residual_basis_artifact,
)
from residual_geometry.subspace.residual_geometry import deduplicate_ranked_probes
from residual_geometry.utils.io import save_json


def _load_fat_subspace_runner():
    script_dir = Path(__file__).parents[3] / "scripts" / "persistent_state" / "residual_geometry"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("fat_subspace_diagnostics", script_dir / "05b_fat_subspace_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sample_unit_directions_are_deterministic_and_confined_to_span() -> None:
    basis = np.eye(4, dtype=np.float32)[:, :2]

    directions = sample_unit_directions_in_span(basis, count=16, seed=7)

    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(directions[:, 2:], 0.0, atol=1e-7)
    np.testing.assert_allclose(directions, sample_unit_directions_in_span(basis, count=16, seed=7))


def test_coupled_nested_span_probe_set_reuses_coefficient_prefixes() -> None:
    basis = np.eye(4, dtype=np.float32)

    probes, coefficients = coupled_nested_span_probe_set(basis, [1, 2, 4], count=8, seed=31)
    repeated, repeated_coefficients = coupled_nested_span_probe_set(basis, [1, 2, 4], count=8, seed=31)

    np.testing.assert_allclose(coefficients, repeated_coefficients)
    np.testing.assert_allclose(probes.directions, repeated.directions)
    for offset, k in enumerate([1, 2, 4]):
        selected = probes.directions[offset * 8 : (offset + 1) * 8]
        expected = coefficients[:, :k] @ basis[:, :k].T
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(selected, expected, atol=1e-6)
        np.testing.assert_allclose(selected[:, k:], 0.0, atol=1e-7)


def test_participation_ratios_distinguish_weight_and_geometry_concentration() -> None:
    assert participation_ratio(np.array([3.0, 1.0])) == pytest.approx(1.6)
    assert geometric_participation_ratio(np.eye(3)) == pytest.approx(3.0)


def test_ranked_probe_deduplication_orders_by_tau_before_family_tiebreak() -> None:
    probes = ResidualProbeSet(
        directions=np.eye(4, dtype=np.float32),
        probe_ids=np.array(["time_low", "pca_high", "random_tie", "time_tie"]),
        probe_family=np.array(["time_lagged", "pca", "random", "time_lagged"]),
    )
    timescales = pd.DataFrame(
        {
            "probe_id": probes.probe_ids,
            "probe_family": probes.probe_family,
            "tau_within": [2.0, 5.0, 3.0, 3.0],
        }
    )

    ranked = deduplicate_ranked_probes(probes, timescales)

    assert ranked["probe_id"].tolist() == ["pca_high", "time_tie", "random_tie", "time_low"]


def test_residual_basis_artifact_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "basis.npz"
    basis = np.eye(4, dtype=np.float32)[:, :2]

    save_residual_basis_artifact(
        str(path),
        basis=basis,
        source_probe_ids=np.array(["probe_a", "probe_b"]),
        source_probe_family=np.array(["time_lagged", "pca"]),
        ranking_rule=np.array("test"),
    )
    artifact = load_residual_basis_artifact(str(path))

    assert artifact.k == 2
    assert artifact.source_probe_ids.tolist() == ["probe_a", "probe_b"]
    assert artifact.source_probe_family.tolist() == ["time_lagged", "pca"]
    np.testing.assert_allclose(artifact.basis, basis)


def test_locked_probe_set_rejects_changed_directions(tmp_path: Path) -> None:
    runner = _load_fat_subspace_runner()
    path = tmp_path / "locked.npz"
    expected = probe_set_from_directions(np.eye(2, dtype=np.float32), family="sample", id_prefix="sample")
    changed = probe_set_from_directions(np.array([[1.0, 0.0], [1.0, 1.0]]), family="sample", id_prefix="sample")

    runner._load_or_create_locked_probe_set(str(path), expected)

    with pytest.raises(ValueError, match="do not match"):
        runner._load_or_create_locked_probe_set(str(path), changed)


def test_locked_nested_probe_set_rejects_changed_coefficient_matrix(tmp_path: Path) -> None:
    runner = _load_fat_subspace_runner()
    path = tmp_path / "nested.npz"
    probes, coefficients = coupled_nested_span_probe_set(np.eye(2, dtype=np.float32), [1, 2], count=4, seed=31)

    runner._load_or_create_locked_nested_probe_set(
        str(path),
        probes,
        coefficient_matrix=coefficients,
        sweep_k=[1, 2],
        candidate_pool_basis=np.eye(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="nested directions"):
        runner._load_or_create_locked_nested_probe_set(
            str(path),
            probes,
            coefficient_matrix=coefficients + 1.0,
            sweep_k=[1, 2],
            candidate_pool_basis=np.eye(2, dtype=np.float32),
        )


def test_default_nested_sweep_omits_values_above_candidate_pool_dimension() -> None:
    runner = _load_fat_subspace_runner()

    assert runner._nested_sweep_values(4) == [1, 2, 3, 4]


def test_source_train_profiles_are_consolidated_in_source_order(tmp_path: Path) -> None:
    runner = _load_fat_subspace_runner()

    class Store:
        def residual_autocorr_profiles_path(self, split: str, family: str, estimator: str) -> str:
            return str(tmp_path / f"{split}_{family}_{estimator}.npz")

    source = ResidualProbeSet(
        directions=np.eye(2, dtype=np.float32),
        probe_ids=np.array(["probe_b", "probe_a"]),
        probe_family=np.array(["pca", "time_lagged"]),
    )
    for probe_id, family, profile in [
        ("probe_a", "time_lagged", [1.0, 0.8]),
        ("probe_b", "pca", [1.0, 0.4]),
    ]:
        np.savez_compressed(
            Store().residual_autocorr_profiles_path("train", family, "within"),
            profiles=np.array([profile]),
            valid_doc_counts=np.array([[4, 4]]),
            probe_ids=np.array([probe_id]),
        )

    bundle = runner._source_probe_train_profile_bundle(Store(), source)

    assert bundle is not None
    assert bundle.probe_ids.tolist() == ["source_probe_00000", "source_probe_00001"]
    assert bundle.probe_family.tolist() == ["source_probe", "source_probe"]
    assert bundle.split.tolist() == ["train", "train"]
    np.testing.assert_allclose(bundle.profiles, np.array([[1.0, 0.4], [1.0, 0.8]]))


def test_positive_profile_area_uses_positive_lags_only() -> None:
    assert positive_profile_area(np.array([1.0, 0.5, -0.2, 0.1])) == pytest.approx(0.6)


def test_autocorr_result_aggregates_document_matrices() -> None:
    corr = np.array(
        [
            [[1.0, 0.8, 0.2], [1.0, 0.4, np.nan]],
            [[1.0, 0.6, 0.0], [1.0, 0.2, 0.1]],
        ]
    )
    valid = ~np.isnan(corr)

    result = autocorr_result_from_document_matrices(corr, valid)

    np.testing.assert_allclose(result.profiles, np.array([[1.0, 0.7, 0.1], [1.0, 0.3, 0.1]]))
    np.testing.assert_array_equal(result.valid_doc_counts, np.array([[2, 2, 2], [2, 2, 1]]))


def test_pca_geometry_reports_only_observed_principal_angles() -> None:
    basis = np.eye(4, dtype=np.float32)
    pca = np.eye(4, dtype=np.float32)

    geometry = pca_embedding_geometry(basis, pca, p_values=[2, 4])
    loadings = pca_axis_loading_profile(basis, pca)

    p2 = geometry[geometry["pca_k_requested"] == 2]
    p4 = geometry[geometry["pca_k_requested"] == 4]
    assert len(p2) == 2
    assert p2["principal_angle_spectrum_complete"].tolist() == [False, False]
    assert p2["principal_angle_degrees"].isna().all()
    assert p2["containment"].iloc[0] == pytest.approx(0.5)
    assert len(p4) == 4
    assert p4["principal_angle_spectrum_complete"].all()
    assert p4["containment"].iloc[0] == pytest.approx(1.0)
    assert loadings["cumulative_mean_squared_loading"].iloc[-1] == pytest.approx(1.0)


def test_mean_profile_summary_reports_family_curve() -> None:
    runner = _load_fat_subspace_runner()
    bundle = runner.ProfileBundle(
        profiles=np.array([[1.0, 0.8], [1.0, 0.4]], dtype=np.float32),
        valid_doc_counts=np.array([[4, 4], [4, 2]]),
        probe_ids=np.array(["a", "b"]),
        probe_family=np.array(["sample", "sample"]),
        split=np.array(["val", "val"]),
    )

    table = runner._mean_profile_summary(bundle)

    lag_one = table[table["lag"] == 1].iloc[0]
    assert lag_one["mean_autocorrelation"] == pytest.approx(0.6)
    assert lag_one["q50_autocorrelation"] == pytest.approx(0.6)
    assert lag_one["mean_valid_doc_count"] == pytest.approx(3.0)


def test_classification_uses_available_direct_pca_span_controls() -> None:
    runner = _load_fat_subspace_runner()
    quantiles = [
        {"split": "test", "diagnostic_family": "random_in_q31", "q50_tau_within": 8.0},
        {"split": "test", "diagnostic_family": "ambient_random", "q90_tau_within": 4.0},
        {"split": "test", "diagnostic_family": "source_probe", "q50_tau_within": 9.0},
        {"split": "test", "diagnostic_family": "random_in_pca31", "q50_tau_within": 6.0},
        {"split": "test", "diagnostic_family": "random_in_pca128", "q50_tau_within": 7.0},
    ]
    source_stability = pd.DataFrame(
        [
            {"split": "train", "tau_within": 9.0, "tau_valid_within": True},
            {"split": "test", "tau_within": 8.0, "tau_valid_within": True},
        ]
    )

    labels, detail = runner._classification(quantiles, split="test", k_fat=31, source_stability=source_stability)

    assert labels == ["fat", "hierarchical", "generic_pca_span_slow"]
    assert detail["direct_pca_span_max_q50_tau"] == pytest.approx(7.0)
    assert detail["headline_span_specific_vs_pca_controls"] is True


def test_nested_sweep_and_source_rank_curve_report_ambient_null_band() -> None:
    runner = _load_fat_subspace_runner()
    timescales = pd.DataFrame(
        [
            {"split": "val", "diagnostic_family": "random_in_q1", "tau_within": 8.0, "positive_profile_area": 4.0, "tau_valid_within": True, "right_censored_within": False},
            {"split": "val", "diagnostic_family": "random_in_q1", "tau_within": 6.0, "positive_profile_area": 3.0, "tau_valid_within": True, "right_censored_within": False},
            {"split": "val", "diagnostic_family": "random_in_q2", "tau_within": 5.0, "positive_profile_area": 2.0, "tau_valid_within": True, "right_censored_within": False},
            {"split": "val", "diagnostic_family": "ambient_random", "tau_within": 2.0, "positive_profile_area": 1.0, "tau_valid_within": True, "right_censored_within": False},
            {"split": "val", "diagnostic_family": "ambient_random", "tau_within": 4.0, "positive_profile_area": 1.0, "tau_valid_within": True, "right_censored_within": False},
        ]
    )
    source_stability = pd.DataFrame(
        [
            {"split": "val", "source_rank": 0, "tau_within": 7.0, "tau_valid_within": True},
            {"split": "val", "source_rank": 1, "tau_within": 3.0, "tau_valid_within": True},
        ]
    )

    null_bands = runner._ambient_null_bands(timescales)
    sweep = runner._nested_dimension_sweep(timescales, null_bands)
    rank_curve = runner._source_probe_rank_curve(source_stability, null_bands)

    assert null_bands["val"] == pytest.approx(3.9)
    assert sweep["nested_k"].tolist() == [1, 2]
    assert sweep["q50_tau_within"].tolist() == pytest.approx([7.0, 5.0])
    assert rank_curve["source_rank"].tolist() == [1, 2]
    assert rank_curve["enters_ambient_null_band"].tolist() == [False, True]
    assert runner._first_source_rank_in_null_band(rank_curve) == {"val": 2}


def test_append_run_preserves_validation_lock_provenance(tmp_path: Path) -> None:
    runner = _load_fat_subspace_runner()

    class Store:
        subspace_dir = str(tmp_path)
        fat_subspace_summary_path = str(tmp_path / "fat_subspace_summary.json")

    save_json({"splits": ["val", "test"]}, str(tmp_path / "projection_collapse_summary.json"))
    assert runner._test_previously_inspected_before_fat_subspace_lock(Store()) is True

    save_json(
        {"test_previously_inspected_before_fat_subspace_lock": False},
        Store.fat_subspace_summary_path,
    )
    assert runner._test_previously_inspected_before_fat_subspace_lock(Store()) is False


def test_bootstrap_family_quantiles_uses_document_resampling() -> None:
    corr = np.tile(np.array([[[1.0, 0.8, 0.2], [1.0, 0.6, 0.1]]]), (4, 1, 1))
    valid = np.ones_like(corr, dtype=bool)

    table = bootstrap_family_quantiles(
        corr,
        valid,
        np.array(["slow", "slow"]),
        split="val",
        max_lag=2,
        min_valid_docs=2,
        min_valid_lag_fraction=1.0,
        smoothing_width=1,
        replicates=8,
        seed=3,
        direction_chunk=1,
    )

    assert len(table) == 9
    assert set(table["diagnostic_family"]) == {"slow"}
    tau_q50 = table[(table["metric"] == "tau_within") & (table["quantile"] == 0.5)].iloc[0]
    assert tau_q50["bootstrap_ci_low"] == pytest.approx(2.0)
    assert tau_q50["bootstrap_ci_high"] == pytest.approx(2.0)
    censoring = table[table["metric"] == "right_censoring_rate"].iloc[0]
    assert censoring["bootstrap_mean"] == pytest.approx(0.0)
