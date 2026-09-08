from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_script(name: str):
    script_dir = Path(__file__).parents[3] / "scripts" / "persistent_state" / "residual_geometry"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(name, script_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _timescales() -> pd.DataFrame:
    rows = []
    values = {
        "val": {
            "ambient_random": [1.0, 1.0, 1.0, 2.0],
            "random_in_q4": [10.0, 8.0, 6.0, 4.0],
        },
        "test": {
            "ambient_random": [1.0, 1.0, 1.0, 2.0],
            "random_in_q4": [9.0, 7.0, 5.0, 3.0],
        },
    }
    for split, families in values.items():
        for family, tau_values in families.items():
            for index, tau in enumerate(tau_values):
                rows.append(
                    {
                        "split": split,
                        "probe_id": f"{family}_{index:05d}",
                        "diagnostic_family": family,
                        "tau_within": tau,
                        "tau_valid_within": True,
                        "right_censored_within": False,
                        "positive_profile_area": tau * 2.0,
                    }
                )
    return pd.DataFrame(rows)


def test_lower_tail_and_direction_bootstrap_report_generic_support() -> None:
    runner = _load_script("analyze_fat_subspace_artifacts")
    timescales = _timescales()

    lower_tail = runner.family_lower_tail_summary(timescales)
    endpoint = lower_tail[
        (lower_tail["split"] == "test") & (lower_tail["diagnostic_family"] == "random_in_q4")
    ].iloc[0]
    assert endpoint["q10_tau_within"] > endpoint["ambient_random_q95_tau_within"]
    assert endpoint["fraction_tau_above_ambient_q95"] == 1.0

    first = runner.direction_bootstrap_intervals(timescales, replicates=50, seed=7)
    second = runner.direction_bootstrap_intervals(timescales, replicates=50, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_split_stability_and_low_core_loading_sensitivity() -> None:
    runner = _load_script("analyze_fat_subspace_artifacts")
    timescales = _timescales()
    coefficients = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.1, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    stability = runner.split_stability_summary(timescales)
    endpoint = stability[stability["diagnostic_family"] == "random_in_q4"].iloc[0]
    assert endpoint["spearman_val_test_tau"] == 1.0
    assert endpoint["fraction_above_split_null_on_both"] == 1.0

    sensitivity = runner.core_loading_sensitivity(timescales, coefficients, k_fat=4, prefixes=[1])
    test_row = sensitivity[sensitivity["split"] == "test"].iloc[0]
    assert test_row["lowest_core_loading_quartile_q50_tau_within"] == 9.0
    assert test_row["lowest_core_loading_quartile_fraction_tau_above_ambient_q95"] == 1.0


def test_semantic_comparison_can_read_saved_projection_subset(tmp_path: Path) -> None:
    runner = _load_script("semantic_axis_comparison")
    split_dir = tmp_path / "val"
    split_dir.mkdir()
    probe_ids = np.array(["probe_a", "probe_b"])
    for chunk_index in range(2):
        np.savez_compressed(
            split_dir / f"chunk_{chunk_index:05d}.npz",
            context_ids=np.array([f"ctx_{chunk_index}_0", f"ctx_{chunk_index}_1"]),
            tokens=np.array([[1, 2], [3, 4]], dtype=np.int64),
            probe_ids=probe_ids,
            probe_family=np.array(["random", "time_lagged"]),
            projections=np.array(
                [
                    [[1.0, 10.0], [2.0, 20.0]],
                    [[3.0, 30.0], [4.0, 40.0]],
                ],
                dtype=np.float16,
            ),
        )

    class Store:
        @staticmethod
        def residual_projection_split_dir(split: str) -> str:
            assert split == "val"
            return str(split_dir)

    tokens, context_ids, projections = runner._saved_projection_bundle(
        Store(),
        split="val",
        probe_ids=np.array(["probe_b"]),
        max_documents=3,
    )

    assert tokens.shape == (3, 2)
    assert context_ids.tolist() == ["ctx_0_0", "ctx_0_1", "ctx_1_0"]
    np.testing.assert_allclose(projections[:, :, 0], np.array([[10, 20], [30, 40], [10, 20]]))


def test_semantic_comparison_reconstructs_locked_span_projection(tmp_path: Path) -> None:
    runner = _load_script("semantic_axis_comparison")
    split_dir = tmp_path / "val"
    split_dir.mkdir()
    np.savez_compressed(
        split_dir / "chunk_00000.npz",
        context_ids=np.array(["ctx"]),
        tokens=np.array([[1, 2]], dtype=np.int64),
        probe_ids=np.array(["axis_a", "axis_b"]),
        probe_family=np.array(["time_lagged", "time_lagged"]),
        projections=np.array([[[2.0, 3.0], [5.0, 7.0]]], dtype=np.float16),
    )

    group = runner._derived_projection_group(
        target_probe_ids=np.array(["mixed"]),
        target_directions=np.array([[1.0, 1.0]], dtype=np.float32),
        source_probe_ids=np.array(["axis_a", "axis_b"]),
        source_directions=np.eye(2, dtype=np.float32),
    )

    class Store:
        @staticmethod
        def residual_projection_split_dir(split: str) -> str:
            assert split == "val"
            return str(split_dir)

    _, _, projections = runner._saved_projection_bundle(
        Store(),
        split="val",
        probe_ids=np.array(["mixed"]),
        max_documents=None,
        derived_groups=[group],
    )

    np.testing.assert_allclose(projections[:, :, 0], np.array([[5.0, 12.0]]))


def test_semantic_comparison_collapses_span_statistics_to_documents() -> None:
    runner = _load_script("semantic_axis_comparison")
    metrics = runner._document_level_span_metrics(
        [
            (0, 0, 9, 1.0, 9.0),
            (0, 20, 30, 1.0, 10.0),
            (2, 5, 13, 1.0, 8.0),
        ],
        n_documents=4,
        top_entries=[(0, 0, 1.0), (0, 1, 0.9), (1, 0, 0.8)],
        bottom_entries=[(2, 0, -1.0), (3, 0, -0.9), (3, 1, -0.8)],
    )

    assert metrics["n_docs_with_retained_q95_span"] == 2
    assert metrics["fraction_docs_with_retained_q95_span"] == 0.5
    assert metrics["spans_per_covered_doc"] == 1.5
    assert metrics["median_doc_max_q95_span_length_covered"] == 9.0
    assert metrics["q90_doc_max_q95_span_length_covered"] == 9.8
    assert metrics["mean_doc_max_q95_span_length_all_docs"] == 4.5
    assert metrics["n_unique_top_window_docs"] == 2
    assert metrics["fraction_unique_top_window_docs"] == 2 / 3
    assert metrics["max_top_window_doc_share"] == 2 / 3
    assert metrics["n_unique_bottom_window_docs"] == 2
    assert metrics["fraction_unique_bottom_window_docs"] == 2 / 3
    assert metrics["max_bottom_window_doc_share"] == 2 / 3
