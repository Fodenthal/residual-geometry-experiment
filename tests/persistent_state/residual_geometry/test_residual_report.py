from __future__ import annotations

from pathlib import Path

import pandas as pd

from residual_geometry.reporting.residual_report import residual_geometry_summary, write_residual_report


def test_residual_report_summary_uses_validation_rows_for_headline_counts() -> None:
    rows = []
    for split in ["train", "val", "test"]:
        for idx in range(4):
            rows.append(
                {
                    "split": split,
                    "probe_family": "random",
                    "probe_id": f"{split}_{idx}",
                    "tau_within": 1,
                    "tau_valid_within": True,
                    "right_censored_within": False,
                }
            )
    all_real = pd.DataFrame(rows)
    val = all_real[all_real["split"] == "val"]
    summary = residual_geometry_summary(val, decision_split="val", all_real_timescales=all_real)
    assert summary["probe_count"] == 4
    assert summary["families"]["random"]["count"] == 4
    assert summary["split_probe_counts"]["train"]["random"] == 4
    assert summary["split_probe_counts"]["val"]["random"] == 4
    assert summary["split_probe_counts"]["test"]["random"] == 4


def test_residual_report_includes_fat_subspace_status(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    summary_path = tmp_path / "summary.json"

    write_residual_report(
        report_path=str(report_path),
        summary_path=str(summary_path),
        timescales=pd.DataFrame(),
        extra_summary={
            "fat_subspace_summary": {
                "status": "validation_complete_test_pending",
                "classification": ["fat"],
                "k_star": 31,
                "k_fat": 31,
                "nested_sweep_k": [1, 2, 3, 5, 8, 13, 21, 31],
                "nested_slow_core_cutoff_status": "descriptive_only_no_validation_locked_cutoff_rule",
                "nested_dimension_sweep_path": "nested.parquet",
                "evaluated_splits": ["val"],
                "generic_slow_region_observed": True,
                "coordinate_free_generic_slow_region_supported": True,
                "claim_level_5a_supported": True,
                "exploratory_after_test_inspection": False,
                "mean_autocorrelation_profiles_path": "profiles.parquet",
            }
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "## Fat-Subspace Diagnostics" in report
    assert "- Classification: `['fat']`" in report
    assert "- Locked nested sweep k: `[1, 2, 3, 5, 8, 13, 21, 31]`" in report
    assert "- Nested-dimension sweep: `nested.parquet`" in report
    assert "- Claim level 5a supported: `True`" in report
    assert "Claim level 5a" in report
