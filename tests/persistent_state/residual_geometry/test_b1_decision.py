from __future__ import annotations

import pandas as pd

from residual_geometry.residuals.autocorr import b1_decision_summary


def test_b1_decision_uses_validation_real_rows_and_permutation_reduction() -> None:
    rows = []
    for family, tau in [("random", 10), ("pca", 14), ("time_lagged", 18)]:
        for idx in range(10):
            rows.append(
                {
                    "split": "val",
                    "control": "real",
                    "probe_family": family,
                    "probe_id": f"{family}_{idx}",
                    "tau_within": tau + idx,
                    "tau_valid_within": True,
                }
            )
            rows.append(
                {
                    "split": "val",
                    "control": "document_permutation",
                    "probe_family": family,
                    "probe_id": f"{family}_{idx}",
                    "tau_perm_within": 2,
                    "tau_valid_perm_within": True,
                }
            )
    summary = b1_decision_summary(pd.DataFrame(rows))
    assert summary["criteria"]["valid_probe_coverage_ge_90pct"] is True
    assert summary["criteria"]["time_lagged_q90_exceeds_random_q90_by_30pct"] is True
    assert summary["criteria"]["document_permutation_reduces_top_probe_tau"] is True
