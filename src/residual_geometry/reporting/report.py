from __future__ import annotations

import pandas as pd

from residual_geometry.config.schema import PersistentStateConfig


def render_report_markdown(
    config: PersistentStateConfig,
    protocol_log: dict,
    timescales: pd.DataFrame | None,
    timescale_summary: dict | None,
    null_summary: dict | None,
    burn_in_summary: dict | None = None,
    stage_b_gate: dict | None = None,
    high_confidence_slow: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# Persistent State SAE Timescale Report",
        "",
        f"- Mode: `{config.mode}`",
        f"- Run: `{config.run_name}`",
        f"- Model: `{config.model.name}`",
        f"- Hook: `{config.model.hook_name}`",
        f"- SAE release: `{config.sae.release}`",
        f"- SAE id: `{protocol_log.get('resolved_sae_id', config.sae.sae_id)}`",
        f"- Distribution: `{config.dataset.dataset_name}` / `{config.dataset.dataset_config_name}` / `{config.dataset.split}`",
        f"- Synthetic activations: `{config.runtime.synthetic_activations}`",
        f"- Git commit: `{protocol_log.get('git_commit', 'unknown')}`",
        "",
    ]
    if config.runtime.synthetic_activations:
        lines.extend(
            [
                "> This is a synthetic local pipeline smoke run. Do not interpret any feature timescale, null, or gate result scientifically.",
                "",
            ]
        )
    if timescale_summary:
        lines.extend(
            [
                "## Autocorrelation",
                "",
                f"- Documents analyzed: `{timescale_summary.get('document_count', 'unknown')}`",
                f"- Features analyzed: `{timescale_summary.get('feature_count', 'unknown')}`",
                f"- Invalid within-document features: `{timescale_summary.get('invalid_within', 'unknown')}`",
                f"- Right-censored within-document features: `{timescale_summary.get('right_censored_within', 'unknown')}`",
                f"- Active-document-invalid features: `{timescale_summary.get('active_doc_invalid', 'unknown')}`",
                f"- High-confidence slow features: `{timescale_summary.get('high_confidence_slow_count', 'unknown')}`",
                f"- Bootstrap run: `{timescale_summary.get('bootstrap_run', False)}`",
                f"- Approximate quantile features: `{timescale_summary.get('quantile_approximation_features', 'unknown')}`",
                "",
            ]
        )
    if timescales is not None and not timescales.empty:
        median_tau = timescales["tau_within"].median() if "tau_within" in timescales else float("nan")
        top = timescales.sort_values("slow_score", ascending=False).head(10)
        tau_label = "Median `tau_within`" if config.runtime.synthetic_activations else "Median `tau_within(D_C4)`"
        lines.extend(
            [
                f"{tau_label}: `{median_tau:.3g}`",
                "",
                "Top slow-score features:",
                "",
                "| feature_index | tau_within | tau_binary | tau_raw | slow_score |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in top.itertuples(index=False):
            lines.append(
                f"| {int(row.feature_index)} | {getattr(row, 'tau_within', '')} | "
                f"{getattr(row, 'tau_binary', '')} | {getattr(row, 'tau_raw', '')} | "
                f"{getattr(row, 'slow_score', float('nan')):.3g} |"
            )
        lines.append("")
    if null_summary:
        gate = null_summary.get("stage_b_gate", {}) if isinstance(null_summary.get("stage_b_gate", {}), dict) else {}
        lines.extend(
            [
                "## Nulls",
                "",
                f"- Matched sparsity rows: `{null_summary.get('matched_sparsity_rows', 0)}`",
                f"- Document permutation rows: `{null_summary.get('document_permutation_rows', 0)}`",
                f"- Random residual baseline: `{null_summary.get('random_residual_status', 'unknown')}`",
                f"- Residual PCA baseline: `{null_summary.get('residual_pca_status', 'unknown')}`",
                f"- Matched-sparsity survival fraction: `{gate.get('matched_sparsity_survival_fraction', 'unknown')}`",
                f"- Real/document-permutation median tau ratio: `{gate.get('real_to_document_permutation_median_ratio', 'unknown')}`",
                "",
            ]
        )
    if burn_in_summary:
        lines.extend(
            [
                "## Burn-In",
                "",
                f"- Median burn-in delta: `{burn_in_summary.get('median_burn_in_delta', 'unknown')}`",
                f"- Slow positive burn-in fraction: `{burn_in_summary.get('slow_positive_burn_in_fraction', 'unknown')}`",
                "",
            ]
        )
    if high_confidence_slow is not None and not high_confidence_slow.empty:
        top_hc = high_confidence_slow.sort_values("slow_score", ascending=False).head(10)
        lines.extend(
            [
                "High-confidence slow features:",
                "",
                "| feature_index | tau_within | tau_binary | slow_score | tau_within_ci_low |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in top_hc.itertuples(index=False):
            lines.append(
                f"| {int(row.feature_index)} | {getattr(row, 'tau_within', '')} | "
                f"{getattr(row, 'tau_binary', '')} | {getattr(row, 'slow_score', float('nan')):.3g} | "
                f"{getattr(row, 'tau_within_ci_low', float('nan')):.3g} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Stage Gates",
            "",
            f"- Stage B autocorrelation/null gate: `{(stage_b_gate or {}).get('status', 'unknown')}`",
            f"- May run truncation: `{(stage_b_gate or {}).get('may_run_truncation', False)}`",
            "",
            "This report supports descriptive timescale and cheap-null claims only. Do not describe features as persistent state unless later truncation or causal-persistence stages pass.",
            "",
        ]
    )
    return "\n".join(lines)
