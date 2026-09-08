from __future__ import annotations

import os

import pandas as pd

from residual_geometry.utils.io import load_json, save_json


def residual_geometry_summary(
    timescales: pd.DataFrame,
    architecture: dict | None = None,
    decision_split: str = "val",
    all_real_timescales: pd.DataFrame | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "architecture": architecture or {},
        "probe_count": int(len(timescales)),
        "b1_decision_split": decision_split,
    }
    if all_real_timescales is not None and not all_real_timescales.empty and {"split", "probe_family"}.issubset(all_real_timescales.columns):
        split_counts: dict[str, dict[str, int]] = {}
        for (split, family), group in all_real_timescales.groupby(["split", "probe_family"]):
            split_counts.setdefault(str(split), {})[str(family)] = int(len(group))
        summary["split_probe_counts"] = split_counts
    if timescales.empty:
        return summary
    family_rows: dict[str, object] = {}
    for family, group in timescales.groupby("probe_family"):
        item: dict[str, object] = {
            "count": int(len(group)),
            "valid_within": int(group["tau_valid_within"].sum()) if "tau_valid_within" in group else int(len(group)),
            "right_censored_within": int(group["right_censored_within"].sum()) if "right_censored_within" in group else 0,
        }
        valid = group[group["tau_valid_within"].astype(bool)] if "tau_valid_within" in group else group
        if not valid.empty and "tau_within" in valid:
            for q in [0.5, 0.75, 0.9, 0.95]:
                item[f"q{int(q * 100)}_tau_within"] = float(valid["tau_within"].quantile(q))
        family_rows[str(family)] = item
    summary["families"] = family_rows
    return summary


def write_residual_report(
    report_path: str,
    summary_path: str,
    timescales: pd.DataFrame,
    architecture_path: str | None = None,
    extra_summary: dict | None = None,
    decision_split: str = "val",
    all_real_timescales: pd.DataFrame | None = None,
) -> None:
    architecture = load_json(architecture_path) if architecture_path and os.path.exists(architecture_path) else {}
    summary = residual_geometry_summary(
        timescales,
        architecture=architecture,
        decision_split=decision_split,
        all_real_timescales=all_real_timescales,
    )
    if extra_summary:
        summary.update(extra_summary)
    save_json(summary, summary_path)
    lines = [
        "# Residual Geometry Report",
        "",
        "## Run Metadata",
        "",
        f"- Model: `{architecture.get('model_name', 'unknown')}`",
        f"- Hook: `{architecture.get('hook_name', 'unknown')}`",
        f"- Target layer attention type: `{architecture.get('target_layer_attention_type', 'unknown')}`",
        f"- Probe count: `{summary.get('probe_count', 0)}`",
        f"- B1 decision split: `{summary.get('b1_decision_split', 'val')}`",
        "",
        "## Validation Timescale Summary",
        "",
    ]
    families = summary.get("families", {})
    if isinstance(families, dict):
        for family, item in families.items():
            if not isinstance(item, dict):
                continue
            lines.extend(
                [
                    f"### {family}",
                    "",
                    f"- Count: `{item.get('count', 0)}`",
                    f"- Valid within: `{item.get('valid_within', 0)}`",
                    f"- Right-censored within: `{item.get('right_censored_within', 0)}`",
                    f"- Q50 tau_within: `{item.get('q50_tau_within', 'n/a')}`",
                    f"- Q90 tau_within: `{item.get('q90_tau_within', 'n/a')}`",
                    f"- Q95 tau_within: `{item.get('q95_tau_within', 'n/a')}`",
                    "",
                ]
            )
    split_counts = summary.get("split_probe_counts", {})
    if isinstance(split_counts, dict) and split_counts:
        lines.extend(["## Split Probe Counts", ""])
        for split, counts in split_counts.items():
            lines.append(f"- {split}: `{counts}`")
        lines.append("")
    lines.extend(
        [
            "## B1 Decision",
            "",
        ]
    )
    b1 = summary.get("residual_probe_summary", {})
    if isinstance(b1, dict):
        decision = b1.get("b1_decision", {})
        if isinstance(decision, dict):
            lines.extend(
                [
                    f"- Status: `{decision.get('status', 'unknown')}`",
                    f"- Combined Q95/Q50: `{decision.get('combined_q95_over_q50', 'unknown')}`",
                    f"- Criteria: `{decision.get('criteria', {})}`",
                    f"- Interpretation flags: `{decision.get('interpretation', [])}`",
                    "",
                ]
            )
    tl = summary.get("time_lagged_fit_summary", {})
    if isinstance(tl, dict):
        lines.extend(
            [
                "## Time-Lagged Fit",
                "",
                f"- Lag set: `{tl.get('lag_set', 'unknown')}`",
                f"- Whitening PCs used: `{tl.get('whitening_pcs_used', 'unknown')}`",
                f"- Epsilon scale: `{tl.get('epsilon_scale', 'unknown')}`",
                f"- Final condition number: `{tl.get('final_condition_number', 'unknown')}`",
                f"- Positive-validation persistence fraction: `{tl.get('positive_validation_persistence_fraction', 'not_computed')}`",
                f"- Ridge sensitivity status: `{tl.get('ridge_sensitivity_status', 'not_required_or_not_run')}`",
                "",
            ]
        )
    dimensionality = summary.get("dimensionality_summary", {})
    if isinstance(dimensionality, dict) and dimensionality:
        lines.extend(
            [
                "## Residual-First Dimensionality",
                "",
                f"- k_80pct_lifetime_excess: `{dimensionality.get('k_80pct_lifetime_excess', 'unknown')}`",
                f"- Top-kstar geometric participation ratio: `{dimensionality.get('top_kstar_geometric_participation_ratio', 'unknown')}`",
                f"- Top-kstar slowness participation ratio: `{dimensionality.get('top_kstar_slowness_participation_ratio', 'unknown')}`",
                "",
            ]
        )
    projection = summary.get("projection_collapse_summary", {})
    if isinstance(projection, dict) and projection:
        lines.extend(
            [
                "## Projection Collapse",
                "",
                f"- Status: `{projection.get('status', 'unknown')}`",
                f"- Basis families: `{projection.get('basis_families', [])}`",
                f"- Eval families: `{projection.get('eval_families', [])}`",
                f"- Splits: `{projection.get('splits', [])}`",
                f"- Summary table: `{projection.get('summary_table_path', 'unknown')}`",
                "",
            ]
        )
    fat_subspace = summary.get("fat_subspace_summary", {})
    if isinstance(fat_subspace, dict) and fat_subspace:
        lines.extend(
            [
                "## Fat-Subspace Diagnostics",
                "",
                f"- Status: `{fat_subspace.get('status', 'unknown')}`",
                f"- Classification: `{fat_subspace.get('classification', [])}`",
                f"- k_star / k_fat: `{fat_subspace.get('k_star', 'unknown')} / {fat_subspace.get('k_fat', 'unknown')}`",
                f"- Locked nested sweep k: `{fat_subspace.get('nested_sweep_k', [])}`",
                f"- Ambient-random Q95 null band by split: `{fat_subspace.get('ambient_random_q95_tau_within_by_split', {})}`",
                f"- First source rank entering null band by split: `{fat_subspace.get('first_source_rank_in_ambient_null_band_by_split', {})}`",
                f"- Nested slow-core cutoff status: `{fat_subspace.get('nested_slow_core_cutoff_status', 'unknown')}`",
                f"- Nested slow-core candidate interval: `{fat_subspace.get('nested_slow_core_candidate_interval', None)}`",
                f"- Nested slow-core candidate supported: `{fat_subspace.get('nested_slow_core_candidate_supported', False)}`",
                f"- Nested slow-core confirmed: `{fat_subspace.get('nested_slow_core_confirmed', False)}`",
                f"- Top-kstar geometric participation ratio: `{fat_subspace.get('top_kstar_geometric_participation_ratio', 'unknown')}`",
                f"- Top-kstar slowness participation ratio: `{fat_subspace.get('top_kstar_slowness_participation_ratio', 'unknown')}`",
                f"- Evaluated splits: `{fat_subspace.get('evaluated_splits', [])}`",
                f"- Generic slow region observed: `{fat_subspace.get('generic_slow_region_observed', False)}`",
                f"- Coordinate-free generic slow region supported: `{fat_subspace.get('coordinate_free_generic_slow_region_supported', False)}`",
                f"- Claim level 5a supported: `{fat_subspace.get('claim_level_5a_supported', False)}`",
                f"- Exploratory after test inspection: `{fat_subspace.get('exploratory_after_test_inspection', False)}`",
                f"- Source-probe split stability: `{fat_subspace.get('source_probe_split_stability', {})}`",
                f"- Direction timescales: `{fat_subspace.get('direction_timescales_path', 'unknown')}`",
                f"- Nested-dimension sweep: `{fat_subspace.get('nested_dimension_sweep_path', 'unknown')}`",
                f"- Source-probe rank curve: `{fat_subspace.get('source_probe_rank_curve_path', 'unknown')}`",
                f"- Mean autocorrelation profiles: `{fat_subspace.get('mean_autocorrelation_profiles_path', 'unknown')}`",
                "",
            ]
        )
    attention = summary.get("attention_alignment_summary", {})
    if isinstance(attention, dict) and attention:
        lines.extend(
            [
                "## Attention Alignment",
                "",
                f"- Status: `{attention.get('status', 'unknown')}`",
                f"- Layers: `{attention.get('layers', [])}`",
                f"- Components per head: `{attention.get('components_per_head', 'unknown')}`",
                f"- Sampled train positions: `{attention.get('sampled_train_positions', 'unknown')}`",
                f"- Persistent vs random: `{attention.get('persistent_vs_random_max_head_alignment', {})}`",
                f"- Robustness controls: `{list(attention.get('attention_alignment_robustness', {}).keys())}`",
                "",
            ]
        )
    block = summary.get("block_output_subspace_summary", {})
    if isinstance(block, dict) and block:
        lines.extend(
            [
                "## M0 Block-Output Subspace",
                "",
                f"- Status: `{block.get('status', 'unknown')}`",
                f"- Layers: `{block.get('layers', [])}`",
                f"- Components per subspace: `{block.get('components_per_subspace', 'unknown')}`",
                f"- Across-layer summary: `{block.get('across_layers', {})}`",
                "",
            ]
        )
    transport = summary.get("attention_transport_summary", {})
    if isinstance(transport, dict) and transport:
        lines.extend(
            [
                "## M3 Attention Transport",
                "",
                f"- Status: `{transport.get('status', 'unknown')}`",
                f"- M0 status: `{transport.get('m0_status', 'unknown')}`",
                f"- Primary metric: `{transport.get('primary_metric', 'unknown')}`",
                f"- Positive vs all controls: `{transport.get('positive_vs_all_controls', 'unknown')}`",
                f"- Claim level 7b supported: `{transport.get('claim_level_7b_supported', False)}`",
                f"- By head group: `{transport.get('by_head_group', {})}`",
                f"- Aligned vs controls: `{transport.get('aligned_vs_control', {})}`",
                "",
            ]
        )
    m1_lite = summary.get("head_ablation_m1_lite_summary", {})
    if isinstance(m1_lite, dict) and m1_lite:
        lines.extend(
            [
                "## Exploratory M1-lite Head Ablation",
                "",
                f"- Status: `{m1_lite.get('status', 'unknown')}`",
                f"- Exploratory: `{m1_lite.get('exploratory', True)}`",
                f"- Post-hoc head selection: `{m1_lite.get('post_hoc_head_selection', True)}`",
                f"- Claim level 8 supported: `{m1_lite.get('claim_level_8_supported', False)}`",
                f"- M0 status: `{m1_lite.get('m0_status', 'unknown')}`",
                f"- By head group: `{m1_lite.get('by_head_group', {})}`",
                f"- Exploratory comparisons: `{m1_lite.get('exploratory_comparisons', {})}`",
                f"- L12H6 signed effect: `{m1_lite.get('l12h6_signed_effect', {})}`",
                "",
            ]
        )
    plots = summary.get("plots", {})
    if isinstance(plots, dict) and plots:
        lines.extend(["## Plots", ""])
        for name, path in plots.items():
            lines.append(f"- {name}: `{path}`")
        lines.append("")
    lines.extend(
        [
            "## Claim Status",
            "",
            "- Claim levels 1-5 are supported by B1, residual-first subspace construction, and fixed held-out projection collapse.",
            "- Claim level 5a is supported only when fat-subspace diagnostics have `claim_level_5a_supported=true`; this requires untouched test confirmation relative to ambient-random and direct PCA-span controls.",
            "- Claim level 6 is supported when attention alignment survives residual-PCA controls.",
            "- Claim level 7a is supported when M0 status is `attention_specific`.",
            "- Claim level 7b is not supported unless locked M3 has `claim_level_7b_supported=true`.",
            "- Claim level 8 is not supported by exploratory M1-lite; it requires a prospectively locked positive M1.",
            "",
        ]
    )
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
