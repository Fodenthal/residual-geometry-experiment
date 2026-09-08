# Residual Geometry Blog Context Index

This file tracks the useful context for writing a technical blog post about the
`persistent_state_residual_geometry` pilot. It is a reading map, not a replacement for
the protocol spec or the pilot summary.

## Primary Reading Order

1. `docs/specs/persistent_state/residual_geometry_pilot_run_summary.md`

   Best single source for the current story. It contains the run identity, key methods,
   main results, caveats, exploratory follow-ups, and the current claim boundaries.

2. `docs/specs/persistent_state/persistent_state_residual_geometry_spec.md`

   Protocol and methods reference. Use this for the original motivation, claim hierarchy,
   stage definitions, metric definitions, and intended interpretation rules.

3. `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/persistent_direction_analysis/summary.md`

   Post-hoc direction contribution and redundancy analysis. Use for the blog section on
   how concentrated the persistent signal is: top-31 share of lifetime excess, top-1/top-5
   concentration, pairwise cosine redundancy, and effective rank.

4. `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/pca_vs_persistent_alignment/summary.md`

   Post-hoc geometry stress test. Use for the PCA caveat and anti-triviality story:
   top residual PCs can align strongly with attention heads without being persistent,
   while persistent directions have the lifetime signal and attention-specific excess.

5. `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/semantic_dossiers/analysis_summary.md`

   Qualitative semantic readout. Use for the hypothesis that persistent directions look
   like document-state, register, domain, or source-template axes rather than clean
   single-topic features.

6. `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/semantic_dossiers/shallow_label_enrichment/shallow_label_enrichment_summary.md`

   Coarse label sanity check for the semantic dossiers. Use only as suggestive support;
   labels are shallow regex/density checks and should not be presented as validated
   semantic classifiers.

## Run Metadata

Use these when the post needs exact setup details:

- `configs/persistent_state/residual_geometry/pilot.yaml`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/configs/config.json`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/configs/config_hash.json`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/configs/resolved_model_architecture.json`

Key setup details already captured in the pilot summary:

- Model: `google/gemma-2-2b`
- Residual hook: `blocks.12.hook_resid_post`
- Residual width: `2304`
- Dataset: `allenai/c4`, English validation split
- Context length used in experiment: `1024`
- Split: train `80%`, validation `10%`, test `10%`
- Layer-12 attention type: global for the experimental context

## Source-of-Truth Result Artifacts

The pilot summary is the best human-readable synthesis. The JSON files below are useful
when exact values, statuses, paths, or machine-readable fields are needed.

- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/residual_probes/autocorr/residual_probe_summary.json`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/residual_probes/time_lagged_fit_summary.json`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/dimensionality_summary.json`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/projection_collapse_summary.json`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/attention_alignment_summary.json`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/block_output_subspace_summary.json`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/attention_transport_summary.json`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/head_ablation_m1_lite_summary.json`

Important note: some generated status fields are conservative or stale relative to the
manual synthesis. For example, the projection-collapse JSON status is stricter than the
validation/test tables warrant. Prefer the pilot summary for interpretation, and use JSON
artifacts for exact numbers.

## Tables Worth Pulling Directly

These CSVs are useful when making blog tables or checking exact row-level values:

- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/persistent_direction_analysis/top31_lifetime_excess_table.csv`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/persistent_direction_analysis/top31_pairwise_abs_cosine.csv`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/persistent_direction_analysis/top31_subspace_singular_values.csv`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/pca_vs_persistent_alignment/group_summary.csv`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/pca_vs_persistent_alignment/persistent_minus_group_contrasts.csv`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/semantic_dossiers/direction_activation_summary.csv`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/semantic_dossiers/shallow_label_enrichment/selected_labels_by_direction.csv`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/semantic_dossiers/shallow_label_enrichment/global_label_rates.csv`

## Figures Worth Considering

The following figures are likely useful for a technical post:

- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/persistent_direction_analysis/lifetime_excess_cumulative_curve.png`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/persistent_direction_analysis/top31_lifetime_excess_bar.png`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/persistent_direction_analysis/top31_singular_spectrum.png`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/pca_vs_persistent_alignment/directionwise_head_alignment_by_group.png`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/pca_vs_persistent_alignment/m0_excess_by_group.png`

## Per-Direction Semantic Dossiers

The full per-direction dossier files are useful for qualitative examples, but they are
not required for the main technical narrative.

- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/semantic_dossiers/semantic_dossier_top10.md`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/semantic_dossiers/per_direction/*.md`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/semantic_dossiers/top_bottom_token_windows.csv`
- `outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/subspace/semantic_dossiers/persistent_high_spans.csv`

Use these for examples only. Do not present the labels as validated semantic discoveries.

## Code References For Methods Audit

Only needed if the post discusses implementation details or if an external reviewer wants
to audit the stages:

- `scripts/persistent_state/residual_geometry/02_compute_residual_probes.py`
- `scripts/persistent_state/residual_geometry/03_compute_residual_autocorr.py`
- `scripts/persistent_state/residual_geometry/04_residual_subspace_pilot.py`
- `scripts/persistent_state/residual_geometry/05_projection_collapse.py`
- `scripts/persistent_state/residual_geometry/06_attention_alignment.py`
- `scripts/persistent_state/residual_geometry/M0_block_output_subspace.py`
- `scripts/persistent_state/residual_geometry/M3_attention_transport.py`
- `scripts/persistent_state/residual_geometry/M1_head_ablation.py`

## Blog Claim Boundaries

Safe claims:

- There is a nontrivial residual-stream persistence signal in this pilot.
- Time-lagged residual probes reveal a heavy upper tail of within-document lifetimes.
- The signal depends on ordered document structure and collapses under document
  permutation.
- The persistent signal is heavy-tailed but not one duplicated direction; the top-31 set
  remains high-rank and nonredundant.
- Held-out time-lagged probes collapse under residual-first and PCA projection but barely
  under random-control projection.
- Persistent directions are strongly PCA-contained as a subspace, even though individual
  PCA axes are mostly not persistent.
- Persistent directions align with attention-output geometry beyond residual-PCA controls.
- M0 supports attention-specific block-output geometry, especially in late layers.
- Semantic dossiers suggest durable document-state/register/template axes.

Claims to avoid:

- The directions are validated semantic concepts or clean discourse variables.
- The identified heads causally write, route, or maintain persistent state.
- Locked group-specific attention transport has been demonstrated.
- Head ablation has shown locked claim-level causal support.
- The phenomenon has been shown to explain SAE feature autocorrelation.
- The result generalizes beyond this model/layer/corpus before full-scale confirmation.

