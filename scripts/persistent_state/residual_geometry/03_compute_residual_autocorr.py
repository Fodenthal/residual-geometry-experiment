#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import artifact_metadata, load_config_and_store, require_residual_geometry_config, residual_batches, split_contexts, token_matrix
from residual_geometry.autocorr.estimators import AutocorrAccumulator, build_timescale_table, compute_document_autocorr
from residual_geometry.residuals.autocorr import (
    accumulate_projection_autocorr_multi,
    b1_decision_summary,
    bootstrap_projection_timescales,
    compute_permutation_timescales,
    projection_chunk_paths,
    save_profiles_npz,
    summarize_probe_timescales,
)
from residual_geometry.residuals.projections import project_residuals
from residual_geometry.residuals.provider import ResidualStreamProvider
from residual_geometry.residuals.time_lagged import fit_time_lagged_from_covariances
from residual_geometry.utils.io import load_json, save_json, save_parquet
from residual_geometry.utils.logging import get_logger


def maybe_run_time_lagged_ridge_sensitivity(config, store, residual_cfg, max_lag: int, metadata: dict[str, object]) -> None:
    if config.mode != "pilot" or not Path(store.time_lagged_fit_summary_path).exists():
        return
    fit_summary = load_json(store.time_lagged_fit_summary_path)
    if float(fit_summary.get("epsilon_scale", 0.0)) <= 1e-2:
        return
    if Path(store.time_lagged_ridge_sensitivity_path).exists():
        return
    if not Path(store.time_lagged_covariances_path).exists():
        fit_summary["ridge_sensitivity_status"] = "required_but_missing_saved_covariances"
        save_json(fit_summary, store.time_lagged_fit_summary_path)
        return
    with np.load(store.time_lagged_covariances_path, allow_pickle=False) as cov:
        mean = cov["mean"]
        sigma0 = cov["sigma0"].astype(np.float64)
        sigma_lag = cov["sigma_lag"].astype(np.float64)
        pair_count = int(cov["pair_count"])
        lag_set = cov["lag_set"].astype(int).tolist()
    val_df = split_contexts(store, "val")
    val_tokens = token_matrix(val_df)
    provider = ResidualStreamProvider(config)
    provider.load()
    rows: list[dict[str, object]] = []
    top_direction_sets: dict[float, np.ndarray] = {}
    trace_scale = float(np.trace(sigma0) / sigma0.shape[0])
    for scale in [1e-4, 1e-3, 1e-2, 1e-1]:
        fit = fit_time_lagged_from_covariances(
            mean=mean,
            sigma0=sigma0,
            sigma_lag=sigma_lag,
            pair_count=pair_count,
            lag_set=lag_set,
            whitening_pcs=residual_cfg.time_lagged.whitening_pcs,
            output_directions=residual_cfg.time_lagged.output_directions,
            ridge_scale=scale,
            max_ridge_scale=scale,
            condition_threshold=float("inf"),
        )
        accumulator = AutocorrAccumulator(n_features=fit.probes.n_probes, max_lag=max_lag)
        for _, residuals in residual_batches(provider, val_tokens, residual_cfg.projections.batch_size):
            projected = project_residuals(residuals, fit.probes)
            accumulator.update(compute_document_autocorr(projected, max_lag=max_lag, estimator="within"))
        result = accumulator.finalize()
        table = build_timescale_table(
            feature_indices=np.arange(fit.probes.n_probes),
            estimator_results={"within": result},
            max_lag=max_lag,
            min_valid_docs=residual_cfg.valid_doc_threshold,
            min_valid_lag_fraction=residual_cfg.min_valid_lag_fraction,
            smoothing_width=residual_cfg.smoothing_width,
        )
        valid = table[table["tau_valid_within"].astype(bool)]
        top_k = min(32, len(valid))
        top_indices = valid.sort_values("tau_within", ascending=False).head(top_k)["feature_index"].to_numpy(dtype=int)
        top_direction_sets[scale] = fit.probes.directions[top_indices] if len(top_indices) else np.zeros((0, config.model.d_model), dtype=np.float32)
        rows.append(
            {
                "epsilon_scale_requested": scale,
                "epsilon": float(scale * trace_scale),
                "validation_q50_tau_within": float(valid["tau_within"].quantile(0.5)) if not valid.empty else float("nan"),
                "validation_q90_tau_within": float(valid["tau_within"].quantile(0.9)) if not valid.empty else float("nan"),
                "validation_q95_tau_within": float(valid["tau_within"].quantile(0.95)) if not valid.empty else float("nan"),
                "valid_probe_count": int(len(valid)),
                "top_k_for_overlap": int(top_k),
                "final_condition_number": float(fit.summary.get("final_condition_number", float("nan"))),
                **metadata,
            }
        )
    baseline = top_direction_sets.get(1e-4, np.zeros((0, config.model.d_model), dtype=np.float32))
    for row in rows:
        directions = top_direction_sets.get(float(row["epsilon_scale_requested"]), np.zeros((0, config.model.d_model), dtype=np.float32))
        if len(baseline) and len(directions):
            singular_values = np.linalg.svd(baseline @ directions.T, compute_uv=False)
            row["top_direction_subspace_overlap_vs_1e_4"] = float(np.mean(np.clip(singular_values, 0.0, 1.0) ** 2))
        else:
            row["top_direction_subspace_overlap_vs_1e_4"] = float("nan")
    save_parquet(pd.DataFrame(rows), store.time_lagged_ridge_sensitivity_path)
    fit_summary["ridge_sensitivity_status"] = "run"
    fit_summary["ridge_sensitivity_path"] = store.time_lagged_ridge_sensitivity_path
    fit_summary["ridge_sensitivity_rows"] = len(rows)
    fit_summary.update(metadata)
    save_json(fit_summary, store.time_lagged_fit_summary_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute residual-probe autocorrelation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Only compute the val split. Sufficient for the B1 decision; add train/test later.",
    )
    args = parser.parse_args()

    logger = get_logger(__name__)
    config, store = load_config_and_store(args.config)
    residual_cfg = require_residual_geometry_config(config)
    metadata = artifact_metadata(config, "residual_geometry_03_compute_residual_autocorr")
    if Path(store.residual_probe_timescales_path).exists() and not args.overwrite:
        logger.info("Residual autocorr artifacts already exist; use --overwrite to rebuild.")
        return 0

    all_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[pd.DataFrame] = []
    max_lag = residual_cfg.max_lag
    computed_ci_lags = [lag for lag in residual_cfg.ci_lags if lag <= max_lag]
    val_within_profiles: dict[str, np.ndarray] = {}
    splits = ["val"] if args.val_only else ["train", "val", "test"]
    for split in splits:
        chunk_paths = projection_chunk_paths(store.residual_projection_split_dir(split))
        if not chunk_paths:
            logger.warning("No projection chunks for split=%s", split)
            continue
        first = np.load(chunk_paths[0], allow_pickle=False)
        families = sorted(set(first["probe_family"].astype(str).tolist()))
        first.close()
        for family in families:
            accumulators, probe_ids, probe_families = accumulate_projection_autocorr_multi(
                chunk_paths,
                max_lag=max_lag,
                estimators=["raw", "within"],
                probe_family_filter=family,
            )
            estimator_results = {est: acc.finalize() for est, acc in accumulators.items()}
            for estimator, result in estimator_results.items():
                save_profiles_npz(
                    store.residual_autocorr_profiles_path(split, family, estimator),
                    profiles=result.profiles,
                    valid_doc_counts=result.valid_doc_counts,
                    probe_ids=probe_ids,
                    probe_family=probe_families,
                )
            within_profiles = estimator_results["within"].profiles
            if split == "val":
                val_within_profiles[family] = within_profiles
            table = build_timescale_table(
                feature_indices=np.arange(len(probe_ids), dtype=np.int64),
                estimator_results=estimator_results,
                max_lag=max_lag,
                min_valid_docs=residual_cfg.valid_doc_threshold,
                min_valid_lag_fraction=residual_cfg.min_valid_lag_fraction,
                smoothing_width=residual_cfg.smoothing_width,
            )
            table["probe_id"] = probe_ids
            table["probe_family"] = probe_families
            table["split"] = split
            table["control"] = "real"
            for key, value in metadata.items():
                table[key] = value
            if family == "time_lagged":
                eligibility_lags = [lag for lag in [8, 16, 32] if lag <= max_lag and lag < within_profiles.shape[1]]
                if eligibility_lags:
                    table["gk_positive_validation_persistence"] = np.nanmedian(within_profiles[:, eligibility_lags], axis=1) > 0.0
                else:
                    table["gk_positive_validation_persistence"] = False
            else:
                table["gk_positive_validation_persistence"] = True
            all_rows.append(table)
            if split == "val" and not args.skip_bootstrap:
                boot = bootstrap_projection_timescales(
                    chunk_paths,
                    feature_indices=np.arange(len(probe_ids), dtype=np.int64),
                    estimator="within",
                    max_lag=max_lag,
                    min_valid_docs=residual_cfg.valid_doc_threshold,
                    min_valid_lag_fraction=residual_cfg.min_valid_lag_fraction,
                    smoothing_width=residual_cfg.smoothing_width,
                    replicates=residual_cfg.bootstrap_replicates,
                    seed=residual_cfg.bootstrap_seed,
                    ci_lags=computed_ci_lags,
                    probe_family_filter=family,
                )
                boot["probe_id"] = probe_ids
                boot["probe_family"] = family
                boot["split"] = split
                for key, value in metadata.items():
                    boot[key] = value
                bootstrap_rows.append(boot)
            if split == "val" and not args.skip_permutation:
                perm = compute_permutation_timescales(
                    chunk_paths,
                    feature_indices=np.arange(len(probe_ids), dtype=np.int64),
                    max_lag=max_lag,
                    min_valid_docs=residual_cfg.valid_doc_threshold,
                    min_valid_lag_fraction=residual_cfg.min_valid_lag_fraction,
                    smoothing_width=residual_cfg.smoothing_width,
                    replicates=residual_cfg.permutation_replicates,
                    seed=config.seed,
                    probe_family_filter=family,
                )
                if not perm.empty:
                    perm["probe_id"] = np.tile(probe_ids, residual_cfg.permutation_replicates)
                    perm["probe_family"] = family
                    perm["split"] = split
                    perm["control"] = "document_permutation"
                    perm["gk_positive_validation_persistence"] = False
                    for key, value in metadata.items():
                        perm[key] = value
                    all_rows.append(perm)

    timescales = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    save_parquet(timescales, store.residual_probe_timescales_path)
    if bootstrap_rows:
        save_parquet(pd.concat(bootstrap_rows, ignore_index=True), store.residual_bootstrap_timescales_path)
    real_val = timescales[(timescales["split"] == "val") & (timescales["control"] == "real")] if not timescales.empty else timescales
    summary = summarize_probe_timescales(real_val) if not timescales.empty else {}
    b1_summary = b1_decision_summary(timescales, within_profiles_by_family=val_within_profiles) if not timescales.empty else {}
    summary.update(
        {
            **metadata,
            "max_lag": int(max_lag),
            "sensitivity_max_lag": residual_cfg.sensitivity_max_lag,
            "ci_lags_requested": [int(lag) for lag in residual_cfg.ci_lags],
            "ci_lags_computed": [int(lag) for lag in computed_ci_lags],
            "distribution": config.dataset.dataset_name,
            "b1_decision": b1_summary,
        }
    )
    save_json(summary, store.residual_probe_summary_path)
    if not real_val.empty and (real_val["probe_family"] == "time_lagged").any():
        tl = real_val[real_val["probe_family"] == "time_lagged"]
        if "gk_positive_validation_persistence" in tl:
            fit_summary = load_json(store.time_lagged_fit_summary_path) if Path(store.time_lagged_fit_summary_path).exists() else {}
            fit_summary.update(
                {
                    "validation_positive_persistence_filter_lags": [lag for lag in [8, 16, 32] if lag <= max_lag],
                    "positive_validation_persistence_count": int(tl["gk_positive_validation_persistence"].sum()),
                    "positive_validation_persistence_fraction": float(tl["gk_positive_validation_persistence"].mean()),
                    "anti_persistent_or_sign_changing_count": int((~tl["gk_positive_validation_persistence"].astype(bool)).sum()),
                    **metadata,
                }
            )
            save_json(fit_summary, store.time_lagged_fit_summary_path)
    maybe_run_time_lagged_ridge_sensitivity(config, store, residual_cfg, max_lag=max_lag, metadata=metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
