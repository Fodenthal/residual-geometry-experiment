from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from residual_geometry.autocorr.bootstrap import bootstrap_timescales_for_array, flag_high_confidence_slow
from residual_geometry.autocorr.estimators import (
    AutocorrAccumulator,
    AutocorrResult,
    EventTriggeredAccumulator,
    QuantileReservoir,
    build_timescale_table,
    compute_document_autocorr,
)
from residual_geometry.autocorr.nulls import null_tau_table
from residual_geometry.config.schema import PersistentStateConfig
from residual_geometry.data.artifact_store import PersistentStateArtifactStore
from residual_geometry.utils.io import load_json, load_parquet, save_json, save_parquet


ActivationIteratorFactory = Callable[[], Iterable[np.ndarray]]


@dataclass(frozen=True)
class StageBActivationSource:
    feature_indices: np.ndarray
    make_iterator: ActivationIteratorFactory
    document_count: int
    source_description: str


def estimate_quantiles(
    source: StageBActivationSource,
    reservoir_capacity: int = 20_000,
    seed: int = 13,
) -> pd.DataFrame:
    reservoir = QuantileReservoir(
        n_features=len(source.feature_indices),
        capacity=reservoir_capacity,
        seed=seed,
    )
    for activations in source.make_iterator():
        reservoir.update(activations.astype(np.float32, copy=False))
    return reservoir.table()


def binary_thresholds(config: PersistentStateConfig, quantiles: pd.DataFrame) -> np.ndarray:
    return np.where(
        quantiles["positive_count"].to_numpy() < config.autocorr.binary_min_positive_activations,
        quantiles["q95"].to_numpy(),
        quantiles["q99"].to_numpy(),
    )


def compute_autocorr_results(
    source: StageBActivationSource,
    config: PersistentStateConfig,
    thresholds: np.ndarray,
) -> dict[str, AutocorrResult]:
    n_features = len(source.feature_indices)
    accumulators = {
        name: AutocorrAccumulator(n_features, config.autocorr.max_lag)
        for name in ["raw", "within", "binary", "active_doc"]
    }
    for activations in source.make_iterator():
        acts = activations.astype(np.float32, copy=False)
        for estimator, accumulator in accumulators.items():
            result = compute_document_autocorr(
                acts,
                max_lag=config.autocorr.max_lag,
                estimator=estimator,
                thresholds=thresholds if estimator == "binary" else None,
            )
            accumulator.update(result)
    return {name: accumulator.finalize() for name, accumulator in accumulators.items()}


def build_timescales_with_quantiles(
    source: StageBActivationSource,
    config: PersistentStateConfig,
    estimator_results: dict[str, AutocorrResult],
    quantiles: pd.DataFrame,
) -> pd.DataFrame:
    timescales = build_timescale_table(
        feature_indices=source.feature_indices,
        estimator_results=estimator_results,
        max_lag=config.autocorr.max_lag,
        min_valid_docs=config.autocorr.valid_doc_threshold,
        min_valid_lag_fraction=config.autocorr.min_valid_lag_fraction,
        smoothing_width=config.autocorr.smoothing_width,
    )
    quantiles_with_ids = quantiles.copy()
    quantiles_with_ids["feature_index"] = source.feature_indices
    return timescales.merge(
        quantiles_with_ids[
            [
                "feature_index",
                "q50",
                "q75",
                "q90",
                "q95",
                "q99",
                "positive_count",
                "reservoir_count",
                "threshold_fallback",
                "quantile_approximation",
            ]
        ],
        on="feature_index",
        how="left",
    )


def compute_event_triggered(
    source: StageBActivationSource,
    config: PersistentStateConfig,
    quantiles: pd.DataFrame,
) -> pd.DataFrame:
    accumulator = EventTriggeredAccumulator(source.feature_indices, quantiles, config.autocorr.event_offsets)
    for activations in source.make_iterator():
        accumulator.update(activations.astype(np.float32, copy=False))
    return accumulator.table()


def _collect_feature_block(source: StageBActivationSource, start: int, stop: int) -> np.ndarray:
    arrays = [
        activations[:, :, start:stop].astype(np.float32, copy=False)
        for activations in source.make_iterator()
    ]
    if not arrays:
        raise ValueError("Activation source produced no arrays.")
    return np.concatenate(arrays, axis=0)


def compute_bootstrap(
    source: StageBActivationSource,
    config: PersistentStateConfig,
    thresholds: np.ndarray,
    feature_block_size: int = 128,
) -> pd.DataFrame:
    parts = []
    ci_lags = [lag for lag in [1, 2, 4, 8, 16, 32, 64, 128, 256] if lag <= config.autocorr.max_lag]
    for start in range(0, len(source.feature_indices), feature_block_size):
        stop = min(start + feature_block_size, len(source.feature_indices))
        block = _collect_feature_block(source, start, stop)
        block_features = source.feature_indices[start:stop]
        for estimator in ["within", "binary", "raw"]:
            parts.append(
                bootstrap_timescales_for_array(
                    block,
                    feature_indices=block_features,
                    estimator=estimator,
                    max_lag=config.autocorr.max_lag,
                    min_valid_docs=config.autocorr.valid_doc_threshold,
                    min_valid_lag_fraction=config.autocorr.min_valid_lag_fraction,
                    replicates=config.autocorr.bootstrap_replicates,
                    seed=13 + start + {"within": 0, "binary": 100_000, "raw": 200_000}[estimator],
                    thresholds=thresholds[start:stop] if estimator == "binary" else None,
                    ci_lags=ci_lags,
                )
            )
    bootstrap = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return bootstrap


def compute_null_tables(
    source: StageBActivationSource,
    config: PersistentStateConfig,
    feature_block_size: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched_parts = []
    permuted_parts = []
    for start in range(0, len(source.feature_indices), feature_block_size):
        stop = min(start + feature_block_size, len(source.feature_indices))
        activations = _collect_feature_block(source, start, stop)
        common = dict(
            activations=activations,
            feature_indices=source.feature_indices[start:stop],
            max_lag=config.autocorr.max_lag,
            min_valid_docs=config.autocorr.valid_doc_threshold,
            min_valid_lag_fraction=config.autocorr.min_valid_lag_fraction,
        )
        matched_parts.append(
            null_tau_table(
                **common,
                null_name="matched_sparsity",
                replicates=config.nulls.matched_sparsity_replicates,
                seed=config.seed + 1000 + start,
            )
        )
        permuted_parts.append(
            null_tau_table(
                **common,
                null_name="document_permutation",
                replicates=config.nulls.document_permutation_replicates,
                seed=config.seed + 2000 + start,
            )
        )
    return pd.concat(matched_parts, ignore_index=True), pd.concat(permuted_parts, ignore_index=True)


def compute_bootstrap_and_null_tables(
    source: StageBActivationSource,
    config: PersistentStateConfig,
    thresholds: np.ndarray,
    feature_block_size: int = 1024,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bootstrap_parts = []
    matched_parts = []
    permuted_parts = []
    ci_lags = [lag for lag in [1, 2, 4, 8, 16, 32, 64, 128, 256] if lag <= config.autocorr.max_lag]
    for start in range(0, len(source.feature_indices), feature_block_size):
        stop = min(start + feature_block_size, len(source.feature_indices))
        activations = _collect_feature_block(source, start, stop)
        block_features = source.feature_indices[start:stop]
        for estimator in ["within", "binary", "raw"]:
            bootstrap_parts.append(
                bootstrap_timescales_for_array(
                    activations,
                    feature_indices=block_features,
                    estimator=estimator,
                    max_lag=config.autocorr.max_lag,
                    min_valid_docs=config.autocorr.valid_doc_threshold,
                    min_valid_lag_fraction=config.autocorr.min_valid_lag_fraction,
                    replicates=config.autocorr.bootstrap_replicates,
                    seed=13 + start + {"within": 0, "binary": 100_000, "raw": 200_000}[estimator],
                    thresholds=thresholds[start:stop] if estimator == "binary" else None,
                    ci_lags=ci_lags,
                )
            )
        common = dict(
            activations=activations,
            feature_indices=block_features,
            max_lag=config.autocorr.max_lag,
            min_valid_docs=config.autocorr.valid_doc_threshold,
            min_valid_lag_fraction=config.autocorr.min_valid_lag_fraction,
        )
        matched_parts.append(
            null_tau_table(
                **common,
                null_name="matched_sparsity",
                replicates=config.nulls.matched_sparsity_replicates,
                seed=config.seed + 1000 + start,
            )
        )
        permuted_parts.append(
            null_tau_table(
                **common,
                null_name="document_permutation",
                replicates=config.nulls.document_permutation_replicates,
                seed=config.seed + 2000 + start,
            )
        )
    return (
        pd.concat(bootstrap_parts, ignore_index=True),
        pd.concat(matched_parts, ignore_index=True),
        pd.concat(permuted_parts, ignore_index=True),
    )


def compute_burn_in(
    source: StageBActivationSource,
    config: PersistentStateConfig,
    timescales: pd.DataFrame,
    high_confidence: pd.DataFrame | None,
) -> tuple[dict[str, AutocorrResult], pd.DataFrame, dict]:
    buckets = {
        "early": (0, 128, 64),
        "middle": (128, 256, 64),
        "late": (256, 512, 128),
    }
    accumulators = {
        name: AutocorrAccumulator(len(source.feature_indices), max_lag)
        for name, (_, _, max_lag) in buckets.items()
    }
    for activations in source.make_iterator():
        acts = activations.astype(np.float32, copy=False)
        for bucket, (start, stop, max_lag) in buckets.items():
            result = compute_document_autocorr(acts[:, start:stop, :], max_lag=max_lag, estimator="within")
            accumulators[bucket].update(result)
    results = {bucket: accumulator.finalize() for bucket, accumulator in accumulators.items()}

    bucket_tables = []
    for bucket, result in results.items():
        max_lag = buckets[bucket][2]
        table = build_timescale_table(
            feature_indices=source.feature_indices,
            estimator_results={"within": result},
            max_lag=max_lag,
            min_valid_docs=config.autocorr.valid_doc_threshold,
            min_valid_lag_fraction=config.autocorr.min_valid_lag_fraction,
            smoothing_width=config.autocorr.smoothing_width,
        )
        table = table.rename(
            columns={
                "tau_within": f"tau_within_{bucket}",
                "right_censored_within": f"right_censored_within_{bucket}",
                "tau_valid_within": f"tau_valid_within_{bucket}",
                "valid_lag_fraction_within": f"valid_lag_fraction_within_{bucket}",
            }
        )
        bucket_tables.append(
            table[
                [
                    "feature_index",
                    f"tau_within_{bucket}",
                    f"right_censored_within_{bucket}",
                    f"tau_valid_within_{bucket}",
                    f"valid_lag_fraction_within_{bucket}",
                ]
            ]
        )
    metrics = bucket_tables[0]
    for table in bucket_tables[1:]:
        metrics = metrics.merge(table, on="feature_index", how="outer")
    metrics = metrics.merge(
        timescales[["feature_index", "slow_score", "tau_within", "tau_valid_within"]],
        on="feature_index",
        how="left",
    )
    metrics["burn_in_delta"] = metrics["tau_within_late"] - metrics["tau_within_early"]
    valid_scores = metrics["slow_score"].rank(pct=True)
    metrics["timescale_bin"] = "other"
    metrics.loc[valid_scores <= 0.2, "timescale_bin"] = "fast"
    metrics.loc[(valid_scores >= 0.4) & (valid_scores <= 0.6), "timescale_bin"] = "medium"
    metrics.loc[valid_scores >= 0.8, "timescale_bin"] = "slow"
    if high_confidence is not None and not high_confidence.empty:
        high_confidence_set = set(high_confidence["feature_index"].tolist())
        metrics.loc[metrics["feature_index"].isin(high_confidence_set), "timescale_bin"] = "high_confidence_slow"
    summary = {
        "stage": "06_burn_in_analysis",
        "status": "PASS",
        "document_count": int(source.document_count),
        "median_burn_in_delta": float(metrics["burn_in_delta"].median()),
        "bin_median_burn_in_delta": {
            str(bin_name): float(group["burn_in_delta"].median())
            for bin_name, group in metrics.groupby("timescale_bin")
        },
        "slow_positive_burn_in_fraction": float(
            (metrics.loc[metrics["timescale_bin"].isin(["slow", "high_confidence_slow"]), "burn_in_delta"] > 0).mean()
        ),
    }
    return results, metrics, summary


def evaluate_stage_b_gate(store: PersistentStateArtifactStore) -> dict:
    protocol_path = Path(store.protocol_log_path)
    if protocol_path.exists():
        protocol = load_json(str(protocol_path))
        if protocol.get("synthetic_activations"):
            return {
                "stage": "stage_b_autocorr_gate",
                "status": "SYNTHETIC_SKIP",
                "reason": "synthetic activations are for local pipeline validation only",
                "may_run_truncation": False,
            }
    timescales = load_parquet(store.timescales_path)
    high_conf_path = Path(store.high_confidence_slow_features_path)
    if not high_conf_path.exists():
        return {
            "stage": "stage_b_autocorr_gate",
            "status": "FAIL",
            "reason": "bootstrap high-confidence slow features artifact is missing",
            "may_run_truncation": False,
        }
    high_conf = load_parquet(str(high_conf_path))
    if high_conf.empty:
        return {
            "stage": "stage_b_autocorr_gate",
            "status": "FAIL",
            "reason": "no high-confidence slow features",
            "may_run_truncation": False,
        }
    matched = load_parquet(str(Path(store.nulls_dir) / "matched_sparsity.parquet"))
    permuted = load_parquet(str(Path(store.nulls_dir) / "document_permutation.parquet"))
    real = timescales[["feature_index", "tau_within"]].merge(
        high_conf[["feature_index"]],
        on="feature_index",
        how="inner",
    )
    matched_within = matched[matched["estimator"] == "within"]
    matched_q95 = matched_within.groupby("feature_index")["tau"].quantile(0.95).rename("matched_sparsity_tau_q95").reset_index()
    survival = real.merge(matched_q95, on="feature_index", how="inner")
    survival["survives_matched_sparsity"] = survival["tau_within"] > survival["matched_sparsity_tau_q95"]
    survival_fraction = float(survival["survives_matched_sparsity"].mean()) if len(survival) else 0.0

    perm_within = permuted[permuted["estimator"] == "within"]
    perm_median = perm_within.groupby("feature_index")["tau"].median().rename("document_permutation_tau_median").reset_index()
    doc_perm = real.merge(perm_median, on="feature_index", how="inner")
    real_median = float(doc_perm["tau_within"].median()) if len(doc_perm) else 0.0
    null_median = float(doc_perm["document_permutation_tau_median"].median()) if len(doc_perm) else 0.0
    median_ratio = real_median / null_median if null_median > 0 else float("inf")
    passed = survival_fraction >= 0.70 and median_ratio >= 1.10
    return {
        "stage": "stage_b_autocorr_gate",
        "status": "PASS" if passed else "FAIL",
        "may_run_truncation": passed,
        "high_confidence_slow_count": int(len(high_conf)),
        "matched_sparsity_survival_fraction": survival_fraction,
        "real_high_confidence_median_tau_within": real_median,
        "document_permutation_median_tau_within": null_median,
        "real_to_document_permutation_median_ratio": median_ratio,
    }


def write_stage_b_outputs(
    source: StageBActivationSource,
    config: PersistentStateConfig,
    store: PersistentStateArtifactStore,
    quantiles: pd.DataFrame,
    estimator_results: dict[str, AutocorrResult],
    timescales: pd.DataFrame,
    event_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    high_confidence: pd.DataFrame,
    matched: pd.DataFrame,
    permuted: pd.DataFrame,
    burn_in_results: dict[str, AutocorrResult],
    burn_in_metrics: pd.DataFrame,
    burn_in_summary: dict,
    split_label: str = "train_val",
    residual_baseline_status: str = "not_run_in_disk_light_stage_b",
) -> dict:
    for estimator, result in estimator_results.items():
        np.savez_compressed(
            store.autocorr_profiles_path(split_label, estimator),
            profiles=result.profiles,
            valid_doc_counts=result.valid_doc_counts,
            feature_indices=source.feature_indices,
        )
    save_parquet(timescales, store.timescales_path)
    save_parquet(event_df, store.event_triggered_path)
    save_parquet(bootstrap_df, store.bootstrap_timescales_path)
    save_parquet(high_confidence, store.high_confidence_slow_features_path)
    save_parquet(matched, str(Path(store.nulls_dir) / "matched_sparsity.parquet"))
    save_parquet(permuted, str(Path(store.nulls_dir) / "document_permutation.parquet"))
    np.savez_compressed(
        store.burn_in_profiles_path,
        feature_indices=source.feature_indices,
        early_profiles=burn_in_results["early"].profiles,
        early_valid_doc_counts=burn_in_results["early"].valid_doc_counts,
        middle_profiles=burn_in_results["middle"].profiles,
        middle_valid_doc_counts=burn_in_results["middle"].valid_doc_counts,
        late_profiles=burn_in_results["late"].profiles,
        late_valid_doc_counts=burn_in_results["late"].valid_doc_counts,
    )
    save_parquet(burn_in_metrics, store.burn_in_metrics_path)
    save_json(burn_in_summary, store.burn_in_summary_path)
    save_json(
        {
            "stage": "04_compute_autocorr",
            "status": "PASS",
            "splits": ["train", "val"],
            "document_count": int(source.document_count),
            "feature_count": int(len(source.feature_indices)),
            "invalid_within": int((~timescales["tau_valid_within"]).sum()),
            "right_censored_within": int(timescales["right_censored_within"].sum()),
            "active_doc_invalid": int((~timescales["tau_valid_active_doc"]).sum()),
            "high_confidence_slow_count": int(len(high_confidence)),
            "bootstrap_run": True,
            "quantile_approximation_features": int(timescales["quantile_approximation"].sum()),
            "activation_source": source.source_description,
        },
        store.timescale_summary_path,
    )
    gate = evaluate_stage_b_gate(store)
    save_json(gate, store.stage_b_gate_path)
    save_json(
        {
            "stage": "05_compute_nulls",
            "status": "PASS",
            "matched_sparsity_rows": int(len(matched)),
            "document_permutation_rows": int(len(permuted)),
            "random_residual_status": residual_baseline_status,
            "residual_pca_status": residual_baseline_status,
            "stage_b_gate": gate,
        },
        str(Path(store.nulls_dir) / "null_summary.json"),
    )
    save_json({"stage": "04_compute_autocorr", "status": "PASS"}, store.qc_summary_path("stage_04"))
    save_json({"stage": "05_compute_nulls", "status": "PASS"}, store.qc_summary_path("stage_05"))
    save_json({"stage": "06_burn_in_analysis", "status": "PASS"}, store.qc_summary_path("stage_06"))
    return gate
