from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from residual_geometry.autocorr.bootstrap import bootstrap_timescales_for_array
from residual_geometry.autocorr.estimators import AutocorrAccumulator, build_timescale_table, compute_document_autocorr
from residual_geometry.residuals.projections import load_projection_chunk
from residual_geometry.utils.io import save_parquet


def projection_chunk_paths(split_dir: str) -> list[str]:
    return [str(path) for path in sorted(Path(split_dir).glob("chunk_*.npz"))]


def accumulate_projection_autocorr(
    chunk_paths: list[str],
    max_lag: int,
    estimator: str,
    probe_family_filter: str | None = None,
) -> tuple[AutocorrAccumulator, np.ndarray, np.ndarray]:
    if not chunk_paths:
        raise ValueError("no projection chunks found")
    first = load_projection_chunk(chunk_paths[0])
    families = first["probe_family"].astype(str)
    probe_ids = first["probe_ids"].astype(str)
    mask = families == probe_family_filter if probe_family_filter is not None else np.ones(len(families), dtype=bool)
    accumulator = AutocorrAccumulator(n_features=int(mask.sum()), max_lag=max_lag)
    for path in chunk_paths:
        chunk = load_projection_chunk(path)
        projections = chunk["projections"].astype(np.float32)[:, :, mask]
        accumulator.update(compute_document_autocorr(projections, max_lag=max_lag, estimator=estimator))
    return accumulator, probe_ids[mask], families[mask]


def accumulate_projection_autocorr_multi(
    chunk_paths: list[str],
    max_lag: int,
    estimators: list[str],
    probe_family_filter: str | None = None,
) -> tuple[dict[str, AutocorrAccumulator], np.ndarray, np.ndarray]:
    """Like accumulate_projection_autocorr but computes multiple estimators in one chunk-read pass."""
    if not chunk_paths:
        raise ValueError("no projection chunks found")
    first = load_projection_chunk(chunk_paths[0])
    families = first["probe_family"].astype(str)
    probe_ids = first["probe_ids"].astype(str)
    mask = families == probe_family_filter if probe_family_filter is not None else np.ones(len(families), dtype=bool)
    n_features = int(mask.sum())
    accumulators = {est: AutocorrAccumulator(n_features=n_features, max_lag=max_lag) for est in estimators}
    for path in chunk_paths:
        chunk = load_projection_chunk(path)
        projections = chunk["projections"].astype(np.float32)[:, :, mask]
        for est in estimators:
            accumulators[est].update(compute_document_autocorr(projections, max_lag=max_lag, estimator=est))
    return accumulators, probe_ids[mask], families[mask]


def compute_permutation_timescales(
    chunk_paths: list[str],
    feature_indices: np.ndarray,
    max_lag: int,
    min_valid_docs: int,
    min_valid_lag_fraction: float,
    smoothing_width: int,
    replicates: int,
    seed: int,
    probe_family_filter: str | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    if not chunk_paths:
        return pd.DataFrame()
    first = load_projection_chunk(chunk_paths[0])
    families = first["probe_family"].astype(str)
    mask = families == probe_family_filter if probe_family_filter is not None else np.ones(len(families), dtype=bool)
    for rep in range(replicates):
        accumulator = AutocorrAccumulator(n_features=int(mask.sum()), max_lag=max_lag)
        for path in chunk_paths:
            projections = load_projection_chunk(path)["projections"].astype(np.float32)[:, :, mask]
            permuted = projections.copy()
            for doc_i in range(permuted.shape[0]):
                permuted[doc_i] = permuted[doc_i, rng.permutation(permuted.shape[1]), :]
            accumulator.update(compute_document_autocorr(permuted, max_lag=max_lag, estimator="within"))
        result = accumulator.finalize()
        table = build_timescale_table(
            feature_indices=feature_indices,
            estimator_results={"perm_within": result},
            max_lag=max_lag,
            min_valid_docs=min_valid_docs,
            min_valid_lag_fraction=min_valid_lag_fraction,
            smoothing_width=smoothing_width,
        )
        table["permutation_replicate"] = rep
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def bootstrap_projection_timescales(
    chunk_paths: list[str],
    feature_indices: np.ndarray,
    estimator: str,
    max_lag: int,
    min_valid_docs: int,
    min_valid_lag_fraction: float,
    smoothing_width: int,
    replicates: int,
    seed: int,
    ci_lags: list[int],
    probe_family_filter: str | None = None,
) -> pd.DataFrame:
    if not chunk_paths:
        return pd.DataFrame()
    first = load_projection_chunk(chunk_paths[0])
    families = first["probe_family"].astype(str)
    mask = families == probe_family_filter if probe_family_filter is not None else np.ones(len(families), dtype=bool)
    arrays = [load_projection_chunk(path)["projections"].astype(np.float32)[:, :, mask] for path in chunk_paths]
    if not arrays:
        return pd.DataFrame()
    projections = np.concatenate(arrays, axis=0)
    return bootstrap_timescales_for_array(
        projections,
        feature_indices=feature_indices,
        estimator=estimator,
        max_lag=max_lag,
        min_valid_docs=min_valid_docs,
        min_valid_lag_fraction=min_valid_lag_fraction,
        replicates=replicates,
        seed=seed,
        ci_lags=[lag for lag in ci_lags if lag <= max_lag],
        smoothing_width=smoothing_width,
    )


def save_profiles_npz(path: str, profiles: np.ndarray, valid_doc_counts: np.ndarray, probe_ids: np.ndarray, probe_family: np.ndarray) -> None:
    import os

    from residual_geometry.utils.io import ensure_dir

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.savez_compressed(
        path,
        profiles=profiles.astype(np.float32),
        valid_doc_counts=valid_doc_counts.astype(np.int64),
        probe_ids=probe_ids.astype(str),
        probe_family=probe_family.astype(str),
    )


def summarize_probe_timescales(timescales: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {"probe_count": int(len(timescales))}
    if timescales.empty:
        return summary
    for family, group in timescales.groupby("probe_family"):
        valid = group[group["tau_valid_within"].astype(bool)] if "tau_valid_within" in group else group
        family_summary: dict[str, object] = {
            "attempted": int(len(group)),
            "valid_within": int(len(valid)),
            "right_censored_within": int(group["right_censored_within"].sum()) if "right_censored_within" in group else 0,
        }
        if "tau_within" in valid and not valid.empty:
            for q in [0.5, 0.75, 0.9, 0.95]:
                family_summary[f"q{int(q * 100)}_tau_within"] = float(valid["tau_within"].quantile(q))
        summary[str(family)] = family_summary
    if {"probe_family", "tau_within"}.issubset(timescales.columns):
        random = timescales[(timescales["probe_family"] == "random") & timescales["tau_valid_within"]]
        if not random.empty:
            random_q90 = float(random["tau_within"].quantile(0.9))
            summary["random_q90_tau_within"] = random_q90
            for family in ["pca", "time_lagged"]:
                group = timescales[(timescales["probe_family"] == family) & timescales["tau_valid_within"]]
                if not group.empty:
                    summary[f"{family}_q90_over_random_q90"] = float(group["tau_within"].quantile(0.9) / max(random_q90, 1e-12))
    return summary


def _valid_real(timescales: pd.DataFrame) -> pd.DataFrame:
    table = timescales.copy()
    if "control" in table:
        table = table[table["control"] == "real"]
    if "split" in table:
        table = table[table["split"] == "val"]
    if "tau_valid_within" in table:
        table = table[table["tau_valid_within"].astype(bool)]
    return table


def b1_decision_summary(timescales: pd.DataFrame, within_profiles_by_family: dict[str, np.ndarray] | None = None) -> dict[str, object]:
    real = _valid_real(timescales)
    summary: dict[str, object] = {
        "stage": "B1_residual_probe_autocorrelation_pilot",
        "valid_probe_fraction_by_family": {},
        "criteria": {},
        "interpretation": [],
    }
    if real.empty:
        summary["status"] = "no_valid_validation_probes"
        return summary
    attempted = timescales.copy()
    if "control" in attempted:
        attempted = attempted[attempted["control"] == "real"]
    if "split" in attempted:
        attempted = attempted[attempted["split"] == "val"]
    for family, group in attempted.groupby("probe_family"):
        valid_count = int(group["tau_valid_within"].sum()) if "tau_valid_within" in group else int(len(group))
        summary["valid_probe_fraction_by_family"][str(family)] = float(valid_count / max(len(group), 1))
    if "probe_family" in attempted.columns and "right_censored_within" in attempted.columns:
        non_random_att = attempted[attempted["probe_family"] != "random"]
        if not non_random_att.empty:
            censored_fraction = float(non_random_att["right_censored_within"].mean())
            summary["non_random_right_censored_fraction_within"] = censored_fraction
            summary["high_censoring_requires_768_lag_sensitivity"] = bool(censored_fraction > 0.30)
            if censored_fraction > 0.30:
                summary["interpretation"].append("high_censoring_768_lag_sensitivity_required")
    q = {
        str(family): {
            f"q{int(quantile * 100)}": float(group["tau_within"].quantile(quantile))
            for quantile in [0.5, 0.75, 0.9, 0.95]
        }
        for family, group in real.groupby("probe_family")
    }
    summary["tau_within_quantiles_by_family"] = q
    random_q90 = q.get("random", {}).get("q90")
    pca_q90 = q.get("pca", {}).get("q90")
    lag_q90 = q.get("time_lagged", {}).get("q90")
    combined_q95 = float(real["tau_within"].quantile(0.95))
    combined_q50 = float(real["tau_within"].quantile(0.5))
    summary["combined_q95_over_q50"] = float(combined_q95 / max(combined_q50, 1e-12))
    valid_coverage = all(value >= 0.9 for value in summary["valid_probe_fraction_by_family"].values())
    pca_exceeds_random = bool(random_q90 is not None and pca_q90 is not None and pca_q90 >= 1.2 * random_q90)
    heavy_combined = bool(summary["combined_q95_over_q50"] >= 2.0)
    lag_exceeds_random = bool(random_q90 is not None and lag_q90 is not None and lag_q90 >= 1.3 * random_q90)
    summary["criteria"] = {
        "valid_probe_coverage_ge_90pct": valid_coverage,
        "pca_q90_exceeds_random_q90_by_20pct": pca_exceeds_random,
        "combined_q95_over_q50_ge_2": heavy_combined,
        "time_lagged_q90_exceeds_random_q90_by_30pct": lag_exceeds_random,
    }
    if lag_exceeds_random:
        summary["interpretation"].append("time_lagged_upper_tail_exceeds_random")
    if pca_exceeds_random:
        summary["interpretation"].append("pca_upper_tail_exceeds_random")
    if heavy_combined:
        summary["interpretation"].append("combined_non_sae_distribution_heavy_tailed")
    perm = timescales.copy()
    if "control" in perm:
        perm = perm[perm["control"] == "document_permutation"]
    else:
        perm = perm.iloc[0:0]
    if "split" in perm:
        perm = perm[perm["split"] == "val"]
    if not perm.empty:
        reductions: dict[str, object] = {}
        for family, group in real.groupby("probe_family"):
            perm_group = perm[perm["probe_family"] == family]
            if perm_group.empty or "tau_perm_within" not in perm_group:
                continue
            threshold = group["tau_within"].quantile(0.9)
            top_ids = set(group.loc[group["tau_within"] >= threshold, "probe_id"].astype(str))
            real_top_median = float(group[group["probe_id"].astype(str).isin(top_ids)]["tau_within"].median())
            perm_top_median = float(perm_group[perm_group["probe_id"].astype(str).isin(top_ids)]["tau_perm_within"].median())
            reductions[str(family)] = {
                "top_decile_real_tau_within_median": real_top_median,
                "top_decile_document_permutation_tau_within_median": perm_top_median,
                "top_decile_tau_reduction_fraction": float(1.0 - perm_top_median / max(real_top_median, 1e-12)),
            }
        summary["document_permutation_control"] = reductions
        summary["criteria"]["document_permutation_reduces_top_probe_tau"] = any(
            isinstance(item, dict) and item.get("top_decile_tau_reduction_fraction", 0.0) > 0.0
            for item in reductions.values()
        )
    if within_profiles_by_family is not None:
        random_profiles = within_profiles_by_family.get("random")
        check_lags = [16, 32, 64, 128]
        if random_profiles is not None and random_profiles.ndim == 2 and random_profiles.shape[0] > 0:
            valid_check_lags = [lag for lag in check_lags if lag < random_profiles.shape[1]]
            if valid_check_lags and not real.empty and "feature_index" in real.columns:
                random_median = np.nanmedian(random_profiles[:, valid_check_lags], axis=0)
                top_threshold = real["tau_within"].quantile(0.9)
                top_rows = real[real["tau_within"] >= top_threshold]
                top_profiles_list = []
                for _, row in top_rows.iterrows():
                    family = str(row.get("probe_family", ""))
                    idx = int(row.get("feature_index", -1))
                    fam_profs = within_profiles_by_family.get(family)
                    if fam_profs is not None and fam_profs.ndim == 2 and 0 <= idx < len(fam_profs):
                        top_profiles_list.append(fam_profs[idx, valid_check_lags])
                if top_profiles_list:
                    top_median = np.nanmedian(np.stack(top_profiles_list), axis=0)
                    lags_above = int(np.sum(top_median > random_median))
                    criterion5_passed = bool(lags_above >= 2)
                    summary["criteria"]["top_persistent_probes_above_random_at_ge2_lags"] = criterion5_passed
                    summary["criterion5_detail"] = {
                        "check_lags_used": valid_check_lags,
                        "top_probe_count": len(top_profiles_list),
                        "lags_above_random_median": lags_above,
                    }
                    if criterion5_passed:
                        summary["interpretation"].append("top_persistent_probes_above_random_at_intermediate_lags")
    minimally_positive = bool(
        valid_coverage
        and (pca_exceeds_random or heavy_combined or lag_exceeds_random)
        and summary["criteria"].get("document_permutation_reduces_top_probe_tau", False)
    )
    summary["status"] = "minimally_positive_or_suggestive" if minimally_positive else "not_positive_by_provisional_b1_criteria"
    return summary


def save_timescale_outputs(timescales: pd.DataFrame, path: str) -> None:
    save_parquet(timescales, path)
