#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from _common import artifact_metadata, load_config_and_store, require_residual_geometry_config, residual_batches, split_contexts, token_matrix
from residual_geometry.autocorr.estimators import AutocorrAccumulator, build_timescale_table, compute_document_autocorr
from residual_geometry.residuals.probes import (
    ResidualProbeSet,
    concatenate_probe_sets,
    load_probe_set,
    probe_set_from_directions,
    random_residual_probe_set,
    save_probe_set,
)
from residual_geometry.residuals.projections import project_residuals
from residual_geometry.residuals.provider import ResidualStreamProvider
from residual_geometry.residuals.time_lagged import estimate_covariances_streaming, fit_time_lagged_from_covariances
from residual_geometry.subspace.fat_subspace import save_residual_basis_artifact
from residual_geometry.subspace.residual_geometry import deduplicate_ranked_probes, orthonormal_basis
from residual_geometry.utils.io import ensure_dir, load_json, save_json, save_parquet


@dataclass(frozen=True)
class BasisSpec:
    family: str
    k: int
    probe_ids: np.ndarray
    probe_families: np.ndarray
    basis: np.ndarray


@dataclass(frozen=True)
class EvalSet:
    family: str
    probes: ResidualProbeSet


def _rename_probe_set(probes: ResidualProbeSet, family: str, prefix: str) -> ResidualProbeSet:
    return ResidualProbeSet(
        directions=probes.directions.astype(np.float32, copy=False),
        probe_ids=np.asarray([f"{prefix}_{idx:05d}" for idx in range(probes.n_probes)], dtype=object),
        probe_family=np.full(probes.n_probes, family, dtype=object),
    )


def _repair_legacy_projection_probe_families(probes: ResidualProbeSet) -> ResidualProbeSet:
    """Recover family names from legacy projection-collapse artifacts with <U1 metadata."""
    families = probes.probe_family.astype(str, copy=True)
    probe_ids = probes.probe_ids.astype(str)
    if not np.all(np.char.str_len(families) <= 1):
        return probes

    repaired = families.astype(object)
    repaired[(families == "r") & np.char.startswith(probe_ids, "heldout_random_residual_")] = "random_heldout"
    repaired[(families == "l") & np.char.startswith(probe_ids, "heldout_time_lagged_residual_")] = "lag_heldout"
    repaired[(families == "t") & np.char.startswith(probe_ids, "basis_time_lagged_residual_")] = "time_lagged"
    if np.array_equal(repaired, families):
        return probes
    return ResidualProbeSet(
        directions=probes.directions.astype(np.float32, copy=False),
        probe_ids=probe_ids.astype(object),
        probe_family=repaired.astype(object),
    )


def _train_shards(store, heldout_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    train_tokens = token_matrix(split_contexts(store, "train"))
    if len(train_tokens) < 2:
        raise ValueError("projection collapse requires at least two train documents")
    heldout_count = max(1, int(round(len(train_tokens) * heldout_fraction)))
    heldout_count = min(heldout_count, len(train_tokens) - 1)
    return train_tokens[:-heldout_count], train_tokens[-heldout_count:]


def _fit_time_lagged_on_tokens(
    provider: ResidualStreamProvider,
    config,
    residual_cfg,
    tokens: np.ndarray,
    output_directions: int,
    id_prefix: str,
    family: str,
) -> ResidualProbeSet:
    def arrays():
        for _, residuals in residual_batches(provider, tokens, residual_cfg.projections.batch_size):
            yield residuals

    mean, sigma0, sigma_lag, pair_count, _ = estimate_covariances_streaming(
        residual_batches=arrays,
        lag_set=residual_cfg.time_lagged.lag_set,
        d_model=config.model.d_model,
    )
    fit = fit_time_lagged_from_covariances(
        mean=mean,
        sigma0=sigma0,
        sigma_lag=sigma_lag,
        pair_count=pair_count,
        lag_set=residual_cfg.time_lagged.lag_set,
        whitening_pcs=residual_cfg.time_lagged.whitening_pcs,
        output_directions=output_directions,
        ridge_scale=residual_cfg.time_lagged.ridge_scale,
        max_ridge_scale=residual_cfg.time_lagged.max_ridge_scale,
        condition_threshold=residual_cfg.time_lagged.condition_number_threshold,
    )
    return _rename_probe_set(fit.probes, family=family, prefix=id_prefix)


def _load_pca_and_random(store) -> tuple[ResidualProbeSet, ResidualProbeSet]:
    if not Path(store.residual_pca_directions_path).exists() or not Path(store.random_residual_directions_path).exists():
        raise FileNotFoundError("Missing PCA/random probe directions; run 02_compute_residual_probes.py first.")
    return load_probe_set(store.residual_pca_directions_path), load_probe_set(store.random_residual_directions_path)


def _existing_validation_timescales(store) -> pd.DataFrame:
    if not Path(store.residual_probe_timescales_path).exists():
        raise FileNotFoundError("Missing residual-probe timescales; run 03_compute_residual_autocorr.py first.")
    timescales = pd.read_parquet(store.residual_probe_timescales_path)
    timescales = timescales[(timescales["split"] == "val") & (timescales["control"] == "real")].copy()
    if "tau_valid_within" in timescales:
        timescales = timescales[timescales["tau_valid_within"].astype(bool)].copy()
    return timescales


def _timescales_for_probes(
    provider: ResidualStreamProvider,
    tokens: np.ndarray,
    probes: ResidualProbeSet,
    residual_cfg,
    split: str,
) -> pd.DataFrame:
    accumulator = AutocorrAccumulator(n_features=probes.n_probes, max_lag=residual_cfg.max_lag)
    for _, residuals in residual_batches(provider, tokens, residual_cfg.projections.batch_size):
        projected = project_residuals(residuals, probes)
        accumulator.update(compute_document_autocorr(projected, max_lag=residual_cfg.max_lag, estimator="within"))
    result = accumulator.finalize()
    table = build_timescale_table(
        feature_indices=np.arange(probes.n_probes, dtype=np.int64),
        estimator_results={"within": result},
        max_lag=residual_cfg.max_lag,
        min_valid_docs=residual_cfg.valid_doc_threshold,
        min_valid_lag_fraction=residual_cfg.min_valid_lag_fraction,
        smoothing_width=residual_cfg.smoothing_width,
    )
    table["probe_id"] = probes.probe_ids.astype(str)
    table["probe_family"] = probes.probe_family.astype(str)
    table["split"] = split
    table["control"] = "real"
    table["gk_positive_validation_persistence"] = True
    return table


def _basis_specs(
    candidate_probes: ResidualProbeSet,
    candidate_timescales: pd.DataFrame,
    k_values: list[int],
    max_random_k: int,
) -> list[BasisSpec]:
    specs: list[BasisSpec] = []
    residual_first = deduplicate_ranked_probes(candidate_probes, candidate_timescales, threshold=0.95)
    pca = candidate_timescales[candidate_timescales["probe_family"] == "pca"].sort_values("feature_index").copy()
    random = candidate_timescales[candidate_timescales["probe_family"] == "random"].sort_values("feature_index").copy()
    id_to_idx = {str(pid): idx for idx, pid in enumerate(candidate_probes.probe_ids.astype(str))}
    for k in k_values:
        for family, table in [("residual_first", residual_first), ("pca", pca), ("random_control", random)]:
            if family == "random_control" and k > max_random_k:
                continue
            if len(table) < k:
                continue
            probe_ids = table.head(k)["probe_id"].astype(str).to_numpy()
            idx = [id_to_idx[probe_id] for probe_id in probe_ids if probe_id in id_to_idx]
            if len(idx) < k:
                continue
            directions = candidate_probes.directions[idx]
            specs.append(
                BasisSpec(
                    family=family,
                    k=k,
                    probe_ids=probe_ids,
                    probe_families=table.head(k)["probe_family"].astype(str).to_numpy(),
                    basis=orthonormal_basis(directions, k),
                )
            )
    return specs


def _basis_none() -> BasisSpec:
    return BasisSpec(
        family="none",
        k=0,
        probe_ids=np.asarray([], dtype=str),
        probe_families=np.asarray([], dtype=str),
        basis=np.zeros((1, 0), dtype=np.float32),
    )


def _save_projection_collapse_bases(
    store,
    basis_specs: list[BasisSpec],
    metadata: dict[str, object],
) -> str:
    basis_dir = os.path.join(store.subspace_dir, "projection_collapse_bases")
    ensure_dir(basis_dir)
    for spec in basis_specs:
        save_residual_basis_artifact(
            os.path.join(basis_dir, f"{spec.family}_k{spec.k}.npz"),
            basis=spec.basis,
            source_probe_ids=spec.probe_ids,
            source_probe_family=spec.probe_families,
            basis_family=np.array(spec.family),
            ranking_rule=np.array("projection_collapse_candidate_validation_ranking"),
            **{key: np.array(str(value)) for key, value in metadata.items()},
        )
    return basis_dir


def _before_after_tables(
    before_result,
    after_result,
    probes: ResidualProbeSet,
    residual_cfg,
) -> pd.DataFrame:
    before = build_timescale_table(
        feature_indices=np.arange(probes.n_probes, dtype=np.int64),
        estimator_results={"before": before_result},
        max_lag=residual_cfg.max_lag,
        min_valid_docs=residual_cfg.valid_doc_threshold,
        min_valid_lag_fraction=residual_cfg.min_valid_lag_fraction,
        smoothing_width=residual_cfg.smoothing_width,
    )
    after = build_timescale_table(
        feature_indices=np.arange(probes.n_probes, dtype=np.int64),
        estimator_results={"after": after_result},
        max_lag=residual_cfg.max_lag,
        min_valid_docs=residual_cfg.valid_doc_threshold,
        min_valid_lag_fraction=residual_cfg.min_valid_lag_fraction,
        smoothing_width=residual_cfg.smoothing_width,
    )
    return pd.DataFrame(
        {
            "eval_probe_id": probes.probe_ids.astype(str),
            "eval_probe_family": probes.probe_family.astype(str),
            "tau_before": before["tau_before"].to_numpy(),
            "tau_after": after["tau_after"].to_numpy(),
            "tau_valid_before": before["tau_valid_before"].to_numpy(),
            "tau_valid_after": after["tau_valid_after"].to_numpy(),
            "right_censored_before": before["right_censored_before"].to_numpy(),
            "right_censored_after": after["right_censored_after"].to_numpy(),
            "valid_lag_fraction_before": before["valid_lag_fraction_before"].to_numpy(),
            "valid_lag_fraction_after": after["valid_lag_fraction_after"].to_numpy(),
        }
    )


def _collapse_fraction(before: np.ndarray, after: np.ndarray, baseline_tau: float) -> tuple[float, float, float]:
    before_excess = np.maximum(before.astype(np.float64) - baseline_tau, 0.0)
    after_excess = np.maximum(after.astype(np.float64) - baseline_tau, 0.0)
    before_total = float(before_excess.sum())
    after_total = float(after_excess.sum())
    return before_total, after_total, float(1.0 - after_total / max(before_total, 1e-12))


def _summarize_rows(rows: pd.DataFrame, baseline_tau: float) -> dict[str, object]:
    valid = rows[rows["tau_valid_before"].astype(bool) & rows["tau_valid_after"].astype(bool)].copy()
    if valid.empty:
        return {"eval_probe_count": int(len(rows)), "valid_eval_probe_count": 0}
    before = valid["tau_before"].to_numpy(dtype=np.float64)
    after = valid["tau_after"].to_numpy(dtype=np.float64)
    before_total, after_total, collapse = _collapse_fraction(before, after, baseline_tau)
    high_threshold = float(np.quantile(before, 0.5))
    high = valid[valid["tau_before"] > high_threshold]
    low = valid[valid["tau_before"] <= high_threshold]
    high_before, high_after, high_collapse = _collapse_fraction(
        high["tau_before"].to_numpy(dtype=np.float64),
        high["tau_after"].to_numpy(dtype=np.float64),
        baseline_tau,
    ) if not high.empty else (0.0, 0.0, 0.0)
    low_before, low_after, low_collapse = _collapse_fraction(
        low["tau_before"].to_numpy(dtype=np.float64),
        low["tau_after"].to_numpy(dtype=np.float64),
        baseline_tau,
    ) if not low.empty else (0.0, 0.0, 0.0)
    return {
        "eval_probe_count": int(len(rows)),
        "valid_eval_probe_count": int(len(valid)),
        "baseline_tau": float(baseline_tau),
        "median_tau_before": float(np.median(before)),
        "median_tau_after": float(np.median(after)),
        "q90_tau_before": float(np.quantile(before, 0.9)),
        "q90_tau_after": float(np.quantile(after, 0.9)),
        "lifetime_excess_before": before_total,
        "lifetime_excess_after": after_total,
        "collapse_fraction": collapse,
        "high_lifetime_excess_before": high_before,
        "high_lifetime_excess_after": high_after,
        "high_lifetime_collapse_fraction": high_collapse,
        "low_lifetime_excess_before": low_before,
        "low_lifetime_excess_after": low_after,
        "low_lifetime_collapse_fraction": low_collapse,
        "high_gt_low_collapse": bool(high_collapse > low_collapse),
    }


def _evaluate_split(
    provider: ResidualStreamProvider,
    tokens: np.ndarray,
    eval_sets: list[EvalSet],
    basis_specs: list[BasisSpec],
    residual_cfg,
    baseline_tau: float,
    split: str,
    metadata: dict[str, object],
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    none = _basis_none()
    # Pre-fuse (Q_k^T @ J^T) per (eval_set, spec) before the batch loop.
    overlaps = {
        (es.family, spec.family, spec.k): spec.basis.T @ es.probes.directions.T.astype(np.float32)
        for es in eval_sets
        for spec in basis_specs
    }
    # One accumulator per (eval_set, spec_key); batch loop runs once — not once per eval_set.
    before_acc = {es.family: AutocorrAccumulator(n_features=es.probes.n_probes, max_lag=residual_cfg.max_lag) for es in eval_sets}
    after_acc = {
        (es.family, spec.family, spec.k): AutocorrAccumulator(n_features=es.probes.n_probes, max_lag=residual_cfg.max_lag)
        for es in eval_sets
        for spec in [none, *basis_specs]
    }
    for _, residuals in residual_batches(provider, tokens, residual_cfg.projections.batch_size):
        r = residuals.astype(np.float32, copy=False)
        for es in eval_sets:
            eval_proj = project_residuals(r, es.probes)
            ac = compute_document_autocorr(eval_proj, max_lag=residual_cfg.max_lag, estimator="within")
            before_acc[es.family].update(ac)
            after_acc[(es.family, none.family, none.k)].update(ac)
            for spec in basis_specs:
                basis_proj = np.einsum("btd,dk->btk", r, spec.basis, optimize=True)
                projected = eval_proj - basis_proj @ overlaps[(es.family, spec.family, spec.k)]
                after_acc[(es.family, spec.family, spec.k)].update(
                    compute_document_autocorr(projected, max_lag=residual_cfg.max_lag, estimator="within")
                )
    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for es in eval_sets:
        before_result = before_acc[es.family].finalize()
        for spec in [none, *basis_specs]:
            rows = _before_after_tables(before_result, after_acc[(es.family, spec.family, spec.k)].finalize(), es.probes, residual_cfg)
            rows["split"] = split
            rows["eval_family"] = es.family
            rows["basis_family"] = spec.family
            rows["basis_k"] = int(spec.k)
            rows["basis_probe_count"] = int(len(spec.probe_ids))
            rows["basis_eval_overlap_policy"] = "fixed_heldout_eval_set"
            for key, value in metadata.items():
                rows[key] = value
            all_rows.append(rows)
            item = _summarize_rows(rows, baseline_tau=baseline_tau)
            item.update({"split": split, "eval_family": es.family, "basis_family": spec.family, "basis_k": int(spec.k), **metadata})
            summary_rows.append(item)
    return all_rows, summary_rows


def _monotonicity_summary(summary: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for (split, eval_family, basis_family), group in summary[summary["basis_family"] != "none"].groupby(
        ["split", "eval_family", "basis_family"]
    ):
        ordered = group.sort_values("basis_k")
        diffs = np.diff(ordered["collapse_fraction"].to_numpy(dtype=np.float64))
        out[f"{split}:{eval_family}:{basis_family}"] = {
            "minimum_step_delta": float(diffs.min()) if len(diffs) else 0.0,
            "approximately_monotone": bool(np.all(diffs >= -0.05)) if len(diffs) else True,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run robust projection-collapse test with fixed held-out eval probes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--heldout-train-fraction", type=float, default=0.2)
    parser.add_argument("--heldout-lag-directions", type=int, default=None)
    parser.add_argument(
        "--refit-probes",
        action="store_true",
        help="Refit projection-collapse candidate and held-out eval probes. By default existing probes are reused, even with --overwrite.",
    )
    args = parser.parse_args()

    config, store = load_config_and_store(args.config)
    residual_cfg = require_residual_geometry_config(config)
    if residual_cfg.subspace is None:
        raise ValueError("residual_geometry.subspace is required for projection collapse.")
    output_path = os.path.join(store.subspace_dir, "projection_autocorr.parquet")
    summary_path = os.path.join(store.subspace_dir, "projection_collapse_summary.json")
    summary_table_path = os.path.join(store.subspace_dir, "projection_collapse_summary.parquet")
    heldout_path = os.path.join(store.subspace_dir, "heldout_projection_eval_probes.npz")
    candidate_path = os.path.join(store.subspace_dir, "projection_collapse_candidate_time_lagged.npz")
    if Path(output_path).exists() and Path(summary_path).exists() and not args.overwrite:
        return 0

    metadata = artifact_metadata(config, "residual_geometry_05_projection_collapse")
    provider = ResidualStreamProvider(config)
    provider.load()
    train_a, train_b = _train_shards(store, args.heldout_train_fraction)
    heldout_lag_count = args.heldout_lag_directions
    if heldout_lag_count is None:
        heldout_lag_count = 64 if config.mode == "smoke" else 128 if config.mode == "pilot" else 256
    pca_probes, random_candidate = _load_pca_and_random(store)
    existing_timescales = _existing_validation_timescales(store)
    val_tokens = token_matrix(split_contexts(store, "val"))

    if Path(candidate_path).exists() and not args.refit_probes:
        candidate_lag = _repair_legacy_projection_probe_families(load_probe_set(candidate_path))
    else:
        candidate_lag = _fit_time_lagged_on_tokens(
            provider,
            config,
            residual_cfg,
            tokens=train_a,
            output_directions=residual_cfg.time_lagged.output_directions,
            id_prefix="basis_time_lagged_residual",
            family="time_lagged",
        )
        save_probe_set(candidate_lag, candidate_path, train_shard=np.array("basis_train_a"), **{k: np.array(str(v)) for k, v in metadata.items()})
    candidate_lag_timescales = _timescales_for_probes(provider, val_tokens, candidate_lag, residual_cfg, split="val")
    pca_random_timescales = existing_timescales[existing_timescales["probe_family"].isin(["pca", "random"])].copy()
    candidate_timescales = pd.concat([pca_random_timescales, candidate_lag_timescales], ignore_index=True)
    candidate_probes = concatenate_probe_sets([candidate_lag, pca_probes, random_candidate])

    if Path(heldout_path).exists() and not args.refit_probes:
        heldout = _repair_legacy_projection_probe_families(load_probe_set(heldout_path))
        heldout_random = ResidualProbeSet(
            directions=heldout.directions[heldout.probe_family.astype(str) == "random_heldout"],
            probe_ids=heldout.probe_ids[heldout.probe_family.astype(str) == "random_heldout"],
            probe_family=heldout.probe_family[heldout.probe_family.astype(str) == "random_heldout"],
        )
        heldout_lag = ResidualProbeSet(
            directions=heldout.directions[heldout.probe_family.astype(str) == "lag_heldout"],
            probe_ids=heldout.probe_ids[heldout.probe_family.astype(str) == "lag_heldout"],
            probe_family=heldout.probe_family[heldout.probe_family.astype(str) == "lag_heldout"],
        )
        if heldout_random.n_probes == 0 or heldout_lag.n_probes == 0:
            families = sorted(set(heldout.probe_family.astype(str).tolist()))
            raise ValueError(
                "Loaded held-out projection eval probes are missing random_heldout or lag_heldout families "
                f"(found {families}). Re-run with --refit-probes to rebuild them."
            )
    else:
        heldout_random = random_residual_probe_set(
            d_model=config.model.d_model,
            count=heldout_lag_count,
            seed=residual_cfg.random.seed + 10_003,
            family="random_heldout",
        )
        heldout_random = _rename_probe_set(heldout_random, family="random_heldout", prefix="heldout_random_residual")
        heldout_lag = _fit_time_lagged_on_tokens(
            provider,
            config,
            residual_cfg,
            tokens=train_b,
            output_directions=heldout_lag_count,
            id_prefix="heldout_time_lagged_residual",
            family="lag_heldout",
        )
        save_probe_set(
            concatenate_probe_sets([heldout_random, heldout_lag]),
            heldout_path,
            train_shard=np.array("heldout_train_b"),
            heldout_train_fraction=np.array(args.heldout_train_fraction),
            **{k: np.array(str(v)) for k, v in metadata.items()},
        )

    random_valid = existing_timescales[existing_timescales["probe_family"] == "random"]
    baseline_tau = float(random_valid["tau_within"].median()) if not random_valid.empty else 1.0
    basis_specs = _basis_specs(
        candidate_probes,
        candidate_timescales,
        residual_cfg.subspace.k_values,
        max_random_k=random_candidate.n_probes,
    )
    projection_collapse_basis_dir = _save_projection_collapse_bases(store, basis_specs, metadata)
    eval_sets = [EvalSet("random_heldout", heldout_random), EvalSet("lag_heldout", heldout_lag)]
    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for split in args.splits:
        tokens = token_matrix(split_contexts(store, split))
        rows, summaries = _evaluate_split(
            provider=provider,
            tokens=tokens,
            eval_sets=eval_sets,
            basis_specs=basis_specs,
            residual_cfg=residual_cfg,
            baseline_tau=baseline_tau,
            split=split,
            metadata=metadata,
        )
        all_rows.extend(rows)
        summary_rows.extend(summaries)

    projection_rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_table = pd.DataFrame(summary_rows)
    save_parquet(projection_rows, output_path)
    save_parquet(summary_table, summary_table_path)

    positive = False
    gate_detail: dict[str, object] = {}
    if not summary_table.empty:
        val_lag = summary_table[(summary_table["split"] == "val") & (summary_table["eval_family"] == "lag_heldout")]
        rf_available = val_lag[val_lag["basis_family"] == "residual_first"]
        if not rf_available.empty:
            max_k = int(rf_available["basis_k"].max())
            rf = val_lag[(val_lag["basis_family"] == "residual_first") & (val_lag["basis_k"] == max_k)]
            rc = val_lag[(val_lag["basis_family"] == "random_control") & (val_lag["basis_k"] == max_k)]
            pca = val_lag[(val_lag["basis_family"] == "pca") & (val_lag["basis_k"] == max_k)]
            if not rf.empty:
                rf_c = float(rf.iloc[0].get("collapse_fraction", 0.0))
                rc_c = float(rc.iloc[0].get("collapse_fraction", 0.0)) if not rc.empty else 0.0
                pca_c = float(pca.iloc[0].get("collapse_fraction", 0.0)) if not pca.empty else float("nan")
                high_gt_low = bool(rf.iloc[0].get("high_gt_low_collapse", False))
                positive = bool(rf_c >= 0.20 and (rf_c - rc_c) >= 0.10 and high_gt_low)
                gate_detail = {
                    "basis_k": max_k,
                    "residual_first_lag_heldout_collapse": rf_c,
                    "random_control_lag_heldout_collapse": rc_c,
                    "pca_lag_heldout_collapse": pca_c,
                    "residual_first_minus_random_control": rf_c - rc_c,
                    "high_lifetime_collapse_exceeds_low_lifetime": high_gt_low,
                }

    save_json(
        {
            **metadata,
            "status": "positive_projection_collapse" if positive else "not_positive_or_not_evaluable",
            "projection_autocorr_path": output_path,
            "summary_table_path": summary_table_path,
            "heldout_projection_eval_probes_path": heldout_path,
            "candidate_time_lagged_basis_probes_path": candidate_path,
            "projection_collapse_basis_dir": projection_collapse_basis_dir,
            "basis_families": sorted(set(summary_table["basis_family"].astype(str))) if not summary_table.empty else [],
            "eval_families": sorted(set(summary_table["eval_family"].astype(str))) if not summary_table.empty else [],
            "splits": sorted(set(summary_table["split"].astype(str))) if not summary_table.empty else [],
            "k_values": [int(k) for k in residual_cfg.subspace.k_values],
            "baseline_tau": baseline_tau,
            "heldout_lag_directions": int(heldout_lag_count),
            "heldout_train_fraction": float(args.heldout_train_fraction),
            "gate_detail": gate_detail,
            "monotonicity": _monotonicity_summary(summary_table) if not summary_table.empty else {},
            "positive_criterion": {
                "residual_first_lag_heldout_collapse_at_max_k_ge_0_20": True,
                "residual_first_minus_random_control_at_max_k_ge_0_10": True,
                "high_lifetime_collapse_exceeds_low_lifetime": True,
                "q90_before_fixed_across_basis_family_and_k": True,
            },
            "note": (
                "Robust implementation uses fixed held-out random and independently fit held-out time-lagged eval probes "
                "for every basis family and k. Candidate time-lagged basis probes are fit on train shard A; held-out "
                "time-lagged eval probes are fit on disjoint train shard B; collapse is measured on validation/test residuals."
            ),
        },
        summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
