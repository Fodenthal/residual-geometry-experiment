#!/usr/bin/env python
"""Frozen S2 equal-rank supervised profile and document bootstrap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from residual_geometry.slow_semantic.protocol import PROTOCOL, load_basis, orthonormal_random_controls, save_json
from residual_geometry.slow_semantic.readout import (
    choose_baseline_c, choose_offset_l2, cross_entropy_rows, decision_logits,
    fit_offset_multinomial, percentile_interval, standardized_fit_apply,
)


def repeat_doc(values: np.ndarray, positions: int) -> np.ndarray:
    return np.repeat(values, positions, axis=0)


def token_classes(tokens: np.ndarray) -> np.ndarray:
    # Vocabulary-independent token-ID buckets are a deliberately weak current-token control;
    # local embedding PCA supplies the semantic local-context nuisance.
    return np.column_stack([
        tokens % 2, tokens % 3, tokens % 5, tokens % 7,
    ]).astype(np.float64)


def fit_representation(
    n_fit: np.ndarray, n_eval: np.ndarray, x_fit: np.ndarray, x_eval: np.ndarray,
    y_fit: np.ndarray, y_eval: np.ndarray, baseline: LogisticRegression,
    baseline_c: float, groups_fit: np.ndarray, l2: float | None,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    classes = len(np.unique(y_fit))
    base_eval_logits = decision_logits(baseline, n_eval, classes)
    base_losses = cross_entropy_rows(base_eval_logits, y_eval)
    xf, xe = standardized_fit_apply(x_fit, x_eval)
    if l2 is None:
        l2 = choose_offset_l2(
            n_fit, xf, y_fit, groups_fit, baseline_c,
            PROTOCOL["regularization_grid"], PROTOCOL["cv_folds"],
        )
    base_fit_logits = decision_logits(baseline, n_fit, classes)
    fit = fit_offset_multinomial(base_fit_logits, xf, y_fit, l2)
    if not fit.success:
        raise RuntimeError("offset multinomial optimizer did not converge")
    aug_losses = cross_entropy_rows(fit.logits(base_eval_logits, xe), y_eval)
    return base_losses - aug_losses, float(l2), fit.coefficients, float(aug_losses.mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run = Path(args.run_dir)
    result_path = run / "decisions" / "s2_profile.json"
    if result_path.exists() and not args.overwrite:
        print(result_path.read_text())
        return 0
    if not (run / "capture" / "manifest.json").exists():
        raise FileNotFoundError("capture manifest missing")
    frame = pd.read_parquet(run / "labels" / "document_labels.parquet")
    residual = np.asarray(np.load(run / "capture" / "residuals.npy", mmap_mode="r"), dtype=np.float64)
    local = np.asarray(np.load(run / "capture" / "local_embedding_means.npy", mmap_mode="r"), dtype=np.float64)
    tokens = np.load(run / "labels" / "token_ids.npy", mmap_mode="r")
    slow, _ = load_basis(run / "provenance" / "slow_basis_rank31.npz")
    n_docs, n_pos, d_model = residual.shape
    splits = frame["split"].to_numpy()
    fit_docs = np.flatnonzero(splits == "fit")
    val_docs = np.flatnonzero(splits == "validation")
    test_docs = np.flatnonzero(splits == "test")
    expected = list(PROTOCOL["split_counts"].values())
    if [len(fit_docs), len(val_docs), len(test_docs)] != expected:
        raise ValueError("split contract violated")

    fit_rows = (fit_docs[:, None] * n_pos + np.arange(n_pos)[None, :]).ravel()
    val_rows = (val_docs[:, None] * n_pos + np.arange(n_pos)[None, :]).ravel()
    test_rows = (test_docs[:, None] * n_pos + np.arange(n_pos)[None, :]).ravel()
    flat_h = residual.reshape(-1, d_model)
    mean = flat_h[fit_rows].mean(axis=0)
    centered_fit = flat_h[fit_rows] - mean
    pca = PCA(n_components=PROTOCOL["pca_parent_rank"], svd_solver="randomized", random_state=PROTOCOL["random_basis_seed"])
    pca.fit(centered_fit)
    pca_basis = pca.components_.T
    centered_all = flat_h - mean
    pca_coordinates = centered_all @ pca_basis
    slow_coordinates = centered_all @ slow
    np.savez_compressed(
        run / "supervised" / "residual_pca256.npz", directions=pca_basis.astype(np.float32),
        mean=mean.astype(np.float32), explained_variance=pca.explained_variance_.astype(np.float32),
    )
    random_controls: dict[int, list[np.ndarray]] = {}
    arrays: dict[str, np.ndarray] = {}
    for rank in PROTOCOL["slow_ranks"]:
        random_controls[rank] = orthonormal_random_controls(
            pca_basis, rank, PROTOCOL["random_bases_per_rank"], PROTOCOL["random_basis_seed"] + rank
        )
        for i, basis in enumerate(random_controls[rank]):
            arrays[f"rank{rank}_draw{i:03d}"] = basis.astype(np.float32)
    np.savez_compressed(run / "supervised" / "random_bases.npz", **arrays)
    save_json(run / "supervised" / "random_basis_manifest.json", {
        "parent": "fit-only residual PCA256", "counts": {str(k): len(v) for k, v in random_controls.items()},
        "seed": PROTOCOL["random_basis_seed"], "energy_matching": False,
        "energy_policy": "projected energy reported; deterministic top-PCA is primary matched comparator",
    })

    local_flat = local.reshape(-1, d_model)
    local_pca = PCA(n_components=PROTOCOL["local_embedding_pca_rank"], svd_solver="randomized", random_state=PROTOCOL["cv_seed"])
    local_pca.fit(local_flat[fit_rows])
    local_scores = local_pca.transform(local_flat)
    pos = np.tile(np.eye(n_pos, dtype=np.float64), (n_docs, 1))
    current_tokens = np.asarray(tokens[:, PROTOCOL["positions"]]).reshape(-1)
    nuisance_raw = np.column_stack([pos, token_classes(current_tokens), local_scores])
    n_scaler = StandardScaler().fit(nuisance_raw[fit_rows])
    nuisance = n_scaler.transform(nuisance_raw)

    target_names = ["topic", "register"]
    source_viable = bool(
        int((frame["source_template"] != "OTHER_LOW_SUPPORT").sum()) >= 100
        and frame["source_template"].nunique() >= 3
    )
    # Source-adjusted topic is a sensitivity endpoint, never a substitute for primary topic.
    source_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    source_encoder.fit(frame.loc[fit_docs, ["source_template"]])
    source_rows = repeat_doc(source_encoder.transform(frame[["source_template"]]), n_pos)

    doc_loss_rows: list[dict[str, object]] = []
    gain_rows: list[dict[str, object]] = []
    coefficient_arrays: dict[str, np.ndarray] = {}
    test_tasks: list[tuple[object, ...]] = []
    procedure: dict[str, object] = {
        "status": "FROZEN_BEFORE_TEST_READ", "targets": {},
        "source_template_nuisance": source_viable,
        "formatting_and_source_role": "descriptive controls; no fitted endpoint in minimum-compute S2",
    }
    for target in target_names:
        raw = frame[target].astype(str).to_numpy()
        fit_counts = pd.Series(raw[fit_docs]).value_counts()
        supported = sorted(fit_counts[fit_counts >= 30].index.tolist())
        if len(supported) < 2:
            continue
        keep_docs = np.flatnonzero(np.isin(raw, supported))
        label_map = {label: i for i, label in enumerate(supported)}
        y_doc = np.asarray([label_map.get(x, -1) for x in raw], dtype=np.int64)
        target_protocol: dict[str, object] = {"classes": supported, "ranks": {}}
        for rank in PROTOCOL["slow_ranks"]:
            allowed_fit_docs = np.intersect1d(fit_docs, keep_docs)
            allowed_val_docs = np.intersect1d(val_docs, keep_docs)
            allowed_test_docs = np.intersect1d(test_docs, keep_docs)
            tr = (allowed_fit_docs[:, None] * n_pos + np.arange(n_pos)[None, :]).ravel()
            va = (allowed_val_docs[:, None] * n_pos + np.arange(n_pos)[None, :]).ravel()
            te = (allowed_test_docs[:, None] * n_pos + np.arange(n_pos)[None, :]).ravel()
            y_rows = repeat_doc(y_doc[:, None], n_pos).ravel()
            n_used = nuisance
            if target == "topic" and source_viable:
                n_used = np.column_stack([nuisance, source_rows])
            baseline_c = choose_baseline_c(
                n_used[tr], y_rows[tr], np.repeat(allowed_fit_docs, n_pos),
                PROTOCOL["regularization_grid"], PROTOCOL["cv_folds"],
            )
            baseline = LogisticRegression(C=baseline_c, max_iter=600, solver="lbfgs", tol=1e-7)
            baseline.fit(n_used[tr], y_rows[tr])
            bases: list[tuple[str, np.ndarray]] = [
                (f"slow_{rank}", slow[:, :rank]), (f"pca_{rank}", pca_basis[:, :rank]),
            ] + [(f"random_{rank}_{i:03d}", b) for i, b in enumerate(random_controls[rank])]
            rank_protocol: dict[str, object] = {"baseline_c": baseline_c, "representation_l2": {}}
            random_shared_l2: float | None = None
            for family, basis in bases:
                if family.startswith("slow_"):
                    x = slow_coordinates[:, :rank]
                elif family == "pca_256":
                    x = pca_coordinates
                elif family.startswith("pca_"):
                    x = pca_coordinates[:, :rank]
                else:
                    x = pca_coordinates @ (pca_basis.T @ basis)
                # One frozen random-family regularizer is selected on draw 0 and then
                # applied to all other exchangeable controls. This prevents both a
                # 50x compute multiplier and per-null hyperparameter fishing.
                requested_l2 = random_shared_l2 if family.startswith("random_") else None
                delta_val, l2, coef, _ = fit_representation(
                    n_used[tr], n_used[va], x[tr], x[va], y_rows[tr], y_rows[va], baseline,
                    baseline_c, np.repeat(allowed_fit_docs, n_pos), requested_l2,
                )
                if family.endswith("_000") and family.startswith("random_"):
                    random_shared_l2 = l2
                rank_protocol["representation_l2"][family] = l2
                coefficient_arrays[f"{target}__{family}"] = coef.astype(np.float32)
                per_doc = delta_val.reshape(len(allowed_val_docs), n_pos).mean(axis=1)
                for document, value in zip(allowed_val_docs, per_doc):
                    doc_loss_rows.append({
                        "target": target, "rank": rank, "family": family, "split": "validation",
                        "document_id": str(frame.iloc[document]["document_id"]), "delta_h": float(value),
                    })
                gain_rows.append({
                    "target": target, "rank": rank, "family": family, "split": "validation",
                    "delta_h": float(per_doc.mean()), "documents": len(allowed_val_docs),
                    "projected_energy": float(np.mean(x[tr] ** 2)), "l2": l2,
                    "augmented_test_loss": np.nan,
                })
                test_tasks.append((target, rank, family, basis, n_used, tr, te, y_rows, baseline,
                                   baseline_c, allowed_fit_docs, allowed_test_docs, l2))
            target_protocol["ranks"][str(rank)] = rank_protocol
        procedure["targets"][target] = target_protocol
    save_json(run / "supervised" / "procedure_freeze.json", procedure)
    # This is the sole sealed-test access. Every target, basis, regularizer and
    # fitting rule above is now serialized and immutable.
    for task in test_tasks:
        (target, rank, family, basis, n_used, tr, te, y_rows, baseline,
         baseline_c, allowed_fit_docs, allowed_test_docs, l2) = task
        if family.startswith("slow_"):
            x = slow_coordinates[:, :rank]
        elif family == "pca_256":
            x = pca_coordinates
        elif family.startswith("pca_"):
            x = pca_coordinates[:, :rank]
        else:
            x = pca_coordinates @ (pca_basis.T @ basis)
        delta_test, _, _, test_aug_loss = fit_representation(
            n_used[tr], n_used[te], x[tr], x[te], y_rows[tr], y_rows[te], baseline,
            baseline_c, np.repeat(allowed_fit_docs, n_pos), l2,
        )
        per_doc = delta_test.reshape(len(allowed_test_docs), n_pos).mean(axis=1)
        for document, value in zip(allowed_test_docs, per_doc):
            doc_loss_rows.append({
                "target": target, "rank": rank, "family": family, "split": "test",
                "document_id": str(frame.iloc[document]["document_id"]), "delta_h": float(value),
            })
        gain_rows.append({
            "target": target, "rank": rank, "family": family, "split": "test",
            "delta_h": float(per_doc.mean()), "documents": len(allowed_test_docs),
            "projected_energy": float(np.mean(x[tr] ** 2)), "l2": l2,
            "augmented_test_loss": test_aug_loss,
        })
    np.savez_compressed(run / "supervised" / "coefficients.npz", **coefficient_arrays)
    docs_table = pd.DataFrame(doc_loss_rows)
    gains = pd.DataFrame(gain_rows)
    docs_table.to_parquet(run / "supervised" / "per_document_gain.parquet", index=False)
    gains.to_parquet(run / "supervised" / "semantic_gain_by_target_rank.parquet", index=False)

    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    bootstrap_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for (target, rank), group in docs_table[docs_table["split"] == "test"].groupby(["target", "rank"]):
        pivot = group.pivot(index="document_id", columns="family", values="delta_h").dropna()
        slow_key, pca_key = f"slow_{rank}", f"pca_{rank}"
        random_keys = sorted(c for c in pivot.columns if c.startswith(f"random_{rank}_"))
        values = pivot.to_numpy()
        columns = list(pivot.columns)
        idx = {c: columns.index(c) for c in columns}
        n = len(pivot)
        draws = rng.integers(0, n, size=(PROTOCOL["bootstrap_replicates"], n))
        means = values[draws].mean(axis=1)
        slow_boot = means[:, idx[slow_key]]
        pca_margin = slow_boot - means[:, idx[pca_key]]
        random_median = np.median(means[:, [idx[c] for c in random_keys]], axis=1)
        excess = slow_boot - random_median
        for b in range(len(draws)):
            bootstrap_rows.extend([
                {"target": target, "rank": int(rank), "replicate": b, "statistic": "slow_delta_h", "value": slow_boot[b]},
                {"target": target, "rank": int(rank), "replicate": b, "statistic": "slow_minus_random_median", "value": excess[b]},
                {"target": target, "rank": int(rank), "replicate": b, "statistic": "slow_minus_pca", "value": pca_margin[b]},
            ])
        for name, vals in (("slow_delta_h", slow_boot), ("slow_minus_random_median", excess), ("slow_minus_pca", pca_margin)):
            lo, hi = percentile_interval(vals)
            summary_rows.append({"target": target, "rank": int(rank), "statistic": name,
                                 "point": float(vals.mean()), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(bootstrap_rows).to_parquet(run / "supervised" / "document_bootstrap.parquet", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(run / "supervised" / "bootstrap_summary.parquet", index=False)

    def passes(target: str, rank: int, semantic: bool) -> bool:
        cell = summary[(summary.target == target) & (summary["rank"] == rank)].set_index("statistic")
        if len(cell) != 3:
            return False
        gain = cell.loc["slow_delta_h"]
        excess = cell.loc["slow_minus_random_median"]
        margin = cell.loc["slow_minus_pca"]
        pca_ok = margin["point"] >= PROTOCOL["pca_competitive_point_margin_nats"] and margin["ci_low"] >= PROTOCOL["pca_competitive_ci_lower_nats"]
        effect = gain["point"] >= PROTOCOL["primary_effect_health_nats"]
        return bool(gain["ci_low"] > 0 and excess["ci_low"] > 0 and effect and (pca_ok if semantic else True))

    semantic_positive = any(passes("topic", rank, True) for rank in PROTOCOL["slow_ranks"])
    structural_targets = ["register"]
    structural_positive = any(passes(target, rank, False) for target in structural_targets for rank in PROTOCOL["slow_ranks"])
    if semantic_positive and structural_positive:
        profile = "PROFILE_MIXED_DURABLE_STATE"
    elif semantic_positive:
        profile = "PROFILE_DURABLE_SEMANTIC"
    elif structural_positive:
        profile = "PROFILE_STRUCTURAL_DURABLE"
    else:
        profile = "PROFILE_GENERIC"
    s3_authorized = semantic_positive
    decision = {
        "status": "PASS", "profile": profile, "semantic_positive": semantic_positive,
        "structural_positive": structural_positive, "s3_authorized": s3_authorized,
        "s4_authorized": False,
        "stop": profile in {"PROFILE_GENERIC", "PROFILE_STRUCTURAL_DURABLE"},
        "test_access_count": 1,
        "note": "S4 remains unimplemented until a separate gate-aware micro-spec is scientifically justified.",
    }
    save_json(result_path, decision)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
