#!/usr/bin/env python
"""Conditional S3: recover and qualify semantic geometry inside frozen S31."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from residual_geometry.slow_semantic.protocol import PROTOCOL, load_basis, save_json
from residual_geometry.slow_semantic.readout import cross_entropy_rows, decision_logits, fit_offset_multinomial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    decision = json.loads((run / "decisions" / "s2_profile.json").read_text())
    if not decision["s3_authorized"]:
        save_json(run / "decisions" / "s3_geometry.json", {
            "status": "NOT_AUTHORIZED", "reason": decision["profile"], "s4_authorized": False,
        })
        return 0

    frame = pd.read_parquet(run / "labels" / "document_labels.parquet")
    h = np.asarray(np.load(run / "capture" / "residuals.npy", mmap_mode="r"), dtype=np.float64)
    local = np.asarray(np.load(run / "capture" / "local_embedding_means.npy", mmap_mode="r"), dtype=np.float64)
    tokens = np.load(run / "labels" / "token_ids.npy", mmap_mode="r")
    slow, _ = load_basis(run / "provenance" / "slow_basis_rank31.npz")
    n_docs, n_pos, d = h.shape
    split = frame["split"].to_numpy()
    fit_docs = np.flatnonzero(split == "fit")
    val_docs = np.flatnonzero(split == "validation")
    fit_rows = (fit_docs[:, None] * n_pos + np.arange(n_pos)[None]).ravel()
    val_rows = (val_docs[:, None] * n_pos + np.arange(n_pos)[None]).ravel()
    mean = h.reshape(-1, d)[fit_rows].mean(axis=0)
    z = (h.reshape(-1, d) - mean) @ slow
    z_scaler = StandardScaler().fit(z[fit_rows])
    z = z_scaler.transform(z)
    local_flat = local.reshape(-1, d)
    lpca = PCA(n_components=PROTOCOL["local_embedding_pca_rank"], svd_solver="randomized", random_state=PROTOCOL["cv_seed"])
    lpca.fit(local_flat[fit_rows])
    local_scores = lpca.transform(local_flat)
    pos = np.tile(np.eye(n_pos), (n_docs, 1))
    current_tokens = np.asarray(tokens[:, PROTOCOL["positions"]]).reshape(-1)
    token_class = np.column_stack([current_tokens % 2, current_tokens % 3, current_tokens % 5, current_tokens % 7])
    nuisance_raw = np.column_stack([pos, token_class, local_scores])
    nuisance_scaler = StandardScaler().fit(nuisance_raw[fit_rows])
    nuisance = nuisance_scaler.transform(nuisance_raw)
    classes = sorted(frame.loc[fit_docs, "topic"].astype(str).value_counts().loc[lambda x: x >= 30].index)
    mapping = {x: i for i, x in enumerate(classes)}
    y_doc = np.asarray([mapping.get(str(x), -1) for x in frame["topic"]])
    y = np.repeat(y_doc, n_pos)
    keep_fit = fit_rows[y[fit_rows] >= 0]
    keep_val = val_rows[y[val_rows] >= 0]
    procedure = json.loads((run / "supervised" / "procedure_freeze.json").read_text())
    if procedure.get("source_template_nuisance", False):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(frame.loc[fit_docs, ["source_template"]])
        source = np.repeat(encoder.transform(frame[["source_template"]]), n_pos, axis=0)
        nuisance = np.column_stack([nuisance, source])
    rank_protocol = procedure["targets"]["topic"]["ranks"]["31"]
    baseline_c = float(rank_protocol["baseline_c"])
    l2 = float(rank_protocol["representation_l2"]["slow_31"])

    def semantic_fit(rows: np.ndarray, labels: np.ndarray | None = None) -> tuple[np.ndarray, LogisticRegression]:
        yy = y[rows] if labels is None else labels
        baseline = LogisticRegression(C=baseline_c, max_iter=600, solver="lbfgs", tol=1e-7).fit(nuisance[rows], yy)
        offset = decision_logits(baseline, nuisance[rows], len(classes))
        fit = fit_offset_multinomial(offset, z[rows], yy, l2)
        if not fit.success:
            raise RuntimeError("S3 optimizer failed")
        u, _, _ = np.linalg.svd(fit.coefficients, full_matrices=False)
        return u, baseline

    full_u, baseline = semantic_fit(keep_fit)
    base_fit = decision_logits(baseline, nuisance[keep_fit], len(classes))
    base_val = decision_logits(baseline, nuisance[keep_val], len(classes))
    base_loss = cross_entropy_rows(base_val, y[keep_val]).mean()
    retention_rows = []
    gains = {}
    for rank in [31, *PROTOCOL["semantic_ranks"]]:
        basis = np.eye(31) if rank == 31 else full_u[:, :rank]
        fit = fit_offset_multinomial(base_fit, z[keep_fit] @ basis, y[keep_fit], l2)
        gain = float(base_loss - cross_entropy_rows(fit.logits(base_val, z[keep_val] @ basis), y[keep_val]).mean())
        gains[rank] = gain
    for rank in PROTOCOL["semantic_ranks"]:
        retention_rows.append({"rank": rank, "validation_gain": gains[rank],
                               "retention": gains[rank] / max(gains[31], 1e-12)})
    retention = pd.DataFrame(retention_rows)
    retention.to_parquet(run / "semantic_geometry" / "rank_retention.parquet", index=False)
    eligible = retention[retention["retention"] >= PROTOCOL["semantic_retention"]]
    selected = int(eligible["rank"].min()) if len(eligible) else 8

    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"] + 901)
    shuffled_docs = fit_docs.copy(); rng.shuffle(shuffled_docs)
    half_a_docs, half_b_docs = np.array_split(shuffled_docs, 2)
    a_rows = (half_a_docs[:, None] * n_pos + np.arange(n_pos)[None]).ravel(); a_rows = a_rows[y[a_rows] >= 0]
    b_rows = (half_b_docs[:, None] * n_pos + np.arange(n_pos)[None]).ravel(); b_rows = b_rows[y[b_rows] >= 0]
    ua, _ = semantic_fit(a_rows); ub, _ = semantic_fit(b_rows)
    singular = np.linalg.svd(ua[:, :selected].T @ ub[:, :selected], compute_uv=False)
    observed = float(np.mean(singular ** 2))
    null = []
    for _ in range(PROTOCOL["semantic_shuffle_replicates"]):
        ya = y[a_rows].reshape(-1, n_pos).copy(); yb = y[b_rows].reshape(-1, n_pos).copy()
        rng.shuffle(ya, axis=0); rng.shuffle(yb, axis=0)
        usa, _ = semantic_fit(a_rows, ya.ravel()); usb, _ = semantic_fit(b_rows, yb.ravel())
        s = np.linalg.svd(usa[:, :selected].T @ usb[:, :selected], compute_uv=False)
        null.append(float(np.mean(s ** 2)))
    null = np.asarray(null)
    stable = bool(observed > np.quantile(null, 0.95))
    label = (
        "SEM0_NO_STABLE_INTERNAL_GEOMETRY" if not stable else
        "SEM3_STABLE_SEMANTIC_DIRECTION" if selected == 1 else
        "SEM2_COMPACT_STABLE_SEMANTIC_SUBSPACE" if selected <= 8 else
        "SEM1_STABLE_SEMANTIC_SUBSPACE"
    )
    np.savez_compressed(
        run / "semantic_geometry" / "semantic_basis.npz", residual_directions=(slow @ full_u[:, :selected]).astype(np.float32),
        within_slow_directions=full_u[:, :selected].astype(np.float32), selected_rank=np.array(selected),
    )
    pd.DataFrame({"replicate": np.arange(len(null)), "mean_squared_canonical_correlation": null}).to_parquet(
        run / "semantic_geometry" / "stability_null.parquet", index=False
    )
    save_json(run / "decisions" / "s3_geometry.json", {
        "status": label, "selected_rank": selected, "full_validation_gain": gains[31],
        "observed_mean_squared_canonical_correlation": observed,
        "shuffle_q95": float(np.quantile(null, 0.95)), "stable": stable,
        "s4_authorized": False,
        "reason_s4_not_authorized": "R1 requires a separate targeted causal micro-spec after descriptive review.",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
