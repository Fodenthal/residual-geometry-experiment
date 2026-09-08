#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _common import artifact_metadata, load_config_and_store, require_residual_geometry_config
from residual_geometry.residuals.probes import load_many_probe_sets
from residual_geometry.subspace.fat_subspace import geometric_participation_ratio, participation_ratio, save_residual_basis_artifact
from residual_geometry.subspace.residual_geometry import deduplicate_ranked_probes, lifetime_excess_summary, orthonormal_basis
from residual_geometry.utils.io import ensure_dir, save_json, save_parquet


def _save_basis(
    path: str,
    ranked_dirs: np.ndarray,
    ranked_probes: pd.DataFrame,
    k: int,
    metadata: dict[str, object],
    dedup_threshold: float,
) -> None:
    selected = ranked_probes.head(k)
    save_residual_basis_artifact(
        path,
        basis=orthonormal_basis(ranked_dirs, k),
        source_probe_ids=selected["probe_id"].astype(str).to_numpy(),
        source_probe_family=selected["probe_family"].astype(str).to_numpy(),
        ranking_rule=np.array("validation_tau_within_desc_then_family_priority_tiebreak"),
        dedup_abs_cosine_threshold=np.array(dedup_threshold),
        **{key: np.array(str(value)) for key, value in metadata.items()},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build residual-first subspace pilot bases.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config, store = load_config_and_store(args.config)
    residual_cfg = require_residual_geometry_config(config)
    if residual_cfg.subspace is None:
        raise ValueError("residual_geometry.subspace is required for the subspace pilot, but not for B1.")
    metadata = artifact_metadata(config, "residual_geometry_04_residual_subspace_pilot")
    basis_dir = os.path.join(store.subspace_dir, "projection_bases")
    coverage_path = os.path.join(store.subspace_dir, "residual_probe_coverage.json")
    dimensionality_path = os.path.join(store.subspace_dir, "dimensionality_summary.json")
    ranked_path = os.path.join(store.subspace_dir, "ranked_residual_source_probes.parquet")
    if Path(dimensionality_path).exists() and not args.overwrite:
        return 0

    paths = [
        path
        for path in [
            store.time_lagged_residual_directions_path,
            store.residual_pca_directions_path,
            store.random_residual_directions_path,
        ]
        if Path(path).exists()
    ]
    probes = load_many_probe_sets(paths)
    timescales = pd.read_parquet(store.residual_probe_timescales_path)
    timescales = timescales[(timescales["split"] == "val") & (timescales["control"] == "real")].copy()
    if "gk_positive_validation_persistence" in timescales:
        timescales = timescales[
            (timescales["probe_family"] != "time_lagged") | timescales["gk_positive_validation_persistence"].astype(bool)
        ].copy()
    subspace_cfg = residual_cfg.subspace
    dedup = deduplicate_ranked_probes(probes, timescales, threshold=subspace_cfg.dedup_abs_cosine_threshold)
    if dedup.empty:
        raise ValueError("No eligible ranked residual probes remain after filtering and deduplication.")
    id_to_idx = {str(pid): idx for idx, pid in enumerate(probes.probe_ids.astype(str))}
    ranked_dirs = np.stack([probes.directions[id_to_idx[str(pid)]] for pid in dedup["probe_id"]])
    ensure_dir(basis_dir)
    dim = lifetime_excess_summary(dedup, subspace_cfg.k_values)
    k_star = int(dim.get("k_80pct_lifetime_excess", 0))
    if k_star > 0:
        top_kstar = dedup.head(k_star)
        baseline = float(dim["random_tau_within_median"])
        excess = np.maximum(top_kstar["tau_within"].to_numpy(dtype=np.float64) - baseline, 0.0)
        dim["top_kstar_geometric_participation_ratio"] = geometric_participation_ratio(ranked_dirs[:k_star])
        dim["top_kstar_slowness_participation_ratio"] = participation_ratio(excess)
    save_parquet(dedup, ranked_path)
    for k in subspace_cfg.k_values:
        if k <= len(ranked_dirs):
            _save_basis(
                os.path.join(basis_dir, f"residual_first_k{k}.npz"),
                ranked_dirs=ranked_dirs,
                ranked_probes=dedup,
                k=k,
                metadata=metadata,
                dedup_threshold=subspace_cfg.dedup_abs_cosine_threshold,
            )
    if k_star > 0:
        _save_basis(
            os.path.join(basis_dir, "residual_first_kstar.npz"),
            ranked_dirs=ranked_dirs,
            ranked_probes=dedup,
            k=k_star,
            metadata=metadata,
            dedup_threshold=subspace_cfg.dedup_abs_cosine_threshold,
        )
    coverage = {
        **metadata,
        "valid_probe_count": int(timescales["tau_valid_within"].sum()) if "tau_valid_within" in timescales else int(len(timescales)),
        "deduplicated_probe_count": int(len(dedup)),
        "dedup_abs_cosine_threshold": float(subspace_cfg.dedup_abs_cosine_threshold),
        "family_counts_after_dedup": dedup["probe_family"].value_counts().to_dict(),
    }
    save_json(coverage, coverage_path)
    dim["ranked_source_probes_path"] = ranked_path
    dim["exact_kstar_basis_path"] = os.path.join(basis_dir, "residual_first_kstar.npz") if k_star > 0 else None
    dim.update(metadata)
    save_json(dim, dimensionality_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
