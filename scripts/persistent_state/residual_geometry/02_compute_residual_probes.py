#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from _common import artifact_metadata, load_config_and_store, require_residual_geometry_config, residual_batches, split_contexts, token_matrix
from residual_geometry.residuals.pca import fit_residual_pca, sample_residual_positions
from residual_geometry.residuals.probes import concatenate_probe_sets, random_residual_probe_set, save_probe_set
from residual_geometry.residuals.projections import project_residuals, save_projection_chunk
from residual_geometry.residuals.provider import ResidualStreamProvider
from residual_geometry.residuals.time_lagged import (
    estimate_covariances_streaming,
    fit_time_lagged_from_covariances,
)
from residual_geometry.utils.io import load_json, save_json, save_npy
from residual_geometry.utils.logging import get_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute residual probe directions and projection chunks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--skip-time-lagged", action="store_true")
    args = parser.parse_args()

    logger = get_logger(__name__)
    config, store = load_config_and_store(args.config)
    residual_cfg = require_residual_geometry_config(config)
    if not Path(store.resolved_model_architecture_path).exists():
        raise FileNotFoundError(
            f"Missing {store.resolved_model_architecture_path}; run 00_validate_env.py without --skip-model-load first."
        )
    provider = ResidualStreamProvider(config)
    provider.load()

    direction_paths = [
        store.random_residual_directions_path,
        store.residual_pca_directions_path,
        store.time_lagged_residual_directions_path,
    ]
    projections_done = all(Path(store.residual_projection_chunk_path(split, 0)).exists() for split in ["train", "val", "test"])
    if all(Path(path).exists() for path in direction_paths if not ("pca" in path and args.skip_pca)) and projections_done and not args.overwrite:
        logger.info("Residual probe and projection artifacts already exist; use --overwrite to rebuild.")
        return 0

    random_probes = random_residual_probe_set(
        d_model=config.model.d_model,
        count=residual_cfg.random.directions,
        seed=residual_cfg.random.seed,
    )
    metadata = artifact_metadata(config, "residual_geometry_02_compute_residual_probes")
    npz_metadata = {key: np.array(str(value)) for key, value in metadata.items()}
    save_probe_set(
        random_probes,
        store.random_residual_directions_path,
        seed=np.array(residual_cfg.random.seed),
        source_population=np.array("unit_gaussian_d_model"),
        **npz_metadata,
    )
    probe_sets = [random_probes]

    train_df = split_contexts(store, "train")
    train_tokens = token_matrix(train_df)

    def train_residual_arrays():
        for _, residuals in residual_batches(provider, train_tokens, residual_cfg.projections.batch_size):
            yield residuals

    if not args.skip_pca:
        logger.info("Sampling residual positions and fitting randomized residual PCA directions.")
        total_train_positions = int(len(train_tokens) * residual_cfg.max_tokens)
        pca_sample_path = os.path.join(store.residual_probes_dir, "pca_residual_sample.memmap")
        residual_samples = sample_residual_positions(
            residual_batches=lambda: train_residual_arrays(),
            total_positions=total_train_positions,
            max_positions=residual_cfg.pca.max_positions,
            seed=residual_cfg.pca.seed,
            d_model=config.model.d_model,
            memmap_path=pca_sample_path,
        )
        pca_fit = fit_residual_pca(
            residual_samples,
            n_components=min(residual_cfg.pca.components, config.model.d_model),
            seed=residual_cfg.pca.seed,
        )
        pca_fit.diagnostics.update(
            {
                **metadata,
                "pca_solver": "sklearn.PCA(svd_solver=randomized)",
                "sample_seed": int(residual_cfg.pca.seed),
                "sample_source_split": "train",
                "sample_population_positions": total_train_positions,
                "sample_path": pca_sample_path,
            }
        )
        save_probe_set(
            pca_fit.probes,
            store.residual_pca_directions_path,
            mean=pca_fit.mean,
            explained_variance=pca_fit.eigenvalues,
            explained_variance_ratio=pca_fit.explained_variance_ratio,
            **npz_metadata,
        )
        save_json(pca_fit.diagnostics, store.residual_pca_fit_summary_path)
        save_json(pca_fit.diagnostics, store.residual_covariance_spectrum_path)
        save_npy(pca_fit.eigenvalues, store.residual_covariance_eigenvalues_path)
        probe_sets.append(pca_fit.probes)
        pca_reference = pca_fit.probes.directions
    else:
        pca_reference = None

    if not args.skip_time_lagged:
        logger.info("Fitting time-lagged residual directions.")
        mean, sigma0, sigma_lag, pair_count, token_count = estimate_covariances_streaming(
            residual_batches=lambda: train_residual_arrays(),
            lag_set=residual_cfg.time_lagged.lag_set,
            d_model=config.model.d_model,
        )
        np.savez_compressed(
            store.time_lagged_covariances_path,
            mean=mean.astype(np.float32),
            sigma0=sigma0.astype(np.float32),
            sigma_lag=sigma_lag.astype(np.float32),
            pair_count=np.array(pair_count, dtype=np.int64),
            token_count=np.array(token_count, dtype=np.int64),
            lag_set=np.asarray(residual_cfg.time_lagged.lag_set, dtype=np.int64),
            **npz_metadata,
        )
        time_fit = fit_time_lagged_from_covariances(
            mean=mean,
            sigma0=sigma0,
            sigma_lag=sigma_lag,
            pair_count=pair_count,
            lag_set=residual_cfg.time_lagged.lag_set,
            whitening_pcs=residual_cfg.time_lagged.whitening_pcs,
            output_directions=residual_cfg.time_lagged.output_directions,
            ridge_scale=residual_cfg.time_lagged.ridge_scale,
            max_ridge_scale=residual_cfg.time_lagged.max_ridge_scale,
            condition_threshold=residual_cfg.time_lagged.condition_number_threshold,
            pca_reference_directions=pca_reference,
        )
        time_fit.summary["train_token_count"] = int(token_count)
        time_fit.summary.update(metadata)
        time_fit.covariance_diagnostics.update(metadata)
        save_probe_set(
            time_fit.probes,
            store.time_lagged_residual_directions_path,
            mean=time_fit.mean,
            generalized_eigenvalues=time_fit.generalized_eigenvalues,
            **npz_metadata,
        )
        save_json(time_fit.summary, store.time_lagged_fit_summary_path)
        existing_spectrum = load_json(store.residual_covariance_spectrum_path) if Path(store.residual_covariance_spectrum_path).exists() else {}
        save_json({**existing_spectrum, **time_fit.covariance_diagnostics}, store.residual_covariance_spectrum_path)
        save_npy(time_fit.covariance_eigenvalues, store.residual_covariance_eigenvalues_path)
        probe_sets.append(time_fit.probes)

    all_probes = concatenate_probe_sets(probe_sets)
    logger.info("Computing projection chunks for %d probes.", all_probes.n_probes)
    for split in ["train", "val", "test"]:
        df = split_contexts(store, split)
        tokens = token_matrix(df)
        chunk_contexts: list[str] = []
        chunk_tokens: list[np.ndarray] = []
        chunk_proj: list[np.ndarray] = []
        chunk_index = 0
        for start, residuals in residual_batches(provider, tokens, residual_cfg.projections.batch_size):
            projections = project_residuals(residuals, all_probes)
            for local_i in range(projections.shape[0]):
                chunk_contexts.append(str(df.iloc[start + local_i]["context_id"]))
                chunk_tokens.append(tokens[start + local_i])
                chunk_proj.append(projections[local_i])
                if len(chunk_contexts) >= residual_cfg.projections.chunk_documents:
                    save_projection_chunk(
                        store.residual_projection_chunk_path(split, chunk_index),
                        context_ids=np.asarray(chunk_contexts, dtype=str),
                        tokens=np.stack(chunk_tokens),
                        probes=all_probes,
                        projections=np.stack(chunk_proj),
                        storage_dtype=residual_cfg.projections.storage_dtype,
                        **npz_metadata,
                    )
                    chunk_contexts, chunk_tokens, chunk_proj = [], [], []
                    chunk_index += 1
        if chunk_contexts:
            save_projection_chunk(
                store.residual_projection_chunk_path(split, chunk_index),
                context_ids=np.asarray(chunk_contexts, dtype=str),
                tokens=np.stack(chunk_tokens),
                probes=all_probes,
                projections=np.stack(chunk_proj),
                storage_dtype=residual_cfg.projections.storage_dtype,
                **npz_metadata,
            )
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
