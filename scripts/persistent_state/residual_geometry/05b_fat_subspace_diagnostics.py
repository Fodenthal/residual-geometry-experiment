#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from _common import artifact_metadata, load_config_and_store, require_residual_geometry_config, residual_batches, split_contexts, token_matrix
from residual_geometry.autocorr.estimators import AutocorrAccumulator, AutocorrResult, build_timescale_table, compute_document_autocorr, compute_document_autocorr_matrix
from residual_geometry.residuals.probes import ResidualProbeSet, concatenate_probe_sets, load_many_probe_sets, load_probe_set, save_probe_set
from residual_geometry.residuals.projections import project_residuals
from residual_geometry.residuals.provider import ResidualStreamProvider
from residual_geometry.subspace.fat_subspace import (
    ResidualBasisArtifact,
    autocorr_result_from_document_matrices,
    bootstrap_family_quantiles,
    coupled_nested_span_probe_set,
    load_residual_basis_artifact,
    pca_axis_loading_profile,
    pca_embedding_geometry,
    positive_profile_areas,
    probe_set_in_span,
    save_residual_basis_artifact,
)
from residual_geometry.subspace.residual_geometry import deduplicate_ranked_probes, orthonormal_basis
from residual_geometry.utils.io import ensure_dir, load_json, save_json, save_parquet
from residual_geometry.utils.logging import get_logger


@dataclass(frozen=True)
class ProfileBundle:
    profiles: np.ndarray
    valid_doc_counts: np.ndarray
    probe_ids: np.ndarray
    probe_family: np.ndarray
    split: np.ndarray


def _load_all_main_probes(store) -> ResidualProbeSet:
    paths = [
        path
        for path in [
            store.time_lagged_residual_directions_path,
            store.residual_pca_directions_path,
            store.random_residual_directions_path,
        ]
        if Path(path).exists()
    ]
    if len(paths) < 3:
        raise FileNotFoundError("Missing residual probe directions; run 02_compute_residual_probes.py first.")
    return load_many_probe_sets(paths)


def _validation_timescales(store) -> pd.DataFrame:
    if not Path(store.residual_probe_timescales_path).exists():
        raise FileNotFoundError("Missing residual-probe timescales; run 03_compute_residual_autocorr.py first.")
    table = pd.read_parquet(store.residual_probe_timescales_path)
    table = table[(table["split"] == "val") & (table["control"] == "real")].copy()
    if "tau_valid_within" in table:
        table = table[table["tau_valid_within"].astype(bool)].copy()
    if "gk_positive_validation_persistence" in table:
        table = table[
            (table["probe_family"] != "time_lagged") | table["gk_positive_validation_persistence"].astype(bool)
        ].copy()
    return table


def _ranked_source_probes(store, probes: ResidualProbeSet, timescales: pd.DataFrame, threshold: float) -> pd.DataFrame:
    path = os.path.join(store.subspace_dir, "ranked_residual_source_probes.parquet")
    if Path(path).exists():
        return pd.read_parquet(path)
    ranked = deduplicate_ranked_probes(probes, timescales, threshold=threshold)
    if ranked.empty:
        raise ValueError("No ranked source probes remain after filtering and deduplication.")
    save_parquet(ranked, path)
    return ranked


def _basis_from_ranked(
    probes: ResidualProbeSet,
    ranked: pd.DataFrame,
    k: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if k <= 0 or k > len(ranked):
        raise ValueError(f"Requested k={k} is invalid for {len(ranked)} ranked source probes.")
    selected = ranked.head(k).copy()
    id_to_idx = {str(probe_id): idx for idx, probe_id in enumerate(probes.probe_ids.astype(str))}
    try:
        directions = np.stack([probes.directions[id_to_idx[str(probe_id)]] for probe_id in selected["probe_id"]])
    except KeyError as exc:
        raise ValueError(f"Ranked source probe {exc.args[0]!r} is absent from saved residual direction files.") from exc
    return orthonormal_basis(directions, k), selected


def _canonical_basis(
    store,
    probes: ResidualProbeSet,
    timescales: pd.DataFrame,
    *,
    dedup_threshold: float,
    k_override: int | None,
    metadata: dict[str, object],
) -> tuple[ResidualBasisArtifact, int, bool]:
    dim_path = os.path.join(store.subspace_dir, "dimensionality_summary.json")
    if not Path(dim_path).exists():
        raise FileNotFoundError("Missing dimensionality summary; run 04_residual_subspace_pilot.py first.")
    dimensionality = load_json(dim_path)
    k_star = int(dimensionality.get("k_80pct_lifetime_excess", 0))
    k_fat = int(k_override if k_override is not None else k_star)
    if k_fat <= 0:
        raise ValueError("Fat-subspace diagnostics require positive k_80pct_lifetime_excess or --k-fat.")
    override = bool(k_override is not None and k_fat != k_star)
    if override:
        basis_path = os.path.join(store.fat_subspace_dir, f"canonical_basis_k{k_fat}.npz")
    else:
        basis_path = os.path.join(store.subspace_dir, "projection_bases", "residual_first_kstar.npz")
    if Path(basis_path).exists():
        try:
            artifact = load_residual_basis_artifact(basis_path)
            if artifact.k != k_fat:
                raise ValueError(f"Saved basis {basis_path} has k={artifact.k}, expected k={k_fat}.")
            return artifact, k_star, override
        except ValueError:
            if override:
                raise
    ranked = _ranked_source_probes(store, probes, timescales, threshold=dedup_threshold)
    basis, selected = _basis_from_ranked(probes, ranked, k_fat)
    save_residual_basis_artifact(
        basis_path,
        basis=basis,
        source_probe_ids=selected["probe_id"].astype(str).to_numpy(),
        source_probe_family=selected["probe_family"].astype(str).to_numpy(),
        ranking_rule=np.array("validation_tau_within_desc_then_family_priority_tiebreak"),
        dedup_abs_cosine_threshold=np.array(dedup_threshold),
        k_star=np.array(k_star),
        k_override=np.array(k_fat if override else -1),
        **{key: np.array(str(value)) for key, value in metadata.items()},
    )
    return load_residual_basis_artifact(basis_path), k_star, override


def _select_original_probes(all_probes: ResidualProbeSet, probe_ids: np.ndarray) -> ResidualProbeSet:
    id_to_idx = {str(probe_id): idx for idx, probe_id in enumerate(all_probes.probe_ids.astype(str))}
    try:
        selected = [id_to_idx[str(probe_id)] for probe_id in probe_ids]
    except KeyError as exc:
        raise ValueError(f"Source probe {exc.args[0]!r} is absent from saved residual directions.") from exc
    return ResidualProbeSet(
        directions=all_probes.directions[selected],
        probe_ids=all_probes.probe_ids[selected],
        probe_family=all_probes.probe_family[selected],
    )


def _renamed_probe_set(
    probes: ResidualProbeSet,
    *,
    family: str,
    id_prefix: str,
    direction_origin: str,
    span_dimension: int | None = None,
    seed: int | None = None,
) -> tuple[ResidualProbeSet, pd.DataFrame]:
    renamed = ResidualProbeSet(
        directions=probes.directions.astype(np.float32, copy=False),
        probe_ids=np.asarray([f"{id_prefix}_{idx:05d}" for idx in range(probes.n_probes)], dtype=object),
        probe_family=np.full(probes.n_probes, family, dtype=object),
    )
    manifest = pd.DataFrame(
        {
            "probe_id": renamed.probe_ids.astype(str),
            "diagnostic_family": renamed.probe_family.astype(str),
            "direction_origin": direction_origin,
            "original_probe_id": probes.probe_ids.astype(str),
            "original_probe_family": probes.probe_family.astype(str),
            "span_dimension": span_dimension,
            "seed": seed,
        }
    )
    return renamed, manifest


def _manifest_for_generated(
    probes: ResidualProbeSet,
    *,
    direction_origin: str,
    span_dimension_by_family: dict[str, int],
    seed_by_family: dict[str, int],
) -> pd.DataFrame:
    families = probes.probe_family.astype(str)
    return pd.DataFrame(
        {
            "probe_id": probes.probe_ids.astype(str),
            "diagnostic_family": families,
            "direction_origin": direction_origin,
            "original_probe_id": "",
            "original_probe_family": "",
            "span_dimension": [span_dimension_by_family[str(family)] for family in families],
            "seed": [seed_by_family[str(family)] for family in families],
        }
    )


def _load_or_create_locked_probe_set(path: str, expected: ResidualProbeSet) -> ResidualProbeSet:
    if Path(path).exists():
        probes = load_probe_set(path)
        if (
            probes.n_probes != expected.n_probes
            or not np.array_equal(probes.probe_ids.astype(str), expected.probe_ids.astype(str))
            or not np.array_equal(probes.probe_family.astype(str), expected.probe_family.astype(str))
            or not np.allclose(probes.directions, expected.directions, rtol=1e-6, atol=1e-7)
        ):
            raise ValueError(f"Saved locked directions in {path} do not match the requested configuration.")
        return probes
    save_probe_set(expected, path)
    return expected


def _nested_sweep_values(k: int, override: list[int] | None = None) -> list[int]:
    requested = override if override is not None else [value for value in [1, 2, 3, 5, 8, 13, 21, k] if value <= k]
    sweep = sorted(set(int(value) for value in requested))
    if not sweep or sweep[0] <= 0 or sweep[-1] > k:
        raise ValueError(f"Nested sweep values must be between 1 and k_fat={k}.")
    if k not in sweep:
        raise ValueError(f"Nested sweep must include the full candidate-pool endpoint k_fat={k}.")
    return sweep


def _load_or_create_locked_nested_probe_set(
    path: str,
    expected: ResidualProbeSet,
    *,
    coefficient_matrix: np.ndarray,
    sweep_k: list[int],
    candidate_pool_basis: np.ndarray,
) -> ResidualProbeSet:
    if Path(path).exists():
        probes = _load_or_create_locked_probe_set(path, expected)
        with np.load(path, allow_pickle=False) as data:
            if (
                "coefficient_matrix" not in data
                or "sweep_k" not in data
                or "candidate_pool_basis" not in data
                or not np.allclose(data["coefficient_matrix"], coefficient_matrix, rtol=1e-6, atol=1e-7)
                or not np.array_equal(data["sweep_k"], np.asarray(sweep_k, dtype=np.int64))
                or not np.allclose(data["candidate_pool_basis"], candidate_pool_basis, rtol=1e-6, atol=1e-7)
            ):
                raise ValueError(f"Saved locked nested directions in {path} do not match the requested configuration.")
        return probes
    save_probe_set(
        expected,
        path,
        coefficient_matrix=coefficient_matrix.astype(np.float32),
        sweep_k=np.asarray(sweep_k, dtype=np.int64),
        candidate_pool_basis=np.asarray(candidate_pool_basis, dtype=np.float32),
    )
    return expected


def _nested_random_in_span_directions(
    store,
    basis: ResidualBasisArtifact,
    *,
    count: int,
    seed: int,
    sweep_k: list[int],
) -> tuple[ResidualProbeSet, pd.DataFrame]:
    path = os.path.join(store.fat_subspace_dir, "nested_random_in_span_directions.npz")
    expected, coefficient_matrix = coupled_nested_span_probe_set(basis.basis, sweep_k, count=count, seed=seed)
    probes = _load_or_create_locked_nested_probe_set(
        path,
        expected,
        coefficient_matrix=coefficient_matrix,
        sweep_k=sweep_k,
        candidate_pool_basis=basis.basis,
    )
    endpoint_family = f"random_in_q{basis.k}"
    endpoint_mask = probes.probe_family.astype(str) == endpoint_family
    endpoint = ResidualProbeSet(
        directions=probes.directions[endpoint_mask],
        probe_ids=probes.probe_ids[endpoint_mask],
        probe_family=probes.probe_family[endpoint_mask],
    )
    _load_or_create_locked_probe_set(os.path.join(store.fat_subspace_dir, "random_in_span_directions.npz"), endpoint)
    return probes, _manifest_for_generated(
        probes,
        direction_origin="coupled_nested_random_in_span",
        span_dimension_by_family={f"random_in_q{k}": k for k in sweep_k},
        seed_by_family={f"random_in_q{k}": seed for k in sweep_k},
    )


def _lower_band_mixing_directions(
    store,
    source_probes: ResidualProbeSet,
    *,
    count: int,
    band_seed: int,
) -> tuple[ResidualProbeSet | None, pd.DataFrame]:
    path = os.path.join(store.fat_subspace_dir, "mixing_directions.npz")
    k = source_probes.n_probes
    groups: list[ResidualProbeSet] = []
    span_dimensions: dict[str, int] = {}
    seeds: dict[str, int] = {}
    band_rng = np.random.default_rng(band_seed)
    for start in [9, 17]:
        if start > k:
            continue
        directions = source_probes.directions[start - 1 :]
        family = f"mix_band_{start}_{k}"
        child_seed = int(band_rng.integers(0, 2**31 - 1))
        groups.append(
            probe_set_in_span(
                orthonormal_basis(directions, len(directions)),
                count=count,
                seed=child_seed,
                family=family,
                id_prefix=family,
            )
        )
        span_dimensions[family] = len(directions)
        seeds[family] = child_seed
    if not groups:
        return None, pd.DataFrame()
    expected = concatenate_probe_sets(groups)
    probes = _load_or_create_locked_probe_set(path, expected)
    return probes, _manifest_for_generated(
        probes,
        direction_origin="source_probe_mixing",
        span_dimension_by_family=span_dimensions,
        seed_by_family=seeds,
    )


def _direct_pca_span_directions(
    store,
    pca_probes: ResidualProbeSet,
    *,
    count: int,
    seed: int,
) -> tuple[ResidualProbeSet, pd.DataFrame]:
    path = os.path.join(store.fat_subspace_dir, "direct_random_pca_span_directions.npz")
    groups: list[ResidualProbeSet] = []
    span_dimensions: dict[str, int] = {}
    seeds: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    for p in [31, 128, 256]:
        if p > pca_probes.n_probes:
            continue
        family = f"random_in_pca{p}"
        child_seed = int(rng.integers(0, 2**31 - 1))
        groups.append(
            probe_set_in_span(
                orthonormal_basis(pca_probes.directions[:p], p),
                count=count,
                seed=child_seed,
                family=family,
                id_prefix=family,
            )
        )
        span_dimensions[family] = p
        seeds[family] = child_seed
    if not groups:
        raise ValueError("No requested direct PCA-span control is available.")
    expected = concatenate_probe_sets(groups)
    probes = _load_or_create_locked_probe_set(path, expected)
    return probes, _manifest_for_generated(
        probes,
        direction_origin="direct_random_pca_span",
        span_dimension_by_family=span_dimensions,
        seed_by_family=seeds,
    )


def _evaluation_probes(
    store,
    all_probes: ResidualProbeSet,
    basis: ResidualBasisArtifact,
    *,
    random_in_span_count: int,
    mixing_count: int,
    direct_pca_count: int,
    random_in_span_seed: int,
    mixing_band_seed: int,
    direct_pca_seed: int,
    nested_sweep_k: list[int],
) -> tuple[ResidualProbeSet, pd.DataFrame, ResidualProbeSet]:
    source_original = _select_original_probes(all_probes, basis.source_probe_ids)
    source, source_manifest = _renamed_probe_set(
        source_original,
        family="source_probe",
        id_prefix="source_probe",
        direction_origin="headline_ranked_source_probe",
        span_dimension=basis.k,
    )
    ambient_original = ResidualProbeSet(
        directions=all_probes.directions[all_probes.probe_family.astype(str) == "random"],
        probe_ids=all_probes.probe_ids[all_probes.probe_family.astype(str) == "random"],
        probe_family=all_probes.probe_family[all_probes.probe_family.astype(str) == "random"],
    )
    ambient, ambient_manifest = _renamed_probe_set(
        ambient_original,
        family="ambient_random",
        id_prefix="ambient_random",
        direction_origin="main_random_residual_probe",
    )
    pca_original = ResidualProbeSet(
        directions=all_probes.directions[all_probes.probe_family.astype(str) == "pca"],
        probe_ids=all_probes.probe_ids[all_probes.probe_family.astype(str) == "pca"],
        probe_family=all_probes.probe_family[all_probes.probe_family.astype(str) == "pca"],
    )
    pca, pca_manifest = _renamed_probe_set(
        pca_original,
        family="pca_axis",
        id_prefix="pca_axis",
        direction_origin="main_residual_pca_axis",
    )
    nested_random_q, nested_random_q_manifest = _nested_random_in_span_directions(
        store,
        basis,
        count=random_in_span_count,
        seed=random_in_span_seed,
        sweep_k=nested_sweep_k,
    )
    mixing, mixing_manifest = _lower_band_mixing_directions(store, source_original, count=mixing_count, band_seed=mixing_band_seed)
    pca_span, pca_span_manifest = _direct_pca_span_directions(store, pca_original, count=direct_pca_count, seed=direct_pca_seed)
    generated = [nested_random_q] + ([mixing] if mixing is not None else []) + [pca_span, source, ambient, pca]
    probes = concatenate_probe_sets(generated)
    manifest = pd.concat(
        [nested_random_q_manifest, mixing_manifest, pca_span_manifest, source_manifest, ambient_manifest, pca_manifest],
        ignore_index=True,
    )
    return probes, manifest, source_original


def _profile_bundle(result: AutocorrResult, probes: ResidualProbeSet, split: str) -> ProfileBundle:
    return ProfileBundle(
        profiles=result.profiles.astype(np.float32),
        valid_doc_counts=result.valid_doc_counts.astype(np.int64),
        probe_ids=probes.probe_ids.astype(str),
        probe_family=probes.probe_family.astype(str),
        split=np.full(probes.n_probes, split, dtype=str),
    )


def _timescale_table(
    result: AutocorrResult,
    probes: ResidualProbeSet,
    manifest: pd.DataFrame,
    residual_cfg,
    *,
    split: str,
    metadata: dict[str, object],
) -> pd.DataFrame:
    table = build_timescale_table(
        feature_indices=np.arange(probes.n_probes, dtype=np.int64),
        estimator_results={"within": result},
        max_lag=residual_cfg.max_lag,
        min_valid_docs=residual_cfg.valid_doc_threshold,
        min_valid_lag_fraction=residual_cfg.min_valid_lag_fraction,
        smoothing_width=residual_cfg.smoothing_width,
    )
    table["probe_id"] = probes.probe_ids.astype(str)
    table["diagnostic_family"] = probes.probe_family.astype(str)
    table["split"] = split
    table["positive_profile_area"] = positive_profile_areas(result.profiles)
    table = table.merge(manifest, on=["probe_id", "diagnostic_family"], how="left", validate="one_to_one")
    for key, value in metadata.items():
        table[key] = value
    return table


def _evaluate_split(
    provider: ResidualStreamProvider,
    tokens: np.ndarray,
    probes: ResidualProbeSet,
    manifest: pd.DataFrame,
    residual_cfg,
    *,
    split: str,
    metadata: dict[str, object],
    direction_chunk: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_direction_chunk: int,
    skip_bootstrap: bool,
    bootstrap_temp_dir: str | None,
    profile_batches: int | None,
) -> tuple[pd.DataFrame, ProfileBundle, pd.DataFrame] | None:
    logger = get_logger(__name__)
    chunks = [(start, min(start + direction_chunk, probes.n_probes)) for start in range(0, probes.n_probes, direction_chunk)]
    accumulators = [AutocorrAccumulator(n_features=stop - start, max_lag=residual_cfg.max_lag) for start, stop in chunks]
    docs = len(tokens)
    lags = residual_cfg.max_lag + 1
    temp_context = tempfile.TemporaryDirectory(prefix=f"fat_subspace_{split}_", dir=bootstrap_temp_dir) if not skip_bootstrap and profile_batches is None else None
    try:
        corr_memmap = None
        valid_memmap = None
        if temp_context is not None:
            corr_memmap = np.memmap(
                os.path.join(temp_context.name, "corr.float32.memmap"),
                mode="w+",
                dtype=np.float32,
                shape=(docs, probes.n_probes, lags),
            )
            valid_memmap = np.memmap(
                os.path.join(temp_context.name, "valid.bool.memmap"),
                mode="w+",
                dtype=np.bool_,
                shape=(docs, probes.n_probes, lags),
            )
        iterator = iter(residual_batches(provider, tokens, residual_cfg.projections.batch_size))
        batch_index = 0
        while True:
            try:
                forward_start = time.perf_counter()
                doc_start, residuals = next(iterator)
                forward_seconds = time.perf_counter() - forward_start
            except StopIteration:
                break
            cpu_start = time.perf_counter()
            doc_stop = doc_start + len(residuals)
            for chunk_index, (direction_start, direction_stop) in enumerate(chunks):
                selected = ResidualProbeSet(
                    directions=probes.directions[direction_start:direction_stop],
                    probe_ids=probes.probe_ids[direction_start:direction_stop],
                    probe_family=probes.probe_family[direction_start:direction_stop],
                )
                projections = project_residuals(residuals, selected)
                if corr_memmap is not None and valid_memmap is not None:
                    corr, valid = compute_document_autocorr_matrix(
                        projections,
                        max_lag=residual_cfg.max_lag,
                        estimator="within",
                    )
                    corr_memmap[doc_start:doc_stop, direction_start:direction_stop] = corr
                    valid_memmap[doc_start:doc_stop, direction_start:direction_stop] = valid
                    accumulators[chunk_index].update(autocorr_result_from_document_matrices(corr, valid))
                else:
                    accumulators[chunk_index].update(
                        compute_document_autocorr(projections, max_lag=residual_cfg.max_lag, estimator="within")
                    )
            cpu_seconds = time.perf_counter() - cpu_start
            logger.info(
                "fat-subspace split=%s batch=%d docs=%d:%d forward_transfer=%.1fs projection_autocorr=%.1fs",
                split,
                batch_index,
                doc_start,
                doc_stop,
                forward_seconds,
                cpu_seconds,
            )
            batch_index += 1
            if profile_batches is not None and batch_index >= profile_batches:
                logger.info("Profile-only run complete after %d batches; no artifacts written.", batch_index)
                return None
        finalized = [accumulator.finalize() for accumulator in accumulators]
        result = AutocorrResult(
            profiles=np.concatenate([item.profiles for item in finalized], axis=0),
            valid_doc_counts=np.concatenate([item.valid_doc_counts for item in finalized], axis=0),
        )
        timescales = _timescale_table(result, probes, manifest, residual_cfg, split=split, metadata=metadata)
        bundle = _profile_bundle(result, probes, split)
        bootstrap = pd.DataFrame()
        if corr_memmap is not None and valid_memmap is not None:
            corr_memmap.flush()
            valid_memmap.flush()
            bootstrap = bootstrap_family_quantiles(
                corr_memmap,
                valid_memmap,
                probes.probe_family,
                split=split,
                max_lag=residual_cfg.max_lag,
                min_valid_docs=residual_cfg.valid_doc_threshold,
                min_valid_lag_fraction=residual_cfg.min_valid_lag_fraction,
                smoothing_width=residual_cfg.smoothing_width,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
                direction_chunk=bootstrap_direction_chunk,
            )
        return timescales, bundle, bootstrap
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def _load_profile_bundle(path: str) -> ProfileBundle | None:
    if not Path(path).exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return ProfileBundle(
            profiles=data["profiles"].astype(np.float32),
            valid_doc_counts=data["valid_doc_counts"].astype(np.int64),
            probe_ids=data["probe_ids"].astype(str),
            probe_family=data["probe_family"].astype(str),
            split=data["split"].astype(str),
        )


def _merge_profiles(existing: ProfileBundle | None, new: list[ProfileBundle]) -> ProfileBundle:
    bundles = ([existing] if existing is not None else []) + new
    profiles = np.concatenate([bundle.profiles for bundle in bundles], axis=0)
    valid_counts = np.concatenate([bundle.valid_doc_counts for bundle in bundles], axis=0)
    probe_ids = np.concatenate([bundle.probe_ids for bundle in bundles], axis=0)
    families = np.concatenate([bundle.probe_family for bundle in bundles], axis=0)
    splits = np.concatenate([bundle.split for bundle in bundles], axis=0)
    keys = np.asarray([f"{split}\0{family}\0{probe_id}" for split, family, probe_id in zip(splits, families, probe_ids)])
    keep = ~pd.Series(keys).duplicated(keep="last").to_numpy()
    return ProfileBundle(
        profiles=profiles[keep],
        valid_doc_counts=valid_counts[keep],
        probe_ids=probe_ids[keep],
        probe_family=families[keep],
        split=splits[keep],
    )


def _save_profile_bundle(path: str, bundle: ProfileBundle) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.savez_compressed(
        path,
        profiles=bundle.profiles.astype(np.float32),
        valid_doc_counts=bundle.valid_doc_counts.astype(np.int64),
        probe_ids=bundle.probe_ids.astype(str),
        probe_family=bundle.probe_family.astype(str),
        split=bundle.split.astype(str),
    )


def _mean_profile_summary(bundle: ProfileBundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in sorted(set(bundle.split.astype(str))):
        for family in sorted(set(bundle.probe_family[bundle.split.astype(str) == split].astype(str))):
            selected = (bundle.split.astype(str) == split) & (bundle.probe_family.astype(str) == family)
            profiles = bundle.profiles[selected]
            valid_counts = bundle.valid_doc_counts[selected]
            for lag in range(profiles.shape[1]):
                values = profiles[:, lag]
                finite = values[np.isfinite(values)]
                rows.append(
                    {
                        "split": split,
                        "diagnostic_family": family,
                        "lag": int(lag),
                        "direction_count": int(len(profiles)),
                        "mean_autocorrelation": float(np.mean(finite)) if len(finite) else float("nan"),
                        "q25_autocorrelation": float(np.quantile(finite, 0.25)) if len(finite) else float("nan"),
                        "q50_autocorrelation": float(np.quantile(finite, 0.5)) if len(finite) else float("nan"),
                        "q75_autocorrelation": float(np.quantile(finite, 0.75)) if len(finite) else float("nan"),
                        "mean_valid_doc_count": float(np.mean(valid_counts[:, lag])),
                    }
                )
    return pd.DataFrame(rows)


def _merge_split_table(existing_path: str, new_tables: list[pd.DataFrame], splits: list[str]) -> pd.DataFrame:
    existing = pd.read_parquet(existing_path) if Path(existing_path).exists() else pd.DataFrame()
    if not existing.empty and "split" in existing:
        existing = existing[~existing["split"].astype(str).isin(splits)].copy()
    tables = ([existing] if not existing.empty else []) + [table for table in new_tables if not table.empty]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _source_probe_train_rows(
    store,
    source_original: ResidualProbeSet,
) -> pd.DataFrame:
    main = pd.read_parquet(store.residual_probe_timescales_path)
    main = main[(main["split"] == "train") & (main["control"] == "real")].copy()
    selected_ids = set(source_original.probe_ids.astype(str))
    main = main[main["probe_id"].astype(str).isin(selected_ids)].copy()
    if set(main["probe_id"].astype(str)) != selected_ids:
        return main.iloc[0:0].copy()
    areas: dict[str, float] = {}
    for family, group in main.groupby("probe_family"):
        path = store.residual_autocorr_profiles_path("train", str(family), "within")
        if not Path(path).exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            ids = data["probe_ids"].astype(str)
            profile_area = positive_profile_areas(data["profiles"])
        areas.update({probe_id: float(area) for probe_id, area in zip(ids, profile_area)})
    rank = {probe_id: idx for idx, probe_id in enumerate(source_original.probe_ids.astype(str))}
    main["diagnostic_family"] = "source_probe"
    main["direction_origin"] = "headline_ranked_source_probe"
    main["original_probe_id"] = main["probe_id"].astype(str)
    main["original_probe_family"] = main["probe_family"].astype(str)
    main["source_rank"] = main["probe_id"].astype(str).map(rank)
    main["positive_profile_area"] = main["probe_id"].astype(str).map(areas)
    return main


def _source_probe_train_profile_bundle(store, source_original: ResidualProbeSet) -> ProfileBundle | None:
    by_probe_id: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    found_profile_artifact = False
    for family in sorted(set(source_original.probe_family.astype(str))):
        path = store.residual_autocorr_profiles_path("train", family, "within")
        if not Path(path).exists():
            continue
        found_profile_artifact = True
        with np.load(path, allow_pickle=False) as data:
            ids = data["probe_ids"].astype(str)
            profiles = data["profiles"].astype(np.float32)
            valid_doc_counts = data["valid_doc_counts"].astype(np.int64)
        by_probe_id.update(
            {
                str(probe_id): (profile, valid_counts)
                for probe_id, profile, valid_counts in zip(ids, profiles, valid_doc_counts)
            }
        )
    if not found_profile_artifact:
        return None
    ordered_ids = source_original.probe_ids.astype(str)
    missing = [probe_id for probe_id in ordered_ids if probe_id not in by_probe_id]
    if missing:
        return None
    return ProfileBundle(
        profiles=np.stack([by_probe_id[probe_id][0] for probe_id in ordered_ids]),
        valid_doc_counts=np.stack([by_probe_id[probe_id][1] for probe_id in ordered_ids]),
        probe_ids=np.asarray([f"source_probe_{index:05d}" for index in range(len(ordered_ids))]),
        probe_family=np.full(len(ordered_ids), "source_probe"),
        split=np.full(len(ordered_ids), "train"),
    )


def _source_stability_table(
    direction_timescales: pd.DataFrame,
    train_rows: pd.DataFrame,
    source_original: ResidualProbeSet,
) -> pd.DataFrame:
    source_eval = direction_timescales[direction_timescales["diagnostic_family"] == "source_probe"].copy()
    diagnostic_to_original = {
        f"source_probe_{index:05d}": str(probe_id)
        for index, probe_id in enumerate(source_original.probe_ids.astype(str))
    }
    original_to_rank = {probe_id: rank for rank, probe_id in enumerate(source_original.probe_ids.astype(str))}
    source_eval["source_rank"] = source_eval["probe_id"].astype(str).map(
        {probe_id: rank for rank, probe_id in enumerate(diagnostic_to_original)}
    )
    source_eval["original_probe_id"] = source_eval["probe_id"].astype(str).map(diagnostic_to_original)
    train_rows = train_rows.copy()
    if "original_probe_id" in train_rows:
        train_rows["source_rank"] = train_rows["original_probe_id"].astype(str).map(original_to_rank)
    columns = sorted(set(source_eval.columns) | set(train_rows.columns))
    return pd.concat([train_rows.reindex(columns=columns), source_eval.reindex(columns=columns)], ignore_index=True)


def _metric_quantiles(timescales: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid = timescales[timescales["tau_valid_within"].astype(bool)].copy()
    for (split, family), group in valid.groupby(["split", "diagnostic_family"]):
        row: dict[str, object] = {
            "split": str(split),
            "diagnostic_family": str(family),
            "direction_count": int(len(group)),
            "right_censoring_rate": float(group["right_censored_within"].mean()),
        }
        for metric in ["tau_within", "positive_profile_area"]:
            for quantile in [0.5, 0.75, 0.9, 0.95]:
                row[f"q{int(quantile * 100)}_{metric}"] = float(group[metric].quantile(quantile))
        rows.append(row)
    return rows


def _random_in_q_dimension(family: str) -> int | None:
    prefix = "random_in_q"
    return int(family[len(prefix) :]) if family.startswith(prefix) else None


def _ambient_null_bands(timescales: pd.DataFrame) -> dict[str, float]:
    valid = timescales[timescales["tau_valid_within"].astype(bool)].copy()
    ambient = valid[valid["diagnostic_family"] == "ambient_random"]
    return {
        str(split): float(group["tau_within"].quantile(0.95))
        for split, group in ambient.groupby("split")
    }


def _nested_dimension_sweep(timescales: pd.DataFrame, ambient_null_bands: dict[str, float]) -> pd.DataFrame:
    valid = timescales[timescales["tau_valid_within"].astype(bool)].copy()
    valid["nested_k"] = valid["diagnostic_family"].astype(str).map(_random_in_q_dimension)
    nested = valid[valid["nested_k"].notna()].copy()
    rows: list[dict[str, object]] = []
    for (split, nested_k), group in nested.groupby(["split", "nested_k"]):
        row: dict[str, object] = {
            "split": str(split),
            "nested_k": int(nested_k),
            "diagnostic_family": f"random_in_q{int(nested_k)}",
            "direction_count": int(len(group)),
            "right_censoring_rate": float(group["right_censored_within"].mean()),
            "ambient_random_q95_tau_within": ambient_null_bands.get(str(split)),
        }
        for metric in ["tau_within", "positive_profile_area"]:
            for quantile in [0.5, 0.75, 0.9, 0.95]:
                row[f"q{int(quantile * 100)}_{metric}"] = float(group[metric].quantile(quantile))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "nested_k"]).reset_index(drop=True)


def _source_probe_rank_curve(source_stability: pd.DataFrame, ambient_null_bands: dict[str, float]) -> pd.DataFrame:
    curve = source_stability.copy()
    curve["source_rank"] = curve["source_rank"].astype(np.int64) + 1
    curve["ambient_random_q95_tau_within"] = curve["split"].astype(str).map(ambient_null_bands)
    curve["enters_ambient_null_band"] = (
        curve["tau_valid_within"].astype(bool)
        & curve["ambient_random_q95_tau_within"].notna()
        & (curve["tau_within"] <= curve["ambient_random_q95_tau_within"])
    )
    return curve.sort_values(["split", "source_rank"]).reset_index(drop=True)


def _first_source_rank_in_null_band(source_probe_rank_curve: pd.DataFrame) -> dict[str, int | None]:
    first: dict[str, int | None] = {}
    for split, group in source_probe_rank_curve.groupby("split"):
        entering = group[group["enters_ambient_null_band"].astype(bool)]
        first[str(split)] = int(entering["source_rank"].min()) if not entering.empty else None
    return first


def _family_quantile(rows: list[dict[str, object]], split: str, family: str, key: str) -> float | None:
    for row in rows:
        if row["split"] == split and row["diagnostic_family"] == family:
            return float(row[key])
    return None


def _classification(
    quantiles: list[dict[str, object]],
    *,
    split: str,
    k_fat: int,
    source_stability: pd.DataFrame,
) -> tuple[list[str], dict[str, object]]:
    span_q50 = _family_quantile(quantiles, split, f"random_in_q{k_fat}", "q50_tau_within")
    ambient_q90 = _family_quantile(quantiles, split, "ambient_random", "q90_tau_within")
    source_q50 = _family_quantile(quantiles, split, "source_probe", "q50_tau_within")
    pca_controls = {
        str(row["diagnostic_family"]): float(row["q50_tau_within"])
        for row in quantiles
        if row["split"] == split and str(row["diagnostic_family"]).startswith("random_in_pca")
    }
    pca_control_max_q50 = max(pca_controls.values()) if pca_controls else None
    labels: list[str] = []
    if span_q50 is not None and ambient_q90 is not None:
        if span_q50 > ambient_q90:
            labels.append("fat")
            if source_q50 is not None and span_q50 < source_q50:
                labels.append("hierarchical")
        else:
            labels.append("thin_frame_dependent")
    if pca_control_max_q50 is not None and span_q50 is not None and ambient_q90 is not None:
        if pca_control_max_q50 > ambient_q90 and pca_control_max_q50 >= 0.8 * span_q50:
            labels.append("generic_pca_span_slow")
    valid_source = source_stability[source_stability["tau_valid_within"].astype(bool)]
    train_source = valid_source[valid_source["split"] == "train"]
    test_source = valid_source[valid_source["split"] == "test"]
    train_q50 = float(train_source["tau_within"].median()) if not train_source.empty else None
    test_q50 = float(test_source["tau_within"].median()) if not test_source.empty else None
    if ambient_q90 is not None and train_q50 is not None and test_q50 is not None:
        if train_q50 > ambient_q90 and test_q50 <= ambient_q90:
            labels.append("shrinkage_dominated")
    return labels or ["unclassified"], {
        "classification_split": split,
        "random_in_span_q50_tau": span_q50,
        "ambient_random_q90_tau": ambient_q90,
        "source_probe_q50_tau": source_q50,
        "direct_pca_span_q50_tau": pca_controls,
        "direct_pca_span_max_q50_tau": pca_control_max_q50,
        "source_probe_train_q50_tau": train_q50,
        "source_probe_test_q50_tau": test_q50,
        "fat_rule": "random_in_span_q50_tau > ambient_random_q90_tau",
        "hierarchical_rule": "fat and random_in_span_q50_tau < source_probe_q50_tau",
        "generic_pca_span_slow_rule": "max direct_pca_span_q50_tau > ambient_random_q90_tau and >= 0.8 * random_in_span_q50_tau",
        "headline_span_specific_vs_pca_controls": bool(
            span_q50 is not None and pca_control_max_q50 is not None and span_q50 > pca_control_max_q50
        ),
        "shrinkage_dominated_rule": "source_probe_train_q50_tau > ambient_random_q90_tau and source_probe_test_q50_tau <= ambient_random_q90_tau",
    }


def _source_stability_summary(source_stability: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    valid = source_stability[source_stability["tau_valid_within"].astype(bool)].copy()
    for split, group in valid.groupby("split"):
        summary[str(split)] = {
            "direction_count": int(len(group)),
            "q50_tau_within": float(group["tau_within"].quantile(0.5)),
            "q75_tau_within": float(group["tau_within"].quantile(0.75)),
            "q90_tau_within": float(group["tau_within"].quantile(0.9)),
            "q95_tau_within": float(group["tau_within"].quantile(0.95)),
            "right_censoring_rate": float(group["right_censored_within"].mean()),
        }
    pivot = valid.pivot_table(index="original_probe_id", columns="split", values="tau_within", aggfunc="first")
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        if left in pivot and right in pivot:
            summary[f"spearman_{left}_{right}"] = float(pivot[[left, right]].corr(method="spearman").iloc[0, 1])
    if "train" in summary:
        train_q50 = float(summary["train"]["q50_tau_within"])
        for split in ["val", "test"]:
            if split in summary:
                summary[f"q50_shrinkage_ratio_{split}_over_train"] = float(summary[split]["q50_tau_within"] / max(train_q50, 1e-12))
    return summary


def _projection_test_already_inspected(store) -> bool:
    path = os.path.join(store.subspace_dir, "projection_collapse_summary.json")
    if not Path(path).exists():
        return False
    return "test" in set(str(split) for split in load_json(path).get("splits", []))


def _test_previously_inspected_before_fat_subspace_lock(store) -> bool:
    if Path(store.fat_subspace_summary_path).exists():
        previous = load_json(store.fat_subspace_summary_path)
        if "test_previously_inspected_before_fat_subspace_lock" in previous:
            return bool(previous["test_previously_inspected_before_fat_subspace_lock"])
    return _projection_test_already_inspected(store)


def _make_plots(
    timescales: pd.DataFrame,
    geometry: pd.DataFrame,
    profile_summary: pd.DataFrame,
    nested_sweep: pd.DataFrame,
    source_probe_rank_curve: pd.DataFrame,
    out_dir: str,
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"plot_status": "matplotlib_unavailable"}
    ensure_dir(out_dir)
    outputs: dict[str, str] = {}
    latest_split = "test" if "test" in set(timescales["split"].astype(str)) else "val"
    table = timescales[(timescales["split"] == latest_split) & timescales["tau_valid_within"].astype(bool)]
    nested_families = {
        str(family): _random_in_q_dimension(str(family))
        for family in table["diagnostic_family"].unique()
        if _random_in_q_dimension(str(family)) is not None
    }
    endpoint_family = max(nested_families, key=lambda family: nested_families[family]) if nested_families else ""
    primary = [
        family
        for family in [endpoint_family, "ambient_random", "pca_axis", "source_probe", "random_in_pca256"]
        if family and family in set(table["diagnostic_family"])
    ]
    if primary:
        plt.figure(figsize=(9, 5))
        for family in primary:
            values = table[table["diagnostic_family"] == family]["tau_within"]
            plt.hist(values, bins=30, alpha=0.42, label=family)
        plt.xlabel("tau_within")
        plt.ylabel("direction count")
        plt.legend(fontsize=8)
        path = os.path.join(out_dir, "random_in_span_tau_histogram.png")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs["random_in_span_tau_histogram"] = path
        plt.figure(figsize=(9, 5))
        for family in primary:
            family_profile = profile_summary[
                (profile_summary["split"] == latest_split) & (profile_summary["diagnostic_family"] == family)
            ]
            if not family_profile.empty:
                plt.plot(family_profile["lag"], family_profile["mean_autocorrelation"], label=family)
        plt.xlabel("lag")
        plt.ylabel("mean within-document autocorrelation")
        plt.legend(fontsize=8)
        path = os.path.join(out_dir, "mean_autocorrelation_profiles.png")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs["mean_autocorrelation_profiles"] = path
    sweep = nested_sweep[nested_sweep["split"] == latest_split].copy()
    if not sweep.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(sweep["nested_k"], sweep["q50_tau_within"], marker="o")
        plt.fill_between(sweep["nested_k"], sweep["q50_tau_within"], sweep["q75_tau_within"], alpha=0.25)
        plt.xscale("log")
        plt.xlabel("nested candidate-pool dimension")
        plt.ylabel("tau_within")
        path = os.path.join(out_dir, "nested_dimension_tau_curve.png")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs["nested_dimension_tau_curve"] = path
    rank_curve = source_probe_rank_curve[source_probe_rank_curve["split"] == latest_split].copy()
    if not rank_curve.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(rank_curve["source_rank"], rank_curve["tau_within"], marker="o", markersize=3)
        null_band = rank_curve["ambient_random_q95_tau_within"].dropna()
        if not null_band.empty:
            plt.axhline(float(null_band.iloc[0]), linestyle="--", color="black", label="ambient random Q95")
            plt.legend(fontsize=8)
        plt.xlabel("ordered source-probe rank")
        plt.ylabel("tau_within")
        path = os.path.join(out_dir, "source_probe_rank_tau_curve.png")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs["source_probe_rank_tau_curve"] = path
    if not geometry.empty:
        containment = geometry.groupby("pca_k_used", as_index=False)["containment"].first()
        plt.figure(figsize=(7, 4))
        plt.plot(containment["pca_k_used"], containment["containment"], marker="o")
        plt.xlabel("top residual PCA dimensions")
        plt.ylabel("headline-span containment")
        path = os.path.join(out_dir, "pca_containment_curve.png")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        outputs["pca_containment_curve"] = path
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed fat-subspace diagnostics after projection collapse.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", default=["val"], choices=["val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--k-fat", type=int, default=None)
    parser.add_argument("--random-in-span-directions", type=int, default=512)
    parser.add_argument("--mixing-directions-per-group", type=int, default=128)
    parser.add_argument("--direct-pca-directions-per-group", type=int, default=128)
    parser.add_argument("--random-in-span-seed", type=int, default=31)
    parser.add_argument("--nested-sweep-k", nargs="+", type=int, default=None)
    parser.add_argument("--mixing-band-seed", type=int, default=33)
    parser.add_argument("--bootstrap-seed", type=int, default=34)
    parser.add_argument("--direct-pca-span-seed", type=int, default=35)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-temp-dir", default=None)
    parser.add_argument("--direction-chunk", type=int, default=128)
    parser.add_argument("--bootstrap-direction-chunk", type=int, default=32)
    parser.add_argument("--profile-batches", type=int, default=None)
    args = parser.parse_args()

    logger = get_logger(__name__)
    config, store = load_config_and_store(args.config)
    residual_cfg = require_residual_geometry_config(config)
    if residual_cfg.subspace is None:
        raise ValueError("residual_geometry.subspace is required for fat-subspace diagnostics.")
    if args.direction_chunk <= 0 or args.bootstrap_direction_chunk <= 0:
        raise ValueError("direction chunks must be positive")
    if min(args.random_in_span_directions, args.mixing_directions_per_group, args.direct_pca_directions_per_group) <= 0:
        raise ValueError("diagnostic direction counts must be positive")
    if args.profile_batches is not None and args.profile_batches <= 0:
        raise ValueError("--profile-batches must be positive")
    if Path(store.fat_subspace_summary_path).exists() and not args.overwrite and args.profile_batches is None:
        previous = load_json(store.fat_subspace_summary_path)
        completed_splits = set(str(split) for split in previous.get("evaluated_splits", []))
        if set(args.splits).issubset(completed_splits):
            logger.info("Fat-subspace diagnostics already contain splits=%s; skipping.", args.splits)
            return 0
    metadata = artifact_metadata(config, "residual_geometry_05b_fat_subspace_diagnostics")
    all_main_probes = _load_all_main_probes(store)
    validation_timescales = _validation_timescales(store)
    basis, k_star, k_override = _canonical_basis(
        store,
        all_main_probes,
        validation_timescales,
        dedup_threshold=residual_cfg.subspace.dedup_abs_cosine_threshold,
        k_override=args.k_fat,
        metadata=metadata,
    )
    nested_sweep_k = _nested_sweep_values(basis.k, args.nested_sweep_k)
    probes, manifest, source_original = _evaluation_probes(
        store,
        all_main_probes,
        basis,
        random_in_span_count=args.random_in_span_directions,
        mixing_count=args.mixing_directions_per_group,
        direct_pca_count=args.direct_pca_directions_per_group,
        random_in_span_seed=args.random_in_span_seed,
        mixing_band_seed=args.mixing_band_seed,
        direct_pca_seed=args.direct_pca_span_seed,
        nested_sweep_k=nested_sweep_k,
    )
    manifest_path = os.path.join(store.fat_subspace_dir, "direction_manifest.parquet")
    save_parquet(manifest, manifest_path)
    pca_original = ResidualProbeSet(
        directions=all_main_probes.directions[all_main_probes.probe_family.astype(str) == "pca"],
        probe_ids=all_main_probes.probe_ids[all_main_probes.probe_family.astype(str) == "pca"],
        probe_family=all_main_probes.probe_family[all_main_probes.probe_family.astype(str) == "pca"],
    )
    geometry = pca_embedding_geometry(basis.basis, pca_original.directions, p_values=[1, 2, 4, 8, 16, 31, 64, 128, 256, 512])
    geometry_path = os.path.join(store.fat_subspace_dir, "pca_embedding_geometry.parquet")
    loading_path = os.path.join(store.fat_subspace_dir, "pca_axis_loading_profile.parquet")
    save_parquet(geometry, geometry_path)
    save_parquet(pca_axis_loading_profile(basis.basis, pca_original.directions), loading_path)

    bootstrap_replicates = int(args.bootstrap_replicates or residual_cfg.bootstrap_replicates)
    if bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    source_train_rows = _source_probe_train_rows(store, source_original)
    source_train_profiles = _source_probe_train_profile_bundle(store, source_original)
    source_train_eval_needed = source_train_profiles is None or source_train_rows.empty
    streamed_splits = list(args.splits)
    if source_train_eval_needed and args.profile_batches is None:
        streamed_splits.append("train")
    split_tokens = {split: token_matrix(split_contexts(store, split)) for split in streamed_splits}
    estimated_bytes = residual_cfg.projections.batch_size * residual_cfg.max_tokens * config.model.d_model * 4
    estimated_batches = sum(
        (len(tokens) + residual_cfg.projections.batch_size - 1) // residual_cfg.projections.batch_size
        for tokens in split_tokens.values()
    )
    logger.info(
        "Fat-subspace evaluation: directions=%d splits=%s forward_passes=%d residual_transfer_per_batch=%.2f GiB "
        "estimated_total_transfer=%.2f GiB bootstrap=%s reps=%d",
        probes.n_probes,
        streamed_splits,
        estimated_batches,
        estimated_bytes / (1024**3),
        estimated_batches * estimated_bytes / (1024**3),
        not args.skip_bootstrap and args.profile_batches is None,
        bootstrap_replicates,
    )
    provider = ResidualStreamProvider(config)
    provider.load()
    new_timescales: list[pd.DataFrame] = []
    new_profiles: list[ProfileBundle] = []
    new_bootstrap: list[pd.DataFrame] = []
    if source_train_eval_needed and args.profile_batches is None:
        logger.info("Saved Stage 03 source-probe train artifacts are incomplete; evaluating fixed source probes on train.")
        source_train_probes, source_train_manifest = _renamed_probe_set(
            source_original,
            family="source_probe",
            id_prefix="source_probe",
            direction_origin="headline_ranked_source_probe",
            span_dimension=basis.k,
        )
        source_train_evaluated = _evaluate_split(
            provider,
            split_tokens["train"],
            source_train_probes,
            source_train_manifest,
            residual_cfg,
            split="train",
            metadata=metadata,
            direction_chunk=args.direction_chunk,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_direction_chunk=args.bootstrap_direction_chunk,
            skip_bootstrap=True,
            bootstrap_temp_dir=args.bootstrap_temp_dir,
            profile_batches=None,
        )
        if source_train_evaluated is None:
            raise AssertionError("Source-probe train evaluation unexpectedly entered profile-only mode.")
        source_train_rows, source_train_profiles, _ = source_train_evaluated
    for split in args.splits:
        evaluated = _evaluate_split(
            provider,
            split_tokens[split],
            probes,
            manifest,
            residual_cfg,
            split=split,
            metadata=metadata,
            direction_chunk=args.direction_chunk,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_direction_chunk=args.bootstrap_direction_chunk,
            skip_bootstrap=args.skip_bootstrap,
            bootstrap_temp_dir=args.bootstrap_temp_dir,
            profile_batches=args.profile_batches,
        )
        if evaluated is None:
            return 0
        split_timescales, split_profiles, split_bootstrap = evaluated
        new_timescales.append(split_timescales)
        new_profiles.append(split_profiles)
        if not split_bootstrap.empty:
            new_bootstrap.append(split_bootstrap)

    timescale_path = os.path.join(store.fat_subspace_dir, "direction_timescales.parquet")
    profiles_path = os.path.join(store.fat_subspace_dir, "autocorr_profiles.npz")
    profile_summary_path = os.path.join(store.fat_subspace_dir, "mean_autocorrelation_profiles.parquet")
    bootstrap_path = os.path.join(store.fat_subspace_dir, "bootstrap_quantiles.parquet")
    source_stability_path = os.path.join(store.fat_subspace_dir, "source_probe_split_stability.parquet")
    nested_sweep_path = os.path.join(store.fat_subspace_dir, "nested_dimension_sweep.parquet")
    source_rank_curve_path = os.path.join(store.fat_subspace_dir, "source_probe_rank_curve.parquet")
    merged_timescales = _merge_split_table(timescale_path, new_timescales, splits=args.splits)
    profiles_to_merge = new_profiles + ([source_train_profiles] if source_train_profiles is not None else [])
    merged_profiles = _merge_profiles(_load_profile_bundle(profiles_path), profiles_to_merge)
    save_parquet(merged_timescales, timescale_path)
    _save_profile_bundle(profiles_path, merged_profiles)
    profile_summary = _mean_profile_summary(merged_profiles)
    save_parquet(profile_summary, profile_summary_path)
    if new_bootstrap:
        save_parquet(_merge_split_table(bootstrap_path, new_bootstrap, splits=args.splits), bootstrap_path)
    source_stability = _source_stability_table(
        merged_timescales,
        source_train_rows,
        source_original,
    )
    save_parquet(source_stability, source_stability_path)
    ambient_null_bands = _ambient_null_bands(merged_timescales)
    nested_sweep = _nested_dimension_sweep(merged_timescales, ambient_null_bands)
    source_rank_curve = _source_probe_rank_curve(source_stability, ambient_null_bands)
    save_parquet(nested_sweep, nested_sweep_path)
    save_parquet(source_rank_curve, source_rank_curve_path)
    quantiles = _metric_quantiles(merged_timescales)
    evaluated_splits = sorted(set(merged_timescales["split"].astype(str)))
    classification_split = "test" if "test" in evaluated_splits else "val"
    labels, classification_detail = _classification(
        quantiles,
        split=classification_split,
        k_fat=basis.k,
        source_stability=source_stability,
    )
    test_previously_inspected = _test_previously_inspected_before_fat_subspace_lock(store)
    generic_slow_region_observed = bool(
        "fat" in labels
        and "shrinkage_dominated" not in labels
        and classification_detail["headline_span_specific_vs_pca_controls"]
    )
    claim_level_5a_supported = bool(
        generic_slow_region_observed and classification_split == "test" and not test_previously_inspected
    )
    dimensionality = load_json(os.path.join(store.subspace_dir, "dimensionality_summary.json"))
    mixing_directions_path = os.path.join(store.fat_subspace_dir, "mixing_directions.npz")
    plots = _make_plots(
        merged_timescales,
        geometry,
        profile_summary,
        nested_sweep,
        source_rank_curve,
        os.path.join(store.fat_subspace_dir, "plots"),
    )
    summary = {
        **metadata,
        "status": "complete" if "test" in evaluated_splits else "validation_complete_test_pending",
        "classification": labels,
        "classification_detail": classification_detail,
        "generic_slow_region_observed": generic_slow_region_observed,
        "coordinate_free_generic_slow_region_supported": claim_level_5a_supported,
        "claim_level_5a_supported": claim_level_5a_supported,
        "k_star": int(k_star),
        "k_fat": int(basis.k),
        "k_override": bool(k_override),
        "nested_sweep_k": nested_sweep_k,
        "nested_random_in_span_seed": int(args.random_in_span_seed),
        "ambient_random_null_band_definition": "Q95(tau_within) over ambient_random directions by split",
        "ambient_random_q95_tau_within_by_split": ambient_null_bands,
        "first_source_rank_in_ambient_null_band_by_split": _first_source_rank_in_null_band(source_rank_curve),
        "top_kstar_geometric_participation_ratio": dimensionality.get("top_kstar_geometric_participation_ratio"),
        "top_kstar_slowness_participation_ratio": dimensionality.get("top_kstar_slowness_participation_ratio"),
        "nested_slow_core_candidate_interval": None,
        "nested_slow_core_cutoff_status": "descriptive_only_no_validation_locked_cutoff_rule",
        "nested_slow_core_candidate_supported": False,
        "nested_slow_core_confirmed": False,
        "source_probe_ids": basis.source_probe_ids.astype(str).tolist(),
        "source_probe_family": basis.source_probe_family.astype(str).tolist(),
        "evaluated_splits": evaluated_splits,
        "test_previously_inspected_before_fat_subspace_lock": test_previously_inspected,
        "exploratory_after_test_inspection": test_previously_inspected,
        "bootstrap_status": "preserved_existing" if args.skip_bootstrap and Path(bootstrap_path).exists() else "skipped" if args.skip_bootstrap else "run",
        "bootstrap_replicates": bootstrap_replicates if not args.skip_bootstrap else 0,
        "quantiles": quantiles,
        "source_probe_split_stability": _source_stability_summary(source_stability),
        "source_probe_train_profiles_in_bundle": bool(
            np.any((merged_profiles.split == "train") & (merged_profiles.probe_family == "source_probe"))
        ),
        "direction_manifest_path": manifest_path,
        "random_in_span_directions_path": os.path.join(store.fat_subspace_dir, "random_in_span_directions.npz"),
        "nested_random_in_span_directions_path": os.path.join(store.fat_subspace_dir, "nested_random_in_span_directions.npz"),
        "mixing_directions_path": mixing_directions_path if Path(mixing_directions_path).exists() else None,
        "direction_timescales_path": timescale_path,
        "nested_dimension_sweep_path": nested_sweep_path,
        "source_probe_rank_curve_path": source_rank_curve_path,
        "autocorr_profiles_path": profiles_path,
        "mean_autocorrelation_profiles_path": profile_summary_path,
        "pca_embedding_geometry_path": geometry_path,
        "pca_axis_loading_profile_path": loading_path,
        "source_probe_split_stability_path": source_stability_path,
        "bootstrap_quantiles_path": bootstrap_path if Path(bootstrap_path).exists() else None,
        "plots": plots,
        "note": (
            "FS1-FS4 evaluate fixed directions only. The complete coupled nested curve is primary. A fat classification means "
            "generic sampled directions inside the candidate-pool endpoint remain slow under the locked comparative rule; it "
            "does not prove that every direction in the span is slow. No smaller effective slow-core cutoff is named without "
            "a separately validation-locked cutoff rule or interval."
        ),
    }
    save_json(summary, store.fat_subspace_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
