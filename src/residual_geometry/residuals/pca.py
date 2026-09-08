from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import os

from sklearn.decomposition import PCA

from residual_geometry.residuals.probes import ResidualProbeSet, probe_set_from_directions
from residual_geometry.utils.io import ensure_dir


@dataclass(frozen=True)
class ResidualPCAFit:
    probes: ResidualProbeSet
    mean: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    diagnostics: dict[str, object]


def covariance_spectrum_diagnostics(
    eigenvalues: np.ndarray,
    components: np.ndarray | None = None,
    reference_directions: np.ndarray | None = None,
) -> dict[str, object]:
    vals = np.asarray(eigenvalues, dtype=np.float64)
    positive = vals[vals > 0]
    total = float(vals.sum())
    squared = float(np.sum(vals * vals))
    diagnostics: dict[str, object] = {
        "fitted_eigenvalue_count": int(len(vals)),
        "top_128_eigenvalues": vals[:128].astype(float).tolist(),
        "explained_variance_curve": (np.cumsum(vals) / total).astype(float).tolist() if total > 0 else [],
        "total_fitted_variance": total,
        "effective_rank": float((total * total) / squared) if squared > 0 else 0.0,
        "participation_ratio_fitted_subspace": float((positive.sum() ** 2) / np.sum(positive * positive)) if len(positive) else 0.0,
    }
    for k in [1, 5, 10, 50, 100]:
        diagnostics[f"variance_explained_top_{k}"] = float(vals[: min(k, len(vals))].sum() / total) if total > 0 else 0.0
    if len(positive):
        diagnostics["condition_number_before_ridge_fitted_subspace"] = float(positive[0] / positive[-1])
    else:
        diagnostics["condition_number_before_ridge_fitted_subspace"] = float("inf")
    if components is not None and reference_directions is not None and len(reference_directions):
        cos = np.abs(reference_directions.astype(np.float64) @ components.astype(np.float64).T)
        diagnostics["max_cosine_overlap_with_reference_pca"] = cos.max(axis=1).astype(float).tolist()
    return diagnostics


def fit_residual_pca(
    residual_samples: np.ndarray,
    n_components: int,
    seed: int,
) -> ResidualPCAFit:
    """Fit randomized PCA on residual samples with shape (positions, d_model)."""
    if residual_samples.ndim != 2:
        raise ValueError("residual_samples must have shape (positions, d_model)")
    n = min(n_components, residual_samples.shape[0], residual_samples.shape[1])
    if n <= 0:
        raise ValueError("n_components resolved to zero")
    pca = PCA(n_components=n, svd_solver="randomized", random_state=seed)
    pca.fit(residual_samples.astype(np.float32, copy=False))
    probes = probe_set_from_directions(pca.components_.astype(np.float32), family="pca", id_prefix="residual_pca")
    diagnostics = covariance_spectrum_diagnostics(pca.explained_variance_)
    diagnostics.update(
        {
            "sample_count": int(residual_samples.shape[0]),
            "d_model": int(residual_samples.shape[1]),
            "n_components": int(n),
            "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        }
    )
    return ResidualPCAFit(
        probes=probes,
        mean=pca.mean_.astype(np.float32),
        eigenvalues=pca.explained_variance_.astype(np.float64),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
        diagnostics=diagnostics,
    )


def sample_residual_positions(
    residual_batches,
    total_positions: int,
    max_positions: int,
    seed: int,
    d_model: int,
    memmap_path: str | None = None,
) -> np.ndarray:
    """Sample train-token residual rows without replacement from a re-iterable batch factory."""
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if total_positions <= 0:
        raise ValueError("total_positions must be positive")
    rng = np.random.default_rng(seed)
    sample_count = min(max_positions, total_positions)
    selected = np.sort(rng.choice(total_positions, size=sample_count, replace=False))
    if memmap_path is None:
        samples = np.empty((sample_count, d_model), dtype=np.float32)
    else:
        ensure_dir(os.path.dirname(os.path.abspath(memmap_path)))
        samples = np.memmap(memmap_path, mode="w+", dtype=np.float32, shape=(sample_count, d_model))
    write_offset = 0
    global_offset = 0
    selected_offset = 0
    for batch in residual_batches():
        flat = batch.reshape(-1, batch.shape[-1]).astype(np.float32, copy=False)
        batch_start = global_offset
        batch_end = global_offset + flat.shape[0]
        lo = selected_offset
        while lo < sample_count and selected[lo] < batch_start:
            lo += 1
        hi = lo
        while hi < sample_count and selected[hi] < batch_end:
            hi += 1
        if hi > lo:
            local = selected[lo:hi] - batch_start
            samples[write_offset : write_offset + len(local)] = flat[local]
            write_offset += len(local)
        selected_offset = hi
        global_offset = batch_end
    if write_offset != sample_count:
        raise ValueError(f"sampled {write_offset} positions but expected {sample_count}")
    if isinstance(samples, np.memmap):
        samples.flush()
    return samples
