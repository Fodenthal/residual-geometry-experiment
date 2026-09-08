from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from residual_geometry.autocorr.estimators import AutocorrResult, extract_tau
from residual_geometry.residuals.probes import ResidualProbeSet, probe_set_from_directions
from residual_geometry.utils.io import ensure_dir


@dataclass(frozen=True)
class ResidualBasisArtifact:
    basis: np.ndarray
    k: int
    source_probe_ids: np.ndarray
    source_probe_family: np.ndarray

    def __post_init__(self) -> None:
        if self.basis.ndim != 2:
            raise ValueError("basis must have shape (d_model, k)")
        if self.basis.shape[1] != self.k:
            raise ValueError("basis second dimension must match k")
        if len(self.source_probe_ids) != self.k:
            raise ValueError("source_probe_ids length must match k")
        if len(self.source_probe_family) != self.k:
            raise ValueError("source_probe_family length must match k")


def save_residual_basis_artifact(
    path: str,
    basis: np.ndarray,
    source_probe_ids: np.ndarray,
    source_probe_family: np.ndarray,
    **metadata: object,
) -> None:
    basis = np.asarray(basis, dtype=np.float32)
    k = int(basis.shape[1])
    artifact = ResidualBasisArtifact(
        basis=basis,
        k=k,
        source_probe_ids=np.asarray(source_probe_ids).astype(str),
        source_probe_family=np.asarray(source_probe_family).astype(str),
    )
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.savez_compressed(
        path,
        basis=artifact.basis,
        k=np.array(artifact.k),
        source_probe_ids=artifact.source_probe_ids,
        source_probe_family=artifact.source_probe_family,
        **metadata,
    )


def load_residual_basis_artifact(path: str) -> ResidualBasisArtifact:
    with np.load(path, allow_pickle=False) as data:
        basis = data["basis"].astype(np.float32)
        k = int(data["k"])
        if "source_probe_ids" not in data or "source_probe_family" not in data:
            raise ValueError(f"Basis artifact {path} is missing source probe metadata; rebuild Stage 04.")
        return ResidualBasisArtifact(
            basis=basis,
            k=k,
            source_probe_ids=data["source_probe_ids"].astype(str),
            source_probe_family=data["source_probe_family"].astype(str),
        )


def sample_unit_directions_in_span(basis: np.ndarray, count: int, seed: int) -> np.ndarray:
    q = np.asarray(basis, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError("basis must have shape (d_model, span_dimension)")
    if count <= 0:
        raise ValueError("count must be positive")
    if q.shape[1] <= 0:
        raise ValueError("basis must contain at least one direction")
    rng = np.random.default_rng(seed)
    coordinates = rng.normal(size=(count, q.shape[1]))
    coordinates /= np.linalg.norm(coordinates, axis=1, keepdims=True)
    directions = coordinates @ q.T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions.astype(np.float32)


def coupled_nested_span_probe_set(
    basis: np.ndarray,
    sweep_k: list[int],
    *,
    count: int,
    seed: int,
) -> tuple[ResidualProbeSet, np.ndarray]:
    # `basis` must be the Q factor of a column-ordered QR decomposition of the
    # source-probe matrix [g_1 | ... | g_{k*}].  Column-ordered QR guarantees that
    # span(basis[:, :k]) == span(g_1, ..., g_k) for every k, so the prefix-column
    # slice q[:, :k] correctly spans Q_k^res = orth(g_1, ..., g_k).  This property
    # does NOT hold for SVD-based orthonormalization; if orthonormal_basis is ever
    # changed to use SVD, this function must be updated to build Q_k^res per-k.
    q = np.asarray(basis, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError("basis must have shape (d_model, candidate_pool_dimension)")
    if count <= 0:
        raise ValueError("count must be positive")
    k_values = sorted(set(int(k) for k in sweep_k))
    if not k_values or k_values[0] <= 0 or k_values[-1] > q.shape[1]:
        raise ValueError("sweep_k must contain dimensions between 1 and the candidate-pool dimension")
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(count, q.shape[1]))
    groups: list[ResidualProbeSet] = []
    for k in k_values:
        family = f"random_in_q{k}"
        directions = coefficients[:, :k] @ q[:, :k].T
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        groups.append(probe_set_from_directions(directions, family=family, id_prefix=family))
    directions = np.concatenate([group.directions for group in groups], axis=0).astype(np.float32)
    probe_ids = np.concatenate([group.probe_ids for group in groups], axis=0)
    probe_family = np.concatenate([group.probe_family for group in groups], axis=0)
    return ResidualProbeSet(directions=directions, probe_ids=probe_ids, probe_family=probe_family), coefficients.astype(np.float32)


def probe_set_in_span(
    basis: np.ndarray,
    count: int,
    seed: int,
    family: str,
    id_prefix: str,
) -> ResidualProbeSet:
    directions = sample_unit_directions_in_span(basis, count=count, seed=seed)
    return probe_set_from_directions(directions, family=family, id_prefix=id_prefix)


def participation_ratio(weights: np.ndarray) -> float:
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("weights must have shape (n,)")
    denominator = float(np.sum(arr**2))
    return float(np.sum(arr) ** 2 / denominator) if denominator > 0 else 0.0


def geometric_participation_ratio(directions: np.ndarray) -> float:
    arr = np.asarray(directions, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("directions must have shape (n, d_model)")
    singular_values = np.linalg.svd(arr, compute_uv=False)
    return participation_ratio(singular_values**2)


def positive_profile_area(profile: np.ndarray) -> float:
    arr = np.asarray(profile, dtype=np.float64)
    return float(np.nansum(np.maximum(arr[1:], 0.0)))


def positive_profile_areas(profiles: np.ndarray) -> np.ndarray:
    arr = np.asarray(profiles)
    if arr.ndim != 2:
        raise ValueError("profiles must have shape (directions, lags)")
    return np.asarray([positive_profile_area(profile) for profile in arr])


def autocorr_result_from_document_matrices(corr: np.ndarray, valid: np.ndarray) -> AutocorrResult:
    if corr.shape != valid.shape or corr.ndim != 3:
        raise ValueError("corr and valid must have shape (documents, directions, lags)")
    counts = valid.sum(axis=0).astype(np.int64)
    sums = np.where(valid, corr, 0.0).sum(axis=0, dtype=np.float64)
    profiles = np.full(counts.shape, np.nan, dtype=np.float64)
    has_valid = counts > 0
    profiles[has_valid] = sums[has_valid] / counts[has_valid]
    profiles[:, 0] = 1.0
    return AutocorrResult(profiles=profiles, valid_doc_counts=counts)


def pca_embedding_geometry(
    basis: np.ndarray,
    pca_directions: np.ndarray,
    p_values: list[int],
) -> pd.DataFrame:
    q = np.asarray(basis, dtype=np.float64)
    pca = np.asarray(pca_directions, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError("basis must have shape (d_model, k)")
    if pca.ndim != 2 or pca.shape[1] != q.shape[0]:
        raise ValueError("pca_directions must have shape (components, d_model)")
    rows: list[dict[str, object]] = []
    for requested_p in p_values:
        p = min(int(requested_p), len(pca))
        if p <= 0:
            continue
        pca_basis, _ = np.linalg.qr(pca[:p].T)
        singular_values = np.linalg.svd(q.T @ pca_basis[:, :p], compute_uv=False)
        containment = float(np.sum(singular_values**2) / q.shape[1])
        complete = bool(p >= q.shape[1])
        for index, singular_value in enumerate(singular_values):
            clipped = float(np.clip(singular_value, 0.0, 1.0))
            rows.append(
                {
                    "pca_k_requested": int(requested_p),
                    "pca_k_used": int(p),
                    "persistent_k": int(q.shape[1]),
                    "singular_value_index": int(index),
                    "singular_value": clipped,
                    "principal_angle_degrees": float(np.degrees(np.arccos(clipped))) if complete else float("nan"),
                    "principal_angle_spectrum_complete": complete,
                    "containment": containment,
                }
            )
    return pd.DataFrame(rows)


def pca_axis_loading_profile(basis: np.ndarray, pca_directions: np.ndarray) -> pd.DataFrame:
    q = np.asarray(basis, dtype=np.float64)
    pca = np.asarray(pca_directions, dtype=np.float64)
    if q.ndim != 2 or pca.ndim != 2 or pca.shape[1] != q.shape[0]:
        raise ValueError("basis and pca_directions have incompatible shapes")
    loadings = np.sum((pca @ q) ** 2, axis=1) / q.shape[1]
    return pd.DataFrame(
        {
            "pca_index": np.arange(1, len(loadings) + 1, dtype=np.int64),
            "mean_squared_loading": loadings,
            "cumulative_mean_squared_loading": np.cumsum(loadings),
        }
    )


def bootstrap_family_quantiles(
    corr_matrix: np.ndarray,
    valid_matrix: np.ndarray,
    probe_families: np.ndarray,
    *,
    split: str,
    max_lag: int,
    min_valid_docs: int,
    min_valid_lag_fraction: float,
    smoothing_width: int,
    replicates: int,
    seed: int,
    direction_chunk: int = 32,
) -> pd.DataFrame:
    if corr_matrix.shape != valid_matrix.shape or corr_matrix.ndim != 3:
        raise ValueError("corr_matrix and valid_matrix must have shape (documents, directions, lags)")
    docs, directions, _ = corr_matrix.shape
    families = np.asarray(probe_families).astype(str)
    if len(families) != directions:
        raise ValueError("probe_families length must match directions")
    if docs <= 0 or replicates <= 0 or direction_chunk <= 0:
        raise ValueError("docs, replicates, and direction_chunk must be positive")

    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, docs), dtype=np.float64)
    for replicate in range(replicates):
        weights[replicate] = np.bincount(rng.integers(0, docs, size=docs), minlength=docs)

    rows: list[dict[str, object]] = []
    for family in sorted(set(families)):
        family_indices = np.flatnonzero(families == family)
        tau_samples = np.zeros((replicates, len(family_indices)), dtype=np.float64)
        area_samples = np.zeros((replicates, len(family_indices)), dtype=np.float64)
        censored_samples = np.zeros((replicates, len(family_indices)), dtype=bool)
        for local_start in range(0, len(family_indices), direction_chunk):
            local_stop = min(local_start + direction_chunk, len(family_indices))
            selected = family_indices[local_start:local_stop]
            valid = np.asarray(valid_matrix[:, selected, :], dtype=np.float64)
            values = np.where(valid > 0, np.asarray(corr_matrix[:, selected, :], dtype=np.float64), 0.0)
            counts = np.einsum("rd,dpk->rpk", weights, valid, optimize=True)
            sums = np.einsum("rd,dpk->rpk", weights, values, optimize=True)
            profiles = np.full_like(sums, np.nan, dtype=np.float64)
            has_valid = counts > 0
            profiles[has_valid] = sums[has_valid] / counts[has_valid]
            profiles[:, :, 0] = 1.0
            for replicate in range(replicates):
                for offset in range(local_stop - local_start):
                    stats = extract_tau(
                        profiles[replicate, offset],
                        max_lag=max_lag,
                        valid_counts=counts[replicate, offset],
                        min_valid_docs=min_valid_docs,
                        min_valid_lag_fraction=min_valid_lag_fraction,
                        smoothing_width=smoothing_width,
                    )
                    tau_samples[replicate, local_start + offset] = float(stats["tau"])
                    area_samples[replicate, local_start + offset] = positive_profile_area(profiles[replicate, offset])
                    censored_samples[replicate, local_start + offset] = bool(stats["right_censored"])
        for metric, samples in [("tau_within", tau_samples), ("positive_profile_area", area_samples)]:
            for quantile in [0.5, 0.75, 0.9, 0.95]:
                quantile_samples = np.quantile(samples, quantile, axis=1)
                rows.append(
                    {
                        "split": split,
                        "diagnostic_family": family,
                        "metric": metric,
                        "quantile": float(quantile),
                        "bootstrap_ci_low": float(np.quantile(quantile_samples, 0.025)),
                        "bootstrap_ci_high": float(np.quantile(quantile_samples, 0.975)),
                        "bootstrap_mean": float(np.mean(quantile_samples)),
                        "bootstrap_replicates": int(replicates),
                        "bootstrap_unit": "document",
                    }
                )
        rows.append(
            {
                "split": split,
                "diagnostic_family": family,
                "metric": "right_censoring_rate",
                "quantile": float("nan"),
                "bootstrap_ci_low": float(np.quantile(censored_samples.mean(axis=1), 0.025)),
                "bootstrap_ci_high": float(np.quantile(censored_samples.mean(axis=1), 0.975)),
                "bootstrap_mean": float(np.mean(censored_samples)),
                "bootstrap_replicates": int(replicates),
                "bootstrap_unit": "document",
            }
        )
    return pd.DataFrame(rows)
