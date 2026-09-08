from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from residual_geometry.residuals.pca import covariance_spectrum_diagnostics
from residual_geometry.residuals.probes import ResidualProbeSet, normalize_rows, probe_set_from_directions


@dataclass(frozen=True)
class TimeLaggedFit:
    probes: ResidualProbeSet
    mean: np.ndarray
    generalized_eigenvalues: np.ndarray
    summary: dict[str, object]
    covariance_eigenvalues: np.ndarray
    covariance_diagnostics: dict[str, object]


def _centered_rows(residuals: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return residuals.reshape(-1, residuals.shape[-1]).astype(np.float64, copy=False) - mean.reshape(1, -1)


def estimate_covariances(residuals: np.ndarray, lag_set: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if residuals.ndim != 3:
        raise ValueError("residuals must have shape (docs, positions, d_model)")
    docs, positions, d_model = residuals.shape
    if any(lag >= positions for lag in lag_set):
        raise ValueError("all lags must be smaller than sequence length")
    mean = residuals.reshape(-1, d_model).astype(np.float64).mean(axis=0)
    centered = residuals.astype(np.float64, copy=False) - mean.reshape(1, 1, d_model)
    flat = centered.reshape(-1, d_model)
    sigma0 = (flat.T @ flat) / max(1, flat.shape[0])
    sigma_lag = np.zeros((d_model, d_model), dtype=np.float64)
    pair_count = 0
    for lag in lag_set:
        left = centered[:, : positions - lag, :].reshape(-1, d_model)
        right = centered[:, lag:, :].reshape(-1, d_model)
        cov = (left.T @ right) / max(1, left.shape[0])
        sigma_lag += 0.5 * (cov + cov.T)
        pair_count += int(left.shape[0])
    sigma_lag /= float(len(lag_set))
    return mean.astype(np.float32), sigma0, sigma_lag, pair_count


def estimate_covariances_streaming(
    residual_batches,
    lag_set: list[int],
    d_model: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Estimate global mean, Sigma0, and symmetrized lag covariance from residual batches.

    The input must be a re-iterable factory or sequence. If an iterator is passed, it is
    materialized as residual arrays by the caller; scripts usually pass a function that
    creates a fresh generator for each pass.
    """
    total = 0
    sum_vec = np.zeros(d_model, dtype=np.float64)
    for batch in residual_batches():
        flat = batch.reshape(-1, d_model).astype(np.float64, copy=False)
        sum_vec += flat.sum(axis=0)
        total += int(flat.shape[0])
    if total == 0:
        raise ValueError("no residuals available for covariance estimation")
    mean = sum_vec / total
    sigma0_sum = np.zeros((d_model, d_model), dtype=np.float64)
    lag_sums = {int(lag): np.zeros((d_model, d_model), dtype=np.float64) for lag in lag_set}
    lag_counts = {int(lag): 0 for lag in lag_set}
    pair_count = 0
    for batch in residual_batches():
        centered = batch.astype(np.float64, copy=False) - mean.reshape(1, 1, d_model)
        flat = centered.reshape(-1, d_model)
        sigma0_sum += flat.T @ flat
        _, positions, _ = centered.shape
        for lag in lag_set:
            if lag >= positions:
                raise ValueError("all lags must be smaller than sequence length")
            left = centered[:, : positions - lag, :].reshape(-1, d_model)
            right = centered[:, lag:, :].reshape(-1, d_model)
            cov_sum = left.T @ right
            lag_sums[int(lag)] += 0.5 * (cov_sum + cov_sum.T)
            lag_counts[int(lag)] += int(left.shape[0])
            pair_count += int(left.shape[0])
    sigma0 = sigma0_sum / float(total)
    sigma_lag = np.zeros((d_model, d_model), dtype=np.float64)
    for lag in lag_set:
        sigma_lag += lag_sums[int(lag)] / max(lag_counts[int(lag)], 1)
    sigma_lag /= float(len(lag_set))
    return mean.astype(np.float32), sigma0, sigma_lag, pair_count, total


def resolve_ridge(
    covariance_eigenvalues: np.ndarray,
    trace_scale: float,
    initial_scale: float,
    max_scale: float,
    condition_threshold: float,
) -> dict[str, object]:
    vals = np.asarray(covariance_eigenvalues, dtype=np.float64)
    epsilon = initial_scale * trace_scale
    initial_epsilon = epsilon
    cond = float((vals.max() + epsilon) / max(vals.min() + epsilon, np.finfo(np.float64).tiny))
    initial_cond = cond
    doublings = 0
    while cond > condition_threshold and (epsilon * 2.0) <= max_scale * trace_scale:
        epsilon *= 2.0
        doublings += 1
        cond = float((vals.max() + epsilon) / max(vals.min() + epsilon, np.finfo(np.float64).tiny))
    epsilon_scale = float(epsilon / trace_scale) if trace_scale > 0 else float("inf")
    return {
        "initial_epsilon": float(initial_epsilon),
        "epsilon_final": float(epsilon),
        "ridge_doublings": int(doublings),
        "initial_condition_number": float(initial_cond),
        "final_condition_number": float(cond),
        "epsilon_scale": epsilon_scale,
        "condition_number_warning": bool(cond > condition_threshold),
        "time_lagged_fit_unstable": bool(epsilon_scale > max_scale or cond > condition_threshold),
    }


def solve_time_lagged_eigenproblem(
    sigma0: np.ndarray,
    sigma_lag: np.ndarray,
    whitening_pcs: int,
    output_directions: int,
    ridge_scale: float,
    max_ridge_scale: float,
    condition_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(sigma0)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    positive = vals > 0
    vals = vals[positive]
    vecs = vecs[:, positive]
    n_white = min(whitening_pcs, len(vals))
    if n_white == 0:
        raise ValueError("sigma0 has no positive eigenvalues")
    vals_w = vals[:n_white]
    vecs_w = vecs[:, :n_white]
    trace_scale = float(np.trace(sigma0) / sigma0.shape[0])
    ridge = resolve_ridge(vals_w, trace_scale, ridge_scale, max_ridge_scale, condition_threshold)
    eps = float(ridge["epsilon_final"])
    inv_sqrt = 1.0 / np.sqrt(vals_w + eps)
    whitened_lag = (vecs_w.T @ sigma_lag @ vecs_w) * np.outer(inv_sqrt, inv_sqrt)
    gen_vals, gen_vecs = np.linalg.eigh(0.5 * (whitened_lag + whitened_lag.T))
    gen_order = np.argsort(gen_vals)[::-1]
    n_out = min(output_directions, len(gen_order))
    gen_vals = gen_vals[gen_order[:n_out]]
    gen_vecs = gen_vecs[:, gen_order[:n_out]]
    directions = (vecs_w * inv_sqrt.reshape(1, -1)) @ gen_vecs
    directions = normalize_rows(directions.T)
    ridge["condition_number_after_ridge_fitted_subspace"] = ridge["final_condition_number"]
    return directions, gen_vals.astype(np.float64), ridge, vals, vecs[:, : min(128, vecs.shape[1])].T


def fit_time_lagged_residual_directions(
    residuals: np.ndarray,
    lag_set: list[int],
    whitening_pcs: int,
    output_directions: int,
    ridge_scale: float = 1e-4,
    max_ridge_scale: float = 1e-1,
    condition_threshold: float = 1e4,
    pca_reference_directions: np.ndarray | None = None,
) -> TimeLaggedFit:
    mean, sigma0, sigma_lag, pair_count = estimate_covariances(residuals, lag_set)
    directions, gen_vals, ridge, cov_eigvals, cov_components = solve_time_lagged_eigenproblem(
        sigma0=sigma0,
        sigma_lag=sigma_lag,
        whitening_pcs=whitening_pcs,
        output_directions=output_directions,
        ridge_scale=ridge_scale,
        max_ridge_scale=max_ridge_scale,
        condition_threshold=condition_threshold,
    )
    probes = probe_set_from_directions(directions, family="time_lagged", id_prefix="time_lagged_residual")
    covariance_diagnostics = covariance_spectrum_diagnostics(
        cov_eigvals,
        components=cov_components,
        reference_directions=pca_reference_directions,
    )
    covariance_diagnostics["condition_number_after_ridge"] = float(ridge["final_condition_number"])
    if pca_reference_directions is not None and len(pca_reference_directions):
        overlap = np.abs(directions.astype(np.float64) @ pca_reference_directions.astype(np.float64).T)
        covariance_diagnostics["cosine_overlap_top_time_lagged_to_top_pca"] = overlap.max(axis=1).astype(float).tolist()
    summary = {
        "lag_set": [int(lag) for lag in lag_set],
        "whitening_pcs_requested": int(whitening_pcs),
        "whitening_pcs_used": int(min(whitening_pcs, len(cov_eigvals))),
        "output_directions_requested": int(output_directions),
        "output_directions": int(probes.n_probes),
        "lagged_pair_count": int(pair_count),
        "top_generalized_eigenvalues": gen_vals[: min(32, len(gen_vals))].astype(float).tolist(),
        **ridge,
    }
    return TimeLaggedFit(
        probes=probes,
        mean=mean,
        generalized_eigenvalues=gen_vals,
        summary=summary,
        covariance_eigenvalues=cov_eigvals,
        covariance_diagnostics=covariance_diagnostics,
    )


def fit_time_lagged_from_covariances(
    mean: np.ndarray,
    sigma0: np.ndarray,
    sigma_lag: np.ndarray,
    pair_count: int,
    lag_set: list[int],
    whitening_pcs: int,
    output_directions: int,
    ridge_scale: float = 1e-4,
    max_ridge_scale: float = 1e-1,
    condition_threshold: float = 1e4,
    pca_reference_directions: np.ndarray | None = None,
) -> TimeLaggedFit:
    directions, gen_vals, ridge, cov_eigvals, cov_components = solve_time_lagged_eigenproblem(
        sigma0=sigma0,
        sigma_lag=sigma_lag,
        whitening_pcs=whitening_pcs,
        output_directions=output_directions,
        ridge_scale=ridge_scale,
        max_ridge_scale=max_ridge_scale,
        condition_threshold=condition_threshold,
    )
    probes = probe_set_from_directions(directions, family="time_lagged", id_prefix="time_lagged_residual")
    covariance_diagnostics = covariance_spectrum_diagnostics(
        cov_eigvals,
        components=cov_components,
        reference_directions=pca_reference_directions,
    )
    covariance_diagnostics["condition_number_after_ridge"] = float(ridge["final_condition_number"])
    if pca_reference_directions is not None and len(pca_reference_directions):
        overlap = np.abs(directions.astype(np.float64) @ pca_reference_directions.astype(np.float64).T)
        covariance_diagnostics["cosine_overlap_top_time_lagged_to_top_pca"] = overlap.max(axis=1).astype(float).tolist()
    summary = {
        "lag_set": [int(lag) for lag in lag_set],
        "whitening_pcs_requested": int(whitening_pcs),
        "whitening_pcs_used": int(min(whitening_pcs, len(cov_eigvals))),
        "output_directions_requested": int(output_directions),
        "output_directions": int(probes.n_probes),
        "lagged_pair_count": int(pair_count),
        "top_generalized_eigenvalues": gen_vals[: min(32, len(gen_vals))].astype(float).tolist(),
        **ridge,
    }
    return TimeLaggedFit(
        probes=probes,
        mean=mean.astype(np.float32),
        generalized_eigenvalues=gen_vals,
        summary=summary,
        covariance_eigenvalues=cov_eigvals,
        covariance_diagnostics=covariance_diagnostics,
    )
