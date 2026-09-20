"""Conditional carrier discovery and matched-ridge inference for R2.6-M-r2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from arm_d_r24.data import Pool, document_hash_fraction
from arm_d_r24.geometry import complement_basis, mean_subspace_correlation
from arm_d_r24.scoring import Frame, fit_frame, symmetric_inverse_sqrt
from arm_d_r4.common_target import GroupedRidge, bootstrap_weights

from .contract import (
    ALPHA,
    BOOTSTRAP_REPLICATES,
    CANDIDATE_RANKS,
    EIGEN_EPSILON,
    ESTIMATOR_RHO,
    LAGS,
    ORTHOGONALITY_ATOL,
    RANDOM_DRAWS,
    RANK_GAIN_FRACTION,
    RIDGE_GAP_RATIO_MIN,
    RIDGE_GRID,
    SEED,
    SPLIT_FRACTIONS,
    SPLIT_SALT,
    STABILITY_NULL_DRAWS,
    namespaced_seed,
)


def four_way_split(documents, *, salt: str = SPLIT_SALT) -> dict[str, np.ndarray]:
    order = ("fitA", "fitB", "validation", "test")
    values = np.array([document_hash_fraction(str(doc), salt) for doc in documents])
    out, low = {}, 0.0
    for name in order:
        high = low + SPLIT_FRACTIONS[name]
        out[name] = (values >= low) & (values < high if high < 1 else values <= 1)
        low = high
    return out


def orthonormal_union(*parts: np.ndarray) -> np.ndarray:
    arrays = [np.asarray(part, dtype=np.float64) for part in parts if np.asarray(part).size]
    if not arrays:
        return np.zeros((256, 0), dtype=np.float64)
    q, r = np.linalg.qr(np.concatenate(arrays, axis=1))
    keep = np.abs(np.diag(r)) > 1e-10
    return np.ascontiguousarray(q[:, keep])


def _fit_at_ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    values, vectors = _design_eigen(design)
    cross = design.T @ target
    return vectors @ ((vectors.T @ cross) / (values[:, None] + float(ridge)))


def _design_eigen(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram = design.T @ design
    return np.linalg.eigh(0.5 * (gram + gram.T))


def _fit_from_eigen(
    eigen: tuple[np.ndarray, np.ndarray], cross: np.ndarray, ridge: float
) -> np.ndarray:
    values, vectors = eigen
    return vectors @ ((vectors.T @ cross) / (values[:, None] + float(ridge)))


def _energy(rows: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(rows, dtype=np.float64) ** 2, axis=1)


def _aggregate(per_lag: Mapping[str, Mapping[str, np.ndarray]], key: str) -> float:
    return float(np.mean([
        float(np.mean(entry[key])) / max(float(np.mean(entry["target_energy"])), 1e-300)
        for entry in per_lag.values()
    ]))


def _replicates(
    per_lag: Mapping[str, Mapping[str, np.ndarray]], key: str, weights: np.ndarray
) -> np.ndarray:
    values = []
    for entry in per_lag.values():
        numerator = weights @ np.asarray(entry[key], dtype=np.float64)
        denominator = weights @ np.asarray(entry["target_energy"], dtype=np.float64)
        values.append(numerator / np.maximum(denominator, 1e-300))
    return np.mean(np.stack(values), axis=0)


def matched_ridge_nested_score(
    frame: Frame,
    pool: Pool,
    base: np.ndarray,
    addition: np.ndarray,
    *,
    validation_mask: np.ndarray,
    score_mask: np.ndarray,
    ridge_grid: Sequence[float] = RIDGE_GRID,
) -> dict[str, object]:
    """Score base and augmented models at the same validation-selected ridge per lag."""

    fit = np.asarray(frame.fit_mask, dtype=bool)
    validation = np.asarray(validation_mask, dtype=bool)
    score = np.asarray(score_mask, dtype=bool)
    base = np.asarray(base, dtype=np.float64)
    addition = np.asarray(addition, dtype=np.float64)
    augmented = orthonormal_union(base, addition)
    zb, za = frame.source @ base, frame.source @ augmented
    aug_eigen = _design_eigen(za[fit])
    base_eigen = _design_eigen(zb[fit]) if base.shape[1] else None
    per_lag, ridges, weights_out = {}, {}, {}
    for lag in LAGS:
        residual = frame.residual[int(lag)]
        aug_cross = za[fit].T @ residual[fit]
        best_ridge, best_loss = float(ridge_grid[0]), np.inf
        for ridge in ridge_grid:
            w = _fit_from_eigen(aug_eigen, aug_cross, float(ridge))
            loss = float(np.mean(_energy(residual[validation] - za[validation] @ w)))
            if loss < best_loss:
                best_ridge, best_loss = float(ridge), loss
        wa = _fit_from_eigen(aug_eigen, aug_cross, best_ridge)
        if base.shape[1]:
            base_cross = zb[fit].T @ residual[fit]
            wb = _fit_from_eigen(base_eigen, base_cross, best_ridge)
            rb = residual[score] - zb[score] @ wb
        else:
            wb = np.zeros((0, residual.shape[1]), dtype=np.float64)
            rb = residual[score]
        ra = residual[score] - za[score] @ wa
        baseline_energy = _energy(residual[score])
        base_energy, augmented_energy = _energy(rb), _energy(ra)
        per_lag[str(lag)] = {
            "target_energy": _energy(frame.target[int(lag)][score]),
            "base_gain_numerator": baseline_energy - base_energy,
            "augmented_gain_numerator": baseline_energy - augmented_energy,
            "increment_numerator": base_energy - augmented_energy,
        }
        ridges[str(lag)] = best_ridge
        weights_out[str(lag)] = {"base": wb, "augmented": wa}
    return {
        "base_rank": int(base.shape[1]),
        "addition_rank": int(addition.shape[1]),
        "augmented_rank": int(augmented.shape[1]),
        "documents": int(score.sum()),
        "ridges": ridges,
        "per_lag": per_lag,
        "base_gain": _aggregate(per_lag, "base_gain_numerator"),
        "augmented_gain": _aggregate(per_lag, "augmented_gain_numerator"),
        "increment": _aggregate(per_lag, "increment_numerator"),
        "decoder_weights": weights_out,
    }


def score_projection(
    frame: Frame,
    pool: Pool,
    basis: np.ndarray,
    *,
    validation_mask: np.ndarray,
    score_mask: np.ndarray,
) -> dict[str, object]:
    return matched_ridge_nested_score(
        frame, pool, np.zeros((frame.source.shape[1], 0)), basis,
        validation_mask=validation_mask, score_mask=score_mask,
    )


def _cross_fitted_conditional_targets(
    frame: Frame, pool: Pool, accepted: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    fit = np.asarray(frame.fit_mask, dtype=bool)
    z = frame.source[fit] @ np.asarray(accepted, dtype=np.float64)
    targets, ridges = {}, {}
    for lag in LAGS:
        residual = frame.residual[int(lag)][fit]
        if z.shape[1]:
            solver = GroupedRidge(z, pool.documents[fit], seed=SEED, ridge_grid=RIDGE_GRID)
            ridge = solver.select(residual)
            prediction = solver.cv_predictions(residual, ridge)
        else:
            ridge, prediction = 0.0, np.zeros_like(residual)
        targets[int(lag)] = residual - prediction
        ridges[str(lag)] = float(ridge)
    return targets, ridges


def _cross_fitted_conditional_source(
    frame: Frame, pool: Pool, accepted: np.ndarray, complement: np.ndarray
) -> tuple[np.ndarray, float]:
    """Partial out accepted coordinates from the candidate source without leakage.

    W3 contains source coordinates that mix accepted shared state with private innovation.
    Euclidean projection removes accepted *directions* but not their correlated variance.
    The conditional operator must therefore receive the out-of-fold innovation of the
    complement coordinates, just as it receives out-of-fold conditional future targets.
    The returned eigenspace is still mapped into the original Euclidean complement.
    """

    fit = np.asarray(frame.fit_mask, dtype=bool)
    accepted_scores = frame.source[fit] @ np.asarray(accepted, dtype=np.float64)
    candidate_scores = frame.source[fit] @ np.asarray(complement, dtype=np.float64)
    if accepted_scores.shape[1] == 0:
        return candidate_scores, 0.0
    solver = GroupedRidge(
        accepted_scores, pool.documents[fit], seed=SEED, ridge_grid=RIDGE_GRID
    )
    ridge = solver.select(candidate_scores)
    prediction = solver.cv_predictions(candidate_scores, ridge)
    return candidate_scores - prediction, float(ridge)


def conditional_operator(
    frame: Frame,
    pool: Pool,
    accepted: np.ndarray,
    *,
    rho: float = ESTIMATOR_RHO,
) -> dict[str, object]:
    """R2.4's trace-normalized covariance-whitened operator on cross-fitted residuals."""

    accepted = orthonormal_union(accepted)
    complement = complement_basis(accepted, width=frame.source.shape[1])
    fit = np.asarray(frame.fit_mask, dtype=bool)
    source, source_decoder_ridge = _cross_fitted_conditional_source(
        frame, pool, accepted, complement
    )
    conditional, decoder_ridges = _cross_fitted_conditional_targets(frame, pool, accepted)
    n = max(int(fit.sum()) - 1, 1)
    sigma = source.T @ source / n
    eps_source = float(rho) * float(np.trace(sigma)) / sigma.shape[0]
    w0 = symmetric_inverse_sqrt(sigma, eps_source)
    operator = np.zeros_like(sigma)
    lag_traces = {}
    for lag in LAGS:
        target = conditional[int(lag)]
        cov = target.T @ target / n
        eps = float(rho) * float(np.trace(cov)) / cov.shape[0]
        wd = symmetric_inverse_sqrt(cov, eps)
        cross = source.T @ target / n
        left = w0 @ cross @ wd
        term = left @ left.T
        term = 0.5 * (term + term.T)
        trace = float(np.trace(term))
        lag_traces[str(lag)] = trace
        if trace > 0:
            operator += term / (len(LAGS) * trace)
    values, vectors = np.linalg.eigh(0.5 * (operator + operator.T))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return {
        "operator": operator,
        "eigenvalues": values,
        "eigenvectors": vectors,
        "source_whitener": w0,
        "complement": complement,
        "conditional_decoder_ridges": decoder_ridges,
        "conditional_source_decoder_ridge": source_decoder_ridge,
        "lag_traces": lag_traces,
        "rho": float(rho),
        "fit_documents": int(fit.sum()),
        "remaining_dimension": int(complement.shape[1]),
    }


def candidate_from_operator(record: Mapping[str, object], rank: int) -> np.ndarray:
    vectors = np.asarray(record["eigenvectors"])[:, : int(rank)]
    inner = np.linalg.qr(np.asarray(record["source_whitener"]) @ vectors)[0]
    candidate = np.asarray(record["complement"]) @ inner
    return np.ascontiguousarray(np.linalg.qr(candidate)[0][:, : int(rank)])


def spectral_gaps(values: np.ndarray, ranks: Sequence[int] = CANDIDATE_RANKS) -> dict[int, dict]:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    all_gaps = np.log((values[:-1] + EIGEN_EPSILON) / (values[1:] + EIGEN_EPSILON))
    out = {}
    for rank in ranks:
        index = int(rank) - 1
        neighbors = [i for i in range(max(0, index - 4), min(len(all_gaps), index + 5)) if i != index]
        local = float(all_gaps[index] - np.median(all_gaps[neighbors])) if neighbors else float("nan")
        out[int(rank)] = {
            "gap": float(all_gaps[index]),
            "local_excess_gap": local,
            "lambda_r": float(values[index]),
            "lambda_next": float(values[index + 1]),
        }
    return out


def select_rank(
    frame: Frame,
    pool: Pool,
    accepted: np.ndarray,
    operator: Mapping[str, object],
    *,
    validation_mask: np.ndarray,
    max_rank: int | None = None,
    gap_envelopes: Mapping[str, float] | None = None,
) -> dict[str, object]:
    gaps = spectral_gaps(np.asarray(operator["eigenvalues"]))
    significant_boundaries = sorted(
        rank for rank in CANDIDATE_RANKS
        if gap_envelopes is not None and str(rank) in gap_envelopes
        and float(gaps[rank]["gap"]) > float(gap_envelopes[str(rank)])
    )
    structural_cap = significant_boundaries[0] if significant_boundaries else None
    gains, bases = {}, {}
    for rank in CANDIDATE_RANKS:
        if int(rank) > int(operator["remaining_dimension"]) or (
            max_rank is not None and int(rank) > int(max_rank)
        ):
            continue
        basis = candidate_from_operator(operator, rank)
        record = matched_ridge_nested_score(
            frame, pool, accepted, basis,
            validation_mask=validation_mask, score_mask=validation_mask,
        )
        gains[int(rank)] = float(record["increment"])
        bases[int(rank)] = basis
    if not gains:
        raise RuntimeError("no candidate rank fits the remaining rank budget")
    best = max(gains.values())
    rank_pool = [rank for rank in sorted(gains) if structural_cap is None or rank <= structural_cap]
    capped_best = max(gains[rank] for rank in rank_pool)
    eligible = [rank for rank in rank_pool if gains[rank] >= RANK_GAIN_FRACTION * capped_best]
    selected = int(eligible[0] if eligible else max(gains, key=gains.get))
    return {
        "selected_rank": selected,
        "validation_increments": gains,
        "maximum_validation_increment": float(best),
        "maximum_validation_increment_within_boundary": float(capped_best),
        "structural_rank_cap": structural_cap,
        "selection_rule": (
            "do not cross the earliest W5-significant spectral boundary; within that block, "
            "take the smallest rank reaching 90% of maximum validation conditional gain"
        ),
        "basis": bases[selected],
        "gaps": gaps,
    }


def random_complement_bases(
    accepted: np.ndarray,
    rank: int,
    *,
    draws: int = RANDOM_DRAWS,
    namespace: str,
    seed: int = SEED,
) -> list[np.ndarray]:
    complement = complement_basis(accepted, width=accepted.shape[0])
    out = []
    for index in range(int(draws)):
        rng = np.random.default_rng(namespaced_seed(namespace, seed, index))
        inner = np.linalg.qr(rng.normal(size=(complement.shape[1], int(rank))))[0]
        basis = np.ascontiguousarray(complement @ inner[:, : int(rank)])
        if float(np.max(np.abs(accepted.T @ basis))) > ORTHOGONALITY_ATOL:
            raise RuntimeError("random control escaped the current complement")
        out.append(basis)
    return out


def evaluate_l1_step(
    frame: Frame,
    pool: Pool,
    accepted: np.ndarray,
    candidate: np.ndarray,
    *,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    random_bases: Sequence[np.ndarray],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, object]:
    real = matched_ridge_nested_score(
        frame, pool, accepted, candidate,
        validation_mask=validation_mask, score_mask=test_mask,
    )
    controls = [
        matched_ridge_nested_score(
            frame, pool, accepted, basis,
            validation_mask=validation_mask, score_mask=test_mask,
        )
        for basis in random_bases
    ]
    random_points = np.array([float(record["increment"]) for record in controls])
    floor = float(np.quantile(random_points, 0.95))
    weights = bootstrap_weights(int(test_mask.sum()), int(bootstrap_replicates), seed=seed)
    real_reps = _replicates(real["per_lag"], "increment_numerator", weights)
    random_reps = np.stack([
        _replicates(record["per_lag"], "increment_numerator", weights)
        for record in controls
    ])
    t_reps = real_reps - np.quantile(random_reps, 0.95, axis=0)
    return {
        "delta_mr": float(real["increment"]),
        "base_gain_mr": float(real["base_gain"]),
        "augmented_gain_mr": float(real["augmented_gain"]),
        "random_increment": random_points,
        "random_floor_q95": floor,
        "t": float(real["increment"] - floor),
        "t_bootstrap": t_reps,
        "t_lcb95": float(np.quantile(t_reps, ALPHA)),
        "t_median": float(np.quantile(t_reps, 0.50)),
        "t_ucb95": float(np.quantile(t_reps, 0.95)),
        "real_record": real,
        "control_records": controls,
    }


def cumulative_msc(left: np.ndarray, right: np.ndarray) -> float:
    return mean_subspace_correlation(orthonormal_union(left), orthonormal_union(right))


def replay_path(
    pool: Pool,
    q1: np.ndarray,
    q2: np.ndarray,
    ranks: Sequence[int],
    *,
    fit_mask: np.ndarray,
    rho: float = ESTIMATOR_RHO,
) -> list[dict[str, object]]:
    frame = fit_frame(pool, fit_mask, rho=rho)
    accepted = orthonormal_union(q1, q2)
    out = []
    for step, rank in enumerate(ranks, start=3):
        operator = conditional_operator(frame, pool, accepted, rho=rho)
        candidate = candidate_from_operator(operator, int(rank))
        accepted = orthonormal_union(accepted, candidate)
        out.append({
            "step": step,
            "rank": int(rank),
            "candidate": candidate,
            "union": accepted,
            "eigenvalues": np.asarray(operator["eigenvalues"]),
            "gaps": spectral_gaps(np.asarray(operator["eigenvalues"])),
        })
    return out


def random_full_refit_null(
    q1: np.ndarray,
    q2: np.ndarray,
    ranks: Sequence[int],
    *,
    draws: int = STABILITY_NULL_DRAWS,
) -> dict[int, np.ndarray]:
    seed_union = orthonormal_union(q1, q2)
    output = {step: [] for step in range(3, 3 + len(ranks))}
    for draw in range(int(draws)):
        unions = []
        for half in ("A", "B"):
            accepted = seed_union
            path = []
            for step, rank in enumerate(ranks, start=3):
                basis = random_complement_bases(
                    accepted, int(rank), draws=1,
                    namespace=f"r26_stability_null_{half}_{draw}_{step}",
                )[0]
                accepted = orthonormal_union(accepted, basis)
                path.append(accepted)
            unions.append(path)
        for step_index, step in enumerate(output):
            output[step].append(cumulative_msc(unions[0][step_index], unions[1][step_index]))
    return {step: np.asarray(values) for step, values in output.items()}


def ridge_robustness(
    pool: Pool,
    q1: np.ndarray,
    q2: np.ndarray,
    ranks: Sequence[int],
    primary_path: Sequence[Mapping[str, object]],
    *,
    fit_mask: np.ndarray,
    rhos: Sequence[float],
) -> list[dict[str, object]]:
    rows = []
    for rho in rhos:
        path = replay_path(pool, q1, q2, ranks, fit_mask=fit_mask, rho=float(rho))
        for primary, alternate in zip(primary_path, path):
            rank = int(primary["rank"])
            primary_gap = float(primary["gaps"][rank]["gap"])
            alternate_gap = float(alternate["gaps"][rank]["gap"])
            union_msc = cumulative_msc(primary["union"], alternate["union"])
            gap_ratio = alternate_gap / max(primary_gap, EIGEN_EPSILON)
            rows.append({
                "rho": float(rho),
                "step": int(primary["step"]),
                "rank": rank,
                "cumulative_union_msc": union_msc,
                "gap": alternate_gap,
                "gap_ratio": gap_ratio,
                # L1 is a predictive-component claim and therefore uses the
                # prespecified cumulative-union robustness quantity.  Spectral
                # boundary persistence is a distinct L2 pocket requirement.
                "passes": bool(union_msc >= 0.50),
                "gap_passes": bool(
                    alternate_gap > 0 and gap_ratio >= RIDGE_GAP_RATIO_MIN
                ),
            })
    return rows


def lag_gains(record: Mapping[str, object], key: str = "augmented_gain_numerator") -> np.ndarray:
    return np.array([
        float(np.mean(record["per_lag"][str(lag)][key])) /
        max(float(np.mean(record["per_lag"][str(lag)]["target_energy"])), 1e-300)
        for lag in LAGS
    ])


def l1_pass(record: Mapping[str, object], practical_effect: float, *, stability: bool, ridge: bool) -> bool:
    return bool(
        stability and ridge and float(record["t_lcb95"]) > 0
        and float(record["delta_mr"]) >= float(practical_effect)
    )
