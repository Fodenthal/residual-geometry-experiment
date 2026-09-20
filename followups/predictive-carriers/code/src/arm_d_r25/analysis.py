"""Core R2.5 carrier correction, conditional scoring, and paired inference."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

from arm_d_r24.data import Pool, document_hash_fraction
from arm_d_r24.estimators import carrier_family
from arm_d_r24.geometry import complement_basis
from arm_d_r24.scoring import (
    document_codes,
    fit_frame,
    gain_replicates,
    per_document_energy,
    to_aperture,
)
from arm_d_r4.common_target import GroupedRidge, bootstrap_weights

from .contract import (
    BOOTSTRAP_REPLICATES,
    LAGS,
    ORTHOGONALITY_ATOL,
    Q2_RANK,
    RANDOM_DRAWS,
    RHO,
    SEED,
    SPLIT_FRACTIONS,
    SPLIT_SALT,
)


def three_way_split(documents, *, salt: str = SPLIT_SALT) -> dict[str, np.ndarray]:
    values = np.array([document_hash_fraction(str(doc), salt) for doc in documents])
    train_edge = SPLIT_FRACTIONS["train"]
    validation_edge = train_edge + SPLIT_FRACTIONS["validation"]
    return {
        "train": values < train_edge,
        "validation": (values >= train_edge) & (values < validation_edge),
        "test": values >= validation_edge,
    }


def regenerate_q2(pool: Pool, q1: np.ndarray, *, rho: float = RHO) -> tuple[np.ndarray, dict]:
    """Run R2.4 C6's estimator after deleting the intended Q_E(16)."""

    q1 = np.asarray(q1, dtype=np.float64)
    fit = pool.mask("fitA", "fitB")
    surviving = complement_basis(q1, width=pool.offsets[0].shape[1])
    projected = fit_frame(pool, fit, rho=rho, projection=surviving)
    family = carrier_family(projected, metric="whitened", ranks=(Q2_RANK,))
    q2 = to_aperture(projected, family["bases"][Q2_RANK])
    overlap = q1.T @ q2
    orthogonality = float(np.max(np.abs(overlap)))
    if orthogonality > ORTHOGONALITY_ATOL:
        raise RuntimeError(f"corrected Q2 is not orthogonal to Q1: {orthogonality}")
    return q2, {
        "estimator": "R2.4 C6 covariance-whitened multi-horizon estimator",
        "deleted_carrier": "Q_E(16)",
        "rank": Q2_RANK,
        "rho": float(rho),
        "max_abs_q1_t_q2": orthogonality,
        "q2_orthonormality_error": float(np.max(np.abs(q2.T @ q2 - np.eye(Q2_RANK)))),
        "fit_documents": int(fit.sum()),
    }


def frozen_random_bases(
    deleted: np.ndarray, *, rank: int, draws: int, namespace: str, seed: int = SEED
) -> list[np.ndarray]:
    deleted = np.asarray(deleted, dtype=np.float64)
    surviving = complement_basis(deleted, width=deleted.shape[0])
    out = []
    for index in range(int(draws)):
        digest = hashlib.sha256(f"{namespace}:{seed}:{index}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        inner = rng.normal(size=(surviving.shape[1], int(rank)))
        inner = np.linalg.qr(inner)[0][:, : int(rank)]
        basis = surviving @ inner
        if np.max(np.abs(deleted.T @ basis)) > ORTHOGONALITY_ATOL:
            raise RuntimeError("random control escaped the requested complement")
        out.append(np.ascontiguousarray(basis))
    return out


def _gain_replicates(record: Mapping[str, object], weights: np.ndarray) -> np.ndarray:
    per_lag = []
    for entry in record["per_lag"].values():
        per_lag.append(
            gain_replicates(
                entry["augmented_residual"],
                entry["baseline_residual"],
                entry["target_energy"],
                weights,
            )
        )
    return np.mean(np.stack(per_lag), axis=0)


def _point_gain(record: Mapping[str, object]) -> float:
    values = []
    for entry in record["per_lag"].values():
        numerator = np.mean(entry["baseline_residual"] - entry["augmented_residual"])
        denominator = np.mean(entry["target_energy"])
        values.append(float(numerator / denominator))
    return float(np.mean(values))


def _score(frame, pool, mask, basis):
    """The frozen primary augmented path only.

    R2.5's sole statistic uses ``gamma_augmented``.  R2.4's scorer also fits raw and
    destination-whitened co-reported paths; fitting those for 400 control models would
    triple qualification cost without entering any R2.5 quantity.  This is the identical
    grouped-CV residual decoder and per-document energy calculation for the one inherited
    path R2.5 actually freezes.
    """
    fit = frame.fit_mask
    score = np.asarray(mask, dtype=bool)
    z = frame.source @ np.asarray(basis, dtype=np.float64)
    solver = GroupedRidge(z[fit], pool.documents[fit], seed=SEED)
    codes, n_documents = document_codes(pool.documents[score])
    per_lag, ridges, decoders = {}, {}, {}
    for lag in LAGS:
        target = frame.target[int(lag)]
        residual = frame.residual[int(lag)]
        decoder, ridge = solver.select_and_fit(residual[fit])
        remaining = residual[score] - z[score] @ decoder
        per_lag[str(lag)] = {
            "lag": int(lag),
            "target_energy": per_document_energy(target[score], codes, n_documents),
            "baseline_residual": per_document_energy(residual[score], codes, n_documents),
            "augmented_residual": per_document_energy(remaining, codes, n_documents),
        }
        ridges[str(lag)] = {"residual": float(ridge)}
        decoders[str(lag)] = np.ascontiguousarray(decoder)
    return {
        "dimension": int(np.asarray(basis).shape[1]),
        "n_documents": n_documents,
        "per_lag": per_lag,
        "decoder_ridges": ridges,
        "decoder_weights": decoders,
    }


def prediction_diagnostics(frame, pool, mask, q1, q2, q1_record, q2_record) -> dict:
    fit = frame.fit_mask
    z1, z2 = frame.source @ q1, frame.source @ q2

    def cross_r2(source, target):
        solver = GroupedRidge(source[fit], pool.documents[fit], seed=SEED)
        weights, _ = solver.select_and_fit(target[fit])
        residual = target[mask] - source[mask] @ weights
        denom = float(np.sum((target[mask] - target[mask].mean(0)) ** 2))
        return float(1.0 - np.sum(residual**2) / denom)

    def canonical():
        a, b = z1[mask], z2[mask]
        a = a - a.mean(0); b = b - b.mean(0)
        ca = a.T @ a / max(len(a) - 1, 1)
        cb = b.T @ b / max(len(b) - 1, 1)
        cab = a.T @ b / max(len(a) - 1, 1)
        def invsqrt(c):
            v, u = np.linalg.eigh((c + c.T) / 2)
            return (u * np.clip(v, 1e-10, None) ** -0.5) @ u.T
        return np.linalg.svd(invsqrt(ca) @ cab @ invsqrt(cb), compute_uv=False)[:16]

    per_lag = {}
    prediction_1, prediction_2, targets = [], [], []
    for lag in LAGS:
        key = str(lag)
        baseline = frame.baseline_prediction(lag)[mask]
        p1 = baseline + z1[mask] @ q1_record["decoder_weights"][key]
        p2 = baseline + z2[mask] @ q2_record["decoder_weights"][key]
        target = frame.target[lag][mask]
        disagreement = float(np.mean(np.sum((p1 - p2) ** 2, axis=1)))
        energy = float(np.mean(np.sum(target**2, axis=1)))
        cosine = float(np.sum(p1 * p2) / max(np.linalg.norm(p1) * np.linalg.norm(p2), 1e-300))
        per_lag[key] = {"normalized_prediction_disagreement": disagreement / energy,
                        "prediction_cosine": cosine}
        prediction_1.append(p1); prediction_2.append(p2); targets.append(target)
    p1 = np.concatenate(prediction_1, axis=1)
    p2 = np.concatenate(prediction_2, axis=1)
    target = np.concatenate(targets, axis=1)
    p1c, p2c = p1 - p1.mean(0), p2 - p2.mean(0)
    gram1, gram2 = p1c @ p1c.T, p2c @ p2c.T
    cka = float(np.sum(gram1 * gram2) / max(np.linalg.norm(gram1) * np.linalg.norm(gram2), 1e-300))
    return {
        "cross_decode_q2_from_q1_r2": cross_r2(z1, z2),
        "cross_decode_q1_from_q2_r2": cross_r2(z2, z1),
        "canonical_correlations": canonical().tolist(),
        "normalized_prediction_disagreement": float(np.sum((p1 - p2) ** 2) / np.sum(target**2)),
        "prediction_cosine": float(np.sum(p1 * p2) / max(np.linalg.norm(p1) * np.linalg.norm(p2), 1e-300)),
        "linear_cka": cka,
        "per_lag": per_lag,
    }


def analyze(
    pool: Pool,
    q1: np.ndarray,
    q2: np.ndarray,
    pca64: np.ndarray,
    *,
    score_split: str,
    random_draws: int = RANDOM_DRAWS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
    include_secondary: bool = True,
    include_diagnostics: bool = True,
) -> dict:
    train = pool.splits["train"]
    score = pool.splits[score_split]
    frame = fit_frame(pool, train, rho=RHO)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    if np.max(np.abs(q1.T @ q2)) > ORTHOGONALITY_ATOL:
        raise RuntimeError("Q1 and Q2 are not orthogonal")

    random_q2 = frozen_random_bases(
        q1, rank=q2.shape[1], draws=random_draws, namespace="r2_5_q1_complement", seed=seed
    )
    random_q1 = (
        frozen_random_bases(
            q2, rank=q1.shape[1], draws=random_draws,
            namespace="r2_5_q2_complement", seed=seed,
        )
        if include_secondary else []
    )
    records = {
        "q1": _score(frame, pool, score, q1),
        "q2": _score(frame, pool, score, q2),
        "joint": _score(frame, pool, score, np.concatenate([q1, q2], axis=1)),
        "pca64": _score(frame, pool, score, pca64),
    }
    primary_random = [
        _score(frame, pool, score, np.concatenate([q1, basis], axis=1))
        for basis in random_q2
    ]
    symmetric_random = [
        _score(frame, pool, score, np.concatenate([q2, basis], axis=1))
        for basis in random_q1
    ]

    n = int(score.sum())
    weights = bootstrap_weights(n, bootstrap_replicates, seed=seed)
    point = {name: _point_gain(record) for name, record in records.items()}
    reps = {name: _gain_replicates(record, weights) for name, record in records.items()}
    random_points = np.array([_point_gain(r) - point["q1"] for r in primary_random])
    random_reps = np.stack(
        [_gain_replicates(r, weights) - reps["q1"] for r in primary_random], axis=0
    )
    symmetric_points = np.array([_point_gain(r) - point["q2"] for r in symmetric_random])
    delta = point["joint"] - point["q1"]
    reverse_delta = point["joint"] - point["q2"]
    floor = float(np.quantile(random_points, 0.95))
    t_module = float(delta - floor)
    t_reps = (reps["joint"] - reps["q1"]) - np.quantile(random_reps, 0.95, axis=0)
    return {
        "score_split": score_split,
        "documents": n,
        "gains": point,
        "delta_2_given_1": float(delta),
        "delta_1_given_2": float(reverse_delta),
        "random_increment": random_points,
        "random_floor_q95": floor,
        "symmetric_random_increment": symmetric_points,
        "symmetric_floor_q95": (
            float(np.quantile(symmetric_points, 0.95)) if symmetric_points.size else None
        ),
        "t_module": t_module,
        "t_module_bootstrap": t_reps,
        "t_module_lcb95": float(np.quantile(t_reps, 0.05)),
        "t_module_median": float(np.quantile(t_reps, 0.50)),
        "t_module_ucb95": float(np.quantile(t_reps, 0.95)),
        "diagnostics": (
            prediction_diagnostics(frame, pool, score, q1, q2, records["q1"], records["q2"])
            if include_diagnostics else None
        ),
        "decoder_ridges": {name: record["decoder_ridges"] for name, record in records.items()},
        "random_bases": random_q2,
        "symmetric_random_bases": random_q1,
        "model_weights": {
            "primary": {name: record["decoder_weights"] for name, record in records.items()},
            "conditional_null": [record["decoder_weights"] for record in primary_random],
            "symmetric_null": [record["decoder_weights"] for record in symmetric_random],
        },
    }


def assign_label(result: Mapping[str, object], m_module: float) -> str:
    delta = float(result["delta_2_given_1"])
    t = float(result["t_module"])
    lcb = float(result["t_module_lcb95"])
    gamma2 = float(result["gains"]["q2"])
    if lcb > 0 and delta >= float(m_module):
        return "M1_ORTHOGONAL_CARRIER_ADDS_UNIQUE_PREDICTIVE_STATE"
    if gamma2 > 0 and delta < float(m_module) and t <= 0:
        return "R1_CARRIERS_PREDICTIVELY_REDUNDANT_AT_TESTED_RESOLUTION"
    if t <= 0:
        return "D1_COMPLEMENT_STRUCTURE_NOT_SPECIAL_BEYOND_DIMENSION"
    return "X1_SHARED_AND_PRIVATE_PREDICTIVE_STRUCTURE"
