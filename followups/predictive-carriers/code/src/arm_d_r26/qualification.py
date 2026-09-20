"""Known-answer worlds and the frozen sequential qualification runner."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from arm_d_r24.data import Pool
from arm_d_r24.geometry import complement_basis
from arm_d_r24.scoring import fit_frame

from .contract import (
    BOOTSTRAP_REPLICATES,
    CANDIDATE_RANKS,
    ESTIMATOR_RHO,
    LAGS,
    PRACTICAL_EFFECT,
    RANDOM_DRAWS,
    RIDGE_RHOS,
    SEED,
    WIDTH,
    SYNTHETIC_EFFECT_CALIBRATION,
)
from .discovery import (
    conditional_operator,
    cumulative_msc,
    evaluate_l1_step,
    four_way_split,
    lag_gains,
    l1_pass,
    matched_ridge_nested_score,
    orthonormal_union,
    random_complement_bases,
    random_full_refit_null,
    replay_path,
    ridge_robustness,
    score_projection,
    select_rank,
)


def _unique_lag_profiles(
    frame, pool: Pool, components: list[np.ndarray], *, validation_mask, score_mask
) -> list[np.ndarray]:
    """Exact leave-one-out lag profiles for one fixed component family."""

    full = score_projection(
        frame, pool, orthonormal_union(*components),
        validation_mask=validation_mask, score_mask=score_mask,
    )
    full_profile = lag_gains(full)
    profiles = []
    for index in range(len(components)):
        remainder = [component for j, component in enumerate(components) if j != index]
        without = score_projection(
            frame, pool, orthonormal_union(*remainder),
            validation_mask=validation_mask, score_mask=score_mask,
        )
        profiles.append(full_profile - lag_gains(without))
    return profiles


def _shape_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(1.0 - (left @ right) / max(denominator, 1e-300))


def _effect_scaled_coefficients(
    shape: np.ndarray, variances: np.ndarray, effect: float, seed_signal_energy: float
) -> np.ndarray:
    shape = np.asarray(shape, dtype=np.float64)
    current = float(np.sum(variances[: len(shape)] * shape**2))
    target = float(effect) * (WIDTH + seed_signal_energy) / max(1.0 - float(effect), 1e-9)
    return shape * np.sqrt(target / max(current, 1e-300))


def _shared_private_coefficients(
    shape: np.ndarray,
    shared_variances: np.ndarray,
    private_variances: np.ndarray,
    sharing: float,
    effect: float,
    seed_signal_energy: float,
    tested_rank: int | None = None,
) -> np.ndarray:
    """Plant an exact population gain in the candidate the instrument tests."""

    shape = np.asarray(shape, dtype=np.float64)
    private_variances = np.asarray(private_variances, dtype=np.float64)[: len(shape)]
    shared_variances = np.asarray(shared_variances, dtype=np.float64)
    innovation = private_variances.copy()
    total = private_variances.copy()
    mixed = min(32, len(shape), len(shared_variances))
    rho2 = float(sharing) ** 2
    innovation[:mixed] *= 1.0 - rho2
    # synthetic_pool variance-standardizes the shared factor before mixing, so
    # every mixed coordinate retains its spent-regime private variance while its
    # private innovation contains the prespecified (1-rho^2) fraction.
    total[:mixed] = private_variances[:mixed]
    tested = len(shape) if tested_rank is None else int(tested_rank)
    if tested <= 0 or tested > len(shape):
        raise ValueError(f"invalid tested rank {tested} for shape of length {len(shape)}")
    private_energy = float(np.sum(innovation[:tested] * shape[:tested] ** 2))
    total_energy = float(np.sum(total * shape**2))
    base_energy = WIDTH + float(seed_signal_energy)
    denominator = private_energy - float(effect) * total_energy
    if denominator <= 0:
        raise ValueError(
            f"requested private effect {effect} exceeds the shared geometry's attainable gain"
        )
    scale_squared = float(effect) * base_energy / denominator
    return shape * np.sqrt(scale_squared)


def _smooth_shape(family: str, size: int = 64) -> np.ndarray:
    index = np.arange(1, size + 1, dtype=np.float64)
    if family == "exponential":
        return np.exp(-index / 18.0)
    if family == "power_law":
        return index ** -0.55
    if family == "curved_log":
        return np.exp(-0.028 * index - 0.00055 * index**2)
    if family == "broken_slope":
        return np.exp(-0.025 * np.minimum(index, 24) - 0.06 * np.maximum(index - 24, 0))
    raise ValueError(f"unknown W5 family {family}")


def synthetic_pool(
    world: str,
    *,
    documents: int,
    seed: int,
    sharing: float = 0.8,
    effect_multiplier: float = 1.0,
    discrete: bool = True,
    w5_family: str = "exponential",
    regime: Mapping[str, object] | None = None,
) -> tuple[Pool, np.ndarray, np.ndarray]:
    """Generate a 256D, seven-lag world with empirical-like source anisotropy."""

    rng = np.random.default_rng(int(seed))
    n, width = int(documents), WIDTH
    if regime and "source_covariance_eigenvalues" in regime:
        variances = np.asarray(regime["source_covariance_eigenvalues"], dtype=np.float64)[:width]
        variances = np.clip(
            variances / max(float(np.mean(variances)), 1e-300), 1e-4, None
        )
    else:
        variances = np.exp(np.linspace(np.log(2.0), np.log(0.30), width))
    x = rng.normal(size=(n, width)) * np.sqrt(variances)
    q1, q2 = np.eye(width)[:, :16], np.eye(width)[:, 16:48]
    if world in {"W3", "W4"}:
        shared = x[:, :32]
        shared_scale = np.sqrt(
            variances[48:80] / np.maximum(variances[:32], 1e-300)
        )
        x[:, 48:80] = (
            float(sharing) * shared * shared_scale
            + np.sqrt(max(1.0 - float(sharing) ** 2, 1e-6))
            * rng.normal(size=(n, 32)) * np.sqrt(variances[48:80])
        )
    update = rng.normal(scale=0.6, size=(n, width))
    offsets = {-1: x - update, 0: x}
    seed_gain = float((regime or {}).get("r2_5_gains", {}).get("joint", 0.20903624))
    seed_coeff = np.sqrt(
        seed_gain * width /
        max((1.0 - seed_gain) * float(np.sum(variances[:16])), 1e-300)
    )
    seed_energy = float(np.sum(variances[:16] * seed_coeff**2))
    target_mixer = np.linalg.qr(rng.normal(size=(width, width)))[0]
    for lag_index, lag in enumerate(LAGS):
        target = rng.normal(size=(n, width))
        target += (x[:, :16] * seed_coeff) @ target_mixer[:16]
        beta = np.zeros(width - 48, dtype=np.float64)
        if world == "W1":
            beta[:8] = _effect_scaled_coefficients(
                np.ones(8), variances[48:56],
                SYNTHETIC_EFFECT_CALIBRATION * PRACTICAL_EFFECT * effect_multiplier, seed_energy
            )
        elif world == "W2":
            beta[:8] = _effect_scaled_coefficients(
                np.ones(8), variances[48:56],
                SYNTHETIC_EFFECT_CALIBRATION * 2.0 * PRACTICAL_EFFECT, seed_energy
            )
            beta[8:16] = _effect_scaled_coefficients(
                np.ones(8), variances[56:64],
                SYNTHETIC_EFFECT_CALIBRATION * PRACTICAL_EFFECT, seed_energy
            )
            # Different lag shapes make the two planted blocks functionally differentiated.
            beta[:8] *= 1.25 - 0.06 * lag_index
            beta[8:16] *= 0.75 + 0.08 * lag_index
        elif world == "W3":
            if discrete:
                shape = np.r_[np.ones(8), np.zeros(56)]
            else:
                shape = _smooth_shape("curved_log")
            beta[:64] = _shared_private_coefficients(
                shape, variances[:32], variances[48:112], float(sharing),
                SYNTHETIC_EFFECT_CALIBRATION * PRACTICAL_EFFECT * effect_multiplier,
                seed_energy, tested_rank=8 if discrete else 16,
            )
        elif world == "W5":
            beta[:64] = _effect_scaled_coefficients(
                _smooth_shape(w5_family), variances[48:112], 4.0 * PRACTICAL_EFFECT, seed_energy
            )
        # W0/W6 have no remaining predictive signal.  W4 has a strong correlated
        # complement realization, but the target depends only on the accepted seed state.
        if np.any(beta):
            target += (x[:, 48:] * beta) @ target_mixer[48:]
        offsets[int(lag)] = target
    ids = np.asarray([f"{world}-{seed}-{i:06d}" for i in range(n)])
    pool = Pool(
        name=f"qualification-{world}", documents=ids, offsets=offsets,
        splits=four_way_split(ids), provenance={
            "synthetic": True, "world": world, "seed": int(seed),
            "sharing": float(sharing), "effect_multiplier": float(effect_multiplier),
            "discrete": bool(discrete), "w5_family": w5_family,
        },
    )
    return pool, q1, q2


def _boundary_reproduces(
    path_a, path_b, index: int, rank: int, gap_envelopes: Mapping[str, float] | None
) -> bool:
    gaps_a = path_a[index]["gaps"]
    gaps_b = path_b[index]["gaps"]
    if gap_envelopes is None:
        return False
    significant_a = [
        r for r in CANDIDATE_RANKS if r in gaps_a and str(r) in gap_envelopes
        and float(gaps_a[r]["gap"]) > float(gap_envelopes[str(r)])
    ]
    significant_b = [
        r for r in CANDIDATE_RANKS if r in gaps_b and str(r) in gap_envelopes
        and float(gaps_b[r]["gap"]) > float(gap_envelopes[str(r)])
    ]
    return bool(
        significant_a and significant_b
        and min(significant_a) == int(rank) and min(significant_b) == int(rank)
    )


def run_qualification_case(
    world: str,
    *,
    documents: int,
    seed: int,
    gap_envelopes: Mapping[str, float] | None,
    profile_envelope: float | None,
    sharing: float = 0.8,
    effect_multiplier: float = 1.0,
    discrete: bool = True,
    w5_family: str = "exponential",
    calibration_only: bool = False,
    regime: Mapping[str, object] | None = None,
) -> dict[str, object]:
    pool, q1, q2 = synthetic_pool(
        world, documents=documents, seed=seed, sharing=sharing,
        effect_multiplier=effect_multiplier, discrete=discrete, w5_family=w5_family,
        regime=regime,
    )
    fit_mask = pool.mask("fitA", "fitB")
    validation, test = pool.splits["validation"], pool.splits["test"]
    frame = fit_frame(pool, fit_mask, rho=ESTIMATOR_RHO)
    accepted = orthonormal_union(q1, q2)
    combined_path, ranks = [], []
    for step in range(3, 7):
        operator = conditional_operator(frame, pool, accepted)
        selected = select_rank(
            frame, pool, accepted, operator, validation_mask=validation,
            max_rank=128 - accepted.shape[1] - 4 * (6 - step),
            gap_envelopes=gap_envelopes,
        )
        candidate = selected["basis"]
        accepted = orthonormal_union(accepted, candidate)
        ranks.append(int(selected["selected_rank"]))
        combined_path.append({
            "step": step, "rank": int(selected["selected_rank"]),
            "candidate": candidate, "union": accepted,
            "eigenvalues": np.asarray(operator["eigenvalues"]),
            "gaps": selected["gaps"],
            "validation_increments": selected["validation_increments"],
        })
    if calibration_only:
        profiles, profile_eligible, steps = [], [], []
        accepted = orthonormal_union(q1, q2)
        for item in combined_path:
            record = matched_ridge_nested_score(
                frame, pool, accepted, item["candidate"],
                validation_mask=validation, score_mask=validation,
            )
            profile = lag_gains(record, "increment_numerator")
            profiles.append(profile)
            profile_eligible.append(float(record["increment"]) >= PRACTICAL_EFFECT)
            steps.append({
                "step": int(item["step"]), "rank": int(item["rank"]),
                "all_rank_gaps": {
                    str(k): float(v["gap"]) for k, v in item["gaps"].items()
                },
                "profile": profile.tolist(),
            })
            accepted = orthonormal_union(accepted, item["candidate"])
        distances = []
        for left in range(len(profiles)):
            for right in range(left + 1, len(profiles)):
                a, b = profiles[left], profiles[right]
                if profile_eligible[left] and profile_eligible[right] and np.linalg.norm(a) and np.linalg.norm(b):
                    distances.append(float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        return {
            "world": world, "documents": int(documents), "seed": int(seed),
            "sharing": float(sharing), "effect_multiplier": float(effect_multiplier),
            "discrete": bool(discrete), "w5_family": w5_family,
            "calibration_only": True, "steps": steps,
            "profile_distances": distances,
        }
    path_a = replay_path(pool, q1, q2, ranks, fit_mask=pool.splits["fitA"])
    path_b = replay_path(pool, q1, q2, ranks, fit_mask=pool.splits["fitB"])
    null = random_full_refit_null(q1, q2, ranks)
    ridge_rows = ridge_robustness(
        pool, q1, q2, ranks, combined_path, fit_mask=fit_mask, rhos=RIDGE_RHOS,
    )
    steps = []
    accepted_components = [q1, q2]
    accepted = orthonormal_union(q1, q2)
    stopped = False
    for index, item in enumerate(combined_path):
        step, rank, candidate = int(item["step"]), int(item["rank"]), item["candidate"]
        if stopped:
            steps.append({
                "step": step, "rank": rank, "tested": False,
                "delta_mr": None, "random_floor_q95": None, "t": None,
                "t_lcb95": None, "stability_msc": None,
                "stability_null_q95": None, "stability_pass": None,
                "ridge_pass": None, "gap": float(item["gaps"][rank]["gap"]),
                "gap_threshold": float((gap_envelopes or {}).get(str(rank), np.inf)),
                "boundary_reproduces": None, "l1": False, "l2": False,
                "all_rank_gaps": {
                    str(k): float(v["gap"]) for k, v in item["gaps"].items()
                },
                "profile": None,
            })
            continue
        real_msc = cumulative_msc(path_a[index]["union"], path_b[index]["union"])
        null_q95 = float(np.quantile(null[step], 0.95))
        stability = bool(real_msc > null_q95 and real_msc >= 0.50)
        ridge_ok = all(
            row["passes"] for row in ridge_rows if int(row["step"]) == step
        )
        ridge_boundary_ok = all(
            row["gap_passes"] for row in ridge_rows if int(row["step"]) == step
        )
        controls = random_complement_bases(
            accepted, rank, draws=RANDOM_DRAWS,
            namespace=f"r26_qualification_{world}_{seed}_step{step}", seed=seed,
        )
        evaluation = evaluate_l1_step(
            frame, pool, accepted, candidate, validation_mask=validation, test_mask=test,
            random_bases=controls, bootstrap_replicates=BOOTSTRAP_REPLICATES,
            seed=seed + step,
        )
        profile = lag_gains(evaluation["real_record"], "increment_numerator")
        pass_l1 = l1_pass(evaluation, PRACTICAL_EFFECT, stability=stability, ridge=ridge_ok)
        gap = float(item["gaps"][rank]["gap"])
        threshold = float((gap_envelopes or {}).get(str(rank), np.inf))
        boundary = _boundary_reproduces(path_a, path_b, index, rank, gap_envelopes)
        pass_l2 = bool(pass_l1 and ridge_boundary_ok and gap > threshold and boundary)
        tested = True
        stopped = not pass_l1
        steps.append({
            "step": step, "rank": rank, "tested": tested,
            "delta_mr": float(evaluation["delta_mr"]),
            "random_floor_q95": float(evaluation["random_floor_q95"]),
            "t": float(evaluation["t"]), "t_lcb95": float(evaluation["t_lcb95"]),
            "stability_msc": real_msc, "stability_null_q95": null_q95,
            "stability_pass": stability, "ridge_pass": ridge_ok,
            "ridge_boundary_pass": ridge_boundary_ok,
            "gap": gap, "gap_threshold": threshold,
            "boundary_reproduces": boundary, "l1": pass_l1, "l2": pass_l2,
            "all_rank_gaps": {str(k): float(v["gap"]) for k, v in item["gaps"].items()},
            "profile": profile.tolist(),
        })
        if pass_l1:
            accepted_components.append(candidate)
        accepted = orthonormal_union(accepted, candidate)
    threshold = float(profile_envelope if profile_envelope is not None else np.inf)
    l2_component_indices = {
        index + 2 for index, step in enumerate(steps) if step["tested"] and step["l2"]
    }
    distances, reproduced = [], []
    if l2_component_indices:
        point_profiles = _unique_lag_profiles(
            frame, pool, accepted_components, validation_mask=validation, score_mask=test,
        )
        frame_a = fit_frame(pool, pool.splits["fitA"], rho=ESTIMATOR_RHO)
        frame_b = fit_frame(pool, pool.splits["fitB"], rho=ESTIMATOR_RHO)
        profiles_a = _unique_lag_profiles(
            frame_a, pool, accepted_components,
            validation_mask=validation, score_mask=pool.splits["fitB"],
        )
        profiles_b = _unique_lag_profiles(
            frame_b, pool, accepted_components,
            validation_mask=validation, score_mask=pool.splits["fitA"],
        )
        for left in range(len(accepted_components)):
            for right in range(left + 1, len(accepted_components)):
                point = _shape_distance(point_profiles[left], point_profiles[right])
                half_a = _shape_distance(profiles_a[left], profiles_a[right])
                half_b = _shape_distance(profiles_b[left], profiles_b[right])
                distances.append(point)
                reproduced.append({
                    "left": left + 1, "right": right + 1,
                    "point": point, "fitA": half_a, "fitB": half_b,
                    "involves_l2": bool(left in l2_component_indices or right in l2_component_indices),
                    "passes": bool(
                        (left in l2_component_indices or right in l2_component_indices)
                        and point > threshold and half_a > threshold and half_b > threshold
                    ),
                })
    l3 = bool(any(item["passes"] for item in reproduced))
    return {
        "world": world, "documents": int(documents), "seed": int(seed),
        "sharing": float(sharing), "effect_multiplier": float(effect_multiplier),
        "discrete": bool(discrete), "w5_family": w5_family,
        "calibration_only": bool(calibration_only), "steps": steps,
        "accepted_l1": int(sum(step["tested"] and step["l1"] for step in steps)),
        "accepted_l2": int(sum(step["tested"] and step["l2"] for step in steps)),
        "l3": l3, "profile_distances": distances,
        "profile_reproduction": reproduced,
        "fwer_event": bool(any(step["tested"] and step["l1"] for step in steps)),
    }
