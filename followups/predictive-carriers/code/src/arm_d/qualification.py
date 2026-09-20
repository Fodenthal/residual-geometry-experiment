"""Synthetic qualification harness and the frozen selections (R1 §31, §33).

Design of the power/false-positive measurement.  Running a full 199-replicate
null bank inside each of 200 replicates of each of 11 worlds is not affordable
and buys nothing: the quantity that matters is whether one decision threshold
separates the worlds that must be detected from the worlds that must not be.
So the threshold is calibrated once, at the 95th percentile of the pipeline
statistic on ``S-D4`` (the pure local-window confound, which the paired design
should cancel exactly), and then power and false-positive rate are measured
against that single frozen threshold on the other worlds.  ``S-D4``'s own rate
is 0.05 by construction and is not counted as evidence; ``S-D5`` and ``S-D6``
are the honest false-positive measurements.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from arm_d.composition import evaluate_composition
from arm_d.operators import MSC_CHANCE_LEVEL, build_shared_space, fit_screening_operator, mean_subspace_correlation
from arm_d.pipeline import CellResult, run_cell
from arm_d.synthetic import (
    CALIBRATE_THRESHOLD_ON,
    MUST_DETECT,
    MUST_NOT_DETECT,
    MUST_NOT_SHOW_MIXING,
    MUST_SHOW_MIXING,
    WorldConfig,
    build_world,
    world_plant,
)

QUALIFICATION_APERTURES = (32, 64)
QUALIFICATION_RANKS = (2, 4)
APERTURE_CANDIDATES = (256, 512, 1024)
REQUIRED_POWER = 0.80
MAX_FALSE_POSITIVE_RATE = 0.05


@dataclass(frozen=True)
class ReplicateOutcome:
    world: str
    seed: int
    statistic: float
    mean_subspace_correlation: float
    best_rank: int
    best_lag: int
    mixing_supported: bool = False
    gate_passed: bool = False


def _grid_results(data, *, msc_threshold: float, seed: int) -> list[CellResult]:
    results: list[CellResult] = []
    for aperture in QUALIFICATION_APERTURES:
        if aperture > data.aperture_width:
            continue
        for lag in data.lags:
            for rank in QUALIFICATION_RANKS:
                cell = run_cell(
                    data,
                    aperture=aperture,
                    rank=rank,
                    lag=lag,
                    suffix_length=int(data.suffix_length[0]),
                    condition="main",
                    msc_threshold=msc_threshold,
                    seed=seed,
                )
                if cell is not None:
                    results.append(cell)
    return results


def run_replicate(arguments: tuple[str, int, dict, float]) -> ReplicateOutcome:
    world, seed, config_kwargs, msc_threshold = arguments
    config = WorldConfig(**config_kwargs)
    data = build_world(world, config, seed)
    results = _grid_results(data, msc_threshold=msc_threshold, seed=seed)
    if not results:
        return ReplicateOutcome(world, seed, float("-inf"), float("nan"), -1, -1)
    best = max(results, key=lambda item: item.statistic)
    return ReplicateOutcome(
        world=world,
        seed=seed,
        statistic=float(best.statistic),
        mean_subspace_correlation=float(best.mean_subspace_correlation),
        best_rank=int(best.rank),
        best_lag=int(best.lag),
        mixing_supported=bool(best.mixing_supported),
        gate_passed=bool(best.gate_passed),
    )


def run_world_bank(
    worlds: tuple[str, ...],
    *,
    replicates: int,
    config: WorldConfig,
    msc_threshold: float,
    workers: int,
    seed_base: int = 1000,
) -> dict[str, list[ReplicateOutcome]]:
    jobs = [
        (world, seed_base + index + 7919 * position, asdict(config), msc_threshold)
        for position, world in enumerate(worlds)
        for index in range(replicates)
    ]
    outcomes: dict[str, list[ReplicateOutcome]] = {world: [] for world in worlds}
    if workers <= 1:
        for job in jobs:
            outcome = run_replicate(job)
            outcomes[outcome.world].append(outcome)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for outcome in pool.map(run_replicate, jobs, chunksize=4):
                outcomes[outcome.world].append(outcome)
    return outcomes


def calibrate_msc_gate(
    fixed_coordinate_msc: np.ndarray,
    transport_msc: np.ndarray,
) -> dict[str, float]:
    """Freeze the shared-space gate between the two calibration populations.

    The threshold maximizes balanced accuracy between the fixed-coordinate
    worlds and the transport calibration world, and the achieved separation is
    reported as an AUC.  A strict quantile-non-overlap rule was tried first and
    is wrong here: the two populations overlap in the tails at realistic
    signal-to-noise, so demanding non-overlap would either fail every run or
    push the threshold to a value no real fit could clear.  A weakly separating
    gate is a limitation on the *label*, reported as such, not a reason to
    refuse the whole run.
    """

    fixed = np.asarray(fixed_coordinate_msc, dtype=np.float64)
    transport = np.asarray(transport_msc, dtype=np.float64)
    fixed = fixed[np.isfinite(fixed)]
    transport = transport[np.isfinite(transport)]
    if fixed.size == 0 or transport.size == 0:
        return {"shared_space_msc_threshold": MSC_CHANCE_LEVEL, "auc": float("nan"), "separated": False}
    candidates = np.unique(np.concatenate([fixed, transport]))
    accuracy = np.array(
        [0.5 * (np.mean(fixed >= value) + np.mean(transport < value)) for value in candidates]
    )
    threshold = float(candidates[int(np.argmax(accuracy))])
    comparisons = (fixed[:, None] > transport[None, :]).astype(float)
    comparisons += 0.5 * (fixed[:, None] == transport[None, :])
    auc = float(comparisons.mean())
    return {
        "fixed_coordinate_msc_median": float(np.median(fixed)),
        "transport_msc_median": float(np.median(transport)),
        "chance_level": MSC_CHANCE_LEVEL,
        "auc": auc,
        "balanced_accuracy": float(accuracy.max()),
        "separated": bool(auc >= 0.80),
        "shared_space_msc_threshold": threshold,
    }


def measure_bank(outcomes: dict[str, list[ReplicateOutcome]]) -> tuple[float, dict[str, dict[str, float]]]:
    """Calibrate the threshold, then measure power and false-positive rate."""

    calibration = np.array(
        [item.statistic for item in outcomes[CALIBRATE_THRESHOLD_ON] if np.isfinite(item.statistic)]
    )
    if calibration.size == 0:
        raise ValueError("calibration world produced no finite statistics")
    threshold = float(np.quantile(calibration, 0.95))
    summary: dict[str, dict[str, float]] = {}
    for world, items in outcomes.items():
        values = np.array([item.statistic for item in items if np.isfinite(item.statistic)])
        rate = float(np.mean(values > threshold)) if values.size else float("nan")
        summary[world] = {
            "replicates": int(values.size),
            "detection_rate": rate,
            "median_statistic": float(np.median(values)) if values.size else float("nan"),
            "p95_statistic": float(np.quantile(values, 0.95)) if values.size else float("nan"),
            "median_mean_subspace_correlation": float(
                np.nanmedian([item.mean_subspace_correlation for item in items])
            )
            if items
            else float("nan"),
            "shared_space_gate_rate": float(np.mean([item.gate_passed for item in items]))
            if items
            else float("nan"),
            "mixing_supported_rate": float(np.mean([item.mixing_supported for item in items]))
            if items
            else float("nan"),
            "role": (
                "power"
                if world in MUST_DETECT
                else "false_positive"
                if world in MUST_NOT_DETECT
                else "descriptive"
            ),
        }
    return threshold, summary


def qualification_verdict(summary: dict[str, dict[str, float]]) -> dict[str, object]:
    power_failures = [
        world for world in MUST_DETECT if summary.get(world, {}).get("detection_rate", 0.0) < REQUIRED_POWER
    ]
    fpr_failures = [
        world
        for world in MUST_NOT_DETECT
        if world != CALIBRATE_THRESHOLD_ON
        and summary.get(world, {}).get("detection_rate", 1.0) > MAX_FALSE_POSITIVE_RATE
    ]
    mixing_failures = [
        world for world in MUST_SHOW_MIXING
        if summary.get(world, {}).get("mixing_supported_rate", 0.0) < 0.5
    ] + [
        world for world in MUST_NOT_SHOW_MIXING
        if summary.get(world, {}).get("mixing_supported_rate", 1.0) > 0.5
    ]
    return {
        "required_power": REQUIRED_POWER,
        "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
        "power_worlds": list(MUST_DETECT),
        "false_positive_worlds": [world for world in MUST_NOT_DETECT if world != CALIBRATE_THRESHOLD_ON],
        "power_failures": power_failures,
        "false_positive_failures": fpr_failures,
        "mixing_worlds": {"must_mix": list(MUST_SHOW_MIXING), "must_not_mix": list(MUST_NOT_SHOW_MIXING)},
        "mixing_failures": mixing_failures,
        "passes": not power_failures and not fpr_failures,
        "mixing_discriminates": not mixing_failures,
    }


def select_base_lag(config: WorldConfig, *, seed: int = 4242, replicates: int = 24) -> dict[str, object]:
    """Choose the composition base lag from ``{4, 8}`` on the closed world.

    Both candidates are scored by their Chapman-Kolmogorov action error on
    ``S-D8`` (closed token-driven state, where the semigroup genuinely holds)
    and cross-checked on ``S-D7`` (non-Markov, where it must not).
    """

    outcomes: dict[str, dict[int, list[float]]] = {"S-D8": {4: [], 8: []}, "S-D7": {4: [], 8: []}}
    long_config = WorldConfig(**{**asdict(config), "lags": (1, 2, 4, 8, 16, 32)})
    for replicate in range(replicates):
        for world in ("S-D8", "S-D7"):
            data = build_world(world, long_config, seed + replicate)
            for base_lag in (4, 8):
                coordinates = _composition_inputs(data, base_lag=base_lag)
                if coordinates is None:
                    continue
                results = evaluate_composition(
                    coordinates=coordinates,
                    base_lag=base_lag,
                    lags=tuple(lag for lag in long_config.lags if lag % base_lag == 0 and lag != base_lag),
                    envelope_replicates=24,
                    seed=seed + replicate,
                )
                if results:
                    outcomes[world][base_lag].append(float(np.mean([item.action_error for item in results])))
    scores = {
        base_lag: float(np.median(outcomes["S-D8"][base_lag])) if outcomes["S-D8"][base_lag] else float("inf")
        for base_lag in (4, 8)
    }
    selected = int(min(scores, key=lambda key: scores[key]))
    return {
        "composition_base_lag": selected,
        "median_ck_action_error_closed_world": scores,
        "median_ck_action_error_non_markov_world": {
            base_lag: float(np.median(outcomes["S-D7"][base_lag])) if outcomes["S-D7"][base_lag] else None
            for base_lag in (4, 8)
        },
        "rule": "smaller median CK action error on the closed token-driven world S-D8",
    }


def _composition_inputs(data, *, base_lag: int, aperture: int = 64, rank: int = 4) -> dict | None:
    """Project a synthetic world into a fitted subspace for the CK check."""

    train = data.split == "train"
    evaluate = data.split == "val"
    if train.sum() < 32 or evaluate.sum() < 8:
        return None
    p = min(aperture, data.aperture_width)
    lag_index = {lag: index for index, lag in enumerate(data.lags)}
    if base_lag not in lag_index:
        return None
    screening = fit_screening_operator(
        data.source[train][:, :p], data.future[train][:, lag_index[base_lag], :p], rank
    )
    if not screening.healthy:
        return None
    shared = build_shared_space(
        screening.source_space, screening.destination_space, rank=rank, msc_threshold=0.0
    )
    basis = shared.basis
    coordinates = {}
    for lag, index in lag_index.items():
        coordinates[lag] = {
            "source_train": data.source[train][:, :p] @ basis,
            "destination_train": data.future[train][:, index, :p] @ basis,
            "documents_train": data.document_ids[train],
            "source_eval": data.source[evaluate][:, :p] @ basis,
            "destination_eval": data.future[evaluate][:, index, :p] @ basis,
            "documents_eval": data.document_ids[evaluate],
        }
    return coordinates


def aperture_replicate(arguments: tuple[int, int, dict, float]) -> dict[str, float]:
    """One aperture-qualification replicate at production document count.

    Returns the cell statistic and the containment of the *planted* subspace in
    the recovered ``Q``.  Containment is the quality metric R1 §7 asks for:
    chance containment for a random rank-``r`` subspace is ``r/p``, so 4/256 at
    the smallest candidate, and the frozen bar sits far above it.
    """

    aperture, seed, config_kwargs, msc_threshold = arguments
    config = WorldConfig(**{**config_kwargs, "width": aperture})
    data = build_world("S-D8", config, seed)
    plant = world_plant("S-D8", config, seed)
    rank = config.rank
    cell = run_cell(
        data,
        aperture=aperture,
        rank=rank,
        lag=config.lags[2],
        suffix_length=int(data.suffix_length[0]),
        condition="main",
        msc_threshold=msc_threshold,
        seed=seed,
    )
    train = data.split == "train"
    screening = fit_screening_operator(
        data.source[train], data.future[train][:, config.lags.index(config.lags[2])], rank
    )
    containment = float("nan")
    if screening.healthy:
        shared = build_shared_space(
            screening.source_space, screening.destination_space, rank=rank, msc_threshold=0.0
        )
        containment = mean_subspace_correlation(shared.basis, plant)
    if cell is None:
        return {"aperture": float(aperture), "statistic": float("nan"), "msc": float("nan"),
                "plant_containment": containment}
    return {
        "aperture": float(aperture),
        "statistic": float(cell.statistic),
        "msc": float(cell.mean_subspace_correlation),
        "plant_containment": containment,
    }


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
