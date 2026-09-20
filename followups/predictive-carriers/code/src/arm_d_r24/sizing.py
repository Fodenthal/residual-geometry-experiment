"""Learning curves and the confirmatory sample size (amendment A8).

The endpoint that decides the handoff is not predictive gain -- it is split-half subspace
agreement at the largest rank, which converges far more slowly.  Sizing on the gain curve
would leave the binding endpoint underpowered, so A8 names the slow endpoint in advance and
sizes on it.

The curve is measured against FIT-HALF size, not pool size, because that is the quantity the
estimator actually sees: a confirmatory pool of ``N`` documents gives each half ``0.35N``.
DEV-Q supports half sizes up to 1000, so 4000 and 8000 are extrapolations of the fitted
``MSC(m) = a - b/m`` form and are reported as such.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from arm_d_r24.contract import (
    CONFIRM_CANDIDATE_SIZES,
    PRIMARY_SCORE,
    CONFIRM_SPLIT_FRACTIONS,
    LAGS_Q,
    LARGEST_RANK,
    SEED,
    namespaced_seed,
)
from arm_d_r24.data import Pool
from arm_d_r24.estimators import carrier_family
from arm_d_r24.geometry import mean_subspace_correlation
from arm_d_r24.scoring import bootstrap_weights, fit_frame, score_carrier

SIZING_SEED_NAMESPACE = "arm_d_r2_4_sizing"


def half_size(total: int) -> int:
    return int(round(float(total) * float(CONFIRM_SPLIT_FRACTIONS["fitA"])))


def learning_curve(
    pool: Pool,
    *,
    half_sizes: Sequence[int],
    metric: str,
    rho: float,
    repeats: int = 6,
    rank: int = LARGEST_RANK,
    gain_rank: int = 16,
    lags: Sequence[int] = LAGS_Q,
    gain_bootstrap: int = 500,
    seed: int = SEED,
    workers: int = 1,
) -> dict[str, object]:
    """Split-half MSC and multi-horizon gain against fit-half size."""

    payloads = []
    for m in half_sizes:
        m = int(m)
        if 2 * m > pool.n_documents:
            continue
        for repeat in range(int(repeats)):
            payloads.append((
                _pool_key(pool), m, int(repeat), metric, float(rho), int(rank), int(gain_rank),
                tuple(int(value) for value in lags), int(gain_bootstrap), int(seed),
            ))
    from arm_d_r24.stability import _run_parallel

    results = _run_parallel(
        _learning_point, payloads, workers=int(workers), pool=pool,
        key=f"sizing_{pool.name}",
    )
    collected: dict[int, list[tuple[float, float, float]]] = {}
    for m, _repeat, msc, gain, error in results:
        collected.setdefault(int(m), []).append((msc, gain, error))
    points: dict[str, object] = {}
    for m, values in sorted(collected.items()):
        msc_values = [value[0] for value in values]
        gain_values = [value[1] for value in values]
        gain_errors = [value[2] for value in values]
        points[str(m)] = {
            "fit_half_documents": m,
            "repeats": int(repeats),
            "split_half_msc_mean": float(np.mean(msc_values)),
            "split_half_msc_sd": float(np.std(msc_values, ddof=1)) if len(msc_values) > 1 else 0.0,
            "split_half_msc_values": [float(value) for value in msc_values],
            "gain_mean": float(np.mean(gain_values)),
            "gain_bootstrap_standard_error_mean": float(np.mean(gain_errors)),
        }
    return {
        "metric": metric,
        "rank": int(rank),
        "gain_rank": int(gain_rank),
        "rho": float(rho),
        "points": points,
    }


def _pool_key(pool: Pool) -> str:
    return f"sizing_{pool.name}"


def _learning_point(payload):
    """One (fit-half size, repeat) point, run in a forked worker."""

    from arm_d_r24.stability import _pool

    (key, m, repeat, metric, rho, rank, gain_rank, lags, gain_bootstrap, seed) = payload
    pool = _pool(key)
    rng = np.random.default_rng(namespaced_seed(SIZING_SEED_NAMESPACE, m, repeat))
    draw = rng.choice(pool.n_documents, size=2 * m, replace=False)
    mask_a = np.zeros(pool.n_documents, dtype=bool)
    mask_b = np.zeros(pool.n_documents, dtype=bool)
    mask_a[draw[:m]] = True
    mask_b[draw[m:]] = True
    frame_a = fit_frame(pool, mask_a, lags=lags, rho=rho)
    frame_b = fit_frame(pool, mask_b, lags=lags, rho=rho)
    bases_a = carrier_family(frame_a, metric=metric, ranks=(rank, gain_rank))["bases"]
    bases_b = carrier_family(frame_b, metric=metric, ranks=(rank, gain_rank))["bases"]
    msc = mean_subspace_correlation(bases_a[rank], bases_b[rank])
    weights = bootstrap_weights(int(mask_b.sum()), int(gain_bootstrap), seed=seed + repeat)
    scored = score_carrier(
        frame_a, bases_a[gain_rank], documents=pool.documents, score_mask=mask_b, weights=weights,
    )
    return (
        int(m), int(repeat), float(msc),
        float(scored["summary"]["multi_horizon_mean"][PRIMARY_SCORE]),
        float(scored["summary"]["bootstrap"][PRIMARY_SCORE]["standard_error"]),
    )


def fit_inverse_curve(sizes: Sequence[float], values: Sequence[float]) -> dict[str, float]:
    """Least squares for ``value(m) = a - b/m`` -- the standard high-dimensional bias form."""

    m = np.asarray(list(sizes), dtype=np.float64)
    y = np.asarray(list(values), dtype=np.float64)
    design = np.stack([np.ones_like(m), -1.0 / m], axis=1)
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ solution
    residual = y - prediction
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "asymptote_a": float(solution[0]),
        "slope_b": float(solution[1]),
        "r_squared": float(1.0 - float(np.sum(residual**2)) / total) if total > 0 else float("nan"),
        "max_absolute_residual": float(np.max(np.abs(residual))),
    }


def choose_size(
    curve: dict[str, object],
    *,
    msc_tolerance: float,
    se_max: float,
    candidates: Sequence[int] = CONFIRM_CANDIDATE_SIZES,
) -> dict[str, object]:
    """Amendment A8's frozen sizing rule, applied to the fitted curve."""

    points = curve["points"]  # type: ignore[index]
    sizes = [float(entry["fit_half_documents"]) for entry in points.values()]  # type: ignore[union-attr]
    msc = [float(entry["split_half_msc_mean"]) for entry in points.values()]  # type: ignore[union-attr]
    errors = [
        float(entry["gain_bootstrap_standard_error_mean"]) for entry in points.values()  # type: ignore[union-attr]
    ]
    msc_fit = fit_inverse_curve(sizes, msc)
    # A bootstrap standard error scales as 1/sqrt(n); fit that form directly.
    se_fit = float(np.mean([e * np.sqrt(s) for e, s in zip(errors, sizes)]))

    evaluated: dict[str, object] = {}
    chosen: int | None = None
    for candidate in candidates:
        m = half_size(int(candidate))
        predicted_msc = msc_fit["asymptote_a"] - msc_fit["slope_b"] / float(m)
        # The scored split is the other fit half, so the standard error scales with it.
        predicted_se = se_fit / float(np.sqrt(m))
        qualifies = bool(
            (msc_fit["asymptote_a"] - predicted_msc) <= float(msc_tolerance)
            and predicted_se <= float(se_max)
        )
        evaluated[str(int(candidate))] = {
            "documents": int(candidate),
            "fit_half_documents": int(m),
            "predicted_split_half_msc": float(predicted_msc),
            "gap_to_asymptote": float(msc_fit["asymptote_a"] - predicted_msc),
            "predicted_gain_standard_error": float(predicted_se),
            "qualifies": qualifies,
        }
        if qualifies and chosen is None:
            chosen = int(candidate)
    fallback = int(max(candidates))
    return {
        "rule": (
            "the smallest candidate whose predicted split-half MSC at the largest rank is "
            "within msc_tolerance of the fitted asymptote AND whose predicted gain bootstrap "
            "standard error is at most se_max; otherwise the largest candidate"
        ),
        "endpoint": "split-half mean subspace correlation at rank 64 (amendment A8)",
        "msc_curve_fit": msc_fit,
        "standard_error_scale": se_fit,
        "msc_tolerance": float(msc_tolerance),
        "se_max": float(se_max),
        "candidates": evaluated,
        "chosen_documents": int(chosen if chosen is not None else fallback),
        "chosen_by_fallback": bool(chosen is None),
        "extrapolation_note": (
            "DEV-Q supports fit halves up to 1000 documents, so the 4000 and 8000 candidates "
            "are extrapolations of the fitted 1/m form rather than measurements"
        ),
    }
