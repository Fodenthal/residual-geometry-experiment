"""Direction versus magnitude in the old composition failure (source section 6).

DEV-only, descriptive, and deliberately unable to make an autonomy claim.  The question is
narrow: when the inherited carrier's one-step law was iterated and missed, did it point the
wrong way, or point roughly the right way with the wrong gain?

Three quantities per lag, in the carrier's own coordinates, all fitted on fit documents and
evaluated held out:

``C_dir``       mean cosine between the composed and the directly fitted prediction
``E_radial``    mean absolute log ratio of their norms
``E_shape``     residual energy after the single best global rescaling of the composed
                prediction, which is the part of the disagreement no gain can fix
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from arm_d_r4.common_target import GroupedRidge, bootstrap_weights

from arm_d_r24.contract import LAGS_Q, RIDGE_GRID, SEED, STATE_ORDER
from arm_d_r24.data import Pool


def build_state(pool: Pool, basis: np.ndarray, lag: int, order: int = STATE_ORDER) -> np.ndarray:
    """``s_{t+lag} = [z_{t+lag}; ...; z_{t+lag-order+1}]`` in carrier coordinates."""

    q = np.asarray(basis, dtype=np.float64)
    blocks = []
    for shift in range(int(order)):
        offset = int(lag) - shift
        if offset not in pool.offsets:
            raise KeyError(f"the pool has no source offset {offset} for lag {lag}")
        blocks.append(pool.offsets[offset] @ q)
    return np.concatenate(blocks, axis=1)


def decompose(
    pool: Pool,
    basis: np.ndarray,
    *,
    fit_mask: np.ndarray,
    score_mask: np.ndarray,
    lags: Sequence[int] = LAGS_Q,
    order: int = STATE_ORDER,
    replicates: int = 2000,
    seed: int = SEED,
) -> dict[str, object]:
    """The three quantities per lag, with a document bootstrap on each."""

    basis = np.asarray(basis, dtype=np.float64)
    rank = int(basis.shape[1])
    size = int(order) * rank
    state = build_state(pool, basis, 0, order)
    solver = GroupedRidge(
        state[fit_mask], pool.documents[fit_mask], seed=seed, ridge_grid=RIDGE_GRID
    )
    next_state = (pool.offsets[1] @ basis)[fit_mask]
    transition, transition_ridge = solver.select_and_fit(next_state)
    companion = np.zeros((size, size), dtype=np.float64)
    companion[:rank, :] = transition.T
    if order > 1:
        companion[rank:, : size - rank] = np.eye(size - rank)

    weights = bootstrap_weights(int(score_mask.sum()), int(replicates), seed=seed)
    per_lag: dict[str, object] = {}
    for lag in lags:
        lag = int(lag)
        target_fit = (pool.offsets[lag] @ basis)[fit_mask]
        direct_weights, direct_ridge = solver.select_and_fit(target_fit)
        state_score = state[score_mask]
        composed = (state_score @ np.linalg.matrix_power(companion, lag).T)[:, :rank]
        direct = state_score @ direct_weights

        composed_norm = np.linalg.norm(composed, axis=1)
        direct_norm = np.linalg.norm(direct, axis=1)
        finite = (composed_norm > 0) & (direct_norm > 0)
        cosine = np.zeros_like(composed_norm)
        cosine[finite] = np.sum(composed[finite] * direct[finite], axis=1) / (
            composed_norm[finite] * direct_norm[finite]
        )
        radial = np.zeros_like(composed_norm)
        radial[finite] = np.abs(np.log(composed_norm[finite] / direct_norm[finite]))

        # The best global rescaling is fitted on FIT documents, not on the scored ones.
        composed_fit = (state[fit_mask] @ np.linalg.matrix_power(companion, lag).T)[:, :rank]
        denominator = float(np.sum(composed_fit**2))
        scale = float(np.sum(composed_fit * target_fit) / denominator) if denominator > 0 else 0.0
        shape_error = np.sum((direct - scale * composed) ** 2, axis=1)
        target_energy = np.sum((pool.offsets[lag] @ basis)[score_mask] ** 2, axis=1)

        def summarise(values: np.ndarray) -> dict[str, float]:
            totals = weights.sum(axis=1)
            replicate = weights @ values / totals
            return {
                "point": float(np.mean(values)),
                "q05": float(np.quantile(replicate, 0.05)),
                "q95": float(np.quantile(replicate, 0.95)),
            }

        per_lag[str(lag)] = {
            "lag": lag,
            "directional_agreement": summarise(cosine),
            "radial_error": summarise(radial),
            "scale_corrected_action_error": summarise(shape_error / max(float(np.mean(target_energy)), 1e-300)),
            "best_global_scale": scale,
            "direct_ridge": float(direct_ridge),
            "documents_scored": int(score_mask.sum()),
        }
    return {
        "rank": rank,
        "state_order": int(order),
        "transition_ridge": float(transition_ridge),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(companion)))),
        "bootstrap_replicates": int(replicates),
        "per_lag": per_lag,
    }


def interpret(record: dict[str, object], *, direction_floor: float = 0.5, radial_floor: float = 0.5) -> str:
    """Source section 6's three descriptive readings, applied to the long lags."""

    lags = [int(key) for key in record["per_lag"]]  # type: ignore[union-attr]
    long_lags = [lag for lag in lags if lag >= 8] or lags
    cosines = [
        float(record["per_lag"][str(lag)]["directional_agreement"]["point"])  # type: ignore[index]
        for lag in long_lags
    ]
    radial = [
        float(record["per_lag"][str(lag)]["radial_error"]["point"])  # type: ignore[index]
        for lag in long_lags
    ]
    direction_kept = float(np.mean(cosines)) >= direction_floor
    scale_broken = float(np.mean(radial)) >= radial_floor
    if direction_kept and scale_broken:
        return "DIRECTION_PRESERVED_SCALE_FAILS"
    if not direction_kept and not scale_broken:
        return "FULL_ACTION_FAILS"
    if not direction_kept:
        return "FULL_ACTION_FAILS"
    return "MIXED_DIRECTION_SCALE_FAILURE"
