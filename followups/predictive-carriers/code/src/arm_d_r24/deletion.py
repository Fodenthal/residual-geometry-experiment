"""The variance-dose deletion audit (D-AUDIT) and the L axis.

The claim under audit is three runs old: deleting the top-variance rank-16 subspace destroys
the remote-history prediction, therefore those directions are specially load-bearing.  The
alternative has never been excluded, because it cannot be excluded at equal rank: PCA1:16
maximizes captured variance among all rank-16 subspaces, so the nearest energy-matched
rank-16 rival is nearly the same subspace (R2.3c measured 0.9375 subspace correlation).

Amendment A3 takes the escape the impossibility argument leaves open.  Matching rank and
captured variance simultaneously is impossible; matching captured variance at HIGHER rank is
routine.  Family F deletes lower principal directions until it has removed the same fraction
of aperture variance, and becomes the primary comparator.  The dose-response curve over
every family is the continuous context around it.

Every deletion is followed by a FULL refit from the surviving source coordinates to the
unchanged 256-dimensional target -- never by rescoring a frozen estimator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from arm_d_r24.contract import (
    DELETION_DOSE_GRID,
    DELETION_DOSE_RANKS,
    PRIMARY_SCORE,
    DELETION_HAAR_DRAWS,
    DELETION_MIXED_M,
    DELETION_ROTATION_ANGLES,
    POSITIVE_GAIN_FLOOR,
    PRIMARY_RANK,
    label,
    SECTION_DELETION,
    unavailable,
)
from arm_d_r24.data import Pool
from arm_d_r24.geometry import (
    complement_basis,
    deleted_variance,
    deletion_families,
    local_linear_prediction,
)
from arm_d_r24.scoring import Frame, bootstrap_summary, gains, score_projection


def audit(
    pool: Pool,
    frame: Frame,
    eigen: Mapping[str, np.ndarray],
    *,
    score_mask: np.ndarray,
    width: int,
    rank: int = PRIMARY_RANK,
    weights: np.ndarray | None = None,
    extra_bases: Mapping[str, np.ndarray] | None = None,
    dose_ranks: Sequence[int] = DELETION_DOSE_RANKS,
    dose_grid: Sequence[float] = DELETION_DOSE_GRID,
) -> dict[str, object]:
    """Run every deletion family, plus any extra named basis, and score the refits."""

    target_variance = float(
        deleted_variance(frame.source_covariance, np.asarray(eigen["basis"])[:, :rank])
    )
    families = deletion_families(
        dict(eigen),
        width=int(width),
        rank=int(rank),
        haar_draws=DELETION_HAAR_DRAWS,
        mixed_m=DELETION_MIXED_M,
        rotation_angles=DELETION_ROTATION_ANGLES,
        dose_ranks=dose_ranks,
        dose_grid=dose_grid,
        target_variance=target_variance,
        cumulative=np.asarray(eigen["cumulative_fraction"]),
    )
    for name, basis in (extra_bases or {}).items():
        families[name] = {
            "family": "X",
            "role": "a named carrier under audit",
            "basis": np.asarray(basis, dtype=np.float64),
        }

    full = score_projection(
        frame, np.eye(int(width)), documents=pool.documents, score_mask=score_mask
    )
    full_gains = gains(full, weights=weights)

    entries: dict[str, object] = {}
    energies: dict[str, object] = {}
    for name, spec in families.items():
        basis = np.asarray(spec["basis"], dtype=np.float64)
        surviving = complement_basis(basis, width=int(width))
        record = score_projection(
            frame, surviving, documents=pool.documents, score_mask=score_mask
        )
        scored = gains(record, weights=weights)
        entries[name] = {
            "family": spec["family"],
            "role": spec["role"],
            "deleted_rank": int(basis.shape[1]),
            "dose_target": spec.get("dose_target"),
            "energy_match_attained": spec.get("energy_match_attained"),
            "attained_variance": spec.get("attained_variance"),
            "maximum_attainable_lower_variance": spec.get("maximum_attainable_lower_variance"),
            "surviving_dimension": int(surviving.shape[1]),
            "deleted_variance": deleted_variance(frame.source_covariance, basis),
            "gains": scored,
            "surviving_fraction_multi_horizon": (
                float(scored["multi_horizon_mean"][PRIMARY_SCORE])
                / float(full_gains["multi_horizon_mean"][PRIMARY_SCORE])
                if float(full_gains["multi_horizon_mean"][PRIMARY_SCORE]) >= POSITIVE_GAIN_FLOOR
                else None
            ),
            "surviving_fraction_by_lag": {
                key: (
                    float(scored["per_lag"][key][PRIMARY_SCORE])
                    / float(full_gains["per_lag"][key][PRIMARY_SCORE])
                    if float(full_gains["per_lag"][key][PRIMARY_SCORE]) >= POSITIVE_GAIN_FLOOR
                    else None
                )
                for key in scored["per_lag"]
            },
            "projection_idempotence": float(
                np.max(np.abs(surviving @ surviving.T @ surviving - surviving))
            ),
            "deleted_orthonormality_error": float(
                np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1])))
            ),
        }
        energies[name] = record["per_lag"]
    return {
        "target_variance": target_variance,
        "full_aperture": full_gains,
        "full_energies": full["per_lag"],
        "families": entries,
        "energies": energies,
        "positive_gain_floor": float(POSITIVE_GAIN_FLOOR),
    }


def _multi_horizon_replicates(per_lag: Mapping[str, object], weights: np.ndarray) -> np.ndarray:
    from arm_d_r4.common_target import gain_replicates

    stack = [
        gain_replicates(
            entry["augmented_residual"], entry["baseline_residual"], entry["target_energy"],
            weights,
        )
        for entry in per_lag.values()
    ]
    return np.mean(np.stack(stack, axis=0), axis=0)


def dose_response(
    result: Mapping[str, object],
    *,
    target: str = "A_pca_1_16",
    weights: np.ndarray | None = None,
    bandwidth: float = 0.08,
    margin: float | None = None,
) -> dict[str, object]:
    """Surviving gain against deleted variance, and the A12 dose-matched geometry contrast.

    Two readings come out of the same deletions.  The dose curve is descriptive context: what
    an ordinary deletion of a given dose leaves.  The contrast is the inferential object: at
    a dose BOTH constructions can reach, does removing top-variance directions destroy more
    predictive gain than removing lower ones carrying the same energy?
    """

    families = result["families"]  # type: ignore[index]
    points = [
        (str(name), float(entry["deleted_variance"]),
         float(entry["gains"]["multi_horizon_mean"][PRIMARY_SCORE]))
        for name, entry in families.items()  # type: ignore[union-attr]
    ]
    context = [
        (name, v, g)
        for name, v, g in points
        if families[name]["family"] in {"C", "D", "E", "F"} and name != target  # type: ignore[index]
    ]
    full_gain = float(result["full_aperture"]["multi_horizon_mean"][PRIMARY_SCORE])  # type: ignore[index]
    target_entry = families[target]  # type: ignore[index]
    target_v = float(target_entry["deleted_variance"])
    target_gain = float(target_entry["gains"]["multi_horizon_mean"][PRIMARY_SCORE])
    context_v = np.array([v for _, v, _ in context])
    predicted = local_linear_prediction(
        context_v, np.array([g for _, _, g in context]), target_v, bandwidth=bandwidth
    )
    extrapolating = bool(target_v > float(np.max(context_v)) + 1e-9) if context_v.size else True

    out: dict[str, object] = {
        "target": target,
        "target_deleted_variance": target_v,
        "target_surviving_gain": target_gain,
        "target_surviving_fraction": target_gain / full_gain if full_gain > 0 else None,
        "full_aperture_gain": full_gain,
        "dose_curve": [
            {"name": name, "deleted_variance": v, "surviving_gain": g,
             "family": families[name]["family"]}  # type: ignore[index]
            for name, v, g in sorted(points, key=lambda item: item[1])
        ],
        "local_prediction_at_target_dose": float(predicted),
        "excess_destructiveness_versus_dose_curve": float(predicted - target_gain),
        "local_prediction_is_extrapolation": extrapolating,
        "context_dose_range": [float(context_v.min()), float(context_v.max())] if context_v.size else None,
        "bandwidth": float(bandwidth),
        "context_families": ["C", "D", "E", "F"],
    }

    energy_family = families.get("F_energy_matched")  # type: ignore[union-attr]
    if energy_family is not None:
        attained = energy_family.get("energy_match_attained")
        out["energy_match_attainability"] = {
            "attained": bool(attained),
            "maximum_attainable_lower_variance": energy_family.get(
                "maximum_attainable_lower_variance"
            ),
            "target_variance": target_v,
            "rank_used": int(energy_family["deleted_rank"]),
            "note": (
                "amendment A12: the leading directions hold more aperture variance than every "
                "other direction combined, so no lower-direction deletion reaches the "
                "target's dose at any rank; the comparison is made at attainable doses instead"
            ) if not attained else "the energy match was attainable at higher rank",
        }
        if not attained:
            out["energy_matched_comparison"] = {
                "unavailable": unavailable(
                    "energy_matched_deletion_at_target_dose",
                    "the aperture spectrum does not permit a lower-direction deletion of "
                    f"{target_v:.6f} captured variance; the maximum attainable is "
                    f"{float(energy_family.get('maximum_attainable_lower_variance', float('nan'))):.6f}",
                )
            }

    # The A12 contrast, dose by dose.
    contrasts: dict[str, object] = {}
    for name, entry in families.items():  # type: ignore[union-attr]
        if entry["family"] != "H_top":
            continue
        dose = float(entry["dose_target"])
        partner = name.replace("_top", "_lower")
        if partner not in families:  # type: ignore[operator]
            continue
        lower = families[partner]  # type: ignore[index]
        top_gain = float(entry["gains"]["multi_horizon_mean"][PRIMARY_SCORE])
        lower_gain = float(lower["gains"]["multi_horizon_mean"][PRIMARY_SCORE])
        record: dict[str, object] = {
            "dose_target": dose,
            "top_deleted_variance": float(entry["deleted_variance"]),
            "lower_deleted_variance": float(lower["deleted_variance"]),
            "top_deleted_rank": int(entry["deleted_rank"]),
            "lower_deleted_rank": int(lower["deleted_rank"]),
            "top_surviving_gain": top_gain,
            "lower_surviving_gain": lower_gain,
            "delta_top_more_destructive": lower_gain - top_gain,
        }
        if weights is not None:
            difference = _multi_horizon_replicates(
                result["energies"][partner], weights  # type: ignore[index]
            ) - _multi_horizon_replicates(result["energies"][name], weights)  # type: ignore[index]
            record["bootstrap"] = {
                "replicates": int(weights.shape[0]),
                "mean": float(np.mean(difference)),
                "q05": float(np.quantile(difference, 0.05)),
                "q50": float(np.quantile(difference, 0.50)),
                "q95": float(np.quantile(difference, 0.95)),
                "fraction_above_zero": float(np.mean(difference > 0)),
            }
        contrasts[f"{dose:g}"] = record
    out["matched_dose_contrasts"] = contrasts
    if contrasts:
        primary_key = max(contrasts, key=lambda key: float(key))
        primary = dict(contrasts[primary_key])
        primary["dose"] = float(primary_key)
        if margin is not None:
            lower_bound = primary.get("bootstrap", {}).get("q05")  # type: ignore[union-attr]
            primary["margin"] = float(margin)
            primary["clears_margin"] = bool(
                float(primary["delta_top_more_destructive"]) > float(margin)
                and lower_bound is not None
                and float(lower_bound) > 0.0
            )
        primary["note"] = (
            "the largest dose both constructions can reach; positive delta means removing "
            "top-variance directions destroys more predictive gain than removing lower "
            "directions carrying the same aperture variance"
        )
        out["primary_comparison"] = primary
    else:
        out["primary_comparison"] = {
            "unavailable": unavailable(
                "matched_dose_contrast", "no dose in the frozen grid was attainable"
            )
        }
    return out


def l_axis_label(
    dose: Mapping[str, object], *, confirmatory_replay: bool, margin: float
) -> dict[str, object]:
    """The L axis, with amendment A4's replay requirement enforced in code."""

    comparison = dose.get("primary_comparison", {})
    if "unavailable" in comparison:  # type: ignore[operator]
        return label(
            "L_UNDERRESOLVED",
            section=SECTION_DELETION,
            evidence={"reason": comparison["unavailable"], "dose": dict(dose)},  # type: ignore[index]
        )
    clears = bool(comparison.get("clears_margin", False))  # type: ignore[union-attr]
    if clears and not confirmatory_replay:
        return label(
            "LOAD_BEARINGNESS_GEOMETRY_SUGGESTIVE",
            section=SECTION_DELETION,
            evidence={
                "amendment": "A4",
                "why": (
                    "the target is more destructive than the energy-matched family, but "
                    "amendment A4 forbids L1_GEOMETRY_SPECIFIC_DELETION without the "
                    "confirmatory replay of the frozen deletion family"
                ),
                "comparison": dict(comparison),  # type: ignore[arg-type]
                "margin": float(margin),
            },
        )
    if clears:
        return label(
            "L1_GEOMETRY_SPECIFIC_DELETION",
            section=SECTION_DELETION,
            evidence={"comparison": dict(comparison), "margin": float(margin)},  # type: ignore[arg-type]
        )
    return label(
        "L0_ENERGY_EXPLAINS_DELETION",
        section=SECTION_DELETION,
        evidence={
            "comparison": dict(comparison),  # type: ignore[arg-type]
            "margin": float(margin),
            "excess_destructiveness_versus_dose_curve": dose[
                "excess_destructiveness_versus_dose_curve"
            ],
        },
    )
