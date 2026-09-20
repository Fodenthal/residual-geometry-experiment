"""The four independent result axes, plus the ridge axis and the promotion rule.

Source section 13 reports four axes rather than one ladder, and every label is preceded by
the continuous quantity it summarizes.  The label never replaces the curve; it exists so the
next stage has an unambiguous handoff.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from arm_d_r24.contract import (
    COMPRESSION_FRACTION,
    LARGEST_RANK,
    SECTION_LABELS,
    SECTION_PROMOTION,
    SECTION_RIDGE,
    SECTION_STABILITY,
    label,
    unavailable,
)


def compression(rank_curve: Mapping[str, float], *, fraction: float = COMPRESSION_FRACTION,
                ceiling_fraction: float | None = None) -> dict[str, object]:
    """``r_90`` and the P axis (source sections 9 and 13.1).

    ``rank_curve`` maps rank to the multi-horizon mean gain.  Compression means the curve
    reaches the frozen fraction of its own rank-64 maximum at a rank materially below 64 --
    and the report always carries the whole curve beside the label.
    """

    ranks = sorted(int(key) for key in rank_curve)
    values = {int(key): float(value) for key, value in rank_curve.items()}
    best = max(values[r] for r in ranks if r <= LARGEST_RANK)
    if not np.isfinite(best) or best <= 0:
        return dict(
            label(
                "P_TECHNICAL_FAILURE",
                section=SECTION_LABELS,
                evidence={
                    "reason": "no positive predictive gain at any rank in the grid",
                    "rank_curve": values,
                },
            ),
            **{"r90": None},
        )
    threshold = float(fraction) * best
    reaching = [r for r in ranks if values[r] >= threshold]
    r90 = int(min(reaching)) if reaching else None
    compact = bool(r90 is not None and r90 < LARGEST_RANK)
    record = label(
        "P1_COMPACT_PREDICTIVE_COMPRESSION" if compact else "P0_NO_COMPACT_PREDICTIVE_COMPRESSION",
        section=SECTION_LABELS,
        evidence={
            "rank_curve": values,
            "best_gain_within_grid": float(best),
            "compression_fraction": float(fraction),
            "threshold": threshold,
            "r90": r90,
            "fraction_of_full_aperture_ceiling_at_r90": ceiling_fraction,
            "note": (
                "the 90% number is a compression summary, not a significance threshold; the "
                "full rank curve is reported beside this label"
            ),
        },
    )
    record["r90"] = r90
    return record


def stability_axis(
    observed_msc: float,
    null_values: Sequence[float],
    ceiling_mean: float,
    *,
    rank: int,
    metric: str,
) -> dict[str, object]:
    """Whether one estimator's carrier is stable beyond its full-refit null."""

    from arm_d_r24.stability import empirical_p_value

    p = empirical_p_value(float(observed_msc), null_values)
    exceeds = bool(float(observed_msc) > float(np.max(null_values)))
    return {
        "metric": metric,
        "rank": int(rank),
        "observed_split_half_msc": float(observed_msc),
        "null_max": float(np.max(null_values)),
        "null_q99": float(np.quantile(null_values, 0.99)),
        "null_mean": float(np.mean(null_values)),
        "empirical_p_value": float(p),
        "bootstrap_self_overlap_mean": float(ceiling_mean),
        "fraction_of_attainable_ceiling": (
            float(observed_msc) / float(ceiling_mean) if float(ceiling_mean) > 0 else float("nan")
        ),
        "stable": exceeds,
    }


def metric_geometry(
    cross_msc: float,
    ceiling_mean: float,
    *,
    fraction: float,
    energy_stable: bool,
    whitened_stable: bool,
    rank_energy: int,
    rank_whitened: int,
) -> dict[str, object]:
    """The G axis, judged against the attainable sampling ceiling rather than a constant."""

    if not (energy_stable and whitened_stable):
        return label(
            "G_UNDERRESOLVED",
            section=SECTION_LABELS,
            evidence={
                "reason": "at least one estimator is not stable beyond its full-refit null",
                "energy_stable": bool(energy_stable),
                "whitened_stable": bool(whitened_stable),
                "cross_metric_msc": float(cross_msc),
            },
        )
    threshold = float(fraction) * float(ceiling_mean)
    concordant = bool(float(cross_msc) >= threshold)
    return label(
        "G1_METRIC_CONCORDANT_CARRIER" if concordant else "G0_METRIC_DEPENDENT_CARRIERS",
        section=SECTION_LABELS,
        evidence={
            "cross_metric_msc": float(cross_msc),
            "bootstrap_self_overlap_mean": float(ceiling_mean),
            "metric_concordance_fraction": float(fraction),
            "threshold": threshold,
            "rank_energy": int(rank_energy),
            "rank_whitened": int(rank_whitened),
            "note": (
                "no consensus basis is manufactured when the two metrics disagree; both are "
                "frozen as separate predictive representations"
            ),
        },
    )


def identifiability(
    *,
    stable: bool,
    complement_relative: float | None,
    complement_stable: bool | None,
    deletion_control_relative: float | None,
    high: float,
    low: float,
) -> dict[str, object]:
    """The I axis (source sections 12.2 and 13.3)."""

    if not stable:
        return label(
            "I0_NONIDENTIFIED_CARRIER",
            section=SECTION_LABELS,
            evidence={
                "reason": "predictive gain may exist but the fitted subspace is not stable "
                "beyond its full-refit null"
            },
        )
    if complement_relative is None:
        return label(
            "I1_STABLE_CARRIER",
            section=SECTION_LABELS,
            evidence={
                "complement_refit": unavailable(
                    "complement_refit_relative_gain", "the complement refit was not run"
                )
            },
        )
    evidence = {
        "complement_relative_gain": float(complement_relative),
        "complement_carrier_stable": complement_stable,
        "rank_matched_deletion_relative_gain": deletion_control_relative,
        "redundancy_relative_high": float(high),
        "redundancy_relative_low": float(low),
    }
    if float(complement_relative) >= float(high) and bool(complement_stable):
        name = "I3_REDUNDANT_CARRIER_FAMILY"
    elif float(complement_relative) <= float(low):
        name = "I4_DOMINANT_CHANNEL"
    else:
        name = "I2_STABLE_CORE_BROAD_SHOULDER"
    return label(name, section=SECTION_LABELS, evidence=evidence)


def ridge_axis(sweep: Mapping[str, object], *, rank: int, minimum: float) -> dict[str, object]:
    """Amendment A5's axis: is the carrier a carrier, or a regularization choice?"""

    entry = sweep["by_rank"][str(int(rank))]  # type: ignore[index]
    observed = float(entry["min_msc_adjacent_decade"])
    stable = bool(observed >= float(minimum))
    return label(
        "R1_RIDGE_STABLE_CARRIER" if stable else "R0_RIDGE_DEPENDENT_CARRIER",
        section=SECTION_RIDGE,
        evidence={
            "rank": int(rank),
            "selected_rho": sweep["selected_rho"],
            "min_msc_adjacent_decade": observed,
            "min_msc_over_grid": float(entry["min_msc_over_grid"]),
            "ridge_stability_min": float(minimum),
            "msc_against_selected": entry["msc_against_selected"],
        },
    )


def promotion(
    *,
    validation: Mapping[str, float],
    test: Mapping[str, float],
    fraction: float,
) -> dict[str, object]:
    """Source section 15 point 5, given a number by amendment A8."""

    checks: dict[str, object] = {}
    passes = True
    for name in ("multi_horizon_gain", "split_half_msc"):
        reference = float(validation[name])
        observed = float(test[name])
        required = float(fraction) * reference
        agrees = bool(
            np.sign(observed) == np.sign(reference) and abs(observed) >= abs(required)
        )
        passes = passes and agrees
        checks[name] = {
            "validation": reference,
            "test": observed,
            "required": required,
            "ratio": observed / reference if reference != 0 else float("nan"),
            "agrees": agrees,
        }
    return label(
        "CONFIRMATORY_REPRODUCTION" if passes else "CONFIRMATORY_REPRODUCTION_FAILED",
        section=SECTION_PROMOTION,
        evidence={"reproduction_min_fraction": float(fraction), "checks": checks},
    )


def handoff_case(p_label: str, g_label: str, i_label: str) -> dict[str, object]:
    """Source section 14's four handoff cases, and what the next stage is allowed to do."""

    if p_label != "P1_COMPACT_PREDICTIVE_COMPRESSION" or i_label == "I0_NONIDENTIFIED_CARRIER":
        case, description = "D", (
            "no compact stable carrier: do not run another fixed-low-rank autonomy search"
        )
    elif i_label == "I3_REDUNDANT_CARRIER_FAMILY":
        case, description = "C", (
            "a redundant carrier family: the next dynamics question is whether the dynamics "
            "are equivalent across redundant predictive realizations"
        )
    elif g_label == "G0_METRIC_DEPENDENT_CARRIERS":
        case, description = "B", (
            "stable but metric-dependent carriers: the next stage tests both under a "
            "multiplicity-controlled contract or fixes one metric for a stated scientific "
            "reason before looking at autonomy results"
        )
    else:
        case, description = "A", (
            "one stable metric-concordant compact carrier becomes the sole primary carrier "
            "for the next autonomy test"
        )
    return {
        "case": case,
        "description": description,
        "p_label": p_label,
        "g_label": g_label,
        "i_label": i_label,
        "data_rule": (
            "amendment A9: the autonomy stage either draws a fresh document pool or takes "
            "every composition statistic only from R2.4's untouched test split"
        ),
    }
