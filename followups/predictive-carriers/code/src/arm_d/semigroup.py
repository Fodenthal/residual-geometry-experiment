"""Does one fixed generator exist? Gain-free diagnostics for the CK failure.

The preregistered composition verdict (R1 §19.3) stands as computed; everything
here is DIAGNOSTIC and is labelled as such in every artifact it writes.

The composition failure was pooled into one action error, and unpooling it (§19.2)
showed the magnitude channel failing while direction largely survived. The
proximate cause is arithmetically clear: the directly fitted operator has a
leading eigenvalue modulus above one, and iterating it 32 times amplifies that
excess geometrically. But "the fit is expansive" and "no single generator exists"
are different claims, and the spectra already hint at the second: a single
generator forces log|lambda_j(A_k)| to be linear in k through the origin, and the
observed leading mode is near +0.138 at both k=1 and k=4 and then NEGATIVE at
k=64. No fixed A can do that.

Five diagnostics, ordered by how little they assume:

1. ``difference_norm_profile`` -- model-free. No operator, no refit. If the
   remote-history difference GROWS before it decays, then no single fixed
   operator can compose, by construction, and the CK failure is a real property
   of the propagation rather than an estimator artifact. This is the decisive
   measurement and it is the cheapest.
2. ``eigenvalue_linearity`` -- the semigroup test on log-moduli, which involves
   no gain comparison at all.
3. ``invariant_subspace_profile`` -- principal angles between the leading
   invariant subspaces of ``A_delta`` and ``A_{delta*m}`` (§21: subspace and
   spectrum are identifiable where exact axes are not). Separates "same flow
   geometry, wrong gain" from "different directions at different horizons"
   without any norm entering.
4. ``joint_horizon_operator`` -- fit ONE operator across all horizons at once.
   A cleaner test of "does a generator exist" than radius clipping, which is
   only a crude regularizer.
5. ``directional_calibration`` -- a positive cosine is not self-evidently
   meaningful when one mode dominates: both predictions align with it. Reference
   the observed cosine against mismatched and spectrum-matched-random operators.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.persistent_state.live_state.temporal_code import principal_angles

from arm_d.operators import fit_full, select_ridge
from arm_d.scoring import document_folds, document_index, equal_document_r2

SECTION_19_2 = "R1 section 19.2 (radial and directional error are never pooled)"
SECTION_21 = "R1 section 21 (subspace and spectrum identifiable; exact axes are not)"
SECTION_22 = "R1 section 22 (eigenvalue cluster matching)"
DIAGNOSTIC_ROLE = "DIAGNOSTIC; the preregistered CK verdict stands as computed"


# --------------------------------------------------------------------------
# 1. The model-free difference-norm profile
# --------------------------------------------------------------------------


def _document_mean(values: np.ndarray, documents: np.ndarray) -> float:
    """Equal-document-weighted mean of a per-row quantity (R1 section 30)."""

    unique, codes = document_index(documents)
    n_documents = int(unique.size)
    counts = np.bincount(codes, minlength=n_documents).astype(np.float64)
    totals = np.bincount(codes, weights=np.asarray(values, dtype=np.float64), minlength=n_documents)
    present = counts > 0
    return float(np.mean(totals[present] / counts[present]))


def _bootstrap_document_interval(
    values: np.ndarray,
    documents: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    unique, codes = document_index(documents)
    n_documents = int(unique.size)
    counts = np.bincount(codes, minlength=n_documents).astype(np.float64)
    totals = np.bincount(codes, weights=np.asarray(values, dtype=np.float64), minlength=n_documents)
    present = counts > 0
    per_document = totals[present] / counts[present]
    n_documents = int(per_document.size)
    if n_documents < 2 or replicates <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        pick = rng.integers(0, n_documents, size=n_documents)
        draws[replicate] = float(np.mean(per_document[pick]))
    return (float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95)))


def difference_norm_profile(
    *,
    source: np.ndarray,
    futures: np.ndarray,
    lags: Sequence[int],
    documents: np.ndarray,
    mode_directions: np.ndarray | None = None,
    bootstrap_replicates: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Equal-document ``||dz_{t+k}|| / ||dz_t||`` against ``k``, with no operator.

    ``futures`` is ``(n_rows, n_lags, n_coordinates)`` aligned with ``lags``.

    If this profile RISES before it falls, the remote-history difference grows
    over the first few tokens and only later decays. No single fixed operator can
    represent that, so the composition failure is then a real finding about the
    shape of propagation rather than an estimation failure. If it decays
    monotonically, the transient reading is dead and estimation instability
    becomes the leading explanation.
    """

    source_array = np.asarray(source, dtype=np.float64)
    futures_array = np.asarray(futures, dtype=np.float64)
    document_array = np.asarray(documents)
    reference = np.linalg.norm(source_array, axis=1)
    usable = reference > 0

    rows: list[dict[str, object]] = []
    for index, lag in enumerate(lags):
        ratio = np.zeros(source_array.shape[0], dtype=np.float64)
        ratio[usable] = (
            np.linalg.norm(futures_array[usable, index, :], axis=1) / reference[usable]
        )
        low, high = _bootstrap_document_interval(
            ratio[usable],
            document_array[usable],
            replicates=bootstrap_replicates,
            seed=seed + index,
        )
        rows.append(
            {
                "lag": int(lag),
                "norm_ratio": _document_mean(ratio[usable], document_array[usable]),
                "norm_ratio_q05": low,
                "norm_ratio_q95": high,
                "rows_used": int(np.count_nonzero(usable)),
            }
        )

    per_mode: list[dict[str, object]] = []
    if mode_directions is not None and np.asarray(mode_directions).size:
        directions = np.asarray(mode_directions, dtype=np.float64)
        for mode in range(directions.shape[1]):
            axis = directions[:, mode]
            axis = axis / max(float(np.linalg.norm(axis)), 1e-300)
            base = np.abs(source_array @ axis)
            mode_usable = base > 0
            profile = []
            for index, lag in enumerate(lags):
                projected = np.abs(futures_array[:, index, :] @ axis)
                ratio = np.zeros_like(base)
                ratio[mode_usable] = projected[mode_usable] / base[mode_usable]
                profile.append(
                    {
                        "lag": int(lag),
                        "norm_ratio": _document_mean(
                            ratio[mode_usable], document_array[mode_usable]
                        ),
                    }
                )
            per_mode.append({"mode": int(mode), "profile": profile})

    ratios = [row["norm_ratio"] for row in rows]
    finite = [value for value in ratios if np.isfinite(value)]
    peak_index = int(np.argmax(ratios)) if finite else -1
    # "Rises before it falls" is judged against the bootstrap and against the
    # FIRST measured lag, not against an absolute ratio of one.  The scientific
    # question is whether propagation is non-monotone in the horizon, and the
    # shortest available lag is the reference for that.  A rise whose interval
    # overlaps the first lag's interval is not a rise.
    # A peak at the FINAL lag is a monotone rise, not a rise before a fall, and
    # calling it one would misdescribe the profile. The peak must be interior and
    # must be separated from BOTH ends beyond the bootstrap.
    rises = False
    peak_is_interior = bool(0 < peak_index < len(rows) - 1)
    if peak_is_interior:
        peak = rows[peak_index]
        first = rows[0]
        last = rows[-1]
        rises = bool(
            np.isfinite(peak["norm_ratio_q05"])
            and np.isfinite(first["norm_ratio_q95"])
            and np.isfinite(last["norm_ratio_q95"])
            and peak["norm_ratio_q05"] > first["norm_ratio_q95"]
            and peak["norm_ratio_q05"] > last["norm_ratio_q95"]
        )
    return {
        "role": DIAGNOSTIC_ROLE,
        "profile": rows,
        "per_mode_profile": per_mode,
        "peak_lag": int(rows[peak_index]["lag"]) if peak_index >= 0 else None,
        "peak_is_after_the_first_lag": bool(peak_index > 0),
        "peak_is_interior": peak_is_interior,
        "monotonically_rising": bool(
            len(finite) == len(ratios)
            and all(ratios[i + 1] >= ratios[i] - 1e-12 for i in range(len(ratios) - 1))
        ),
        "rises_before_it_falls": rises,
        "monotonically_decaying": bool(
            len(finite) == len(ratios) and all(
                ratios[i + 1] <= ratios[i] + 1e-12 for i in range(len(ratios) - 1)
            )
        ),
        "peak_exceeds_first_lag_beyond_bootstrap": rises,
        "caveat": (
            "a rise does not by itself refute a single generator: a NON-NORMAL fixed "
            "operator can also produce transient growth before eventual decay, and the "
            "departure from normality here is recorded in the spectral artifacts. A rise "
            "establishes that propagation is non-monotone in the horizon, which is why "
            "short-lag and long-lag fits describe different regimes and a single operator "
            "must trade them off. The decisive tests of one generator are the eigenvalue "
            "linearity check and the joint-horizon fit, not this profile alone."
        ),
        "reading": (
            "the difference grows before it decays, so propagation is non-monotone in the "
            "horizon and short-lag and long-lag operators describe different regimes; this "
            "is consistent with the composition failure being a real property of the "
            "propagation rather than only an estimation artifact, subject to the "
            "non-normality caveat"
            if rises
            else "the difference rises monotonically across the whole grid without turning "
            "over, so the profile shows sustained growth rather than a transient"
            if peak_index == len(rows) - 1 and len(rows) > 1
            else "the difference decays monotonically, so the transient reading is not "
            "supported and estimation instability of an expansive fit is the leading "
            "explanation for the composition failure"
        ),
    }


# --------------------------------------------------------------------------
# 2. The semigroup test on log-moduli (no gain comparison)
# --------------------------------------------------------------------------


def eigenvalue_linearity(
    operators: dict[int, np.ndarray],
    *,
    reference_lag: int = 1,
    n_modes: int = 4,
) -> dict[str, object]:
    """Under a single generator, ``log|lambda_j(A_k)|`` is linear in ``k`` through 0.

    Modes are matched across lags by descending modulus, which is the §22
    clustering rule for well-separated leading modes. The reported deviation from
    the predicted line is the composition diagnostic; it involves no gain
    comparison whatsoever.
    """

    lags = sorted(int(lag) for lag in operators)
    if not lags:
        return {"role": DIAGNOSTIC_ROLE, "modes": [], "semigroup_consistent": None}

    spectra: dict[int, np.ndarray] = {}
    for lag in lags:
        values = np.linalg.eigvals(np.asarray(operators[lag], dtype=np.float64))
        spectra[lag] = values[np.argsort(-np.abs(values))]

    available = min(n_modes, min(spectrum.size for spectrum in spectra.values()))
    modes: list[dict[str, object]] = []
    for mode in range(available):
        points = []
        for lag in lags:
            modulus = float(np.abs(spectra[lag][mode]))
            points.append(
                {
                    "lag": int(lag),
                    "modulus": modulus,
                    "log_modulus": float(np.log(modulus)) if modulus > 0 else float("-inf"),
                    "per_step_log_modulus": (
                        float(np.log(modulus) / lag) if modulus > 0 and lag else float("nan")
                    ),
                }
            )
        reference = next(
            (point for point in points if point["lag"] == reference_lag), points[0]
        )
        slope = reference["log_modulus"] / max(reference["lag"], 1)
        deviations = [
            point["log_modulus"] - slope * point["lag"]
            for point in points
            if np.isfinite(point["log_modulus"])
        ]
        per_step = [
            point["per_step_log_modulus"]
            for point in points
            if np.isfinite(point["per_step_log_modulus"])
        ]
        sign_flips = sum(
            1
            for i in range(len(per_step) - 1)
            if np.sign(per_step[i]) != 0
            and np.sign(per_step[i + 1]) != 0
            and np.sign(per_step[i]) != np.sign(per_step[i + 1])
        )
        # Normalized headline: under a semigroup the PER-STEP log modulus is the
        # same at every lag.  Its spread relative to the reference slope is
        # scale-free, unlike the raw deviation, which grows with lag by
        # construction and so cannot carry a threshold.
        spread = float(max(per_step) - min(per_step)) if per_step else float("nan")
        relative_spread = (
            spread / abs(slope) if per_step and abs(slope) > 1e-12 else float("nan")
        )
        modes.append(
            {
                "mode": int(mode),
                "points": points,
                "predicted_slope_per_step": float(slope),
                "per_step_log_modulus_spread": spread,
                "per_step_spread_relative_to_reference": relative_spread,
                "departs_from_linearity": bool(
                    np.isfinite(relative_spread) and relative_spread > 0.25
                ),
                "max_absolute_deviation_from_line": (
                    float(np.max(np.abs(deviations))) if deviations else float("nan")
                ),
                "per_step_log_modulus_range": (
                    [float(min(per_step)), float(max(per_step))] if per_step else None
                ),
                "per_step_sign_flips": int(sign_flips),
                "changes_between_growth_and_decay": bool(sign_flips > 0),
            }
        )

    leading = modes[0] if modes else None
    return {
        "role": DIAGNOSTIC_ROLE,
        "citation": SECTION_22,
        "reference_lag": int(reference_lag),
        "modes": modes,
        "leading_mode_changes_sign": bool(leading["changes_between_growth_and_decay"])
        if leading
        else None,
        "leading_mode_departs_from_linearity": bool(leading["departs_from_linearity"])
        if leading
        else None,
        "leading_mode_per_step_spread_relative": float(
            leading["per_step_spread_relative_to_reference"]
        )
        if leading
        else None,
        # Both channels gate the verdict.  An earlier version consulted only the
        # sign flip and so called a 14-log-unit departure from linearity
        # "consistent", which contradicted its own headline number.
        "semigroup_consistent": bool(
            leading is not None
            and not leading["changes_between_growth_and_decay"]
            and not leading["departs_from_linearity"]
        ),
        "reading": (
            "the leading mode's per-step log modulus changes sign across the lag grid, "
            "so it grows at short horizons and contracts at long ones; a single fixed "
            "generator cannot produce that and the semigroup model is refuted on the "
            "spectrum alone, with no gain comparison involved"
            if leading and leading["changes_between_growth_and_decay"]
            else "the leading mode's per-step log modulus keeps one sign but is far from "
            "constant across lags, so the log moduli are not linear in the horizon and a "
            "single generator is not supported"
            if leading and leading["departs_from_linearity"]
            else "the leading mode's per-step log modulus is approximately constant and "
            "keeps one sign, so the spectrum does not refute a single generator"
        ),
    }


# --------------------------------------------------------------------------
# 3. Invariant-subspace geometry (no norm enters)
# --------------------------------------------------------------------------


def leading_invariant_subspace(operator: np.ndarray, dimension: int) -> np.ndarray:
    """Orthonormal basis for the invariant subspace of the top-modulus modes.

    Complex conjugate pairs contribute their real and imaginary parts, which span
    the same real invariant plane, so the returned basis is real.
    """

    array = np.asarray(operator, dtype=np.float64)
    values, vectors = np.linalg.eig(array)
    order = np.argsort(-np.abs(values))
    collected: list[np.ndarray] = []
    for index in order:
        if len(collected) >= dimension:
            break
        vector = vectors[:, index]
        if np.iscomplexobj(vector) and np.max(np.abs(vector.imag)) > 1e-12:
            collected.append(vector.real)
            if len(collected) < dimension:
                collected.append(vector.imag)
        else:
            collected.append(vector.real)
    if not collected:
        return np.zeros((array.shape[0], 0))
    stacked = np.column_stack(collected[:dimension])
    basis, _ = np.linalg.qr(stacked)
    return basis


def modulus_degeneracy(operator: np.ndarray, dimension: int) -> dict[str, object]:
    """Is the cut at ``dimension`` separated enough for mode matching to be stable?

    Matching modes across lags by descending modulus is only meaningful when the
    modulus at the cut is separated from the next one. Under near-degeneracy the
    ordering within the tied group is arbitrary, so two operators with the SAME
    invariant subspaces can report large principal angles purely from the sort.
    This is the same failure shape as the Schur block separation check (R1 section
    22), and it must be reported rather than silently producing a large angle.
    """

    moduli = np.sort(np.abs(np.linalg.eigvals(np.asarray(operator, dtype=np.float64))))[::-1]
    if moduli.size <= dimension:
        return {"separated": True, "gap": float("inf"), "relative_gap": float("inf")}
    gap = float(moduli[dimension - 1] - moduli[dimension])
    scale = float(max(moduli[0], 1e-300))
    relative = gap / scale
    return {
        "separated": bool(relative > 0.02),
        "gap": gap,
        "relative_gap": relative,
        "moduli_at_cut": [float(moduli[dimension - 1]), float(moduli[dimension])],
        "note": (
            None
            if relative > 0.02
            else "moduli at the cut are nearly degenerate, so mode matching by modulus is "
            "arbitrary within the tied group and a large principal angle here may be an "
            "artifact of the ordering rather than a change of invariant subspace"
        ),
    }


def invariant_subspace_profile(
    operators: dict[int, np.ndarray],
    *,
    base_lag: int,
    dimension: int = 4,
) -> dict[str, object]:
    """Principal angles between the leading invariant subspaces of A_base and A_k.

    Under a single generator these subspaces coincide, because ``A^m`` has exactly
    the eigenvectors of ``A``. Large angles mean the flow points in different
    directions at different horizons, which no fixed operator can represent. No
    magnitude enters, so this is independent of the gain failure.
    """

    if base_lag not in operators:
        return {
            "role": DIAGNOSTIC_ROLE,
            "comparisons": [],
            "label": "SUBSPACE_PROFILE_UNAVAILABLE",
            "unavailable_because": "the base lag is absent from the fitted operators",
        }
    rank = int(np.asarray(operators[base_lag]).shape[0])
    if rank < 2:
        return {
            "role": DIAGNOSTIC_ROLE,
            "comparisons": [],
            "label": "SUBSPACE_PROFILE_UNAVAILABLE",
            "unavailable_because": (
                f"principal angles need at least two dimensions and the operator rank is {rank}"
            ),
        }
    # Never request more modes than exist; a truncated basis would silently
    # compare different numbers of directions at different lags.
    dimension = int(min(dimension, rank))
    base = leading_invariant_subspace(operators[base_lag], dimension)
    powered = {
        int(lag): np.linalg.matrix_power(
            np.asarray(operators[base_lag], dtype=np.float64), int(lag) // int(base_lag)
        )
        for lag in operators
        if int(lag) % int(base_lag) == 0 and int(lag) != int(base_lag)
    }

    comparisons: list[dict[str, object]] = []
    for lag in sorted(int(value) for value in operators):
        if lag == int(base_lag):
            continue
        direct = leading_invariant_subspace(operators[lag], dimension)
        if (
            direct.shape[1] == 0
            or base.shape[1] == 0
            or direct.shape[0] < 2
            or base.shape[0] < 2
            or direct.shape[1] != base.shape[1]
        ):
            comparisons.append(
                {
                    "lag": int(lag),
                    "label": "SUBSPACE_COMPARISON_UNAVAILABLE",
                    "unavailable_because": (
                        f"basis shapes {base.shape} and {direct.shape} cannot be compared"
                    ),
                }
            )
            continue
        angles = np.degrees(principal_angles(base, direct))
        row = {
            "lag": int(lag),
            "principal_angles_degrees": [float(value) for value in angles],
            "max_angle_degrees": float(np.max(angles)),
            "mean_angle_degrees": float(np.mean(angles)),
        }
        if lag in powered:
            against_power = np.degrees(
                principal_angles(leading_invariant_subspace(powered[lag], dimension), direct)
            )
            row["angles_against_iterated_base_degrees"] = [
                float(value) for value in against_power
            ]
            row["max_angle_against_iterated_base_degrees"] = float(np.max(against_power))
        comparisons.append(row)

    usable = [row for row in comparisons if "max_angle_degrees" in row]
    worst = max((row["max_angle_degrees"] for row in usable), default=float("nan"))
    degeneracy = {
        int(lag): modulus_degeneracy(operators[lag], dimension) for lag in operators
    }
    if not usable:
        return {
            "role": DIAGNOSTIC_ROLE,
            "comparisons": comparisons,
            "label": "SUBSPACE_PROFILE_UNAVAILABLE",
            "unavailable_because": "no lag pair produced comparable subspace bases",
            "modulus_degeneracy_by_lag": degeneracy,
        }
    ambiguous = [lag for lag, entry in degeneracy.items() if not entry["separated"]]
    return {
        "role": DIAGNOSTIC_ROLE,
        "citation": SECTION_21,
        "base_lag": int(base_lag),
        "dimension": int(dimension),
        "comparisons": comparisons,
        "modulus_degeneracy_by_lag": degeneracy,
        "lags_with_ambiguous_mode_matching": sorted(ambiguous),
        "mode_matching_is_reliable": bool(not ambiguous),
        "worst_max_angle_degrees": float(worst),
        "reading": (
            "mode matching is unreliable at these lags because the eigenvalue moduli at the "
            "cut are nearly degenerate, so the principal angles here do not support a "
            "conclusion either way"
            if ambiguous
            else "the leading invariant subspaces largely agree across horizons, so the flow "
            "geometry is shared and the composition failure is confined to gain"
            if np.isfinite(worst) and worst < 45.0
            else "the leading invariant subspaces differ materially across horizons, so the "
            "operator points in different directions at different lags and the failure is "
            "not only about gain"
        ),
    }


# --------------------------------------------------------------------------
# 4. One operator for every horizon at once
# --------------------------------------------------------------------------


def _joint_loss_and_gradient(
    transposed: np.ndarray,
    design: np.ndarray,
    targets: dict[int, np.ndarray],
    ridge: float,
) -> tuple[float, np.ndarray]:
    """Loss and analytic gradient for ``sum_m ||X B^m - Y_m||^2 + ridge ||B||^2``.

    ``B`` is ``A^T`` so that rows of ``X`` act on the right.
    """

    rank = transposed.shape[0]
    powers = {0: np.eye(rank)}
    highest = max(targets)
    for exponent in range(1, highest + 1):
        powers[exponent] = powers[exponent - 1] @ transposed

    loss = ridge * float(np.sum(transposed * transposed))
    gradient = 2.0 * ridge * transposed
    for multiplicity, target in targets.items():
        residual = design @ powers[multiplicity] - target
        loss += float(np.sum(residual * residual))
        common = design.T @ residual
        for j in range(multiplicity):
            gradient += 2.0 * (
                powers[j].T @ common @ powers[multiplicity - 1 - j].T
            )
    return loss, gradient


def joint_horizon_operator(
    *,
    coordinates: dict[int, dict[str, np.ndarray]],
    base_lag: int,
    lags: Sequence[int],
    ridge_grid: Sequence[float] = (1e-4, 1e-2, 1.0, 1e2),
    max_iterations: int = 400,
    seed: int = 42,
) -> dict[str, object]:
    """Fit ONE operator whose ``m``-th power predicts lag ``base_lag * m``.

    Compared on held-out documents against the per-lag directly fitted operators.
    If a single operator comes close everywhere, a generator exists and the
    per-lag fits were tracking a transient the CK test then punished. If the best
    joint operator is substantially worse at every horizon, that is direct
    evidence against the semigroup model. A cleaner test than radius clipping,
    which only regularizes.
    """

    from scipy.optimize import minimize

    base = coordinates[int(base_lag)]
    multiplicities = {
        int(lag) // int(base_lag): int(lag)
        for lag in lags
        if int(lag) % int(base_lag) == 0
    }
    multiplicities.setdefault(1, int(base_lag))

    design = np.asarray(base["source_train"], dtype=np.float64)
    documents = np.asarray(base["documents_train"])
    rank = design.shape[1]

    # Equal-document weights (R1 section 30) folded into row scalings.
    unique, codes = document_index(documents)
    counts = np.bincount(codes, minlength=int(unique.size)).astype(np.float64)
    weights = 1.0 / np.maximum(counts[codes], 1.0)
    weights = weights / float(np.mean(weights))
    scale = np.sqrt(weights)[:, None]

    targets = {
        multiplicity: np.asarray(
            coordinates[lag]["destination_train"], dtype=np.float64
        )
        * scale
        for multiplicity, lag in multiplicities.items()
    }
    weighted_design = design * scale

    # Warm start from the per-lag base fit, which is the semigroup hypothesis.
    warm_ridge = select_ridge(
        design, np.asarray(base["destination_train"], dtype=np.float64), documents,
        "full", ridge_grid=tuple(ridge_grid), seed=seed,
    )
    warm = fit_full(design, np.asarray(base["destination_train"], dtype=np.float64), warm_ridge).T

    fold_indices = document_folds(documents, n_folds=3, seed=seed)
    all_rows = np.arange(design.shape[0])
    folds = [
        (np.setdiff1d(all_rows, held_out), held_out)
        for held_out in fold_indices
        if held_out.size and held_out.size < all_rows.size
    ]
    best = {"ridge": None, "score": -np.inf, "operator": None}
    for ridge in ridge_grid:
        fold_scores: list[float] = []
        for train_mask, validate_mask in folds:
            fold_targets = {
                multiplicity: value[train_mask] for multiplicity, value in targets.items()
            }

            def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
                loss, gradient = _joint_loss_and_gradient(
                    flat.reshape(rank, rank),
                    weighted_design[train_mask],
                    fold_targets,
                    float(ridge),
                )
                return loss, gradient.ravel()

            outcome = minimize(
                objective, warm.ravel(), jac=True, method="L-BFGS-B",
                options={"maxiter": max_iterations},
            )
            candidate = outcome.x.reshape(rank, rank)
            scores = []
            for multiplicity, lag in multiplicities.items():
                entry = coordinates[lag]
                prediction = (
                    np.asarray(entry["source_train"], dtype=np.float64)[validate_mask]
                    @ np.linalg.matrix_power(candidate, multiplicity)
                )
                scores.append(
                    float(
                        equal_document_r2(
                            np.asarray(entry["destination_train"], dtype=np.float64)[
                                validate_mask
                            ],
                            prediction,
                            documents[validate_mask],
                        ).r2
                    )
                )
            fold_scores.append(float(np.mean(scores)))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best["score"]:
            best = {"ridge": float(ridge), "score": mean_score, "operator": None}

    def final_objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        loss, gradient = _joint_loss_and_gradient(
            flat.reshape(rank, rank), weighted_design, targets, float(best["ridge"])
        )
        return loss, gradient.ravel()

    final = minimize(
        final_objective, warm.ravel(), jac=True, method="L-BFGS-B",
        options={"maxiter": max_iterations},
    )
    joint = final.x.reshape(rank, rank)

    rows: list[dict[str, object]] = []
    for multiplicity, lag in sorted(multiplicities.items()):
        entry = coordinates[lag]
        evaluation_documents = np.asarray(entry["documents_eval"])
        source_eval = np.asarray(entry["source_eval"], dtype=np.float64)
        destination_eval = np.asarray(entry["destination_eval"], dtype=np.float64)
        joint_prediction = source_eval @ np.linalg.matrix_power(joint, multiplicity)
        joint_r2 = float(
            equal_document_r2(destination_eval, joint_prediction, evaluation_documents).r2
        )
        direct_ridge = select_ridge(
            np.asarray(entry["source_train"], dtype=np.float64),
            np.asarray(entry["destination_train"], dtype=np.float64),
            np.asarray(entry["documents_train"]), "full",
            ridge_grid=tuple(ridge_grid), seed=seed,
        )
        direct = fit_full(
            np.asarray(entry["source_train"], dtype=np.float64),
            np.asarray(entry["destination_train"], dtype=np.float64),
            direct_ridge,
        )
        direct_r2 = float(
            equal_document_r2(
                destination_eval, source_eval @ direct.T, evaluation_documents
            ).r2
        )
        rows.append(
            {
                "lag": int(lag),
                "multiplicity": int(multiplicity),
                "joint_r2": joint_r2,
                "direct_r2": direct_r2,
                "shortfall": direct_r2 - joint_r2,
            }
        )

    shortfalls = [row["shortfall"] for row in rows if np.isfinite(row["shortfall"])]
    worst = float(np.max(shortfalls)) if shortfalls else float("nan")
    return {
        "role": DIAGNOSTIC_ROLE,
        "selected_ridge": best["ridge"],
        "joint_operator_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(joint)))
        ),
        "converged": bool(final.success),
        "per_lag": rows,
        "worst_shortfall": worst,
        "mean_shortfall": float(np.mean(shortfalls)) if shortfalls else float("nan"),
        "reading": (
            "a single operator comes close to the per-lag fits at every horizon, so one "
            "generator plausibly exists and the per-lag fits were tracking a transient "
            "that the composition test then punished"
            if np.isfinite(worst) and worst < 0.05
            else "the best single operator is materially worse than the per-lag fits, which "
            "is direct evidence against a single fixed generator"
        ),
    }


# --------------------------------------------------------------------------
# 5. Calibrating the directional agreement
# --------------------------------------------------------------------------


def _row_cosine(
    left: np.ndarray, right: np.ndarray, documents: np.ndarray
) -> float:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    usable = (left_norm > 0) & (right_norm > 0)
    if not np.any(usable):
        return float("nan")
    cosine = np.sum(left[usable] * right[usable], axis=1) / (
        left_norm[usable] * right_norm[usable]
    )
    return _document_mean(cosine, np.asarray(documents)[usable])


def _matched_spectrum_random_operator(
    operator: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """An operator with the same eigenvalue moduli but random invariant directions."""

    array = np.asarray(operator, dtype=np.float64)
    values = np.linalg.eigvals(array)
    rank = array.shape[0]
    random_basis = np.linalg.qr(rng.normal(size=(rank, rank)))[0]
    # Build a real block-diagonal matrix with the same spectrum, then rotate it.
    blocks: list[np.ndarray] = []
    used = np.zeros(values.size, dtype=bool)
    for index in range(values.size):
        if used[index]:
            continue
        value = values[index]
        if abs(value.imag) > 1e-12:
            partner = int(
                np.argmin(np.where(used, np.inf, np.abs(values - np.conj(value))))
            )
            used[index] = used[partner] = True
            blocks.append(
                np.array([[value.real, value.imag], [-value.imag, value.real]])
            )
        else:
            used[index] = True
            blocks.append(np.array([[value.real]]))
    canonical = np.zeros((rank, rank))
    offset = 0
    for block in blocks:
        size = block.shape[0]
        if offset + size > rank:
            break
        canonical[offset : offset + size, offset : offset + size] = block
        offset += size
    return random_basis @ canonical @ random_basis.T


def directional_calibration(
    *,
    coordinates: dict[int, dict[str, np.ndarray]],
    operators: dict[int, np.ndarray],
    base_lag: int,
    lags: Sequence[int],
    random_draws: int = 32,
    seed: int = 42,
) -> dict[str, object]:
    """Reference the observed direction cosine against mismatched operators.

    In rank 16 with one dominant mode, the composed and the directly fitted
    prediction both align with that mode, so a positive cosine can come from
    shared dominant structure rather than from correct composition. The observed
    agreement is therefore reported against two references: iterating the base
    operator to the WRONG multiplicity, and operators with matched eigenvalue
    moduli but random invariant directions.
    """

    rng = np.random.default_rng(seed)
    base_operator = np.asarray(operators[int(base_lag)], dtype=np.float64)
    rows: list[dict[str, object]] = []
    multiplicities = {
        int(lag) // int(base_lag): int(lag)
        for lag in lags
        if int(lag) % int(base_lag) == 0 and int(lag) != int(base_lag)
    }

    for multiplicity, lag in sorted(multiplicities.items()):
        entry = coordinates[lag]
        source_eval = np.asarray(entry["source_eval"], dtype=np.float64)
        evaluation_documents = np.asarray(entry["documents_eval"])
        direct_prediction = source_eval @ np.asarray(
            operators[lag], dtype=np.float64
        ).T
        observed = _row_cosine(
            direct_prediction,
            source_eval @ np.linalg.matrix_power(base_operator, multiplicity).T,
            evaluation_documents,
        )
        mismatched = []
        for other, _ in sorted(multiplicities.items()):
            if other == multiplicity:
                continue
            mismatched.append(
                {
                    "multiplicity_used": int(other),
                    "cosine": _row_cosine(
                        direct_prediction,
                        source_eval
                        @ np.linalg.matrix_power(base_operator, other).T,
                        evaluation_documents,
                    ),
                }
            )
        random_cosines = [
            _row_cosine(
                direct_prediction,
                source_eval
                @ np.linalg.matrix_power(
                    _matched_spectrum_random_operator(base_operator, rng), multiplicity
                ).T,
                evaluation_documents,
            )
            for _ in range(random_draws)
        ]
        finite_random = [value for value in random_cosines if np.isfinite(value)]
        mismatched_values = [
            row["cosine"] for row in mismatched if np.isfinite(row["cosine"])
        ]
        reference = float(np.quantile(finite_random, 0.95)) if finite_random else float("nan")
        rows.append(
            {
                "lag": int(lag),
                "multiplicity": int(multiplicity),
                "observed_cosine": observed,
                "mismatched_multiplicity_cosines": mismatched,
                "mismatched_maximum": (
                    float(np.max(mismatched_values)) if mismatched_values else float("nan")
                ),
                "matched_spectrum_random_q95": reference,
                "matched_spectrum_random_median": (
                    float(np.median(finite_random)) if finite_random else float("nan")
                ),
                "exceeds_random_reference": bool(
                    np.isfinite(observed) and np.isfinite(reference) and observed > reference
                ),
                "exceeds_mismatched_reference": bool(
                    np.isfinite(observed)
                    and mismatched_values
                    and observed > float(np.max(mismatched_values))
                ),
            }
        )
    return {
        "role": DIAGNOSTIC_ROLE,
        "citation": SECTION_19_2,
        "note": (
            "a positive cosine against zero is not the right comparison when one mode "
            "dominates; these references say whether the agreement is specific to the "
            "correct multiplicity"
        ),
        "per_multiplicity": rows,
    }
