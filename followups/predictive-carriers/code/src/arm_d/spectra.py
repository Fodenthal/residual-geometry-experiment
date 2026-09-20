"""Transition spectra, mode clustering, and the conditional frequency branch.

R1 §20-§22.  Nothing here may be reported unless the shared-space gate passed;
:func:`spectrum_report` refuses to emit eigenvalues under
``SOURCE_DESTINATION_TRANSPORT``, which is the enforcement point for the
prohibition rather than a convention the report is trusted to follow.
"""

from __future__ import annotations

import numpy as np

from src.persistent_state.arm_b.stability_r2 import cluster_degenerate_modes
from src.persistent_state.live_state.temporal_code import principal_angles, projection_overlap


def antisymmetric_energy(lag_covariance: np.ndarray, epsilon: float = 1e-12) -> float:
    r"""``R_asym(k) = ||C_k^asym||_F / (||C_k^sym||_F + eps)`` (R1 §6, P0.2)."""

    matrix = np.asarray(lag_covariance, dtype=np.float64)
    symmetric = 0.5 * (matrix + matrix.T)
    antisymmetric = 0.5 * (matrix - matrix.T)
    return float(np.linalg.norm(antisymmetric, "fro") / (np.linalg.norm(symmetric, "fro") + epsilon))


def spectrum_report(
    operator: np.ndarray,
    *,
    shared_space_label: str,
    base_lag: int,
    cluster_gap: float = 0.05,
    cluster_span: float = 0.10,
    cluster_size: int = 4,
) -> dict[str, object]:
    """Eigen-structure of a fitted operator, withheld under transport."""

    if shared_space_label != "FIXED_COORDINATE_DYNAMICS":
        return {
            "reported": False,
            "reason": "shared-space gate did not pass; R1 §12 forbids eigenvalues under SOURCE_DESTINATION_TRANSPORT",
            "shared_space_label": shared_space_label,
        }
    matrix = np.asarray(operator, dtype=np.float64)
    eigenvalues = np.linalg.eigvals(matrix)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    modulus = np.abs(eigenvalues)
    unstable = modulus >= 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.where(
            (modulus > 0) & (modulus < 1),
            -float(base_lag) / np.log(np.clip(modulus, 1e-12, None)),
            np.nan,
        )
    clusters = cluster_degenerate_modes(
        modulus, local_gap=cluster_gap, max_span=cluster_span, max_size=cluster_size
    )
    return {
        "reported": True,
        "shared_space_label": shared_space_label,
        "base_lag": int(base_lag),
        "eigenvalue_modulus": [float(value) for value in modulus],
        "eigenvalue_phase_radians": [float(np.angle(value)) for value in eigenvalues],
        "decay_time_tokens": [None if not np.isfinite(value) else float(value) for value in decay],
        "unstable_count": int(unstable.sum()),
        "unstable_fraction": float(unstable.mean()),
        "unstable_note": "a large unstable fraction is a numerical-health warning, not explosive model dynamics",
        "complex_pair_count": int(np.sum(np.abs(np.imag(eigenvalues)) > 1e-9) // 2),
        "mode_clusters": [
            {"indices": list(cluster.indices), "relative_span": float(cluster.relative_span)}
            for cluster in clusters
        ],
    }


def frequency_branch(
    spectrum: dict[str, object],
    *,
    antisymmetric_ratio: float,
    bootstrap_phase_halfwidths: np.ndarray | None,
    whitening_healthy: bool,
    min_antisymmetric_ratio: float = 0.10,
    max_phase_halfwidth: float = 0.20,
) -> dict[str, object]:
    """R1 §20: frequency is reported only when every gate passes."""

    gates = {
        "shared_space_gate": bool(spectrum.get("reported")),
        "complex_structure_present": bool(spectrum.get("complex_pair_count", 0) > 0),
        "antisymmetric_energy": bool(antisymmetric_ratio >= min_antisymmetric_ratio),
        "solver_health": bool(whitening_healthy),
        "bootstrap_phase_precision": bool(
            bootstrap_phase_halfwidths is not None
            and np.size(bootstrap_phase_halfwidths) > 0
            and float(np.max(bootstrap_phase_halfwidths)) <= max_phase_halfwidth
        ),
    }
    passed = all(gates.values())
    payload: dict[str, object] = {
        "gates": gates,
        "passed": passed,
        "antisymmetric_ratio": float(antisymmetric_ratio),
        "min_antisymmetric_ratio": float(min_antisymmetric_ratio),
        "max_phase_halfwidth": float(max_phase_halfwidth),
    }
    if not passed:
        payload["reason"] = "frequency branch disabled; no frequencies, periods, or decay times reported"
        return payload
    phases = np.asarray(spectrum["eigenvalue_phase_radians"], dtype=np.float64)
    payload["cycles_per_token"] = [float(phase / (2 * np.pi)) for phase in phases]
    payload["period_tokens"] = [
        None if abs(phase) < 1e-9 else float(2 * np.pi / abs(phase)) for phase in phases
    ]
    return payload


def split_half_stability(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    operator_a: np.ndarray | None = None,
    operator_b: np.ndarray | None = None,
) -> dict[str, object]:
    """R1 §23 subspace and spectrum agreement across independent train halves."""

    angles = principal_angles(basis_a, basis_b)
    report: dict[str, object] = {
        "principal_angles_degrees": [float(np.degrees(angle)) for angle in angles],
        "projection_overlap": float(projection_overlap(basis_a, basis_b)),
        "max_principal_angle_degrees": float(np.degrees(angles.max())) if angles.size else float("nan"),
    }
    if operator_a is not None and operator_b is not None:
        modulus_a = np.sort(np.abs(np.linalg.eigvals(operator_a)))[::-1]
        modulus_b = np.sort(np.abs(np.linalg.eigvals(operator_b)))[::-1]
        width = min(modulus_a.size, modulus_b.size)
        if width >= 2:
            report["eigenvalue_modulus_correlation"] = float(
                np.corrcoef(modulus_a[:width], modulus_b[:width])[0, 1]
            )
        report["eigenvalue_modulus_a"] = [float(value) for value in modulus_a[:width]]
        report["eigenvalue_modulus_b"] = [float(value) for value in modulus_b[:width]]
    return report
