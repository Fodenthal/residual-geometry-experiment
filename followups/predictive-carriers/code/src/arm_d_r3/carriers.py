"""Carrier construction inside the frozen aperture, and what a carrier is like.

Four families plus one diagnostic, all orthonormal column bases of the same frozen
256-dimensional aperture, so every carrier is a coordinate choice inside one fixed
observation window rather than a different observation:

``inherited``          R1's frozen rank-16 carrier.  R2.2's coordinate system, and
                      the reference point of every comparison here.
``variance``           the top-``r`` principal directions of the aperture source
                      difference, fitted on TRAINING documents only.  This is the
                      same construction R2.2's Q axis deleted as its matched PCA
                      control, so "the subspace whose deletion cost 0.85 of the
                      predictive gain" and "the subspace tested here" are the same
                      object.
``haar``               a rank-matched Haar-random subspace.  The floor: any
                      improvement has to beat what an arbitrary subspace of the
                      same width buys.
``complement_refit``   the carrier the R1 estimator recovers after the inherited
                      one is deleted, that is, R2.2's ``c_perp`` basis.
``predictive_refit``   the R1 estimator's own carrier refit on THIS pool, which
                      separates "the inherited basis is the wrong kind of object"
                      from "the inherited basis is stale for this pool".

Two rules hold for every family.  Nothing is fitted on evaluation documents, and
the aperture itself is never refitted: rotating the aperture would move the
coordinate system out from under the inherited carrier and make the comparison
meaningless (the reason R2.2 section 4.2 freezes it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from arm_d.operators import haar_subspace, mean_subspace_correlation
from arm_d.pipeline import PairedDataset

from arm_d_r2.carrier import project_out, recover_carrier

from arm_d_r3.contract import (
    APERTURE_WIDTH,
    HAAR_CARRIER_DRAWS,
    HAAR_NULL_DRAWS,
    MSC_THRESHOLD,
    PRIMARY_RANK,
    REFERENCE_LAG,
    SEED,
    SECTION_CARRIERS,
    assert_distinct,
    carrier_key,
    haar_carrier_seed,
    label,
    unavailable,
)

SECTION = SECTION_CARRIERS


def _orthonormal(basis: np.ndarray, *, width: int = APERTURE_WIDTH) -> np.ndarray:
    array = np.asarray(basis, dtype=np.float64)[:width]
    gram = array.T @ array
    if not np.allclose(gram, np.eye(array.shape[1]), atol=1e-8):
        array = np.linalg.qr(array)[0]
    return array


def variance_basis(
    source_train: np.ndarray, rank: int, *, width: int = APERTURE_WIDTH
) -> np.ndarray:
    """Top-``rank`` principal directions of the training aperture differences.

    Identical construction to R2.2's matched PCA deletion control: centre the
    training source block and take the leading right singular vectors.
    """

    data = np.asarray(source_train, dtype=np.float64)[:, :width]
    centred = data - data.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    return np.ascontiguousarray(right.T[:, : int(rank)])


def variance_spectrum(source_train: np.ndarray, *, width: int = APERTURE_WIDTH) -> dict[str, object]:
    """The aperture variance spectrum, so "top variance" is a stated quantity."""

    data = np.asarray(source_train, dtype=np.float64)[:, :width]
    centred = data - data.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    cumulative = np.cumsum(energy) / total if total > 0 else np.zeros_like(energy)
    return {
        "singular_values": [float(value) for value in singular[:64]],
        "variance_fraction_by_rank": {
            str(rank): float(cumulative[rank - 1]) for rank in (1, 2, 4, 8, 16, 32, 64)
            if rank <= cumulative.shape[0]
        },
        "total_energy": total,
        "participation_ratio": float(total**2 / float(np.sum(energy**2))) if total > 0 else 0.0,
    }


def variance_captured(source: np.ndarray, basis: np.ndarray, *, width: int = APERTURE_WIDTH) -> float:
    """Fraction of aperture-difference energy the carrier retains."""

    data = np.asarray(source, dtype=np.float64)[:, :width]
    centred = data - data.mean(axis=0, keepdims=True)
    total = float(np.sum(centred**2))
    if total <= 0:
        return float("nan")
    projected = centred @ np.asarray(basis, dtype=np.float64)
    return float(np.sum(projected**2) / total)


def haar_null_msc(
    rank: int, *, width: int = APERTURE_WIDTH, draws: int = HAAR_NULL_DRAWS, seed: int = SEED
) -> dict[str, float]:
    """Mean subspace correlation between two independent Haar subspaces.

    The reference for every overlap statement below: two arbitrary rank-``r``
    subspaces of a 256-dimensional space already share ``r / 256`` on average, so
    an overlap has to be read against that.
    """

    rng = np.random.default_rng(seed + 7919 * int(rank))
    values = np.array(
        [
            mean_subspace_correlation(
                haar_subspace(width, int(rank), rng), haar_subspace(width, int(rank), rng)
            )
            for _ in range(int(draws))
        ]
    )
    return {
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "mean": float(values.mean()),
        "draws": int(draws),
        "analytic_expectation": float(int(rank) / int(width)),
    }


def subspace_overlap(left: np.ndarray, right: np.ndarray, *, what: str) -> dict[str, object]:
    """Principal angles and mean subspace correlation, with the I3 assertion."""

    assertion = assert_distinct(left, right, what=what, left_name="left_basis", right_name="right_basis")
    u = _orthonormal(left)
    v = _orthonormal(right)
    cosines = np.clip(np.linalg.svd(u.T @ v, compute_uv=False), -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    return {
        "mean_subspace_correlation": float(mean_subspace_correlation(u, v)),
        "principal_angles_degrees": [float(value) for value in angles],
        "min_principal_angle_degrees": float(angles.min()),
        "max_principal_angle_degrees": float(angles.max()),
        "n_angles": int(angles.shape[0]),
        "distinctness": assertion,
    }


def training_halves(documents: np.ndarray, *, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic document-disjoint split of the training documents.

    The same construction R2.2's ``split_half_companion`` and its rank stability
    curve use, so a stability number here is comparable with one there.
    """

    unique = np.unique(documents)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique.shape[0])
    half = set(unique[order[: unique.shape[0] // 2]].tolist())
    in_a = np.array([value in half for value in documents])
    return in_a, ~in_a


def variance_stability(
    source_train: np.ndarray,
    documents_train: np.ndarray,
    *,
    rank: int,
    seed: int = SEED,
    width: int = APERTURE_WIDTH,
) -> dict[str, object]:
    """Does the variance carrier reproduce across independent training halves?"""

    in_a, in_b = training_halves(documents_train, seed=seed)
    if min(int(in_a.sum()), int(in_b.sum())) < 8 * int(rank):
        return {"unavailable": unavailable(f"variance_stability_rank_{rank}", "a half is too small")}
    basis_a = variance_basis(source_train[in_a], rank, width=width)
    basis_b = variance_basis(source_train[in_b], rank, width=width)
    overlap = subspace_overlap(
        basis_a, basis_b, what=f"variance carrier split-half stability at rank {rank}"
    )
    null = haar_null_msc(rank, width=width, seed=seed)
    overlap["haar_null"] = null
    overlap["exceeds_haar_null"] = bool(overlap["mean_subspace_correlation"] > null["q95"])
    overlap["half_sizes"] = [int(in_a.sum()), int(in_b.sum())]
    return overlap


def estimator_stability(
    view: PairedDataset,
    *,
    rank: int,
    lag: int = REFERENCE_LAG,
    suffix_length: int = REFERENCE_LAG,
    msc_threshold: float = MSC_THRESHOLD,
    seed: int = SEED,
    width: int = APERTURE_WIDTH,
) -> dict[str, object]:
    """Split-half stability of the R1 estimator's own recovered carrier."""

    rows = (view.condition == "main") & (view.suffix_length == int(suffix_length))
    train = rows & (view.split == "train")
    in_a, in_b = training_halves(view.document_ids[train], seed=seed)
    bases = []
    for mask in (in_a, in_b):
        selector = np.zeros(view.document_ids.shape[0], dtype=bool)
        selector[np.flatnonzero(train)[mask]] = True
        half = PairedDataset(
            document_ids=view.document_ids[selector],
            condition=view.condition[selector],
            suffix_length=view.suffix_length[selector],
            band=view.band[selector],
            split=view.split[selector],
            source=view.source[selector],
            history=view.history[selector],
            future=view.future[selector],
            lags=view.lags,
            history_offsets=view.history_offsets,
        )
        bases.append(
            recover_carrier(
                half, lag=lag, suffix_length=suffix_length, aperture=width, rank=int(rank),
                msc_threshold=msc_threshold,
            )
        )
    if bases[0] is None or bases[1] is None:
        return {"unavailable": unavailable(f"estimator_stability_rank_{rank}", "a half did not fit")}
    overlap = subspace_overlap(
        bases[0], bases[1], what=f"estimator carrier split-half stability at rank {rank}"
    )
    null = haar_null_msc(int(rank), width=width, seed=seed)
    overlap["haar_null"] = null
    overlap["exceeds_haar_null"] = bool(overlap["mean_subspace_correlation"] > null["q95"])
    return overlap


def build_carriers(
    *,
    view: PairedDataset,
    inherited: np.ndarray,
    source_train: np.ndarray,
    documents_train: np.ndarray,
    ranks: Sequence[int],
    primary_rank: int = PRIMARY_RANK,
    haar_draws: int = HAAR_CARRIER_DRAWS,
    msc_threshold: float = MSC_THRESHOLD,
    seed: int = SEED,
    width: int = APERTURE_WIDTH,
) -> dict[str, dict[str, object]]:
    """Every carrier this run compares, keyed by :func:`carrier_key`.

    ``view`` is the R1 estimator's own paired object (destination-hook futures);
    it is used only for the two refit families.  ``source_train`` is the
    source-hook aperture difference on training rows of the analysed cell, which
    is what the variance family is fitted on.
    """

    carriers: dict[str, dict[str, object]] = {}

    inherited_basis = _orthonormal(inherited, width=width)
    carriers[carrier_key("inherited", inherited_basis.shape[1])] = {
        "family": "inherited",
        "rank": int(inherited_basis.shape[1]),
        "basis": inherited_basis,
        "construction": "R1 frozen carrier, inherited unchanged through R2.2",
        "fitted_on": "R1 confirmatory pool (not this pool)",
    }

    for rank in ranks:
        carriers[carrier_key("variance", rank)] = {
            "family": "variance",
            "rank": int(rank),
            "basis": variance_basis(source_train, int(rank), width=width),
            "construction": "leading right singular vectors of the centred training aperture block",
            "fitted_on": "1100 training documents of this pool",
        }
        for index in range(int(haar_draws) if int(rank) == int(primary_rank) else 1):
            rng = np.random.default_rng(haar_carrier_seed(int(rank), index))
            carriers[carrier_key("haar", rank, index)] = {
                "family": "haar",
                "rank": int(rank),
                "index": int(index),
                "basis": haar_subspace(width, int(rank), rng),
                "construction": "Haar-random orthonormal basis inside the frozen aperture",
                "fitted_on": "nothing (random by construction)",
            }

    complement = recover_carrier(
        project_out(view, inherited_basis, width=width),
        lag=REFERENCE_LAG, suffix_length=REFERENCE_LAG, aperture=width, rank=int(primary_rank),
        msc_threshold=msc_threshold,
    )
    if complement is not None:
        carriers[carrier_key("complement_refit", primary_rank)] = {
            "family": "complement_refit",
            "rank": int(primary_rank),
            "basis": _orthonormal(complement, width=width),
            "construction": "R1 estimator refit from scratch on the inherited carrier's complement",
            "fitted_on": "1100 training documents of this pool",
        }

    for rank in (int(primary_rank),):
        recovered = recover_carrier(
            view, lag=REFERENCE_LAG, suffix_length=REFERENCE_LAG, aperture=width, rank=rank,
            msc_threshold=msc_threshold,
        )
        if recovered is not None:
            carriers[carrier_key("predictive_refit", rank)] = {
                "family": "predictive_refit",
                "rank": rank,
                "basis": _orthonormal(recovered, width=width),
                "construction": "R1 estimator's carrier, refit on this pool",
                "fitted_on": "1100 training documents of this pool",
            }
    return carriers


def characterise(
    carriers: Mapping[str, Mapping[str, object]],
    *,
    view: PairedDataset,
    source_train: np.ndarray,
    source_eval: np.ndarray,
    documents_train: np.ndarray,
    inherited_key: str,
    seed: int = SEED,
    width: int = APERTURE_WIDTH,
) -> dict[str, object]:
    """Geometry of every carrier before any composition number is read.

    Deliberately ordered: variance captured, overlap with the inherited carrier
    against a Haar reference, and split-half stability of the construction.  None
    of it involves the dynamics.
    """

    inherited_basis = np.asarray(carriers[inherited_key]["basis"], dtype=np.float64)
    records: dict[str, object] = {}
    for key, carrier in carriers.items():
        basis = np.asarray(carrier["basis"], dtype=np.float64)
        rank = int(carrier["rank"])
        entry: dict[str, object] = {
            "family": carrier["family"],
            "rank": rank,
            "construction": carrier["construction"],
            "fitted_on": carrier["fitted_on"],
            "state_dimension_at_order_5": int(5 * rank),
            "variance_captured_train": variance_captured(source_train, basis, width=width),
            "variance_captured_eval": variance_captured(source_eval, basis, width=width),
        }
        if key == inherited_key:
            entry["overlap_with_inherited"] = {
                "identical_carrier": True,
                "reading": "the reference carrier compared with itself is not measured",
            }
        else:
            entry["overlap_with_inherited"] = subspace_overlap(
                basis, inherited_basis, what=f"{key} against the inherited carrier"
            )
            entry["overlap_with_inherited"]["haar_null"] = haar_null_msc(
                min(rank, int(inherited_basis.shape[1])), width=width, seed=seed
            )
        if carrier["family"] == "variance":
            entry["stability"] = variance_stability(
                source_train, documents_train, rank=rank, seed=seed, width=width
            )
        elif carrier["family"] in {"inherited", "predictive_refit"}:
            entry["stability"] = estimator_stability(
                view, rank=rank, seed=seed, width=width
            )
            if carrier["family"] == "inherited":
                entry["stability"]["reading"] = (
                    "stability of the CONSTRUCTION (the R1 estimator refit on each half of "
                    "this pool), not of the frozen basis, which is fixed by definition"
                )
        elif carrier["family"] == "haar":
            entry["stability"] = {
                "not_applicable_by_construction": True,
                "reading": "a Haar carrier has no construction to reproduce; its overlap "
                           "with an independent draw is the null itself",
                "haar_null": haar_null_msc(rank, width=width, seed=seed),
            }
        else:
            entry["stability"] = {
                "not_measured": True,
                "reading": "the complement refit inherits the inherited carrier's deletion, so "
                           "a split-half of its construction repeats the estimator stability "
                           "already reported for the predictive refit",
            }
        records[key] = entry

    return label(
        "CARRIERS_CHARACTERISED",
        section=SECTION,
        evidence={
            "carriers": records,
            "variance_spectrum": variance_spectrum(source_train, width=width),
            "aperture_width": int(width),
            "inherited_key": inherited_key,
        },
    )
