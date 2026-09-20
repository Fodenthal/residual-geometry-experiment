"""The Arm D estimator pipeline and its full-refit null bank (R1 §14, §17, §32).

One cell of the pipeline is a choice of aperture ``p``, predictive rank ``r``,
lag ``k``, and suffix length ``L``.  Within a cell the estimator is:

1. whitened asymmetric screening on train rows (R1 §11.1);
2. the shared-space gate, producing ``Q`` and the shared-space label (§12);
3. the nested operator family fitted in ``Q`` coordinates (§11.2);
4. the mandatory trajectory baselines in the *same* coordinates (§15);
5. the cell statistic ``T = R^2(full operator) - R^2(strongest applicable
   baseline)``, where applicable means the baselines that consume the
   operator's own information set (persistence, update-only).

``T`` is deliberately incremental rather than raw: a decaying difference makes
persistence a strong predictor, so a raw ``R^2`` would clear any pairing null
without saying anything about a transition law.  The richer trajectory
baselines are scored in every cell and reported, but they see five slots of
state history against the operator's one, so they are consumed at D5 and the
Markov-order label rather than in the D1 headline (see ``arm_d.baselines``).

The pipeline-level maximum statistic ``max_{p,r,k,L} T`` is compared against the
same maximum recomputed inside every full-refit null replicate (§32), so the
search over the grid is paid for rather than assumed away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from arm_d.baselines import TRAJECTORY_BASELINES, evaluate_baselines, strongest_baseline
from arm_d.operators import (
    build_shared_space,
    gate_width_is_calibrated,
    pooled_reducer,
    fit_operator_family,
    haar_subspace,
    fit_screening_operator,
    mixing_evidence,
    score_family,
)
from arm_d.scoring import empirical_p_value

HISTORY_OFFSETS = np.array([-4, -3, -2, -1, 0], dtype=np.int64)
LAGS = (1, 2, 4, 8, 16, 32, 64, 128)
APERTURES = (256, 512, 1024)
RANKS = (1, 2, 4, 8, 16, 32)
SPLICE_CONDITIONS = ("main", "noop", "sham", "matched")
NULL_FAMILIES = ("lag_shuffle", "lag_block_shuffle", "xdoc_derangement")


@dataclass
class PairedDataset:
    """Paired remote-prefix differences in aperture coordinates.

    ``source`` is the source-hook difference at the source token.  ``history``
    is the *source-hook* difference over :data:`HISTORY_OFFSETS`, whose last
    slot is ``source`` itself -- so persistence is exactly the operator with
    ``A = I`` and the two are compared on identical inputs.  ``future`` is the
    destination-hook difference at every lag in ``lags``.
    """

    document_ids: np.ndarray
    condition: np.ndarray
    suffix_length: np.ndarray
    band: np.ndarray
    split: np.ndarray
    source: np.ndarray
    history: np.ndarray
    future: np.ndarray
    lags: tuple[int, ...] = LAGS
    history_offsets: np.ndarray = field(default_factory=lambda: HISTORY_OFFSETS.copy())

    def __post_init__(self) -> None:
        n = self.document_ids.shape[0]
        for name in ("condition", "suffix_length", "band", "split"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"{name} must have one entry per row")
        if self.source.shape[0] != n or self.history.shape[0] != n or self.future.shape[0] != n:
            raise ValueError("coordinate arrays must have one entry per row")
        if self.history.shape[1] != self.history_offsets.shape[0]:
            raise ValueError("history slots must match history_offsets")
        if self.future.shape[1] != len(self.lags):
            raise ValueError("future slots must match lags")

    @property
    def aperture_width(self) -> int:
        return int(self.source.shape[1])

    def select(self, mask: np.ndarray) -> "PairedDataset":
        return PairedDataset(
            document_ids=self.document_ids[mask],
            condition=self.condition[mask],
            suffix_length=self.suffix_length[mask],
            band=self.band[mask],
            split=self.split[mask],
            source=self.source[mask],
            history=self.history[mask],
            future=self.future[mask],
            lags=self.lags,
            history_offsets=self.history_offsets,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            document_ids=self.document_ids.astype("U64"),
            condition=self.condition.astype("U16"),
            suffix_length=self.suffix_length.astype(np.int64),
            band=self.band.astype(np.int64),
            split=self.split.astype("U8"),
            source=self.source.astype(np.float32),
            history=self.history.astype(np.float32),
            future=self.future.astype(np.float32),
            lags=np.asarray(self.lags, dtype=np.int64),
            history_offsets=self.history_offsets.astype(np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PairedDataset":
        with np.load(Path(path), allow_pickle=False) as handle:
            return cls(
                document_ids=handle["document_ids"],
                condition=handle["condition"],
                suffix_length=handle["suffix_length"],
                band=handle["band"],
                split=handle["split"],
                source=handle["source"].astype(np.float64),
                history=handle["history"].astype(np.float64),
                future=handle["future"].astype(np.float64),
                lags=tuple(int(value) for value in handle["lags"]),
                history_offsets=handle["history_offsets"],
            )


@dataclass(frozen=True)
class CellResult:
    aperture: int
    rank: int
    lag: int
    suffix_length: int
    condition: str
    statistic: float
    statistic_over_trajectory: float
    r2_operator: float
    r2_best_baseline: float
    best_baseline_name: str
    richer_baseline_r2: dict[str, float]
    mixing_supported: bool
    shared_space_label: str
    gate_passed: bool
    mean_subspace_correlation: float
    baseline_r2: dict[str, float]
    family_r2: dict[str, float]
    whitening: dict[str, float | bool]
    n_train: int
    n_eval: int

    def to_dict(self) -> dict[str, object]:
        payload = {
            "aperture": self.aperture,
            "rank": self.rank,
            "lag": self.lag,
            "suffix_length": self.suffix_length,
            "condition": self.condition,
            "statistic_delta_r2": self.statistic,
            "statistic_delta_r2_over_trajectory": self.statistic_over_trajectory,
            "r2_operator": self.r2_operator,
            "r2_best_baseline": self.r2_best_baseline,
            "best_baseline": self.best_baseline_name,
            "shared_space_label": self.shared_space_label,
            "gate_passed": self.gate_passed,
            "mixing_supported": self.mixing_supported,
            "mean_subspace_correlation": self.mean_subspace_correlation,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
        }
        payload.update({f"baseline_r2_{name}": value for name, value in self.baseline_r2.items()})
        payload["best_richer_baseline_r2"] = (
            max(self.richer_baseline_r2.values()) if self.richer_baseline_r2 else float("nan")
        )
        payload.update({f"family_r2_{name}": value for name, value in self.family_r2.items()})
        payload.update(self.whitening)
        return payload


def _project(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return values @ basis


def run_cell(
    data: PairedDataset,
    *,
    aperture: int,
    rank: int,
    lag: int,
    suffix_length: int,
    condition: str,
    msc_threshold: float,
    train_split: str = "train",
    eval_split: str = "val",
    seed: int = 42,
    subspace_override: np.ndarray | None = None,
    mixing_bootstrap: int = 400,
    reducer: np.ndarray | None = None,
    capture: dict | None = None,
) -> CellResult | None:
    """Run the estimator for one grid cell and return its statistic.

    ``subspace_override`` replaces the fitted ``Q`` with a supplied basis (the
    PCA and Haar-random rank-matched subspace controls of R1 §17.1 N-D0/N-D1),
    keeping every other step identical.
    """

    lag_index = data.lags.index(int(lag))
    rows = (data.condition == condition) & (data.suffix_length == int(suffix_length))
    train = rows & (data.split == train_split)
    evaluate = rows & (data.split == eval_split)
    if train.sum() < 32 or evaluate.sum() < 16:
        return None

    p = int(aperture)
    source_train = data.source[train][:, :p]
    future_train = data.future[train][:, lag_index, :p]
    source_eval = data.source[evaluate][:, :p]
    future_eval = data.future[evaluate][:, lag_index, :p]
    history_train = data.history[train][:, :, :p]
    history_eval = data.history[evaluate][:, :, :p]
    if not np.allclose(history_train[:, -1, :], source_train, atol=1e-5):
        raise AssertionError(
            "the last history slot must be the source-token state itself, so persistence is "
            "exactly the operator with A = I"
        )
    documents_train = data.document_ids[train]
    documents_eval = data.document_ids[evaluate]

    if subspace_override is None:
        screening = fit_screening_operator(source_train, future_train, rank, reducer=reducer)
        if not screening.healthy or screening.source_space.shape[1] < rank:
            return None
        shared = build_shared_space(
            screening.source_space,
            screening.destination_space,
            rank=rank,
            msc_threshold=msc_threshold,
        )
        basis = shared.basis
        whitening = screening.health_report()
        label, gate_passed, msc = shared.label, shared.gate_passed, shared.mean_subspace_correlation
        if not gate_width_is_calibrated(screening.whitening_dimension, rank):
            # Too few rows for the 4r reduction, so the frozen 0.25 threshold no
            # longer sits at chance.  The cell still yields a statistic; only the
            # coordinate label, and the spectral language it unlocks, is withheld.
            label, gate_passed = "SHARED_SPACE_UNDERSIZED_CELL", False
        whitening["gate_width_calibrated"] = bool(
            gate_width_is_calibrated(screening.whitening_dimension, rank)
        )
        whitening["msc_chance_level"] = float(
            rank / max(int(screening.whitening_dimension), 1)
        )
    else:
        basis = np.asarray(subspace_override, dtype=np.float64)[:, :rank]
        whitening = {}
        label, gate_passed, msc = "SUBSPACE_CONTROL", False, float("nan")
    if basis.shape[1] < rank:
        return None

    zs_train = _project(source_train, basis)
    zd_train = _project(future_train, basis)
    zs_eval = _project(source_eval, basis)
    zd_eval = _project(future_eval, basis)
    zh_train = np.einsum("nhp,pr->nhr", history_train, basis)
    zh_eval = np.einsum("nhp,pr->nhr", history_eval, basis)

    family = fit_operator_family(zs_train, zd_train, documents_train, lag=lag, seed=seed)
    family = score_family(family, zs_eval, zd_eval, documents_eval)
    baselines = evaluate_baselines(
        history_train=zh_train,
        target_train=zd_train,
        documents_train=documents_train,
        history_eval=zh_eval,
        target_eval=zd_eval,
        documents_eval=documents_eval,
        offsets=data.history_offsets,
        lag=lag,
        seed=seed,
    )
    best_baseline = strongest_baseline(baselines)
    if capture is not None:
        # Hand back the per-document scores the R1 section 14.3 bootstrap needs.
        # Recomputing them in the caller would duplicate this fit path and could
        # drift from it silently, which is exactly the failure this experiment
        # keeps finding, so the numbers are taken from the fit that produced the
        # recorded statistic rather than from a second implementation of it.
        capture["operator_score"] = family.fits["full"].score
        capture["baselines"] = baselines
        capture["best_source_state_baseline"] = best_baseline.name
        # Eval-side rows for the R1.1 splice-validity diagnostics.  These are the
        # SAME arrays the scores above were computed from, so a band-restricted
        # rescoring cannot drift from the headline fit.  ``source_norm`` is the
        # intervention magnitude at the aperture, one value per evaluation row.
        capture["eval_rows"] = {
            "source_projected": zs_eval,
            "destination_projected": zd_eval,
            "history_projected": zh_eval,
            "documents": documents_eval,
            "source_norm": np.linalg.norm(source_eval, axis=1),
            "operator": family.fits["full"].operator,
        }
    operator_r2 = family["full"].r2
    mixing = mixing_evidence(family, replicates=mixing_bootstrap, seed=seed)
    richer = {
        name: float(result.r2)
        for name, result in baselines.items()
        if name in TRAJECTORY_BASELINES
    }
    best_trajectory = max(richer.values()) if richer else 0.0
    return CellResult(
        aperture=p,
        rank=int(rank),
        lag=int(lag),
        suffix_length=int(suffix_length),
        condition=str(condition),
        statistic=float(operator_r2 - best_baseline.r2),
        statistic_over_trajectory=float(operator_r2 - best_trajectory),
        r2_operator=float(operator_r2),
        r2_best_baseline=float(best_baseline.r2),
        best_baseline_name=best_baseline.name,
        shared_space_label=label,
        gate_passed=bool(gate_passed),
        mean_subspace_correlation=float(msc),
        baseline_r2={name: float(result.r2) for name, result in baselines.items()},
        richer_baseline_r2=richer,
        mixing_supported=bool(mixing["mixing_supported"]),
        family_r2={name: float(fit.r2) for name, fit in family.fits.items()},
        whitening=whitening,
        n_train=int(train.sum()),
        n_eval=int(evaluate.sum()),
    )


def run_grid(
    data: PairedDataset,
    *,
    apertures: tuple[int, ...],
    ranks: tuple[int, ...],
    lags: tuple[int, ...],
    suffix_lengths: tuple[int, ...],
    condition: str,
    msc_threshold: float,
    seed: int = 42,
) -> list[CellResult]:
    """Every grid cell for one splice condition."""

    results: list[CellResult] = []
    max_width = 4 * int(max(ranks))
    for aperture in apertures:
        if aperture > data.aperture_width:
            continue
        for suffix_length in suffix_lengths:
            for lag in lags:
                reducer = grid_reducer(
                    data,
                    aperture=aperture,
                    suffix_length=suffix_length,
                    lag=lag,
                    condition=condition,
                    max_width=max_width,
                )
                for rank in ranks:
                    cell = run_cell(
                        data,
                        aperture=aperture,
                        rank=rank,
                        lag=lag,
                        suffix_length=suffix_length,
                        condition=condition,
                        msc_threshold=msc_threshold,
                        seed=seed,
                        reducer=reducer,
                    )
                    if cell is not None:
                        results.append(cell)
    return results


def grid_reducer(
    data,
    *,
    aperture: int,
    suffix_length: int,
    lag: int,
    condition: str,
    max_width: int,
):
    """The pooled reducer `run_grid` shares across every rank in one group.

    A replay MUST build its reducer through this function rather than passing
    `reducer=None`, or `run_cell` will construct a narrower one at 4*rank and the
    replayed fit will not reproduce the grid's recorded statistic.
    """

    rows = (data.condition == condition) & (data.suffix_length == int(suffix_length))
    train = rows & (data.split == "train")
    if train.sum() < 32:
        return None
    lag_index = data.lags.index(int(lag))
    source_train = data.source[train][:, :aperture]
    future_train = data.future[train][:, lag_index, :aperture]
    if not (np.sum(source_train**2) > 0 and np.sum(future_train**2) > 0):
        return None
    return pooled_reducer(source_train, future_train, max_width)

def pipeline_maximum(results: list[CellResult]) -> float:
    if not results:
        return float("-inf")
    return float(max(result.statistic for result in results))


# --------------------------------------------------------------------------
# Null constructions (R1 §17)
# --------------------------------------------------------------------------


def apply_lag_shuffle(data: PairedDataset, rng: np.random.Generator, block: int = 1) -> PairedDataset:
    """Destroy lag-specific structure by permuting the future slots per row.

    This is R1 §17.1's within-document order destruction adapted to the paired
    object: each spliced sequence contributes one trajectory, so the order that
    can be destroyed is the order of its evaluated lags.  ``block > 1`` permutes
    contiguous blocks of lags instead of single lags (N-D3).
    """

    future = data.future.copy()
    n_lags = future.shape[1]
    if block <= 1:
        for row in range(future.shape[0]):
            future[row] = future[row][rng.permutation(n_lags)]
    else:
        n_blocks = int(np.ceil(n_lags / block))
        for row in range(future.shape[0]):
            order = rng.permutation(n_blocks)
            pieces = [future[row][index * block : (index + 1) * block] for index in order]
            future[row] = np.concatenate(pieces, axis=0)[:n_lags]
    shuffled = PairedDataset(
        document_ids=data.document_ids,
        condition=data.condition,
        suffix_length=data.suffix_length,
        band=data.band,
        split=data.split,
        source=data.source,
        history=data.history,
        future=future,
        lags=data.lags,
        history_offsets=data.history_offsets,
    )
    return shuffled


def apply_document_derangement(data: PairedDataset, rng: np.random.Generator) -> PairedDataset:
    """Coherent cross-document future pairing (R1 §17.1 N-D4).

    One donor document is drawn per source document and reused across every
    condition, suffix length and lag, so realistic serial structure survives
    while same-pair temporal identity is destroyed.  Donors are matched within
    position band so the null does not also destroy the position contract.

    Only the future is deranged.  The history stack belongs to the source
    trajectory and stays with it, both because that is what "different-document
    future pairing" means and because the last history slot is the source state
    itself.
    """

    documents = np.unique(data.document_ids)
    band_of = {}
    for document in documents:
        band_of[document] = int(data.band[data.document_ids == document][0])
    donor: dict[str, str] = {}
    for band in sorted({value for value in band_of.values()}):
        members = np.array([document for document in documents if band_of[document] == band])
        if members.size < 2:
            for document in members:
                donor[document] = document
            continue
        order = rng.permutation(members.size)
        rolled = np.roll(order, -1)
        for position, target in enumerate(order):
            donor[members[target]] = members[rolled[position]]

    key_to_row = {
        (document, condition, int(length)): row
        for row, (document, condition, length) in enumerate(
            zip(data.document_ids, data.condition, data.suffix_length)
        )
    }
    future = data.future.copy()
    history = data.history
    for row, (document, condition, length) in enumerate(
        zip(data.document_ids, data.condition, data.suffix_length)
    ):
        source_row = key_to_row.get((donor[document], condition, int(length)))
        if source_row is None:
            continue
        future[row] = data.future[source_row]
    return PairedDataset(
        document_ids=data.document_ids,
        condition=data.condition,
        suffix_length=data.suffix_length,
        band=data.band,
        split=data.split,
        source=data.source,
        history=history,
        future=future,
        lags=data.lags,
        history_offsets=data.history_offsets,
    )


def null_replicate(
    data: PairedDataset,
    family: str,
    seed: int,
    *,
    apertures: tuple[int, ...],
    ranks: tuple[int, ...],
    lags: tuple[int, ...],
    suffix_lengths: tuple[int, ...],
    condition: str,
    msc_threshold: float,
) -> float:
    """One full-refit null replicate, returning its pipeline maximum."""

    rng = np.random.default_rng(seed)
    if family == "lag_shuffle":
        permuted = apply_lag_shuffle(data, rng, block=1)
    elif family == "lag_block_shuffle":
        permuted = apply_lag_shuffle(data, rng, block=2)
    elif family == "xdoc_derangement":
        permuted = apply_document_derangement(data, rng)
    else:
        raise ValueError(f"unknown null family: {family!r}")
    results = run_grid(
        permuted,
        apertures=apertures,
        ranks=ranks,
        lags=lags,
        suffix_lengths=suffix_lengths,
        condition=condition,
        msc_threshold=msc_threshold,
        seed=seed,
    )
    return pipeline_maximum(results)


def subspace_control_maximum(
    data: PairedDataset,
    *,
    kind: str,
    seed: int,
    apertures: tuple[int, ...],
    ranks: tuple[int, ...],
    lags: tuple[int, ...],
    suffix_lengths: tuple[int, ...],
    condition: str,
    msc_threshold: float,
) -> float:
    """Pipeline maximum with ``Q`` replaced by a PCA or Haar-random subspace."""

    rng = np.random.default_rng(seed)
    best = float("-inf")
    for aperture in apertures:
        if aperture > data.aperture_width:
            continue
        if kind == "pca":
            rows = (data.condition == condition) & (data.split == "train")
            centered = data.source[rows][:, :aperture]
            centered = centered - centered.mean(axis=0, keepdims=True)
            _, _, right = np.linalg.svd(centered, full_matrices=False)
            override = right.T
        elif kind == "random":
            override = haar_subspace(aperture, max(ranks), rng)
        else:
            raise ValueError(f"unknown subspace control: {kind!r}")
        for suffix_length in suffix_lengths:
            for lag in lags:
                for rank in ranks:
                    cell = run_cell(
                        data,
                        aperture=aperture,
                        rank=rank,
                        lag=lag,
                        suffix_length=suffix_length,
                        condition=condition,
                        msc_threshold=msc_threshold,
                        seed=seed,
                        subspace_override=override,
                    )
                    if cell is not None:
                        best = max(best, cell.statistic)
    return best


DETERMINISTIC_CONTROLS = ("pca_subspace",)
"""Controls that are a single fixed comparison, not a resampled null.

The top-r PCA subspace of the paired difference is one deterministic
alternative to the fitted ``Q``.  Two things follow.  It cannot be scored as an
empirical p-value: one replicate pins the value at 0.5 and would block
detection at any margin, so it is reported as a margin.  And it does not belong
in the D1 detection conjunction at all, because PCA coordinates are themselves a
fixed low-dimensional coordinate system: on a synthetic world whose generator is
a genuine rank-r fixed-coordinate operator, the top-r PCA subspace scores
slightly *above* the whitened fit, and requiring the fit to beat it would refute
D1 on a world where D1 is true by construction.  What the control does decide is
narrower and is reported as its own fact: whether the fitted coordinates are
better than a generic high-variance subspace, which qualifies the
coordinate-specificity claim at D2 rather than the existence claim at D1.
"""


def summarize_detection(
    observed_maximum: float,
    null_by_family: dict[str, np.ndarray],
) -> dict[str, object]:
    """Family-wise empirical p-values against the pipeline maximum (R1 §32)."""

    per_family = {}
    deterministic: dict[str, dict[str, float | bool]] = {}
    for family, values in null_by_family.items():
        if family in DETERMINISTIC_CONTROLS:
            array = np.asarray(values, dtype=np.float64)
            array = array[np.isfinite(array)]
            if array.size:
                deterministic[family] = {
                    "control_value": float(array.max()),
                    "margin": float(observed_maximum - array.max()),
                    "observed_exceeds_control": bool(observed_maximum > array.max()),
                }
            continue
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            per_family[family] = {"replicates": 0, "p_value": None}
            continue
        per_family[family] = {
            "replicates": int(array.size),
            "null_median": float(np.median(array)),
            "null_q95": float(np.quantile(array, 0.95)),
            "null_max": float(array.max()),
            "p_value": empirical_p_value(observed_maximum, array),
        }
    p_values = [entry["p_value"] for entry in per_family.values() if entry["p_value"] is not None]
    controls_cleared = all(entry["observed_exceeds_control"] for entry in deterministic.values())
    return {
        "observed_pipeline_maximum": float(observed_maximum),
        "families": per_family,
        "deterministic_controls": deterministic,
        "coordinates_beat_generic_subspace": bool(controls_cleared),
        "coordinate_specificity_note": (
            "the deterministic PCA control qualifies the coordinate-specificity claim at "
            "D2; it is deliberately not part of the D1 detection conjunction"
        ),
        "worst_family_p_value": float(max(p_values)) if p_values else None,
        "detected_at_0_05": bool(p_values) and all(value <= 0.05 for value in p_values),
    }


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
