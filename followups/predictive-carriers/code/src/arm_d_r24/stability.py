"""Split-half stability, the association-destruction null, bootstrap overlap, ridge sweep.

Zero is not the stability null.  Two independent Haar rank-``r`` subspaces of a
256-dimensional aperture already overlap at ``r/256`` in expectation, and a real estimator
run twice on halves of one pool overlaps far more than that for reasons that have nothing to
do with a reproducible carrier.  Source section 10.2 therefore fixes three references:

``analytic``   ``r/256`` -- a sanity check, never an inferential threshold;
``null``       a FULL refit after the source-future association is destroyed, preserving
               both marginal geometries;
``ceiling``    within-half document-bootstrap refits -- the attainable sampling ceiling.

Amendment A5 adds a fourth comparison that split-half agreement structurally cannot make:
the same estimator refit across the frozen ridge grid.  A ridge artifact is common to both
halves and passes split-half stability by construction.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arm_d_r24.contract import (
    LAGS_Q,
    PRIMARY_SCORE,
    REFERENCE_SCORING_RHO,
    NULL_SEED_NAMESPACE,
    BOOTSTRAP_SEED_NAMESPACE,
    RANK_GRID,
    RHO_GRID,
    SEED,
    namespaced_seed,
)
from arm_d_r24.data import Pool
from arm_d_r24.estimators import carrier_family
from arm_d_r24.geometry import mean_subspace_correlation, principal_angles
from arm_d_r24.scoring import fit_frame


def analytic_haar_msc(rank: int, width: int) -> float:
    return float(int(rank) / int(width))


def derange(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation with no fixed point -- every document is paired with another's future."""

    while True:
        order = rng.permutation(n)
        if n < 2 or not np.any(order == np.arange(n)):
            return order


def permuted_pool(pool: Pool, order: np.ndarray, *, lags: Sequence[int] = LAGS_Q) -> Pool:
    """The same pool with every future block deranged across documents.

    The source, the increment and both marginal covariances are untouched; only the pairing
    between a document's present and its own future is destroyed, so the null answers "how
    much subspace agreement does this estimator manufacture from marginals alone".
    """

    offsets = dict(pool.offsets)
    for lag in list(lags) + [key for key in pool.offsets if key > max(lags)]:
        if int(lag) in offsets:
            offsets[int(lag)] = offsets[int(lag)][order]
    return Pool(
        name=f"{pool.name}_deranged",
        documents=pool.documents,
        offsets=offsets,
        splits=pool.splits,
        provenance={**pool.provenance, "pairing_destroyed": True},
    )


def half_solver(pool: Pool, mask: np.ndarray):
    """The increment baseline's ridge machinery for one set of fit documents.

    Reused across replicates: a derangement of the future changes the target, never this
    design, so building it once per half is an identity, not an approximation.
    """

    from arm_d_r4.common_target import GroupedRidge
    from arm_d_r24.contract import RIDGE_GRID

    mask = np.asarray(mask, dtype=bool)
    increment = pool.offsets[0] - pool.offsets[-1]
    increment = increment - increment[mask].mean(axis=0, keepdims=True)
    return GroupedRidge(
        increment[mask], pool.documents[mask], seed=SEED, ridge_grid=RIDGE_GRID
    )


def carriers_on(
    pool: Pool,
    mask: np.ndarray,
    *,
    metric: str,
    rho: float,
    ranks: Sequence[int] = RANK_GRID,
    lags: Sequence[int] = LAGS_Q,
    projection: np.ndarray | None = None,
    solver=None,
) -> dict[int, np.ndarray]:
    """Carrier bases in APERTURE coordinates, whatever the estimator saw."""

    from arm_d_r24.scoring import to_aperture

    frame = fit_frame(pool, mask, lags=lags, rho=rho, projection=projection, solver=solver)
    bases = carrier_family(frame, metric=metric, ranks=ranks)["bases"]
    return {int(rank): to_aperture(frame, basis) for rank, basis in bases.items()}


@dataclass
class SplitHalf:
    metric: str
    rho: float
    ranks: tuple[int, ...]
    a: dict[int, np.ndarray]
    b: dict[int, np.ndarray]

    def overlaps(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for rank in self.ranks:
            angles = principal_angles(self.a[rank], self.b[rank])
            out[str(int(rank))] = {
                "mean_subspace_correlation": mean_subspace_correlation(
                    self.a[rank], self.b[rank]
                ),
                "analytic_haar_expectation": analytic_haar_msc(rank, self.a[rank].shape[0]),
                "min_principal_angle_degrees": float(angles.min()),
                "median_principal_angle_degrees": float(np.median(angles)),
                "max_principal_angle_degrees": float(angles.max()),
            }
        return out


def split_half(
    pool: Pool,
    *,
    metric: str,
    rho: float,
    ranks: Sequence[int] = RANK_GRID,
    lags: Sequence[int] = LAGS_Q,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    projection: np.ndarray | None = None,
) -> SplitHalf:
    mask_a = pool.splits["fitA"] if mask_a is None else mask_a
    mask_b = pool.splits["fitB"] if mask_b is None else mask_b
    return SplitHalf(
        metric=metric,
        rho=float(rho),
        ranks=tuple(int(rank) for rank in ranks),
        a=carriers_on(
            pool, mask_a, metric=metric, rho=rho, ranks=ranks, lags=lags, projection=projection
        ),
        b=carriers_on(
            pool, mask_b, metric=metric, rho=rho, ranks=ranks, lags=lags, projection=projection
        ),
    )


#: Workers receive the pool as a directory of memory-mapped ``.npy`` blocks rather than in
#: every task payload.  Two failed approaches are worth recording.  Pickling a 2000x256x33
#: pool into 400 payloads is minutes of serialization for nothing.  Publishing it to a module
#: global and inheriting it by ``fork`` is worse: forking a process whose linear-algebra
#: runtime is already initialized deadlocked every worker, and the R2.4 development run spent
#: two separate jobs sitting in a null that never produced a single replicate.  Spawned
#: workers that mmap one on-disk copy have neither problem, and the operating system shares
#: the pages between them.
_WORKER_POOL: dict[str, Pool] = {}


def set_worker_pool(pool: Pool, key: str = "default") -> str:
    """Publish a pool in this process (used by the serial path and by spawned workers)."""

    _WORKER_POOL[key] = pool
    return key


def _pool(key: str) -> Pool:
    if key not in _WORKER_POOL:
        raise RuntimeError(f"worker pool {key!r} is not published in this process")
    return _WORKER_POOL[key]


def publish_pool_to_disk(pool: Pool, root: Path | str | None = None) -> str:
    """Write one memory-mappable copy of a pool and return its directory."""

    import json

    if root is None:
        root = Path(tempfile.mkdtemp(prefix="arm_d_r2_4_pool_"))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for key, block in pool.offsets.items():
        np.save(root / f"offset_{int(key)}.npy", np.ascontiguousarray(block))
    np.save(root / "documents.npy", np.asarray(pool.documents))
    for name, mask in pool.splits.items():
        np.save(root / f"split_{name}.npy", np.asarray(mask, dtype=bool))
    (root / "meta.json").write_text(
        json.dumps({
            "name": pool.name,
            "offsets": [int(key) for key in pool.offsets],
            "splits": list(pool.splits),
            "provenance": {str(k): str(v) for k, v in pool.provenance.items()},
        }),
        encoding="utf-8",
    )
    return str(root)


def load_pool_from_disk(root: str) -> Pool:
    """Reopen a published pool with every block memory-mapped read-only."""

    import json

    base = Path(root)
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    return Pool(
        name=str(meta["name"]),
        documents=np.load(base / "documents.npy", allow_pickle=False),
        offsets={
            int(key): np.load(base / f"offset_{int(key)}.npy", mmap_mode="r")
            for key in meta["offsets"]
        },
        splits={
            str(name): np.load(base / f"split_{name}.npy", allow_pickle=False)
            for name in meta["splits"]
        },
        provenance=dict(meta.get("provenance", {})),
    )


def _run_parallel(function, payloads, *, workers: int, pool: Pool, key: str, chunksize: int = 1):
    """Map ``function`` over ``payloads`` in THREADS, sharing one in-process pool.

    Threads rather than processes, for a reason worth recording.  A refit spends nearly all of
    its time inside dense factorizations that release the interpreter lock, so thread
    parallelism scales on this workload; and process parallelism is not available here at all.
    Forked workers deadlocked on an already-initialized linear-algebra runtime, and spawned
    workers were killed by the sandbox with leaked-semaphore warnings.  Two development jobs
    were lost to that before this path was measured.
    """

    set_worker_pool(pool, key=key)
    if int(workers) <= 1 or len(payloads) <= 1:
        return [function(payload) for payload in payloads]
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        return list(executor.map(function, payloads, chunksize=int(chunksize)))


#: Solvers shared across replicates, keyed by pool name and half.
_SOLVERS: dict[tuple[str, str], object] = {}


def shared_solver(pool: Pool, mask: np.ndarray, tag: str):
    key = (pool.name, tag)
    if key not in _SOLVERS:
        _SOLVERS[key] = half_solver(pool, mask)
    return _SOLVERS[key]


def _null_replicate(payload) -> tuple[str, int, dict[str, float]]:
    pool_key, metric, rho, ranks, lags, index, mask_a, mask_b, projection = payload
    pool = _pool(pool_key)
    rng = np.random.default_rng(namespaced_seed(NULL_SEED_NAMESPACE, index))
    n = pool.n_documents
    order_a = np.arange(n)
    order_b = np.arange(n)
    idx_a = np.flatnonzero(mask_a)
    idx_b = np.flatnonzero(mask_b)
    order_a[idx_a] = idx_a[derange(idx_a.size, rng)]
    order_b[idx_b] = idx_b[derange(idx_b.size, rng)]
    a = carriers_on(
        permuted_pool(pool, order_a, lags=lags), mask_a, metric=metric, rho=rho,
        ranks=ranks, lags=lags, projection=projection,
        solver=shared_solver(pool, mask_a, "null_a"),
    )
    b = carriers_on(
        permuted_pool(pool, order_b, lags=lags), mask_b, metric=metric, rho=rho,
        ranks=ranks, lags=lags, projection=projection,
        solver=shared_solver(pool, mask_b, "null_b"),
    )
    return metric, index, {
        str(int(rank)): mean_subspace_correlation(a[rank], b[rank]) for rank in ranks
    }


def association_destruction_null(
    pool: Pool,
    *,
    metric: str,
    rho: float,
    replicates: int,
    ranks: Sequence[int] = RANK_GRID,
    lags: Sequence[int] = LAGS_Q,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    projection: np.ndarray | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """``replicates`` full refits of both halves after destroying the pairing.

    ``workers`` above one runs the refits in spawned workers that memory-map one published
    copy of the pool; see the note above ``_WORKER_POOL`` for why fork is not used.
    """

    mask_a = pool.splits["fitA"] if mask_a is None else mask_a
    mask_b = pool.splits["fitB"] if mask_b is None else mask_b
    key = f"null_{pool.name}"
    payloads = [
        (key, metric, float(rho), tuple(int(r) for r in ranks), tuple(int(l) for l in lags),
         int(index), mask_a, mask_b, projection)
        for index in range(int(replicates))
    ]
    collected: dict[str, list[float]] = {str(int(rank)): [] for rank in ranks}
    for _, _, values in _run_parallel(
        _null_replicate, payloads, workers=int(workers), pool=pool, key=key, chunksize=2
    ):
        for name, value in values.items():
            collected[name].append(float(value))
    return {
        "metric": metric,
        "replicates": int(replicates),
        "construction": (
            "document-coherent derangement of the future block inside each fit half, with "
            "every mean, baseline, covariance, whitener and eigensystem refit from scratch"
        ),
        "by_rank": {
            key: {
                "mean": float(np.mean(values)),
                "q95": float(np.quantile(values, 0.95)),
                "q99": float(np.quantile(values, 0.99)),
                "max": float(np.max(values)),
                "values": [float(value) for value in values],
            }
            for key, values in collected.items()
        },
    }


def _bootstrap_replicate(payload) -> tuple[int, dict[str, float]]:
    pool_key, metric, rho, ranks, lags, index, mask, reference, projection = payload
    pool = _pool(pool_key)
    rng = np.random.default_rng(namespaced_seed(BOOTSTRAP_SEED_NAMESPACE, index))
    idx = np.flatnonzero(mask)
    draw = rng.choice(idx, size=idx.size, replace=True)
    resampled = Pool(
        name=f"{pool.name}_boot{index}",
        documents=np.array([f"{doc}#{position}" for position, doc in enumerate(pool.documents[draw])]),
        offsets={key: block[draw] for key, block in pool.offsets.items()},
        splits={"fit": np.ones(draw.size, dtype=bool)},
        provenance={**pool.provenance, "bootstrap_replicate": int(index)},
    )
    bases = carriers_on(
        resampled, resampled.splits["fit"], metric=metric, rho=rho, ranks=ranks, lags=lags,
        projection=projection,
    )
    return index, {
        str(int(rank)): mean_subspace_correlation(bases[rank], reference[rank]) for rank in ranks
    }


def bootstrap_overlap(
    pool: Pool,
    *,
    metric: str,
    rho: float,
    replicates: int,
    reference: dict[int, np.ndarray],
    mask: np.ndarray,
    ranks: Sequence[int] = RANK_GRID,
    lags: Sequence[int] = LAGS_Q,
    projection: np.ndarray | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """The attainable sampling ceiling: refits on resampled documents of one half."""

    key = f"boot_{pool.name}"
    payloads = [
        (key, metric, float(rho), tuple(int(r) for r in ranks), tuple(int(l) for l in lags),
         int(index), mask, reference, projection)
        for index in range(int(replicates))
    ]
    collected: dict[str, list[float]] = {str(int(rank)): [] for rank in ranks}
    for _, values in _run_parallel(
        _bootstrap_replicate, payloads, workers=int(workers), pool=pool, key=key, chunksize=2
    ):
        for name, value in values.items():
            collected[name].append(float(value))
    return {
        "metric": metric,
        "replicates": int(replicates),
        "by_rank": {
            key: {
                "mean": float(np.mean(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q50": float(np.quantile(values, 0.50)),
            }
            for key, values in collected.items()
        },
    }


def empirical_p_value(observed: float, null_values: Sequence[float]) -> float:
    values = np.asarray(list(null_values), dtype=np.float64)
    return float((1.0 + float(np.sum(values >= float(observed)))) / (values.size + 1.0))


def ridge_sweep(
    pool: Pool,
    *,
    metric: str,
    selected_rho: float,
    mask: np.ndarray,
    ranks: Sequence[int] = RANK_GRID,
    lags: Sequence[int] = LAGS_Q,
    grid: Sequence[float] = RHO_GRID,
    projection: np.ndarray | None = None,
) -> dict[str, object]:
    """Amendment A5: refit across the frozen ridge grid and compare subspaces."""

    bases = {
        float(rho): carriers_on(
            pool, mask, metric=metric, rho=float(rho), ranks=ranks, lags=lags,
            projection=projection,
        )
        for rho in grid
    }
    reference = bases[float(selected_rho)]
    by_rank: dict[str, object] = {}
    for rank in ranks:
        entries = {}
        for rho, family in bases.items():
            if rho == float(selected_rho):
                continue
            entries[f"{rho:g}"] = mean_subspace_correlation(family[int(rank)], reference[int(rank)])
        adjacent = [
            f"{float(value):g}"
            for value in grid
            if float(value) != float(selected_rho)
            and abs(np.log10(float(value)) - np.log10(float(selected_rho))) <= 1.0 + 1e-9
        ]
        by_rank[str(int(rank))] = {
            "msc_against_selected": entries,
            "min_msc_over_grid": float(min(entries.values())) if entries else float("nan"),
            "min_msc_adjacent_decade": (
                float(min(entries[key] for key in adjacent)) if adjacent else float("nan")
            ),
        }
    return {
        "metric": metric,
        "selected_rho": float(selected_rho),
        "grid": [float(value) for value in grid],
        "by_rank": by_rank,
    }


def select_rho(
    pool: Pool,
    *,
    mask: np.ndarray,
    metric: str = "whitened",
    rank: int = 16,
    lags: Sequence[int] = LAGS_Q,
    grid: Sequence[float] = RHO_GRID,
    folds: int = 5,
    seed: int = SEED,
    reference_rho: float = REFERENCE_SCORING_RHO,
) -> dict[str, object]:
    """Resolve the whitening ridge on FIT documents only, by grouped cross-validation.

    Two details make this honest rather than circular.  First, every candidate carrier is
    scored through ONE fixed reference metric -- the frame at ``reference_rho`` fitted on the
    same inner folds -- so the candidates are compared on a common yardstick instead of each
    being graded by the metric its own ridge defines.  Second, each estimator is selected on
    the score it exists to maximize: the energy estimator on the primary gain, the whitened
    estimator on the whitened score.  Selecting Q-W by an energy-weighted score would always
    prefer the least whitening, which is exactly the answer Q-W was built to avoid assuming.
    """

    from arm_d_r24.scoring import score_carrier

    criterion = PRIMARY_SCORE if metric == "energy" else "gamma_white"
    index = np.flatnonzero(np.asarray(mask, dtype=bool))
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(index.size) % int(folds)
    scores: dict[str, list[float]] = {f"{float(rho):g}": [] for rho in grid}
    companion_scores: dict[str, list[float]] = {f"{float(rho):g}": [] for rho in grid}
    for fold in range(int(folds)):
        inner = np.zeros(pool.n_documents, dtype=bool)
        outer = np.zeros(pool.n_documents, dtype=bool)
        inner[index[assignment != fold]] = True
        outer[index[assignment == fold]] = True
        reference_frame = fit_frame(pool, inner, lags=lags, rho=float(reference_rho))
        for rho in grid:
            frame = fit_frame(pool, inner, lags=lags, rho=float(rho))
            basis = carrier_family(frame, metric=metric, ranks=(int(rank),))["bases"][int(rank)]
            scored = score_carrier(
                reference_frame, basis, documents=pool.documents, score_mask=outer
            )
            key = f"{float(rho):g}"
            scores[key].append(float(scored["summary"]["multi_horizon_mean"][criterion]))
            companion = PRIMARY_SCORE if criterion == "gamma_white" else "gamma_white"
            companion_scores[key].append(
                float(scored["summary"]["multi_horizon_mean"][companion])
            )
    means = {key: float(np.mean(values)) for key, values in scores.items()}
    chosen = max(means, key=lambda key: means[key])
    grid_values = [float(value) for value in grid]
    return {
        "metric": metric,
        "criterion": criterion,
        "reference_scoring_rho": float(reference_rho),
        "reference_rank": int(rank),
        "folds": int(folds),
        "grid": grid_values,
        "fold_scores": scores,
        "mean_cross_validated_criterion": means,
        "mean_cross_validated_companion_score": {
            key: float(np.mean(values)) for key, values in companion_scores.items()
        },
        "selected_rho": float(chosen),
        "selected_at_grid_edge": bool(
            float(chosen) in (min(grid_values), max(grid_values))
        ),
        "rule": (
            "argmax of document-grouped cross-validated multi-horizon score of this "
            "estimator's carrier at the reference rank, every candidate scored through one "
            "fixed reference metric, on FIT documents only"
        ),
        "primary_score": PRIMARY_SCORE,
    }
