"""The R2.4 analysis object: one row per document, one aperture, one hook.

R2.2's capture stores the source hook at every offset a composition target needs, so the
whole R2.4 object -- ``x_t`` and ``x_{t+k}`` for every lag in the grid -- is already inside
it.  Nothing here recaptures or reprojects: the aperture is R1's frozen mean and basis and
the rows are the ``main`` condition at suffix length 64, which is the object R2.3 and R2.3c
scored their carriers on.

Two pools use this loader.  DEV-Q is the spent 2000-document R2.2 pool, whose R2.2 split
names ("train"/"eval"/"sealed") are irrelevant here because nothing on it is confirmatory;
R2.4 re-splits it by document hash for its own learning curves.  CONFIRM-Q is the fresh
capture, split 35/35/15/15 by document.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from arm_d_r2.dataset import R2Capture

from arm_d_r24.contract import (
    ANALYSIS_CONDITION,
    ANALYSIS_SUFFIX,
    APERTURE_WIDTH,
    CONFIRM_SPLIT_FRACTIONS,
    DESCRIPTIVE_LAG,
    LAGS_Q,
    SEED,
)

#: Every offset the analysis needs: the source, the persistence increment, the lag grid.
REQUIRED_OFFSETS = (-1, 0) + tuple(LAGS_Q)


@dataclass
class Pool:
    """One document-indexed aperture pool.

    ``offsets`` maps an integer offset to an ``(n_documents, 256)`` float64 block, and
    ``splits`` maps a split name to a boolean row mask.  Rows and documents are the same
    thing here: one spliced sequence per document at the frozen condition and suffix.
    """

    name: str
    documents: np.ndarray
    offsets: dict[int, np.ndarray]
    splits: dict[str, np.ndarray]
    provenance: dict[str, object]

    @property
    def n_documents(self) -> int:
        return int(self.documents.shape[0])

    def mask(self, *names: str) -> np.ndarray:
        out = np.zeros(self.n_documents, dtype=bool)
        for name in names:
            out |= self.splits[name]
        return out

    def subset(self, mask: np.ndarray, name: str | None = None) -> "Pool":
        mask = np.asarray(mask, dtype=bool)
        return Pool(
            name=name or self.name,
            documents=self.documents[mask],
            offsets={offset: block[mask] for offset, block in self.offsets.items()},
            splits={key: value[mask] for key, value in self.splits.items()},
            provenance=dict(self.provenance),
        )

    def summary(self) -> dict[str, object]:
        return {
            "pool": self.name,
            "documents": self.n_documents,
            "split_counts": {
                key: int(value.sum()) for key, value in sorted(self.splits.items())
            },
            "offsets": sorted(int(offset) for offset in self.offsets),
            "aperture_width": int(next(iter(self.offsets.values())).shape[1]),
            **self.provenance,
        }


def document_hash_fraction(document_id: str, salt: str) -> float:
    """A deterministic uniform in [0,1) from the document's content hash.

    Splitting on the hash rather than on a shuffled index means the same document lands in
    the same split however the pool was ordered or subsampled.
    """

    digest = hashlib.sha256(f"{salt}:{document_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def four_way_split(
    documents: Sequence[str],
    *,
    salt: str,
    fractions: Mapping[str, float] = CONFIRM_SPLIT_FRACTIONS,
) -> dict[str, np.ndarray]:
    """The frozen 35/35/15/15 document split of unified section 12."""

    order = ("fitA", "fitB", "validation", "test")
    edges: list[tuple[str, float, float]] = []
    running = 0.0
    for name in order:
        width = float(fractions[name])
        edges.append((name, running, running + width))
        running += width
    if abs(running - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {running}")
    values = np.array([document_hash_fraction(str(doc), salt) for doc in documents])
    return {
        name: (values >= low) & (values < high if high < 1.0 else values <= 1.0)
        for name, low, high in edges
    }


def load_pool(
    path,
    *,
    name: str,
    split_salt: str,
    condition: str = ANALYSIS_CONDITION,
    suffix: int = ANALYSIS_SUFFIX,
    width: int = APERTURE_WIDTH,
    with_descriptive_lag: bool = True,
    splits: Mapping[str, np.ndarray] | None = None,
) -> Pool:
    """Load an R2.2-format capture and reduce it to the R2.4 analysis object."""

    capture = R2Capture.load(path)
    mask = (capture.condition == condition) & (capture.suffix_length == int(suffix))
    selected = capture.select(mask)
    documents = np.asarray(selected.document_ids)
    if np.unique(documents).size != documents.size:
        raise ValueError(
            "the analysis object must have exactly one row per document; found "
            f"{documents.size} rows over {np.unique(documents).size} documents"
        )
    order = np.argsort(documents, kind="stable")
    documents = documents[order]

    # Load every source-hook offset the capture carries, not only the analysis minimum: the
    # DEV direction-versus-magnitude diagnostic needs the five-deep history stack, and the
    # descriptive lag 128 is in the same block.
    wanted = sorted(
        set(int(value) for value in selected.source_offsets)
        | set(REQUIRED_OFFSETS)
        | ({DESCRIPTIVE_LAG} if with_descriptive_lag else set())
    )
    required = set(REQUIRED_OFFSETS)
    offsets: dict[int, np.ndarray] = {}
    missing: list[int] = []
    for offset in wanted:
        if int(offset) not in selected.source_offsets:
            if int(offset) in required:
                missing.append(int(offset))
            continue
        slot = selected.slot(int(offset))
        offsets[int(offset)] = np.ascontiguousarray(
            selected.source_block[order][:, slot, :width].astype(np.float64)
        )
    if missing:
        raise ValueError(f"capture is missing required source offsets {missing}")

    resolved = (
        {key: np.asarray(value, dtype=bool) for key, value in splits.items()}
        if splits is not None
        else four_way_split(documents, salt=split_salt)
    )
    provenance = {
        "capture_path": str(path),
        "condition": condition,
        "suffix_length": int(suffix),
        "split_salt": split_salt,
        "rows_in_capture": int(capture.n_rows),
        "rows_selected": int(documents.shape[0]),
        "capture_split_names": sorted(set(np.asarray(capture.split).tolist())),
    }
    return Pool(
        name=name, documents=documents, offsets=offsets, splits=resolved, provenance=provenance
    )


def subsample_documents(pool: Pool, size: int, *, seed: int = SEED) -> Pool:
    """A document subsample for the learning curves of unified section 9."""

    rng = np.random.default_rng(seed)
    size = int(min(size, pool.n_documents))
    index = np.sort(rng.choice(pool.n_documents, size=size, replace=False))
    mask = np.zeros(pool.n_documents, dtype=bool)
    mask[index] = True
    return pool.subset(mask, name=f"{pool.name}_n{size}")


def synthetic_pool(
    offsets: Mapping[int, np.ndarray],
    *,
    name: str,
    split_salt: str,
    documents: Sequence[str] | None = None,
) -> Pool:
    """Wrap simulated blocks in the same container the real pool uses."""

    n = int(next(iter(offsets.values())).shape[0])
    ids = np.asarray(
        list(documents) if documents is not None else [f"{name}-{index:06d}" for index in range(n)]
    )
    return Pool(
        name=name,
        documents=ids,
        offsets={int(key): np.asarray(value, dtype=np.float64) for key, value in offsets.items()},
        splits=four_way_split(ids, salt=split_salt),
        provenance={"synthetic": True, "split_salt": split_salt},
    )
